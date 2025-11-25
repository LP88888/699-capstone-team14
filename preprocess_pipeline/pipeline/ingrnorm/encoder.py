
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Any, Optional, Dict, Union, List
from collections import Counter
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

class IngredientEncoder:
    """Deterministic token→id encoder for already-normalized ingredient tokens."""
    def __init__(self, min_freq: int = 1):
        self.min_freq = int(min_freq)
        self.token_to_id: Dict[str, int] = {"<UNK>": 0}
        self.id_to_token: Dict[int, str] = {0: "<UNK>"}
        self._frozen = False

    def _fit_from_counter(self, cnt: Counter, min_freq: int) -> "IngredientEncoder":
        """Internal helper to add tokens from a counter to the encoder."""
        for tok, c in cnt.items():
            if c >= min_freq and tok not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[tok] = idx
                self.id_to_token[idx] = tok
        return self

    def _count_tokens(self, series: Iterable[Any]) -> Counter:
        """Count tokens from an iterable of lists."""
        cnt = Counter()
        for lst in series:
            if isinstance(lst, (list, tuple, np.ndarray)):
                for t in lst:
                    s = str(t).strip().lower()
                    if s:
                        cnt[s] += 1
        return cnt

    def fit_from_series(self, series: Iterable[Any], min_freq: Optional[int] = None) -> "IngredientEncoder":
        mf = self.min_freq if min_freq is None else int(min_freq)
        cnt = self._count_tokens(series)
        return self._fit_from_counter(cnt, mf)

    def fit_from_parquet_streaming(self, parquet_path: Union[str, Path], col: str = "NER_clean", min_freq: Optional[int] = None) -> "IngredientEncoder":
        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        mf = self.min_freq if min_freq is None else int(min_freq)
        pf = pq.ParquetFile(parquet_path)
        if pf.num_row_groups == 0:
            raise ValueError(f"Parquet file has no row groups: {parquet_path}")
        cnt = Counter()
        for rg in range(pf.num_row_groups):
            df_rg = pf.read_row_group(rg, columns=[col]).to_pandas()
            if col not in df_rg.columns:
                raise ValueError(f"Column '{col}' not found in parquet file {parquet_path} (row group {rg})")
            for lst in df_rg[col]:
                if isinstance(lst, (list, tuple, np.ndarray)):
                    for t in lst:
                        s = str(t).strip().lower()
                        if s:
                            cnt[s] += 1
        return self._fit_from_counter(cnt, mf)

    def freeze(self) -> "IngredientEncoder":
        self._frozen = True
        return self

    def transform_series_to_idlists(self, series: Iterable[Any]) -> List[List[int]]:
        out: List[List[int]] = []
        for lst in series:
            ids: List[int] = []
            if isinstance(lst, (list, tuple, np.ndarray)):
                for t in lst:
                    tok = str(t).strip().lower()
                    if tok:
                        ids.append(self.token_to_id.get(tok, 0))
            else:
                ids = [0]
            out.append(ids if ids else [0])
        return out

    def transform_df(self, df: pd.DataFrame, ingredients_col: str = "NER_clean", dataset_id: int = 1) -> pd.DataFrame:
        id_lists = self.transform_series_to_idlists(df[ingredients_col])
        res = pd.DataFrame({
            "Dataset ID": np.int32(dataset_id),
            "Index": np.arange(len(df), dtype=np.int64),
            "Ingredients": id_lists,
        })
        return res

    def encode_parquet_streaming(self, parquet_path: Union[str, Path], out_parquet_path: Union[str, Path], dataset_id: int = 1, col: str = "NER_clean", compression: str = "zstd") -> Path:
        parquet_path = Path(parquet_path)
        out_parquet_path = Path(out_parquet_path)
        pf = pq.ParquetFile(parquet_path)
        target_schema = pa.schema([
            pa.field("Dataset ID", pa.int32()),
            pa.field("Index", pa.int64()),
            pa.field("Ingredients", pa.list_(pa.int64())),
        ])
        out_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(out_parquet_path, target_schema, compression=compression)
        global_index_start = 0
        for rg in range(pf.num_row_groups):
            df_rg = pf.read_row_group(rg, columns=[col]).to_pandas()
            id_lists = self.transform_series_to_idlists(df_rg[col])
            ds_ids = pa.array(np.full(len(id_lists), dataset_id, dtype=np.int32))
            idxs   = pa.array(np.arange(global_index_start, global_index_start + len(id_lists), dtype=np.int64))
            ingr   = pa.array([(lst if isinstance(lst, (list, tuple, np.ndarray)) else []) for lst in id_lists], type=pa.list_(pa.int64()))
            tbl = pa.Table.from_arrays([ds_ids, idxs, ingr], names=["Dataset ID", "Index", "Ingredients"])
            writer.write_table(tbl)
            global_index_start += len(id_lists)
        writer.close()
        return out_parquet_path

    def save_maps(self, id_to_token_path: Union[str, Path], token_to_id_path: Optional[Union[str, Path]] = None) -> None:
        id_to_token_path = Path(id_to_token_path)
        token_to_id_path = Path(token_to_id_path) if token_to_id_path else id_to_token_path.with_name("ingredient_token_to_id.json")
        id_to_token_path.parent.mkdir(parents=True, exist_ok=True)
        with open(id_to_token_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.id_to_token.items()}, f, indent=2)
        with open(token_to_id_path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, indent=2)

    @classmethod
    def load_maps(cls, id_to_token_path: Union[str, Path], token_to_id_path: Optional[Union[str, Path]] = None) -> "IngredientEncoder":
        id_to_token_path = Path(id_to_token_path)
        token_to_id_path = Path(token_to_id_path) if token_to_id_path else id_to_token_path.with_name("ingredient_token_to_id.json")
        with open(id_to_token_path, "r", encoding="utf-8") as f:
            id_to_token_raw = json.load(f)
        with open(token_to_id_path, "r", encoding="utf-8") as f:
            token_to_id_raw = json.load(f)
        enc = cls()
        enc.id_to_token = {int(k): str(v) for k, v in id_to_token_raw.items()}
        enc.token_to_id = {str(k): int(v) for k, v in token_to_id_raw.items()}
        enc._frozen = True
        return enc
