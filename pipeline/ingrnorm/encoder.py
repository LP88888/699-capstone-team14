from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import Iterable, List, Dict, Any, Optional

import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class IngredientEncoder:
    """
    Deterministic token→id encoder for ingredient tokens that are already normalized
    (e.g., lists in a column like NER_clean).

    Capabilities
    ------------
    - fit_from_series(series): build/extend vocab from an in-memory iterable of list-like rows
    - fit_from_parquet_streaming(parquet_path, col): stream Parquet row-groups to build vocab
    - transform_df(df, ingredients_col, dataset_id): encode a small DataFrame in memory
    - encode_parquet_streaming(parquet_path, out_parquet_path, dataset_id, col): stream-encode a large Parquet
    - save_maps(id_to_token_path, token_to_id_path): persist both vocab directions
    - load_maps(id_to_token_path, token_to_id_path): restore a frozen encoder with the same IDs

    Notes
    -----
    - This class does NOT normalize text; it assumes the input column already contains clean tokens/phrases.
    - Token ID 0 is reserved for <UNK>.
    """

    def __init__(self, min_freq: int = 1):
        self.min_freq = int(min_freq)
        # Reserve 0 for unknown
        self.token_to_id: Dict[str, int] = {"<UNK>": 0}
        self.id_to_token: Dict[int, str] = {0: "<UNK>"}
        self._frozen: bool = False

    # Fit / vocab building
    def fit_from_series(self, series: Iterable[Any], min_freq: Optional[int] = None) -> "IngredientEncoder":
        """
        Build/extend vocabulary from an in-memory iterable whose elements are list/tuple/ndarray of strings.
        """
        mf = self.min_freq if min_freq is None else int(min_freq)
        cnt = Counter()
        for lst in series:
            if isinstance(lst, (list, tuple, np.ndarray)):
                for t in lst:
                    s = str(t).strip().lower()
                    if s:
                        cnt[s] += 1

        for tok, c in cnt.items():
            if c >= mf and tok not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[tok] = idx
                self.id_to_token[idx] = tok
        return self

    def fit_from_parquet_streaming(
        self,
        parquet_path: Path | str,
        col: str = "NER_clean",
        min_freq: Optional[int] = None,
    ) -> "IngredientEncoder":
        """
        Build/extend vocabulary by streaming Parquet row-groups from `parquet_path`
        and counting tokens in column `col` (list-like per row).
        """
        parquet_path = Path(parquet_path)
        mf = self.min_freq if min_freq is None else int(min_freq)

        pf = pq.ParquetFile(parquet_path)
        cnt = Counter()
        for rg in range(pf.num_row_groups):
            df_rg = pf.read_row_group(rg, columns=[col]).to_pandas()
            for lst in df_rg[col]:
                if isinstance(lst, (list, tuple, np.ndarray)):
                    for t in lst:
                        s = str(t).strip().lower()
                        if s:
                            cnt[s] += 1
            del df_rg

        for tok, c in cnt.items():
            if c >= mf and tok not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[tok] = idx
                self.id_to_token[idx] = tok
        return self

    def freeze(self) -> "IngredientEncoder":
        """Optionally mark the encoder as frozen (IDs fixed)."""
        self._frozen = True
        return self

    # Transform / encode
    def transform_series_to_idlists(self, series: Iterable[Any]) -> List[List[int]]:
        """
        Convert a series of list-like rows to list[int] using current vocab. Unknowns → 0.
        """
        out: List[List[int]] = []
        for lst in series:
            ids: List[int] = []
            if isinstance(lst, (list, tuple, np.ndarray)):
                for t in lst:
                    tok = str(t).strip().lower()
                    if not tok:
                        continue
                    ids.append(self.token_to_id.get(tok, 0))
            else:
                # Malformed row fallback
                ids = [0]
            out.append(ids if ids else [0])
        return out

    def transform_df(
        self,
        df: pd.DataFrame,
        ingredients_col: str = "NER_clean",
        dataset_id: int = 1,
    ) -> pd.DataFrame:
        """
        Encode a small in-memory DataFrame to:
          ["Dataset ID"(int32), "Index"(int64), "Ingredients"(list[int])]
        """
        id_lists = self.transform_series_to_idlists(df[ingredients_col])
        res = pd.DataFrame({
            "Dataset ID": np.int32(dataset_id),
            "Index": np.arange(len(df), dtype=np.int64),
            "Ingredients": id_lists,
        })
        return res

    def encode_parquet_streaming(
        self,
        parquet_path: Path | str,
        out_parquet_path: Path | str,
        dataset_id: int = 1,
        col: str = "NER_clean",
        compression: str = "zstd",
    ) -> Path:
        """
        Stream-encode a large Parquet to a unified Parquet with schema:
          ["Dataset ID"(int32), "Index"(int64), "Ingredients"(list<int64>)]
        """
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
            ingr   = pa.array(
                [(lst if isinstance(lst, (list, tuple, np.ndarray)) else []) for lst in id_lists],
                type=pa.list_(pa.int64())
            )

            tbl = pa.Table.from_arrays([ds_ids, idxs, ingr], names=["Dataset ID", "Index", "Ingredients"])
            writer.write_table(tbl)

            global_index_start += len(id_lists)
            del df_rg, id_lists, ds_ids, idxs, ingr, tbl

        writer.close()
        return out_parquet_path

    # Persistence
    def save_maps(self, id_to_token_path: Path | str, token_to_id_path: Optional[Path | str] = None) -> None:
        """
        Save both vocab directions. If token_to_id_path is None, use a sibling name 'ingredient_token_to_id.json'.
        """
        id_to_token_path = Path(id_to_token_path)
        if token_to_id_path is None:
            token_to_id_path = id_to_token_path.with_name("ingredient_token_to_id.json")
        else:
            token_to_id_path = Path(token_to_id_path)

        id_to_token_path.parent.mkdir(parents=True, exist_ok=True)

        with open(id_to_token_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in self.id_to_token.items()}, f, indent=2)

        with open(token_to_id_path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, indent=2)

    @classmethod
    def load_maps(cls, id_to_token_path: Path | str, token_to_id_path: Optional[Path | str] = None) -> "IngredientEncoder":
        """
        Restore an encoder with the exact same IDs. If token_to_id_path is None,
        infer it as a sibling named 'ingredient_token_to_id.json'.
        """
        id_to_token_path = Path(id_to_token_path)
        if token_to_id_path is None:
            token_to_id_path = id_to_token_path.with_name("ingredient_token_to_id.json")
        else:
            token_to_id_path = Path(token_to_id_path)

        with open(id_to_token_path, "r", encoding="utf-8") as f:
            id_to_token_raw = json.load(f)
        id_to_token = {int(k): str(v) for k, v in id_to_token_raw.items()}

        with open(token_to_id_path, "r", encoding="utf-8") as f:
            token_to_id_raw = json.load(f)
        token_to_id = {str(k): int(v) for k, v in token_to_id_raw.items()}

        enc = cls()
        enc.id_to_token = id_to_token
        enc.token_to_id = token_to_id
        enc._frozen = True
        return enc
