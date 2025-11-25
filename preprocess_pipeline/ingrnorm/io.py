
from __future__ import annotations
import ast, json
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def parse_listish(v) -> list[str]:
    """Accept list/tuple/np.ndarray; try JSON/Python list in string; fallback to single-item list."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, (list, tuple, np.ndarray)):
        return [str(x) for x in v if str(x).strip()]
    s = str(v).strip()
    if not s:
        return []
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        parsed = None
        try:
            parsed = json.loads(s)
        except Exception:
            try:
                parsed = ast.literal_eval(s)
            except Exception:
                parsed = None
        if isinstance(parsed, (list, tuple, np.ndarray)):
            return [str(x) for x in parsed if str(x).strip()]
    return [s]

def materialize_parquet_source(input_path: Path, ner_col: str, chunksize: int, tmp_out: Path) -> Path:
    """Ensure we have a Parquet file with a list<string> column `ner_col` from CSV/Excel/Parquet."""
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        return input_path

    if suffix == ".csv":
        tmp_out.parent.mkdir(parents=True, exist_ok=True)
        schema = pa.schema([pa.field(ner_col, pa.list_(pa.string()))])
        writer = pq.ParquetWriter(str(tmp_out), schema, compression="zstd")
        for chunk in pd.read_csv(input_path, chunksize=chunksize, dtype=str):
            col = chunk[ner_col] if ner_col in chunk.columns else pd.Series([None] * len(chunk))
            lists = [parse_listish(x) for x in col]
            arr = pa.array(lists, type=pa.list_(pa.string()))
            tbl = pa.Table.from_arrays([arr], names=[ner_col])
            writer.write_table(tbl)
        writer.close()
        return tmp_out

    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(input_path, dtype=str)
        col = df[ner_col] if ner_col in df.columns else pd.Series([None] * len(df))
        lists = [parse_listish(x) for x in col]
        arr = pa.array(lists, type=pa.list_(pa.string()))
        tbl = pa.Table.from_arrays([arr], names=[ner_col])
        tmp_out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(tbl, str(tmp_out), compression="zstd")
        return tmp_out

    raise ValueError(f"Unsupported file type: {input_path.name}")