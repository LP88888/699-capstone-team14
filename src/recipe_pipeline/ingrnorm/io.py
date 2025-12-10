
from __future__ import annotations
import ast, json
from pathlib import Path
from typing import List
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import pyarrow.csv as pa_csv
    _HAS_PACSV = True
except Exception:
    pa_csv = None
    _HAS_PACSV = False

logger = logging.getLogger(__name__)

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
        pf = pq.ParquetFile(str(input_path))
        # If column is already list<string>, keep as-is
        schema = pf.schema_arrow
        if ner_col in schema.names:
            field = schema.field(ner_col)
            if pa.types.is_list(field.type) and pa.types.is_string(field.type.value_type):
                return input_path
        # Otherwise convert to list<string>
        tmp_out.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(str(tmp_out), pa.schema([pa.field(ner_col, pa.list_(pa.string()))]), compression="zstd")
        for batch in pf.iter_batches(columns=[ner_col]):
            arr_in = batch.column(0).to_pylist()
            lists = [parse_listish(x) for x in arr_in]
            arr = pa.array(lists, type=pa.list_(pa.string()))
            writer.write_table(pa.Table.from_arrays([arr], names=[ner_col]))
        writer.close()
        return tmp_out

    if suffix == ".csv":
        tmp_out.parent.mkdir(parents=True, exist_ok=True)
        schema = pa.schema([pa.field(ner_col, pa.list_(pa.string()))])
        writer = pq.ParquetWriter(str(tmp_out), schema, compression="zstd")

        def _write(chunk_values):
            lists = [parse_listish(x) for x in chunk_values]
            arr = pa.array(lists, type=pa.list_(pa.string()))
            tbl = pa.Table.from_arrays([arr], names=[ner_col])
            writer.write_table(tbl)

        if _HAS_PACSV:
            try:
                block_size = max(int(chunksize or 250_000), 50_000)
                read_opts = pa_csv.ReadOptions(block_size=block_size, use_threads=True)
                convert_opts = pa_csv.ConvertOptions(include_columns=[ner_col])
                reader = pa_csv.open_csv(str(input_path), read_options=read_opts, convert_options=convert_opts)
                for batch in reader:
                    _write(batch.column(0).to_pylist())
                writer.close()
                return tmp_out
            except Exception as exc:
                logger.warning("pyarrow.csv fast path failed (%s). Falling back to pandas reader.", exc)
                writer.close()
                tmp_out.unlink(missing_ok=True)
                writer = pq.ParquetWriter(str(tmp_out), schema, compression="zstd")

        read_kwargs = dict(chunksize=chunksize, dtype=str)
        try:
            reader = pd.read_csv(input_path, **read_kwargs)
            for chunk in reader:
                col = chunk[ner_col] if ner_col in chunk.columns else pd.Series([None] * len(chunk))
                _write(col.tolist())
        except pd.errors.ParserError as exc:
            logger.warning(
                "Failed to parse %s with default CSV reader (%s). Falling back to python engine with on_bad_lines='skip'.",
                input_path,
                exc,
            )
            read_kwargs.update({"engine": "python", "on_bad_lines": "skip"})
            reader = pd.read_csv(input_path, **read_kwargs)
            for chunk in reader:
                col = chunk[ner_col] if ner_col in chunk.columns else pd.Series([None] * len(chunk))
                _write(col.tolist())
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
