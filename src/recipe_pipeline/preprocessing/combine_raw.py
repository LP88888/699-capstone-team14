"""Pipeline stage that combines raw CSV datasets into a single parquet file."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..core import PipelineContext, StageResult
from ..utils import stage_logger


def read_csv_with_fallback(path: Path, logger: logging.Logger) -> pd.DataFrame:
    encodings = ["utf-8", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            logger.debug(f"Attempting to read {path.name} with encoding={enc}")
            df = pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
            logger.info(f"Successfully read {path.name} with encoding={enc} ({len(df):,} rows)")
            return df
        except UnicodeDecodeError as e:
            logger.warning(f"Failed to read {path.name} with encoding={enc}: {e}")
            continue
        except Exception as e:
            logger.exception(f"Unexpected error reading {path.name} with encoding={enc}: {e}")
            raise
    logger.warning(f"All standard encodings failed for {path.name}. Trying utf-8 with errors='replace'.")
    df = pd.read_csv(path, encoding="utf-8", errors='replace', dtype=str, low_memory=False)
    logger.info(f"Read {path.name} with utf-8 (errors=replace) ({len(df):,} rows)")
    return df


def find_ingredients_column(df: pd.DataFrame, logger: logging.Logger) -> Optional[str]:
    cols_lower = {col.lower(): col for col in df.columns}
    candidates = ["ingredients", "ingredient", "ing", "ingr", "ingredient_list", "ingredients_list"]
    for candidate in candidates:
        if candidate in cols_lower:
            actual_col = cols_lower[candidate]
            logger.info(f"Found ingredients column: '{actual_col}' (matched '{candidate}')")
            return actual_col
    logger.warning(f"Ingredients column not found. Available columns: {list(df.columns)[:10]}")
    return None


def process_dataset(
    src_path: Path,
    dataset_id: int,
    logger: logging.Logger,
    cuisine_default: str = "unknown",
    cuisine_vocab_path: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    try:
        if src_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(src_path)
            logger.info(f"Read {src_path.name} as parquet ({len(df):,} rows)")
        else:
            df = read_csv_with_fallback(src_path, logger)
        if df.empty:
            logger.warning(f"{src_path.name} is empty, skipping")
            return None
        df.columns = df.columns.str.lower()
        ing_col = find_ingredients_column(df, logger)
        if ing_col is None:
            logger.error(f"Could not find ingredients column in {src_path.name}, skipping")
            return None

        # Try to find cuisine column
        cuisine_col = None
        cols_lower = {col.lower(): col for col in df.columns}
        cuisine_candidates = ["cuisine", "cuisines", "country", "countries", "type", "category", "region"]
        for candidate in cuisine_candidates:
            if candidate in cols_lower:
                cuisine_col = cols_lower[candidate]
                logger.info(f"Found cuisine column: '{cuisine_col}' (matched '{candidate}')")
                break

        # Extract cuisine
        if cuisine_col:
            cuisine_values = df[cuisine_col].astype(str)

            def extract_cuisine(val):
                s = str(val).strip()
                if not s or s.lower() in ["nan", "none", ""]:
                    return cuisine_default
                if s == "[]" or s.lower() == "[]":
                    return cuisine_default
                if s.startswith("[") and s.endswith("]"):
                    try:
                        import ast
                        parsed = ast.literal_eval(s)
                        if isinstance(parsed, list):
                            non_empty = [str(x).strip() for x in parsed if str(x).strip() and str(x).strip().lower() not in ["nan", "none", ""]]
                            if non_empty:
                                return str(non_empty) if len(non_empty) > 1 else non_empty[0]
                            else:
                                return cuisine_default
                    except:
                        pass
                return s if s else cuisine_default

            cuisine_series = cuisine_values.apply(extract_cuisine)
            non_default = (cuisine_series != cuisine_default).sum()
            logger.info(f"Extracted cuisine values: {non_default:,} non-default out of {len(cuisine_series):,} rows")
        else:
            logger.info(f"No cuisine column found in {src_path.name}, using default: '{cuisine_default}'")
            cuisine_series = pd.Series([cuisine_default] * len(df), dtype=str)

        # Vectorized cuisine extraction for missing values
        empty_mask = (cuisine_series == cuisine_default) | (cuisine_series.astype(str).str.strip() == "[]")
        if empty_mask.sum() > 0:
            known_cuisines = set()
            vocab_path = Path(cuisine_vocab_path) if cuisine_vocab_path else None
            if vocab_path and vocab_path.exists():
                try:
                    with open(vocab_path, "r", encoding="utf-8") as f:
                        cuisine_vocab = json.load(f)
                        known_cuisines = {
                            k.lower().strip() for k in cuisine_vocab.keys() if k != "<UNK>"
                        }
                        logger.info("Loaded %s known cuisines from vocabulary", len(known_cuisines))
                except Exception:
                    logger.warning("Failed to load cuisine vocabulary from %s", vocab_path)

            # Add common cuisines
            known_cuisines.update({
                "american", "mexican", "italian", "chinese", "japanese", "indian", "thai",
                "french", "greek", "spanish", "mediterranean", "asian", "korean", "vietnamese",
                "german", "british", "irish", "cuban", "caribbean", "brazilian", "african",
                "middle eastern", "middleeastern", "arab", "arabic", "persian", "turkish",
                "moroccan", "lebanese", "israeli", "jewish", "scandinavian", "nordic",
                "eastern european", "russian", "polish", "cajun", "creole", "southern",
                "tex-mex", "texmex", "latin", "south american", "peruvian", "argentinian"
            })

            # Select text columns
            exclude_cols = {ing_col.lower(), "cuisine", "cuisines", "country", "countries"}
            text_columns = [c for c in df.columns if c.lower() not in exclude_cols]
            for col in text_columns:
                df[col + "_lc"] = df[col].astype(str).str.lower()

            # Compile regex patterns
            pattern_dict = {c: re.compile(rf'\b{re.escape(c)}\b') for c in known_cuisines}

            def vectorized_cuisine_search(row):
                for col in text_columns:
                    val = row[col + "_lc"]
                    for c, pat in pattern_dict.items():
                        if pat.search(val):
                            return c
                return cuisine_default

            cuisine_series.loc[empty_mask] = df.loc[empty_mask].apply(vectorized_cuisine_search, axis=1)
            df.drop(columns=[col + "_lc" for col in text_columns], inplace=True)

        # Build result DataFrame
        result = pd.DataFrame({
            "Dataset_ID": dataset_id,
            "index": df.index,
            "ingredients": df[ing_col].astype(str),
            "cuisine": cuisine_series,
        })

        before_drop = len(result)
        empty_mask_final = (
            (result["cuisine"] == cuisine_default) |
            (result["cuisine"].astype(str).str.strip() == "") |
            (result["cuisine"].astype(str).str.strip() == "[]") |
            (result["cuisine"].astype(str).str.strip().isin(["nan", "None", "null"]))
        )
        result = result[~empty_mask_final].copy()
        dropped_count = before_drop - len(result)
        if dropped_count > 0:
            logger.info(f"Dropped {dropped_count:,} rows without cuisine labels")
        logger.info(f"Processed {src_path.name}: {len(result):,} rows after drop")
        return result

    except Exception as e:
        logger.exception(f"Error processing {src_path.name}: {e}")
        return None


def run(
    context: PipelineContext,
    *,
    force: bool = False,
    run_inference: bool | None = None,
) -> StageResult:
    logger = stage_logger(context, "combine_raw", force=force)
    cfg = context.stage("combine_raw")

    data_dir = Path(str(cfg.get("data_dir", "/data/raw")))
    output_path = Path(str(cfg.get("output_path", "/data/combined_raw_datasets.parquet")))
    cuisine_default = cfg.get("cuisine_default", "unknown")
    excluded = set(cfg.get("excluded_files", []))
    cuisine_vocab_path = cfg.get("cuisine_vocab_path")

    logger.info("Combining raw datasets from %s", data_dir)
    if not data_dir.exists():
        details = f"Data directory not found: {data_dir}"
        logger.error(details)
        return StageResult(name="combine_raw", status="failed", details=details)

    files = sorted([f for f in data_dir.glob("*.*") if f.suffix.lower() in {".csv", ".parquet"} and f.name not in excluded])
    logger.info("Found %s raw files to process", len(files))

    writer: Optional[pq.ParquetWriter] = None
    total_rows = 0
    for dataset_id, src_path in enumerate(files, start=1):
        logger.info("[%s/%s] Processing %s", dataset_id, len(files), src_path.name)
        df_processed = process_dataset(
            src_path,
            dataset_id,
            logger,
            cuisine_default=cuisine_default,
            cuisine_vocab_path=Path(cuisine_vocab_path) if cuisine_vocab_path else None,
        )
        if df_processed is not None and not df_processed.empty:
            table = pa.Table.from_pandas(df_processed, preserve_index=False)
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(str(output_path), table.schema, compression="zstd")
            writer.write_table(table)
            total_rows += len(df_processed)

    if writer is None:
        details = "No datasets successfully processed."
        logger.error(details)
        return StageResult(name="combine_raw", status="failed", details=details)

    writer.close()
    logger.info("Saved combined dataset to %s (%s rows)", output_path, total_rows)

    outputs = {"combined_path": str(output_path), "rows": total_rows}

    inference_cfg = cfg.get("inference", {}) or {}
    should_infer = bool(inference_cfg.get("enabled", False))
    if run_inference is not None:
        should_infer = bool(run_inference)

    if should_infer:
        logger.warning(
            "combine_raw inference is deprecated. Please run the dedicated ingredient_ner_infer stage instead."
        )

    return StageResult(name="combine_raw", status="success", outputs=outputs)
