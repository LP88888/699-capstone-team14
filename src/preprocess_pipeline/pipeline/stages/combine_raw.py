"""Pipeline stage that combines raw CSV datasets into a single parquet file."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from preprocess_pipeline.ingredient_ner.config import (
    OUT as INFER_OUT,
    load_inference_configs_from_dict,
)
from preprocess_pipeline.ingredient_ner.inference import (
    load_dedupe_and_maps_from_config,
    predict_normalize_encode_structured,
)

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
    csv_path: Path,
    dataset_id: int,
    logger: logging.Logger,
    cuisine_default: str = "unknown",
    cuisine_vocab_path: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    try:
        df = read_csv_with_fallback(csv_path, logger)
        if df.empty:
            logger.warning(f"{csv_path.name} is empty, skipping")
            return None
        df.columns = df.columns.str.lower()
        ing_col = find_ingredients_column(df, logger)
        if ing_col is None:
            logger.error(f"Could not find ingredients column in {csv_path.name}, skipping")
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
            logger.info(f"No cuisine column found in {csv_path.name}, using default: '{cuisine_default}'")
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
        logger.info(f"Processed {csv_path.name}: {len(result):,} rows after drop")
        return result

    except Exception as e:
        logger.exception(f"Error processing {csv_path.name}: {e}")
        return None


def _build_inference_payload(
    inference_cfg: dict,
    temp_path: Path,
    text_col: str,
) -> dict:
    model_dir = inference_cfg.get("model_dir", "./models/ingredient_ner_trf/model-best")
    out_base = inference_cfg.get("out_base")
    if not out_base:
        out_base = str(temp_path.with_suffix("").with_name(temp_path.stem + "_ner"))

    artifacts_cfg = {
        "cosine_map_path": inference_cfg.get("cosine_map_path"),
        "ingredient_id_to_token": inference_cfg.get("ingredient_id_to_token"),
        "ingredient_token_to_id": inference_cfg.get("ingredient_token_to_id"),
    }

    if not all(artifacts_cfg.values()):
        missing = [k for k, v in artifacts_cfg.items() if not v]
        raise ValueError(f"Inference artifacts missing values for {missing}")

    return {
        "model": {"model_dir": model_dir},
        "data": {"input_path": str(temp_path), "data_is_parquet": True},
        "output": {"out_base": out_base},
        "artifacts": artifacts_cfg,
        "inference": {
            "text_col": text_col,
            "batch_size": int(inference_cfg.get("batch_size", 256)),
            "n_process": int(inference_cfg.get("n_process", 1)),
            "use_gpu": bool(inference_cfg.get("use_gpu", False)),
            "use_spacy_normalizer": bool(inference_cfg.get("use_spacy_normalizer", True)),
            "spacy_model": inference_cfg.get("spacy_model", "en_core_web_sm"),
            "sample_n": inference_cfg.get("sample_n"),
            "sample_frac": inference_cfg.get("sample_frac"),
            "head_n": inference_cfg.get("head_n"),
        },
    }


def _run_inference(
    df: pd.DataFrame,
    inference_cfg: dict,
    output_path: Path,
    logger: logging.Logger,
) -> dict:
    text_col = inference_cfg.get("text_col", "ingredients")
    temp_path = output_path.with_suffix(".temp.parquet")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(temp_path, index=False, compression="zstd")

    payload = _build_inference_payload(inference_cfg, temp_path, text_col)
    load_inference_configs_from_dict(payload)
    dedupe_map, tok2id = load_dedupe_and_maps_from_config()
    df_wide, df_tall = predict_normalize_encode_structured(
        nlp_dir=INFER_OUT.MODEL_DIR,
        data_path=temp_path,
        is_parquet=True,
        text_col=text_col,
        dedupe=dedupe_map,
        tok2id=tok2id,
        out_path=None,
        batch_size=int(payload["inference"]["batch_size"]),
        n_process=int(payload["inference"]["n_process"]),
        use_spacy_normalizer=bool(payload["inference"]["use_spacy_normalizer"]),
        spacy_model=payload["inference"]["spacy_model"],
    )

    if temp_path.exists():
        temp_path.unlink()

    if len(df_wide) == len(df):
        df["inferred_ingredients"] = df_wide["NER_clean"].tolist()
        if "Ingredients" in df_wide:
            df["encoded_ingredients"] = df_wide["Ingredients"].tolist()

    logger.info("Inference complete for %s rows", len(df_wide))
    return {
        "inferred_rows": len(df_wide),
        "inferred_entities": len(df_tall),
    }


def run(
    context: PipelineContext,
    *,
    force: bool = False,
    run_inference: bool | None = None,
) -> StageResult:
    logger = stage_logger(context, "combine_raw", force=force)
    cfg = context.stage("combine_raw")

    data_dir = Path(cfg.get("data_dir", "./data/raw"))
    output_path = Path(cfg.get("output_path", "./data/combined_raw_datasets.parquet"))
    cuisine_default = cfg.get("cuisine_default", "unknown")
    excluded = set(cfg.get("excluded_files", []))
    cuisine_vocab_path = cfg.get("cuisine_vocab_path")

    logger.info("Combining raw datasets from %s", data_dir)
    if not data_dir.exists():
        details = f"Data directory not found: {data_dir}"
        logger.error(details)
        return StageResult(name="combine_raw", status="failed", details=details)

    csv_files = sorted([f for f in data_dir.glob("*.csv") if f.name not in excluded])
    logger.info("Found %s CSV files to process", len(csv_files))

    all_dfs: List[pd.DataFrame] = []
    for dataset_id, csv_path in enumerate(csv_files, start=1):
        logger.info("[%s/%s] Processing %s", dataset_id, len(csv_files), csv_path.name)
        df_processed = process_dataset(
            csv_path,
            dataset_id,
            logger,
            cuisine_default=cuisine_default,
            cuisine_vocab_path=Path(cuisine_vocab_path) if cuisine_vocab_path else None,
        )
        if df_processed is not None:
            all_dfs.append(df_processed)

    if not all_dfs:
        details = "No datasets successfully processed."
        logger.error(details)
        return StageResult(name="combine_raw", status="failed", details=details)

    combined = pd.concat(all_dfs, ignore_index=True)
    before_final_drop = len(combined)
    empty_mask_final = (
        (combined["cuisine"] == cuisine_default)
        | (combined["cuisine"].astype(str).str.strip().isin(["", "[]", "nan", "None", "null"]))
    )
    combined = combined[~empty_mask_final].copy()
    dropped = before_final_drop - len(combined)
    if dropped > 0:
        logger.warning("Dropped %s rows without cuisine labels", dropped)

    combined["inferred_ingredients"] = None
    combined["encoded_ingredients"] = None

    outputs = {
        "combined_path": str(output_path),
        "rows": len(combined),
    }
    details = None

    inference_cfg = cfg.get("inference", {}) or {}
    should_infer = bool(inference_cfg.get("enabled", True))
    if run_inference is not None:
        should_infer = bool(run_inference)

    if should_infer:
        try:
            inference_outputs = _run_inference(combined, inference_cfg, output_path, logger)
            outputs.update(inference_outputs)
        except Exception as exc:
            details = f"Inference failed: {exc}"
            logger.exception(details)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False, compression="zstd")
    logger.info("Saved combined dataset to %s", output_path)

    status = "success" if details is None else "partial"
    return StageResult(name="combine_raw", status=status, outputs=outputs, details=details)
