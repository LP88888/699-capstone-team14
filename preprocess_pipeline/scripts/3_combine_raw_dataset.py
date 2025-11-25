"""
Script to combine multiple raw datasets into a unified format.

Processes CSV files from data/raw, extracts ingredients column,
and combines them into a single DataFrame with Dataset_ID, index, ingredients, cuisine.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Optional
import sys as _sys

import pandas as pd
# Make `preprocess_pipeline/` importable when running from repo root
_SYS_PATH_ROOT = Path.cwd() / "preprocess_pipeline"
if str(_SYS_PATH_ROOT) not in _sys.path:
    _sys.path.append(str(_SYS_PATH_ROOT))

from common.logging_setup import setup_logging

# Import inference functionality
try:
    from ingredient_ner.inference import run_full_inference_from_config
    from ingredient_ner.config import load_inference_configs_from_yaml, DATA, OUT
    _HAS_INFERENCE = True
except ImportError as e:
    _HAS_INFERENCE = False
    print(f"Warning: Could not import inference modules: {e}")


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


def process_dataset(csv_path: Path, dataset_id: int, logger: logging.Logger, cuisine_default: str = "unknown") -> Optional[pd.DataFrame]:
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
            # Load known cuisines
            cuisine_vocab_path = Path("./data/cuisine_encoded/cuisine_token_to_id.json")
            known_cuisines = set()
            if cuisine_vocab_path.exists():
                try:
                    with open(cuisine_vocab_path, "r", encoding="utf-8") as f:
                        cuisine_vocab = json.load(f)
                        known_cuisines = {k.lower().strip() for k in cuisine_vocab.keys() if k != "<UNK>"}
                        logger.info(f"Loaded {len(known_cuisines)} known cuisines from vocabulary")
                except:
                    logger.warning("Failed to load cuisine vocabulary")

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


def main():
    parser = argparse.ArgumentParser(description="Combine raw datasets into unified format")
    parser.add_argument("--data-dir", type=str, default="./data/raw")
    parser.add_argument("--output", type=str, default="./data/combined_raw_datasets.parquet")
    parser.add_argument("--cuisine-default", type=str, default="unknown")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--inference-config", type=str, default="./preprocess_pipeline/config/ingredient_ner_inference.yaml")
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()

    logger = setup_logging(args.config) if args.config else logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        logger = logging.getLogger(__name__)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1

    excluded_files = {"wilmerarltstrmberg_data.csv", "sample_data.csv", "recipe_api_data.csv"}
    csv_files = sorted([f for f in data_dir.glob("*.csv") if f.name not in excluded_files])
    logger.info(f"Found {len(csv_files)} CSV files to process")

    all_dfs: List[pd.DataFrame] = []
    for dataset_id, csv_path in enumerate(csv_files, start=1):
        logger.info(f"[{dataset_id}/{len(csv_files)}] Processing {csv_path.name}...")
        df_processed = process_dataset(csv_path, dataset_id, logger, args.cuisine_default)
        if df_processed is not None:
            all_dfs.append(df_processed)

    if not all_dfs:
        logger.error("No datasets successfully processed")
        return 1

    combined = pd.concat(all_dfs, ignore_index=True)
    before_final_drop = len(combined)
    empty_mask_final = (
        (combined["cuisine"] == args.cuisine_default) |
        (combined["cuisine"].astype(str).str.strip().isin(["", "[]", "nan", "None", "null"]))
    )
    combined = combined[~empty_mask_final].copy()
    if before_final_drop - len(combined) > 0:
        logger.warning(f"Dropped {before_final_drop - len(combined)} rows without cuisine labels")

    combined["inferred_ingredients"] = None
    combined["encoded_ingredients"] = None

    # Inference
    if not args.skip_inference:
        if not _HAS_INFERENCE:
            logger.error("Inference modules not available")
            return 1
        try:
            from ingredient_ner.inference import predict_normalize_encode_structured, load_dedupe_and_maps_from_config
            load_inference_configs_from_yaml(Path(args.inference_config))
            temp_path = Path(args.output).with_suffix(".temp.parquet")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(temp_path, index=False, compression="zstd")
            dedupe, tok2id = load_dedupe_and_maps_from_config()
            df_wide, df_tall = predict_normalize_encode_structured(
                nlp_dir=OUT.MODEL_DIR,
                data_path=temp_path,
                is_parquet=True,
                text_col="ingredients",
                dedupe=dedupe,
                tok2id=tok2id,
                out_path=None,
                batch_size=256,
                n_process=1,
                use_spacy_normalizer=True,
                spacy_model="en_core_web_sm",
            )
            if len(df_wide) == len(combined):
                combined["inferred_ingredients"] = df_wide["NER_clean"].tolist()
                combined["encoded_ingredients"] = df_wide["Ingredients"].tolist() if "Ingredients" in df_wide else None
            if temp_path.exists():
                temp_path.unlink()
        except Exception as e:
            logger.exception(f"Inference failed: {e}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        combined.to_parquet(output_path, index=False, compression="zstd")
    else:
        combined.to_csv(output_path, index=False)
    logger.info(f"Saved combined dataset to {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
