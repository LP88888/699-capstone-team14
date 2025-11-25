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
# Make `pipeline/` importable when running from repo root
_SYS_PATH_ROOT = Path.cwd() / "pipeline"
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
    """
    Tries to read CSV using UTF-8, then cp1252 (windows-1252), then latin-1.
    Returns a DataFrame and the encoding used (string).
    """
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

    # Last resort: read with errors='replace'
    logger.warning(f"All standard encodings failed for {path.name}. Trying utf-8 with errors='replace'.")
    df = pd.read_csv(path, encoding="utf-8", errors='replace', dtype=str, low_memory=False)
    logger.info(f"Read {path.name} with utf-8 (errors=replace) ({len(df):,} rows)")
    return df


def find_ingredients_column(df: pd.DataFrame, logger: logging.Logger) -> Optional[str]:
    """Find the ingredients column (case-insensitive)."""
    cols_lower = {col.lower(): col for col in df.columns}
    
    # Common variations
    candidates = [
        "ingredients",
        "ingredient",
        "ing",
        "ingr",
        "ingredient_list",
        "ingredients_list",
    ]
    
    for candidate in candidates:
        if candidate in cols_lower:
            actual_col = cols_lower[candidate]
            logger.info(f"Found ingredients column: '{actual_col}' (matched '{candidate}')")
            return actual_col
    
    # If not found, log available columns
    logger.warning(f"Ingredients column not found. Available columns: {list(df.columns)[:10]}")
    return None


def process_dataset(
    csv_path: Path,
    dataset_id: int,
    logger: logging.Logger,
    cuisine_default: str = "unknown",
) -> Optional[pd.DataFrame]:
    """
    Process a single CSV dataset.
    
    Returns:
        DataFrame with columns: Dataset_ID, index, ingredients, cuisine
        or None if processing fails
    """
    try:
        # Read CSV
        df = read_csv_with_fallback(csv_path, logger)
        
        if df.empty:
            logger.warning(f"{csv_path.name} is empty, skipping")
            return None
        
        # Convert column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Find ingredients column
        ing_col = find_ingredients_column(df, logger)
        if ing_col is None:
            logger.error(f"Could not find ingredients column in {csv_path.name}, skipping")
            return None
        
        # Try to find cuisine column (case-insensitive)
        cuisine_col = None
        cols_lower = {col.lower(): col for col in df.columns}
        cuisine_candidates = ["cuisine", "cuisines", "country", "countries", "type", "category", "region"]
        for candidate in cuisine_candidates:
            if candidate in cols_lower:
                cuisine_col = cols_lower[candidate]
                logger.info(f"Found cuisine column: '{cuisine_col}' (matched '{candidate}')")
                break
        
        # Debug: log all columns if no cuisine column found
        if cuisine_col is None:
            logger.warning(f"No cuisine column found in {csv_path.name}. Available columns: {list(df.columns)}")
        
        # Extract cuisine values if column exists, otherwise use default
        if cuisine_col:
            # Handle list-like cuisine values (e.g., "[American]" -> "American")
            cuisine_values = df[cuisine_col].astype(str)
            # Try to parse as list and extract first element
            def extract_cuisine(val):
                s = str(val).strip()
                if not s or s.lower() in ["nan", "none", ""]:
                    return cuisine_default
                # Check for empty list string
                if s == "[]" or s.lower() == "[]":
                    return cuisine_default
                # Try to parse as list
                if s.startswith("[") and s.endswith("]"):
                    try:
                        import ast
                        parsed = ast.literal_eval(s)
                        if isinstance(parsed, list):
                            if len(parsed) == 0:
                                return cuisine_default
                            # Extract all non-empty elements and join them
                            non_empty = [str(x).strip() for x in parsed if str(x).strip() and str(x).strip().lower() not in ["nan", "none", ""]]
                            if non_empty:
                                # Return as list string format for consistency
                                return str(non_empty) if len(non_empty) > 1 else non_empty[0]
                            else:
                                return cuisine_default
                    except Exception as e:
                        logger.debug(f"Could not parse cuisine value '{s}' as list: {e}")
                        # If parsing fails, return the string as-is (might be a single cuisine)
                        pass
                return s if s else cuisine_default
            cuisine_series = cuisine_values.apply(extract_cuisine)
            
            # Debug: log cuisine extraction stats
            non_default = (cuisine_series != cuisine_default).sum()
            logger.info(f"Extracted cuisine values: {non_default:,} non-default out of {len(cuisine_series):,} rows")
            if non_default > 0:
                sample_cuisines = cuisine_series[cuisine_series != cuisine_default].head(5).tolist()
                logger.debug(f"Sample cuisine values: {sample_cuisines}")
        else:
            logger.info(f"No cuisine column found in {csv_path.name}, using default: '{cuisine_default}'")
            cuisine_series = pd.Series([cuisine_default] * len(df), dtype=str)
        
        # If cuisine is empty/default, try to extract from all text columns
        empty_cuisine_mask = (cuisine_series == cuisine_default) | (cuisine_series.astype(str).str.strip() == "[]")
        if empty_cuisine_mask.sum() > 0:
            # Load known cuisine vocabulary for matching
            cuisine_vocab_path = Path("./data/cuisine_encoded/cuisine_token_to_id.json")
            known_cuisines = set()
            if cuisine_vocab_path.exists():
                try:
                    with open(cuisine_vocab_path, "r", encoding="utf-8") as f:
                        cuisine_vocab = json.load(f)
                        # Get all cuisine names (lowercase for matching)
                        known_cuisines = {k.lower().strip() for k in cuisine_vocab.keys() if k != "<UNK>"}
                        logger.info(f"Loaded {len(known_cuisines)} known cuisines from vocabulary for text matching")
                except Exception as e:
                    logger.warning(f"Could not load cuisine vocabulary: {e}")
            
            # Also add common cuisine/country names that might appear in text
            common_cuisines = {
                "american", "mexican", "italian", "chinese", "japanese", "indian", "thai",
                "french", "greek", "spanish", "mediterranean", "asian", "korean", "vietnamese",
                "german", "british", "irish", "cuban", "caribbean", "brazilian", "african",
                "middle eastern", "middleeastern", "arab", "arabic", "persian", "turkish",
                "moroccan", "lebanese", "israeli", "jewish", "scandinavian", "nordic",
                "eastern european", "russian", "polish", "cajun", "creole", "southern",
                "tex-mex", "texmex", "latin", "south american", "peruvian", "argentinian"
            }
            known_cuisines.update(common_cuisines)
            
            # Identify text columns to search (exclude cuisine column and ingredients column)
            text_columns = []
            exclude_cols = {ing_col.lower(), "cuisine", "cuisines", "country", "countries"}
            for col in df.columns:
                col_lower = col.lower()
                if col_lower not in exclude_cols:
                    # Check if column contains text data
                    sample_val = df[col].dropna().head(1)
                    if len(sample_val) > 0:
                        val_str = str(sample_val.iloc[0])
                        # Consider it a text column if it's not purely numeric and has reasonable length
                        if len(val_str) > 3 and not val_str.replace(".", "").replace("-", "").isdigit():
                            text_columns.append(col)
            
            logger.info(f"Searching for cuisine in {len(text_columns)} text columns: {text_columns[:5]}{'...' if len(text_columns) > 5 else ''}")
            
            def extract_cuisine_from_text(row_data, cuisine_val):
                """Extract cuisine from any text column if cuisine is empty/default."""
                # If cuisine already has a value, use it
                if cuisine_val and cuisine_val != cuisine_default and str(cuisine_val).strip() not in ["", "[]", "nan"]:
                    return cuisine_val
                
                found_cuisines = []
                
                # Search through all text columns
                for col in text_columns:
                    if col not in row_data.index:
                        continue
                    
                    text_val = row_data[col]
                    if pd.isna(text_val) or not str(text_val).strip():
                        continue
                    
                    text_str = str(text_val).lower()
                    
                    # Handle list-like strings (e.g., tags, categories)
                    if text_str.startswith("[") and text_str.endswith("]"):
                        try:
                            import ast
                            parsed = ast.literal_eval(str(text_val))
                            if isinstance(parsed, list):
                                text_parts = [str(t).strip().lower() for t in parsed]
                            else:
                                text_parts = [text_str]
                        except:
                            # Split by comma if not a valid list
                            text_parts = [t.strip().lower() for t in text_str.split(",")]
                    else:
                        # For regular text, split by common delimiters and also search as whole
                        text_parts = [text_str]
                        # Also add comma-split parts
                        text_parts.extend([t.strip().lower() for t in text_str.split(",")])
                        # Also add space-split parts (for recipe names, descriptions)
                        text_parts.extend([t.strip().lower() for t in text_str.split()])
                    
                    # Check each text part against known cuisines
                    for text_part in text_parts:
                        text_clean = text_part.strip()
                        if not text_clean or len(text_clean) < 3:
                            continue
                        
                        # Direct exact match
                        if text_clean in known_cuisines:
                            if text_clean not in found_cuisines:
                                found_cuisines.append(text_clean)
                        else:
                            # Word boundary matching (cuisine name as whole word)
                            for known_cuisine in known_cuisines:
                                # Use word boundaries to avoid partial matches (e.g., "indian" in "indiana")
                                pattern = r'\b' + re.escape(known_cuisine) + r'\b'
                                if re.search(pattern, text_clean):
                                    if known_cuisine not in found_cuisines:
                                        found_cuisines.append(known_cuisine)
                                    break
                            
                            # Also check substring match (but prefer word boundary matches)
                            if not found_cuisines:
                                for known_cuisine in known_cuisines:
                                    if known_cuisine in text_clean or text_clean in known_cuisine:
                                        if known_cuisine not in found_cuisines:
                                            found_cuisines.append(known_cuisine)
                                        break
                    
                    # If we found a match, stop searching (take first match)
                    if found_cuisines:
                        break
                
                if found_cuisines:
                    # Return first match, or list if multiple
                    return found_cuisines[0] if len(found_cuisines) == 1 else str(found_cuisines)
                return cuisine_default
            
            # Apply text-based extraction for rows with empty cuisine
            logger.info(f"Attempting to extract cuisine from all text columns for {empty_cuisine_mask.sum():,} rows with empty cuisine...")
            cuisine_from_text = pd.Series([
                extract_cuisine_from_text(df.iloc[i], cuisine_series.iloc[i])
                for i in range(len(df))
            ])
            # Update only rows that had empty cuisine
            cuisine_series[empty_cuisine_mask] = cuisine_from_text[empty_cuisine_mask]
            
            # Log how many were found
            found_in_text = (cuisine_from_text[empty_cuisine_mask] != cuisine_default).sum()
            logger.info(f"Found cuisine in text columns for {found_in_text:,} rows (out of {empty_cuisine_mask.sum():,} empty)")
            if found_in_text > 0:
                sample_found = cuisine_from_text[empty_cuisine_mask & (cuisine_from_text != cuisine_default)].head(5).tolist()
                logger.info(f"Sample cuisines found in text: {sample_found}")
        
        # Create result DataFrame
        result = pd.DataFrame({
            "Dataset_ID": dataset_id,
            "index": df.index,
            "ingredients": df[ing_col].astype(str),
            "cuisine": cuisine_series,
        })
        
        # Drop rows where cuisine is still empty/default (no way to add cuisine label)
        before_drop = len(result)
        # Filter out rows with empty/default cuisine
        empty_cuisine_mask = (
            (result["cuisine"] == cuisine_default) | 
            (result["cuisine"].astype(str).str.strip() == "") |
            (result["cuisine"].astype(str).str.strip() == "[]") |
            (result["cuisine"].astype(str).str.strip().isin(["nan", "None", "null"]))
        )
        result = result[~empty_cuisine_mask].copy()
        dropped_count = before_drop - len(result)
        
        if dropped_count > 0:
            logger.info(f"Dropped {dropped_count:,} rows without cuisine labels (out of {before_drop:,} total)")
        logger.info(f"Processed {csv_path.name}: {len(result):,} rows (after dropping rows without cuisine)")
        return result
        
    except Exception as e:
        logger.exception(f"Error processing {csv_path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Combine raw datasets into unified format"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/raw",
        help="Directory containing raw CSV files (default: ./data/raw)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/combined_raw_datasets.parquet",
        help="Output path for combined dataset (default: ./data/combined_raw_datasets.parquet)",
    )
    parser.add_argument(
        "--cuisine-default",
        type=str,
        default="unknown",
        help="Default cuisine value (default: unknown)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config file for logging",
    )
    parser.add_argument(
        "--inference-config",
        type=str,
        default="./pipeline/config/ingredient_ner_inference.yaml",
        help="Path to inference config YAML (default: ./pipeline/config/ingredient_ner_inference.yaml)",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip ingredient NER inference (default: run inference)",
    )
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.config) if args.config else logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger(__name__)
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1
    
    # Find all CSV files, excluding specified ones
    excluded_files = {
        "wilmerarltstrmberg_data.csv",
        "sample_data.csv",
        "recipe_api_data.csv",
    }
    
    csv_files = sorted([
        f for f in data_dir.glob("*.csv")
        if f.name not in excluded_files
    ])
    
    logger.info(f"Found {len(csv_files)} CSV files to process (excluding .xlsx and {len(excluded_files)} specified files)")
    
    if not csv_files:
        logger.warning("No CSV files found to process")
        return 1
    
    # Process each dataset
    all_dfs: List[pd.DataFrame] = []
    for dataset_id, csv_path in enumerate(csv_files, start=1):
        logger.info(f"[{dataset_id}/{len(csv_files)}] Processing {csv_path.name}...")
        result_df = process_dataset(csv_path, dataset_id, logger, args.cuisine_default)
        if result_df is not None:
            all_dfs.append(result_df)
    
    if not all_dfs:
        logger.error("No datasets were successfully processed")
        return 1
    
    # Combine all DataFrames
    logger.info("Combining all datasets...")
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Final safety check: drop any rows that still don't have cuisine labels
    before_final_drop = len(combined)
    empty_cuisine_mask = (
        (combined["cuisine"] == args.cuisine_default) | 
        (combined["cuisine"].astype(str).str.strip() == "") |
        (combined["cuisine"].astype(str).str.strip() == "[]") |
        (combined["cuisine"].astype(str).str.strip().isin(["nan", "None", "null"]))
    )
    combined = combined[~empty_cuisine_mask].copy()
    final_dropped = before_final_drop - len(combined)
    
    if final_dropped > 0:
        logger.warning(f"Final safety check: dropped {final_dropped:,} additional rows without cuisine labels")
    
    logger.info(f"Combined dataset: {len(combined):,} total rows (after dropping rows without cuisine)")
    logger.info(f"Dataset_ID distribution:")
    for dataset_id, count in combined["Dataset_ID"].value_counts().sort_index().items():
        logger.info(f"  Dataset {dataset_id}: {count:,} rows")
    
    # Initialize inferred_ingredients and encoded_ingredients columns
    combined["inferred_ingredients"] = None
    combined["encoded_ingredients"] = None  # For debugging: ingredient IDs
    
    # Run inference unless skipped
    if not args.skip_inference:
        if not _HAS_INFERENCE:
            logger.error("Inference modules not available. Cannot run inference.")
            return 1
        
        logger.info("=" * 60)
        logger.info("Running ingredient NER inference...")
        logger.info("=" * 60)
        
        try:
            # Load inference config
            inference_config_path = Path(args.inference_config)
            if not inference_config_path.exists():
                logger.error(f"Inference config not found: {inference_config_path}")
                return 1
            
            logger.info(f"Loading inference config from {inference_config_path}")
            load_inference_configs_from_yaml(inference_config_path)
            
            # Save combined dataset temporarily for inference (as parquet)
            temp_path = Path(args.output).with_suffix(".temp.parquet")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(temp_path, index=False, compression="zstd")
            logger.info(f"Saved temporary dataset to {temp_path} for inference")
            
            # Run inference - use predict_normalize_encode_structured directly to ensure parquet is detected
            from ingredient_ner.inference import predict_normalize_encode_structured, load_dedupe_and_maps_from_config
            
            # Load dedupe and maps
            dedupe, tok2id = load_dedupe_and_maps_from_config()
            
            logger.info("Running inference on combined dataset...")
            df_wide, df_tall = predict_normalize_encode_structured(
                nlp_dir=OUT.MODEL_DIR,
                data_path=temp_path,
                is_parquet=True,  # Explicitly set to True since we saved as parquet
                text_col="ingredients",
                dedupe=dedupe,
                tok2id=tok2id,
                out_path=None,  # Don't write separate output files
                batch_size=256,
                n_process=1,
                use_spacy_normalizer=True,
                spacy_model="en_core_web_sm",
            )
            
            # Map inference results back to combined dataset
            # df_wide has the same row order as input, with NER_clean and Ingredients columns
            logger.info("Mapping inference results to combined dataset...")
            if len(df_wide) == len(combined):
                combined["inferred_ingredients"] = df_wide["NER_clean"].tolist()
                # Extract encoded ingredient IDs if available (when tok2id is provided)
                if "Ingredients" in df_wide.columns:
                    combined["encoded_ingredients"] = df_wide["Ingredients"].tolist()
                    logger.info(f"Successfully mapped {len(df_wide):,} inference results (with encoding)")
                else:
                    logger.warning("Ingredients column not found in inference results (encoding not available)")
                    combined["encoded_ingredients"] = None
            else:
                logger.warning(
                    f"Row count mismatch: combined={len(combined)}, inference={len(df_wide)}. "
                    f"Using index-based mapping."
                )
                # Try to match by index if possible
                if "index" in df_wide.columns:
                    # This shouldn't happen, but handle it
                    logger.warning("Index-based mapping not implemented. Inference results may not align.")
                else:
                    # Assume same order
                    combined["inferred_ingredients"] = df_wide["NER_clean"].tolist()[:len(combined)]
                    if "Ingredients" in df_wide.columns:
                        combined["encoded_ingredients"] = df_wide["Ingredients"].tolist()[:len(combined)]
                    else:
                        combined["encoded_ingredients"] = None
            
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
                logger.debug(f"Removed temporary file: {temp_path}")
            
            logger.info("Inference complete")
            
        except Exception as e:
            logger.exception(f"Error during inference: {e}")
            logger.warning("Continuing without inference results...")
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == ".parquet":
        combined.to_parquet(output_path, index=False, compression="zstd")
        logger.info(f"Saved combined dataset to {output_path} (Parquet format)")
    else:
        combined.to_csv(output_path, index=False)
        logger.info(f"Saved combined dataset to {output_path} (CSV format)")
    
    logger.info("=" * 60)
    logger.info("Dataset combination complete")
    logger.info("=" * 60)
    logger.info("\nFirst few rows:")
    logger.info("\n" + str(combined.head()))
    return 0


if __name__ == "__main__":
    exit(main())

