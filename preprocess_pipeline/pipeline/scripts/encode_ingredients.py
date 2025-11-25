"""
Script to encode already-inferred ingredients to IDs.

This script takes a dataset with inferred_ingredients (normalized/canonical forms)
and encodes them to ingredient IDs using the token-to-ID mapping.
This is separate from inference to allow faster iteration when only encoding changes.
"""

import pyarrow.parquet as pq
import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyarrow.parquet as pq

import sys as _sys
# Make `pipeline/` importable when running from repo root
_SYS_PATH_ROOT = Path.cwd() / "pipeline"
if str(_SYS_PATH_ROOT) not in _sys.path:
    _sys.path.append(str(_SYS_PATH_ROOT))

from common.logging_setup import setup_logging


def load_encoder_maps(
    id2tok_path: Optional[Path],
    tok2id_path: Optional[Path],
) -> tuple[Optional[dict], Optional[dict]]:
    """Load IngredientEncoder id2tok / tok2id maps."""
    if not id2tok_path or not tok2id_path:
        return None, None
    if (not Path(id2tok_path).exists()) or (not Path(tok2id_path).exists()):
        return None, None
    with open(id2tok_path, "r", encoding="utf-8") as f:
        id2tok_raw = json.load(f)
    with open(tok2id_path, "r", encoding="utf-8") as f:
        tok2id_raw = json.load(f)
    id2tok = {int(k): str(v) for k, v in id2tok_raw.items()}
    tok2id = {str(k): int(v) for k, v in tok2id_raw.items()}
    return id2tok, tok2id


def encode_ingredient_list(ingredient_list, tok2id: dict) -> List[int]:
    """
    Encode a list of normalized ingredients to IDs.
    Handles various input formats: list, numpy array, tuple, string representation of list, None, etc.
    """
    import numpy as np
    
    # Handle None or NaN
    if ingredient_list is None or (isinstance(ingredient_list, float) and pd.isna(ingredient_list)):
        return []
    
    # Convert numpy array to list
    if isinstance(ingredient_list, np.ndarray):
        ingredient_list = ingredient_list.tolist()
    
    # If it's a string, try to parse it as a list
    if isinstance(ingredient_list, str):
        s = ingredient_list.strip()
        if not s or s.lower() in ["none", "nan", "null", "[]"]:
            return []
        # Try to parse as JSON or Python list
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                import ast
                ingredient_list = ast.literal_eval(s)
            except:
                try:
                    ingredient_list = json.loads(s)
                except:
                    # If parsing fails, treat as single string
                    ingredient_list = [s]
        else:
            # Single string, not a list
            ingredient_list = [s]
    
    # Now ingredient_list should be a list-like object
    if not isinstance(ingredient_list, (list, tuple, np.ndarray)):
        return []
    
    encoded = []
    for ing in ingredient_list:
        if ing is None:
            continue
        ing_str = str(ing).strip()
        if not ing_str:
            continue
        # Try exact match first
        ing_id = tok2id.get(ing_str, None)
        if ing_id is None:
            # Try with different whitespace normalization
            ing_normalized = " ".join(ing_str.split())
            ing_id = tok2id.get(ing_normalized, 0)
        encoded.append(ing_id)
    
    return encoded if encoded else []


def main():
    parser = argparse.ArgumentParser(
        description="Encode inferred ingredients to IDs"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input parquet file with inferred_ingredients column",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output parquet file (will add/update encoded_ingredients column)",
    )
    parser.add_argument(
        "--ingredient-token-to-id",
        type=str,
        default="./data/encoded/ingredient_token_to_id.json",
        help="Path to token-to-ID mapping JSON (default: ./data/encoded/ingredient_token_to_id.json)",
    )
    parser.add_argument(
        "--ingredient-id-to-token",
        type=str,
        default="./data/encoded/ingredient_id_to_token.json",
        help="Path to ID-to-token mapping JSON (default: ./data/encoded/ingredient_id_to_token.json)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config file for logging",
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
    
    # Load encoder maps
    tok2id_path = Path(args.ingredient_token_to_id)
    id2tok_path = Path(args.ingredient_id_to_token)
    
    if not tok2id_path.exists():
        logger.error(f"Token-to-ID map not found: {tok2id_path}")
        return 1
    
    logger.info(f"Loading encoder maps...")
    id2tok, tok2id = load_encoder_maps(id2tok_path, tok2id_path)
    
    if not tok2id:
        logger.error("Failed to load token-to-ID mapping")
        return 1
    
    logger.info(f"Loaded token→ID map: {len(tok2id):,} tokens")
    
    # Load input data
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    logger.info(f"Loading input data from {input_path}...")
    
    # Read parquet file
    pf = pq.ParquetFile(str(input_path))
    logger.info(f"Input file has {pf.num_row_groups} row groups")
    
    # Check if inferred_ingredients column exists
    schema = pf.schema_arrow
    col_names = [field.name for field in schema]
    
    if "inferred_ingredients" not in col_names:
        logger.error("Column 'inferred_ingredients' not found in input file")
        logger.error(f"Available columns: {col_names}")
        return 1
    
    # Process row groups and encode
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Encoding ingredients...")
    
    # Read all data (or process in chunks if too large)
    # Use pyarrow to preserve list types correctly
    pf = pq.ParquetFile(str(input_path))
    
    # Read in chunks to handle large files
    chunks = []
    for rg in range(pf.num_row_groups):
        df_chunk = pf.read_row_group(rg).to_pandas()
        chunks.append(df_chunk)
    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"Loaded {len(df):,} rows")
    
    # Debug: Check data types and sample values
    logger.info("Debugging data format...")
    logger.info(f"  Column dtypes: {df.dtypes.to_dict()}")
    
    # Check sample of inferred_ingredients
    import numpy as np
    sample_ing = df["inferred_ingredients"].iloc[0] if len(df) > 0 else None
    logger.info(f"  Sample inferred_ingredients (row 0):")
    logger.info(f"    Type: {type(sample_ing)}")
    logger.info(f"    Value: {sample_ing}")
    
    # Convert to list if numpy array
    if isinstance(sample_ing, np.ndarray):
        sample_ing_list = sample_ing.tolist()
    elif isinstance(sample_ing, (list, tuple)):
        sample_ing_list = list(sample_ing)
    else:
        sample_ing_list = []
    
    if len(sample_ing_list) > 0:
        logger.info(f"    First item: '{sample_ing_list[0]}' (type: {type(sample_ing_list[0])})")
        # Check if it's in tok2id
        first_key = str(sample_ing_list[0]).strip()
        in_vocab = first_key in tok2id
        logger.info(f"    In tok2id: {in_vocab}")
        if in_vocab:
            logger.info(f"    ID: {tok2id[first_key]}")
        else:
            # Show some sample keys from tok2id
            sample_keys = list(tok2id.keys())[:10]
            logger.info(f"    Sample tok2id keys: {sample_keys}")
            # Try to find similar keys
            first_lower = first_key.lower()
            similar = [k for k in tok2id.keys() if first_lower in k.lower() or k.lower() in first_lower][:5]
            if similar:
                logger.info(f"    Similar keys in vocab: {similar}")
    elif sample_ing is not None:
        logger.warning(f"    Sample is not a list/array! Type: {type(sample_ing)}, Value: {sample_ing}")
    
    # Encode ingredients
    logger.info("Encoding inferred_ingredients to IDs...")
    encoded_lists = []
    zero_count = 0
    total_ingredients = 0
    sample_issues = []  # Track sample encoding issues for debugging
    
    for idx, ing_list in enumerate(df["inferred_ingredients"]):
        if idx % 10000 == 0 and idx > 0:
            logger.info(f"Processed {idx:,} / {len(df):,} rows...")
        
        # Debug first few entries
        if idx < 5:
            logger.debug(f"Row {idx}: type={type(ing_list)}, value={ing_list}")
            if isinstance(ing_list, (list, tuple)) and len(ing_list) > 0:
                logger.debug(f"  First ingredient: '{ing_list[0]}' (type={type(ing_list[0])})")
                # Check if first ingredient is in tok2id
                first_ing = str(ing_list[0]).strip()
                if first_ing in tok2id:
                    logger.debug(f"  ✓ Found in tok2id: {tok2id[first_ing]}")
                else:
                    logger.debug(f"  ✗ Not found in tok2id. Sample keys: {list(tok2id.keys())[:5]}")
        
        encoded = encode_ingredient_list(ing_list, tok2id)
        encoded_lists.append(encoded)
        
        # Count zeros for diagnostics
        zeros = sum(1 for x in encoded if x == 0)
        zero_count += zeros
        total_ingredients += len(encoded)
        
        # Track issues for first few rows
        if idx < 5 and len(encoded) == 0 and ing_list is not None:
            if isinstance(ing_list, (list, tuple)) and len(ing_list) > 0:
                sample_issues.append((idx, ing_list[:3], encoded))
    
    # Add encoded column
    df["encoded_ingredients"] = encoded_lists
    
    # Log statistics
    logger.info(f"Encoding complete:")
    logger.info(f"  Total rows: {len(df):,}")
    logger.info(f"  Total ingredients: {total_ingredients:,}")
    if total_ingredients > 0:
        logger.info(f"  Ingredients with ID=0 (unknown): {zero_count:,} ({zero_count/total_ingredients*100:.1f}%)")
    else:
        logger.warning("  No ingredients were encoded! This suggests a data format issue.")
        logger.warning("  Checking sample data...")
        
        # Show sample of what we're trying to encode
        sample_df = df[["inferred_ingredients"]].head(10)
        for idx, row in sample_df.iterrows():
            ing_list = row["inferred_ingredients"]
            logger.warning(f"  Row {idx}: type={type(ing_list)}, value={ing_list}")
            if isinstance(ing_list, (list, tuple)) and len(ing_list) > 0:
                first = str(ing_list[0]).strip()
                logger.warning(f"    First item: '{first}' (in tok2id: {first in tok2id})")
                # Show similar keys
                if first not in tok2id:
                    # Find similar keys
                    similar = [k for k in tok2id.keys() if first.lower() in k.lower() or k.lower() in first.lower()][:3]
                    if similar:
                        logger.warning(f"    Similar keys in tok2id: {similar}")
    
    logger.info(f"  First few encoded ingredients:")
    logger.info(f"\n{df[['inferred_ingredients', 'encoded_ingredients']].head()}")
    logger.info(f"Saving to {output_path}...")
    df.to_parquet(output_path, index=False, compression="zstd")
    logger.info(f"Saved encoded dataset to {output_path}")
    
    logger.info("=" * 60)
    logger.info("Encoding complete")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())

