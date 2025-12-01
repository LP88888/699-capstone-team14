"""Encode normalized ingredients to integer IDs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..core import PipelineContext, StageResult
from ..utils import stage_logger


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


def run(
    context: PipelineContext,
    *,
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    force: bool = False,
) -> StageResult:
    cfg = context.stage("ingredient_encoding")
    logger = stage_logger(context, "ingredient_encoding", force=force)

    try:
        input_path = Path(input_path or cfg.get("input_path", ""))
        output_path = Path(output_path or cfg.get("output_path", ""))
        inferred_column = cfg.get("inferred_column", "inferred_ingredients")
        encoded_column = cfg.get("encoded_column", "encoded_ingredients")
        tok2id_path = Path(cfg.get("ingredient_token_to_id", ""))
        id2tok_path = Path(cfg.get("ingredient_id_to_token", ""))

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if not tok2id_path.exists():
            raise FileNotFoundError(f"Token-to-ID map not found: {tok2id_path}")

        logger.info("Loading encoder maps from %s", tok2id_path)
        _, tok2id = load_encoder_maps(id2tok_path, tok2id_path)
        if not tok2id:
            raise ValueError("Failed to load token-to-ID mapping. Ensure encoder maps exist.")

        logger.info("Reading %s", input_path)
        pf = pq.ParquetFile(str(input_path))
        if inferred_column not in [field.name for field in pf.schema_arrow]:
            raise KeyError(f"Column '{inferred_column}' not found in {input_path}")

        writer: Optional[pq.ParquetWriter] = None
        total_rows = 0
        zero_count = 0
        total_ingredients = 0

        for rg_idx in range(pf.num_row_groups):
            df_chunk = pf.read_row_group(rg_idx).to_pandas()
            encoded_lists = []
            for ing_list in df_chunk[inferred_column]:
                encoded = encode_ingredient_list(ing_list, tok2id)
                encoded_lists.append(encoded)
                zero_count += sum(1 for token in encoded if token == 0)
                total_ingredients += len(encoded)
            df_chunk[encoded_column] = encoded_lists

            table = pa.Table.from_pandas(df_chunk, preserve_index=False)
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(str(output_path), table.schema, compression="zstd")
            writer.write_table(table)
            total_rows += len(df_chunk)

            if rg_idx % 5 == 0:
                logger.info(
                    "Encoded chunks %s/%s (%s rows so far)",
                    rg_idx + 1,
                    pf.num_row_groups,
                    total_rows,
                )

        if writer:
            writer.close()
        else:
            raise ValueError(f"No row groups found in {input_path}")

        logger.info(
            "Encoding complete: %s rows, %s ingredients, %s unknown tokens",
            total_rows,
            total_ingredients,
            zero_count,
        )
        logger.info("Saved encoded dataset to %s", output_path)

        return StageResult(
            name="ingredient_encoding",
            status="success",
            outputs={
                "encoded_path": str(output_path),
                "rows": total_rows,
                "total_ingredients": total_ingredients,
                "unknown_tokens": zero_count,
            },
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("Ingredient encoding failed: %s", exc)
        return StageResult(name="ingredient_encoding", status="failed", details=str(exc))