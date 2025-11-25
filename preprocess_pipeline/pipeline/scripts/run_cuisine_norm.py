"""
Script to normalize and encode cuisines using the ingrnorm pipeline.

Reuses the same normalization, deduplication, and encoding infrastructure
as run_ingrnorm.py, but applied to the cuisine column.
Handles splitting entries that contain multiple cuisines per row.
"""

from __future__ import annotations
import argparse
import logging
import ast
import json
import os
import glob
from pathlib import Path
import sys as _sys
import pathlib as _pathlib
import yaml
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

# Make `pipeline/` importable when running from repo root
_SYS_PATH_ROOT = _pathlib.Path.cwd() / "pipeline"
if str(_SYS_PATH_ROOT) not in _sys.path:
    _sys.path.append(str(_SYS_PATH_ROOT))

from common.logging_setup import setup_logging
from ingrnorm.io import materialize_parquet_source
from ingrnorm.parquet_utils import vocab_from_parquet_listcol
from ingrnorm.spacy_normalizer import apply_spacy_normalizer_to_parquet
from ingrnorm.sbert_dedupe import sbert_dedupe
from ingrnorm.w2v_dedupe import w2v_dedupe
from ingrnorm.dedupe_map import apply_map_to_parquet_streaming, write_jsonl_map
from ingrnorm.encoder import IngredientEncoder

logger = logging.getLogger("cuisine_norm")


def _cleanup_paths(cfg: dict, logger: logging.Logger, preserve_files: list[str] = None):
    """
    Clean up old artifacts if cleanup is enabled in config.
    By default, deletes parquet files but keeps JSON/JSONL files.
    
    Args:
        cfg: Configuration dict
        logger: Logger instance
        preserve_files: List of file paths/patterns to preserve (e.g., main datasets)
    """
    cleanup_cfg = cfg.get("cleanup", {})
    if not cleanup_cfg.get("enabled", False):
        logger.info("Cleanup disabled – skipping file deletions.")
        return
    
    preserve_files = preserve_files or []
    preserve_patterns = []
    for p in preserve_files:
        try:
            preserve_patterns.append(Path(p).resolve())
        except (OSError, ValueError):
            # If path doesn't exist or can't be resolved, store as string for pattern matching
            preserve_patterns.append(p)
    
    paths = cleanup_cfg.get("paths", [])
    if not paths:
        logger.info("Cleanup enabled but no paths specified.")
        return
    
    deleted_count = 0
    for pattern in paths:
        for f in glob.glob(pattern):
            f_path = Path(f).resolve()
            
            # Skip if file should be preserved
            should_preserve = False
            for preserve in preserve_patterns:
                if isinstance(preserve, Path):
                    if preserve.exists() and f_path.samefile(preserve):
                        should_preserve = True
                        break
                    # Also check if preserve is a pattern that matches
                    if str(f_path).endswith(str(preserve)) or str(preserve) in str(f_path):
                        should_preserve = True
                        break
                else:
                    # String pattern matching
                    if str(preserve) in str(f_path) or str(f_path).endswith(str(preserve)):
                        should_preserve = True
                        break
            
            if should_preserve:
                logger.debug(f"[cleanup] Preserving {f}")
                continue
            
            # Only delete parquet files, keep JSON/JSONL files
            if f.endswith('.parquet'):
                try:
                    os.remove(f)
                    logger.info(f"[cleanup] Deleted {f}")
                    deleted_count += 1
                except FileNotFoundError:
                    pass
                except Exception as e:
                    logger.warning(f"[cleanup] Failed to delete {f}: {e}")
            else:
                logger.debug(f"[cleanup] Keeping non-parquet file: {f}")
    
    if deleted_count > 0:
        logger.info(f"[cleanup] Cleaned up {deleted_count} parquet file(s)")


def _needs_reapply_map(dedupe_map_path: Path, deduped_parquet: Path) -> bool:
    """Check if dedupe map needs to be reapplied (map is newer than deduped parquet)."""
    if not dedupe_map_path.exists():
        return False
    if not deduped_parquet.exists():
        return True
    try:
        map_mtime = dedupe_map_path.stat().st_mtime
        parquet_mtime = deduped_parquet.stat().st_mtime
        return map_mtime > parquet_mtime
    except (OSError, AttributeError):
        return False


def split_cuisine_entries(cuisine_value: str) -> list[str]:
    """
    Split cuisine entries that may contain multiple cuisines.
    Handles formats like:
    - "[American, Italian]" -> ["American", "Italian"]
    - "American, Italian" -> ["American", "Italian"]
    - "American & Italian" -> ["American", "Italian"]
    - "American" -> ["American"]
    - "Goan recipes" -> ["Goan"] (removes "recipes" suffix)
    """
    if pd.isna(cuisine_value) or not str(cuisine_value).strip():
        return []
    
    s = str(cuisine_value).strip()
    
    # Try parsing as list/JSON first
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                cuisines = [str(x).strip() for x in parsed if str(x).strip()]
            else:
                cuisines = [s]
        except:
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    cuisines = [str(x).strip() for x in parsed if str(x).strip()]
                else:
                    cuisines = [s]
            except:
                cuisines = [s]
    else:
        # Try splitting on comma
        if "," in s:
            parts = [x.strip() for x in s.split(",") if x.strip()]
            if len(parts) > 1:
                cuisines = parts
            else:
                cuisines = [s]
        else:
            # Try splitting on " & " or " and "
            found_split = False
            for sep in [" & ", " and ", " AND "]:
                if sep in s:
                    parts = [x.strip() for x in s.split(sep) if x.strip()]
                    if len(parts) > 1:
                        cuisines = parts
                        found_split = True
                        break
            if not found_split:
                # Single value
                cuisines = [s] if s else []
    
    # Clean up each cuisine: remove "recipes" suffix (case-insensitive)
    cleaned = []
    for cuisine in cuisines:
        if not cuisine:
            continue
        # Remove "recipes" from the end (case-insensitive, with optional whitespace)
        cuisine_clean = cuisine.strip()
        # Remove trailing "recipes" or " recipe" (singular or plural)
        for suffix in [" recipes", " recipe", "Recipes", "Recipe"]:
            if cuisine_clean.lower().endswith(suffix.lower()):
                cuisine_clean = cuisine_clean[:-len(suffix)].strip()
                break
        if cuisine_clean:
            cleaned.append(cuisine_clean)
    
    return cleaned


def prepare_cuisine_list_column(
    input_path: Path,
    cuisine_col: str,
    output_path: Path,
    logger: logging.Logger,
) -> Path:
    """
    Prepare cuisine column as a list column for normalization pipeline.
    Splits entries that contain multiple cuisines.
    """
    logger.info(f"Preparing cuisine list column from {input_path}...")
    
    # Read parquet file
    pf = pq.ParquetFile(str(input_path))
    logger.info(f"Input file has {pf.num_row_groups} row groups")
    
    # Process row groups and split cuisines
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    writer = None
    schema = None
    
    for rg_idx in range(pf.num_row_groups):
        logger.info(f"Processing row group {rg_idx + 1}/{pf.num_row_groups}...")
        
        # Read row group
        df = pf.read_row_group(rg_idx).to_pandas()
        
        if cuisine_col not in df.columns:
            logger.error(f"Column '{cuisine_col}' not found in input file")
            logger.error(f"Available columns: {list(df.columns)}")
            raise KeyError(f"Column '{cuisine_col}' not found")
        
        # Split cuisine entries into lists
        logger.info(f"Splitting cuisine entries in row group {rg_idx + 1}...")
        cuisine_lists = df[cuisine_col].apply(split_cuisine_entries)
        
        # Count splits for statistics
        splits = sum(1 for lst in cuisine_lists if len(lst) > 1)
        if splits > 0:
            logger.info(f"  Found {splits:,} entries with multiple cuisines (out of {len(cuisine_lists):,})")
        
        # Create new DataFrame with list column
        df_output = pd.DataFrame({
            cuisine_col: cuisine_lists,
        })
        
        # Convert to PyArrow table
        import pyarrow as pa
        table = pa.Table.from_pandas(df_output, preserve_index=False)
        
        # Set schema (list of strings)
        if schema is None:
            schema = pa.schema([
                pa.field(cuisine_col, pa.list_(pa.string())),
            ])
            writer = pq.ParquetWriter(str(output_path), schema, compression="zstd")
        
        writer.write_table(table)
    
    if writer:
        writer.close()
    
    logger.info(f"Saved prepared cuisine list column to {output_path}")
    return output_path


def main():
    ap = argparse.ArgumentParser(
        description="Normalize and encode cuisines using ingrnorm pipeline"
    )
    ap.add_argument(
        "--config",
        type=str,
        default="pipeline/config/cuisnorm.yaml",
        help="Path to cuisine normalization config YAML (default: pipeline/config/cuisnorm.yaml)",
    )
    ap.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input parquet file with cuisine column (default: from config data.input_path)",
    )
    ap.add_argument(
        "--cuisine-col",
        type=str,
        default="cuisine",
        help="Name of cuisine column (default: cuisine)",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="./data/cuisine_normalized",
        help="Output directory for cuisine normalization artifacts (default: derived from config)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Rebuild artifacts even if they exist",
    )
    args = ap.parse_args()
    
    # Load config
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    setup_logging(cfg)
    logger = logging.getLogger("cuisine_norm")
    
    # Get paths from config (with CLI overrides)
    data_cfg = cfg.get("data", {})
    out_cfg = cfg.get("output", {})
    
    # Validate config and cleanup old artifacts (preserve main dataset)
    combined_dataset_path = Path(data_cfg.get("input_path", "./data/encoded_combined_datasets.parquet"))
    _cleanup_paths(cfg, logger, preserve_files=[str(combined_dataset_path)])
    
    
    # Input path: use CLI arg if provided, otherwise from config
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = Path(data_cfg.get("input_path", "./data/encoded_combined_datasets.parquet"))
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Output directory: use CLI arg if provided, otherwise derive from config
    if args.output_dir != "./data/cuisine_normalized":  # User provided custom path
        output_dir = Path(args.output_dir)
    else:
        # Derive from config output paths
        baseline_path = Path(out_cfg.get("baseline_parquet", "./data/cuisine_normalized/cuisine_baseline.parquet"))
        output_dir = baseline_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Cuisine column: use CLI arg if provided, otherwise from config
    cuisine_col = args.cuisine_col if args.cuisine_col != "cuisine" else data_cfg.get("cuisine_col", "cuisine")
    
    # Get output paths from config
    baseline_parquet = Path(out_cfg.get("baseline_parquet", "./data/cuisine_normalized/cuisine_baseline.parquet"))
    final_dedup_parquet = Path(out_cfg.get("dedup_parquet", "./data/cuisine_normalized/cuisine_deduped.parquet"))
    dedupe_map_path = Path(out_cfg.get("cosine_map_path", "./data/cuisine_normalized/cuisine_dedupe_map.jsonl"))
    list_col_for_vocab = out_cfg.get("list_col_for_vocab", "cuisine_clean")
    unified_parquet = Path(out_cfg.get("unified_parquet", "./data/cuisine_encoded/cuisine_unified.parquet"))
    cuisine_id_to_token = Path(out_cfg.get("cuisine_id_to_token", "./data/cuisine_encoded/cuisine_id_to_token.json"))
    cuisine_token_to_id = Path(out_cfg.get("cuisine_token_to_id", "./data/cuisine_encoded/cuisine_token_to_id.json"))
    
    # Step 1: Prepare cuisine list column (split multi-cuisine entries)
    prepared_parquet = baseline_parquet.parent / "_cuisine_prepared.parquet"
    if args.force or not prepared_parquet.exists():
        logger.info("=" * 60)
        logger.info("Step 1: Preparing cuisine list column (splitting multi-cuisine entries)")
        logger.info("=" * 60)
        prepare_cuisine_list_column(input_path, cuisine_col, prepared_parquet, logger)
    else:
        logger.info("Step 1: Skipped (prepared file exists)")
    
    # Step 2: Apply spaCy normalization (reuse ingrnorm pipeline)
    if args.force or not baseline_parquet.exists():
        logger.info("=" * 60)
        logger.info("Step 2: spaCy normalization")
        logger.info("=" * 60)
        sbert_cfg = cfg.get("sbert", {})
        apply_spacy_normalizer_to_parquet(
            in_parquet=str(prepared_parquet),
            out_parquet=str(baseline_parquet),
            list_col=cuisine_col,
            out_col=list_col_for_vocab,
            spacy_model=sbert_cfg.get("spacy_model", "en_core_web_sm"),
            batch_size=int(sbert_cfg.get("spacy_batch_size", 512)),
            n_process=int(sbert_cfg.get("spacy_n_process", 1)),
        )
        logger.info(f"Saved baseline cuisine Parquet → {baseline_parquet}")
    else:
        logger.info("Step 2: Skipped (baseline file exists)")
    
    # Step 3: Build vocabulary
    min_freq_for_vocab = int(cfg.get("sbert", {}).get("min_freq_for_vocab", 1))
    vocab = vocab_from_parquet_listcol(
        str(baseline_parquet),
        col=list_col_for_vocab,
        min_freq=min_freq_for_vocab,
    )
    logger.info(f"Vocab size: {len(vocab):,} (min_freq={min_freq_for_vocab})")
    
    if len(vocab) == 0:
        logger.warning("Vocabulary is empty. Check cuisine_clean column.")
        return 1
    
    # Step 4: Deduplication (SBERT)
    stages_cfg = cfg.get("stages", {})
    use_sbert = bool(stages_cfg.get("sbert_dedupe", True))
    
    if use_sbert and (args.force or not dedupe_map_path.exists()):
        logger.info("=" * 60)
        logger.info("Step 3: SBERT deduplication")
        logger.info("=" * 60)
        sbert_cfg = cfg.get("sbert", {})
        mapping = sbert_dedupe(
            vocab_counter=vocab,
            out_path=str(dedupe_map_path),
            model_name=sbert_cfg.get("model", "all-MiniLM-L6-v2"),
            threshold=float(sbert_cfg.get("threshold", 0.88)),
            topk=int(sbert_cfg.get("topk", 25)),
            min_len=int(sbert_cfg.get("min_len", 2)),
            require_token_overlap=bool(sbert_cfg.get("require_token_overlap", True)),
            block_generic_as_canon=bool(sbert_cfg.get("block_generic_as_canon", True)),
        )
        write_jsonl_map(mapping, dedupe_map_path)
        logger.info(f"Saved dedupe map → {dedupe_map_path}")
    else:
        logger.info("Step 3: Skipped (dedupe map exists or disabled)")
    
    # Step 5: Apply dedupe map
    # The JSON map is the source of truth - always apply it to the parquet if it exists
    do_apply_map = bool(stages_cfg.get("apply_cosine_map", True))
    if do_apply_map and dedupe_map_path.exists():
        needs_reapply = _needs_reapply_map(dedupe_map_path, final_dedup_parquet)
        logger.info(f"Step 4: Checking dedupe map application...")
        logger.info(f"  - Map exists: {dedupe_map_path.exists()}")
        logger.info(f"  - Deduped parquet exists: {final_dedup_parquet.exists()}")
        logger.info(f"  - Needs reapply: {needs_reapply}")
        logger.info(f"  - Force flag: {args.force}")
        
        if args.force or not final_dedup_parquet.exists() or needs_reapply:
            logger.info("=" * 60)
            logger.info("Step 4: Applying dedupe map to parquet")
            logger.info("=" * 60)
            if needs_reapply and final_dedup_parquet.exists():
                logger.info(f"Dedupe map is newer than deduped parquet; re-applying map...")
            
            # Load and log a sample of the mapping to verify it's being read
            from ingrnorm.dedupe_map import load_jsonl_map
            mapping_dict = load_jsonl_map(dedupe_map_path)
            logger.info(f"Loaded {len(mapping_dict):,} mappings from {dedupe_map_path}")
            if mapping_dict:
                sample_items = list(mapping_dict.items())[:5]
                logger.info(f"Sample mappings: {sample_items}")
            
            # Check baseline parquet before applying map
            try:
                pf_baseline = pq.ParquetFile(str(baseline_parquet))
                if pf_baseline.num_row_groups > 0:
                    df_baseline_sample = pf_baseline.read_row_group(0).to_pandas()
                    logger.info(f"Baseline parquet: shape={df_baseline_sample.shape}, columns={list(df_baseline_sample.columns)}")
                    if list_col_for_vocab in df_baseline_sample.columns:
                        baseline_sample = df_baseline_sample[list_col_for_vocab].head(5)
                        logger.info(f"Baseline sample (before mapping):")
                        for idx, lst in enumerate(baseline_sample):
                            if isinstance(lst, (list, tuple, np.ndarray)) and len(lst) > 0:
                                logger.info(f"  Row {idx}: {lst[:3]}")
            except Exception as e:
                logger.warning(f"Could not inspect baseline parquet: {e}")
            
            apply_map_to_parquet_streaming(
                in_path=str(baseline_parquet),
                out_path=str(final_dedup_parquet),
                mapping=str(dedupe_map_path),
                list_col=list_col_for_vocab,
            )
            logger.info(f"Saved deduped cuisine Parquet → {final_dedup_parquet}")
            
            # Verify the mapping was applied by checking a sample
            try:
                pf_check = pq.ParquetFile(str(final_dedup_parquet))
                logger.info(f"Verification: Parquet has {pf_check.num_row_groups} row group(s)")
                if pf_check.num_row_groups > 0:
                    df_sample = pf_check.read_row_group(0).to_pandas()
                    logger.info(f"Verification: DataFrame shape: {df_sample.shape}")
                    logger.info(f"Verification: Columns: {list(df_sample.columns)}")
                    logger.info(f"Verification: Looking for column '{list_col_for_vocab}'")
                    
                    if list_col_for_vocab in df_sample.columns:
                        sample_lists = df_sample[list_col_for_vocab].head(10)
                        logger.info(f"Sample of deduped data (first 10 rows):")
                        non_empty_count = 0
                        for idx, lst in enumerate(sample_lists):
                            if isinstance(lst, (list, tuple, np.ndarray)):
                                if len(lst) > 0:
                                    logger.info(f"  Row {idx}: {lst[:5]}")  # Show first 5 items
                                    non_empty_count += 1
                                else:
                                    logger.info(f"  Row {idx}: [] (empty)")
                            else:
                                logger.info(f"  Row {idx}: {type(lst).__name__} = {lst}")
                        logger.info(f"Verification: {non_empty_count}/10 rows have non-empty lists")
                    else:
                        logger.warning(f"Verification: Column '{list_col_for_vocab}' not found in deduped parquet!")
            except Exception as e:
                logger.warning(f"Could not verify deduped data: {e}", exc_info=True)
        else:
            logger.info("Step 4: Skipped (deduped file exists and is up-to-date with map)")
    elif do_apply_map and not dedupe_map_path.exists():
        logger.warning(f"Step 4: Dedupe map not found at {dedupe_map_path} - skipping map application")
    else:
        logger.info("Step 4: Skipped (map application disabled)")
    
    # Step 6: Encode to IDs
    # Always re-encode if the deduped parquet is newer than the encoded parquet
    # (This ensures manual edits to the JSON map are reflected in the encoded output)
    do_encode_ids = bool(stages_cfg.get("encode_ids", True))
    
    # CRITICAL: If dedupe map exists and we're encoding, ensure it's been applied first
    if do_encode_ids and do_apply_map and dedupe_map_path.exists():
        if not final_dedup_parquet.exists() or _needs_reapply_map(dedupe_map_path, final_dedup_parquet):
            logger.info("=" * 60)
            logger.info("Step 4 (re-check): Ensuring dedupe map is applied before encoding")
            logger.info("=" * 60)
            apply_map_to_parquet_streaming(
                in_path=str(baseline_parquet),
                out_path=str(final_dedup_parquet),
                mapping=str(dedupe_map_path),
                list_col=list_col_for_vocab,
            )
            logger.info(f"Re-applied dedupe map → {final_dedup_parquet}")
    
    needs_reencode = False
    if unified_parquet.exists() and final_dedup_parquet.exists():
        try:
            unified_mtime = unified_parquet.stat().st_mtime
            deduped_mtime = final_dedup_parquet.stat().st_mtime
            if deduped_mtime > unified_mtime:
                needs_reencode = True
                logger.info(f"Deduped parquet is newer than encoded parquet; re-encoding needed...")
        except (OSError, AttributeError):
            pass
    # Also check if the dedupe map is newer than the encoded parquet (map was manually edited)
    if not needs_reencode and unified_parquet.exists() and dedupe_map_path.exists():
        try:
            unified_mtime = unified_parquet.stat().st_mtime
            map_mtime = dedupe_map_path.stat().st_mtime
            if map_mtime > unified_mtime:
                needs_reencode = True
                logger.info(f"Dedupe map is newer than encoded parquet; will re-apply map and re-encode...")
        except (OSError, AttributeError):
            pass
    
    if do_encode_ids and (args.force or not unified_parquet.exists() or needs_reencode):
        logger.info("=" * 60)
        logger.info("Step 5: Encoding cuisines to IDs")
        logger.info("=" * 60)
        # Always prefer deduped parquet if it exists (it has the applied dedupe map)
        # CRITICAL: If dedupe map exists, we MUST use the deduped parquet, not the baseline
        if dedupe_map_path.exists() and do_apply_map:
            if not final_dedup_parquet.exists():
                logger.warning(f"Dedupe map exists but deduped parquet not found! Applying map now...")
                apply_map_to_parquet_streaming(
                    in_path=str(baseline_parquet),
                    out_path=str(final_dedup_parquet),
                    mapping=str(dedupe_map_path),
                    list_col=list_col_for_vocab,
                )
            source_parquet = final_dedup_parquet
            logger.info(f"Using DEDUPED parquet (with applied map) → {source_parquet}")
        else:
            source_parquet = final_dedup_parquet if final_dedup_parquet.exists() else baseline_parquet
            logger.info(f"Using source parquet → {source_parquet}")
        
        enc_cfg = cfg.get("encoder", {})
        enc_min_freq = int(enc_cfg.get("min_freq", 1))
        dataset_id = int(enc_cfg.get("dataset_id", 1))
        ingredients_col = enc_cfg.get("ingredients_col", list_col_for_vocab)
        
        enc = IngredientEncoder(min_freq=enc_min_freq)
        logger.info(f"Fitting encoder from {source_parquet} (col={ingredients_col}, min_freq={enc_min_freq})...")
        enc.fit_from_parquet_streaming(source_parquet, col=ingredients_col, min_freq=enc_min_freq).freeze()
        
        cuisine_id_to_token.parent.mkdir(parents=True, exist_ok=True)
        enc.save_maps(id_to_token_path=cuisine_id_to_token, token_to_id_path=cuisine_token_to_id)
        logger.info(f"Saved encoder maps → {cuisine_id_to_token}, {cuisine_token_to_id}")
        
        logger.info(f"Writing encoded cuisines → {unified_parquet}")
        enc.encode_parquet_streaming(
            parquet_path=source_parquet,
            out_parquet_path=unified_parquet,
            dataset_id=dataset_id,
            col=ingredients_col,
            compression="zstd",
        )
        logger.info(f"Saved encoded cuisine Parquet → {unified_parquet}")
    else:
        logger.info("Step 5: Skipped (encoded file exists and is up-to-date, or disabled)")
    
    logger.info("=" * 60)
    logger.info("Cuisine normalization and encoding complete")
    
    # Step 7: Apply dedupe and encoding to the original combined dataset
    logger.info("=" * 60)
    logger.info("Step 6: Applying dedupe and encoding to combined dataset")
    logger.info("=" * 60)
    
    combined_dataset_path = Path(data_cfg.get("input_path", "./data/encoded_combined_datasets.parquet"))
    if combined_dataset_path.exists():
        logger.info(f"Loading combined dataset from {combined_dataset_path}")
        
        # Load the combined dataset
        pf_combined = pq.ParquetFile(str(combined_dataset_path))
        logger.info(f"Combined dataset has {pf_combined.num_row_groups} row group(s)")
        
        # Load dedupe map and encoder
        from ingrnorm.dedupe_map import load_jsonl_map
        dedupe_map = load_jsonl_map(dedupe_map_path)
        logger.info(f"Loaded {len(dedupe_map):,} dedupe mappings")
        
        # Load encoder maps
        if cuisine_token_to_id.exists():
            with open(cuisine_token_to_id, "r", encoding="utf-8") as f:
                token_to_id = json.load(f)
            logger.info(f"Loaded encoder with {len(token_to_id):,} tokens")
        else:
            logger.warning(f"Encoder map not found at {cuisine_token_to_id}")
            token_to_id = {}
        
        # Process each row group and apply dedupe + encoding
        output_path = combined_dataset_path.parent / f"{combined_dataset_path.stem}_with_cuisine_encoded.parquet"
        writer = None
        
        for rg_idx in range(pf_combined.num_row_groups):
            logger.info(f"Processing row group {rg_idx + 1}/{pf_combined.num_row_groups}...")
            df_rg = pf_combined.read_row_group(rg_idx).to_pandas()
            
            # Apply dedupe map to cuisine column
            if cuisine_col in df_rg.columns:
                # Convert cuisine to list format (handle both string and list inputs)
                def normalize_cuisine_to_list(x):
                    """Convert cuisine value to a list of strings, handling various input formats."""
                    if pd.isna(x):
                        return []
                    # If already a list/array, extract strings directly
                    if isinstance(x, (list, tuple, np.ndarray)):
                        return [str(t).strip() for t in x if str(t).strip()]
                    # If it's a string, parse it (could be "[American]" or "American" or "[American, Italian]")
                    s = str(x).strip()
                    if not s or s.lower() in ["nan", "none", "[]", ""]:
                        return []
                    # Use split_cuisine_entries to handle all string formats
                    return split_cuisine_entries(s)
                
                cuisine_lists = df_rg[cuisine_col].apply(normalize_cuisine_to_list)
                
                # Normalize and apply dedupe map (lowercase for lookup)
                cuisine_deduped = cuisine_lists.apply(
                    lambda lst: [
                        dedupe_map.get(str(tok).lower().strip(), str(tok).lower().strip())
                        for tok in lst
                        if str(tok).strip()  # Skip empty strings
                    ]
                )
                df_rg[f"{cuisine_col}_deduped"] = cuisine_deduped
                
                # Encode to IDs
                cuisine_encoded = cuisine_deduped.apply(
                    lambda lst: [
                        token_to_id.get(tok.lower().strip(), 0)
                        for tok in lst
                    ]
                )
                df_rg[f"{cuisine_col}_encoded"] = cuisine_encoded
                
                logger.info(f"  Applied dedupe and encoding to {len(df_rg):,} rows")
                logger.info(f"  Sample: {cuisine_col} -> {cuisine_col}_deduped -> {cuisine_col}_encoded")
            else:
                logger.warning(f"  Column '{cuisine_col}' not found in row group {rg_idx + 1}")
                logger.info(f"  Available columns: {list(df_rg.columns)}")
            
            # Write to output parquet
            table = pa.Table.from_pandas(df_rg, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), table.schema, compression="zstd")
            writer.write_table(table)
        
        if writer:
            writer.close()
        
        logger.info(f"Saved updated dataset → {output_path}")
        
        # Debug: Show head and columns
        logger.info("=" * 60)
        logger.info("Debug: Inspecting updated dataset")
        logger.info("=" * 60)
        df_debug = pq.read_table(output_path).to_pandas()
        logger.info(f"Dataset shape: {df_debug.shape}")
        logger.info(f"Columns: {list(df_debug.columns)}")
        logger.info("\nFirst 10 rows (showing key columns):")
        # Show relevant columns
        cols_to_show = [cuisine_col]
        if f"{cuisine_col}_deduped" in df_debug.columns:
            cols_to_show.append(f"{cuisine_col}_deduped")
        if f"{cuisine_col}_encoded" in df_debug.columns:
            cols_to_show.append(f"{cuisine_col}_encoded")
        # Also show a few other columns if they exist
        for col in ["Dataset_ID", "index", "ingredients"]:
            if col in df_debug.columns and col not in cols_to_show:
                cols_to_show.append(col)
        
        logger.info(f"\n{df_debug[cols_to_show].head(10).to_string()}")
        
        # Show detailed sample of cuisine transformations
        if f"{cuisine_col}_deduped" in df_debug.columns:
            logger.info(f"\nDetailed sample of cuisine transformations (first 5 rows):")
            for idx in range(min(5, len(df_debug))):
                orig = df_debug[cuisine_col].iloc[idx]
                deduped = df_debug[f"{cuisine_col}_deduped"].iloc[idx]
                encoded = df_debug[f"{cuisine_col}_encoded"].iloc[idx]
                logger.info(f"  Row {idx}:")
                logger.info(f"    Original: {orig}")
                logger.info(f"    Deduped:  {deduped}")
                logger.info(f"    Encoded:  {encoded}")
    else:
        logger.warning(f"Combined dataset not found at {combined_dataset_path} - skipping Step 6")

    # Final cleanup: Remove intermediate parquet files, keep JSON files and main dataset
    logger.info("=" * 60)
    logger.info("Final cleanup: Removing intermediate parquet files")
    logger.info("=" * 60)
    preserve_files = [
        str(combined_dataset_path),
        str(output_path) if 'output_path' in locals() and output_path.exists() else None,
    ]
    preserve_files = [f for f in preserve_files if f]  # Remove None values
    
    # Also clean up temporary prepared file if it exists
    prepared_parquet = baseline_parquet.parent / "_cuisine_prepared.parquet"
    if prepared_parquet.exists():
        try:
            prepared_parquet.unlink()
            logger.info(f"[cleanup] Deleted temporary file: {prepared_parquet}")
        except Exception as e:
            logger.warning(f"[cleanup] Failed to delete {prepared_parquet}: {e}")
    
    _cleanup_paths(cfg, logger, preserve_files=preserve_files)
    
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"  - Dedupe map (JSONL): {dedupe_map_path}")
    logger.info(f"  - Token→ID map (JSON): {cuisine_token_to_id}")
    logger.info(f"  - ID→Token map (JSON): {cuisine_id_to_token}")
    if 'output_path' in locals() and output_path.exists():
        logger.info(f"  - Final dataset: {output_path}")


if __name__ == "__main__":
    main()

