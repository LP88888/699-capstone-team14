"""
Normalize and encode cuisines using the ingrnorm pipeline primitives.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import glob
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

from src.preprocess_pipeline.ingrnorm.io import materialize_parquet_source
from src.preprocess_pipeline.ingrnorm.parquet_utils import vocab_from_parquet_listcol
from src.preprocess_pipeline.ingrnorm.spacy_normalizer import apply_spacy_normalizer_to_parquet
from src.preprocess_pipeline.ingrnorm.sbert_dedupe import sbert_dedupe
from src.preprocess_pipeline.ingrnorm.w2v_dedupe import w2v_dedupe
from src.preprocess_pipeline.ingrnorm.dedupe_map import apply_map_to_parquet_streaming, write_jsonl_map
from src.preprocess_pipeline.ingrnorm.encoder import IngredientEncoder

from ..core import PipelineContext, StageResult
from ..utils import stage_logger


def _filter_and_normalize_cuisine_mapping(mapping: dict[str, str]) -> dict[str, str]:
    """
    Post-process cuisine deduplication mapping to:
    1. Filter out non-cuisine entries (e.g., "friendly", "kid friendly")
    2. Normalize entries (e.g., "gujarati recipes" -> "gujarati")
    
    Args:
        mapping: Raw mapping from sbert_dedupe
        
    Returns:
        Filtered and normalized mapping
    """
    # Non-cuisine terms to exclude
    NON_CUISINES = {"friendly", "kid friendly"}
    
    # Normalization rules: patterns to clean up
    def normalize_cuisine(cuisine: str) -> str:
        """Normalize cuisine name by removing common suffixes."""
        cuisine_lower = cuisine.lower().strip()
        
        # Remove "recipe(s)" suffix
        if cuisine_lower.endswith(" recipes"):
            cuisine_lower = cuisine_lower[:-8]  # Remove " recipes"
        elif cuisine_lower.endswith(" recipe"):
            cuisine_lower = cuisine_lower[:-7]  # Remove " recipe"
        
        return cuisine_lower.strip()
    
    filtered_mapping = {}
    for src, tgt in mapping.items():
        src_lower = src.lower().strip()
        tgt_lower = tgt.lower().strip()
        
        # Skip non-cuisines
        if src_lower in NON_CUISINES or tgt_lower in NON_CUISINES:
            continue
        
        # Normalize target (canonical form)
        tgt_norm = normalize_cuisine(tgt)
        
        # Normalize source, but keep original if normalization doesn't change it
        src_norm = normalize_cuisine(src)
        
        # Only add mapping if source and target are different (after normalization)
        if src_norm != tgt_norm:
            filtered_mapping[src_norm] = tgt_norm
    
    return filtered_mapping


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


def run(
    context: PipelineContext,
    *,
    force: bool = False,
    input_path: Optional[Path] = None,
) -> StageResult:
    cfg = context.stage("cuisine_normalization")
    logger = stage_logger(context, "cuisine_norm", force=force)

    try:
        data_cfg = cfg.get("data", {})
        out_cfg = cfg.get("output", {})
        stages_cfg = cfg.get("stages", {})
        sbert_cfg = cfg.get("sbert", {})
        enc_cfg = cfg.get("encoder", {})

        combined_dataset_path = Path(data_cfg.get("input_path", "./data/encoded_combined_datasets.parquet"))
        _cleanup_paths(cfg, logger, preserve_files=[str(combined_dataset_path)])

        input_path = Path(input_path or data_cfg.get("input_path", "./data/encoded_combined_datasets.parquet"))
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        cuisine_col = data_cfg.get("cuisine_col", "cuisine")
        baseline_parquet = Path(out_cfg.get("baseline_parquet", "./data/cuisine_normalized/cuisine_baseline.parquet"))
        final_dedup_parquet = Path(out_cfg.get("dedup_parquet", "./data/cuisine_normalized/cuisine_deduped.parquet"))
        dedupe_map_path = Path(out_cfg.get("cosine_map_path", "./data/cuisine_normalized/cuisine_dedupe_map.jsonl"))
        list_col_for_vocab = out_cfg.get("list_col_for_vocab", "cuisine_clean")
        unified_parquet = Path(out_cfg.get("unified_parquet", "./data/cuisine_encoded/cuisine_unified.parquet"))
        cuisine_id_to_token = Path(out_cfg.get("cuisine_id_to_token", "./data/cuisine_encoded/cuisine_id_to_token.json"))
        cuisine_token_to_id = Path(out_cfg.get("cuisine_token_to_id", "./data/cuisine_encoded/cuisine_token_to_id.json"))

        output_dir = baseline_parquet.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        prepared_parquet = baseline_parquet.parent / "_cuisine_prepared.parquet"

        if force or not prepared_parquet.exists():
            logger.info("=" * 60)
            logger.info("Step 1: Preparing cuisine list column (splitting multi-cuisine entries)")
            logger.info("=" * 60)
            prepare_cuisine_list_column(input_path, cuisine_col, prepared_parquet, logger)
        else:
            logger.info("Step 1: Skipped (prepared file exists)")

        if force or not baseline_parquet.exists():
            logger.info("=" * 60)
            logger.info("Step 2: spaCy normalization")
            logger.info("=" * 60)
            apply_spacy_normalizer_to_parquet(
                in_parquet=str(prepared_parquet),
                out_parquet=str(baseline_parquet),
                list_col=cuisine_col,
                out_col=list_col_for_vocab,
                spacy_model=sbert_cfg.get("spacy_model", "en_core_web_sm"),
                batch_size=int(sbert_cfg.get("spacy_batch_size", 512)),
                n_process=int(sbert_cfg.get("spacy_n_process", 1)),
            )
            logger.info("Saved baseline cuisine parquet → %s", baseline_parquet)
        else:
            logger.info("Step 2: Skipped (baseline file exists)")

        min_freq_for_vocab = int(sbert_cfg.get("min_freq_for_vocab", 1))
        vocab = vocab_from_parquet_listcol(
            str(baseline_parquet),
            col=list_col_for_vocab,
            min_freq=min_freq_for_vocab,
        )
        vocab_filtered = {k: v for k, v in vocab.items() if k.lower().strip() not in {"friendly", "kid friendly"}}
        if len(vocab) != len(vocab_filtered):
            logger.info("Filtered %s non-cuisine entries from vocab", len(vocab) - len(vocab_filtered))
        vocab = vocab_filtered

        if not vocab:
            raise ValueError("Vocabulary is empty. Check cuisine_clean column.")

        use_sbert = bool(stages_cfg.get("sbert_dedupe", True))
        if use_sbert and (force or not dedupe_map_path.exists()):
            logger.info("=" * 60)
            logger.info("Step 3: SBERT deduplication")
            logger.info("=" * 60)
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
            mapping = _filter_and_normalize_cuisine_mapping(mapping)
            write_jsonl_map(mapping, dedupe_map_path)
            logger.info("Saved dedupe map → %s", dedupe_map_path)
        else:
            logger.info("Step 3: Skipped (dedupe map exists or SBERT disabled)")

        do_apply_map = bool(stages_cfg.get("apply_cosine_map", True))
        if do_apply_map and dedupe_map_path.exists():
            needs_reapply = _needs_reapply_map(dedupe_map_path, final_dedup_parquet)
            if force or not final_dedup_parquet.exists() or needs_reapply:
                logger.info("=" * 60)
                logger.info("Step 4: Applying dedupe map to parquet")
                logger.info("=" * 60)
                apply_map_to_parquet_streaming(
                    in_path=str(baseline_parquet),
                    out_path=str(final_dedup_parquet),
                    mapping=str(dedupe_map_path),
                    list_col=list_col_for_vocab,
                )
                logger.info("Saved deduped cuisine parquet → %s", final_dedup_parquet)
            else:
                logger.info("Step 4: Skipped (deduped parquet up-to-date)")
        elif do_apply_map:
            logger.warning("Step 4: Dedupe map not found at %s - skipping map application", dedupe_map_path)
        else:
            logger.info("Step 4: Skipped (map application disabled)")

        do_encode_ids = bool(stages_cfg.get("encode_ids", True))
        if do_encode_ids and do_apply_map and dedupe_map_path.exists():
            if not final_dedup_parquet.exists() or _needs_reapply_map(dedupe_map_path, final_dedup_parquet):
                apply_map_to_parquet_streaming(
                    in_path=str(baseline_parquet),
                    out_path=str(final_dedup_parquet),
                    mapping=str(dedupe_map_path),
                    list_col=list_col_for_vocab,
                )

        needs_reencode = False
        if unified_parquet.exists() and final_dedup_parquet.exists():
            try:
                if final_dedup_parquet.stat().st_mtime > unified_parquet.stat().st_mtime:
                    needs_reencode = True
            except (OSError, AttributeError):
                pass
        if not needs_reencode and unified_parquet.exists() and dedupe_map_path.exists():
            try:
                if dedupe_map_path.stat().st_mtime > unified_parquet.stat().st_mtime:
                    needs_reencode = True
            except (OSError, AttributeError):
                pass

        if do_encode_ids and (force or not unified_parquet.exists() or needs_reencode):
            logger.info("=" * 60)
            logger.info("Step 5: Encoding cuisines to IDs")
            logger.info("=" * 60)
            source_parquet = final_dedup_parquet if final_dedup_parquet.exists() else baseline_parquet
            enc_min_freq = int(enc_cfg.get("min_freq", 1))
            dataset_id = int(enc_cfg.get("dataset_id", 1))
            ingredients_col = enc_cfg.get("ingredients_col", list_col_for_vocab)

            enc = IngredientEncoder(min_freq=enc_min_freq)
            enc.fit_from_parquet_streaming(source_parquet, col=ingredients_col, min_freq=enc_min_freq).freeze()
            cuisine_id_to_token.parent.mkdir(parents=True, exist_ok=True)
            enc.save_maps(id_to_token_path=cuisine_id_to_token, token_to_id_path=cuisine_token_to_id)
            enc.encode_parquet_streaming(
                parquet_path=source_parquet,
                out_parquet_path=unified_parquet,
                dataset_id=dataset_id,
                col=ingredients_col,
                compression="zstd",
            )
            logger.info("Saved encoded cuisine parquet → %s", unified_parquet)
        else:
            logger.info("Step 5: Skipped (encoded file exists and is up-to-date, or disabled)")

        logger.info("=" * 60)
        logger.info("Cuisine normalization and encoding complete")

        combined_output_path = None
        if combined_dataset_path.exists():
            logger.info("=" * 60)
            logger.info("Step 6: Applying dedupe and encoding to combined dataset")
            logger.info("=" * 60)

            from ingrnorm.dedupe_map import load_jsonl_map

            dedupe_map = load_jsonl_map(dedupe_map_path) if dedupe_map_path.exists() else {}
            if cuisine_token_to_id.exists():
                with open(cuisine_token_to_id, "r", encoding="utf-8") as fh:
                    token_to_id = json.load(fh)
            else:
                token_to_id = {}

            pf_combined = pq.ParquetFile(str(combined_dataset_path))
            combined_output_path = combined_dataset_path.parent / f"{combined_dataset_path.stem}_with_cuisine_encoded.parquet"
            writer = None

            for rg_idx in range(pf_combined.num_row_groups):
                logger.info("Processing row group %s/%s…", rg_idx + 1, pf_combined.num_row_groups)
                df_rg = pf_combined.read_row_group(rg_idx).to_pandas()

                if cuisine_col in df_rg.columns:
                    def normalize_cuisine_to_list(x):
                        if pd.isna(x):
                            return []
                        if isinstance(x, (list, tuple, np.ndarray)):
                            return [str(t).strip() for t in x if str(t).strip()]
                        s = str(x).strip()
                        if not s or s.lower() in ["nan", "none", "[]", ""]:
                            return []
                        return split_cuisine_entries(s)

                    cuisine_lists = df_rg[cuisine_col].apply(normalize_cuisine_to_list)
                    cuisine_deduped = cuisine_lists.apply(
                        lambda lst: [
                            dedupe_map.get(str(tok).lower().strip(), str(tok).lower().strip())
                            for tok in lst
                            if str(tok).strip()
                        ]
                    )
                    df_rg[f"{cuisine_col}_deduped"] = cuisine_deduped
                    cuisine_encoded = cuisine_deduped.apply(
                        lambda lst: [token_to_id.get(tok.lower().strip(), 0) for tok in lst]
                    )
                    df_rg[f"{cuisine_col}_encoded"] = cuisine_encoded
                else:
                    logger.warning("Column '%s' not found in row group %s", cuisine_col, rg_idx + 1)

                table = pa.Table.from_pandas(df_rg, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(combined_output_path), table.schema, compression="zstd")
                writer.write_table(table)

            if writer:
                writer.close()
            logger.info("Saved updated dataset → %s", combined_output_path)
        else:
            logger.warning("Combined dataset not found at %s - skipping Step 6", combined_dataset_path)

        logger.info("=" * 60)
        logger.info("Final cleanup: Removing intermediate parquet files")
        logger.info("=" * 60)
        preserve_files = [str(combined_dataset_path)]
        if "combined_output_path" in locals() and combined_output_path and combined_output_path.exists():
            preserve_files.append(str(combined_output_path))

        if prepared_parquet.exists():
            try:
                prepared_parquet.unlink()
                logger.info("[cleanup] Deleted temporary file: %s", prepared_parquet)
            except Exception as exc:
                logger.warning("[cleanup] Failed to delete %s: %s", prepared_parquet, exc)

        _cleanup_paths(cfg, logger, preserve_files=preserve_files)

        outputs = {
            "baseline_parquet": str(baseline_parquet),
            "dedup_parquet": str(final_dedup_parquet),
            "cosine_map": str(dedupe_map_path),
            "unified_parquet": str(unified_parquet),
        }
        if combined_output_path:
            outputs["combined_dataset"] = str(combined_output_path)

        return StageResult(name="cuisine_normalization", status="success", outputs=outputs)

    except Exception as exc:  # pragma: no cover
        logger.exception("Cuisine normalization failed: %s", exc)
        return StageResult(name="cuisine_normalization", status="failed", details=str(exc))