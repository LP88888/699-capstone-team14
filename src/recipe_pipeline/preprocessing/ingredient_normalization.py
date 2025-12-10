
from __future__ import annotations

import glob
import logging
import os
from pathlib import Path

import pyarrow.parquet as pq

from ingrnorm.io import materialize_parquet_source
from ingrnorm.parquet_utils import vocab_from_parquet_listcol
from ingrnorm.spacy_normalizer import apply_spacy_normalizer_to_parquet
from ingrnorm.sbert_dedupe import sbert_dedupe
from ingrnorm.w2v_dedupe import w2v_dedupe
from ingrnorm.dedupe_map import apply_map_to_parquet_streaming, write_jsonl_map
from ingrnorm.encoder import IngredientEncoder
from ingrnorm.spacy_normalizer import SpacyIngredientNormalizer

from ..core import PipelineContext, StageResult
from ..utils import stage_logger

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
            if Path(p).exists():
                preserve_patterns.append(Path(p).resolve())
            else:
                # If path doesn't exist, store as string for pattern matching
                preserve_patterns.append(p)
        except (OSError, ValueError):
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

def _as_path(d: dict, key: str) -> Path:
    v = d.get(key)
    if v is None:
        raise ValueError(f"Missing required path key: '{key}'")
    return Path(v)

def _exists(p: Path) -> bool:
    """Check if path exists, handling permission errors gracefully."""
    try:
        return p.exists()
    except (OSError, PermissionError):
        return False

def _validate_config(cfg: dict, logger: logging.Logger) -> None:
    """Validate required config sections and values."""
    required_sections = ["data", "output"]
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required config section: '{section}'")
    
    data_cfg = cfg.get("data", {})
    if "input_path" not in data_cfg:
        raise ValueError("Missing required config: data.input_path")
    
    out_cfg = cfg.get("output", {})
    required_outputs = ["baseline_parquet", "dedup_parquet", "cosine_map_path", "unified_parquet"]
    for key in required_outputs:
        if key not in out_cfg:
            raise ValueError(f"Missing required config: output.{key}")
    
    input_path = Path(data_cfg["input_path"])
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.debug("Config validation passed")


def _build_spacy_map_from_parquet(
    parquet_path: Path,
    list_col: str,
    spacy_model: str,
    batch_size: int,
    n_process: int,
    logger: logging.Logger,
) -> dict:
    """
    Normalize all unique phrases once via spaCy and return a mapping raw -> normalized.
    """
    vocab = vocab_from_parquet_listcol(str(parquet_path), col=list_col, min_freq=1)
    phrases = list(vocab.keys())
    logger.info("[stage1] unique phrases collected: %s", len(phrases))
    if not phrases:
        return {}

    normalizer = SpacyIngredientNormalizer(spacy_model, batch_size=batch_size, n_process=n_process)
    mapping: dict[str, str] = {}

    batch: list[list[str]] = []
    originals: list[str] = []
    for p in phrases:
        batch.append([p])
        originals.append(p)
        if len(batch) >= batch_size:
            norms = normalizer.normalize_batch(batch)
            for raw, norm_list in zip(originals, norms):
                if norm_list:
                    mapping[raw] = norm_list[0]
            batch.clear()
            originals.clear()
    if batch:
        norms = normalizer.normalize_batch(batch)
        for raw, norm_list in zip(originals, norms):
            if norm_list:
                mapping[raw] = norm_list[0]

    logger.info("[stage1] spaCy map built: %s entries", len(mapping))
    return mapping

def _run_stage1_spacy_normalization(
    input_path: Path,
    ner_col: str,
    chunksize: int,
    baseline_parquet: Path,
    list_col_for_vocab: str,
    sbert_cfg: dict,
    do_write_parquet: bool,
    force: bool,
    logger: logging.Logger,
) -> None:
    """Stage 1: spaCy normalization."""
    tmp_raw_parquet = baseline_parquet.with_name("_raw_source_for_spacy.parquet")
    if do_write_parquet and (force or not _exists(baseline_parquet)):
        logger.info("Stage 1: spaCy normalization → baseline_parquet (NER_clean)")

        logger.info("[stage1] materialize_parquet_source starting…")
        src_parquet = materialize_parquet_source(input_path, ner_col, chunksize, tmp_raw_parquet)
        logger.info(f"[stage1] materialize_parquet_source done → {src_parquet}")

        baseline_parquet.parent.mkdir(parents=True, exist_ok=True)
        # Build global spaCy normalization map from all unique phrases, then apply to parquet
        spacy_batch_size = int(sbert_cfg.get("spacy_batch_size", 512))
        spacy_n_process = int(sbert_cfg.get("spacy_n_process", 0))  # 0=auto, 1=single-threaded
        spacy_model = sbert_cfg.get("spacy_model", "en_core_web_sm")

        mapping = _build_spacy_map_from_parquet(
            parquet_path=src_parquet,
            list_col=ner_col,
            spacy_model=spacy_model,
            batch_size=spacy_batch_size,
            n_process=spacy_n_process,
            logger=logger,
        )

        logger.info("[stage1] applying global map to parquet…")
        apply_map_to_parquet_streaming(
            in_path=str(src_parquet),
            out_path=str(baseline_parquet),
            mapping=mapping,
            list_col=ner_col,
            out_col=list_col_for_vocab,
            collapse_duplicate_words=True,
            dedupe_tokens=True,
        )
        logger.info("[stage1] global map application done")

        if src_parquet == tmp_raw_parquet and tmp_raw_parquet.exists():
            try:
                tmp_raw_parquet.unlink()
                logger.info(f"[stage1] Removed temp parquet {tmp_raw_parquet}")
            except (OSError, PermissionError) as e:
                logger.warning(f"[stage1] Could not remove temp file {tmp_raw_parquet}: {e}")
        logger.info(f"Saved baseline Parquet → {baseline_parquet}")
    else:
        logger.info("Stage 1: spaCy normalization – skipped (exists or disabled)")

def _run_stage2_dedupe(
    baseline_parquet: Path,
    dedupe_map_path: Path,
    list_col_for_vocab: str,
    vocab: dict,
    use_sbert: bool,
    use_w2v: bool,
    sbert_cfg: dict,
    w2v_cfg: dict,
    force: bool,
    logger: logging.Logger,
) -> None:
    """Stage 2: Build dedupe map."""
    if (use_sbert or use_w2v) and (force or not _exists(dedupe_map_path)):
        if len(vocab) == 0:
            logger.warning("[dedupe] Vocab empty; skipping dedupe + map application.")
            return
        
        if use_sbert:
            logger.info("Stage 2: SBERT de-dupe – building phrase map")
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
        elif use_w2v:
            logger.info("Stage 2: W2V de-dupe – building phrase map")
            mapping = w2v_dedupe(
                vocab_counter=vocab,
                corpus_parquet=str(baseline_parquet),
                list_col=list_col_for_vocab,
                model_cache_path=Path(dedupe_map_path).with_suffix(".w2v"),
                vector_size=int(w2v_cfg.get("vector_size", 100)),
                window=int(w2v_cfg.get("window", 5)),
                min_count=int(w2v_cfg.get("min_count", 1)),
                workers=int(w2v_cfg.get("workers", 4)),
                sg=int(w2v_cfg.get("sg", 1)),
                epochs=int(w2v_cfg.get("epochs", 8)),
                threshold=float(w2v_cfg.get("threshold", 0.85)),
                topk=int(w2v_cfg.get("topk", 25)),
                out_path=str(dedupe_map_path),
            )
            write_jsonl_map(mapping, dedupe_map_path)
        logger.info(f"Saved dedupe map → {dedupe_map_path}")
    else:
        logger.info("Stage 2: Dedupe – skipped (exists or disabled)")

def _run_stage3_apply_map(
    baseline_parquet: Path,
    final_dedup_parquet: Path,
    dedupe_map_path: Path,
    list_col_for_vocab: str,
    do_apply_map: bool,
    force: bool,
    logger: logging.Logger,
) -> None:
    """Stage 3: Apply dedupe map."""
    if do_apply_map and (force or not _exists(final_dedup_parquet)):
        if not _exists(dedupe_map_path):
            logger.warning(f"[apply] Map not found at {dedupe_map_path}; skipping Stage 3.")
        else:
            logger.info("Stage 3: Apply dedupe map → final deduped Parquet")
            apply_map_to_parquet_streaming(
                in_path=str(baseline_parquet),
                out_path=str(final_dedup_parquet),
                mapping=str(dedupe_map_path),
                list_col=list_col_for_vocab,
            )
            logger.info(f"Saved final deduped Parquet → {final_dedup_parquet}")
    else:
        logger.info("Stage 3: Apply map – skipped (exists or disabled)")

def _run_stage4_encode(
    baseline_parquet: Path,
    final_dedup_parquet: Path,
    unified_parquet: Path,
    ing_id_to_token: Path,
    ing_token_to_id: Path,
    list_col_for_vocab: str,
    enc_cfg: dict,
    do_encode_ids: bool,
    force: bool,
    logger: logging.Logger,
) -> None:
    """Stage 4: Encode to IDs."""
    if do_encode_ids and (force or not _exists(unified_parquet)):
        logger.info("Stage 4: Encode to IDs – starting")
        source_parquet = final_dedup_parquet if _exists(final_dedup_parquet) else baseline_parquet
        logger.info(f"[encode] Source selected → {source_parquet}")
        enc_min_freq    = int(enc_cfg.get("min_freq", 1))
        dataset_id      = int(enc_cfg.get("dataset_id", 1))
        ingredients_col = enc_cfg.get("ingredients_col", list_col_for_vocab)

        enc = IngredientEncoder(min_freq=enc_min_freq)
        logger.info(f"[encode] Fitting encoder from {source_parquet} (col={ingredients_col}, min_freq={enc_min_freq}) …")
        enc.fit_from_parquet_streaming(source_parquet, col=ingredients_col, min_freq=enc_min_freq).freeze()

        ing_id_to_token.parent.mkdir(parents=True, exist_ok=True)
        enc.save_maps(id_to_token_path=ing_id_to_token, token_to_id_path=ing_token_to_id)
        logger.info(f"Saved encoder maps → {ing_id_to_token}, {ing_token_to_id}")

        logger.info(f"[encode] Writing → {unified_parquet}")
        unified_parquet.parent.mkdir(parents=True, exist_ok=True)
        enc.encode_parquet_streaming(
            parquet_path=source_parquet,
            out_parquet_path=unified_parquet,
            dataset_id=dataset_id,
            col=ingredients_col,
            compression="zstd",
        )
        logger.info(f"Saved unified encoded Parquet → {unified_parquet}")
    else:
        logger.info("Stage 4: Encode to IDs – skipped (exists or disabled)")

def run(context: PipelineContext, *, force: bool = False) -> StageResult:
    cfg = context.stage("ingredient_normalization")
    logger = stage_logger(context, "ingrnorm", force=force)

    try:
        _validate_config(cfg, logger)

        data_cfg = cfg.get("data", {})
        out_cfg = cfg.get("output", {})
        stages_cfg = cfg.get("stages", {})
        w2v_cfg = cfg.get("w2v", {})
        sbert_cfg = cfg.get("sbert", {})
        enc_cfg = cfg.get("encoder", {})

        input_path = _as_path(data_cfg, "input_path")
        ner_col = data_cfg.get("ner_col", "NER")
        chunksize = int(data_cfg.get("chunksize", 200_000))

        baseline_parquet = _as_path(out_cfg, "baseline_parquet")
        final_dedup_parquet = _as_path(out_cfg, "dedup_parquet")
        dedupe_map_path = _as_path(out_cfg, "cosine_map_path")
        list_col_for_vocab = out_cfg.get("list_col_for_vocab", "NER_clean")

        unified_parquet = _as_path(out_cfg, "unified_parquet")
        ing_id_to_token = _as_path(out_cfg, "ingredient_id_to_token")
        ing_token_to_id = _as_path(out_cfg, "ingredient_token_to_id")

        do_write_parquet = bool(stages_cfg.get("write_parquet", True))
        use_sbert = bool(stages_cfg.get("sbert_dedupe", True))
        use_w2v = bool(stages_cfg.get("w2v_dedupe", False))
        do_apply_map = bool(stages_cfg.get("apply_cosine_map", True))
        do_encode_ids = bool(stages_cfg.get("encode_ids", True))

        if use_sbert and use_w2v:
            logger.warning("[stages] Both sbert_dedupe and w2v_dedupe are True; proceeding with SBERT only.")
            use_w2v = False

        preserve_files = [str(input_path)] if input_path.exists() else []
        # Run cleanup BEFORE processing to clear old artifacts, not after
        _cleanup_paths(cfg, logger, preserve_files=preserve_files)

        _run_stage1_spacy_normalization(
            input_path,
            ner_col,
            chunksize,
            baseline_parquet,
            list_col_for_vocab,
            sbert_cfg,
            do_write_parquet,
            force,
            logger,
        )

        try:
            if _exists(baseline_parquet):
                pf = pq.ParquetFile(str(baseline_parquet))
                if pf.num_row_groups > 0:
                    tbl = pf.read_row_group(0, columns=[list_col_for_vocab])
                    s = tbl.to_pandas()[list_col_for_vocab]
                    non_empty = s.explode().dropna().astype(str).str.strip()
                    logger.info(
                        "[probe] RG0 tokens: non-empty=%s, unique=%s",
                        (non_empty != "").sum(),
                        non_empty.nunique(),
                    )
        except Exception as exc:  # pragma: no cover - diagnostics only
            logger.info("[probe] Skipped due to error: %s", exc)

        if not _exists(baseline_parquet):
            raise FileNotFoundError(
                f"baseline_parquet not found at {baseline_parquet}. "
                f"Run Stage 1 (spaCy normalization) first or check input_path: {input_path}"
            )

        min_freq_for_vocab = int(sbert_cfg.get("min_freq_for_vocab", w2v_cfg.get("min_freq_for_vocab", 1)))
        vocab = vocab_from_parquet_listcol(
            str(baseline_parquet),
            col=list_col_for_vocab,
            min_freq=min_freq_for_vocab,
        )
        logger.info("[dedupe] Vocab size: %s (min_freq=%s)", len(vocab), min_freq_for_vocab)
        if len(vocab) == 0:
            logger.warning("[dedupe] Vocabulary is empty. Check column '%s' in %s", list_col_for_vocab, baseline_parquet)

        _run_stage2_dedupe(
            baseline_parquet,
            dedupe_map_path,
            list_col_for_vocab,
            vocab,
            use_sbert,
            use_w2v,
            sbert_cfg,
            w2v_cfg,
            force,
            logger,
        )

        _run_stage3_apply_map(
            baseline_parquet,
            final_dedup_parquet,
            dedupe_map_path,
            list_col_for_vocab,
            do_apply_map,
            force,
            logger,
        )

        _run_stage4_encode(
            baseline_parquet,
            final_dedup_parquet,
            unified_parquet,
            ing_id_to_token,
            ing_token_to_id,
            list_col_for_vocab,
            enc_cfg,
            do_encode_ids,
            force,
            logger,
        )

        logger.info("=" * 60)
        logger.info("Final cleanup: Removing intermediate parquet files")
        logger.info("=" * 60)

        tmp_raw_parquet = baseline_parquet.with_name("_raw_source_for_spacy.parquet")
        if tmp_raw_parquet.exists():
            try:
                tmp_raw_parquet.unlink()
                logger.info("[cleanup] Deleted temporary file: %s", tmp_raw_parquet)
            except Exception as exc:
                logger.warning("[cleanup] Failed to delete %s: %s", tmp_raw_parquet, exc)

        # Do NOT run general cleanup at the end; it deletes the outputs we just created!
        # _cleanup_paths(cfg, logger, preserve_files=preserve_files) 
        
        logger.info("Workflow complete. Kept JSON artifacts: %s, %s, %s", dedupe_map_path, ing_token_to_id, ing_id_to_token)

        return StageResult(
            name="ingredient_normalization",
            status="success",
            outputs={
                "baseline_parquet": str(baseline_parquet),
                "dedup_parquet": str(final_dedup_parquet),
                "cosine_map": str(dedupe_map_path),
                "unified_parquet": str(unified_parquet),
            },
        )
    except Exception as exc:  # pragma: no cover - pipeline runner will log
        logger.exception("Ingredient normalization failed: %s", exc)
        return StageResult(name="ingredient_normalization", status="failed", details=str(exc))
