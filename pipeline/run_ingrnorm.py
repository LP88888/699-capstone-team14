from __future__ import annotations
import argparse
import ast
import json
import logging
import sys
from pathlib import Path
import pathlib as _pathlib
from typing import List

# Make `pipeline/` importable
sys.path.append(str(_pathlib.Path.cwd() / "pipeline"))

# External deps
try:
    import yaml
except Exception:
    print("[FATAL] Please `pip install pyyaml` to read config.yaml")
    raise

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ingrnorm.parquet_utils import vocab_from_parquet_listcol
from ingrnorm.dedupe import w2v_dedupe, apply_map_to_parquet
from ingrnorm.encoder import IngredientEncoder
from ingrnorm.spacy_normalizer import apply_spacy_normalizer_to_parquet
from common.logging_setup import setup_logging

logger = logging.getLogger("ingrnorm") 

import os
import glob

def _cleanup_paths(cfg: dict, logger: logging.Logger):
    cleanup_cfg = cfg.get("cleanup", {})
    if not cleanup_cfg.get("enabled", False):
        logger.info("Cleanup disabled – skipping file deletions.")
        return
    paths = cleanup_cfg.get("paths", [])
    if not paths:
        logger.info("Cleanup enabled but no paths specified.")
        return
    for pattern in paths:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
                logger.info(f"[cleanup] Deleted {f}")
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"[cleanup] Failed to delete {f}: {e}")


def _as_path(d: dict, key: str) -> Path:
    v = d.get(key)
    if v is None:
        raise ValueError(f"Missing required path key: '{key}'")
    return Path(v)

def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

def _parse_listish(v: object) -> List[str]:
    """
    Accept:
      - true list/tuple/ndarray → list[str]
      - JSON/Python-serialized list in a string → list[str]
      - plain string → [string]
      - None/NaN → []
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, (list, tuple, np.ndarray)):
        return [str(x) for x in v if str(x).strip()]
    s = str(v).strip()
    if not s:
        return []
    # Try JSON or Python list
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
    # Fallback: treat as one phrase
    return [s]

def _materialize_parquet_source(
    input_path: Path,
    ner_col: str,
    chunksize: int,
    tmp_out: Path
) -> Path:
    """
    Ensure we have a Parquet file with a list<string> column `ner_col`.
    If input is already Parquet, return it as-is.
    If CSV/Excel, stream-convert to Parquet with schema [ner_col: list<string>].
    """
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        return input_path

    if suffix == ".csv":
        logger.info(f"[ingest] Converting CSV → Parquet (list<{ner_col}>) at {tmp_out}")
        tmp_out.parent.mkdir(parents=True, exist_ok=True)
        schema = pa.schema([pa.field(ner_col, pa.list_(pa.string()))])
        writer = pq.ParquetWriter(str(tmp_out), schema, compression="zstd")

        for chunk in pd.read_csv(input_path, chunksize=chunksize, dtype=str):
            col = chunk[ner_col] if ner_col in chunk.columns else pd.Series([None] * len(chunk))
            lists = [_parse_listish(x) for x in col]
            arr = pa.array(lists, type=pa.list_(pa.string()))
            tbl = pa.Table.from_arrays([arr], names=[ner_col])
            writer.write_table(tbl)
            del chunk, col, lists, arr, tbl

        writer.close()
        return tmp_out

    if suffix in (".xlsx", ".xls"):
        logger.info(f"[ingest] Converting Excel → Parquet (list<{ner_col}>) at {tmp_out}")
        df = pd.read_excel(input_path, dtype=str)
        col = df[ner_col] if ner_col in df.columns else pd.Series([None] * len(df))
        lists = [_parse_listish(x) for x in col]
        arr = pa.array(lists, type=pa.list_(pa.string()))
        tbl = pa.Table.from_arrays([arr], names=[ner_col])
        tmp_out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(tbl, str(tmp_out), compression="zstd")
        return tmp_out

    raise ValueError(f"Unsupported file type for ingest: {input_path.name}")


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Run ingrnorm workflow (spaCy → dedupe → apply → encode)")
    ap.add_argument("--config", type=str, default="pipeline/config/config.yaml", help="Path to config YAML")
    ap.add_argument("--force", action="store_true", help="Rebuild artifacts even if they exist")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # init logging
    setup_logging(cfg)
    logger = logging.getLogger("ingrnorm")

    # Cleanup files if configured
    _cleanup_paths(cfg, logger)

    # Config blocks
    data_cfg    = cfg.get("data", {})
    out_cfg     = cfg.get("output", {})
    stages_cfg  = cfg.get("stages", {})
    w2v_cfg     = cfg.get("w2v", {})
    sbert_cfg   = cfg.get("sbert", {})
    enc_cfg     = cfg.get("encoder", {})

    # Paths & columns
    input_path          = _as_path(data_cfg, "input_path")
    ner_col             = data_cfg.get("ner_col", "NER")
    chunksize           = int(data_cfg.get("chunksize", 200_000))

    baseline_parquet    = _as_path(out_cfg, "baseline_parquet")      # spaCy-normalized: NER_clean
    final_dedup_parquet = _as_path(out_cfg, "dedup_parquet")
    dedupe_map_path     = _as_path(out_cfg, "cosine_map_path")       # JSONL map written by deduper
    list_col_for_vocab  = out_cfg.get("list_col_for_vocab", "NER_clean")

    unified_parquet     = _as_path(out_cfg, "unified_parquet")
    ing_id_to_token     = _as_path(out_cfg, "ingredient_id_to_token")
    ing_token_to_id     = _as_path(out_cfg, "ingredient_token_to_id")

    # Stage toggles
    do_write_parquet = bool(stages_cfg.get("write_parquet", True))      # spaCy normalize → baseline
    use_sbert        = bool(stages_cfg.get("sbert_dedupe", True))       # prefer SBERT by default
    use_w2v          = bool(stages_cfg.get("w2v_dedupe", False))        # W2V optional fallback
    do_apply_map     = bool(stages_cfg.get("apply_cosine_map", True))   # apply → dedup_parquet
    do_encode_ids    = bool(stages_cfg.get("encode_ids", True))         # encode → unified

    # Enforce mutual exclusivity
    if use_sbert and use_w2v:
        logger.warning("[stages] Both sbert_dedupe and w2v_dedupe are True; proceeding with SBERT and ignoring W2V.")
        use_w2v = False

    # ----------------------------
    # Stage 1: spaCy normalize → baseline_parquet (NER_clean)
    # ----------------------------
    if do_write_parquet and (args.force or not _exists(baseline_parquet)):
        logger.info("Stage 1: spaCy normalization → baseline_parquet (NER_clean)")

        # Ensure a Parquet source with list[str] column `ner_col`
        tmp_raw_parquet = baseline_parquet.with_name("_raw_source_for_spacy.parquet")
        source_parquet = _materialize_parquet_source(input_path, ner_col, chunksize, tmp_raw_parquet)

        baseline_parquet.parent.mkdir(parents=True, exist_ok=True)
        try:
            apply_spacy_normalizer_to_parquet(
                in_parquet=str(source_parquet),
                out_parquet=str(baseline_parquet),
                list_col=ner_col,                  # raw list column ('NER')
                out_col=list_col_for_vocab,        # 'NER_clean'
                spacy_model=sbert_cfg.get("spacy_model", "en_core_web_sm")
            )
        except OSError as e:
            # Typical spaCy model missing error (E050)
            logger.error(
                "spaCy model not found. Install it, e.g.:\n"
                "  python -m spacy download en_core_web_sm\n"
                "…or set stages.write_parquet=false to skip Stage 1 if you already have baseline_parquet."
            )
            raise
        logger.info(f"Saved baseline (spaCy-normalized) Parquet → {baseline_parquet}")
    else:
        logger.info("Stage 1: spaCy normalization – skipped (exists or disabled)")

    # quick probe
    try:
        if _exists(baseline_parquet):
            pf = pq.ParquetFile(str(baseline_parquet))
            if pf.num_row_groups > 0:
                tbl = pf.read_row_group(0, columns=[list_col_for_vocab])
                s = tbl.to_pandas()[list_col_for_vocab]
                non_empty = s.explode().dropna().astype(str).str.strip()
                logger.info(f"[probe] RG0: non-empty tokens={ (non_empty != '').sum() }, unique={ non_empty.nunique() }")
    except Exception as e:
        logger.info(f"[probe] Skipped due to error: {e}")

    if not _exists(baseline_parquet):
        raise FileNotFoundError(
            f"baseline_parquet not found at {baseline_parquet}. "
            "Enable Stage 1 or provide it via config."
        )

    # Build vocab for dedupe (from NER_clean)
    min_freq_for_vocab = int(sbert_cfg.get("min_freq_for_vocab", w2v_cfg.get("min_freq_for_vocab", 1)))
    vocab = vocab_from_parquet_listcol(
        str(baseline_parquet),
        col=list_col_for_vocab,
        min_freq=min_freq_for_vocab
    )
    logger.info(f"[dedupe] Vocab size from baseline ({list_col_for_vocab}) with min_freq={min_freq_for_vocab}: {len(vocab)}")

    # ----------------------------
    # Stage 2: Deduper (SBERT preferred)
    # ----------------------------
    if (use_sbert or use_w2v) and (args.force or not _exists(dedupe_map_path)):
        if len(vocab) == 0:
            logger.warning("[dedupe] Vocab is empty; skipping dedupe and map application.")
            do_apply_map = False
        else:
            dedupe_map_path.parent.mkdir(parents=True, exist_ok=True)
            if use_sbert:
                logger.info("Stage 2: SBERT de-dupe – building phrase map from baseline Parquet")
                from ingrnorm.sbert_dedupe import sbert_dedupe  # import here to avoid hard dep if disabled
                sbert_dedupe(
                    vocab_counter=vocab,
                    out_path=str(dedupe_map_path),
                    model_name=sbert_cfg.get("model", "all-MiniLM-L6-v2"),
                    threshold=float(sbert_cfg.get("threshold", 0.88)),
                    topk=int(sbert_cfg.get("topk", 25)),
                    min_len=int(sbert_cfg.get("min_len", 2)),
                    require_token_overlap=bool(sbert_cfg.get("require_token_overlap", True)),
                    block_generic_as_canon=bool(sbert_cfg.get("block_generic_as_canon", True)),
                )
                logger.info(f"Saved SBERT dedupe map → {dedupe_map_path}")

            elif use_w2v:
                logger.info("Stage 2: W2V de-dupe – building phrase map from baseline Parquet")
                w2v_dedupe(
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
                logger.info(f"Saved W2V dedupe map → {dedupe_map_path}")
    else:
        logger.info("Stage 2: Dedupe – skipped (exists or disabled)")

    # ----------------------------
    # Stage 3: Apply map → final deduped parquet
    # ----------------------------
    if do_apply_map and (args.force or not _exists(final_dedup_parquet)):
        if not _exists(dedupe_map_path):
            logger.warning(f"[apply] Map not found at {dedupe_map_path}; skipping Stage 3.")
        else:
            logger.info("Stage 3: Apply dedupe map → final deduped Parquet")
            apply_map_to_parquet(
                in_path=str(baseline_parquet),
                out_path=str(final_dedup_parquet),
                mapping=str(dedupe_map_path),
                list_col=list_col_for_vocab,
            )
            logger.info(f"Saved final deduped Parquet → {final_dedup_parquet}")
    else:
        logger.info("Stage 3: Apply map – skipped (exists or disabled)")

    # ----------------------------
    # Stage 4: Encode to IDs
    # ----------------------------
    if do_encode_ids and (args.force or not _exists(unified_parquet)):
        logger.info("Stage 4: Encode to IDs – starting")
        candidates = []
        if _exists(final_dedup_parquet):
            candidates.append(final_dedup_parquet)
        if _exists(baseline_parquet):
            candidates.append(baseline_parquet)
        if not candidates:
            raise FileNotFoundError("No parquet available for encoding (dedup/baseline).")
        source_parquet = candidates[0]
        logger.info(f"[encode] Source selected → {source_parquet}")

        enc_min_freq    = int(enc_cfg.get("min_freq", 1))
        dataset_id      = int(enc_cfg.get("dataset_id", 1))
        ingredients_col = enc_cfg.get("ingredients_col", list_col_for_vocab)

        enc = IngredientEncoder(min_freq=enc_min_freq)
        logger.info(f"[encode] Fitting encoder from {source_parquet} (col={ingredients_col}, min_freq={enc_min_freq}) …")
        enc.fit_from_parquet_streaming(source_parquet, col=ingredients_col, min_freq=enc_min_freq).freeze()

        # Persist maps
        ing_id_to_token.parent.mkdir(parents=True, exist_ok=True)
        enc.save_maps(id_to_token_path=ing_id_to_token, token_to_id_path=ing_token_to_id)
        logger.info(f"Saved encoder maps → {ing_id_to_token}, {ing_token_to_id}")

        # Stream-encode to unified
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

    logger.info("Workflow complete.")


if __name__ == "__main__":
    main()
