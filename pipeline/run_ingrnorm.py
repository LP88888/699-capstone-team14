from __future__ import annotations
import logging
from pathlib import Path
import argparse
import json
import sys
import time
import pathlib as _pathlib

# Make `pipeline/` importable
sys.path.append(str(_pathlib.Path.cwd() / "pipeline"))

# External deps
try:
    import yaml
except Exception:
    print("[FATAL] Please `pip install pyyaml` to read config.yaml")
    raise

import pandas as pd

from ingrnorm.stats_normalizer import StatsNormalizer
from ingrnorm.spellmap import build_spell_map
from ingrnorm.parquet_utils import vocab_from_parquet_listcol
from ingrnorm.cosdedupe import cosine_dedupe, apply_cosine_map_to_parquet
from ingrnorm.encoder import IngredientEncoder
from common.logging_setup import setup_logging  
logger = logging.getLogger("ingrnorm.encoder")


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





def _ingest_any(normalizer: StatsNormalizer, input_path: Path, ner_col: str, chunksize: int) -> None:
    """
    Ingest either CSV or Parquet into the StatsNormalizer counters.
    - CSV uses stream read via StatsNormalizer.ingest_csv
    - Parquet row-groups are streamed and fed to ingest_df
    """
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        logger.info(f"Ingesting CSV in chunksize={chunksize} …")
        normalizer.ingest_csv(input_path, ner_col=ner_col, chunksize=chunksize)
        return

    if suffix == ".parquet":
        logger.info("Ingesting Parquet row-groups …")
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(input_path)
        for rg in range(pf.num_row_groups):
            df = pf.read_row_group(rg, columns=[ner_col]).to_pandas()
            if ner_col not in df.columns:
                df[ner_col] = None
            normalizer.ingest_df(df, ner_col=ner_col)
            del df
        return

    if suffix in (".xlsx", ".xls"):
        logger.info("Ingesting Excel (single shot) …")
        df = pd.read_excel(input_path, dtype=str)
        if ner_col not in df.columns:
            df[ner_col] = None
        normalizer.ingest_df(df, ner_col=ner_col)
        return

    raise ValueError(f"Unsupported file type for ingest: {input_path.name}")


def main():
    ap = argparse.ArgumentParser(description="Run ingrnorm workflow")
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

    data_cfg    = cfg.get("data", {})
    out_cfg     = cfg.get("output", {})
    norm_cfg    = cfg.get("normalizer", {})
    stages_cfg  = cfg.get("stages", {})
    cosine_cfg  = cfg.get("cosine", {})
    enc_cfg     = cfg.get("encoder", {})          # NEW

    input_path  = _as_path(data_cfg, "input_path")
    ner_col     = data_cfg.get("ner_col", "NER")
    chunksize   = int(data_cfg.get("chunksize", 200_000))

    vocab_json          = _as_path(out_cfg, "vocab_json")
    baseline_parquet    = _as_path(out_cfg, "baseline_parquet")
    spell_map_path      = _as_path(out_cfg, "spell_map_path")
    cosine_map_path     = _as_path(out_cfg, "cosine_map_path")
    final_dedup_parquet = _as_path(out_cfg, "dedup_parquet")
    list_col_for_vocab  = out_cfg.get("list_col_for_vocab", "NER_clean")

    # NEW encoder outputs
    unified_parquet     = _as_path(out_cfg, "unified_parquet")
    ing_id_to_token     = _as_path(out_cfg, "ingredient_id_to_token")
    ing_token_to_id     = _as_path(out_cfg, "ingredient_token_to_id")

    # Stages
    do_build_vocab   = bool(stages_cfg.get("build_vocab", True))
    do_spell_map     = bool(stages_cfg.get("build_spell_map", False))
    do_write_parquet = bool(stages_cfg.get("write_parquet", True))
    do_cos_dedupe    = bool(stages_cfg.get("cosine_dedupe", True))
    do_apply_cos_map = bool(stages_cfg.get("apply_cosine_map", True))
    do_encode_ids    = bool(stages_cfg.get("encode_ids", True))      # NEW

    # 1) Build vocab
    if do_build_vocab and (args.force or not _exists(vocab_json)):
        logger.info("Stage 1: Build vocab – starting")
        normalizer = StatsNormalizer(**norm_cfg)
        logger.info(f"Ingesting from {input_path}")
        _ingest_any(normalizer, input_path, ner_col=ner_col, chunksize=chunksize)

        logger.info("Building canonical vocabulary …")
        normalizer.build_vocab()

        vocab_json.parent.mkdir(parents=True, exist_ok=True)
        normalizer.save_vocab(vocab_json)
        logger.info(f"Saved vocab → {vocab_json}")
    else:
        logger.info("Stage 1: Build vocab – skipped (exists or disabled)")

    # Load normalizer for subsequent stages
    logger.info("Loading vocab for downstream stages …")
    normalizer = StatsNormalizer.load_vocab(vocab_json)

    # 2) Spell map (optional)
    if do_spell_map and (args.force or not _exists(spell_map_path)):
        logger.info("Stage 2: Build spell/fuzzy map – starting")
        canon_phrases = [" ".join(p) for p in normalizer.canon]
        build_spell_map(
            canon_phrases=canon_phrases,
            csv_path=str(input_path),
            ner_col=ner_col,
            out_path=spell_map_path,
            chunksize=chunksize,
        )
        logger.info(f"Saved spell map → {spell_map_path}")
    else:
        logger.info("Stage 2: Build spell/fuzzy map – skipped (exists or disabled)")

    # 3) Baseline cleaned parquet
    if do_write_parquet and (args.force or not _exists(baseline_parquet)):
        logger.info("Stage 3: Write baseline cleaned Parquet – starting")
        baseline_parquet.parent.mkdir(parents=True, exist_ok=True)
        normalizer.transform_csv_to_parquet(
            csv_path=str(input_path),
            out_path=str(baseline_parquet),
            ner_col=ner_col,
            chunksize=chunksize,
        )
        logger.info(f"Saved baseline cleaned Parquet → {baseline_parquet}")
    else:
        logger.info("Stage 3: Write baseline cleaned Parquet – skipped (exists or disabled)")

    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(baseline_parquet))
        if pf.num_row_groups > 0:
            tbl = pf.read_row_group(0, columns=[list_col_for_vocab])
            s = tbl.to_pandas()[list_col_for_vocab]
            non_empty = s.explode().dropna().astype(str).str.strip()
            logger.info(f"[probe] RG0: non-empty tokens={ (non_empty != '').sum() }, unique={ non_empty.nunique() }")
    except Exception as e:
        logger.info(f"[probe] Skipped due to error: {e}")

    # 4) Cosine de-dupe (build map from baseline parquet)
    if do_cos_dedupe and (args.force or not _exists(cosine_map_path)):
        logger.info("Stage 4: Cosine de-dupe – starting (scanning vocab from baseline Parquet)")
        vocab = vocab_from_parquet_listcol(str(baseline_parquet), col=list_col_for_vocab)
        logger.info(f"Collected vocab size={len(vocab)} from {baseline_parquet}")

        if len(vocab) == 0:
            logger.info("[WARN] Vocab is empty; skipping cosine de-dupe and map application.")
            # Make sure Stage 5 also skips
            do_apply_cos_map = False
        else:
            cosine_map_path.parent.mkdir(parents=True, exist_ok=True)
            # NOTE: cosine_dedupe signature takes the vocab counter as the first positional arg.
            cosine_dedupe(
                vocab,  # <-- pass as positional, not keyword
                threshold=float(cfg.get("cosine", {}).get("threshold", 0.88)),
                topk=int(cfg.get("cosine", {}).get("topk", 20)),
                out_path=str(cosine_map_path),
            )
            logger.info(f"Saved cosine-dedupe map → {cosine_map_path}")
    else:
        logger.info("Stage 4: Cosine de-dupe – skipped (exists or disabled)")

    # 5) Apply cosine map to produce final deduped parquet
    if do_apply_cos_map and (args.force or not _exists(final_dedup_parquet)):
        if not _exists(cosine_map_path):
            logger.info(f"[WARN] Cosine map not found at {cosine_map_path}; skipping Stage 5.")
        else:
            logger.info("Stage 5: Apply cosine map – starting")
            apply_cosine_map_to_parquet(
                in_path=str(baseline_parquet),
                out_path=str(final_dedup_parquet),
                mapping=str(cosine_map_path),
                list_col=list_col_for_vocab,
            )
            logger.info(f"Saved final deduped Parquet → {final_dedup_parquet}")
    else:
        logger.info("Stage 5: Apply cosine map – skipped (exists or disabled)")

    # 6) Encode to IDs (IngredientEncoder) — ingredients only
    if do_encode_ids and (args.force or not _exists(unified_parquet)):
        logger.info("Stage 6: Encode to IDs – starting")
        # Choose source for encoding: prefer final dedup if built; fall back to baseline
        source_parquet = final_dedup_parquet if _exists(final_dedup_parquet) else baseline_parquet
        if not _exists(source_parquet):
            raise FileNotFoundError(f"No source parquet to encode: {source_parquet}")

        # Configure encoder
        enc_min_freq = int(enc_cfg.get("min_freq", 1))
        dataset_id   = int(enc_cfg.get("dataset_id", 1))
        ingredients_col = enc_cfg.get("ingredients_col", list_col_for_vocab)

        enc = IngredientEncoder(min_freq=enc_min_freq)
        logger.info(f"Fitting encoder from {source_parquet} (col={ingredients_col}, min_freq={enc_min_freq}) …")
        enc.fit_from_parquet_streaming(source_parquet, col=ingredients_col, min_freq=enc_min_freq).freeze()

        # Persist maps
        ing_id_to_token.parent.mkdir(parents=True, exist_ok=True)
        enc.save_maps(id_to_token_path=ing_id_to_token, token_to_id_path=ing_token_to_id)
        logger.info(f"Saved encoder maps → {ing_id_to_token}, {ing_token_to_id}")

        # Stream-encode to unified
        logger.info(f"Encoding → {unified_parquet}")
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
        logger.info("Stage 6: Encode to IDs – skipped (exists or disabled)")

    logger.info("Workflow complete.")


if __name__ == "__main__":
    main()
