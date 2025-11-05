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

# simplified cleaner
from ingrnorm.simplified_cleaner import (
    load_lexicon,
    build_symspell_from_lexicon,
    clean_token_or_phrase,
    SemanticBackstop,
)
import pyarrow as pa
import pyarrow.parquet as pq


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

def _make_row_cleaner(simple_cfg: dict, logger: logging.Logger):
    # Load lexicon
    sources = [Path(p) for p in simple_cfg.get("lexicon_sources", [])]
    lexicon = load_lexicon(sources)
    logger.info(f"simple_clean: loaded lexicon size={len(lexicon)} from {len(sources)} source(s)")

    # SymSpell toward the lexicon
    sym = build_symspell_from_lexicon(
        lexicon,
        max_edit_distance=int(simple_cfg.get("max_edit_distance", 2))
    )

    # Optional semantic backstop
    semantic = None
    if bool(simple_cfg.get("use_semantic_backstop", False)):
        logger.info("simple_clean: initializing semantic backstop (MiniLM)…")
        semantic = SemanticBackstop(lexicon)

    drop_if_unmapped = bool(simple_cfg.get("drop_if_unmapped", True))
    semantic_threshold = float(simple_cfg.get("semantic_threshold", 0.86))

    def _clean_list(lst):
        if not isinstance(lst, (list, tuple)):
            return []
        out = []
        for x in lst:
            s = str(x).strip()
            if not s:
                continue
            fixed = clean_token_or_phrase(
                s,
                lexicon=lexicon,
                symspell=sym,
                semantic=(semantic if semantic else None),
                drop_if_unmapped=drop_if_unmapped
            )
            if not fixed and semantic:
                # allow override threshold per call
                fixed = semantic.map(s, threshold=semantic_threshold)
            if fixed and (not out or out[-1] != fixed):
                out.append(fixed)
        return out

    return _clean_list

def _apply_simple_clean_to_parquet(
    in_parquet: Path,
    out_parquet: Path,
    list_col: str,
    cleaner_fn,
    logger: logging.Logger,
    compression: str = "zstd",
):
    pf = pq.ParquetFile(str(in_parquet))
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    target_schema = pa.schema([
        pa.field(list_col, pa.list_(pa.string())),
    ])
    writer = pq.ParquetWriter(str(out_parquet), target_schema, compression=compression)

    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=[list_col])
        df = tbl.to_pandas()

        if list_col not in df.columns:
            df[list_col] = [[] for _ in range(len(df))]

        cleaned = df[list_col].apply(lambda v: cleaner_fn(v))
        arr = pa.array(
            [lst if isinstance(lst, (list, tuple)) else [] for lst in cleaned],
            type=pa.list_(pa.string())
        )
        out_tbl = pa.Table.from_arrays([arr], names=[list_col])
        writer.write_table(out_tbl)

        del df, tbl, cleaned, arr, out_tbl

    writer.close()



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
    simple_cfg = cfg.get("simple_clean", {})
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

        # Extract min_freq (optional) from YAML
        min_freq = norm_cfg.pop("min_freq", None)

        # Initialize the StatsNormalizer
        normalizer = StatsNormalizer(**norm_cfg)
        if min_freq is not None:
            normalizer.min_freq = int(min_freq)  # attach custom attribute for filtering
        
        logger.info(f"Ingesting from {input_path} (min_freq={min_freq or 'default'})")
        _ingest_any(normalizer, input_path, ner_col=ner_col, chunksize=chunksize)

        logger.info("Building canonical vocabulary …")
        normalizer.build_vocab()

        if getattr(normalizer, "min_freq", None):
            logger.info(f"Filtering vocab with min_freq ≥ {normalizer.min_freq}")
            # Drop rare tokens directly from canon if frequency stats exist
            if hasattr(normalizer, "freqs"):
                normalizer.canon = {
                    t for t in normalizer.canon
                    if sum(normalizer.freqs.get(tok, 0) for tok in t) >= normalizer.min_freq
                }


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
        min_freq = int(norm_cfg.get("min_freq", 1))
        vocab = vocab_from_parquet_listcol(
            str(baseline_parquet),
            col=list_col_for_vocab,
            min_freq=min_freq
        )
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
    
    # 5b) Simple cleaner (spell → filter → optional semantic), producing a cleaned Parquet
    simple_enabled = bool(simple_cfg.get("enabled", False))
    simple_out_parquet = Path(simple_cfg.get("out_parquet", out_cfg.get("simple_clean_parquet", ""))) if simple_enabled else None

    if simple_enabled and simple_out_parquet:
        # choose best available source: dedup (if produced), else baseline
        source_for_clean = final_dedup_parquet if _exists(final_dedup_parquet) else baseline_parquet
        if not _exists(source_for_clean):
            logger.info(f"[simple_clean] source parquet missing → {source_for_clean}, skipping.")
        elif (not args.force) and _exists(simple_out_parquet):
            logger.info(f"[simple_clean] exists → {simple_out_parquet}; skipping.")
        else:
            logger.info(f"[simple_clean] building cleaner and processing {source_for_clean} → {simple_out_parquet}")
            cleaner_fn = _make_row_cleaner(simple_cfg, logger)
            _apply_simple_clean_to_parquet(
                in_parquet=source_for_clean,
                out_parquet=simple_out_parquet,
                list_col=list_col_for_vocab,
                cleaner_fn=cleaner_fn,
                logger=logger,
                compression="zstd",
            )
            logger.info(f"[simple_clean] wrote → {simple_out_parquet}")
    else:
        logger.info("[simple_clean] disabled or no output path configured; skipping.")
    
    # 6) Encode to IDs (IngredientEncoder) — ingredients only
    if do_encode_ids and (args.force or not _exists(unified_parquet)):
        logger.info("Stage 6: Encode to IDs – starting")
 # Prefer simple_clean parquet if present, then dedup, else baseline
        candidates = []
        if simple_enabled and simple_out_parquet and _exists(simple_out_parquet):
            candidates.append(simple_out_parquet)
        if _exists(final_dedup_parquet):
            candidates.append(final_dedup_parquet)
        if _exists(baseline_parquet):
            candidates.append(baseline_parquet)
        if not candidates:
            raise FileNotFoundError("No parquet available for encoding (simple_clean/dedup/baseline).")
        source_parquet = candidates[0]
        logger.info(f"Encoding source selected → {source_parquet}")
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
