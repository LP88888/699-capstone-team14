import logging
import random
import shutil
from pathlib import Path
from typing import Iterator, List, Tuple, Set, Optional
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq
import spacy
from spacy.tokens import Doc, DocBin
from tqdm import tqdm

from .config import DATA, TRAIN, OUT
from .utils import load_data, parse_listlike, normalize_cuisine_label


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- 
# Column + label helpers
# ----------------------------------------------------------------------------- 

def _detect_columns(available_cols: List[str]) -> Tuple[str, str]:
    """
    Choose text and cuisine columns, preferring normalized/deduped cuisine labels
    and short ingredient token lists. If configured columns exist, enforce them
    rather than silently falling back to raw fields.
    """
    logger.info("Available columns: %s", available_cols)

    # 1) Enforce configured text column if present
    if DATA.TEXT_COL and DATA.TEXT_COL in available_cols:
        text_col = DATA.TEXT_COL
        logger.info("Using configured text column: %s", text_col)
    else:
        candidates = [
            "inferred_ingredients",
            "NER_clean",
            "ingredients",
            "ingredient_list",
            "ingredients_list",
            "text",
        ]
        text_col = None
        for candidate in candidates:
            for col in available_cols:
                if col.lower() == candidate.lower():
                    text_col = col
                    logger.info("Auto-detected text column: %s", text_col)
                    break
            if text_col:
                break

    # 2) Prefer deduped/clean cuisine labels
    if DATA.CUISINE_COL and DATA.CUISINE_COL in available_cols:
        cuisine_col = DATA.CUISINE_COL
        logger.info("Using configured cuisine column: %s", cuisine_col)
    else:
        candidates = [
            "cuisine_deduped",
            "cuisine_clean",
            "cuisine",
            "cuisines",
            "category",
            "type",
        ]
        cuisine_col = None
        for candidate in candidates:
            for col in available_cols:
                if col.lower() == candidate.lower():
                    cuisine_col = col
                    logger.info("Auto-detected cuisine column: %s", cuisine_col)
                    break
            if cuisine_col:
                break

    if not text_col or text_col not in available_cols:
        raise RuntimeError(
            f"Text column not found. Requested '{DATA.TEXT_COL}', searched candidates; "
            f"available: {available_cols}"
        )
    if not cuisine_col or cuisine_col not in available_cols:
        raise RuntimeError(
            f"Cuisine column not found. Requested '{DATA.CUISINE_COL}', searched candidates; "
            f"available: {available_cols}"
        )

    return text_col, cuisine_col


def _load_allowed_labels(path: Optional[str | Path]) -> Set[str]:
    """Load an optional allowlist of labels (values of id->token map or simple list)."""
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        logger.warning("allowed_labels_path not found: %s", p)
        return set()
    try:
        import json
        data = json.load(open(p, "r", encoding="utf-8"))
        if isinstance(data, dict):
            labels = {str(v).strip() for v in data.values() if str(v).strip()}
        elif isinstance(data, list):
            labels = {str(v).strip() for v in data if str(v).strip()}
        else:
            labels = set()
        logger.info("Loaded %s allowed labels from %s", len(labels), p)
        return labels
    except Exception as e:
        logger.warning("Failed to load allowed labels from %s: %s", p, e)
        return set()


def _load_parent_map(path: Optional[str | Path]) -> dict[str, str]:
    """Optional hierarchy map: child -> parent."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        logger.warning("parent_map_path not found: %s", p)
        return {}
    try:
        import json
        data = json.load(open(p, "r", encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if str(k).strip() and str(v).strip()}
    except Exception as e:
        logger.warning("Failed to load parent map from %s: %s", p, e)
    return {}


def _collect_labels(df: pd.DataFrame, cuisine_col: str, allowed: Set[str], min_freq: int, parent_map: dict[str, str]) -> Tuple[List[str], Counter]:
    """Collect normalized labels with optional allowlist and min frequency filter."""
    counts: Counter = Counter()
    for cuisine_val in df[cuisine_col]:
        cuisine = normalize_cuisine_label(cuisine_val)
        if not cuisine:
            continue
        if parent_map:
            cuisine = parent_map.get(cuisine, cuisine)
        if allowed and cuisine not in allowed:
            continue
        counts[cuisine] += 1

    labels = [lbl for lbl, c in counts.items() if c >= max(1, min_freq)]
    labels_sorted = sorted(labels)
    logger.info(
        "Label filtering: %s raw labels -> %s kept (min_freq=%s, allowed=%s)",
        len(counts),
        len(labels_sorted),
        min_freq,
        "yes" if allowed else "no",
    )
    return labels_sorted, counts


def _collect_labels(df: pd.DataFrame, cuisine_col: str) -> List[str]:
    labels = set()
    for val in df[cuisine_col]:
        cuisine = normalize_cuisine_label(val)
        if cuisine:
            labels.add(cuisine)
    all_labels = sorted(labels)
    logger.info("Found %s unique cuisine labels (normalized)", len(all_labels))
    if not all_labels:
        raise ValueError(f"No cuisine labels found in column '{cuisine_col}'")
    return all_labels


# ----------------------------------------------------------------------------- 
# Streaming doc creation (per ingredient)
# ----------------------------------------------------------------------------- 

def iter_docs_from_list_column(
    df: pd.DataFrame,
    text_col: str,
    cuisine_col: str,
    all_labels: List[str],
) -> Iterator[Doc]:
    """
    Stream spaCy Docs one by one, treating each ingredient/token as its own example.
    Keeps text short and ensures we only use normalized cuisine labels.
    """
    blank = spacy.blank("en")
    label_set = set(all_labels)

    for text_val, cuisine_val in tqdm(
        zip(df[text_col], df[cuisine_col]),
        total=len(df),
        desc="Streaming cuisine docs (per-ingredient)",
    ):
        cuisine = normalize_cuisine_label(cuisine_val)
        if not cuisine or cuisine not in label_set:
            continue

        for ingredient in parse_listlike(text_val):
            ingredient = str(ingredient).strip()
            if not ingredient:
                continue

            doc = blank.make_doc(ingredient)
            doc.cats = {label: 1.0 if label == cuisine else 0.0 for label in all_labels}
            yield doc


def _flush_buffer(db: DocBin, split_name: str, index: int, out_dir: Path) -> None:
    path = out_dir / f"shard_{index:04d}.spacy"
    db.to_disk(path)


def stream_docs_to_disk(
    doc_iterator: Iterator[Doc],
    train_dir: Path,
    valid_dir: Path,
    valid_fraction: float = 0.2,
    shard_size: int = 2500,
) -> Tuple[int, int]:
    """Split docs into train/valid on the fly and write shards to disk."""
    for p in (train_dir, valid_dir):
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    buffers = {
        "train": DocBin(store_user_data=False),
        "valid": DocBin(store_user_data=False),
    }
    counts = {"train": 0, "valid": 0}
    shard_indices = {"train": 0, "valid": 0}

    rng = random.Random(TRAIN.RANDOM_SEED)
    logger.info("Streaming docs to disk (shard_size=%s, valid_fraction=%.2f)", shard_size, valid_fraction)

    for doc in doc_iterator:
        split = "valid" if rng.random() < valid_fraction else "train"
        buffers[split].add(doc)
        counts[split] += 1

        if len(buffers[split]) >= shard_size:
            _flush_buffer(buffers[split], split, shard_indices[split], train_dir if split == "train" else valid_dir)
            buffers[split] = DocBin(store_user_data=False)
            shard_indices[split] += 1

    for split, db in buffers.items():
        if len(db) > 0:
            _flush_buffer(db, split, shard_indices[split], train_dir if split == "train" else valid_dir)

    logger.info("Doc streaming complete: train=%s, valid=%s", counts["train"], counts["valid"])
    return counts["train"], counts["valid"]


# ----------------------------------------------------------------------------- 
# Main entry point
# ----------------------------------------------------------------------------- 

def prepare_docbins_from_config() -> None:
    """Load data, build label set, and stream docbins to disk using config values."""
    logger.info("Preparing cuisine docbins from %s", DATA.TRAIN_PATH)

    if DATA.DATA_IS_PARQUET:
        # Read first row group to get actual column names (handles list columns)
        pf = pq.ParquetFile(str(DATA.TRAIN_PATH))
        df_sample = pf.read_row_group(0).to_pandas().head(1)
        available_cols = list(df_sample.columns)
    else:
        df_sample = pd.read_csv(DATA.TRAIN_PATH, nrows=0, dtype=str)
        available_cols = list(df_sample.columns)

    text_col, cuisine_col = _detect_columns(available_cols)

    df = load_data(
        DATA.TRAIN_PATH,
        DATA.DATA_IS_PARQUET,
        text_col,
        cuisine_col,
        max_rows=DATA.MAX_ROWS,
    )
    logger.info("Loaded %s rows for cuisine classifier prep", len(df))

    allowed = _load_allowed_labels(getattr(TRAIN, "ALLOWED_LABELS_PATH", None))
    parent_map = _load_parent_map(getattr(TRAIN, "PARENT_MAP_PATH", None))
    min_freq = max(1, int(getattr(TRAIN, "MIN_LABEL_FREQ", 1)))
    all_labels, counts = _collect_labels(df, cuisine_col, allowed, min_freq, parent_map)
    if not all_labels:
        raise RuntimeError("No labels left after filtering. Check allowed_labels_path/min_label_freq.")
    logger.info("Label sample after filtering: %s", all_labels[:10])

    doc_iter = iter_docs_from_list_column(df, text_col, cuisine_col, all_labels)
    stream_docs_to_disk(
        doc_iter,
        train_dir=OUT.TRAIN_DIR,
        valid_dir=OUT.VALID_DIR,
        valid_fraction=TRAIN.VALID_FRACTION,
        shard_size=getattr(TRAIN, "SHARD_SIZE", 2500),
    )


__all__ = [
    "prepare_docbins_from_config",
    "iter_docs_from_list_column",
    "stream_docs_to_disk",
]
