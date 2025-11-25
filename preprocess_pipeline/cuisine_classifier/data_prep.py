import math
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from spacy.tokens import Doc, DocBin
from tqdm import tqdm

from .config import DATA, TRAIN, OUT
from .utils import (
    load_data,
    parse_listlike,
    normalize_cuisine_label,
)


def docs_from_text_and_labels(df: pd.DataFrame, text_col: str, cuisine_col: str) -> List[Doc]:
    """Create spaCy Docs from text column with cuisine labels for text classification."""
    blank = spacy.blank("en")
    out: List[Doc] = []
    
    # First pass: Get unique cuisine labels
    unique_cuisines = set()
    for cuisine_val in df[cuisine_col]:
        cuisine = normalize_cuisine_label(cuisine_val)
        if cuisine:
            unique_cuisines.add(cuisine)
    
    all_labels = sorted(list(unique_cuisines))
    logger = __import__("logging").getLogger(__name__)
    logger.info(f"Found {len(all_labels)} unique cuisine labels")
    
    # Second pass: Create docs with all labels set (exclusive classification)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating docs from text and labels"):
        # Parse text column (may be list-like or string)
        text_val = row[text_col]
        text_tokens = parse_listlike(text_val)
        
        if not text_tokens:
            # Empty text, skip
            continue
        
        # Join tokens into text
        text = ", ".join(text_tokens)
        
        # Get cuisine label
        cuisine = normalize_cuisine_label(row[cuisine_col])
        if not cuisine:
            # Skip rows without valid cuisine labels
            continue
        
        # Create doc
        doc = blank.make_doc(text)
        
        # Set textcat labels (exclusive classification: one True, rest False)
        doc.cats = {label: 1.0 if label == cuisine else 0.0 for label in all_labels}
        
        out.append(doc)
    
    return out


def build_docs_from_config() -> Tuple[List[Doc], List[Doc], str]:
    """High-level helper: build train & valid docs based on DATA + TRAIN config."""
    logger = __import__("logging").getLogger(__name__)
    
    # Debug info
    logger.info(f"[DEBUG] ===== Data Loading Configuration =====")
    logger.info(f"[DEBUG] TRAIN_PATH: {DATA.TRAIN_PATH}")
    logger.info(f"[DEBUG] Absolute path: {DATA.TRAIN_PATH.resolve()}")
    logger.info(f"[DEBUG] File exists: {DATA.TRAIN_PATH.exists()}")
    logger.info(f"[DEBUG] Data format: {'Parquet' if DATA.DATA_IS_PARQUET else 'CSV'}")
    logger.info(f"[DEBUG] TEXT_COL: {DATA.TEXT_COL}")
    logger.info(f"[DEBUG] CUISINE_COL: {DATA.CUISINE_COL}")
    logger.info(f"[DEBUG] ======================================")

    # Load data
    df = load_data(
        DATA.TRAIN_PATH, 
        DATA.DATA_IS_PARQUET, 
        DATA.TEXT_COL, 
        DATA.CUISINE_COL,
        max_rows=DATA.MAX_ROWS
    )
    
    logger.info(f"Loaded {len(df):,} rows from {DATA.TRAIN_PATH}")
    
    # Create docs
    docs_all = docs_from_text_and_labels(df, DATA.TEXT_COL, DATA.CUISINE_COL)
    
    # Apply max_train_docs limit if set (for debug mode)
    if hasattr(TRAIN, 'MAX_TRAIN_DOCS') and TRAIN.MAX_TRAIN_DOCS is not None:
        if len(docs_all) > TRAIN.MAX_TRAIN_DOCS:
            logger.info(f"[DEBUG] Limiting training docs to {TRAIN.MAX_TRAIN_DOCS} for debug mode")
            docs_all = docs_all[:TRAIN.MAX_TRAIN_DOCS]
    
    logger.info(f"Docs prepared: {len(docs_all):,}")
    
    # Get all unique cuisine labels for logging
    all_cuisines = set()
    for doc in docs_all:
        all_cuisines.update(doc.cats.keys())
    logger.info(f"Total unique cuisine labels: {len(all_cuisines)}")
    
    # Split train/valid
    train_docs, valid_docs = train_test_split(
        docs_all, test_size=TRAIN.VALID_FRACTION, random_state=TRAIN.RANDOM_SEED
    )
    logger.info(f"train={len(train_docs):,} | valid={len(valid_docs):,}")

    return train_docs, valid_docs, "text-classification"


def write_docbins(docs: List[Doc], out_dir: Path, shard_size: int) -> None:
    """Write docs into sharded DocBin files."""

    def clean_dir(path: Path):
        path.mkdir(parents=True, exist_ok=True)
        for p in path.glob("*.spacy"):
            logger = __import__("logging").getLogger(__name__)
            logger.info(f"[CLEAN] Removing old shard: {p}")
            p.unlink()
    
    clean_dir(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(docs)
    shards = max(1, math.ceil(n / max(1, shard_size)))
    for i in range(shards):
        db = DocBin(store_user_data=False)
        for d in docs[i * shard_size: (i + 1) * shard_size]:
            db.add(d)
        db.to_disk(out_dir / f"shard_{i:04d}.spacy")
    logger = __import__("logging").getLogger(__name__)
    logger.info(f"Wrote {n} docs to {out_dir} in {shards} shard(s).")


def prepare_docbins_from_config() -> None:
    """End-to-end: build docs and write train/valid DocBins according to config."""
    train_docs, valid_docs, _ = build_docs_from_config()
    write_docbins(train_docs, OUT.TRAIN_DIR, shard_size=TRAIN.SHARD_SIZE)
    write_docbins(valid_docs, OUT.VALID_DIR, shard_size=TRAIN.SHARD_SIZE)


__all__ = [
    "docs_from_text_and_labels",
    "build_docs_from_config",
    "write_docbins",
    "prepare_docbins_from_config",
]

