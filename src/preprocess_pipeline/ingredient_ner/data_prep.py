import math
import warnings
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Iterator

import pandas as pd
import spacy
from spacy.tokens import Doc, DocBin
from tqdm import tqdm

from .config import DATA, TRAIN, OUT
from .utils import (
    load_data,
    parse_listlike,
    join_with_offsets,
    normalize_token,
)

# -----------------------------------------------------------------------------
# Streaming Helpers
# -----------------------------------------------------------------------------
def load_lexicon(path: Path | None = None) -> list[str]:
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        warnings.warn(f"Lexicon not found at {p}. Skipping.")
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Expect either {"terms": [...]} or a simple list [...]
    if isinstance(data, dict) and "terms" in data:
        terms = data["terms"]
    else:
        terms = data
    # normalize
    terms = [normalize_token(t) for t in terms if str(t).strip()]
    terms = sorted(set(terms))
    print(f"Loaded {len(terms):,} lexicon terms.")
    return terms

def iter_docs_from_list_column(df: pd.DataFrame, col: str) -> Iterator[Doc]:
    """
    Generator that yields spaCy Docs one by one from a list-like column.
    Memory efficient: does not hold the full list in RAM.
    """
    blank = spacy.blank("en")
    
    # Iterate rows
    for lst in tqdm(df[col], desc="Streaming docs (per-ingredient)"):
        toks = parse_listlike(lst)
        if not toks:
            continue
        
        # Iterate ingredients in the row
        for ingredient_text in toks:
            ingredient_text = str(ingredient_text).strip()
            if not ingredient_text:
                continue
            
            # Create doc
            d = blank.make_doc(ingredient_text)
            
            # Label entire string as INGREDIENT
            span = d.char_span(0, len(ingredient_text), label="INGREDIENT", alignment_mode="contract")
            if span:
                d.ents = [span]
            else:
                # Fallback
                try:
                    d.ents = [spacy.tokens.Span(d, 0, len(d), label="INGREDIENT")]
                except Exception:
                    d.ents = []
            
            yield d

def stream_docs_to_disk(
    doc_iterator: Iterator[Doc],
    train_dir: Path,
    valid_dir: Path,
    valid_fraction: float = 0.2,
    shard_size: int = 2500
) -> None:
    """
    Consumes a doc iterator, splits into Train/Valid on the fly, 
    and writes to disk in chunks (shards).
    """
    # 1. Setup Directories
    for p in [train_dir, valid_dir]:
        if p.exists():
            shutil.rmtree(p) # Clean start
        p.mkdir(parents=True, exist_ok=True)

    # 2. Initialize Buffers
    buffers = {
        "train": DocBin(store_user_data=False),
        "valid": DocBin(store_user_data=False)
    }
    counts = {"train": 0, "valid": 0}
    shard_indices = {"train": 0, "valid": 0}

    print(f"[STREAM] Processing docs... (Shard size: {shard_size})")

    # 3. Stream & Split
    for doc in doc_iterator:
        # Deterministic split based on content hash (reproducible) 
        # or simple random if prefered.
        split = "valid" if random.random() < valid_fraction else "train"
        
        buffers[split].add(doc)
        counts[split] += 1

        # 4. Check Buffer Limit
        if len(buffers[split]) >= shard_size:
            _flush_buffer(buffers[split], split, shard_indices[split], train_dir if split == "train" else valid_dir)
            buffers[split] = DocBin(store_user_data=False) # Reset
            shard_indices[split] += 1

    # 5. Flush Leftovers
    for split, db in buffers.items():
        if len(db) > 0:
            _flush_buffer(db, split, shard_indices[split], train_dir if split == "train" else valid_dir)

    print(f"[STREAM] Complete.")
    print(f"  - Train: {counts['train']:,} docs")
    print(f"  - Valid: {counts['valid']:,} docs")

def _flush_buffer(db: DocBin, split_name: str, index: int, out_dir: Path):
    path = out_dir / f"shard_{index:04d}.spacy"
    db.to_disk(path)
    # Optional: print(f"    Wrote {split_name} shard {index} ({len(db)} docs)")

# -----------------------------------------------------------------------------
# Main Config Handler
# -----------------------------------------------------------------------------

def prepare_docbins_from_config() -> None:
    """
    Main entry point. Loads data iterator and streams to disk.
    """
    print(f"[DEBUG] Loading data from {DATA.TRAIN_PATH}...")
    
    # 1. Identify Column
    df_head = pd.read_parquet(DATA.TRAIN_PATH) if DATA.DATA_IS_PARQUET else pd.read_csv(DATA.TRAIN_PATH, nrows=5)
    cols = list(df_head.columns)
    
    ner_col = DATA.NER_LIST_COL
    if not ner_col or ner_col not in cols:
        # Auto-detect
        candidates = ["ingredients", "ner", "ingredient_list"]
        for c in candidates:
            if c in cols:
                ner_col = c
                break
    
    if not ner_col:
        raise ValueError(f"Could not find ingredient column. Available: {cols}")

    print(f"[DEBUG] Streaming from column: '{ner_col}'")
    
    # 2. Load Data (Full DF needed for iteration, but we won't create Docs list)
    # Optimization: If parquet, you can iterate row groups, but loading DF is usually fine 
    # as long as we don't create the huge list of Docs.
    df = load_data(DATA.TRAIN_PATH, DATA.DATA_IS_PARQUET, ner_col, max_rows=DATA.MAX_ROWS)
    
    # 3. Create Iterator
    doc_gen = iter_docs_from_list_column(df, ner_col)
    
    # 4. Stream to Disk
    stream_docs_to_disk(
        doc_gen,
        train_dir=OUT.TRAIN_DIR,
        valid_dir=OUT.VALID_DIR,
        valid_fraction=TRAIN.VALID_FRACTION,
        shard_size=getattr(TRAIN, "SHARD_SIZE", 2500) # Default to 2500 if missing
    )


# Expose for import
__all__ = ["prepare_docbins_from_config"]