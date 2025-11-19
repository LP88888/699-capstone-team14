import json
import math
import warnings
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
    join_with_offsets,
    normalize_token,
)


def docs_from_list_column(df: pd.DataFrame, col: str) -> List[Doc]:
    """Create spaCy Docs from a list-like ingredient column."""
    blank = spacy.blank("en")
    out: List[Doc] = []
    for lst in tqdm(df[col].tolist(), desc="Synthesizing from list column"):
        toks = parse_listlike(lst)
        if not toks:
            out.append(blank.make_doc(""))
            continue
        text, offs = join_with_offsets(toks)
        d = blank.make_doc(text)
        ents = []
        for (a, b) in offs:
            sp = d.char_span(a, b, label="INGREDIENT", alignment_mode="contract")
            if sp is not None:
                ents.append(sp)
        d.ents = spacy.util.filter_spans(ents)
        out.append(d)
    return out


def load_lexicon(path: Path | None) -> list[str]:
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


def build_entity_ruler(nlp: spacy.language.Language, phrases: list[str]):
    ruler = nlp.add_pipe("entity_ruler")
    patterns = [{"label": "INGREDIENT", "pattern": t} for t in phrases]
    ruler.add_patterns(patterns)
    return ruler


def docs_from_text_plus_lexicon(df: pd.DataFrame, text_col: str, lexicon_terms: list[str]) -> List[Doc]:
    """Bootstrap labels from raw text using an EntityRuler over a lexicon."""
    nlp = spacy.blank("en")
    if not lexicon_terms:
        raise ValueError("No lexicon terms provided; cannot bootstrap from raw text.")
    build_entity_ruler(nlp, lexicon_terms)
    out: List[Doc] = []
    for text in tqdm(df[text_col].astype(str).tolist(), desc="Bootstrapping with EntityRuler"):
        d = nlp.make_doc(text)
        d = nlp(d)  # apply ruler
        # Keep only INGREDIENT, deduplicate spans
        d.ents = spacy.util.filter_spans([e for e in d.ents if e.label_ == "INGREDIENT"])
        out.append(d)
    return out


def build_docs_from_config() -> Tuple[list[Doc], list[Doc], str]:
    """High-level helper: build train & valid docs based on DATA + TRAIN config."""
    # Debug info
    print(f"[DEBUG] ===== Data Loading Configuration =====")
    print(f"[DEBUG] TRAIN_PATH: {DATA.TRAIN_PATH}")
    print(f"[DEBUG] Absolute path: {DATA.TRAIN_PATH.resolve()}")
    print(f"[DEBUG] File exists: {DATA.TRAIN_PATH.exists()}")
    print(f"[DEBUG] Data format: {'Parquet' if DATA.DATA_IS_PARQUET else 'CSV'}")
    print(f"[DEBUG] NER_LIST_COL: {DATA.NER_LIST_COL}")
    print(f"[DEBUG] TEXT_COL: {DATA.TEXT_COL}")
    print(f"[DEBUG] LEXICON_JSON: {DATA.LEXICON_JSON}")
    print(f"[DEBUG] ======================================")

    # Decide source mode
    if DATA.DATA_IS_PARQUET:
        df_sample = pd.read_parquet(DATA.TRAIN_PATH).head(1)
    else:
        df_sample = pd.read_csv(DATA.TRAIN_PATH, nrows=0, dtype=str)  # Read header only

    print(f"[DEBUG] Sample columns: {list(df_sample.columns)}")

    if DATA.NER_LIST_COL and DATA.NER_LIST_COL in df_sample.columns:
        print(f"[DEBUG] Using list-column mode on '{DATA.NER_LIST_COL}'")
        print(f"[DEBUG] This teaches the model to handle real-world messy input.")
        df_list = load_data(DATA.TRAIN_PATH, DATA.DATA_IS_PARQUET, DATA.NER_LIST_COL, max_rows=DATA.MAX_ROWS)
        docs_all = docs_from_list_column(df_list, DATA.NER_LIST_COL)
        source_mode = "list-column"
        
        # Apply max_train_docs limit if set (for debug mode)
        if hasattr(TRAIN, 'MAX_TRAIN_DOCS') and TRAIN.MAX_TRAIN_DOCS is not None:
            if len(docs_all) > TRAIN.MAX_TRAIN_DOCS:
                print(f"[DEBUG] Limiting training docs to {TRAIN.MAX_TRAIN_DOCS} for debug mode")
                docs_all = docs_all[:TRAIN.MAX_TRAIN_DOCS]
    elif DATA.TEXT_COL and DATA.LEXICON_JSON:
        print(f"[DEBUG] Using text+lexicon mode with TEXT_COL='{DATA.TEXT_COL}'")
        df_text = load_data(DATA.TRAIN_PATH, DATA.DATA_IS_PARQUET, DATA.TEXT_COL)
        lex = load_lexicon(DATA.LEXICON_JSON)
        docs_all = docs_from_text_plus_lexicon(df_text, DATA.TEXT_COL, lex)
        source_mode = "text+lexicon"
    else:
        raise RuntimeError(
            "No valid data source inferred. Set DATA.NER_LIST_COL (list-like labels) "
            "or DATA.TEXT_COL + DATA.LEXICON_JSON (bootstrapping)."
        )

    print(f"Docs prepared: {len(docs_all):,} | Source mode: {source_mode}")
    print("Total labeled entities:", sum(len(d.ents) for d in docs_all))

    # Optionally cap docs for local testing, if you want:
    # MAX_DOCS = 500
    # docs_all = docs_all[:MAX_DOCS]

    print(f"[DEBUG] Using only first {len(docs_all)} docs for training")

    train_docs, valid_docs = train_test_split(
        docs_all, test_size=TRAIN.VALID_FRACTION, random_state=TRAIN.RANDOM_SEED
    )
    print(f"train={len(train_docs):,} | valid={len(valid_docs):,}")

    return train_docs, valid_docs, source_mode


from pathlib import Path


def write_docbins(docs: list[Doc], out_dir: Path, shard_size: int) -> None:
    """Write docs into sharded DocBin files."""

    def clean_dir(path: Path):
        path.mkdir(parents=True, exist_ok=True)
        for p in path.glob("*.spacy"):
            print(f"[CLEAN] Removing old shard: {p}")
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
    print(f"Wrote {n} docs to {out_dir} in {shards} shard(s).")


def prepare_docbins_from_config() -> None:
    """End-to-end: build docs and write train/valid DocBins according to config."""
    train_docs, valid_docs, _ = build_docs_from_config()
    write_docbins(train_docs, OUT.TRAIN_DIR, shard_size=TRAIN.SHARD_SIZE)
    write_docbins(valid_docs, OUT.VALID_DIR, shard_size=TRAIN.SHARD_SIZE)


__all__ = [
    "docs_from_list_column",
    "load_lexicon",
    "build_entity_ruler",
    "docs_from_text_plus_lexicon",
    "build_docs_from_config",
    "write_docbins",
    "prepare_docbins_from_config",
]
