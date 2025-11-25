from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Set, Union, Sequence

import time
import re
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import spacy
from spacy.tokens import Token, Doc
import logging

DROP_ADJ: Set[str] = {
    "fresh","ripe","frozen","thawed","organic","unsalted","salted","sweetened","unsweetened",
    "minced","chopped","diced","sliced","crushed","ground","grated","shredded","powdered",
    "large","small","medium","extra","virgin","golden","dark","light","skinless","boneless",
}
UNITS: Set[str] = {"cup","cups","tbsp","tablespoon","tablespoons","tsp","teaspoon","teaspoons",
                   "g","kg","oz","ounce","ounces","ml","l","lb","lbs","pound","pounds"}
ALNUM = re.compile(r"[a-z0-9]+")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Regex patterns for bronze layer cleaning
PARENTHESES_PATTERN = re.compile(r'\([^)]*\)')
COMMERCIAL_SYMBOLS_PATTERN = re.compile(r'[®™©]')
BRAND_ARTIFACTS_PATTERN = re.compile(r'\b(Inc\.|LLC|Ltd\.|Corp\.|Corporation|Company|Co\.)\b', re.IGNORECASE)
MULTISPACE_PATTERN = re.compile(r'\s+')


@dataclass
class SpacyIngredientNormalizer:
    model: str = "en_core_web_sm"
    batch_size: int = 128
    n_process: int = 1  # Multiprocessing: 0=auto, 1=single-threaded, >1=multiprocess (may not work on Windows)

    def __post_init__(self):
        # Disable components we don't need for ingredient normalization
        # We only need tokenizer + parser for dependency parsing
        self.nlp = spacy.load(
            self.model,
            disable=["ner", "textcat", "lemmatizer", "senter", "attribute_ruler"],
        )
        # Optimize for speed: disable unnecessary pipeline components
        # We only need tokenizer and parser for dependency analysis
        Token.set_extension("keep", default=True, force=True)
    
    @staticmethod
    def clean_raw_text(text: str) -> str:
        """
        Bronze layer cleaner: Remove artifacts from raw ingredient text.
        
        Removes:
        - Text inside parentheses (e.g., "Spinach (frozen)" -> "Spinach")
        - Commercial symbols: ®, ™, ©
        - Brand artifacts: Inc., LLC, Ltd., Corp., etc.
        - Collapses multiple spaces to one
        
        Args:
            text: Raw ingredient text
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return text if text else ""
        
        # Remove text inside parentheses
        cleaned = PARENTHESES_PATTERN.sub('', text)
        
        # Remove commercial symbols
        cleaned = COMMERCIAL_SYMBOLS_PATTERN.sub('', cleaned)
        
        # Remove brand artifacts
        cleaned = BRAND_ARTIFACTS_PATTERN.sub('', cleaned)
        
        # Collapse multiple spaces to one
        cleaned = MULTISPACE_PATTERN.sub(' ', cleaned)
        
        return cleaned.strip()

    def _basic_tokens(self, s: str) -> List[str]:
        return ALNUM.findall(s.lower())

    def _is_unit_or_number(self, t: str) -> bool:
        return t.isdigit() or t in UNITS

    def _normalize_doc(self, doc: Doc, raw: str) -> Optional[str]:
        """Core normalize logic, reusing an existing Doc."""
        # Apply bronze layer cleaning first
        raw = self.clean_raw_text(raw)
        s = raw.strip().lower()
        if not s:
            return None

        head: Optional[Token] = None
        for token in doc:
            if token.dep_ == "ROOT":
                head = token
                break
        if head is None:
            return None

        compounds = [t for t in head.lefts if t.dep_ in ("compound",)]
        parts: List[str] = []
        for t in compounds + [head]:
            tok = t.text.lower()
            if tok in DROP_ADJ:
                continue
            if self._is_unit_or_number(tok):
                continue
            if not ALNUM.fullmatch(tok):
                continue
            parts.append(tok)

        # Special cases like "powder", "sauce", etc.
        if len(parts) == 1 and parts[0] in {"powder","sauce","paste","oil","vinegar","cheese"}:
            for child in head.lefts:
                if child.dep_ in ("amod","compound") and ALNUM.fullmatch(child.text.lower()):
                    mod = child.text.lower()
                    if mod not in DROP_ADJ:
                        parts.insert(0, mod)
                        break

        if not parts:
            toks = [
                t for t in self._basic_tokens(s)
                if t not in DROP_ADJ and not self._is_unit_or_number(t)
            ]
            if not toks:
                return None
            parts = toks[-2:] if len(toks) >= 2 else toks

        return " ".join(parts).strip()

    def _normalize_phrase(self, s: str) -> Optional[str]:
        """Single-phrase wrapper; still used by normalize_list."""
        # Apply bronze layer cleaning first
        s = self.clean_raw_text(s)
        s = s.strip()
        if not s:
            return None
        doc = self.nlp(s.lower())
        return self._normalize_doc(doc, s)

    def normalize_list(self, lst: Union[List[str], np.ndarray, tuple]) -> List[str]:
        """Existing API: normalize a single list of ingredient strings."""
        if not isinstance(lst, (list, tuple, np.ndarray)):
            return []
        out: List[str] = []
        seen: Set[str] = set()
        for x in lst:
            norm = self._normalize_phrase(str(x))
            if norm and norm not in seen:
                out.append(norm)
                seen.add(norm)
        return out

    def normalize_batch(
        self,
        lists: Sequence[Union[List[str], np.ndarray, tuple]],
    ) -> List[List[str]]:
        """
        New batched API: normalize many ingredient lists at once using nlp.pipe.
        Returns a list of normalized lists, same length as `lists`.
        """
        # Flatten all phrases and remember boundaries
        flat_phrases: List[str] = []
        boundaries: List[tuple[int, int]] = []

        for lst in lists:
            if not isinstance(lst, (list, tuple, np.ndarray)):
                # mimic normalize_list: non-list -> empty output
                boundaries.append((len(flat_phrases), len(flat_phrases)))
                continue
            start = len(flat_phrases)
            for x in lst:
                # Apply bronze layer cleaning before processing
                cleaned = SpacyIngredientNormalizer.clean_raw_text(str(x))
                flat_phrases.append(cleaned)
            end = len(flat_phrases)
            boundaries.append((start, end))

        if not flat_phrases:
            # no work to do
            return [[] for _ in lists]

        # Run spaCy once over all phrases
        logger.info(f"[spacy_norm] normalize_batch: {len(flat_phrases)} phrases total (batch_size={self.batch_size}, n_process={self.n_process})")
        # Pre-lowercase all texts for better performance
        texts = [p.lower() for p in flat_phrases]
        
        # Process with spaCy pipe - use list() to materialize all docs at once
        # This is more efficient than iterating one-by-one
        # n_process: 0 or None = auto-detect, 1 = single-threaded, >1 = multiprocess
        n_proc = self.n_process if self.n_process > 0 else 1  # Default to 1 for safety
        docs = list(self.nlp.pipe(
            texts,
            batch_size=self.batch_size,
            n_process=n_proc,
        ))

        # Compute normalized forms for each phrase (parallelized if n_process > 1)
        flat_norms: List[Optional[str]] = []
        for raw, doc in zip(flat_phrases, docs):
            flat_norms.append(self._normalize_doc(doc, raw))

        # Rebuild per-list outputs with deduping
        out_lists: List[List[str]] = []
        for start, end in boundaries:
            seen: Set[str] = set()
            cur: List[str] = []
            for i in range(start, end):
                norm = flat_norms[i]
                if norm and norm not in seen:
                    cur.append(norm)
                    seen.add(norm)
            out_lists.append(cur)

        return out_lists


def apply_spacy_normalizer_to_parquet(
    in_parquet: Union[str, Path],
    out_parquet: Union[str, Path],
    list_col: str = "NER",
    out_col: str = "NER_clean",
    compression: str = "zstd",
    spacy_model: str = "en_core_web_sm",
    batch_size: int = 512,  # Increased batch size for better throughput
    n_process: int = 0,  # 0=auto-detect (uses all CPUs), 1=single-threaded
) -> Path:
    """
    Apply spaCy normalization to parquet file.
    
    Args:
        n_process: Number of processes (0=auto, 1=single-threaded, >1=multiprocess)
                  Note: Multiprocessing may not work on Windows due to spawn method.
    """
    normalizer = SpacyIngredientNormalizer(spacy_model, batch_size=batch_size, n_process=n_process)
    pf = pq.ParquetFile(str(in_parquet))
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    schema = pa.schema([pa.field(out_col, pa.list_(pa.string()))])

    total_row_groups = pf.num_row_groups
    for rg in range(total_row_groups):
        t0 = time.time()
        logger.info(f"[spacy_norm] RG{rg+1}/{total_row_groups} start")

        # Only read the column we need
        tbl = pf.read_row_group(rg, columns=[list_col])
        df = tbl.to_pandas()
        n_rows = len(df)
        logger.info(f"[spacy_norm] RG{rg+1}/{total_row_groups}: {n_rows:,} rows")

        if list_col not in df.columns:
            df[list_col] = [[] for _ in range(n_rows)]

        # Ensure we pass list-like objects (or [] for bad types) into normalize_batch
        # Use list comprehension for better performance
        lists = [
            list(x) if isinstance(x, (list, tuple, np.ndarray)) else []
            for x in df[list_col]
        ]

        t1 = time.time()
        logger.info(f"[spacy_norm] RG{rg+1}/{total_row_groups}: running normalizer.normalize_batch…")
        cleaned_lists = normalizer.normalize_batch(lists)
        elapsed = time.time() - t1
        logger.info(f"[spacy_norm] RG{rg+1}/{total_row_groups}: normalizer step took {elapsed:.1f}s ({n_rows/elapsed:.0f} rows/sec)")

        df[out_col] = cleaned_lists

        table = pa.Table.from_pandas(df[[out_col]], preserve_index=False).replace_schema_metadata(None)
        if table.schema != schema:
            table = table.cast(schema, safe=False)

        if writer is None:
            writer = pq.ParquetWriter(str(out_parquet), schema, compression=compression)
        writer.write_table(table)

        logger.info(f"[spacy_norm] RG{rg}: total row group time {time.time() - t0:.1f}s")

    if writer:
        writer.close()
        
    return out_parquet
