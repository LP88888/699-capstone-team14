
from __future__ import annotations
import json
from collections import defaultdict
import re
from pathlib import Path
from typing import Dict, Union
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


_DROP_TOKENS = {
    # Prep words / measurements that add noise
    "til",  # sesame synonym; drop noisy token
    "extra",
    "chopped",
    "finely",
    "tablespoon",
    "tablespoons",
    "teaspoon",
    "teaspoons",
    "tbsp",
    "tsp",
    "spoon",
    "spoons",
    "to",
    "cut",
    "pinch",
    "ounce",
    "ounces",
    "pound",
    "pounds",
    "toasted",
    "whole",
    "all",
    "sauce",
    "piece",
    "water",
    "cut",
    "peeled",
    "water water",
    "salt",
    "for",
    "minced",
    "grated",
    "slices",
    "baking",
    "cup",
    "cups",
    "thinly",
    "sifted",
    "melted",
    "stick",
    "28-ounce",
    "couple",
    "1 (",
    "halved",
    "sifted",
    "plus",
    "2"
    "inch piece",
    "seeded",
    "turns",
    "28-ounce",
    "sugar",
    "wrappers ",
    "purpose",
    "all",
    "degrees",
    "one",
    "two",
    "see"
    "disclaimer",
    "note",
    "ingredient",
    "ingredient note",
    "network",
    "eyeball",
    "split",
    "legs",
    "thighs",
    "food",
    "inspirations",
    "hellman",
    "kitchen",
    "pods"
    

}
_DROP_SUBSTRINGS = {"spoon", "baking", "ounce", "pound", "cup", "slice",}
_PHRASE_NOISE_TOKENS = {
    # Quantities / units
    "slice", "slices", "sliced",
    "chunk", "chunks", "chunky", "chunked",
    "piece", "pieces",
    "cup", "cups",
    "tablespoon", "tablespoons", "tbsp",
    "teaspoon", "teaspoons", "tsp",
    "lb", "lbs", "pound", "pounds",
    "oz", "ounce", "ounces",
    "gram", "grams", "kg", "ml", "l", "liter", "liters",
    "pinch",
    # Forms / descriptors that are not the core ingredient
    "powder", "powders", "powdered",
}
_DIGIT_RE = re.compile(r"^[0-9]+([./][0-9]+)?$")
_UNIGRAM_CACHE: Dict[int, Dict[str, int]] = {}


def _dedupe_preserve_order(tokens):
    seen = set()
    deduped = []
    for tok in tokens:
        key = str(tok)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(tok)
    return deduped


def _collapse_duplicate_words(text: str) -> str:
    parts = str(text).split()
    if len(parts) > 1 and len(set(parts)) == 1:
        return parts[0]
    return text


def _build_unigram_frequency(mapping: Dict[str, str]) -> Dict[str, int]:
    """
    Build a unigram frequency map from mapping targets; helps choose the dominant
    token when stripping noisy descriptors (e.g., 'chicken slices' -> 'chicken').
    """
    freq: Dict[str, int] = defaultdict(int)
    for val in mapping.values():
        for tok in str(val).lower().split():
            freq[tok] += 1
    return dict(freq)


def _get_unigram_freq(mapping: Dict[str, str], fallback: Dict[str, int] | None = None) -> Dict[str, int]:
    if not mapping:
        return fallback or {}
    key = id(mapping)
    if key not in _UNIGRAM_CACHE:
        _UNIGRAM_CACHE[key] = _build_unigram_frequency(mapping)
    return _UNIGRAM_CACHE[key]


def _strip_noise_tokens(parts: list[str], unigram_freq: Dict[str, int]) -> list[str]:
    """
    Remove measurement / non-food tokens from within a phrase and prefer the
    most frequent remaining tokens.
    """
    kept: list[str] = []
    for tok in parts:
        lower = tok.lower()
        if lower in _PHRASE_NOISE_TOKENS:
            continue
        if _DIGIT_RE.match(lower):
            continue
        kept.append(tok)

    if not kept and parts:
        kept = [parts[0]]

    # Collapse consecutive duplicates case-insensitively
    collapsed: list[str] = []
    for tok in kept:
        if collapsed and collapsed[-1].lower() == tok.lower():
            continue
        collapsed.append(tok)

    if len(collapsed) > 1 and unigram_freq:
        max_freq = max(unigram_freq.get(tok.lower(), 0) for tok in collapsed)
        dominant = [tok for tok in collapsed if unigram_freq.get(tok.lower(), 0) == max_freq]
        collapsed = dominant or collapsed

    return collapsed


def normalize_token_with_map(
    tok,
    mapping: Dict[str, str],
    *,
    collapse_duplicate_words: bool = False,
    canonicalizer=None,
    unigram_freq: Dict[str, int] | None = None,
) -> str | None:
    """
    Apply a dedupe map to a token and drop obviously noisy tokens.

    - Applies an optional canonicalizer (e.g., plural → singular tweaks)
    - Looks up the token in the provided mapping (variant → canonical)
    - Optionally collapses duplicate single-word strings like "pepper pepper"
    - Filters out known noisy tokens/substrings and strips measurement/descriptive words
    """
    raw = canonicalizer(str(tok), mapping) if canonicalizer else str(tok)
    freq_map = unigram_freq or _get_unigram_freq(mapping)

    # Prefer curated map directly if present
    mapped_direct = mapping.get(raw)
    raw = mapped_direct if mapped_direct is not None else raw

    # Collapse immediate duplicate first/second tokens (e.g., "salt salt" -> "salt")
    parts = raw.split()
    if len(parts) >= 2 and parts[0] == parts[1]:
        parts = [parts[0]]
        raw = parts[0]

    # Strip obvious non-food words (units, counts, form descriptors)
    if len(parts) > 1:
        parts = _strip_noise_tokens(parts, freq_map)
        if not parts:
            return None
        raw = " ".join(parts)

    # Drop overly long phrases that are likely junk (e.g., long prep notes)
    if len(parts) > 4:
        return None

    mapped = mapping.get(raw, raw)
    if collapse_duplicate_words:
        mapped = _collapse_duplicate_words(mapped)

    mapped = str(mapped).strip()
    lower = mapped.lower()
    if not mapped:
        return None
    if lower in _DROP_TOKENS:
        return None
    if any(substr in lower for substr in _DROP_SUBSTRINGS):
        return None
    return mapped


def load_jsonl_map(path: Union[str, Path]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return mapping
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            raw_from = str(obj.get("from", "")).strip()
            raw_to = str(obj.get("to", "")).strip()
            if not raw_from or not raw_to:
                continue
            # store both original and lowercase key for flexible lookups
            mapping[raw_from] = raw_to
            mapping[raw_from.lower()] = raw_to
    return mapping

def write_jsonl_map(mapping: Dict[str, str], out_path: Union[str, Path]) -> Path:
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    data = "\n".join(json.dumps({"from": k, "to": v}) for k, v in mapping.items())
    # Write directly - file comparison is inefficient for large files
    # If needed, use --force flag in scripts to rebuild
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(data + "\n")
    return out_path

def apply_map_to_parquet_streaming(
    in_path: Union[str, Path],
    out_path: Union[str, Path],
    mapping: Union[Dict[str, str], str, Path],
    list_col: str = "NER_clean",
    compression: str = "zstd",
    dedupe_tokens: bool = False,
    collapse_duplicate_words: bool = False,
    canonicalizer=None,
    max_tokens_per_record: int | None = 120,
) -> None:
    if not isinstance(mapping, dict):
        mapping = load_jsonl_map(mapping)
    unigram_freq = _get_unigram_freq(mapping)

    pf = pq.ParquetFile(str(in_path))
    writer = None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg).to_pandas()
        if list_col in df.columns:
            cleaned_col = []
            for lst in df[list_col]:
                if not (isinstance(lst, (list, tuple, np.ndarray)) and len(lst) > 0):
                    cleaned_col.append([] if not isinstance(lst, (list, tuple)) else list(lst))
                    continue
                tokens_in = list(lst) if not isinstance(lst, np.ndarray) else lst.tolist()
                if max_tokens_per_record and len(tokens_in) > max_tokens_per_record:
                    tokens_in = tokens_in[:max_tokens_per_record]
                mapped_tokens = []
                for tok in tokens_in:
                    mapped = normalize_token_with_map(
                        tok,
                        mapping,
                        collapse_duplicate_words=collapse_duplicate_words,
                        canonicalizer=canonicalizer,
                        unigram_freq=unigram_freq,
                    )
                    if mapped:
                        mapped_tokens.append(mapped)
                if dedupe_tokens:
                    mapped_tokens = _dedupe_preserve_order(mapped_tokens)
                cleaned_col.append(mapped_tokens)
            df[list_col] = cleaned_col
        table = pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata(None)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), table.schema, compression=compression)
        writer.write_table(table)
    if writer is not None:
        writer.close()
