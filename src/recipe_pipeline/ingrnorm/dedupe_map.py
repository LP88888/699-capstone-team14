
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Union
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


_DROP_TOKENS = {
    # Prep words / measurements that add noise
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
    "wrappers "

}
_DROP_SUBSTRINGS = {"spoon", "baking"}


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
) -> None:
    if not isinstance(mapping, dict):
        mapping = load_jsonl_map(mapping)

    def _normalize_token(tok):
        raw = canonicalizer(str(tok), mapping) if canonicalizer else str(tok)
        mapped = mapping.get(
            raw,
            _collapse_duplicate_words(raw) if collapse_duplicate_words else raw,
        )
        mapped = str(mapped).strip()
        lower = mapped.lower()
        if not mapped:
            return None
        if lower in _DROP_TOKENS:
            return None
        if any(substr in lower for substr in _DROP_SUBSTRINGS):
            return None
        return mapped

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
                mapped_tokens = []
                for tok in tokens_in:
                    mapped = _normalize_token(tok)
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
