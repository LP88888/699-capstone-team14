from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set, Union

import json
import re
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# pip install spacy==3.* && python -m spacy download en_core_web_sm
import spacy
from spacy.tokens import Token

# words we can usually drop without changing identity
DROP_ADJ: Set[str] = {
    "fresh","ripe","frozen","thawed","organic","unsalted","salted","sweetened","unsweetened",
    "minced","chopped","diced","sliced","crushed","ground","grated","shredded","powdered",
    "large","small","medium","extra","virgin","golden","dark","light","skinless","boneless",
}

UNITS: Set[str] = {"cup","cups","tbsp","tablespoon","tablespoons","tsp","teaspoon","teaspoons",
                   "g","kg","oz","ounce","ounces","ml","l","lb","lbs","pound","pounds"}

ALNUM = re.compile(r"[a-z0-9]+")

@dataclass
class SpacyIngredientNormalizer:
    model: str = "en_core_web_sm"

    def __post_init__(self):
        self.nlp = spacy.load(self.model, disable=["ner","textcat","lemmatizer","senter"])
        Token.set_extension("keep", default=True, force=True)

    def _basic_tokens(self, s: str) -> List[str]:
        return ALNUM.findall(s.lower())

    def _is_unit_or_number(self, t: str) -> bool:
        return t.isdigit() or t in UNITS

    def _normalize_phrase(self, s: str) -> Optional[str]:
        s = s.strip().lower()
        if not s:
            return None
        doc = self.nlp(s)

        # Find the head noun; keep its left compounds (e.g., 'olive' in 'olive oil')
        head: Optional[Token] = None
        for token in doc:
            if token.dep_ == "ROOT":
                head = token
                break
        if head is None:
            return None

        # Build candidate from compounds + head; drop adjectives if in DROP_ADJ
        compounds = [t for t in head.lefts if t.dep_ in ("compound",)]
        parts = []
        for t in compounds + [head]:
            # keep only alnum tokens
            tok = t.text.lower()
            if tok in DROP_ADJ:
                continue
            if self._is_unit_or_number(tok):
                continue
            if not ALNUM.fullmatch(tok):
                continue
            parts.append(tok)

        # If head alone is too generic (e.g., 'powder'), consider one right modifier if it's a noun (e.g., 'baking' + 'powder')
        if len(parts) == 1 and parts[0] in {"powder","sauce","paste","oil","vinegar","cheese"}:
            for child in head.lefts:
                if child.dep_ in ("amod","compound") and ALNUM.fullmatch(child.text.lower()):
                    mod = child.text.lower()
                    if mod not in DROP_ADJ:
                        parts.insert(0, mod)
                        break

        if not parts:
            # fallback: simple token filter
            toks = [t for t in self._basic_tokens(s) if t not in DROP_ADJ and not self._is_unit_or_number(t)]
            if not toks:
                return None
            parts = toks[-2:] if len(toks) >= 2 else toks

        return " ".join(parts).strip()

    def normalize_list(self, lst: Union[List[str], np.ndarray, tuple]) -> List[str]:
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


def apply_spacy_normalizer_to_parquet(
    in_parquet: Union[str, Path],
    out_parquet: Union[str, Path],
    list_col: str = "NER",
    out_col: str = "NER_clean",
    compression: str = "zstd",
    spacy_model: str = "en_core_web_sm",
) -> Path:
    normalizer = SpacyIngredientNormalizer(spacy_model)
    pf = pq.ParquetFile(str(in_parquet))

    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    target_schema = None
    writer = None

    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=[list_col])
        df = tbl.to_pandas()
        if list_col not in df.columns:
            df[list_col] = [[] for _ in range(len(df))]

        cleaned = df[list_col].apply(normalizer.normalize_list)
        df[out_col] = cleaned

        # make sure out_col is list<string>
        table = pa.Table.from_pandas(df[[out_col]], preserve_index=False).replace_schema_metadata(None)
        flds = [pa.field(out_col, pa.list_(pa.string()))]
        schema = pa.schema(flds)
        if writer is None:
            writer = pq.ParquetWriter(str(out_parquet), schema, compression=compression)
        # cast if needed
        if table.schema != schema:
            table = table.cast(schema, safe=False)
        writer.write_table(table)

    if writer:
        writer.close()
    return out_parquet
