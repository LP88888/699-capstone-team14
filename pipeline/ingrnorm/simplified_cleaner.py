# pipeline/ingrnorm/simplified_cleaner.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Dict, Set, Optional, Tuple
import json
import re

try:
    from symspellpy import SymSpell, Verbosity
except Exception:
    SymSpell = None
try:
    from rapidfuzz import process, fuzz
except Exception:
    process = None
    fuzz = None


# 1) Build / load ingredient lexicon

def load_lexicon(
    sources: List[Path | str],
    min_len: int = 2,
    to_lower: bool = True
) -> Set[str]:
    """
    Load ingredient-like strings from JSON/CSV/TXT files into a set.
    Accepts:
      - JSON (id->token maps or list[str])
      - TXT (one per line)
      - CSV (single column; simple)
    """
    import pandas as pd
    L: Set[str] = set()
    for src in sources:
        p = Path(src)
        if not p.exists():
            continue
        if p.suffix.lower() == ".json":
            obj = json.load(open(p, "r", encoding="utf-8"))
            if isinstance(obj, dict):
                vals = [str(v) for v in obj.values()]
            elif isinstance(obj, list):
                vals = [str(x) for x in obj]
            else:
                vals = []
            for v in vals:
                s = v.strip().lower() if to_lower else v.strip()
                if len(s) >= min_len:
                    L.add(s)
        elif p.suffix.lower() in {".txt"}:
            for line in open(p, "r", encoding="utf-8"):
                s = line.strip().lower() if to_lower else line.strip()
                if len(s) >= min_len:
                    L.add(s)
        elif p.suffix.lower() in {".csv"}:
            df = pd.read_csv(p)
            # use the first column by default
            col = df.columns[0]
            for v in df[col].astype(str):
                s = v.strip().lower() if to_lower else v.strip()
                if len(s) >= min_len:
                    L.add(s)
    return L


# 2) SymSpell for spelling toward the lexicon


def build_symspell_from_lexicon(lexicon: Iterable[str], max_edit_distance: int = 2) -> Optional[SymSpell]:
    """
    Build a SymSpell index from your lexicon so misspellings correct toward valid ingredients.
    """
    if SymSpell is None:
        return None
    sym = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=7)
    # SymSpell expects frequency; we can give a flat weight (1)
    for term in lexicon:
        # Add multi-word terms token-wise so single-token corrections work,
        # AND add the phrase itself for phrase lookup_compound() to leverage.
        sym.create_dictionary_entry(term, 1)
        for tok in term.split():
            sym.create_dictionary_entry(tok, 1)
    return sym

def symspell_fix(sym: Optional[SymSpell], text: str) -> str:
    """
    Correct a token or short phrase with SymSpell; no-op if SymSpell not available.
    """
    if not sym or not text:
        return text
    # If it's multi-word, try compound lookup; otherwise simple lookup
    if " " in text:
        suggestions = sym.lookup_compound(text, max_edit_distance=sym._max_dictionary_edit_distance)
        return suggestions[0].term if suggestions else text
    else:
        suggestions = sym.lookup(text, Verbosity.TOP, max_edit_distance=sym._max_dictionary_edit_distance)
        return suggestions[0].term if suggestions else text


# 3) Lightweight filters (units/adjectives)


_UNITS = {
    "cup","cups","tbsp","tablespoon","tablespoons","tsp","teaspoon","teaspoons",
    "lb","lbs","pound","pounds","oz","ounce","ounces","g","kg","ml","l","litre","liter"
}
_ADJECTIVES_MISC = {
    "ripe","fresh","frozen","chopped","diced","minced","cooked","crushed","sliced",
    "halved","pitted","boneless","skinless","reduced","lowfat","low-fat","nonfat",
    "large","small","extra","additional","original","homemade","smoked","roasted",
    "decorative","prepared","oriental","mexican","solid","round","square","young"
}
_NUMERIC_RE = re.compile(r"^[+\-]?\d+([.,]\d+)?$")

def looks_like_noise(tok: str) -> bool:
    t = tok.strip().lower()
    if not t:
        return True
    if t in _UNITS or t in _ADJECTIVES_MISC:
        return True
    if _NUMERIC_RE.match(t):
        return True
    if len(t) == 1:
        return True
    return False


# 4) Optional semantic backstop (MiniLM)


class SemanticBackstop:
    """Optional: map OOV terms to nearest lexicon term by cosine similarity."""
    def __init__(self, lexicon: Iterable[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.lex = sorted(set(lexicon))
        self.model = SentenceTransformer(model_name)
        self.emb = self.model.encode(self.lex, convert_to_tensor=True, show_progress_bar=False)

    def map(self, term: str, threshold: float = 0.84) -> Optional[str]:
        if not term:
            return None
        from sentence_transformers import util
        q = self.model.encode([term], convert_to_tensor=True, show_progress_bar=False)
        sim = util.cos_sim(q, self.emb)[0]
        idx = int(sim.argmax())
        score = float(sim[idx])
        if score >= threshold:
            return self.lex[idx]
        return None


# 5) One function to clean a token/phrase


def clean_token_or_phrase(
    phrase: str,
    lexicon: Set[str],
    symspell: Optional[SymSpell] = None,
    semantic: Optional[SemanticBackstop] = None,
    drop_if_unmapped: bool = True
) -> Optional[str]:
    """
    - phrase-level spell correction toward lexicon (SymSpell)
    - split into words, drop pure noise tokens, correct per-token, rejoin
    - if final phrase not in lexicon: optionally map semantically; else drop/keep raw
    """
    if not phrase:
        return None
    p = phrase.strip().lower()

    # quick noise drop
    if p in _ADJECTIVES_MISC or p in _UNITS:
        return None

    # spell toward lexicon (phrase-aware)
    p = symspell_fix(symspell, p)

    # token-level cleanup
    toks = [w for w in re.findall(r"[a-z]+", p)]
    toks = [w for w in toks if not looks_like_noise(w)]
    if not toks:
        return None
    toks = [symspell_fix(symspell, w) for w in toks]
    p2 = " ".join(toks).strip()
    if not p2:
        return None

    # if exact in lexicon, done
    if p2 in lexicon:
        return p2

    # semantic backstop
    if semantic is not None:
        mapped = semantic.map(p2)
        if mapped:
            return mapped

    # fallback behavior
    return None if drop_if_unmapped else p2
