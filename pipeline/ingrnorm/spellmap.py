import ast, os, threading, json, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm
from rapidfuzz import process, fuzz
from spellchecker import SpellChecker


_WORD_RE = re.compile(r"[a-z']+")

def _tok(s: str):
    return _WORD_RE.findall(str(s).lower())

def build_spell_map(
    canon_phrases,
    csv_path,
    ner_col="NER",
    out_path=Path("../data/ner_spell_map.jsonl"),
    chunksize=200_000,
    batch_size=2000,
    max_workers=None,
    score_cutoff=92,
    near_perfect=96,
):
    if not canon_phrases:
        raise ValueError("build_spell_map: canon_phrases is empty. Did you run build_vocab()?")

    buckets = {}
    for p in canon_phrases:
        toks = str(p).split()
        if toks:
            buckets.setdefault(toks[0], []).append(p)
    canon_set = set(canon_phrases)

    seen = set()
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, dtype=str):
        col = chunk.get(ner_col)
        if col is None:
            continue
        for entry in col:
            items = []
            if entry is None:
                pass
            else:
                s = str(entry).strip()
                if s:
                    try:
                        parsed = ast.literal_eval(s)
                        if isinstance(parsed, (list, tuple)):
                            items = [str(x) for x in parsed if str(x).strip()]
                    except Exception:
                        items = [x.strip() for x in s.split(",") if x.strip()]
            for it in items:
                if it: seen.add(str(it))
        del chunk

    items = sorted(seen)

    _tls = threading.local()
    vocab_tokens = [t for ph in canon_phrases for t in str(ph).split()]

    def _get_spell():
        sc = getattr(_tls, "sc", None)
        if sc is None:
            sc = SpellChecker(distance=2)
            if vocab_tokens:
                sc.word_frequency.load_words(vocab_tokens)
            _tls.sc = sc
        return sc

    def _fix_batch(batch):
        sc = _get_spell()
        out = []
        for raw in batch:
            raw_str = str(raw)
            toks = _tok(raw_str)
            if not toks:
                out.append((raw_str, raw_str)); continue
            toks2 = [sc.correction(t) or t for t in toks]
            if not toks2:
                out.append((raw_str, raw_str)); continue

            corrected = " ".join(toks2)
            if corrected in canon_set:
                out.append((raw_str, corrected)); continue

            choices = buckets.get(toks2[0], canon_phrases) if toks2 else canon_phrases
            match = process.extractOne(corrected, choices, scorer=fuzz.WRatio, score_cutoff=score_cutoff)
            if match:
                cand = match[0]
                score = match[1] if len(match) > 1 else 100
                len_ok = (len(cand.split()) == len(toks2))
                out.append((raw_str, cand if (len_ok or score >= near_perfect) else corrected))
            else:
                out.append((raw_str, corrected))
        return out

    def _chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    if max_workers is None:
        max_workers = min(16, 2 * (os.cpu_count() or 2))

    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    batches = list(_chunks(items, batch_size))
    with ThreadPoolExecutor(max_workers=max_workers) as ex, open(out_path, "w", encoding="utf-8") as out:
        for res in tqdm(ex.map(_fix_batch, batches), total=len(batches), desc="Spell/Fuzzy map"):
            for raw, fixed in res:
                out.write(json.dumps({"raw": raw, "fixed": fixed}) + "\n")

    return out_path
