import ast, gc, json, math, re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class StatsNormalizer:
    """
    PMI-based n-gram normalizer with:
      - Streaming counts (uni/bi/tri/4-grams)
      - PMI + child-share + entropy gates (canonical vocab)
      - Greedy 4→3→2→1 segmentation + span-preserving fuzzy snap
      - CSV→Parquet writer (list<string> NER_clean)
      - Save/load learned vocabulary
    """

    @staticmethod
    def _tok(s):
        return re.findall(r"[a-z']+", str(s).lower())

    @staticmethod
    def _ngrams(tokens, n):
        for i in range(len(tokens) - n + 1):
            yield tuple(tokens[i:i+n])

    @staticmethod
    def _parse_ner_entry(entry):
        if entry is None or (isinstance(entry, float) and pd.isna(entry)):
            return []
        s = str(entry).strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
        return [x.strip() for x in s.split(",") if x.strip()]

    def __init__(
        self,
        max_ngram=4,
        min_unigram=50, min_bigram=50, min_trigram=30, min_fourgram=20,
        pmi_bigram=3.0, pmi_trigram=2.0, pmi_fourgram=2.0,
        min_child_share=0.12, max_right_entropy=1.0,
        min_child_share4=0.05, max_right_entropy3=1.3,
        pmi_bigram_fallback=2.6,  min_bigram_fallback=20,
        pmi_trigram_fallback=2.2, min_trigram_fallback=12,
        min_child_share_fallback=0.06, max_right_entropy_fallback=1.4,
        pmi_fourgram_fallback=1.8, min_fourgram_fallback=10,
        min_child_share4_fallback=0.04, max_right_entropy3_fallback=1.5,
        snap_score_cutoff=92, snap_near_perfect=96
    ):
        self.max_ngram = int(max_ngram)
        self.min_unigram, self.min_bigram, self.min_trigram, self.min_fourgram = (
            min_unigram, min_bigram, min_trigram, min_fourgram
        )
        self.pmi_bigram, self.pmi_trigram, self.pmi_fourgram = (
            pmi_bigram, pmi_trigram, pmi_fourgram
        )
        self.min_child_share, self.max_right_entropy = min_child_share, max_right_entropy
        self.min_child_share4, self.max_right_entropy3 = min_child_share4, max_right_entropy3

        self.pmi_bigram_fallback, self.min_bigram_fallback = pmi_bigram_fallback, min_bigram_fallback
        self.pmi_trigram_fallback, self.min_trigram_fallback = pmi_trigram_fallback, min_trigram_fallback
        self.min_child_share_fallback, self.max_right_entropy_fallback = (
            min_child_share_fallback, max_right_entropy_fallback
        )
        self.pmi_fourgram_fallback, self.min_fourgram_fallback = pmi_fourgram_fallback, min_fourgram_fallback
        self.min_child_share4_fallback, self.max_right_entropy3_fallback = (
            min_child_share4_fallback, max_right_entropy3_fallback
        )

        self.snap_score_cutoff, self.snap_near_perfect = snap_score_cutoff, snap_near_perfect

        self.token_total = 0
        self.c1, self.c2, self.c3, self.c4 = Counter(), Counter(), Counter(), Counter()
        self._followers  = defaultdict(Counter)   # (a,b)->c
        self._followers3 = defaultdict(Counter)  # (a,b,c)->d

        self.canon = set()
        self._canon_ready = False
        self._canon_phrases = None
        self._canon_buckets = None

    # Ingest
    def ingest_df(self, df, ner_col="NER"):
        for entry in df[ner_col]:
            for item in self._parse_ner_entry(entry):
                toks = self._tok(item)
                if not toks:
                    continue
                self.c1.update(toks)
                self.token_total += len(toks)

                if self.max_ngram >= 2 and len(toks) >= 2:
                    self.c2.update(self._ngrams(toks, 2))
                if self.max_ngram >= 3 and len(toks) >= 3:
                    for i in range(len(toks) - 2):
                        a, b, c = toks[i], toks[i+1], toks[i+2]
                        self.c3[(a, b, c)] += 1
                        self._followers[(a, b)][c] += 1
                if self.max_ngram >= 4 and len(toks) >= 4:
                    for i in range(len(toks) - 3):
                        a, b, c, d = toks[i], toks[i+1], toks[i+2], toks[i+3]
                        self.c4[(a, b, c, d)] += 1
                        self._followers3[(a, b, c)][d] += 1

    def ingest_csv(self, csv_path, ner_col="NER", chunksize=200_000):
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, dtype=str):
            self.ingest_df(chunk, ner_col=ner_col)
            del chunk; gc.collect()

    # Stats
    def _right_entropy(self, ab):
        foll = self._followers.get(ab)
        if not foll:
            return 0.0
        tot = sum(foll.values()); 
        if tot == 0: return 0.0
        H = 0.0
        for v in foll.values():
            p = v / tot
            H -= p * math.log(p + 1e-12)
        return H

    def _right_entropy3(self, abc):
        foll = self._followers3.get(abc)
        if not foll:
            return 0.0
        tot = sum(foll.values()); 
        if tot == 0: return 0.0
        H = 0.0
        for v in foll.values():
            p = v / tot
            H -= p * math.log(p + 1e-12)
        return H

    def _child_share(self, abc):
        cabc = self.c3.get(abc, 0)
        cab  = self.c2.get(abc[:2], 0)
        return (cabc / cab) if cab else 0.0

    def _child_share4(self, abcd):
        cabcd = self.c4.get(abcd, 0)
        cabc  = self.c3.get(abcd[:3], 0)
        return (cabcd / cabc) if cabc else 0.0

    def _pmi_bigram(self, ab):
        a, b = ab
        cab = self.c2.get(ab, 0)
        if cab == 0 or self.token_total == 0: return -1e9
        pa = self.c1.get(a, 0) / self.token_total
        pb = self.c1.get(b, 0) / self.token_total
        pab = cab / self.token_total
        return math.log((pab / (pa * pb)) + 1e-12)

    def _pmi_trigram(self, abc):
        a, b, c = abc
        return (self._pmi_bigram((a, b)) + self._pmi_bigram((b, c))) / 2.0

    def _pmi_fourgram(self, abcd):
        a, b, c, d = abcd
        return (self._pmi_bigram((a, b)) + self._pmi_bigram((b, c)) + self._pmi_bigram((c, d))) / 3.0

    #    build canon ----------
    def build_vocab(self):
        self.canon.clear()
        for w, c in self.c1.items():
            if c >= self.min_unigram:
                self.canon.add((w,))
        for ab, c in self.c2.items():
            if c >= self.min_bigram and self._pmi_bigram(ab) >= self.pmi_bigram:
                self.canon.add(ab)
        for abc, c in self.c3.items():
            if c < self.min_trigram: continue
            if self._pmi_trigram(abc) < self.pmi_trigram: continue
            if self._child_share(abc) < self.min_child_share: continue
            if self._right_entropy(abc[:2]) > self.max_right_entropy: continue
            self.canon.add(abc)
        for abcd, c in self.c4.items():
            if c < self.min_fourgram: continue
            if self._pmi_fourgram(abcd) < self.pmi_fourgram: continue
            if self._child_share4(abcd) < self.min_child_share4: continue
            if self._right_entropy3(abcd[:3]) > self.max_right_entropy3: continue
            self.canon.add(abcd)
        self._canon_ready = True
        self._canon_phrases = None
        self._canon_buckets = None

    # Snap helpers
    def _canon_bucket_init(self):
        self._canon_phrases = [" ".join(p) for p in self.canon]
        buckets = {}
        for ph in self._canon_phrases:
            ft = ph.split()[0] if ph else ""
            buckets.setdefault(ft, []).append(ph)
        self._canon_buckets = buckets

    def _snap_span(self, tokens, i, n):
        if i + n > len(tokens):
            return None
        if self._canon_phrases is None or self._canon_buckets is None:
            self._canon_bucket_init()
        try:
            from rapidfuzz import process, fuzz
        except Exception:
            return None
        span = " ".join(tokens[i:i+n])
        bucket = self._canon_buckets.get(tokens[i], self._canon_phrases)
        match = process.extractOne(span, bucket, scorer=fuzz.WRatio, score_cutoff=self.snap_score_cutoff)
        if not match:
            return None
        cand, score = match[0], match[1]
        if len(cand.split()) == n or score >= self.snap_near_perfect:
            return cand.split(), n
        return None

    # Segmentation
    def _longest_match(self, toks, i):
        if not self._canon_ready:
            raise RuntimeError("build_vocab() first")
        if self.max_ngram >= 4 and i+3 < len(toks) and tuple(toks[i:i+4]) in self.canon:
            return tuple(toks[i:i+4]), 4
        if self.max_ngram >= 3 and i+2 < len(toks) and tuple(toks[i:i+3]) in self.canon:
            return tuple(toks[i:i+3]), 3
        if self.max_ngram >= 2 and i+1 < len(toks) and tuple(toks[i:i+2]) in self.canon:
            return tuple(toks[i:i+2]), 2
        if (toks[i],) in self.canon:
            return (toks[i],), 1
        if self.max_ngram >= 3:
            snapped = self._snap_span(toks, i, 3)
            if snapped: return tuple(snapped[0]), snapped[1]
        if self.max_ngram >= 2:
            snapped = self._snap_span(toks, i, 2)
            if snapped: return tuple(snapped[0]), snapped[1]
        if self.max_ngram >= 4 and i+3 < len(toks):
            abcd = (toks[i], toks[i+1], toks[i+2], toks[i+3])
            cabcd = self.c4.get(abcd, 0)
            if cabcd >= self.min_fourgram_fallback:
                if (self._pmi_fourgram(abcd) >= self.pmi_fourgram_fallback and
                    self._child_share4(abcd)  >= self.min_child_share4_fallback and
                    self._right_entropy3(abcd[:3]) <= self.max_right_entropy3_fallback):
                    return abcd, 4
        if self.max_ngram >= 3 and i+2 < len(toks):
            abc = (toks[i], toks[i+1], toks[i+2])
            cabc = self.c3.get(abc, 0)
            if cabc >= self.min_trigram_fallback:
                if (self._pmi_trigram(abc) >= self.pmi_trigram_fallback and
                    self._child_share(abc)  >= self.min_child_share_fallback and
                    self._right_entropy(abc[:2]) <= self.max_right_entropy_fallback):
                    return abc, 3
        if self.max_ngram >= 2 and i+1 < len(toks):
            ab = (toks[i], toks[i+1])
            cab = self.c2.get(ab, 0)
            if cab >= self.min_bigram_fallback and self._pmi_bigram(ab) >= self.pmi_bigram_fallback:
                return ab, 2
        return (toks[i],), 1

    def segment_item(self, text):
        t = self._tok(text)
        out, i = [], 0
        while i < len(t):
            phrase, k = self._longest_match(t, i)
            out.append(" ".join(phrase)); i += k
        seen, clean = set(), []
        for x in out:
            if x not in seen:
                clean.append(x); seen.add(x)
        pruned = []
        for x in clean:
            if pruned and x == pruned[-1].split()[-1]:
                continue
            pruned.append(x)
        return pruned

    # DataFrame & IO
    def transform_df(self, df, ner_col="NER", out_col="NER_clean", dedupe_row=False):
        results = []
        for v in df[ner_col]:
            segs = [seg for item in self._parse_ner_entry(v) for seg in self.segment_item(item)]
            if dedupe_row:
                seen, uniq = set(), []
                for s in segs:
                    if s not in seen:
                        uniq.append(s); seen.add(s)
                results.append(uniq)
            else:
                results.append(segs)
        df[out_col] = results
        return df

    @staticmethod
    def _sanitize_for_arrow(df, list_col="NER_clean"):
        df = df.copy()
        def _to_list_of_str(x):
            if isinstance(x, (list, tuple, np.ndarray)):
                return [str(y) for y in x]
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return []
            try:
                parsed = ast.literal_eval(str(x))
                if isinstance(parsed, (list, tuple, np.ndarray)):
                    return [str(y) for y in parsed]
            except Exception:
                pass
            return [str(x)]
        if list_col in df.columns:
            df[list_col] = df[list_col].apply(_to_list_of_str)
        for col in df.columns:
            if col == list_col: continue
            s = df[col]
            if s.dtype == object:
                def _to_scalar_str(v):
                    if isinstance(v, (list, tuple, dict, set, np.ndarray)):
                        return json.dumps(list(v) if isinstance(v, np.ndarray) else v, ensure_ascii=False)
                    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
                df[col] = s.map(_to_scalar_str)
        return df

    def transform_csv_to_parquet(self, csv_path, out_path, ner_col="NER", chunksize=200_000):
        writer = None
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, dtype=str):
            chunk = self.transform_df(chunk, ner_col=ner_col, out_col="NER_clean")
            chunk = self._sanitize_for_arrow(chunk, list_col="NER_clean")
            table = pa.Table.from_pandas(chunk, preserve_index=False).replace_schema_metadata(None)
            fields = []
            for f in table.schema:
                if f.name == "NER_clean" and not pa.types.is_list(f.type):
                    fields.append(pa.field("NER_clean", pa.list_(pa.string())))
                else:
                    fields.append(f)
            target_schema = pa.schema(fields)
            try:
                table = table.cast(target_schema, safe=False)
            except Exception:
                arrays = [pa.array(arr, type=pa.list_(pa.string())) for arr in table.column("NER_clean").to_pylist()]
                table = table.set_column(table.schema.get_field_index("NER_clean"), "NER_clean", pa.chunked_array(arrays))
            if writer is None:
                writer = pq.ParquetWriter(out_path, target_schema, compression="zstd")
            writer.write_table(table)
            del chunk, table; gc.collect()
        if writer is not None:
            writer.close()

    # Persistence
    def save_vocab(self, path):
        data = {"token_total": int(self.token_total),
                "canon": [" ".join(p) for p in sorted(self.canon)]}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_vocab(cls, path):
        data = json.load(open(path, "r", encoding="utf-8"))
        obj = cls()
        obj.canon = set(tuple(p.split()) for p in data["canon"])
        obj._canon_ready = True
        obj._canon_phrases = None
        obj._canon_buckets = None
        return obj
