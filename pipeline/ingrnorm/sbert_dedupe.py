
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

GENERIC = {"salt", "water", "oil", "sugar"}
def _tokset(s: str) -> set:
    return set(s.split())

def sbert_dedupe(
    vocab_counter: Dict[str, int],
    out_path: Path | str,
    model_name: str = "all-MiniLM-L6-v2",
    threshold: float = 0.88,
    topk: int = 25,
    min_len: int = 2,
    require_token_overlap: bool = True,
    block_generic_as_canon: bool = True,
) -> Dict[str, str]:
    phrases = [p.strip().lower() for p, c in vocab_counter.items() if c > 0]
    freqs   = np.array([vocab_counter[p] for p in phrases], dtype=np.int64)
    if not phrases:
        raise ValueError("sbert_dedupe: empty vocab")

    model = SentenceTransformer(model_name)
    X = model.encode(phrases, normalize_embeddings=True, show_progress_bar=True)

    nn = NearestNeighbors(n_neighbors=min(topk + 1, len(phrases)), metric="cosine", algorithm="auto")
    nn.fit(X)
    dists, idxs = nn.kneighbors(X, return_distance=True)

    parent = list(range(len(phrases)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(phrases)):
        dist_row, nbrs = dists[i], idxs[i]
        for jpos in range(1, len(nbrs)):
            j = nbrs[jpos]
            cos = 1.0 - float(dist_row[jpos])
            if cos < threshold:
                continue
            a, b = phrases[i], phrases[j]
            if len(a) < min_len or len(b) < min_len:
                continue
            if require_token_overlap and not (_tokset(a) & _tokset(b)):
                continue
            union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(len(phrases)):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    mapping: Dict[str, str] = {}
    for root, members in clusters.items():
        members_sorted = sorted(members, key=lambda m: (-freqs[m], len(phrases[m]), phrases[m]))
        canon = phrases[members_sorted[0]]
        if block_generic_as_canon and canon in GENERIC and len(members_sorted) > 1:
            alt = next((phrases[m] for m in members_sorted if phrases[m] not in GENERIC), canon)
            canon = alt
        for m in members:
            mapping[phrases[m]] = canon

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for src, tgt in mapping.items():
            if src and tgt and src != tgt:
                f.write(json.dumps({"from": src, "to": tgt}) + "\n")
    return mapping
