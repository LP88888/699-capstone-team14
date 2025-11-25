
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from rapidfuzz.distance import Levenshtein

def _iter_parquet_tokenized_docs(parquet_path: Union[str, Path], list_col: str = "NER_clean") -> Iterable[List[str]]:
    pf = pq.ParquetFile(str(parquet_path))
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=[list_col])
        df = tbl.to_pandas()
        if list_col not in df.columns:
            continue
        for lst in df[list_col]:
            items: List[str] = []
            if isinstance(lst, (list, tuple, np.ndarray)):
                items = [str(x) for x in lst if str(x).strip()]
            elif isinstance(lst, str):
                s = lst.strip()
                if s and (s.startswith("[") and s.endswith("]")):
                    try:
                        parsed = json.loads(s)
                        if isinstance(parsed, list):
                            items = [str(x) for x in parsed if str(x).strip()]
                    except Exception:
                        pass
                if not items and s:
                    items = [s]
            if not items:
                continue
            tokens: List[str] = []
            for phrase in items:
                tokens.extend(simple_preprocess(str(phrase), deacc=True, min_len=2))
            if tokens:
                yield tokens

def train_or_load_w2v(
    corpus_parquet: Union[str, Path],
    list_col: str = "NER_clean",
    model_cache_path: Optional[Union[str, Path]] = None,
    vector_size: int = 100, window: int = 5, min_count: int = 1, workers: int = 4, sg: int = 1, epochs: int = 8,
) -> Word2Vec:
    """Train or load Word2Vec model. Uses streaming iterator to avoid loading all sentences into memory."""
    model_cache_path = Path(model_cache_path) if model_cache_path else None
    if model_cache_path and model_cache_path.exists():
        return Word2Vec.load(str(model_cache_path))
    
    # Use streaming iterator instead of loading all into memory
    sentences_iter = _iter_parquet_tokenized_docs(corpus_parquet, list_col=list_col)
    model = Word2Vec(
        sentences=sentences_iter,  # Word2Vec accepts iterables directly
        vector_size=vector_size, 
        window=window, 
        min_count=min_count, 
        workers=workers, 
        sg=sg, 
        epochs=epochs,
    )
    if model_cache_path:
        model_cache_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_cache_path))
    return model

def _tokenize_phrase(phrase: str) -> List[str]:
    return simple_preprocess(str(phrase), deacc=True, min_len=2)

# Stop words to exclude from token overlap check
STOP_WORDS = {"salt", "water", "oil", "sugar", "pepper", "flour", "butter", "garlic", "onion"}

def _tokenize_excluding_stops(phrase: str) -> set:
    """Tokenize phrase and return set of tokens excluding stop words."""
    tokens = _tokenize_phrase(phrase)
    return {t for t in tokens if t not in STOP_WORDS}

def _has_token_overlap(phrase_a: str, phrase_b: str) -> bool:
    """Check if two phrases share at least one non-stop-word token."""
    tokens_a = _tokenize_excluding_stops(phrase_a)
    tokens_b = _tokenize_excluding_stops(phrase_b)
    return bool(tokens_a & tokens_b)

def _levenshtein_ratio(phrase_a: str, phrase_b: str) -> float:
    """Compute normalized Levenshtein similarity ratio (0-1, higher = more similar)."""
    if not phrase_a or not phrase_b:
        return 0.0
    # Use rapidfuzz for normalized similarity (0-1 scale)
    distance = Levenshtein.distance(phrase_a.lower(), phrase_b.lower())
    max_len = max(len(phrase_a), len(phrase_b))
    if max_len == 0:
        return 1.0
    return 1.0 - (distance / max_len)

def _should_merge(phrase_a: str, phrase_b: str, levenshtein_threshold: float = 0.7) -> bool:
    """
    Determine if two phrases should be merged based on lexical guardrails.
    
    Merges if:
    - They share at least one non-stop-word token, OR
    - Their Levenshtein similarity is above threshold
    
    Args:
        phrase_a: First phrase
        phrase_b: Second phrase
        levenshtein_threshold: Minimum Levenshtein similarity ratio (0-1)
        
    Returns:
        True if phrases should be merged
    """
    # Check token overlap
    if _has_token_overlap(phrase_a, phrase_b):
        return True
    
    # Check Levenshtein distance
    lev_ratio = _levenshtein_ratio(phrase_a, phrase_b)
    if lev_ratio >= levenshtein_threshold:
        return True
    
    return False

def phrase_vector(model: Word2Vec, phrase: str, oov_policy: str = "skip") -> Optional[np.ndarray]:
    toks = _tokenize_phrase(phrase)
    vecs = [model.wv[t] for t in toks if t in model.wv]
    if not vecs:
        return None if oov_policy != "zeros" else np.zeros(model.vector_size, dtype=np.float32)
    v = np.mean(np.stack(vecs, 0), 0)
    n = np.linalg.norm(v) + 1e-12
    return (v / n).astype(np.float32)

def _topk_similar_indices(M: np.ndarray, k: int = 25, self_exclude: bool = True, chunk_size: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    n, d = M.shape
    k = min(k, n - 1 if self_exclude else n)
    topk_idx = np.empty((n, k), dtype=np.int32)
    topk_sc  = np.empty((n, k), dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        block = M[start:end]
        sims = block @ M.T
        if self_exclude:
            rows = np.arange(start, end)
            sims[np.arange(end - start), rows] = -np.inf
        part_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(sims, part_idx, axis=1)
        order = np.argsort(-part_scores, axis=1)
        block_topk_idx = np.take_along_axis(part_idx, order, axis=1).astype(np.int32)
        block_topk_sc  = np.take_along_axis(part_scores, order, axis=1).astype(np.float32)
        topk_idx[start:end] = block_topk_idx
        topk_sc[start:end]  = block_topk_sc
    return topk_idx, topk_sc

def _uf_build(n: int): return list(range(n))
def _uf_find(p, a): 
    while p[a] != a:
        p[a] = p[p[a]]; a = p[a]
    return a
def _uf_union(p, a, b):
    ra, rb = _uf_find(p, a), _uf_find(p, b)
    if ra != rb: p[rb] = ra

def w2v_dedupe(
    vocab_counter: Dict[str, int],
    *, corpus_parquet: Union[str, Path], list_col: str = "NER_clean",
    model_cache_path: Optional[Union[str, Path]] = None,
    vector_size: int = 100, window: int = 5, min_count: int = 1, workers: int = 4, sg: int = 1, epochs: int = 8,
    threshold: float = 0.85, topk: int = 25, out_path: Union[str, Path] = Path("../data/w2v_dedupe_map.jsonl"),
) -> Dict[str, str]:
    phrases = [p for p, c in vocab_counter.items() if c > 0]
    if not phrases:
        raise ValueError("w2v_dedupe: empty vocabulary.")
    try:
        model = train_or_load_w2v(corpus_parquet=corpus_parquet, list_col=list_col, model_cache_path=model_cache_path,
                                  vector_size=vector_size, window=window, min_count=min_count, workers=workers, sg=sg, epochs=epochs)
    except ValueError:
        from gensim.utils import simple_preprocess
        sentences = [simple_preprocess(p, deacc=True, min_len=2) for p in phrases]
        sentences = [s for s in sentences if s]
        model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=1, workers=workers, sg=sg, epochs=max(epochs, 5))

    vecs, kept = [], []
    for ph in phrases:
        v = phrase_vector(model, ph, oov_policy="skip")
        if v is None: continue
        kept.append(ph); vecs.append(v)
    if not vecs:
        raise ValueError("w2v_dedupe: no phrases had in-vocab tokens.")

    M = np.stack(vecs, 0).astype(np.float32)
    nbr_idx, nbr_sc = _topk_similar_indices(M, k=topk, self_exclude=True, chunk_size=2048)
    
    # Centroid-based clustering: avoid union-find chaining
    # Sort phrases by frequency (descending) to process most common first
    freqs = vocab_counter
    phrase_indices = list(range(len(kept)))
    phrase_indices.sort(key=lambda i: (-freqs.get(kept[i], 0), len(kept[i]), kept[i]))
    
    clusters: Dict[int, List[int]] = {}  # cluster_id -> list of phrase indices
    cluster_centroids: Dict[int, int] = {}  # cluster_id -> centroid phrase index
    cluster_vectors: Dict[int, np.ndarray] = {}  # cluster_id -> centroid vector
    next_cluster_id = 0
    
    # Levenshtein threshold for lexical guardrails (0.7 = 70% similarity)
    levenshtein_threshold = 0.7
    # Verification threshold: members must be within this distance of centroid
    centroid_verification_threshold = threshold * 0.95  # Slightly stricter than merge threshold
    
    for i in phrase_indices:
        assigned = False
        phrase_i = kept[i]
        vec_i = M[i]
        
        # Check if phrase_i can join an existing cluster
        for cluster_id, centroid_idx in cluster_centroids.items():
            centroid_vec = cluster_vectors[cluster_id]
            centroid_phrase = kept[centroid_idx]
            
            # Check similarity to centroid
            cos_sim = float(np.dot(vec_i, centroid_vec))
            if cos_sim < threshold:
                continue
            
            # Lexical guardrail: check token overlap or Levenshtein distance
            if not _should_merge(phrase_i, centroid_phrase, levenshtein_threshold):
                continue
            
            # Verify member is actually close to centroid (not just a neighbor)
            if cos_sim >= centroid_verification_threshold:
                clusters[cluster_id].append(i)
                assigned = True
                break
        
        # If not assigned, create new cluster with this phrase as centroid
        if not assigned:
            cluster_id = next_cluster_id
            next_cluster_id += 1
            clusters[cluster_id] = [i]
            cluster_centroids[cluster_id] = i
            cluster_vectors[cluster_id] = vec_i
    
    # Build mapping: all members map to their cluster centroid
    mapping = {}
    for cluster_id, members in clusters.items():
        centroid_idx = cluster_centroids[cluster_id]
        canon = kept[centroid_idx]
        for i in members:
            mapping[kept[i]] = canon
    
    # Ensure all phrases have a mapping (even if not clustered)
    for ph in phrases:
        mapping.setdefault(ph, ph)

    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for frm, to in mapping.items():
            f.write(json.dumps({"from": frm, "to": to}) + "\n")
    return mapping
