
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

    # Centroid-based clustering: avoid union-find chaining
    # Sort phrases by frequency (descending) to process most common first
    phrase_indices = list(range(len(phrases)))
    phrase_indices.sort(key=lambda i: (-freqs[i], len(phrases[i]), phrases[i]))
    
    clusters: Dict[int, List[int]] = {}  # cluster_id -> list of phrase indices
    cluster_centroids: Dict[int, int] = {}  # cluster_id -> centroid phrase index
    cluster_vectors: Dict[int, np.ndarray] = {}  # cluster_id -> centroid vector
    next_cluster_id = 0
    
    # Verification threshold: members must be within this distance of centroid
    centroid_verification_threshold = threshold * 0.95  # Slightly stricter than merge threshold
    
    for i in phrase_indices:
        assigned = False
        phrase_i = phrases[i]
        vec_i = X[i]
        
        # Check if phrase_i can join an existing cluster
        for cluster_id, centroid_idx in cluster_centroids.items():
            centroid_vec = cluster_vectors[cluster_id]
            centroid_phrase = phrases[centroid_idx]
            
            # Check similarity to centroid
            cos_sim = float(np.dot(vec_i, centroid_vec))
            if cos_sim < threshold:
                continue
            
            # Check lexical requirements
            if len(phrase_i) < min_len or len(centroid_phrase) < min_len:
                continue
            if require_token_overlap and not (_tokset(phrase_i) & _tokset(centroid_phrase)):
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

    mapping: Dict[str, str] = {}
    for cluster_id, members in clusters.items():
        # Use centroid as canonical form (already selected as most representative)
        centroid_idx = cluster_centroids[cluster_id]
        canon = phrases[centroid_idx]
        
        # Block generic as canon if there are alternatives
        if block_generic_as_canon and canon in GENERIC and len(members) > 1:
            # Find highest frequency non-generic member
            members_sorted = sorted(members, key=lambda m: (-freqs[m], len(phrases[m]), phrases[m]))
            alt = next((phrases[m] for m in members_sorted if phrases[m] not in GENERIC), canon)
            canon = alt
        
        # Map all members to canonical form
        for m in members:
            mapping[phrases[m]] = canon

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for src, tgt in mapping.items():
            if src and tgt and src != tgt:
                f.write(json.dumps({"from": src, "to": tgt}) + "\n")
    return mapping