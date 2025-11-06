# pipeline/ingrnorm/w2v_dedupe.py
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# gensim is the de-facto for Word2Vec/FastText
from gensim.models import Word2Vec  # swap to FastText if desired
from gensim.utils import simple_preprocess


def _iter_parquet_tokenized_docs(
    parquet_path: Union[str, Path],
    list_col: str = "NER_clean",
) -> Iterable[List[str]]:
    """
    Stream tokenized ingredient lists from a Parquet list[str] column.
    - Accepts list, tuple, or np.ndarray per row
    - If a row is a string that looks like a Python/JSON list, attempt to parse
    - Skips None/NaN/empty rows
    """
    pf = pq.ParquetFile(str(parquet_path))
    total_rows, yielded_docs = 0, 0

    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=[list_col])
        df = tbl.to_pandas()
        if list_col not in df.columns:
            continue

        for lst in df[list_col]:
            total_rows += 1
            if lst is None or (isinstance(lst, float) and pd.isna(lst)):
                continue

            items: List[str] = []
            # Accept list-like
            if isinstance(lst, (list, tuple, np.ndarray)):
                items = [str(x) for x in lst if str(x).strip()]
            # Sometimes serialized list in a string
            elif isinstance(lst, str):
                s = lst.strip()
                if s and (s.startswith("[") and s.endswith("]")):
                    try:
                        parsed = json.loads(s)
                        if isinstance(parsed, list):
                            items = [str(x) for x in parsed if str(x).strip()]
                    except Exception:
                        pass
                # fallback: treat whole string as one phrase
                if not items and s:
                    items = [s]

            if not items:
                continue

            tokens: List[str] = []
            for phrase in items:
                tokens.extend(simple_preprocess(str(phrase), deacc=True, min_len=2))
            if tokens:
                yielded_docs += 1
                yield tokens

    if yielded_docs == 0:
        # Help the user understand what's wrong
        raise ValueError(
            f"w2v corpus empty: scanned {total_rows} rows, found 0 tokenized docs in column '{list_col}'. "
            "Check that baseline_parquet exists and that the column is a list[str] (or parseable list)."
        )



def train_or_load_w2v(
    corpus_parquet: Union[str, Path],
    list_col: str = "NER_clean",
    model_cache_path: Optional[Union[str, Path]] = None,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
    workers: int = 4,
    sg: int = 1,
    epochs: int = 8,
) -> Word2Vec:
    """
    Train (or load cached) Word2Vec on tokenized ingredient lists from NER_clean.
    Defaults work well for ingredient corpora; use sg=1 (skip-gram) for better rare-word handling.
    """
    model_cache_path = Path(model_cache_path) if model_cache_path else None
    if model_cache_path and model_cache_path.exists():
        return Word2Vec.load(str(model_cache_path))

    # Two-pass iterator: we can pre-materialize to list to allow multiple epochs, but to keep memory safe
    # we'll stream once to collect into a list of small docs (ingredients lists are short).
    sentences = list(_iter_parquet_tokenized_docs(corpus_parquet, list_col=list_col))
    if not sentences:
        raise ValueError("Word2Vec training corpus is empty. Did you set the correct parquet/list_col?")

    model = Word2Vec(
        sentences=sentences,
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
    # lightweight, consistent with gensim training tokens
    return simple_preprocess(str(phrase), deacc=True, min_len=2)


def phrase_vector(
    model: Word2Vec,
    phrase: str,
    oov_policy: str = "skip",  # "skip" | "avg_known" | "zeros"
) -> Optional[np.ndarray]:
    """
    Compute a dense vector for a phrase by averaging token vectors present in the model's vocab.
    If no token is known:
      - skip  -> return None
      - avg_known -> return None (same as skip; kept for clarity)
      - zeros -> return zero vector
    """
    toks = _tokenize_phrase(phrase)
    vecs = []
    for tok in toks:
        if tok in model.wv:
            vecs.append(model.wv[tok])
    if not vecs:
        if oov_policy == "zeros":
            return np.zeros(model.vector_size, dtype=np.float32)
        return None
    v = np.mean(np.stack(vecs, axis=0), axis=0)
    # L2 normalize for cosine
    n = np.linalg.norm(v) + 1e-12
    return (v / n).astype(np.float32)


def _topk_similar_indices(
    M: np.ndarray,
    k: int = 25,
    self_exclude: bool = True,
    chunk_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute top-k indices and cosine scores for each row in M vs all rows in M using chunked matmul.
    M should be L2-normalized (n x d). Returns (topk_idx, topk_scores) each (n x k).
    """
    n, d = M.shape
    k = min(k, n - 1 if self_exclude else n)

    # allocate outputs
    topk_idx = np.empty((n, k), dtype=np.int32)
    topk_sc  = np.empty((n, k), dtype=np.float32)

    # chunked dot-products: for each block of rows, score vs full matrix
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        block = M[start:end]  # (b x d)
        sims = block @ M.T     # (b x n) cosine (since unit vectors)
        if self_exclude:
            # set diagonal (self) to -inf for positions within this block
            rows = np.arange(start, end)
            sims[np.arange(end - start), rows] = -np.inf

        # argpartition for speed, then sort those k
        part_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(sims, part_idx, axis=1)
        order = np.argsort(-part_scores, axis=1)
        block_topk_idx = np.take_along_axis(part_idx, order, axis=1).astype(np.int32)
        block_topk_sc  = np.take_along_axis(part_scores, order, axis=1).astype(np.float32)

        topk_idx[start:end] = block_topk_idx
        topk_sc[start:end]  = block_topk_sc

    return topk_idx, topk_sc


def _union_find_build(n: int) -> List[int]:
    return list(range(n))


def _find(parent: List[int], a: int) -> int:
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def _union(parent: List[int], a: int, b: int) -> None:
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[rb] = ra


def w2v_dedupe(
    vocab_counter: Dict[str, int],
    *,
    corpus_parquet: Union[str, Path],
    list_col: str = "NER_clean",
    model_cache_path: Optional[Union[str, Path]] = None,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
    workers: int = 4,
    sg: int = 1,
    epochs: int = 8,
    threshold: float = 0.85,
    topk: int = 25,
    out_path: Union[str, Path] = Path("../data/w2v_dedupe_map.jsonl"),
) -> Dict[str, str]:
    """
    Deduplicate semantically similar phrases using Word2Vec-based cosine similarity.

    Steps:
      1) Train/load W2V on NER_clean token sequences from `corpus_parquet`.
      2) Embed each phrase (avg of token vectors) and L2-normalize.
      3) Chunked top-K cosine neighbors, union-find merge with threshold.
      4) Canonical representative per cluster = highest frequency (vocab_counter), then shortest phrase, then lexicographic.
      5) Write JSONL mapping {"from": phrase, "to": canonical}.

    Returns:
      mapping dict (phrase -> canonical)
    """
    phrases = [p for p, c in vocab_counter.items() if c > 0]
    if not phrases:
        raise ValueError("w2v_dedupe: empty vocabulary.")

    try:
        model = train_or_load_w2v(
            corpus_parquet=corpus_parquet,
            list_col=list_col,
            model_cache_path=model_cache_path,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg,
            epochs=epochs,
        )
    except ValueError as e:
        # Fallback: train on phrases if parquet list_col is empty or malformed
        sentences = [simple_preprocess(p, deacc=True, min_len=2) for p in phrases]
        sentences = [s for s in sentences if s]
        if not sentences:
            raise
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=1,
            workers=workers,
            sg=sg,
            epochs=max(epochs, 5),
        )

    # build phrase embeddings (skip phrases with no known tokens)
    vecs: List[np.ndarray] = []
    keep_mask: List[bool]  = []
    for ph in phrases:
        v = phrase_vector(model, ph, oov_policy="skip")
        if v is None:
            keep_mask.append(False)
            continue
        keep_mask.append(True)
        vecs.append(v)

    if not vecs:
        raise ValueError("w2v_dedupe: no phrases had any in-vocab tokens; consider FastText or lower min_count.")

    # filter down to retained phrases/vecs
    kept_phrases = [p for p, m in zip(phrases, keep_mask) if m]
    M = np.stack(vecs, axis=0).astype(np.float32)  # already L2-normalized

    # top-k neighbors per row
    nbr_idx, nbr_sc = _topk_similar_indices(M, k=topk, self_exclude=True, chunk_size=2048)

    # union-find clustering with threshold
    parent = _union_find_build(len(kept_phrases))
    for i in range(len(kept_phrases)):
        for jpos, score in zip(nbr_idx[i], nbr_sc[i]):
            if score >= threshold:
                _union(parent, i, jpos)

    # build groups
    groups: Dict[int, List[int]] = {}
    for i in range(len(kept_phrases)):
        r = _find(parent, i)
        groups.setdefault(r, []).append(i)

    # select canonical per group
    def _canon_idx(candidates: List[int]) -> int:
        # Prefer highest frequency; then shortest phrase; then lexicographic
        def _key(i: int):
            ph = kept_phrases[i]
            freq = vocab_counter.get(ph, 0)
            return (-freq, len(ph), ph)
        best = min(candidates, key=_key)
        return best

    idx2canon: Dict[int, int] = {gidx: _canon_idx(idxs) for gidx, idxs in groups.items()}

    # final mapping
    mapping: Dict[str, str] = {}
    for gidx, idxs in groups.items():
        canon_phrase = kept_phrases[idx2canon[gidx]]
        for i in idxs:
            mapping[kept_phrases[i]] = canon_phrase

    # phrases that had no vectors map to themselves (identity) so downstream apply is stable
    for ph, m in zip(phrases, keep_mask):
        if not m:
            mapping[ph] = ph

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for frm, to in mapping.items():
            f.write(json.dumps({"from": frm, "to": to}) + "\n")

    return mapping


def load_jsonl_map(path: Union[str, Path]) -> Dict[str, str]:
    """Load {"from":..., "to":...} JSONL mapping into a dict."""
    mapping: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            mapping[str(obj["from"])] = str(obj["to"])
    return mapping


def apply_map_to_parquet(
    in_path: Union[str, Path],
    out_path: Union[str, Path],
    mapping: Union[Dict[str, str], str, Path],
    list_col: str = "NER_clean",
    compression: str = "zstd",
) -> None:
    """
    Apply a phrase→canonical map to a Parquet list[str] column.
    Accepts either a dict mapping or a path to a JSONL mapping file (for compatibility with your current call site).
    """
    if not isinstance(mapping, dict):
        mapping = load_jsonl_map(mapping)

    pf = pq.ParquetFile(str(in_path))
    writer = None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg).to_pandas()
        if list_col in df.columns:
            df[list_col] = [
                [mapping.get(tok, tok) for tok in lst] if isinstance(lst, (list, tuple)) else lst
                for lst in df[list_col]
            ]
        table = pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata(None)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), table.schema, compression=compression)
        writer.write_table(table)

    if writer is not None:
        writer.close()
