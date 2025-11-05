import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import pyarrow as pa
import pyarrow.parquet as pq


def cosine_dedupe(vocab_counter, threshold=0.88, topk=20, out_path=Path("../data/cosine_dedupe_map.jsonl"),
                  model_name="all-MiniLM-L6-v2", device=None):
    phrases = [p for p, c in vocab_counter.items() if c > 0]
    if not phrases:
        raise ValueError("cosine_dedupe: empty vocabulary.")

    model = SentenceTransformer(model_name, device=device)
    embs = model.encode([p.lower().strip() for p in phrases], convert_to_tensor=True, show_progress_bar=True)
    embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-12)

    sims = util.cos_sim(embs, embs).cpu().numpy()
    idx2phrase = phrases
    phrase2idx = {p: i for i, p in enumerate(phrases)}

    # simple union-find by threshold
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

    n = len(phrases)
    for i in range(n):
        # only consider topk neighbors (excluding itself)
        row = sims[i]
        neighbors = np.argsort(-row)[1:topk+1]
        for j in neighbors:
            if row[j] >= threshold:
                union(i, j)

    # pick representative = smallest index in each set
    rep = {}
    for i in range(n):
        r = find(i)
        rep.setdefault(r, []).append(i)

    mapping = {}
    for r, group in rep.items():
        canon = idx2phrase[min(group)]
        for g in group:
            mapping[idx2phrase[g]] = canon

    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for k, v in mapping.items():
            f.write(json.dumps({"from": k, "to": v}) + "\n")

    return mapping


def apply_cosine_map_to_parquet(in_path, out_path, mapping: Dict[str, str], list_col="NER_clean"):
    pf = pq.ParquetFile(in_path)
    writer = None
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg).to_pandas()
        if list_col in df.columns:
            df[list_col] = [
                [mapping.get(tok, tok) for tok in lst] if isinstance(lst, (list, tuple)) else lst
                for lst in df[list_col]
            ]
        table = pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata(None)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
        writer.write_table(table)
    if writer is not None:
        writer.close()
