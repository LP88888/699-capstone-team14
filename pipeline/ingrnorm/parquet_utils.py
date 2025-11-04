# pipeline/ingrnorm/parquet_utils.py
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import numpy as np
import pyarrow.parquet as pq

def vocab_from_parquet_listcol(
    path: Union[str, Path],
    col: str = "NER_clean",
    min_freq: int = 1,
) -> Dict[str, int]:
    """
    Stream all row-groups from a Parquet and count tokens found in `col`.

    - Accepts per-row values that are list/tuple/np.ndarray
    - Ignores None/empty/whitespace tokens
    - Returns a dict {token: count} (filtered by min_freq)
    """
    path = Path(path)
    pf = pq.ParquetFile(path)
    cnt = Counter()

    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=[col]).to_pandas()
        if col not in df.columns:
            continue
        for v in df[col]:
            if isinstance(v, (list, tuple, np.ndarray)):
                for t in v:
                    s = str(t).strip()
                    if s:
                        cnt[s] += 1
        del df  # keep memory tidy

    if min_freq > 1:
        cnt = Counter({k: c for k, c in cnt.items() if c >= min_freq})
    return dict(cnt)

