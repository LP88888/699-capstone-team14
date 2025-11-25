import ast
import json
import math
import os
import random
import warnings
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
    _HAS_PA = True
except Exception:
    _HAS_PA = False

import spacy
from typing import Optional
try:
    import torch
except Exception:  # torch is optional
    torch = None
    warnings.warn(
        "Torch not available. Training will fall back to CPU tok2vec "
        "if transformers cannot be used."
    )


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


import logging
import spacy

try:
    import torch
except Exception:
    torch = None

from .config import TRAIN

logger = logging.getLogger("ingredient_ner.training")


def configure_device() -> None:
    """
    Configure device for training.

    - If USE_TOK2VEC_DEBUG is True, we just stay on CPU.
    - If torch.cuda.is_available() is True, the transformer will use GPU via PyTorch.
    - We *do not* call spacy.require_gpu(), so CuPy is no longer a hard requirement.
    """
    if getattr(TRAIN, "USE_TOK2VEC_DEBUG", False):
        logger.info("[device] Debug/tok2vec mode – forcing CPU (tok2vec).")
        return

    if torch is None:
        logger.info("[device] torch not available – using CPU.")
        return

    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0)
        logger.info(f"[device] CUDA available – PyTorch will use GPU: {dev_name}")
        # Important: DON'T call spacy.require_gpu() here.
        # spacy-transformers uses PyTorch directly.
    else:
        logger.info("[device] CUDA not available – using CPU.")

def load_data(path: Path, is_parquet: bool, col: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Load a single column from CSV/Parquet and return a clean DataFrame.
    
    Args:
        path: Path to the data file
        is_parquet: Whether the file is Parquet format
        col: Column name to extract
        max_rows: Optional limit on number of rows to load (None = load all)
    """
    def read_csv_with_fallback(path,dtype=str, nrows=None):
        """
        Tries to read CSV using UTF-8, then cp1252 (windows-1252), then latin-1.
        Returns a DataFrame and the encoding used (string).
        """
        tried = []
        encodings = ["utf-8", "cp1252", "latin-1"]
        for enc in encodings:
            try:
                logger.info(f"Attempting to read CSV with encoding={enc}")
                df = pd.read_csv(path, dtype=dtype, encoding=enc, nrows=nrows)
                logger.info(f"Successfully read CSV with encoding={enc}")
                return df, enc
            except UnicodeDecodeError as e:
                tried.append((enc, str(e)))
                logger.warning(f"Failed with encoding={enc}: {e}")
            except Exception as e:
                # Non-encoding errors should be raised
                logger.exception(f"Unexpected error reading CSV with encoding={enc}: {e}")
                raise

        # Last resort: read with errors='replace' so no exception but some characters may be lost
        logger.warning("All standard encodings failed. Trying utf-8 with errors='replace'.")
        df = pd.read_csv(path, dtype=dtype, encoding="utf-8", engine="python", error_bad_lines=False, warn_bad_lines=True)
        return df, "utf-8 (errors=replace)"

    if is_parquet:
        if not _HAS_PA:
            raise RuntimeError("pyarrow is required to read Parquet files. Please install pyarrow.")
        pf = pq.ParquetFile(str(path))
        frames = []
        total_rows = 0
        for i in range(pf.num_row_groups):
            if max_rows is not None and total_rows >= max_rows:
                break
            frame = pf.read_row_group(i).to_pandas()
            if max_rows is not None:
                remaining = max_rows - total_rows
                if len(frame) > remaining:
                    frame = frame.head(remaining)
                total_rows += len(frame)
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        df, used_encoding = read_csv_with_fallback(path, dtype=str, nrows=max_rows)
        logger.info(f"Loaded CSV using encoding: {used_encoding}")
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in {list(df.columns)[:20]}...")
    return df[[col]].dropna().reset_index(drop=True)


def parse_listlike(v: Any) -> List[str]:
    """Parse values that may be python-lists, JSON-lists, or comma-separated strings."""
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    if not s:
        return []
    for parser in (ast.literal_eval, json.loads):
        try:
            out = parser(s)
            if isinstance(out, (list, tuple)):
                return [str(x).strip() for x in out if str(x).strip()]
        except Exception:
            pass
    return [x.strip() for x in s.split(",") if x.strip()]


def join_with_offsets(tokens: List[str], sep: str = ", "):
    """Join tokens with a separator and track character offsets of each token."""
    text, spans, pos = [], [], 0
    for i, tok in enumerate(tokens):
        start, end = pos, pos + len(tok)
        text.append(tok)
        spans.append((start, end))
        pos = end
        if i < len(tokens) - 1:
            text.append(sep)
            pos += len(sep)
    return "".join(text), spans


def normalize_token(s: str) -> str:
    """Lowercase, trim and collapse whitespace for consistent string keys."""
    return " ".join(str(s).strip().lower().split())
