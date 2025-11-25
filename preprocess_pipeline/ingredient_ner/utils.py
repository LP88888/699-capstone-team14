import ast
import json
import logging
import random
import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import spacy

try:
    import pyarrow.parquet as pq
    _HAS_PA = True
except ImportError:
    _HAS_PA = False

try:
    import torch
except ImportError:
    torch = None

# We use a global logger for this module
logger = logging.getLogger(__name__)

def set_global_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

def configure_device(use_gpu: bool = True) -> None:
    """
    Configure device for training/inference.
    
    This version respects the user's config but avoids spacy.require_gpu() 
    if it causes conflicts, relying on PyTorch directly.
    """
    if not use_gpu:
        logger.info("[device] GPU disabled by config. Using CPU.")
        return

    if torch is None:
        logger.warning("[device] Torch not installed. Using CPU.")
        return

    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0)
        logger.info(f"[device] CUDA available. PyTorch will use GPU: {dev_name}")
        # NOTE: We intentionally do NOT call spacy.require_gpu() here 
        # to avoid conflicts with thinc/cupy if they aren't perfectly aligned.
    else:
        logger.info("[device] CUDA not available. Using CPU.")

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

def _read_csv_with_fallback(path: Path, dtype=str, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, str]:
    """
    Internal helper: Tries to read CSV using UTF-8, then cp1252 (Windows), then latin-1.
    Returns (DataFrame, encoding_used).
    """
    encodings = ["utf-8", "cp1252", "latin-1"]
    
    for enc in encodings:
        try:
            # Use default C engine for speed
            df = pd.read_csv(path, dtype=dtype, encoding=enc, nrows=nrows)
            return df, enc
        except UnicodeDecodeError:
            continue # Try next encoding
        except Exception as e:
            logger.warning(f"Error reading with {enc}: {e}")
            raise

    # Last resort: Python engine with 'replace' (lossy but works)
    logger.warning("All standard encodings failed. Trying utf-8 with errors='replace'.")
    df = pd.read_csv(
        path, 
        dtype=dtype, 
        encoding="utf-8", 
        engine="python", 
        on_bad_lines="warn", 
        nrows=nrows
    )
    return df, "utf-8-replace"

def load_data(path: Union[str, Path], is_parquet: bool, col: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Load a single column from CSV/Parquet and return a clean DataFrame.
    
    Handles Windows encoding (0xae / ®) issues automatically.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.DataFrame()

    if is_parquet:
        if not _HAS_PA:
            raise RuntimeError("pyarrow is required to read Parquet files.")
        
        # Parquet reading logic
        if max_rows:
            pf = pq.ParquetFile(str(path))
            rows_read = 0
            frames = []
            for i in range(pf.num_row_groups):
                if rows_read >= max_rows: break
                chunk = pf.read_row_group(i, columns=[col]).to_pandas()
                frames.append(chunk)
                rows_read += len(chunk)
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if len(df) > max_rows:
                df = df.head(max_rows)
        else:
            df = pd.read_parquet(path, columns=[col])
            
    else:
        # CSV Reading with Fallback Logic
        # FIX: We unpack the tuple here so we return ONLY the dataframe
        df, used_encoding = _read_csv_with_fallback(path, dtype=str, nrows=max_rows)
        logger.info(f"Loaded CSV using encoding: {used_encoding}")

    # Column Validation
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found. Available: {list(df.columns)[:5]}...")
    
    # Return clean, non-empty dataframe
    return df[[col]].dropna().reset_index(drop=True)

def parse_listlike(v: Any) -> List[str]:
    """Parse values that may be python-lists, JSON-lists, or comma-separated strings."""
    
    # 1. Handle list-likes (arrays, lists, tuples) FIRST.
    # We do this check first to avoid passing an array to pd.isna(), which causes the ValueError.
    if isinstance(v, (list, tuple, np.ndarray)):
        # Convert elements to strings and filter empty ones
        return [str(x).strip() for x in v if str(x).strip()]
    
    # 2. Handle scalars (strings, floats, None).
    # Now we are safe to use pd.isna() because we know v is not an array.
    if pd.isna(v):
        return []
    
    s = str(v).strip()
    if not s:
        return []
        
    # 3. Try JSON/Literal eval (e.g. "['salt', 'pepper']")
    if s.startswith("[") and s.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                out = parser(s)
                # Ensure the result is actually a list/iterable
                if isinstance(out, (list, tuple, np.ndarray)):
                    return [str(x).strip() for x in out if str(x).strip()]
            except Exception:
                pass
                
    # 4. Fallback to comma separation
    return [x.strip() for x in s.split(",") if x.strip()]

def normalize_token(s: str) -> str:
    """Lowercase, trim and collapse whitespace for consistent string keys."""
    return " ".join(str(s).strip().lower().split())