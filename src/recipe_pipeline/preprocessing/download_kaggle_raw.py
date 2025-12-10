"""
Utility script to fetch the raw Kaggle datasets and persist each to Parquet.

Notes:
- Requires `kagglehub` (already in our deps) and Kaggle API creds configured.
- We do NOT merge datasets here; we simply download and convert the first CSV/Excel
  file found in each dataset to a parquet in data/raw/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import kagglehub  # type: ignore
import pandas as pd


def _read_table(path: Path) -> Optional[pd.DataFrame]:
    """Read CSV/Excel with a couple of encoding fallbacks."""
    if path.suffix.lower() in {".csv"}:
        for enc in ("utf-8", "cp1252"):
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
            except Exception:
                break
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path)
        except Exception:
            pass
    return None


def _get_first_table(dataset: str) -> Optional[pd.DataFrame]:
    """Download a Kaggle dataset and load the first CSV/Excel file."""
    base_path = Path(kagglehub.dataset_download(dataset))
    candidates: list[Path] = []
    for root, _, files in os.walk(base_path):
        for fname in files:
            p = Path(root) / fname
            if p.suffix.lower() in {".csv", ".xlsx", ".xls"}:
                candidates.append(p)
    if not candidates:
        print(f"[WARN] No tabular files found in {dataset}")
        return None
    candidates.sort()
    for path in candidates:
        df = _read_table(path)
        if df is not None:
            print(f"[INFO] Loaded {path} from {dataset} with shape {df.shape}")
            return df
        print(f"[WARN] Failed to load {path}")
    print(f"[WARN] Could not read any table for {dataset}")
    return None


def download_all(targets: Iterable[tuple[str, str]], raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for dataset_id, outfile in targets:
        df = _get_first_table(dataset_id)
        if df is None:
            continue
        out_path = raw_dir / outfile
        df.to_parquet(out_path, index=False)
        print(f"[OK] Wrote {out_path} ({len(df):,} rows)")



from recipe_pipeline.core import StageResult, PipelineContext


def run(ctx: PipelineContext) -> StageResult:
    cfg = ctx.stage("download", required=False)
    if not cfg or not cfg.get("enabled", False):
        return StageResult(name="download", status="skipped", details="Disabled or not configured")

    user = cfg.get("kaggle_username")
    key = cfg.get("kaggle_key")
    if user and key:
        os.environ["KAGGLE_USERNAME"] = str(user)
        os.environ["KAGGLE_KEY"] = str(key)

    datasets_cfg = cfg.get("datasets", [])
    targets: list[tuple[str, str]] = []
    for item in datasets_cfg:
        ds = item.get("id")
        out = item.get("output")
        if ds and out:
            targets.append((ds, out))
    if not targets:
        return StageResult(name="download", status="skipped", details="No datasets configured")

    # Allow overriding raw_dir; default to config.paths.raw_data_dir or ./data/raw
    raw_dir_str = cfg.get("raw_dir") or ctx.config.raw.get("paths", {}).get("raw_data_dir", "data/raw")
    raw_dir = Path(raw_dir_str)

    download_all(targets, raw_dir=raw_dir)
    return StageResult(name="download", status="ok", details=f"Fetched {len(targets)} dataset(s) to {raw_dir}")
