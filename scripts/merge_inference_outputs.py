#!/usr/bin/env python3
"""CLI helper that merges ingredient NER inference results back into the source dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

INFERRED_COL = "inferred_ingredients"
ENCODED_COL = "encoded_ingredients"


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False, compression="zstd")
        return
    df.to_csv(path, index=False)


def merge_inference(
    source_path: Path,
    wide_path: Path,
    output_path: Path,
    *,
    inferred_col: str = INFERRED_COL,
    encoded_col: str = ENCODED_COL,
) -> tuple[int, Optional[int]]:
    df_source = read_table(source_path)
    df_wide = pd.read_parquet(wide_path)

    if "NER_clean" not in df_wide.columns:
        raise KeyError("NER_clean column missing from inference output; cannot attach results")

    if len(df_source) == 0:
        raise ValueError("Source dataset is empty; nothing to merge")
    if len(df_wide) == 0:
        raise ValueError("Inference output is empty; nothing to merge")

    rows_to_assign = min(len(df_source), len(df_wide))
    if rows_to_assign < len(df_source):
        logging.warning(
            "Source dataset has %s rows but inference output only has %s. Truncating assignment.",
            len(df_source),
            len(df_wide),
        )

    df_source = df_source.copy()
    df_source[inferred_col] = pd.Series([None] * len(df_source), dtype=object)
    inferred_series = pd.Series(df_wide["NER_clean"].tolist()[:rows_to_assign], dtype=object)
    df_source.loc[: rows_to_assign - 1, inferred_col] = inferred_series.values

    encoded_lists = None
    if "Ingredients" in df_wide.columns:
        encoded_lists = df_wide["Ingredients"].tolist()
        df_source[encoded_col] = pd.Series([None] * len(df_source), dtype=object)
        encoded_series = pd.Series(encoded_lists[:rows_to_assign], dtype=object)
        df_source.loc[: rows_to_assign - 1, encoded_col] = encoded_series.values

    write_table(df_source, output_path)
    return rows_to_assign, len(encoded_lists) if encoded_lists is not None else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge inference outputs into the source dataset")
    parser.add_argument("source", type=Path, help="Original dataset that was fed into inference (CSV or Parquet)")
    parser.add_argument("wide", type=Path, help="Wide Parquet produced by inference (<out_base>_wide.parquet)")
    parser.add_argument(
        "output",
        type=Path,
        help="Destination for merged dataset (e.g., ../data/combined_raw_datasets_with_inference.parquet)",
    )
    parser.add_argument(
        "--inferred-col",
        default=INFERRED_COL,
        help=f"Column name to store normalized ingredients (default: {INFERRED_COL})",
    )
    parser.add_argument(
        "--encoded-col",
        default=ENCODED_COL,
        help=f"Column name to store ingredient IDs if present (default: {ENCODED_COL})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="[%(levelname)s] %(message)s")

    try:
        rows, encoded_rows = merge_inference(
            args.source,
            args.wide,
            args.output,
            inferred_col=args.inferred_col,
            encoded_col=args.encoded_col,
        )
        logging.info("Merged inference outputs into %s (rows written: %s)", args.output, rows)
        if encoded_rows is not None:
            logging.info("Ingredient ID lists also written (%s rows)", encoded_rows)
        else:
            logging.info("Wide file lacked 'Ingredients' column, so encoded IDs were skipped")
        return 0
    except Exception as exc:  # pragma: no cover - CLI convenience
        logging.error("Merge failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
