from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PA = True
except Exception:
    _HAS_PA = False

import spacy
from tqdm import tqdm

from .config import DATA, OUT
from .utils import load_data, normalize_token
from .normalization import apply_dedupe, load_jsonl_map, load_encoder_maps


def _unique_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _extract_ingredient_rows(doc, dedupe: Optional[dict] = None, tok2id: Optional[dict] = None):
    """Return a list of per-entity dicts with offsets + normalized/canonical forms."""
    rows: List[Dict] = []
    for ent in doc.ents:
        if ent.label_ != "INGREDIENT":
            continue
        raw = ent.text
        norm = normalize_token(raw)
        canon = apply_dedupe(norm, dedupe)
        tok_id = tok2id.get(canon, 0) if tok2id else None
        rows.append(
            {
                "raw": raw,
                "start": int(ent.start_char),
                "end": int(ent.end_char),
                "label": ent.label_,
                "norm": norm,
                "canonical": canon,
                "id": int(tok_id) if tok_id is not None else None,
            }
        )
    return rows


def predict_normalize_encode_structured(
    nlp_dir: Path,
    data_path: Path,
    is_parquet: bool,
    text_col: str,
    dedupe: Optional[dict] = None,
    tok2id: Optional[dict] = None,
    out_path: Optional[Path] = None,
    batch_size: int = 256,
    # sampling knobs (use exactly one)
    sample_n: Optional[int] = None,
    sample_frac: Optional[float] = None,
    head_n: Optional[int] = None,
    start: int = 0,
    stop: Optional[int] = None,
    sample_seed: int = 42,
    # performance
    n_process: int = 1,  # keep 1 for GPU/transformers
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_wide: one row per input, columns=[text_col, NER_raw, NER_clean, Ingredients?, spans_json]
      df_tall: one row per extracted entity with offsets and normalized/canonical forms
    If out_path is set, writes two parquet files: <stem>_wide.parquet and <stem>_tall.parquet.
    """
    if not nlp_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {nlp_dir}")

    nlp = spacy.load(nlp_dir)

    df_in = load_data(data_path, is_parquet, text_col)

    # Apply ONE sampling strategy
    if head_n is not None:
        df_in = df_in.head(head_n)
    elif sample_n is not None:
        df_in = df_in.sample(n=min(sample_n, len(df_in)), random_state=sample_seed)
    elif sample_frac is not None:
        df_in = df_in.sample(
            frac=min(max(sample_frac, 0.0), 1.0), random_state=sample_seed
        )
    elif start != 0 or stop is not None:
        df_in = df_in.iloc[start:stop]

    texts = df_in[text_col].astype(str).tolist()

    wide_rows: List[Dict] = []
    tall_records: List[Dict] = []

    for i, doc in enumerate(
        tqdm(
            nlp.pipe(texts, batch_size=batch_size, n_process=n_process),
            total=len(texts),
            desc="Infer (structured)",
        )
    ):
        rows = _extract_ingredient_rows(doc, dedupe=dedupe, tok2id=tok2id)

        raw_list = _unique_preserve_order([r["raw"] for r in rows])
        clean_list = _unique_preserve_order(
            [r["canonical"] for r in rows if r["canonical"]]
        )
        id_list = [r["id"] for r in rows if r["id"] is not None] if tok2id else None

        # wide entry (compact)
        wide_entry: Dict = {
            text_col: texts[i],
            "NER_raw": raw_list,
            "NER_clean": clean_list,
            "spans_json": json.dumps(rows, ensure_ascii=False),
        }
        if tok2id:
            wide_entry["Ingredients"] = id_list
        wide_rows.append(wide_entry)

        # tall entries (one row per entity, great for QA/exploration)
        for r in rows:
            tall_records.append(
                {
                    "row_id": i,
                    text_col: texts[i],
                    "ent_text": r["raw"],
                    "start": r["start"],
                    "end": r["end"],
                    "label": r["label"],
                    "norm": r["norm"],
                    "canonical": r["canonical"],
                    "id": r["id"],
                }
            )

    df_wide = pd.DataFrame(wide_rows)
    df_tall = pd.DataFrame(tall_records)

    if out_path is not None:
        if not _HAS_PA:
            raise RuntimeError("pyarrow is required to write Parquet files.")
        base = Path(out_path)
        wide_path = base.with_name(base.stem + "_wide.parquet")
        tall_path = base.with_name(base.stem + "_tall.parquet")
        pq.write_table(
            pa.Table.from_pandas(df_wide, preserve_index=False).replace_schema_metadata(None),
            wide_path,
        )
        pq.write_table(
            pa.Table.from_pandas(df_tall, preserve_index=False).replace_schema_metadata(None),
            tall_path,
        )
        print(f"Wrote → {wide_path.name} and {tall_path.name} in {wide_path.parent}")

    return df_wide, df_tall


def load_dedupe_and_maps_from_config() -> Tuple[Optional[dict], Optional[dict]]:
    """
    Load dedupe map and token→ID mapping from config paths.
    
    Returns:
        Tuple of (dedupe_dict, tok2id_dict). Either can be None if files don't exist.
        - dedupe_dict: Maps normalized variant phrases → canonical forms
        - tok2id_dict: Maps canonical tokens → integer IDs
    """
    dedupe = None
    tok2id = None
    
    # Load dedupe map (JSONL format)
    if DATA.DEDUPE_JSONL and DATA.DEDUPE_JSONL.exists():
        dedupe = load_jsonl_map(DATA.DEDUPE_JSONL)
        print(f"Loaded dedupe map: {len(dedupe)} mappings from {DATA.DEDUPE_JSONL}")
    else:
        print(f"Dedupe map not found at {DATA.DEDUPE_JSONL} (skipping deduplication)")
    
    # Load encoder maps (token ↔ ID)
    if DATA.ING_TOK2ID_JSON and DATA.ING_TOK2ID_JSON.exists():
        _, tok2id = load_encoder_maps(DATA.ING_ID2TOK_JSON, DATA.ING_TOK2ID_JSON)
        if tok2id:
            print(f"Loaded token→ID map: {len(tok2id)} tokens from {DATA.ING_TOK2ID_JSON}")
        else:
            print(f"Token→ID map not found or empty at {DATA.ING_TOK2ID_JSON}")
    else:
        print(f"Token→ID map not found at {DATA.ING_TOK2ID_JSON} (skipping ID encoding)")
    
    return dedupe, tok2id


def run_full_inference_from_config(
    text_col: str,
    out_base: Path,
    data_path: Optional[Path] = None,
    sample_n: Optional[int] = None,
    sample_frac: Optional[float] = None,
    head_n: Optional[int] = None,
    batch_size: int = 256,
    n_process: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    High-level helper: run inference using config paths and settings.
    
    Args:
        text_col: Column name containing raw ingredient text
        out_base: Base path for output files (will write <base>_wide.parquet and <base>_tall.parquet)
        data_path: Optional override for input data (defaults to DATA.TRAIN_PATH)
        sample_n: Optional number of rows to sample
        sample_frac: Optional fraction of rows to sample
        head_n: Optional number of rows from head
        batch_size: Batch size for spaCy processing
        n_process: Number of processes (keep 1 for GPU/transformers, >1 may not work on Windows)
    
    Returns:
        Tuple of (df_wide, df_tall) DataFrames
    """
    # Determine input path
    if data_path is None:
        data_path = DATA.TRAIN_PATH
    if not data_path.exists():
        raise FileNotFoundError(f"Input data not found: {data_path}")
    
    # Determine if parquet or CSV
    is_parquet = DATA.DATA_IS_PARQUET if hasattr(DATA, 'DATA_IS_PARQUET') else (data_path.suffix.lower() == ".parquet")
    
    # Load dedupe and token→ID maps from config
    dedupe, tok2id = load_dedupe_and_maps_from_config()
    
    # Run inference
    return predict_normalize_encode_structured(
        nlp_dir=OUT.MODEL_DIR,
        data_path=data_path,
        is_parquet=is_parquet,
        text_col=text_col,
        dedupe=dedupe,
        tok2id=tok2id,
        out_path=out_base,
        batch_size=batch_size,
        sample_n=sample_n,
        sample_frac=sample_frac,
        head_n=head_n,
        n_process=n_process,
    )


__all__ = ["predict_normalize_encode_structured", "load_dedupe_and_maps_from_config", "run_full_inference_from_config"]
