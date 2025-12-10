from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple
import time

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
from .utils import load_data, normalize_token, parse_listlike, join_with_offsets
from .normalization import apply_dedupe, load_jsonl_map, load_encoder_maps

logger = logging.getLogger(__name__)

# Import spaCy normalizer for consistent normalization
# This ensures inference uses the same normalization as the training pipeline
try:
    import sys
    from pathlib import Path
    # Add pipeline to path if not already there (for imports when running as script)
    pipeline_path = Path(__file__).parent.parent
    if str(pipeline_path) not in sys.path:
        sys.path.insert(0, str(pipeline_path))
    from ingrnorm.spacy_normalizer import SpacyIngredientNormalizer
    _HAS_SPACY_NORM = True
except (ImportError, ModuleNotFoundError) as e:
    _HAS_SPACY_NORM = False
    SpacyIngredientNormalizer = None
    # Will be handled gracefully in the code


def _unique_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _extract_ingredient_rows(
    doc, 
    dedupe: Optional[dict] = None, 
    tok2id: Optional[dict] = None,
    spacy_normalizer: Optional[Any] = None,
):
    """
    Return a list of per-entity dicts with offsets + normalized/canonical forms.
    
    Args:
        doc: spaCy Doc with entities
        dedupe: Dedupe mapping dict (variant → canonical)
        tok2id: Token to ID mapping dict
        spacy_normalizer: Optional SpacyIngredientNormalizer instance for consistent normalization
    """
    rows: List[Dict] = []
    for ent in doc.ents:
        if ent.label_ != "INGREDIENT":
            continue
        raw = ent.text
        
        # Apply spaCy normalization if available (matches training pipeline)
        # Otherwise fall back to simple normalize_token
        if spacy_normalizer is not None:
            norm_result = spacy_normalizer._normalize_phrase(raw)
            norm = norm_result if norm_result else normalize_token(raw)
        else:
            norm = normalize_token(raw)
        
        # Apply dedupe mapping (variant → canonical)
        canon = apply_dedupe(norm, dedupe)
        
        # Map canonical form to ID
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


def _batch_normalize_phrases(
    phrases: List[str], spacy_normalizer: Optional[Any]
) -> List[Optional[str]]:
    """
    Normalize many ingredient spans in a single spaCy pipe pass.
    
    This avoids calling the normalizer per-entity, which is extremely slow
    (hundreds of thousands of extra pipeline runs on larger datasets).
    """
    if not phrases:
        return []

    # Fast path: simple normalization when spaCy normalizer is disabled/unavailable
    if spacy_normalizer is None:
        return [normalize_token(p) for p in phrases]

    cleaned = [
        SpacyIngredientNormalizer.clean_raw_text(str(p)) if p is not None else ""
        for p in phrases
    ]
    lowered = [c.strip().lower() for c in cleaned]
    n_process = spacy_normalizer.n_process or 1

    t0 = time.time()
    docs = spacy_normalizer.nlp.pipe(
        lowered,
        batch_size=spacy_normalizer.batch_size,
        n_process=max(n_process, 1),
    )

    norms: List[Optional[str]] = []
    for raw_clean, doc in zip(cleaned, docs):
        raw_clean = (raw_clean or "").strip()
        if not raw_clean:
            norms.append(None)
            continue
        norms.append(spacy_normalizer._normalize_doc(doc, raw_clean))

    elapsed = time.time() - t0
    logger.info(
        "Normalized %s entity spans with spaCy normalizer in %.2fs (%.1f spans/sec)",
        len(phrases),
        elapsed,
        len(phrases) / max(elapsed, 1e-6),
    )
    return norms


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
    # normalization
    use_spacy_normalizer: bool = True,  # Use spaCy normalizer to match training pipeline
    spacy_model: str = "en_core_web_sm",  # spaCy model for normalizer
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
    logger.info("Loaded trained spaCy model from %s", nlp_dir)
    
    # Initialize spaCy normalizer if requested (for consistent normalization with training)
    # CRITICAL: This ensures inference uses the same normalization as training pipeline
    # Without this, "buttermilk cornbread" stays as-is instead of becoming "cornbread"
    spacy_normalizer = None
    if use_spacy_normalizer and _HAS_SPACY_NORM:
        try:
            spacy_normalizer = SpacyIngredientNormalizer(model=spacy_model, batch_size=128, n_process=1)
            print(f"✓ Using spaCy normalizer (model={spacy_model}) for consistent normalization with training pipeline")
            print(f"  This ensures ingredients like 'buttermilk cornbread' → 'cornbread' (matching training data)")
        except Exception as e:
            print(f"⚠ Warning: Could not initialize spaCy normalizer: {e}. Falling back to simple normalization.")
            print(f"  This may cause ID=0 for many ingredients if they don't match encoder vocabulary.")
            spacy_normalizer = None
    elif use_spacy_normalizer and not _HAS_SPACY_NORM:
        print("⚠ Warning: spaCy normalizer not available. Using simple normalization.")
        print("  This may cause ID=0 for many ingredients. Install ingrnorm package for full normalization.")

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

    # Process each row: parse ingredient list and run NER on each ingredient separately
    # The model is trained on individual ingredients, not joined lists
    raw_texts = df_in[text_col].astype(str).tolist()
    
    # Flatten ingredients across all rows for a single nlp.pipe call
    flat_ing: List[str] = []
    offsets: List[tuple[int, int]] = []  # (row_idx, ingredient_idx)
    parsed_lists: List[List[str]] = []
    for i, raw_text in enumerate(raw_texts):
        ingredients = parse_listlike(raw_text)
        if ingredients and isinstance(ingredients, list):
            last_ing = ingredients[-1]
            if isinstance(last_ing, str) and " and " in last_ing.lower():
                parts = [p.strip() for p in last_ing.rsplit(" and ", 1)]
                if len(parts) == 2 and parts[1]:
                    ingredients[-1] = parts[0]
                    ingredients.append(parts[1])
        else:
            ingredients = []
        parsed_lists.append(ingredients)
        for j, ing in enumerate(ingredients):
            flat_ing.append(str(ing))
            offsets.append((i, j))

    logger.info(
        "Running NER on %s ingredient strings (batch_size=%s, n_process=%s)",
        len(flat_ing),
        batch_size,
        n_process,
    )
    t0 = time.time()
    docs_iter = nlp.pipe(flat_ing, batch_size=batch_size, n_process=n_process)

    entity_records: List[Dict[str, Any]] = []
    total_docs = 0
    for ing_text, doc, (row_idx, _) in zip(flat_ing, docs_iter, offsets):
        total_docs += 1
        entities = [ent for ent in doc.ents if ent.label_ == "INGREDIENT"]
        if entities:
            for ent in entities:
                entity_records.append(
                    {
                        "row_idx": row_idx,
                        "raw": ent.text,
                        "start": int(ent.start_char),
                        "end": int(ent.end_char),
                        "label": ent.label_,
                    }
                )
        else:
            # Fallback: treat the entire ingredient string as one entity
            entity_records.append(
                {
                    "row_idx": row_idx,
                    "raw": ing_text,
                    "start": 0,
                    "end": len(ing_text),
                    "label": "INGREDIENT",
                }
            )

    ner_elapsed = time.time() - t0
    logger.info(
        "NER pass done in %.2fs (%.1f docs/sec)",
        ner_elapsed,
        len(flat_ing) / max(ner_elapsed, 1e-6),
    )

    # Prepare holders
    wide_rows: List[Dict] = []
    tall_records: List[Dict] = []
    row_raw: List[List[str]] = [[] for _ in raw_texts]
    row_clean: List[List[str]] = [[] for _ in raw_texts]
    row_ids: List[List[int]] = [[] for _ in raw_texts]
    row_spans: List[List[Dict]] = [[] for _ in raw_texts]

    # Normalize all extracted entities in a single batch to avoid repeated spaCy calls
    norms = _batch_normalize_phrases(
        [rec["raw"] for rec in entity_records], spacy_normalizer
    )

    for rec, norm in zip(entity_records, norms):
        canon = apply_dedupe(norm, dedupe)
        tok_id = tok2id.get(canon, 0) if tok2id else None
        row_idx = rec["row_idx"]

        row_raw[row_idx].append(rec["raw"])
        if canon:
            row_clean[row_idx].append(canon)
        if tok2id:
            row_ids[row_idx].append(int(tok_id) if tok_id is not None else 0)

        span = {
            "raw": rec["raw"],
            "start": rec["start"],
            "end": rec["end"],
            "label": rec["label"],
            "norm": norm,
            "canonical": canon,
            "id": int(tok_id) if tok_id is not None else None,
        }
        row_spans[row_idx].append(span)
        tall_records.append(
            {
                "row_id": row_idx,
                text_col: raw_texts[row_idx],
                "ent_text": rec["raw"],
                "start": rec["start"],
                "end": rec["end"],
                "label": rec["label"],
                "norm": norm,
                "canonical": canon,
                "id": int(tok_id) if tok_id is not None else None,
            }
        )

    for i, raw_text in enumerate(raw_texts):
        clean_list = _unique_preserve_order(row_clean[i])
        wide_entry: Dict = {
            text_col: raw_text,
            "NER_raw": row_raw[i],
            "NER_clean": clean_list,
            "spans_json": json.dumps(row_spans[i], ensure_ascii=False),
        }
        if tok2id:
            wide_entry["Ingredients"] = row_ids[i]
        wide_rows.append(wide_entry)

    df_wide = pd.DataFrame(wide_rows)
    df_tall = pd.DataFrame(tall_records)
    
    # Diagnostic: count how many got ID=0 (not found in encoder)
    if tok2id:
        zero_ids = sum(1 for r in tall_records if r.get("id") == 0)
        total_entities = len(tall_records)
        if total_entities > 0:
            zero_pct = (zero_ids / total_entities) * 100
            print(f"\n[Diagnostic] ID mapping results:")
            print(f"  Total entities: {total_entities:,}")
            print(f"  Entities with ID=0 (not found): {zero_ids:,} ({zero_pct:.1f}%)")
            print(f"  Entities with valid ID: {total_entities - zero_ids:,} ({100-zero_pct:.1f}%)")
            
            # Check for recipes with only 1 ingredient ID
            recipes_with_one_id = sum(1 for w in wide_rows if tok2id and len(w.get("Ingredients", [])) == 1)
            total_recipes = len(wide_rows)
            if total_recipes > 0:
                one_id_pct = (recipes_with_one_id / total_recipes) * 100
                print(f"\n[Diagnostic] Recipe-level analysis:")
                print(f"  Total recipes: {total_recipes:,}")
                print(f"  Recipes with only 1 ingredient ID: {recipes_with_one_id:,} ({one_id_pct:.1f}%)")
                if one_id_pct > 10:
                    print(f"  ⚠ Note: Some recipes have only 1 ingredient ID.")
                    print(f"    This can happen if:")
                    print(f"    - Multiple ingredients normalize to the same canonical form")
                    print(f"    - Most ingredients get ID=0 (not in encoder vocabulary)")
                    print(f"    - NER model only detected 1 ingredient")
                    # Show a sample
                    sample = next((w for w in wide_rows if tok2id and len(w.get("Ingredients", [])) == 1 and len(w.get("NER_raw", [])) > 1), None)
                    if sample:
                        print(f"\n  Example recipe with 1 ID but multiple raw ingredients:")
                        print(f"    NER_raw ({len(sample.get('NER_raw', []))} items): {sample.get('NER_raw', [])[:5]}")
                        print(f"    NER_clean ({len(sample.get('NER_clean', []))} items): {sample.get('NER_clean', [])[:5]}")
                        print(f"    Ingredients ({len(sample.get('Ingredients', []))} IDs): {sample.get('Ingredients', [])}")
            
            if zero_pct > 50:
                print(f"\n  ⚠ Warning: >50% entities have ID=0. This suggests normalization mismatch.")
                print(f"    Ensure use_spacy_normalizer=True to match training pipeline normalization.")

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
    
    Note: The dedupe map and encoder maps are built from spaCy-normalized tokens (NER_clean).
    Therefore, inference must also use spaCy normalization to match these tokens.
    """
    dedupe = None
    tok2id = None
    
    # Load dedupe map (JSONL format)
    if DATA.DEDUPE_JSONL and DATA.DEDUPE_JSONL.exists():
        dedupe = load_jsonl_map(DATA.DEDUPE_JSONL)
        print(f"✓ Loaded dedupe map: {len(dedupe):,} mappings from {DATA.DEDUPE_JSONL}")
        # Show a sample for debugging
        if dedupe and len(dedupe) > 0:
            sample_items = list(dedupe.items())[:3]
            print(f"  Sample entries: {sample_items}")
    else:
        print(f"⚠ Dedupe map not found at {DATA.DEDUPE_JSONL} (skipping deduplication)")
    
    # Load encoder maps (token ↔ ID)
    if DATA.ING_TOK2ID_JSON and DATA.ING_TOK2ID_JSON.exists():
        _, tok2id = load_encoder_maps(DATA.ING_ID2TOK_JSON, DATA.ING_TOK2ID_JSON)
        if tok2id:
            print(f"✓ Loaded token→ID map: {len(tok2id):,} tokens from {DATA.ING_TOK2ID_JSON}")
            # Show a sample for debugging
            if tok2id and len(tok2id) > 0:
                sample_items = list(tok2id.items())[:3]
                print(f"  Sample entries: {sample_items}")
        else:
            print(f"⚠ Token→ID map not found or empty at {DATA.ING_TOK2ID_JSON}")
    else:
        print(f"⚠ Token→ID map not found at {DATA.ING_TOK2ID_JSON} (skipping ID encoding)")
    
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
    use_spacy_normalizer: bool = True,
    spacy_model: str = "en_core_web_sm",
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
        use_spacy_normalizer=use_spacy_normalizer,
        spacy_model=spacy_model,
    )


__all__ = ["predict_normalize_encode_structured", "load_dedupe_and_maps_from_config", "run_full_inference_from_config"]
