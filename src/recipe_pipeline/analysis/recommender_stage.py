"""
Recommender Stage: Generates fusion suggestions using the PMI graph.
"""
from __future__ import annotations

import json
import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import itertools
import re

from ..ingrnorm.dedupe_map import load_jsonl_map, _DROP_TOKENS, _DROP_SUBSTRINGS

from ..core import PipelineContext, StageResult
from ..utils import stage_logger
from .recommender import CuisineRecommender 

def run(context: PipelineContext, *, force: bool = False) -> StageResult:
    cfg = context.stage("analysis_recommender")
    logger = stage_logger(context, "analysis_recommender", force=force)
    
    # 1. Configuration & Inputs
    pmi_cfg = context.stage("analysis_pmi").get("output", {})
    baseline_cfg = context.stage("analysis_baseline").get("output", {})
    norm_cfg = context.stage("cuisine_normalization").get("output", {})
    
    pmi_path = Path(pmi_cfg.get("pmi_pairs", "reports/pmi/pairings_pmi_global.csv"))
    data_path = Path(cfg.get("data", {}).get("input_path", "data/combined_raw_datasets_with_inference_with_cuisine_encoded.parquet"))
    dedupe_path = Path(cfg.get("data", {}).get("dedupe_map_path", "data/ingr_normalized/dedupe_map.jsonl"))
    baseline_reports = Path(baseline_cfg.get("reports_dir", "reports/cuisines/baseline"))
    
    out_dir = Path(cfg.get("output", {}).get("reports_dir", "reports/fusion"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not pmi_path.exists():
        raise FileNotFoundError(f"PMI graph not found at {pmi_path}. Run analysis_pmi first.")
        
    # 2. Rebuild Graph from PMI Data (Much faster than recalculating)
    logger.info("Rebuilding graph from PMI pairs...")
    df_pmi = pd.read_csv(pmi_path)
    G = nx.from_pandas_edgelist(df_pmi, 'ingredient_a', 'ingredient_b', ['pmi', 'count'])
    
    # 3. Build Cuisine Profiles
    logger.info("Building cuisine ingredient profiles...")
    df = pd.read_parquet(data_path)
    
    ing_col = cfg.get("data", {}).get("ingredients_col", "inferred_ingredients")
    cuisine_col = cfg.get("data", {}).get("cuisine_col", "cuisine_deduped")
    
    # Map: Cuisine -> Counter({ingredient: freq})
    cuisine_map = defaultdict(Counter)
    
    # Load dedupe map if available
    dedupe_map = {}
    if dedupe_path.exists():
        try:
            dedupe_map = load_jsonl_map(dedupe_path)
            logger.info("Loaded %d dedupe terms from %s", len(dedupe_map), dedupe_path)
        except Exception as exc:
            logger.warning("Failed to load dedupe map %s: %s", dedupe_path, exc)

    # Optimized iteration with safe parsing per row
    import ast
    NOISE_WORDS = {"finely", "follows", "diced", "cut", "for", "chopped", "sliced", "pieces"}

    def _clean_ing(token: str) -> str | None:
        """Lightweight cleaner to drop prep words and noise."""
        if not token:
            return None
        t = str(token).strip().strip('"').strip("'")
        if not t:
            return None
        t = re.sub(r"^[,;:]+|[,;:]+$", "", t)
        low = t.lower()
        # Apply dedupe map if present
        mapped = dedupe_map.get(low, t)
        if isinstance(mapped, str):
            t = mapped.strip()
            low = t.lower()
        if low in _DROP_TOKENS or low in NOISE_WORDS:
            return None
        if any(substr in low for substr in _DROP_SUBSTRINGS):
            return None
        if low.endswith("ed"):
            return None
        if re.fullmatch(r"[\\W_]+", low):
            return None
        return t

    def _parse_ings(val):
        # Normalize array-like first
        if isinstance(val, (list, tuple, set)):
            return list(val)
        if isinstance(val, np.ndarray):
            return list(val.flatten())
        # Scalar NaN/None check
        try:
            if val is None:
                return []
            if isinstance(val, float) and np.isnan(val):
                return []
            if pd.isna(val) and not isinstance(val, str):
                return []
        except Exception:
            pass
        if isinstance(val, str):
            s = val.strip()
            try:
                if s.startswith("["):
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple)):
                        return list(parsed)
                return s.split()
            except Exception:
                return []
        return []

    def _norm_cuisine(val):
        if isinstance(val, str):
            t = val.strip().strip("[](){}\"'")
            t = re.sub(r"^[\s,;:_-]+|[\s,;:_-]+$", "", t)
            t = re.sub(r"[\s\-_]+", " ", t).strip()
            return t
        if isinstance(val, (list, tuple, set, np.ndarray)):
            try:
                # take first element if array-like
                if len(val) > 0:
                    return str(list(val)[0]).strip().strip("[](){}\"'")
            except Exception:
                pass
            return ""
        if pd.isna(val):
            return ""
        return str(val).strip()

    def _clean_cuisine_label(raw: str) -> str:
        """Mirror graph label cleaning to keep names aligned with nodes."""
        t = str(raw).strip()
        t = t.strip("[](){}\"' ")
        t = re.sub(r"^[\s,;:_-]+|[\s,;:_-]+$", "", t)
        t = re.sub(r"[\s\-_]+", " ", t)
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    # Eagerly clean and dedupe ingredient lists once after load
    df[ing_col] = df[ing_col].apply(
        lambda raw: [
            cleaned
            for cleaned in (_clean_ing(tok) for tok in _parse_ings(raw))
            if cleaned
        ]
    )

    ingredient_counts = Counter()
    for cuisine, ings in zip(df[cuisine_col], df[ing_col]):
        norm_cuisine = _norm_cuisine(cuisine)
        clean_cuisine = _clean_cuisine_label(norm_cuisine)
        if clean_cuisine and ings:
            cuisine_map[clean_cuisine].update(ings)
            ingredient_counts.update(ings)
            
    # 4. Initialize Recommender
    logger.info("Initializing Engine...")
    rec = CuisineRecommender(G, cuisine_map, ingredient_counts=ingredient_counts)
    
    # 5. Run Scenarios (Defined in Config or Hardcoded Defaults)
    params = cfg.get("params", {})
    max_cuisines = int(params.get("max_cuisines", 30))
    scenarios = params.get("scenarios")

    # Derive the same cuisine set as the graph: use baseline predictions/top features and limit to max_cuisines
    cuisine_counts_series = None
    preds_path = baseline_reports / "y_pred_logreg.csv"
    top_feat_path = baseline_reports / "top_features_logreg.csv"
    if preds_path.exists():
        df_preds = pd.read_csv(preds_path)
        if "y_true" in df_preds.columns:
            cuisine_counts_series = df_preds["y_true"].value_counts()
        elif "y_true_parent" in df_preds.columns:
            cuisine_counts_series = df_preds["y_true_parent"].value_counts()
    if cuisine_counts_series is None and top_feat_path.exists():
        df_top = pd.read_csv(top_feat_path)
        cuisine_counts_series = df_top["cuisine"].value_counts()
    if cuisine_counts_series is None:
        cuisine_counts_series = pd.Series({c: sum(cnt.values()) for c, cnt in cuisine_map.items()})
    cuisine_counts_series = cuisine_counts_series.head(max_cuisines)
    keep_cuisines = set(cuisine_counts_series.index.tolist())

    # Prune cuisine_map to keep only top cuisines to align with the chart
    cuisine_map = {c: cnt for c, cnt in cuisine_map.items() if c in keep_cuisines}

    if not scenarios:
        top_cuisines = sorted(keep_cuisines)
        scenarios = list(itertools.combinations(top_cuisines, 2))
    # normalize and clean scenarios
    scenarios = [( _norm_cuisine(a), _norm_cuisine(b) ) for a, b in scenarios if _norm_cuisine(a) and _norm_cuisine(b) and _norm_cuisine(a) in keep_cuisines and _norm_cuisine(b) in keep_cuisines]

    def _slug(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "", name) or "cuisine"
    
    results = []
    for c1, c2 in scenarios:
        clean_c1 = _clean_cuisine_label(c1)
        clean_c2 = _clean_cuisine_label(c2)
        logger.info(f"Generating fusion: {clean_c1} + {clean_c2}")
        try:
            fusion = rec.suggest_fusion(clean_c1, clean_c2, strictness=0.6)
            fusion["pair"] = f"{clean_c1} + {clean_c2}"
            # Save individual JSON
            slug_a = _slug(clean_c1)
            slug_b = _slug(clean_c2)
            fname = f"fusion_{slug_a}_{slug_b}.json"
            with open(out_dir / fname, "w", encoding="utf-8") as f:
                json.dump(fusion, f, indent=2, ensure_ascii=False)
            results.append(fname)
        except Exception as e:
            logger.warning(f"Failed fusion for {c1}/{c2}: {e}")
            
    return StageResult(
        name="analysis_recommender",
        status="success",
        outputs={"fusion_reports": [str(out_dir / r) for r in results]}
    )
