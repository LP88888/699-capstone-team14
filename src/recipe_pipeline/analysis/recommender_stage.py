"""
Recommender Stage: Generates fusion suggestions using the PMI graph.
"""
from __future__ import annotations

import json
import pandas as pd
import networkx as nx
from pathlib import Path
from collections import Counter, defaultdict

from ..core import PipelineContext, StageResult
from ..utils import stage_logger
from .recommender import CuisineRecommender 

def run(context: PipelineContext, *, force: bool = False) -> StageResult:
    cfg = context.stage("analysis_recommender")
    logger = stage_logger(context, "analysis_recommender", force=force)
    
    # 1. Configuration & Inputs
    pmi_cfg = context.stage("analysis_pmi").get("output", {})
    norm_cfg = context.stage("cuisine_normalization").get("output", {})
    
    pmi_path = Path(pmi_cfg.get("pmi_pairs", "reports/pmi/pairings_pmi_global.csv"))
    data_path = Path(cfg.get("data", {}).get("input_path", "data/combined_raw_datasets_with_inference_with_cuisine_encoded.parquet"))
    
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
    
    # Optimized iteration
    # Ensure ingredients are list
    if isinstance(df[ing_col].iloc[0], str):
        import ast
        df[ing_col] = df[ing_col].apply(lambda x: ast.literal_eval(x) if x.startswith('[') else x.split())

    for cuisine, ings in zip(df[cuisine_col], df[ing_col]):
        if pd.notna(cuisine) and isinstance(ings, list):
            cuisine_map[cuisine].update(ings)
            
    # 4. Initialize Recommender
    logger.info("Initializing Engine...")
    rec = CuisineRecommender(G, cuisine_map)
    
    # 5. Run Scenarios (Defined in Config or Hardcoded Defaults)
    scenarios = cfg.get("params", {}).get("scenarios", [
        ["Italian", "Japanese"],
        ["Mexican", "Thai"],
        ["Indian", "French"]
    ])
    
    results = []
    for c1, c2 in scenarios:
        logger.info(f"Generating fusion: {c1} + {c2}")
        try:
            fusion = rec.suggest_fusion(c1, c2, strictness=0.6)
            
            # Save individual JSON
            fname = f"fusion_{c1}_{c2}.json".replace(" ", "")
            with open(out_dir / fname, "w") as f:
                json.dump(fusion, f, indent=2)
                
            results.append(fname)
        except Exception as e:
            logger.warning(f"Failed fusion for {c1}/{c2}: {e}")
            
    return StageResult(
        name="analysis_recommender",
        status="success",
        outputs={"fusion_reports": [str(out_dir / r) for r in results]}
    )
