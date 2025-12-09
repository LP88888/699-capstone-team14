"""
PMI Analysis Stage: Calculates Pointwise Mutual Information for ingredient pairs
and generates graph centrality metrics.
"""
from __future__ import annotations

import ast
from collections import Counter
from itertools import combinations
from math import log
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from ..core import PipelineContext, StageResult
from ..ingrnorm.dedupe_map import (
    _dedupe_preserve_order,
    load_jsonl_map,
    normalize_token_with_map,
)
from ..utils import stage_logger


def _normalize_ingredient_list(raw_value, mapping: dict) -> list[str]:
    """Parse/clean a single ingredient list cell with the SBERT dedupe map applied."""
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return []

    # Parse common containers (list/tuple/ndarray/JSON-string/comma-separated string)
    if isinstance(raw_value, (list, tuple, np.ndarray)):
        tokens = list(raw_value)
    elif isinstance(raw_value, str):
        s = raw_value.strip()
        if not s or s.lower() in {"nan", "none", "null", "[]"}:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                tokens = list(parsed) if isinstance(parsed, (list, tuple, np.ndarray)) else [parsed]
            except Exception:
                tokens = [p.strip() for p in s.split(",") if p.strip()]
        else:
            tokens = [p.strip() for p in s.split(",") if p.strip()]
    else:
        tokens = [str(raw_value)]

    cleaned = []
    for tok in tokens:
        mapped = normalize_token_with_map(tok, mapping)
        if mapped:
            cleaned.append(mapped)
    return _dedupe_preserve_order(cleaned)


def calculate_pmi(
    df,
    ingredients_col,
    min_count=5,
    min_pmi=1.0,
    dedupe_map=None,
    max_items_per_record: int | None = None,
):
    """Calculate PMI for all ingredient pairs."""
    # Flatten and count ingredients
    total_recipes = len(df)
    ing_counter = Counter()
    pair_counter = Counter()
    mapping = dedupe_map or {}
    trimmed_records = 0

    # Pass 1: Counts
    for lst in df[ingredients_col]:
        unique_ings = sorted(set(_normalize_ingredient_list(lst, mapping)))
        if max_items_per_record and len(unique_ings) > max_items_per_record:
            unique_ings = unique_ings[:max_items_per_record]
            trimmed_records += 1
        ing_counter.update(unique_ings)
        for a, b in combinations(unique_ings, 2):
            pair_counter[(a, b)] += 1

    rows = []
    # Pass 2: PMI Calculation
    for (a, b), count in pair_counter.items():
        if count < min_count:
            continue
        
        p_a = ing_counter[a] / total_recipes
        p_b = ing_counter[b] / total_recipes
        p_ab = count / total_recipes
        
        if p_ab <= 0 or p_a <= 0 or p_b <= 0:
            continue
            
        pmi = log(p_ab / (p_a * p_b))
        
        if pmi >= min_pmi:
            rows.append({
                "ingredient_a": a,
                "ingredient_b": b,
                "count": count,
                "pmi": pmi
            })
            
    df_out = pd.DataFrame(rows).sort_values("pmi", ascending=False)
    return df_out, trimmed_records

def run(context: PipelineContext, *, force: bool = False) -> StageResult:
    cfg = context.stage("analysis_pmi")
    logger = stage_logger(context, "analysis_pmi", force=force)
    
    # Config
    data_cfg = cfg.get("data", {})
    params = cfg.get("params", {})
    output_cfg = cfg.get("output", {})
    
    input_path = Path(data_cfg.get("input_path"))
    out_dir = Path(output_cfg.get("reports_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ingredients_col = data_cfg.get("ingredients_col", "inferred_ingredients")
    
    # 1. Load Data
    logger.info(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)

    # Optional dedupe map (SBERT-built) to clean noise and collapse variants
    dedupe_map_path = data_cfg.get("dedupe_map_path")
    dedupe_map = {}
    if dedupe_map_path:
        p = Path(dedupe_map_path)
        if p.exists():
            try:
                dedupe_map = load_jsonl_map(p)
                logger.info("Loaded %d dedupe entries from %s", len(dedupe_map), p)
            except Exception as exc:
                logger.warning("Failed to load dedupe map %s: %s", p, exc)
        else:
            logger.warning("Dedupe map %s not found; proceeding without canonicalization", p)
    
    # Ensure list column
    if not isinstance(df[ingredients_col].iloc[0], (list, np.ndarray)):
         # Fallback for string representation
         df[ingredients_col] = df[ingredients_col].apply(
             lambda x: ast.literal_eval(str(x)) if str(x).startswith("[") else str(x).split()
         )

    # 2. Calculate PMI
    logger.info("Calculating PMI pairings...")
    df_pmi, trimmed_records = calculate_pmi(
        df, 
        ingredients_col, 
        min_count=params.get("min_count", 15),
        min_pmi=params.get("min_pmi", 1.0),
        dedupe_map=dedupe_map,
        max_items_per_record=params.get("max_items_per_record", 60),
    )
    pmi_path = out_dir / "pairings_pmi_global.csv"
    df_pmi.to_csv(pmi_path, index=False)
    if trimmed_records:
        logger.info("Trimmed %s records with >%s ingredients to reduce pair explosion", trimmed_records, params.get("max_items_per_record", 60))
    logger.info(f"Saved {len(df_pmi)} PMI pairs to {pmi_path}")
    
    # 3. Build Graph & Calculate Centrality
    logger.info("Building NetworkX graph...")
    G = nx.Graph()
    for _, row in df_pmi.iterrows():
        G.add_edge(
            row["ingredient_a"], 
            row["ingredient_b"], 
            weight=row["pmi"], 
            count=row["count"]
        )
        
    logger.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    logger.info("Calculating centrality (Degree, Betweenness)...")
    deg = nx.degree_centrality(G)
    bet = nx.betweenness_centrality(G)
    
    nodes_data = []
    for n in G.nodes():
        nodes_data.append({
            "ingredient": n,
            "degree": G.degree(n),
            "degree_centrality": deg.get(n, 0),
            "betweenness": bet.get(n, 0)
        })
        
    nodes_path = out_dir / "network_nodes_centrality.csv"
    pd.DataFrame(nodes_data).to_csv(nodes_path, index=False)
    
    # Export GEXF for Gephi (Bonus!)
    gexf_path = out_dir / "ingredient_network.gexf"
    nx.write_gexf(G, gexf_path)
    
    return StageResult(
        name="analysis_pmi",
        status="success",
        outputs={
            "pmi_pairs": str(pmi_path),
            "node_metrics": str(nodes_path),
            "gexf": str(gexf_path)
        }
    )
