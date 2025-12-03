"""
Visualization stage: Generates PyVis network and Matplotlib charts.
"""
from __future__ import annotations

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from pyvis.network import Network

from ...core import PipelineContext, StageResult
from ...utils import stage_logger

def plot_cuisine_network(df_top, cuisine_counts, out_path, min_shared=5):
    """Generates the interactive PyVis HTML graph."""
    # [cite_start]Group features by cuisine [cite: 329]
    cuisine_to_terms = (
        df_top.groupby("cuisine")["feature"]
        .apply(lambda s: set(str(x) for x in s))
        .to_dict()
    )
    
    net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="#222222", notebook=False)
    net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=150)
    
    # Add Nodes
    cuisines = list(cuisine_to_terms.keys())
    for c in cuisines:
        count = cuisine_counts.get(c, 1)
        net.add_node(c, label=c, value=np.log1p(count), title=f"{c}: {count} recipes")
        
    # [cite_start]Add Edges based on shared ingredients [cite: 333]
    for i in range(len(cuisines)):
        for j in range(i + 1, len(cuisines)):
            c1, c2 = cuisines[i], cuisines[j]
            shared = cuisine_to_terms[c1] & cuisine_to_terms[c2]
            
            if len(shared) >= min_shared:
                shared_str = ", ".join(list(shared)[:10])
                title = f"Shared ({len(shared)}): {shared_str}"
                net.add_edge(c1, c2, value=len(shared), title=title)
                
    net.write_html(str(out_path))

def run(context: PipelineContext, *, force: bool = False) -> StageResult:
    cfg = context.stage("analysis_graph")
    logger = stage_logger(context, "analysis_graph", force=force)
    
    # 1. Config & Inputs
    baseline_cfg = context.stage("analysis_baseline").get("output", {})
    reports_dir = Path(baseline_cfg.get("reports_dir", "reports/baseline"))
    
    top_feat_path = reports_dir / "top_features_per_cuisine.csv"
    viz_dir = Path(cfg.get("output", {}).get("viz_dir", "reports/viz"))
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    if not top_feat_path.exists():
        logger.warning(f"Missing {top_feat_path}. Run analysis_baseline first.")
        return StageResult(name="analysis_graph", status="skipped", details="Missing input data")
        
    # 2. Load Data
    df_top = pd.read_csv(top_feat_path)
    
    # Need recipe counts for node sizing. We can infer rough counts or load summary.
    # For now, we'll infer frequency from how often cuisines appear in the top_feat file 
    # (or you can load the summary.json if preferred).
    cuisine_counts = df_top["cuisine"].value_counts().to_dict() 
    
    # 3. Generate Visualizations
    
    # [cite_start]A. Interactive Network [cite: 330]
    html_path = viz_dir / "cuisine_network.html"
    logger.info(f"Generating network graph -> {html_path}")
    import numpy as np # Ensure numpy is available for log1p
    plot_cuisine_network(df_top, cuisine_counts, html_path, min_shared=3)
    
    # [cite_start]B. Distribution Plot [cite: 307]
    dist_path = viz_dir / "cuisine_distribution.png"
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(cuisine_counts.values()), y=list(cuisine_counts.keys()), palette="viridis")
    plt.title("Cuisine Distribution (Top Features)")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(dist_path)
    plt.close()
    
    return StageResult(
        name="analysis_graph", 
        status="success", 
        outputs={"network": str(html_path), "plot": str(dist_path)}
    )