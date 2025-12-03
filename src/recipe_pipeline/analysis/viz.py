"""
Visualization Stage: Generates Plotly HTMLs for Confusion Matrix, Features, and Network.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from pathlib import Path
from ...core import PipelineContext, StageResult
from ...utils import stage_logger

def run(context: PipelineContext, *, force: bool = False) -> StageResult:
    cfg = context.stage("analysis_viz")
    logger = stage_logger(context, "analysis_viz", force=force)
    
    # Inputs (from previous stages)
    pmi_cfg = context.stage("analysis_pmi").get("output", {})
    baseline_cfg = context.stage("analysis_baseline").get("output", {})
    
    pmi_report_dir = Path(pmi_cfg.get("reports_dir"))
    baseline_report_dir = Path(baseline_cfg.get("reports_dir"))
    viz_dir = Path(cfg.get("output", {}).get("viz_dir"))
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix (Top 12 Cuisines)
    preds_path = baseline_report_dir / "y_pred_logreg.csv"
    if preds_path.exists():
        logger.info("Generating Confusion Matrix...")
        df_preds = pd.read_csv(preds_path)
        top_cuisines = df_preds["y_true"].value_counts().head(12).index
        mask = df_preds["y_true"].isin(top_cuisines)
        
        cm = confusion_matrix(df_preds[mask]["y_true"], df_preds[mask]["y_pred"], labels=top_cuisines)
        fig = px.imshow(cm, x=top_cuisines, y=top_cuisines, color_continuous_scale="Blues", title="Confusion Matrix (Top 12)")
        fig.write_html(str(viz_dir / "confusion_matrix.html"))
        
    # 2. Cluster Heatmap
    cluster_path = baseline_report_dir / "cluster_assignments.csv"
    if cluster_path.exists():
        logger.info("Generating Cluster Heatmap...")
        df_clust = pd.read_csv(cluster_path)
        pivot = df_clust.pivot_table(index="cluster", columns="cuisine", aggfunc="size", fill_value=0)
        # Filter for readability (Top 15 cuisines)
        top_cols = pivot.sum().sort_values(ascending=False).head(15).index
        fig = px.imshow(pivot[top_cols], aspect="auto", title="Cluster vs Cuisine Heatmap")
        fig.write_html(str(viz_dir / "cluster_heatmap.html"))
        
    # 3. Top Features Bar Charts
    feat_path = baseline_report_dir / "top_features_logreg.csv"
    if feat_path.exists():
        logger.info("Generating Feature Plots...")
        df_feat = pd.read_csv(feat_path)
        for cuisine in df_feat["cuisine"].unique()[:5]: # Generate for top 5 just to test
            subset = df_feat[df_feat["cuisine"] == cuisine].head(10)
            fig = px.bar(subset, x="weight", y="feature", orientation='h', title=f"Top Ingredients: {cuisine}")
            fig.update_yaxes(autorange="reversed")
            fig.write_html(str(viz_dir / f"features_{cuisine}.html"))
            
    return StageResult(name="analysis_viz", status="success")