"""
Visualization Stage: Generates Plotly HTMLs for Network, Confusion Matrix, and Features.
"""
from __future__ import annotations

import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from pathlib import Path
from ..core import PipelineContext, StageResult
from ..utils import stage_logger

def plot_ingredient_network(pmi_path: Path, centrality_path: Path, out_path: Path, title="Ingredient PMI Network"):
    """
    Generates an interactive Plotly network graph (from Colleague's Snippet 4).
    """
    # 1. Load Data
    df_pmi = pd.read_csv(pmi_path)
    df_nodes = pd.read_csv(centrality_path).set_index("ingredient")
    
    # Filter for visualization speed (top 500 edges by PMI)
    df_pmi = df_pmi.head(1000)
    
    # 2. Build Graph
    G = nx.from_pandas_edgelist(df_pmi, 'ingredient_a', 'ingredient_b', ['pmi'])
    
    # 3. Layout (Spring Layout)
    pos = nx.spring_layout(G, seed=42, k=0.15)
    
    # 4. Create Edges Trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # 5. Create Nodes Trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Get metrics
        cent = df_nodes.loc[node, "betweenness"] if node in df_nodes.index else 0
        deg = df_nodes.loc[node, "degree"] if node in df_nodes.index else 1
        
        node_text.append(f"{node}<br>Degree: {deg}<br>Centrality: {cent:.4f}")
        node_size.append(10 + (cent * 50)) # Size by centrality
        node_color.append(deg)             # Color by degree

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_size,
            color=node_color,
            colorbar=dict(thickness=15, title='Node Degree'),
            line_width=2))

    # 6. Assemble Figure
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title=title,
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
                
    fig.write_html(str(out_path))

def run(context: PipelineContext, *, force: bool = False) -> StageResult:
    cfg = context.stage("analysis_viz")
    logger = stage_logger(context, "analysis_viz", force=force)
    
    # Inputs
    pmi_cfg = context.stage("analysis_pmi").get("output", {})
    baseline_cfg = context.stage("analysis_baseline").get("output", {})
    
    pmi_path = Path(pmi_cfg.get("pmi_pairs", "reports/pmi/pairings_pmi_global.csv"))
    nodes_path = Path(pmi_cfg.get("node_metrics", "reports/pmi/network_nodes_centrality.csv"))
    
    baseline_report_dir = Path(baseline_cfg.get("reports_dir", "reports/baseline"))
    viz_dir = Path(cfg.get("output", {}).get("viz_dir", "reports/viz_phase2"))
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix
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
        top_cols = pivot.sum().sort_values(ascending=False).head(15).index
        fig = px.imshow(pivot[top_cols], aspect="auto", title="Cluster vs Cuisine Heatmap")
        fig.write_html(str(viz_dir / "cluster_heatmap.html"))
        
    # 3. Ingredient Network (The "Colleague Special")
    if pmi_path.exists() and nodes_path.exists():
        logger.info("Generating Interactive Ingredient Network...")
        plot_ingredient_network(pmi_path, nodes_path, viz_dir / "ingredient_network.html")
            
    return StageResult(
        name="analysis_viz", 
        status="success",
        outputs={"viz_dir": str(viz_dir)}
    )