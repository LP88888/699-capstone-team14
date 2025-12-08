"""
Visualization Stage: Generates Plotly HTMLs for Network, Confusion Matrix, and Features.
"""
from __future__ import annotations

import pandas as pd
import networkx as nx
import plotly.express as px
from sklearn.metrics import confusion_matrix
from pathlib import Path
import shutil
import seaborn as sns
from pyvis.network import Network
import numpy as np
import json
import re

from ..core import PipelineContext, StageResult
from ..utils import stage_logger
from ..ingrnorm.dedupe_map import _DROP_TOKENS, _DROP_SUBSTRINGS


def _inject_ingredient_controls(html_text: str) -> str:
    """Injects layout, hover bar, and chained click filter for ingredient graph."""
    layout_styles = """
    <style id="ing-layout">
      html, body { height:100vh; margin:0; padding:0; overflow:hidden; background:#ffffff; }
      body { display:flex; }
      #mynetwork { flex:1 1 auto; height:100vh; }
      #mynetwork > div.vis-network { width:100% !important; height:100% !important; }
      #ing-panel { width:320px; max-width:320px; background:#ffffff; border-left:1px solid #e2e8f0; padding:12px; box-shadow:-8px 0 24px rgba(17,24,39,0.08); font-family:Arial,sans-serif; overflow-y:auto; }
      .vis-tooltip { display:none !important; }
      @media (max-width: 1100px) {
        body { flex-direction:column; overflow:auto; }
        #ing-panel { width:100%; max-width:100%; position:relative; box-shadow:none; border-left:none; border-top:1px solid #e2e8f0; }
        #mynetwork { height:70vh; }
      }
    </style>
    """
    panel = """
    <div id="ing-panel">
      <div style="display:flex;gap:8px;align-items:center;justify-content:flex-start;margin-bottom:6px;">
        <a href="index.html" style="font-size:12px;color:#2563eb;font-weight:700;text-decoration:none;">Cuisine Network</a>
        <span style="color:#cbd5e1;">|</span>
        <a href="ingredient_network.html" style="font-size:12px;color:#111827;font-weight:700;text-decoration:none;">Ingredient Network</a>
      </div>
      <div style="font-weight:700;font-size:18px;color:#0f172a;margin-bottom:6px;">Ingredient PMI Network</div>
      <div style="font-size:12px;color:#475569;line-height:1.5;margin-bottom:12px;">
        Hover an ingredient to highlight its connections. Zoom in to read labels clearly. Click background or reset to clear highlights.
      </div>
      <div id="ing-bar" style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:8px 10px;box-shadow:0 8px 24px rgba(17,24,39,0.08);font-family:Arial,sans-serif;min-width:220px;max-width:100%;margin-bottom:10px;">
        <div id="ing-title" style="font-weight:700;color:#0f172a;font-size:14px;">Hover an ingredient</div>
        <div id="ing-meta" style="font-size:12px;color:#475569;margin-top:2px;"></div>
      </div>
      <button id="ing-reset" style="border:none;background:#0f172a;color:white;border-radius:8px;padding:8px 10px;font-size:12px;cursor:pointer;">Reset highlights</button>
    </div>
    """
    script = """
    <script type="text/javascript">
    (function(){
      const nodes = window.nodes;
      const edges = window.edges;
      const network = window.network;
      if(!nodes || !edges || !network){ return; }
      function syncEdges(){ /* no-op placeholder for compatibility */ }

      function updateBar(data){
        const tEl = document.getElementById("ing-title");
        const mEl = document.getElementById("ing-meta");
        if(!tEl || !mEl){ return; }
        if(!data){
          tEl.textContent = "Hover an ingredient";
          mEl.textContent = "";
          return;
        }
        tEl.textContent = data.label || data.id;
        const deg = data.degree ? `Degree: ${data.degree}` : "";
        const cent = data.centrality ? `Centrality: ${data.centrality.toFixed ? data.centrality.toFixed(4) : data.centrality}` : "";
        mEl.textContent = [deg, cent].filter(Boolean).join(" | ");
      }

      network.on("hoverNode", function(params){
        if(params.node){
          const data = nodes.get(params.node);
          updateBar(data);
          const neighbors = network.getConnectedNodes(params.node);
          const selection = [params.node].concat(neighbors);
          const edgeIds = network.getConnectedEdges(params.node);
          network.unselectAll();
          network.selectNodes(selection);
          network.selectEdges(edgeIds);
        }
      });
      network.on("blurNode", function(){
        updateBar(null);
        network.unselectAll();
      });

      const resetBtn = document.getElementById("ing-reset");
      if(resetBtn){
        resetBtn.addEventListener("click", function(){
          network.unselectAll();
          updateBar(null);
        });
      }
      // initial sync
      network.unselectAll();
    })();
    </script>
    """
    html_text = html_text.replace("<head>", "<head>" + layout_styles)
    # Insert UI elements inside body (after opening tag)
    return html_text.replace("<body>", "<body>" + panel).replace("</body>", script + "</body>")


def plot_ingredient_network(pmi_path: Path, centrality_path: Path, out_path: Path, title="Ingredient PMI Network"):
    """Generate an interactive ingredient network with bright colors and chained filtering."""
    df_pmi = pd.read_csv(pmi_path)
    df_nodes = pd.read_csv(centrality_path).set_index("ingredient")

    def _clean_ing(name: str) -> str | None:
        if not isinstance(name, str):
            return None
        t = name.strip().strip("\"'").lower()
        t = re.sub(r"[()\\[\\]{}]", " ", t)
        t = re.sub(r"\\d+[^a-z]+", " ", t)
        t = re.sub(r"[^a-z\\s]", " ", t)
        t = re.sub(r"\\s+", " ", t).strip()
        if not t:
            return None
        if t in _DROP_TOKENS or any(substr in t for substr in _DROP_SUBSTRINGS):
            return None
        if t.endswith("ed"):
            return None
        if len(t) <= 2:
            return None
        return t

    # Clean edges and filter strongest
    cleaned_edges = []
    for _, row in df_pmi.iterrows():
        a = _clean_ing(row["ingredient_a"])
        b = _clean_ing(row["ingredient_b"])
        if not a or not b or a == b:
            continue
        cleaned_edges.append((a, b, row["pmi"]))
    cleaned_edges = sorted(cleaned_edges, key=lambda x: -x[2])[:600]

    # Clean node metrics
    centrality_map = {}
    for ing, r in df_nodes.iterrows():
        c = _clean_ing(ing)
        if not c:
            continue
        cent = float(r.get("betweenness", 0.0))
        deg = float(r.get("degree", 0.0))
        if c not in centrality_map:
            centrality_map[c] = {"betweenness": cent, "degree": deg}
        else:
            centrality_map[c]["betweenness"] = max(centrality_map[c]["betweenness"], cent)
            centrality_map[c]["degree"] = max(centrality_map[c]["degree"], deg)

    # Build graph
    G = nx.Graph()
    for a, b, pmi in cleaned_edges:
        G.add_edge(a, b, pmi=pmi)
    degrees = dict(G.degree())

    # Color palette by degree bucket (food-friendly warm → fresh)
    palette = [
        "#f97316",  # orange
        "#f59e0b",  # amber
        "#fbbf24",  # sunflower
        "#fde047",  # lemon
        "#a3e635",  # lime
        "#4ade80",  # fresh green
        "#22c55e",  # vibrant green
        "#10b981",  # jade
        "#0ea5e9",  # sky accent
        "#8b5cf6",  # berry accent
    ]
    def bucket(deg):
        if deg >= 30: return 0
        if deg >= 20: return 1
        if deg >= 15: return 2
        if deg >= 10: return 3
        if deg >= 7: return 4
        if deg >= 5: return 5
        return 6

    net = Network(height="760px", width="100%", bgcolor="#ffffff", font_color="#0f172a", notebook=False, cdn_resources="in_line")
    net.force_atlas_2based(gravity=-60, central_gravity=0.01, spring_length=180, damping=0.6, overlap=0.8)

    for n in G.nodes():
        deg = degrees.get(n, 1)
        cent = df_nodes.loc[n, "betweenness"] if n in df_nodes.index else 0.0
        size = float(np.log1p(deg) * 8 + cent * 80)
        col = palette[bucket(deg) % len(palette)]
        title_html = f"{n}<br>Degree: {deg}<br>Centrality: {cent:.4f}"
        net.add_node(
            n,
            label=n,
            value=size,
            title=title_html,
            color=col,
            degree=deg,
            centrality=cent,
        )

    for a, b, pmi in cleaned_edges:
        w = max(0.8, np.log1p(pmi) * 1.2)
        net.add_edge(a, b, value=pmi, width=w, color="#cbd5e1")

    net.set_options(json.dumps({
        "nodes": {
            "shape": "dot",
            "scaling": {"min": 6, "max": 36},
            "shadow": True,
            "borderWidth": 1,
            "borderWidthSelected": 3,
            "font": {"size": 14, "face": "arial"}
        },
        "edges": {
            "smooth": {"type": "dynamic"},
            "color": {"inherit": False},
            "shadow": False,
            "selectionWidth": 2
        },
        "layout": {"improvedLayout": False},
        "physics": {"stabilization": {"iterations": 150}, "timestep": 0.35, "minVelocity": 0.75},
        "interaction": {"hover": True, "hoverConnectedEdges": True, "selectConnectedEdges": True, "multiselect": False}
    }))

    net.conf = False
    net.html = _inject_ingredient_controls(net.generate_html(notebook=False))
    out_path = Path(out_path)
    out_path.write_text(net.html, encoding="utf-8")

def plot_svd_variance(var_path: Path, out_path: Path) -> None:
    """Plot cumulative explained variance for SVD."""
    df = pd.read_csv(var_path)
    if df.empty:
        return
    fig = px.area(
        df,
        x="component",
        y="cumulative_explained_variance",
        hover_data={"explained_variance_ratio": ":.4f", "cumulative_explained_variance": ":.4f"},
        labels={
            "component": "Component",
            "cumulative_explained_variance": "Cumulative Explained Variance",
            "explained_variance_ratio": "Explained Variance Ratio",
        },
        title="TruncatedSVD Explained Variance",
    )
    fig.update_traces(mode="lines+markers")
    fig.update_layout(
        template="simple_white",
        yaxis=dict(range=[0, 1]),
        hovermode="x unified",
        height=520,
    )
    fig.write_html(str(out_path))

def plot_svd_scatter(sample_path: Path, out_path: Path, *, max_labels: int = 18) -> None:
    """Plot a 2D scatter of sampled SVD projections colored by cuisine."""
    df = pd.read_csv(sample_path)
    if df.empty:
        return
    label_cols = [c for c in df.columns if not c.startswith("svd_")]
    label_col = label_cols[0] if label_cols else "label"
    if label_col not in df.columns:
        df[label_col] = ""
    top_labels = df[label_col].value_counts().head(max_labels).index
    mask = df[label_col].isin(top_labels)
    plot_df = df[mask].copy()
    fig = px.scatter(
        plot_df,
        x="svd_1",
        y="svd_2",
        color=label_col,
        opacity=0.7,
        hover_data={label_col: True, "svd_1": ":.3f", "svd_2": ":.3f"},
        title="SVD Projection (sampled, colored by cuisine)",
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0)))
    fig.update_layout(
        template="simple_white",
        legend_title=label_col,
        height=700,
    )
    fig.write_html(str(out_path))

def plot_svd_component_terms(terms_path: Path, out_path: Path, *, max_components: int = 6, top_n: int = 8) -> None:
    """Facet bar charts for top +/- terms per component."""
    df = pd.read_csv(terms_path)
    if df.empty:
        return
    selected = sorted(df["component"].unique())[:max_components]
    panels = []
    for comp in selected:
        comp_df = df[df["component"] == comp]
        pos = comp_df[comp_df["direction"] == "positive"].nlargest(top_n, "weight")
        neg = comp_df[comp_df["direction"] == "negative"].nsmallest(top_n, "weight")
        comp_panel = pd.concat([pos, neg], axis=0)
        if comp_panel.empty:
            continue
        comp_panel = comp_panel.copy()
        comp_panel["component_label"] = f"Component {comp}"
        comp_panel["term_display"] = comp_panel["term"].apply(lambda t: t[:18] + "…" if isinstance(t, str) and len(t) > 20 else t)
        panels.append(comp_panel)

    if not panels:
        return
    plot_df = pd.concat(panels, axis=0)
    fig = px.bar(
        plot_df,
        x="weight",
        y="term_display",
        color="direction",
        facet_col="component_label",
        facet_col_wrap=3,
        orientation="h",
        color_discrete_map={"positive": "#0ea5e9", "negative": "#f97316"},
        title=f"SVD Components: Top +/-{top_n} terms (sample of {len(selected)} components)",
    )
    fig.update_layout(
        template="simple_white",
        height=450 + 180 * ((len(selected) - 1) // 3),
        legend_title="Direction",
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
    
    baseline_report_dir = Path(baseline_cfg.get("reports_dir", "reports/baseline")).resolve()
    viz_dir = Path(cfg.get("output", {}).get("viz_dir", "reports/viz_phase2")).resolve()
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 0. TruncatedSVD visuals
    svd_var_path = baseline_report_dir / "svd_variance.csv"
    if svd_var_path.exists():
        logger.info("Generating SVD explained variance chart...")
        plot_svd_variance(svd_var_path, viz_dir / "svd_explained_variance.html")

    svd_proj_path = baseline_report_dir / "svd_projection_sample.csv"
    if svd_proj_path.exists():
        logger.info("Generating SVD scatter (sampled projections)...")
        plot_svd_scatter(svd_proj_path, viz_dir / "svd_scatter.html")

    svd_terms_path = baseline_report_dir / "svd_components_top_terms.csv"
    if svd_terms_path.exists():
        logger.info("Generating SVD component term panels...")
        plot_svd_component_terms(svd_terms_path, viz_dir / "svd_component_terms.html")

    # 1. Confusion Matrix
    preds_path = baseline_report_dir / "y_pred_logreg.csv"
    if preds_path.exists():
        logger.info("Generating Confusion Matrix...")
        df_preds = pd.read_csv(preds_path)
        top_cuisines = df_preds["y_true"].value_counts().head(12).index
        mask = df_preds["y_true"].isin(top_cuisines)
        cm = confusion_matrix(df_preds[mask]["y_true"], df_preds[mask]["y_pred"], labels=top_cuisines)
        fig = px.imshow(cm, x=top_cuisines, y=top_cuisines, color_continuous_scale="Blues", title="Confusion Matrix (Top 12)")
        cm_path = (viz_dir / "confusion_matrix.html").resolve()
        fig.write_html(str(cm_path))
        
    # 2. Cluster Heatmap
    cluster_path = baseline_report_dir / "cluster_assignments.csv"
    if cluster_path.exists():
        logger.info("Generating Cluster Heatmap...")
        df_clust = pd.read_csv(cluster_path)
        pivot = df_clust.pivot_table(index="cluster", columns="cuisine", aggfunc="size", fill_value=0)
        top_cols = pivot.sum().sort_values(ascending=False).head(15).index
        fig = px.imshow(pivot[top_cols], aspect="auto", title="Cluster vs Cuisine Heatmap")
        clus_path = (viz_dir / "cluster_heatmap.html").resolve()
        fig.write_html(str(clus_path))
        
    # 3. Ingredient Network (The "Colleague Special")
    if pmi_path.exists() and nodes_path.exists():
        logger.info("Generating Interactive Ingredient Network...")
        ing_path = (viz_dir / "ingredient_network.html").resolve()
        plot_ingredient_network(pmi_path, nodes_path, ing_path)
        # copy to public for static hosting
        try:
            public_dir = Path("public")
            public_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(ing_path, public_dir / "ingredient_network.html")
        except Exception as exc:
            logger.warning("Failed to copy ingredient network to public: %s", exc)
            
    return StageResult(
        name="analysis_viz", 
        status="success",
        outputs={"viz_dir": str(viz_dir)}
    )
