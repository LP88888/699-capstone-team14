"""
Visualization stage: Generates PyVis network and Matplotlib charts.
"""
from __future__ import annotations

import json
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path
from pyvis.network import Network

from ..core import PipelineContext, StageResult
from ..utils import stage_logger


def _lighten(hex_color: str, factor: float = 0.5) -> str:
    """Lighten a hex color by blending toward white."""
    if not hex_color:
        return hex_color
    h = hex_color.lstrip("#")
    try:
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return hex_color
    r = min(255, int(r + (255 - r) * factor))
    g = min(255, int(g + (255 - g) * factor))
    b = min(255, int(b + (255 - b) * factor))
    return f"#{r:02x}{g:02x}{b:02x}"

def plot_cuisine_network(df_top, cuisine_counts, out_path, *, min_shared=2, top_k=25, parent_map=None):
    """Generate an interactive cuisine similarity graph using shared top features."""
    parent_map = parent_map or {}

    def _clean_label(raw: str) -> str:
        t = str(raw).strip()
        # Drop surrounding bracket-like characters and trailing punctuation
        t = t.strip("[](){}\"' ")
        t = re.sub(r"^[\\s,;:_-]+|[\\s,;:_-]+$", "", t)
        # Normalize internal spacing
        t = re.sub(r"[_\\-]+", " ", t)
        t = re.sub(r"\\s+", " ", t)
        return t.strip()

    def _norm_label(s: str) -> str:
        t = str(s).strip()
        # Drop stray brackets/quotes and unify delimiters
        t = t.strip("[](){}\"'")
        t = re.sub(r"[,_]+", " ", t)
        t = t.replace("-", " ").replace("_", " ")
        t = re.sub(r"\s+", " ", t).lower().strip()
        return t

    parent_lookup = {_norm_label(k): str(v) for k, v in parent_map.items() if str(k).strip() and str(v).strip()}

    def resolve_parent(label: str) -> str:
        if not parent_lookup:
            return label
        return parent_lookup.get(_norm_label(label), label)

    # Keep only cuisines we care about and top-k features per cuisine
    clean_counts = {}
    for k, v in cuisine_counts.items():
        cleaned = _clean_label(k)
        clean_counts[cleaned] = clean_counts.get(cleaned, 0) + v
    cuisine_counts = clean_counts
    keep_cuisines = set(cuisine_counts.keys())
    df_filtered = df_top.copy()
    df_filtered["cuisine"] = df_filtered["cuisine"].map(_clean_label)
    df_filtered = df_filtered[df_filtered["cuisine"].isin(keep_cuisines)].copy()
    if "rank" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["rank"] <= top_k]

    # Build feature sets and weights per cuisine
    cuisine_to_terms = {}
    cuisine_term_weights = {}
    for cuisine, group in df_filtered.groupby("cuisine"):
        terms = set(str(x) for x in group["feature"])
        cuisine_to_terms[cuisine] = terms
        cuisine_term_weights[cuisine] = {str(r.feature): float(r.weight) for r in group.itertuples()}

    cuisine_parents = {c: resolve_parent(c) for c in keep_cuisines} if parent_lookup else {}
    parent_to_children = {}
    for child, parent in cuisine_parents.items():
        parent_to_children.setdefault(parent, []).append(child)

    net = Network(height="780px", width="1200px", bgcolor="#f8f9fb", font_color="#1b1b1b", notebook=False, cdn_resources="in_line")
    net.force_atlas_2based(gravity=-40, central_gravity=0.03, spring_length=180, damping=0.68, overlap=0.6)

    # Color palette for nodes (parent-aware if available)
    if cuisine_parents:
        parents = sorted(set(cuisine_parents.values()))
        palette = sns.color_palette("tab20", len(parents)).as_hex()
        parent_colors = {p: palette[i % len(palette)] for i, p in enumerate(parents)}
        cuisine_colors = {
            c: _lighten(parent_colors[cuisine_parents[c]], factor=0.5) for c in keep_cuisines
        }
    else:
        palette = sns.color_palette("cubehelix", len(keep_cuisines)).as_hex()
        parent_colors = {}
        cuisine_colors = {c: palette[i % len(palette)] for i, c in enumerate(sorted(keep_cuisines))}

    # Optional parent-level feature aggregation for cross-parent edges
    parent_term_weights = {}
    parent_terms = {}
    if cuisine_parents:
        for cuisine, terms in cuisine_to_terms.items():
            parent = cuisine_parents[cuisine]
            parent_terms.setdefault(parent, set()).update(terms)
            weights = cuisine_term_weights.get(cuisine, {})
            agg = parent_term_weights.setdefault(parent, {})
            for t, w in weights.items():
                agg[t] = agg.get(t, 0.0) + w

    # Add parent nodes sized by summed support
    if cuisine_parents:
        for parent, children in sorted(parent_to_children.items()):
            support = sum(cuisine_counts.get(ch, 1) for ch in children)
            preview = ", ".join(sorted(children)[:12])
            net.add_node(
                f"parent::{parent}",
                label=parent,
                value=float(np.log1p(support)),
                title=f"{parent}: {support} recipes<br>Children: {len(children)}<br>{preview}{' …' if len(children) > 12 else ''}",
                color=parent_colors.get(parent),
                shape="diamond",
                borderWidth=3,
                physics=True,
                font={"size": 18},
            )

    # Add cuisine nodes sized by support, colored by parent cluster when available
    for c in sorted(keep_cuisines):
        count = cuisine_counts.get(c, 1)
        weights = cuisine_term_weights.get(c, {})
        top_terms = sorted(weights.items(), key=lambda kv: -kv[1])[:8]
        top_terms_txt = "<br>".join(f"{t}: {w:.2f}" for t, w in top_terms)
        parent = cuisine_parents.get(c)
        net.add_node(
            c,
            label=c,
            value=float(np.log1p(count)),
            title=f"{c}: {count} recipes" + (f"<br>Parent: {parent}" if parent else "") + (f"<br>Top terms:<br>{top_terms_txt}" if top_terms else ""),
            color=cuisine_colors.get(c),
            group=parent or c,
            font={"size": 14},
        )

    # Add edges based on shared features (count + summed weight)
    cuisines = list(keep_cuisines)
    for i in range(len(cuisines)):
        for j in range(i + 1, len(cuisines)):
            c1, c2 = cuisines[i], cuisines[j]
            shared = cuisine_to_terms.get(c1, set()) & cuisine_to_terms.get(c2, set())
            if len(shared) >= min_shared:
                # Score: combined weight of shared terms for prioritizing strong overlaps
                weights1 = cuisine_term_weights.get(c1, {})
                weights2 = cuisine_term_weights.get(c2, {})
                shared_score = sum(weights1.get(t, 0) + weights2.get(t, 0) for t in shared)
                shared_preview = ", ".join(sorted(shared, key=lambda t: -(weights1.get(t, 0) + weights2.get(t, 0)))[:10])
                title = f"Shared terms: {len(shared)} | Score: {shared_score:.2f}<br>{shared_preview}"
                p1, p2 = cuisine_parents.get(c1, c1), cuisine_parents.get(c2, c2)
                edge_color = parent_colors[p1] if p1 == p2 and p1 in parent_colors else "#8898aa"
                net.add_edge(
                    c1,
                    c2,
                    value=len(shared),
                    width=max(1.5, np.log1p(len(shared)) * 1.4),
                    title=title,
                    color=edge_color,
                )

    # Connect parents to children for interactive cluster exploration
    if cuisine_parents:
        for child, parent in cuisine_parents.items():
            net.add_edge(
                f"parent::{parent}",
                child,
                value=1,
                width=1,
                color=parent_colors.get(parent, "#555"),
                title=f"{child} belongs to {parent}",
                dashes=True,
            )

        # Optional: parent-to-parent edges showing overlap of child feature spaces
        parent_list = list(parent_terms.keys())
        for i in range(len(parent_list)):
            for j in range(i + 1, len(parent_list)):
                p1, p2 = parent_list[i], parent_list[j]
                shared = parent_terms[p1] & parent_terms[p2]
                if len(shared) >= max(2, min_shared):
                    w1 = parent_term_weights.get(p1, {})
                    w2 = parent_term_weights.get(p2, {})
                    shared_score = sum(w1.get(t, 0) + w2.get(t, 0) for t in shared)
                    shared_preview = ", ".join(sorted(shared, key=lambda t: -(w1.get(t, 0) + w2.get(t, 0)))[:8])
                    net.add_edge(
                        f"parent::{p1}",
                        f"parent::{p2}",
                        value=len(shared),
                        width=max(1.5, np.log1p(len(shared)) * 1.1),
                        title=f"Parent overlap: {len(shared)} shared terms<br>{shared_preview}<br>Score: {shared_score:.2f}",
                        color="#4a5568",
                        dashes=True,
                    )

    net.set_options(
        json.dumps(
            {
                "nodes": {
                    "shape": "dot",
                    "scaling": {"min": 8, "max": 36},
                    "shadow": True,
                    "borderWidth": 1,
                    "borderWidthSelected": 3,
                    "font": {"size": 16, "strokeWidth": 0, "face": "arial"},
                },
                "edges": {
                    "smooth": {"type": "dynamic"},
                    "color": {"inherit": False},
                    "shadow": False,
                    "selectionWidth": 2,
                },
                "configure": {"enabled": True, "filter": True},
                "layout": {"improvedLayout": True},
                "physics": {"stabilization": {"iterations": 150}, "timestep": 0.35, "minVelocity": 0.75},
                "interaction": {
                    "hover": True,
                    "hoverConnectedEdges": True,
                    "selectConnectedEdges": True,
                    "multiselect": True,
                    "tooltipDelay": 100,
                    "navigationButtons": True,
                    "hideEdgesOnDrag": False,
                    "zoomView": True,
                    "dragView": True,
                    "keyboard": True,
                },
            }
        )
    )
    # Enable configure widget in the HTML
    net.conf = True
    net.write_html(str(out_path))

def run(context: PipelineContext, *, force: bool = False) -> StageResult:
    cfg = context.stage("analysis_graph")
    logger = stage_logger(context, "analysis_graph", force=force)
    
    # 1. Config & Inputs
    baseline_stage_cfg = context.stage("analysis_baseline")
    baseline_output_cfg = baseline_stage_cfg.get("output", {})
    reports_dir = Path(baseline_output_cfg.get("reports_dir", "reports/baseline"))
    
    top_feat_path = reports_dir / "top_features_logreg.csv"
    viz_dir = Path(cfg.get("output", {}).get("viz_dir", "reports/viz_graph"))
    viz_dir.mkdir(parents=True, exist_ok=True)

    parent_map_path = baseline_stage_cfg.get("params", {}).get("parent_map_path")
    parent_map = {}
    if parent_map_path:
        p = Path(parent_map_path)
        if p.exists():
            try:
                parent_map = json.load(open(p, "r", encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - safety net
                logger.warning("Failed to load parent map %s: %s", p, exc)
        else:
            logger.warning("Parent map not found at %s", p)
    
    if not top_feat_path.exists():
        logger.warning(f"Missing {top_feat_path}. Run analysis_baseline first.")
        return StageResult(name="analysis_graph", status="skipped", details="Missing input data")
        
    # 2. Load Data
    df_top = pd.read_csv(top_feat_path)
    
    # Infer frequency from predictions (preferred) or top_feat file (fallback).
    preds_path = reports_dir / "y_pred_logreg.csv"
    cuisine_counts_series = None
    if preds_path.exists():
        df_preds = pd.read_csv(preds_path)
        if "y_true" in df_preds.columns:
            cuisine_counts_series = df_preds["y_true"].value_counts()
        elif "y_true_parent" in df_preds.columns:
            # Fallback to parent-level counts if only parent labels exist
            cuisine_counts_series = df_preds["y_true_parent"].value_counts()
    if cuisine_counts_series is None:
        cuisine_counts_series = df_top["cuisine"].value_counts()
    # Keep top N children for readability
    max_nodes = int(cfg.get("params", {}).get("max_nodes", 30))
    cuisine_counts_series = cuisine_counts_series.head(max_nodes)
    cuisine_counts = cuisine_counts_series.to_dict()
    
    # 3. Generate Visualizations
    
    # [cite_start]A. Interactive Network [cite: 330]
    html_path = viz_dir / "cuisine_network.html"
    logger.info(f"Generating network graph -> {html_path}")
    import numpy as np # Ensure numpy is available for log1p
    plot_cuisine_network(df_top, cuisine_counts, html_path, min_shared=2, parent_map=parent_map)
    
    # [cite_start]B. Distribution Plot [cite: 307]
    dist_path = viz_dir / "cuisine_distribution.png"
    plt.figure(figsize=(10, 10))
    counts_df = cuisine_counts_series.reset_index()
    counts_df.columns = ["cuisine", "count"]
    sns.barplot(data=counts_df, x="count", y="cuisine", palette="viridis", dodge=False, hue="cuisine", legend=False)
    plt.title("Cuisine Distribution (Top by support)")
    plt.xlabel("Frequency")
    plt.ylabel("Cuisine")
    plt.tight_layout()
    plt.savefig(dist_path)
    plt.close()
    
    return StageResult(
        name="analysis_graph", 
        status="success", 
        outputs={"network": str(html_path), "plot": str(dist_path)}
    )
