import config as cfg
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

import networkx as nx



Y_PRED_LR = cfg.PHASE_2_REPORTS_PATH / "y_pred_phase2.csv"
CLASS_REPORT_LR = cfg.PHASE_2_REPORTS_PATH / "classification_report_phase2_logreg.csv"
KMEANS_QUALITY = cfg.PHASE_2_REPORTS_PATH / "clustering_quality_phase2_kmeans.csv"
CLUSTER_ASSIGN = cfg.PHASE_2_REPORTS_PATH / "cluster_assignments_phase2.csv"
TOP_FEATURES = cfg.PHASE_2_REPORTS_PATH / "top_features_phase2_logreg.csv"  # from modeling_phase2
INGREDIENT_COUNTS = cfg.PHASE_2_REPORTS_PATH / "ingredient_counts_phase2.csv"

# Optional network files (if phase2 network code was run)
NETWORK_EDGES = cfg.PHASE_2_REPORTS_PATH / "network_edges.csv"
NETWORK_NODES = cfg.PHASE_2_REPORTS_PATH / "network_nodes_centrality.csv"



def plot_confusion_matrix_top():
    print("[VIS] Confusion Matrix (top cuisines)")

    if not Y_PRED_LR.exists():
        print(f"[WARN] {Y_PRED_LR} not found; skipping confusion matrix.")
        return

    dfp = pd.read_csv(Y_PRED_LR)

    # Prefer string labels if present
    if "y_test_label" in dfp.columns and "y_pred_label" in dfp.columns:
        y_true = dfp["y_test_label"]
        y_pred = dfp["y_pred_label"]
    else:
        # fallback to integer codes if labels not saved
        y_true = dfp["y_test"]
        y_pred = dfp["y_pred"]

    # Pick top 12 cuisines by frequency
    top_labels = y_true.value_counts().head(12).index.tolist()

    mask = y_true.isin(top_labels)
    y_true_top = y_true[mask]
    y_pred_top = y_pred[mask]

    cm = confusion_matrix(y_true_top, y_pred_top, labels=top_labels)

    fig = px.imshow(
        cm,
        x=top_labels,
        y=top_labels,
        color_continuous_scale="Blues",
        title="Confusion Matrix – Top 12 Cuisines",
        labels=dict(x="Predicted", y="True", color="Count"),
    )
    fig.update_layout(height=700, width=900)

    out = cfg.PHASE_2_VIZ_PATH / "confusion_matrix_top12.html"
    fig.write_html(out)
    print(f"[VIS] → {out}")



def plot_top_features():
    print("[VIS] Per-cuisine top ingredient features")

    if not TOP_FEATURES.exists():
        print(f"[WARN] {TOP_FEATURES} not found; skipping top features plots.")
        return

    df = pd.read_csv(TOP_FEATURES)

    # Figure out which columns we actually have
    if "ingredient" in df.columns:
        feat_col = "ingredient"
    elif "feature" in df.columns:
        feat_col = "feature"
    else:
        raise ValueError(
            f"Could not find ingredient/feature column in {TOP_FEATURES}. "
            f"Expected one of ['ingredient', 'feature'], found {list(df.columns)}"
        )

    if "importance" in df.columns:
        val_col = "importance"
    elif "weight" in df.columns:
        val_col = "weight"
    elif "mean_tfidf" in df.columns:
        val_col = "mean_tfidf"
    else:
        raise ValueError(
            f"Could not find numeric importance column in {TOP_FEATURES}. "
            f"Expected one of ['importance', 'weight', 'mean_tfidf'], "
            f"found {list(df.columns)}"
        )

    cuisines = df["cuisine"].unique()

    for c in cuisines:
        df_c = (
            df[df["cuisine"] == c]
            .sort_values(val_col, ascending=False)
            .head(10)
        )

        if df_c.empty:
            continue

        fig = px.bar(
            df_c,
            x=val_col,
            y=feat_col,
            orientation="h",
            title=f"Top Ingredients for {c}",
            labels={val_col: "Importance", feat_col: "Ingredient"},
        )
        fig.update_yaxes(autorange="reversed")

        out = cfg.PHASE_2_VIZ_PATH / f"top_features_{c}.html"
        fig.write_html(out)

    print("[VIS] → per-cuisine feature bar charts written")


def plot_cluster_heatmap():
    print("[VIS] Cluster composition heatmap")

    if not CLUSTER_ASSIGN.exists():
        print(f"[WARN] {CLUSTER_ASSIGN} not found; skipping cluster heatmap.")
        return

    df = pd.read_csv(CLUSTER_ASSIGN)

    if "cluster" not in df.columns or "cuisine_label" not in df.columns:
        raise ValueError(
            f"Expected columns 'cluster' and 'cuisine_label' in "
            f"{CLUSTER_ASSIGN}, found {list(df.columns)}"
        )

    pivot = df.pivot_table(
        index="cluster",
        columns="cuisine_label",
        aggfunc="size",
        fill_value=0,
    )

    # Keep only top cuisines to avoid a giant unreadable heatmap
    top_cols = pivot.sum().sort_values(ascending=False).head(15).index
    pivot = pivot[top_cols]

    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Cluster → Cuisine Composition Heatmap",
        labels=dict(x="Cuisine", y="Cluster", color="Count"),
    )

    out = cfg.PHASE_2_VIZ_PATH / "cluster_heatmap_phase2.html"
    fig.write_html(out)
    print(f"[VIS] → {out}")


def plot_silhouette_bar():
    print("[VIS] KMeans silhouette bar")

    if not KMEANS_QUALITY.exists():
        print(f"[WARN] {KMEANS_QUALITY} not found; skipping silhouette plot.")
        return

    df = pd.read_csv(KMEANS_QUALITY)
    k = int(df["k"].iloc[0])
    sil = float(df["silhouette"].iloc[0])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[str(k)],
            y=[sil],
            name="Silhouette",
        )
    )

    fig.update_layout(
        title=f"KMeans Silhouette Score (k={k})",
        xaxis_title="k",
        yaxis_title="Silhouette score",
        yaxis_range=[0, max(0.1, sil * 1.5)],
    )

    out = cfg.PHASE_2_VIZ_PATH / "kmeans_silhouette_phase2.html"
    fig.write_html(out)
    print(f"[VIS] → {out}")



def plot_ingredient_network():
    print("[VIS] Ingredient network (if available)")

    if not (NETWORK_EDGES.exists() and NETWORK_NODES.exists()):
        print(f"[WARN] Network files not found; skipping ingredient network plot.")
        return

    edges = pd.read_csv(NETWORK_EDGES)
    nodes = pd.read_csv(NETWORK_NODES)

    if edges.empty or nodes.empty:
        print("[WARN] Network edge/node tables are empty; skipping network.")
        return

    G = nx.Graph()
    for _, row in nodes.iterrows():
        ing = row.get("ingredient")
        if pd.isna(ing):
            continue
        G.add_node(ing)

    for _, row in edges.iterrows():
        a = row.get("ingredient_a")
        b = row.get("ingredient_b")
        if pd.isna(a) or pd.isna(b):
            continue
        G.add_edge(a, b, weight=row.get("pmi", 1.0))

    if G.number_of_nodes() == 0:
        print("[WARN] Network has 0 nodes; skipping.")
        return

    pos = nx.spring_layout(G, k=0.25, iterations=50)

    edge_x = []
    edge_y = []
    for a, b in G.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x = []
    node_y = []
    labels = []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        labels.append(n)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=5),
            hovertext=labels,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title="Ingredient PMI Network",
        showlegend=False,
        height=900,
        width=1100,
    )

    out = cfg.PHASE_2_VIZ_PATH / "ingredient_network_phase2.html"
    fig.write_html(out)
    print(f"[VIS] → {out}")



def main():
    print("\n=== Phase 2 Visualizations ===")

    plot_confusion_matrix_top()
    plot_top_features()
    plot_cluster_heatmap()
    plot_silhouette_bar()
    plot_ingredient_network()

    print("[VIS] Phase 2 visualizations complete.\n")


if __name__ == "__main__":
    main()
