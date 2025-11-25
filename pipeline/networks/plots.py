#!/usr/bin/env python
"""
Master visualization script for cuisine–ingredient project.

Generates:
- Cuisine distribution bar chart
- Top ingredients per cuisine heatmap
- Model macro-F1 comparison plot
- Cuisine similarity network map (based on shared top ingredients)
"""

import os
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# -------------------------
# Paths / config
# -------------------------
RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = "reports/phase2"
VIZ_DIR = "reports/viz"

INPUT_CSV = "data/encoded/combined_raw_datasets_with_cuisine_encoded.parquet"


os.makedirs(VIZ_DIR, exist_ok=True)

sns.set(style="whitegrid", context="talk")


# -------------------------
# Helpers
# -------------------------
def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return None
    return pd.read_parquet(path)


def get_macro_f1_from_report(path: Path) -> float | None:
    """Try to pull macro F1 from a classification_report CSV."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)

    # Case 1: has 'label' column
    if "label" in df.columns:
        row = df.loc[df["label"] == "macro avg"]
        if not row.empty and "f1-score" in row.columns:
            return float(row["f1-score"].iloc[0])

    # Case 2: unnamed first column
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "label"})
        row = df.loc[df["label"] == "macro avg"]
        if not row.empty and "f1-score" in row.columns:
            return float(row["f1-score"].iloc[0])

    # Case 3: index as label
    df_idx = pd.read_csv(path, index_col=0)
    if "macro avg" in df_idx.index and "f1-score" in df_idx.columns:
        return float(df_idx.loc["macro avg", "f1-score"])

    return None


# -------------------------
# Plot 1: Cuisine distribution
# -------------------------
def plot_cuisine_distribution(df: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    print("[PLOT] Cuisine distribution")
    if "cuisine" not in df.columns:
        print("[WARN] No 'cuisine' column in dataframe; skipping cuisine distribution.")
        return

    counts = (
        df["cuisine"]
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts()
    )

    if counts.empty:
        print("[WARN] No cuisine counts; skipping cuisine distribution.")
        return

    top = counts.head(top_n)[::-1]  # reverse for nicer barh order

    plt.figure(figsize=(8, 0.4 * len(top) + 2))
    plt.barh(top.index, top.values)
    plt.xlabel("Number of recipes")
    plt.ylabel("Cuisine")
    plt.title(f"Top {len(top)} cuisines by recipe count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  -> saved to {out_path}")


# -------------------------
# Plot 2: Top ingredients heatmap
# -------------------------
def plot_top_ingredients_heatmap(
    top_features_path: Path,
    cuisine_counts: pd.Series,
    out_path: Path,
    n_cuisines: int = 10,
    top_per_cuisine: int = 10,
) -> None:
    print("[PLOT] Top ingredients per cuisine heatmap")
    df_top = safe_read_csv(top_features_path)
    if df_top is None:
        print("[WARN] No top_features_per_cuisine.csv; skipping heatmap.")
        return

    required_cols = {"cuisine", "feature", "mean_tfidf", "rank"}
    if not required_cols.issubset(df_top.columns):
        print(f"[WARN] top_features file missing columns {required_cols}; skipping heatmap.")
        return

    # Select top cuisines by recipe count
    chosen_cuisines = cuisine_counts.head(n_cuisines).index.tolist()
    df_top = df_top[df_top["cuisine"].isin(chosen_cuisines)]
    df_top = df_top[df_top["rank"] <= top_per_cuisine]

    if df_top.empty:
        print("[WARN] No data after filtering top cuisines/features; skipping heatmap.")
        return

    # Pivot: cuisines x ingredients
    pivot = df_top.pivot(index="cuisine", columns="feature", values="mean_tfidf").fillna(0.0)

    # Order cuisines by overall count
    pivot = pivot.loc[[c for c in chosen_cuisines if c in pivot.index]]

    plt.figure(figsize=(1.2 * pivot.shape[1], 0.5 * pivot.shape[0] + 3))
    sns.heatmap(
        pivot,
        cmap="viridis",
        cbar_kws={"label": "Mean TF-IDF"},
        linewidths=0.3,
        linecolor="gray",
    )
    plt.xlabel("Ingredient")
    plt.ylabel("Cuisine")
    plt.title("Top ingredients (by TF-IDF) for top cuisines")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  -> saved to {out_path}")


# -------------------------
# Plot 3: Model macro-F1 comparison
# -------------------------
def plot_model_performance(out_path: Path) -> None:
    print("[PLOT] Model macro-F1 comparison")
    models = []
    macro_f1s = []

    # TF-IDF only
    tfidf_only_path = REPORTS_DIR + "/classification_report_tfidf_only.csv"
    m_f1 = get_macro_f1_from_report(tfidf_only_path)
    if m_f1 is not None:
        models.append("TF-IDF only")
        macro_f1s.append(m_f1)

    # TF-IDF + graph (if exists)
    tfidf_graph_path = REPORTS_DIR + "/classification_report_tfidf_plus_graph.csv"
    m_f1_graph = get_macro_f1_from_report(tfidf_graph_path)
    if m_f1_graph is not None:
        models.append("TF-IDF + Graph")
        macro_f1s.append(m_f1_graph)

    if not models:
        print("[WARN] No classification reports found; skipping model performance plot.")
        return

    plt.figure(figsize=(6, 4))
    sns.barplot(x=models, y=macro_f1s)
    plt.ylabel("Macro F1-score")
    plt.ylim(0, max(macro_f1s) * 1.1)
    plt.title("Model performance (macro F1)")
    for i, v in enumerate(macro_f1s):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  -> saved to {out_path}")


# -------------------------
# Plot 4: Cuisine similarity network map
# -------------------------
def plot_cuisine_network(
    top_features_path: Path,
    cuisine_counts: pd.Series,
    out_path: Path,
    n_cuisines: int = 20,
    top_terms_per_cuisine: int = 30,
    min_shared_terms: int = 5,
) -> None:
    print("[PLOT] Cuisine similarity network map")
    df_top = safe_read_csv(top_features_path)
    if df_top is None:
        print("[WARN] No top_features_per_cuisine.csv; skipping network.")
        return

    required_cols = {"cuisine", "feature", "rank"}
    if not required_cols.issubset(df_top.columns):
        print(f"[WARN] top_features file missing columns {required_cols}; skipping network.")
        return

    # Select top cuisines and their top terms
    selected_cuisines = cuisine_counts.head(n_cuisines).index.tolist()
    df_top = df_top[
        (df_top["cuisine"].isin(selected_cuisines))
        & (df_top["rank"] <= top_terms_per_cuisine)
    ]
    if df_top.empty:
        print("[WARN] No data for selected cuisines/terms; skipping network.")
        return

    # Group features by cuisine
    cuisine_to_terms = (
        df_top.groupby("cuisine")["feature"]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )
    cuisines = list(cuisine_to_terms.keys())

    # Build graph
    G = nx.Graph()
    for c in cuisines:
        G.add_node(c, recipes=int(cuisine_counts.get(c, 0)))

    for i in range(len(cuisines)):
        for j in range(i + 1, len(cuisines)):
            ci, cj = cuisines[i], cuisines[j]
            shared = cuisine_to_terms[ci] & cuisine_to_terms[cj]
            if len(shared) >= min_shared_terms:
                G.add_edge(ci, cj, weight=len(shared))

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        print("[WARN] Graph has no edges after thresholding; skipping network.")
        return

    # Layout
    pos = nx.spring_layout(G, seed=RANDOM_STATE, k=0.6, weight="weight")

    # Node sizes based on # recipes
    node_sizes = []
    for n in G.nodes():
        rec = G.nodes[n].get("recipes", 0)
        node_sizes.append(100 + 3 * np.sqrt(rec))

    # Edge widths based on number of shared ingredients
    edge_widths = [1 + 0.4 * G[u][v]["weight"] for u, v in G.edges()]

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        edge_color="lightgray",
        alpha=0.8,
    )
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color="tab:blue",
        alpha=0.85,
    )
    nx.draw_networkx_labels(G, pos, font_size=9)

    plt.title("Cuisine similarity network (shared top ingredients)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  -> saved to {out_path}")


# -------------------------
# Main
# -------------------------
def main():
    print(f"[INFO] Base dir: {BASE_DIR}")
    print(f"[INFO] Reading recipes from: {INPUT_CSV}")

    
    df = pd.read_parquet(INPUT_CSV)

    if "cuisine" not in df.columns:
        raise ValueError("Input CSV must contain a 'cuisine' column.")

    # Clean cuisine
    df["cuisine"] = (
        df["cuisine"]
        .astype(str)
        .str.strip()
        .str.lower()
    )
    df = df.dropna(subset=["cuisine"]).copy()

    cuisine_counts = df["cuisine"].value_counts()

    # 1) Cuisine distribution
    plot_cuisine_distribution(
        df, VIZ_DIR + "/cuisine_distribution.png", top_n=20
    )

    # 2) Top ingredients heatmap
    plot_top_ingredients_heatmap(
        REPORTS_DIR + "/ top_features_per_cuisine.csv",
        cuisine_counts,
        VIZ_DIR + "/ top_ingredients_heatmap.png",
        n_cuisines=10,
        top_per_cuisine=10,
    )

    # 3) Model macro-F1 comparison
    plot_model_performance(
        VIZ_DIR + "/model_macro_f1.png"
    )

    # 4) Cuisine similarity network
    plot_cuisine_network(
        REPORTS_DIR + "/ top_features_per_cuisine.csv",
        cuisine_counts,
        VIZ_DIR + "/ cuisine_network.png",
        n_cuisines=20,
        top_terms_per_cuisine=30,
        min_shared_terms=5,
    )

    print("[INFO] All visualizations generated in:", VIZ_DIR)


if __name__ == "__main__":
    main()
