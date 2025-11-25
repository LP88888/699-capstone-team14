#!/usr/bin/env python
"""
Build an interactive cuisine network HTML where:
- Nodes = cuisines
- Edges = shared top ingredients between cuisines

Reads:
    reports/phase2/top_features_per_cuisine.csv

Writes:
    reports/viz/cuisine_network_pyvis.html
"""

from pathlib import Path
import numpy as np
import pandas as pd

from pyvis.network import Network

# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # adjust if needed
REPORTS_DIR = BASE_DIR / "reports" / "phase2"
VIZ_DIR = BASE_DIR / "reports" / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

TOP_FEAT_PATH = REPORTS_DIR / "top_features_per_cuisine.csv"
OUT_HTML = VIZ_DIR / "cuisine_network_pyvis.html"

RANDOM_STATE = 42

# -------------------------
# Parameters
# -------------------------
N_CUISINES = 25           # how many cuisines to include (top by frequency in top_features file)
TOP_TERMS_PER_CUISINE = 40
MIN_SHARED_TERMS = 5      # create an edge only if at least this many shared top ingredients


def main():
    print(f"[INFO] Reading top features from: {TOP_FEAT_PATH}")
    if not TOP_FEAT_PATH.exists():
        raise FileNotFoundError(f"File not found: {TOP_FEAT_PATH}")

    df = pd.read_csv(TOP_FEAT_PATH)

    required_cols = {"cuisine", "feature", "rank"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Expected columns {required_cols} in {TOP_FEAT_PATH}, "
            f"got {set(df.columns)}"
        )

    # Use frequencies in this file as a proxy for "importance"/coverage
    cuisine_counts = df["cuisine"].value_counts()

    # Select top cuisines by count
    selected_cuisines = cuisine_counts.head(N_CUISINES).index.tolist()
    print(f"[INFO] Using {len(selected_cuisines)} cuisines: {selected_cuisines}")

    # Filter to selected cuisines + top terms
    df_sel = df[
        (df["cuisine"].isin(selected_cuisines)) &
        (df["rank"] <= TOP_TERMS_PER_CUISINE)
    ].copy()

    if df_sel.empty:
        raise RuntimeError("No data after filtering by cuisine + rank; check parameters.")

    # Build mapping: cuisine -> set of top ingredients
    cuisine_to_terms = (
        df_sel.groupby("cuisine")["feature"]
        .apply(lambda s: set(str(x) for x in s))
        .to_dict()
    )

    # -------------------------
    # Build PyVis network
    # -------------------------
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
        notebook=False,
        directed=False,
    )

    # A bit nicer physics
    net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=150, spring_strength=0.02)

    # Add cuisine nodes
    for cuisine in selected_cuisines:
        terms = cuisine_to_terms.get(cuisine, set())
        num_terms = len(terms)
        freq = int(cuisine_counts.get(cuisine, 0))

        # Node size scaled by count in top_features file
        value = max(1, freq)

        title_html = (
            f"<b>{cuisine}</b><br>"
            f"features in file: {freq}<br>"
            f"top terms (sample): {', '.join(sorted(list(terms))[:15])}"
        )

        net.add_node(
            cuisine,
            label=cuisine,
            title=title_html,
            value=value,
        )

    # Add edges between cuisines with enough shared ingredients
    cuisines = list(cuisine_to_terms.keys())

    rng = np.random.RandomState(RANDOM_STATE)

    for i in range(len(cuisines)):
        for j in range(i + 1, len(cuisines)):
            ci, cj = cuisines[i], cuisines[j]
            terms_i = cuisine_to_terms.get(ci, set())
            terms_j = cuisine_to_terms.get(cj, set())
            if not terms_i or not terms_j:
                continue

            shared = terms_i & terms_j
            if len(shared) >= MIN_SHARED_TERMS:
                shared_list = sorted(list(shared))
                # Limit the hover text length
                shared_preview = ", ".join(shared_list[:20])
                if len(shared_list) > 20:
                    shared_preview += ", ..."

                title = (
                    f"<b>{ci}</b> – <b>{cj}</b><br>"
                    f"shared top ingredients: {len(shared)}<br>"
                    f"{shared_preview}"
                )

                # edge value scales line thickness
                net.add_edge(
                    ci,
                    cj,
                    value=len(shared),
                    title=title,
                )

    # If the graph ended up empty, warn
    if len(net.nodes) == 0 or len(net.edges) == 0:
        print("[WARN] Network ended up with no nodes or edges; check thresholds.")
    else:
        print(f"[INFO] Network has {len(net.nodes)} nodes and {len(net.edges)} edges.")

    # Write HTML (no auto-open; you can open in browser)
    print(f"[INFO] Writing HTML to: {OUT_HTML}")
    net.write_html(str(OUT_HTML))


if __name__ == "__main__":
    main()
