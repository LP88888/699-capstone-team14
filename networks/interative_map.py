#!/usr/bin/env python
"""
Build an interactive cuisine/ingredient network HTML.

Modes:
1. Default Mode: Plot cuisine network with shared ingredients as edges
2. Fusion Mode: Visualize ingredient overlap between two specific cuisines

Reads:
    reports/phase2/top_features_per_cuisine.csv
    reports/phase2/network_nodes_centrality.csv (for fusion mode)

Writes:
    reports/viz/cuisine_network_pyvis.html (default mode)
    reports/viz/fusion_{cuisine_a}_{cuisine_b}.html (fusion mode)
"""

import argparse
from pathlib import Path
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from pyvis.network import Network

# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # adjust if needed
REPORTS_DIR = BASE_DIR / "reports" / "phase2"
VIZ_DIR = BASE_DIR / "reports" / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

TOP_FEAT_PATH = REPORTS_DIR / "top_features_per_cuisine.csv"
NETWORK_NODES_PATH = REPORTS_DIR / "network_nodes_centrality.csv"
INGREDIENT_COUNTS_PATH = REPORTS_DIR / "ingredient_counts.csv"
OUT_HTML = VIZ_DIR / "cuisine_network_pyvis.html"

RANDOM_STATE = 42

# -------------------------
# Parameters
# -------------------------
N_CUISINES = 25           # how many cuisines to include (top by frequency in top_features file)
TOP_TERMS_PER_CUISINE = 40
MIN_SHARED_TERMS = 5      # create an edge only if at least this many shared top ingredients

# Fusion Mode Colors
COLOR_CUISINE_A = "#2ecc71"   # Green for first cuisine
COLOR_CUISINE_B = "#e74c3c"   # Red for second cuisine
COLOR_BRIDGE = "#f1c40f"      # Gold for bridge ingredients
COLOR_EDGE_A = "#27ae60"      # Darker green for cuisine A edges
COLOR_EDGE_B = "#c0392b"      # Darker red for cuisine B edges
COLOR_EDGE_BRIDGE = "#f39c12" # Orange for bridge edges


def load_top_features() -> pd.DataFrame:
    """Load top features per cuisine."""
    if not TOP_FEAT_PATH.exists():
        raise FileNotFoundError(f"File not found: {TOP_FEAT_PATH}")
    
    df = pd.read_csv(TOP_FEAT_PATH)
    required_cols = {"cuisine", "feature", "rank"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Expected columns {required_cols} in {TOP_FEAT_PATH}, "
            f"got {set(df.columns)}"
        )
    return df


def load_ingredient_network_data() -> tuple:
    """Load ingredient network centrality data."""
    if NETWORK_NODES_PATH.exists():
        nodes_df = pd.read_csv(NETWORK_NODES_PATH)
        centrality = dict(zip(nodes_df["ingredient"], nodes_df["betweenness"]))
        degree = dict(zip(nodes_df["ingredient"], nodes_df["degree"]))
    else:
        print(f"[WARN] Network nodes file not found: {NETWORK_NODES_PATH}")
        centrality = {}
        degree = {}
    
    return centrality, degree


def get_cuisine_top_ingredients(df: pd.DataFrame, cuisine: str, n: int = 20) -> set:
    """Get top N ingredients for a specific cuisine."""
    cuisine_lower = cuisine.lower().strip()
    
    # Find matching cuisine (case-insensitive)
    df_cuisine = df[df["cuisine"].str.lower().str.strip() == cuisine_lower]
    
    if df_cuisine.empty:
        available = df["cuisine"].unique()[:10]
        raise ValueError(
            f"Cuisine '{cuisine}' not found. Available: {list(available)}..."
        )
    
    # Get top N by rank
    top_df = df_cuisine[df_cuisine["rank"] <= n]
    return set(top_df["feature"].astype(str).tolist())


def find_bridge_ingredients(
    top_a: set, 
    top_b: set, 
    all_ingredients: set,
    centrality: dict
) -> list:
    """
    Find bridge ingredients that connect both cuisine ingredient sets.
    
    Bridge ingredients are those that appear in both top sets OR
    have high centrality and connect to both sets.
    """
    # Direct overlap (shared ingredients)
    shared = top_a & top_b
    
    # For simplicity, we consider shared ingredients as bridges
    # In a full implementation, you'd check graph edges
    bridges = []
    for ing in shared:
        cent = centrality.get(ing, 0)
        bridges.append((ing, cent))
    
    # Sort by centrality
    bridges.sort(key=lambda x: x[1], reverse=True)
    return bridges


def build_fusion_network(
    cuisine_a: str,
    cuisine_b: str,
    top_n: int = 20,
    output_path: Path = None
) -> Path:
    """
    Build a fusion mode network showing ingredient overlap between two cuisines.
    
    Args:
        cuisine_a: First cuisine name (displayed in green)
        cuisine_b: Second cuisine name (displayed in red)
        top_n: Number of top ingredients per cuisine
        output_path: Optional output path for HTML
        
    Returns:
        Path to generated HTML file
    """
    print(f"[FUSION MODE] Building fusion network for: {cuisine_a} + {cuisine_b}")
    
    # Load data
    df = load_top_features()
    centrality, degree = load_ingredient_network_data()
    
    # Get top ingredients for each cuisine
    top_a = get_cuisine_top_ingredients(df, cuisine_a, top_n)
    top_b = get_cuisine_top_ingredients(df, cuisine_b, top_n)
    
    print(f"[FUSION] {cuisine_a}: {len(top_a)} top ingredients")
    print(f"[FUSION] {cuisine_b}: {len(top_b)} top ingredients")
    
    # Find bridge ingredients (shared between both)
    bridges = top_a & top_b
    print(f"[FUSION] Bridge ingredients (shared): {len(bridges)}")
    
    # Exclusive ingredients for each cuisine
    exclusive_a = top_a - bridges
    exclusive_b = top_b - bridges
    
    # Create network
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#1a1a2e",  # Dark background for contrast
        font_color="#ffffff",
        notebook=False,
        directed=False,
    )
    
    # Configure physics
    net.barnes_hut(
        gravity=-30000, 
        central_gravity=0.5, 
        spring_length=200, 
        spring_strength=0.03
    )
    
    # Add cuisine label nodes (larger, centered)
    net.add_node(
        f"__cuisine_{cuisine_a}__",
        label=cuisine_a.upper(),
        color=COLOR_CUISINE_A,
        size=50,
        font={"size": 24, "color": "#ffffff", "face": "arial black"},
        shape="box",
        title=f"<b>{cuisine_a}</b><br>Top {top_n} ingredients shown",
    )
    
    net.add_node(
        f"__cuisine_{cuisine_b}__",
        label=cuisine_b.upper(),
        color=COLOR_CUISINE_B,
        size=50,
        font={"size": 24, "color": "#ffffff", "face": "arial black"},
        shape="box",
        title=f"<b>{cuisine_b}</b><br>Top {top_n} ingredients shown",
    )
    
    # Add ingredient nodes for cuisine A (green)
    for ing in exclusive_a:
        cent = centrality.get(ing, 0)
        deg = degree.get(ing, 1)
        size = 10 + min(deg * 0.5, 30)
        
        net.add_node(
            ing,
            label=ing,
            color=COLOR_CUISINE_A,
            size=size,
            title=f"<b>{ing}</b><br>Cuisine: {cuisine_a}<br>Centrality: {cent:.4f}<br>Degree: {deg}",
            font={"size": 12, "color": "#ffffff"},
        )
        
        # Connect to cuisine node
        net.add_edge(
            f"__cuisine_{cuisine_a}__",
            ing,
            color=COLOR_EDGE_A,
            width=1,
        )
    
    # Add ingredient nodes for cuisine B (red)
    for ing in exclusive_b:
        cent = centrality.get(ing, 0)
        deg = degree.get(ing, 1)
        size = 10 + min(deg * 0.5, 30)
        
        net.add_node(
            ing,
            label=ing,
            color=COLOR_CUISINE_B,
            size=size,
            title=f"<b>{ing}</b><br>Cuisine: {cuisine_b}<br>Centrality: {cent:.4f}<br>Degree: {deg}",
            font={"size": 12, "color": "#ffffff"},
        )
        
        # Connect to cuisine node
        net.add_edge(
            f"__cuisine_{cuisine_b}__",
            ing,
            color=COLOR_EDGE_B,
            width=1,
        )
    
    # Add bridge ingredients (gold) - connected to BOTH cuisines
    for ing in bridges:
        cent = centrality.get(ing, 0)
        deg = degree.get(ing, 1)
        size = 15 + min(deg * 0.5, 35)  # Slightly larger than exclusive
        
        net.add_node(
            ing,
            label=ing,
            color=COLOR_BRIDGE,
            size=size,
            title=(
                f"<b>{ing}</b><br>"
                f"<span style='color:{COLOR_BRIDGE}'>★ BRIDGE INGREDIENT ★</span><br>"
                f"Shared by: {cuisine_a} & {cuisine_b}<br>"
                f"Centrality: {cent:.4f}<br>Degree: {deg}"
            ),
            font={"size": 14, "color": "#000000", "face": "arial black"},
            borderWidth=3,
            borderWidthSelected=5,
        )
        
        # Connect to both cuisine nodes
        net.add_edge(
            f"__cuisine_{cuisine_a}__",
            ing,
            color=COLOR_EDGE_BRIDGE,
            width=2,
        )
        net.add_edge(
            f"__cuisine_{cuisine_b}__",
            ing,
            color=COLOR_EDGE_BRIDGE,
            width=2,
        )
    
    # Generate output path
    if output_path is None:
        safe_a = cuisine_a.lower().replace(" ", "_")
        safe_b = cuisine_b.lower().replace(" ", "_")
        output_path = VIZ_DIR / f"fusion_{safe_a}_{safe_b}.html"
    
    # Add legend as HTML
    legend_html = f"""
    <div style="position: fixed; top: 10px; left: 10px; background: rgba(0,0,0,0.8); 
                padding: 15px; border-radius: 10px; color: white; font-family: Arial;">
        <h3 style="margin: 0 0 10px 0;">🍳 Fusion Mode: {cuisine_a} + {cuisine_b}</h3>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 15px; height: 15px; 
                        background: {COLOR_CUISINE_A}; border-radius: 50%; margin-right: 8px;"></span>
            {cuisine_a} ingredients ({len(exclusive_a)})
        </div>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 15px; height: 15px; 
                        background: {COLOR_CUISINE_B}; border-radius: 50%; margin-right: 8px;"></span>
            {cuisine_b} ingredients ({len(exclusive_b)})
        </div>
        <div style="margin: 5px 0;">
            <span style="display: inline-block; width: 15px; height: 15px; 
                        background: {COLOR_BRIDGE}; border-radius: 50%; margin-right: 8px;"></span>
            Bridge ingredients ({len(bridges)})
        </div>
    </div>
    """
    
    print(f"[FUSION] Network: {len(net.nodes)} nodes, {len(net.edges)} edges")
    print(f"[FUSION] Writing to: {output_path}")
    
    # Write HTML with legend
    net.write_html(str(output_path))
    
    # Inject legend into HTML
    with open(output_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    # Insert legend before closing body tag
    html_content = html_content.replace("</body>", f"{legend_html}</body>")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"[FUSION] ✓ Fusion network saved: {output_path}")
    
    return output_path


def main_default():
    """Default mode: Build cuisine network."""
    print(f"[INFO] Reading top features from: {TOP_FEAT_PATH}")
    
    df = load_top_features()
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

    # Build PyVis network
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
        notebook=False,
        directed=False,
    )

    net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=150, spring_strength=0.02)

    # Add cuisine nodes
    for cuisine in selected_cuisines:
        terms = cuisine_to_terms.get(cuisine, set())
        freq = int(cuisine_counts.get(cuisine, 0))
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
                shared_preview = ", ".join(shared_list[:20])
                if len(shared_list) > 20:
                    shared_preview += ", ..."

                title = (
                    f"<b>{ci}</b> – <b>{cj}</b><br>"
                    f"shared top ingredients: {len(shared)}<br>"
                    f"{shared_preview}"
                )

                net.add_edge(ci, cj, value=len(shared), title=title)

    if len(net.nodes) == 0 or len(net.edges) == 0:
        print("[WARN] Network ended up with no nodes or edges; check thresholds.")
    else:
        print(f"[INFO] Network has {len(net.nodes)} nodes and {len(net.edges)} edges.")

    print(f"[INFO] Writing HTML to: {OUT_HTML}")
    net.write_html(str(OUT_HTML))


def main():
    """Main entry point with argument parsing for fusion mode."""
    parser = argparse.ArgumentParser(
        description="Build interactive cuisine/ingredient network visualizations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default mode: Build full cuisine network
  python interative_map.py
  
  # Fusion mode: Visualize Italian + Thai fusion opportunities
  python interative_map.py --fusion Italian Thai
  
  # Fusion mode with more ingredients
  python interative_map.py --fusion Mexican Japanese --top-n 30
        """
    )
    
    parser.add_argument(
        "--fusion",
        nargs=2,
        metavar=("CUISINE_A", "CUISINE_B"),
        help="Enable fusion mode: specify two cuisine names to compare"
    )
    
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top ingredients per cuisine in fusion mode (default: 20)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output path for HTML file"
    )
    
    args = parser.parse_args()
    
    if args.fusion:
        cuisine_a, cuisine_b = args.fusion
        output_path = Path(args.output) if args.output else None
        build_fusion_network(
            cuisine_a=cuisine_a,
            cuisine_b=cuisine_b,
            top_n=args.top_n,
            output_path=output_path
        )
    else:
        main_default()


if __name__ == "__main__":
    main()
