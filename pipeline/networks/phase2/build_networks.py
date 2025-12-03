
from __future__ import annotations

import config as cfg
import networkx as nx
import pandas as pd

OUT_DIR = cfg.PHASE_2_REPORTS_PATH


PMI_FILE = OUT_DIR / "pairings_pmi_global.csv"
MIN_PMI = 1.0         # edge threshold
MIN_COUNT = 15        # additional count filter


def main():
    print(f"[INFO] Loading PMI pairs from {PMI_FILE}")
    df_pmi = pd.read_csv(PMI_FILE)

    df_pmi_filt = df_pmi[
        (df_pmi["pmi"] >= MIN_PMI) & (df_pmi["count"] >= MIN_COUNT)
    ].copy()
    print(f"[INFO] Using {len(df_pmi_filt)} edges with PMI>={MIN_PMI} and count>={MIN_COUNT}")

    G = nx.Graph()
    for _, row in df_pmi_filt.iterrows():
        a = row["ingredient_a"]
        b = row["ingredient_b"]
        w = row["pmi"]
        c = row["count"]
        G.add_edge(a, b, weight=w, count=c)

    print(f"[INFO] Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Centrality measures
    deg_c = nx.degree_centrality(G)
    bet_c = nx.betweenness_centrality(G)
    close_c = nx.closeness_centrality(G)

    nodes_rows = []
    for n in G.nodes():
        nodes_rows.append(
            {
                "ingredient": n,
                "degree_centrality": deg_c.get(n, 0.0),
                "betweenness_centrality": bet_c.get(n, 0.0),
                "closeness_centrality": close_c.get(n, 0.0),
                "degree": G.degree(n),
            }
        )
    df_nodes = pd.DataFrame(nodes_rows)
    out_nodes = OUT_DIR / "network_nodes_centrality.csv"
    df_nodes.to_csv(out_nodes, index=False)
    print(f"[INFO] Wrote node centrality table to {out_nodes}")

    # Edges table
    edges_rows = []
    for u, v, data in G.edges(data=True):
        edges_rows.append(
            {
                "ingredient_a": u,
                "ingredient_b": v,
                "pmi": data.get("weight", 0.0),
                "count": data.get("count", 0),
            }
        )
    df_edges = pd.DataFrame(edges_rows)
    out_edges = OUT_DIR / "network_edges.csv"
    df_edges.to_csv(out_edges, index=False)
    print(f"[INFO] Wrote edges table to {out_edges}")

    # Save graph in GEXF for Gephi
    gexf_path = OUT_DIR / "ingredient_network.gexf"
    nx.write_gexf(G, gexf_path)
    print(f"[INFO] Wrote graph to {gexf_path}")


