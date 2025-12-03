
from __future__ import annotations
import ast
import config as cfg
from typing import List

import numpy as np
import pandas as pd

DATA_PATH = cfg.CLEANED_DATA_PATH
OUT_DIR = cfg.PHASE_2_REPORTS_PATH


NODE_CENTRALITY_FILE = OUT_DIR / "network_nodes_centrality.csv"


def parse_ingredients(val) -> List[str]:
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str):
        s = val.strip()
        try:
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
        except Exception:
            pass
        return [s]
    return [str(val)]


def main():
    print(f"[INFO] Loading data from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    df["ingredients_clean"] = df["ingredients_clean"].apply(parse_ingredients)
    df["ingredients_clean"] = df["ingredients_clean"].apply(
        lambda lst: [x.strip().lower() for x in lst if x and isinstance(x, str)]
    )

    print(f"[INFO] Loading node centrality from {NODE_CENTRALITY_FILE}")
    df_nodes = pd.read_csv(NODE_CENTRALITY_FILE)
    cent = df_nodes.set_index("ingredient")

    def agg_centrality(ings):
        rows = cent.loc[[i for i in ings if i in cent.index]]
        if rows.empty:
            return {
                "deg_mean": 0.0,
                "deg_max": 0.0,
                "betw_mean": 0.0,
                "betw_max": 0.0,
                "close_mean": 0.0,
                "close_max": 0.0,
            }
        return {
            "deg_mean": float(rows["degree_centrality"].mean()),
            "deg_max": float(rows["degree_centrality"].max()),
            "betw_mean": float(rows["betweenness_centrality"].mean()),
            "betw_max": float(rows["betweenness_centrality"].max()),
            "close_mean": float(rows["closeness_centrality"].mean()),
            "close_max": float(rows["closeness_centrality"].max()),
        }

    features = df["ingredients_clean"].apply(agg_centrality).apply(pd.Series)
    features["recipe_index"] = df.index

    out_f = OUT_DIR / "graph_features_table.csv"
    features.to_csv(out_f, index=False)
    print(f"[INFO] Wrote graph-based recipe features to {out_f}")


if __name__ == "__main__":
    main()
