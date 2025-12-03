# pmi_pairings.py
from __future__ import annotations
import ast
from collections import Counter, defaultdict
from itertools import combinations
from math import log

import config as cfg
from typing import List

import pandas as pd

DATA_PATH = cfg.CLEANED_DATA_PATH
OUT_DIR = cfg.PHASE_2_REPORTS_PATH


MIN_PAIR_COUNT_GLOBAL = 5      # min co-occur count to keep globally
MIN_PAIR_COUNT_PER_CUISINE = 3  # per-cuisine pairs
MIN_ING_FREQ = 10               # drop ultra-rare ingredients




def main():
    print(f"[INFO] Loading data from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(df.head())
    if "ingredients_clean" not in df.columns:
        raise ValueError("Expected 'ingredients_clean' column. Run cleaning.py first.")


    print(df.head())
    # Remove rows with empty ingredient lists
    df = df[df["ingredients_clean"].apply(len) > 1].copy()
    print(f"[INFO] {len(df)} recipes with >=2 ingredients")

    ing_counter = Counter()
    for lst in df["ingredients_clean"]:
        ing_counter.update(lst)

    # Filter rare ingredients
    keep_ings = {ing for ing, c in ing_counter.items() if c >= MIN_ING_FREQ}
    print(f"[INFO] Keeping {len(keep_ings)} ingredients with freq >= {MIN_ING_FREQ}")

    # Rebuild with filtered ingredients
    df["ingredients_filt"] = df["ingredients_clean"].apply(
        lambda lst: [x for x in lst if x in keep_ings]
    )
    df = df[df["ingredients_filt"].apply(len) > 1].copy()
    print(f"[INFO] {len(df)} recipes after ingredient freq filtering")

    # Recompute ingredient counts on filtered
    ing_counter = Counter()
    for lst in df["ingredients_filt"]:
        ing_counter.update(lst)

    total_recipes = len(df)
    total_ing_occ = sum(ing_counter.values())

    pair_counter = Counter()
    for lst in df["ingredients_filt"]:
        for a, b in combinations(sorted(set(lst)), 2):
            pair_counter[(a, b)] += 1

    rows = []
    for (a, b), cnt in pair_counter.items():
        if cnt < MIN_PAIR_COUNT_GLOBAL:
            continue
        pa = ing_counter[a] / total_recipes
        pb = ing_counter[b] / total_recipes
        pab = cnt / total_recipes
        if pab <= 0 or pa <= 0 or pb <= 0:
            continue
        pmi = log(pab / (pa * pb))
        rows.append(
            {
                "ingredient_a": a,
                "ingredient_b": b,
                "count": cnt,
                "pmi": pmi,
            }
        )

    if not rows:
        print(
            f"[WARN] No global PMI pairs met the thresholds "
            f"(MIN_PAIR_COUNT_GLOBAL={MIN_PAIR_COUNT_GLOBAL}, "
            f"MIN_ING_FREQ={MIN_ING_FREQ}). "
            "Writing an empty file."
        )
        df_pmi_global = pd.DataFrame(
            columns=["ingredient_a", "ingredient_b", "count", "pmi"]
        )
    else:
        df_pmi_global = pd.DataFrame(rows).sort_values("pmi", ascending=False)

    out_global = OUT_DIR / "pairings_pmi_global.csv"
    df_pmi_global.to_csv(out_global, index=False)
    print(f"[INFO] Wrote global PMI pairings to {out_global}")


    if "cuisine" not in df.columns:
        raise ValueError("Expected 'cuisine' column in data.")

    rows_c = []
    for cuisine, sub in df.groupby("cuisine"):
        if len(sub) < 30:
            # too small for stable PMI
            continue

        ing_c = Counter()
        for lst in sub["ingredients_filt"]:
            ing_c.update(lst)

        pair_c = Counter()
        for lst in sub["ingredients_filt"]:
            for a, b in combinations(sorted(set(lst)), 2):
                pair_c[(a, b)] += 1

        n_rec_c = len(sub)
        for (a, b), cnt in pair_c.items():
            if cnt < MIN_PAIR_COUNT_PER_CUISINE:
                continue
            pa = ing_c[a] / n_rec_c
            pb = ing_c[b] / n_rec_c
            pab = cnt / n_rec_c
            if pab <= 0 or pa <= 0 or pb <= 0:
                continue
            pmi = log(pab / (pa * pb))
            rows_c.append(
                {
                    "cuisine": cuisine,
                    "ingredient_a": a,
                    "ingredient_b": b,
                    "count": cnt,
                    "pmi": pmi,
                }
            )

    
    if not rows_c:
        print(
            f"[WARN] No per-cuisine PMI pairs met thresholds "
            f"(MIN_PAIR_COUNT_PER_CUISINE={MIN_PAIR_COUNT_PER_CUISINE}, "
            f"MIN_ING_FREQ={MIN_ING_FREQ}). Writing an empty file."
        )
        df_pmi_cuisine = pd.DataFrame(
            columns=["cuisine", "ingredient_a", "ingredient_b", "count", "pmi"]
        )
    else:
        df_pmi_cuisine = pd.DataFrame(rows_c).sort_values(
            ["cuisine", "pmi"], ascending=[True, False]
        )

    out_c = OUT_DIR / "pairings_pmi_by_cuisine.csv"
    df_pmi_cuisine.to_csv(out_c, index=False)
    print(f"[INFO] Wrote per-cuisine PMI pairings to {out_c}")



if __name__ == "__main__":
    main()
