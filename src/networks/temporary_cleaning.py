"""
cleaning.py

Reads the merged recipe dataset and produces a cleaned, encoded version
suitable for:

- Phase 1 modeling (TF-IDF, classification, clustering) using
  `ingredients_text` and `cuisine_cleaned_str`.

- Phase 2 network / PMI analysis using `ingredients_clean`
  (a list of cleaned ingredient tokens).

Output: data/encoded/combined_raw_datasets_with_cuisine_clean_encoded.parquet
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RAW_PATH = Path("data/encoded/combined_raw_datasets_with_cuisine_encoded.parquet")
ENCODED_DIR = Path("data/encoded")
ENCODED_DIR.mkdir(parents=True, exist_ok=True)

OUT_PARQUET = ENCODED_DIR / "combined_raw_datasets_with_cuisine_clean_encoded.parquet"
OUT_CUISINE_MAP = ENCODED_DIR / "cuisine_label_mapping.csv"


# ---------------------------------------------------------------------
# Cuisine normalization
# ---------------------------------------------------------------------

CUISINE_NORMALIZATION_MAP = {
    # basic canonical names
    "american": "american",
    "usa": "american",
    "united states": "american",
    "united states of america": "american",

    "british": "british",
    "english": "british",
    "unitedkingdom": "british",
    "uk": "british",

    "chinese": "chinese",
    "sichuan": "chinese",
    "cantonese": "chinese",

    "indian": "indian",
    "north indian recipes": "indian",
    "south indian recipes": "indian",
    "bengali recipes": "indian",
    "maharashtrian recipes": "indian",
    "kerala recipes": "indian",
    "goan recipes": "indian",
    "north east india recipes": "indian",
    "uttarakhand-north kumaon": "indian",

    "italian": "italian",
    "italian recipes": "italian",

    "caribbean": "caribbean",
    "carribean": "caribbean",

    "mexican": "mexican",
    "southwestern": "mexican",

    "greek": "greek",
    "mediterranean": "mediterranean",
    "greek, mediterranean": "mediterranean",

    "french": "french",
    "german": "german",
    "hungarian": "hungarian",
    "thai": "thai",
    "korean": "korean",
    "japanese": "japanese",
    "vietnamese": "vietnamese",
    "spanish": "spanish",
    "turkish": "turkish",
    "middle eastern": "middle eastern",
    "lebanese": "middle eastern",
    "israeli": "middle eastern",
    "persian": "middle eastern",

    "african": "african",
    "brazilian": "brazilian",
    "canadian": "canadian",
    "canada": "canadian",
    "scandinavian": "scandinavian",
    "russian": "russian",

    # multi-label or “X, Y” style – pick the first as canonical
    "american, asian": "american",
    "american, italian": "american",
    "american, mexican": "american",
    "american, cajun & creole": "american",
    "american, hawaiian": "american",
    "american, latin": "american",
    "cajun & creole": "cajun & creole",
    "cajun & creole, american": "cajun & creole",
    "southern & soul food": "southern & soul food",
    "southern & soul food, american": "southern & soul food",

    # catch-all
    "continental": "continental",
    "fusion": "fusion",
    "other": "other",
}


def normalize_cuisine(raw: Any) -> str:
    """
    Normalize raw cuisine strings into canonical lowercase labels.
    """
    if pd.isna(raw):
        return "unknown"

    s = str(raw).strip().lower()
    if not s:
        return "unknown"

    # If there is a comma-separated list, default to first element as primary
    if "," in s and s not in CUISINE_NORMALIZATION_MAP:
        first = s.split(",")[0].strip()
        if first in CUISINE_NORMALIZATION_MAP:
            return CUISINE_NORMALIZATION_MAP[first]
        return first

    # Map directly if known
    if s in CUISINE_NORMALIZATION_MAP:
        return CUISINE_NORMALIZATION_MAP[s]

    return s


# ---------------------------------------------------------------------
# Ingredient parsing
# ---------------------------------------------------------------------

def parse_ingredients(val: Any) -> List[str]:
    """
    Convert raw ingredient representation into a *flat, deduplicated,
    lowercase list of tokens*.

    This is for PHASE 2 (PMI + network). Phase 1 will use ingredients_text.

    Handles:
      - list of strings
      - list of lists
      - string like "egg, milk, sugar"
      - string like "egg milk sugar"
      - string like "['egg', 'milk']"
    """

    # Already a Python list
    if isinstance(val, list):
        out: List[str] = []
        for x in val:
            if isinstance(x, str):
                # Allow both comma- and space-separated content inside
                tokens = x.replace(",", " ").split()
                out.extend(tokens)
            elif isinstance(x, list):
                out.extend([t for t in x if isinstance(t, str)])
        return sorted(
            set(t.strip().lower() for t in out if isinstance(t, str) and t.strip())
        )

    # String: could be text or Python-list literal
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []

        # Looks like a Python list → try literal_eval
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                return parse_ingredients(parsed)
            except Exception:
                # Fall back to splitting
                pass

        tokens = s.replace(",", " ").split()
        return sorted(
            set(t.strip().lower() for t in tokens if t.strip())
        )

    # Anything else → empty
    return []


# ---------------------------------------------------------------------
# Main cleaning routine
# ---------------------------------------------------------------------

def main() -> None:
    print(f"[INFO] Loading raw combined data from {RAW_PATH}")

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_PATH}")

    # Read CSV (change to read_parquet if your raw is parquet)
    df = pd.read_parquet(RAW_PATH)

    print(f"[INFO] Raw shape: {df.shape}")

    # Basic drops: need cuisine + some form of ingredients
    possible_ing_cols = ["ingredients", "inferred_ingredients", "encoded_ingredients"]
    has_ing = [c for c in possible_ing_cols if c in df.columns]

    if not has_ing:
        raise ValueError(
            "Expected at least one of 'ingredients', 'inferred_ingredients', "
            "'encoded_ingredients' in the input."
        )

    if "cuisine" not in df.columns:
        raise ValueError("Expected 'cuisine' column in the input data.")

    df = df.dropna(subset=["cuisine"])
    print(f"[INFO] After dropping rows with missing cuisine: {df.shape}")

    # -----------------------------------------------------------------
    # Cuisine cleaning
    # -----------------------------------------------------------------
    df["cuisine_raw"] = df["cuisine"].astype(str)

    df["cuisine_cleaned_str"] = df["cuisine_raw"].apply(normalize_cuisine)

    # Encode cuisines to integers for modeling
    le = LabelEncoder()
    df["cuisine_encoded"] = le.fit_transform(df["cuisine_cleaned_str"])

    # Save mapping for reference
    cuisine_map = pd.DataFrame(
        {
            "cuisine_cleaned_str": le.classes_,
            "cuisine_encoded": np.arange(len(le.classes_)),
        }
    )
    cuisine_map.to_csv(OUT_CUISINE_MAP, index=False)
    print(f"[INFO] Wrote cuisine label mapping to {OUT_CUISINE_MAP}")

    # -----------------------------------------------------------------
    # Ingredient cleaning for both phases
    # -----------------------------------------------------------------

    def extract_raw_ingredient_text(row):
        """
        Build a plain text version of ingredients for Phase 1.

        Priority:
            1. inferred_ingredients (list or string)
            2. ingredients (list or string)
            3. encoded_ingredients (string-only fallback)

        Output:
            lowercase string suitable for TF-IDF (Phase 1)
        """
        preferred_cols = ["inferred_ingredients", "ingredients"]

        for col in preferred_cols:
            if col in row:
                val = row[col]

                if isinstance(val, list):
                    # join list into a single space-separated string
                    return " ".join(str(x) for x in val).lower().strip()

                if isinstance(val, str) and val.strip():
                    return val.lower().strip()

        # fallback: encoded_ingredients may be a list of ints or weird artifacts
        if "encoded_ingredients" in row:
            val = row["encoded_ingredients"]
            if isinstance(val, str):
                return val.lower().strip()

            if isinstance(val, list):
                return " ".join(str(x) for x in val).lower().strip()

        # If absolutely nothing exists:
        return ""


    # Phase 1 text (string)
    df["ingredients_text"] = df.apply(extract_raw_ingredient_text, axis=1)
    # Very light normalization: lowercase + strip
    df["ingredients_text"] = df["ingredients_text"].astype(str).str.strip().str.lower()

    # Phase 2 list (tokens) — use ingredients_text as source if ingredients_clean not present/usable
    if "ingredients_clean" in df.columns:
        # Parse/normalize whatever is currently there
        df["ingredients_clean"] = df["ingredients_clean"].apply(parse_ingredients)
    else:
        # Build from ingredients_text
        df["ingredients_clean"] = df["ingredients_text"].apply(parse_ingredients)

    print("[INFO] Example rows after cleaning:")
    print(df[["cuisine_cleaned_str", "ingredients_text", "ingredients_clean"]].head())

    # -----------------------------------------------------------------
    # Save encoded parquet
    # -----------------------------------------------------------------
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"[INFO] Wrote cleaned & encoded data to {OUT_PARQUET}")
    print(f"[INFO] Final shape: {df.shape}")

    print(df.head())


if __name__ == "__main__":
    main()
