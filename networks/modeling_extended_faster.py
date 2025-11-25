#!/usr/bin/env python
"""
Lightweight modeling_extended.py for large recipe+cuisine dataset.

What it does:
- Loads a combined recipes CSV with columns: cuisine, ingredients or ingredients_clean
- Builds a TF-IDF representation of ingredients
- Trains a logistic regression classifier (TF-IDF only)
- Evaluates with classification_report + macro F1
- Runs a single MiniBatchKMeans clustering on TF-IDF
- Saves:
    - classification_report_tfidf_only.csv
    - clustering_quality_kmeans.csv
    - top_features_per_cuisine.csv
    - ingredient_counts.csv
    - summary.json
"""

import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    silhouette_score,
    adjusted_rand_score,
)
from sklearn.cluster import MiniBatchKMeans
from sklearn.exceptions import ConvergenceWarning
import warnings


# =========================
# CONFIG
# =========================
RANDOM_STATE = 42

# Path to your merged dataset (update this!)
INPUT_CSV = Path("data/encoded/combined_raw_datasets_with_cuisine_encoded.parquet")

# Where to write modeling outputs
OUT = Path("../../reports/phase2")
OUT.mkdir(parents=True, exist_ok=True)

# Runtime controls
FAST_MODE = True

# Max rows to use (for both classification and clustering)
MAX_RECIPES = 40000

# TF-IDF config
TFIDF_MAX_FEATURES = 20000
MIN_SAMPLES_PER_CLASS = 10  # drop extremely rare cuisines

# Clustering config
KMEANS_K = 40
MAX_SAMPLES_FOR_CLUSTERING = 40000  # can be <= MAX_RECIPES


# =========================
# HELPERS
# =========================
def choose_text_column(df: pd.DataFrame) -> str:
    """
    Choose which column to use as ingredient text.
    Prefers 'ingredients_clean', falls back to 'ingredients'.
    """
    if "ingredients_clean" in df.columns:
        return "ingredients_clean"
    elif "ingredients" in df.columns:
        return "ingredients"
    else:
        raise ValueError(
            "Expected a column named 'ingredients_clean' or 'ingredients' in the input CSV."
        )


def compute_top_features_per_cuisine(
    X_tfidf, feature_names, cuisines, top_n=25
) -> pd.DataFrame:
    """
    For each cuisine, compute mean TF-IDF and take top_n features.
    Returns a long-format DataFrame.
    """
    cuisines = np.array(cuisines)
    rows = []
    unique_cuis = np.unique(cuisines)

    for c in unique_cuis:
        mask = cuisines == c
        if mask.sum() == 0:
            continue
        mean_vec = X_tfidf[mask].mean(axis=0)
        mean_vec = np.asarray(mean_vec).ravel()
        # top indices
        top_idx = np.argsort(mean_vec)[::-1][:top_n]
        for rank, j in enumerate(top_idx, start=1):
            rows.append(
                {
                    "cuisine": c,
                    "feature": feature_names[j],
                    "mean_tfidf": float(mean_vec[j]),
                    "rank": rank,
                }
            )

    return pd.DataFrame(rows)


def safe_macro_f1(report_path: Path) -> float | None:
    """
    Read a classification_report CSV and pull out macro avg F1 if present.
    Handles both 'label' as column or index.
    """
    if not report_path.exists():
        return None

    df = pd.read_csv(report_path)

    # common pattern: label column = 'label'
    if "label" in df.columns:
        row = df.loc[df["label"] == "macro avg"]
        if not row.empty and "f1-score" in row.columns:
            return float(row["f1-score"].iloc[0])

    # sometimes first column is unnamed index
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "label"})
        row = df.loc[df["label"] == "macro avg"]
        if not row.empty and "f1-score" in row.columns:
            return float(row["f1-score"].iloc[0])

    # last fallback: try using index
    df_idx = pd.read_csv(report_path, index_col=0)
    if "macro avg" in df_idx.index and "f1-score" in df_idx.columns:
        return float(df_idx.loc["macro avg", "f1-score"])

    return None


# =========================
# MAIN
# =========================
def main():
    print(f"Loading data from {INPUT_CSV.resolve()}")
    df = pd.read_parquet(INPUT_CSV)

    # Basic checks
    if "cuisine" not in df.columns:
        raise ValueError("Input CSV must contain a 'cuisine' column.")

    text_col = choose_text_column(df)
    print(f"Using text column: {text_col}")

    # Drop rows with missing text or cuisine
    df = df.dropna(subset=[text_col, "cuisine"]).copy()

    # Optional downsample for speed
    if FAST_MODE and len(df) > MAX_RECIPES:
        df = df.sample(MAX_RECIPES, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"[FAST_MODE] Downsampled to {len(df)} recipes.")

    # Normalize cuisine labels
    df["cuisine"] = df["cuisine"].astype(str).str.strip().str.lower()

    # Drop extremely rare cuisines so stratified split doesn't blow up
    counts = df["cuisine"].value_counts()
    keep_cuis = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    dropped = set(counts.index) - set(keep_cuis)
    if dropped:
        print(
            f"Dropping {len(dropped)} cuisines with < {MIN_SAMPLES_PER_CLASS} samples "
            f"(total dropped rows: {len(df) - len(df[df['cuisine'].isin(keep_cuis)])})"
        )
        df = df[df["cuisine"].isin(keep_cuis)].copy()

    print(f"Final dataset size for modeling: {len(df)} rows, {df['cuisine'].nunique()} cuisines")

    # =========================
    # TF-IDF
    # =========================
    print("Vectorizing ingredients with TF-IDF...")
    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.7,
        stop_words="english",
        max_features=TFIDF_MAX_FEATURES,
    )

    X_tfidf = tfidf.fit_transform(df[text_col].astype(str))
    feature_names = np.array(tfidf.get_feature_names_out())
    y = df["cuisine"].values

    # Only keep non-zero rows (should be all, but safe)
    nonzero = X_tfidf.getnnz(axis=1) > 0
    X_tfidf = X_tfidf[nonzero]
    y = y[nonzero]
    df = df.loc[nonzero].reset_index(drop=True)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf,
        y_encoded,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    # =========================
    # Logistic Regression classifier
    # =========================
    print("Training Logistic Regression (TF-IDF only)...")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    clf = LogisticRegression(
        max_iter=200,
        n_jobs=-1,
        solver="saga",
        multi_class="multinomial",
        verbose=0,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    acc = (y_pred == y_test).mean()

    print(f"LogReg TF-IDF only: accuracy={acc:.4f}, macro_F1={macro_f1:.4f}")

    # classification report -> CSV
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T
    report_df.insert(0, "label", report_df.index)
    report_df.to_csv(OUT / "classification_report_tfidf_only.csv", index=False)

    # =========================
    # KMeans clustering (MiniBatch) on TF-IDF
    # =========================
    print("Running MiniBatchKMeans clustering...")

    # optionally downsample for clustering if very large
    n_samples = X_tfidf.shape[0]
    if n_samples > MAX_SAMPLES_FOR_CLUSTERING:
        rng = np.random.RandomState(RANDOM_STATE)
        idx_cluster = rng.choice(n_samples, size=MAX_SAMPLES_FOR_CLUSTERING, replace=False)
        X_cluster = X_tfidf[idx_cluster]
        y_cluster = y_encoded[idx_cluster]
        print(
            f"[FAST_MODE] Clustering on subset of {MAX_SAMPLES_FOR_CLUSTERING} / {n_samples} recipes."
        )
    else:
        idx_cluster = np.arange(n_samples)
        X_cluster = X_tfidf
        y_cluster = y_encoded

    # normalize rows for cosine-like geometry
    X_cluster_norm = normalize(X_cluster, norm="l2")

    km = MiniBatchKMeans(
        n_clusters=KMEANS_K,
        random_state=RANDOM_STATE,
        batch_size=4096,
        n_init="auto",
    )
    km_labels = km.fit_predict(X_cluster_norm)

    sil_kmeans = silhouette_score(X_cluster_norm, km_labels, metric="cosine")
    ari_kmeans = adjusted_rand_score(y_cluster, km_labels)

    print(f"KMeans (k={KMEANS_K}) silhouette={sil_kmeans:.4f}, ARI vs cuisine={ari_kmeans:.4f}")

    pd.DataFrame(
    [
        {
            "k": KMEANS_K,
            "n_samples_used": X_cluster.shape[0],
            "silhouette": float(sil_kmeans),
            "ari_vs_labels": float(ari_kmeans),
        }
    ]
    ).to_csv(OUT / "clustering_quality_kmeans.csv", index=False)

    # =========================
    # Top features per cuisine
    # =========================
    print("Computing top TF-IDF features per cuisine...")
    top_feat_df = compute_top_features_per_cuisine(
        X_tfidf, feature_names, y, top_n=25
    )
    top_feat_df.to_csv(OUT / "top_features_per_cuisine.csv", index=False)

    # =========================
    # Ingredient counts (very rough, string-based)
    # =========================
    print("Computing ingredient frequency counts...")
    counts = Counter()
    for txt in df[text_col].astype(str):
        # naive split on comma, strip
        parts = [p.strip().lower() for p in txt.split(",") if p.strip()]
        counts.update(parts)

    ing_rows = [
        {"ingredient": ing, "count": cnt} for ing, cnt in counts.most_common()
    ]
    pd.DataFrame(ing_rows).to_csv(OUT / "ingredient_counts.csv", index=False)

    # =========================
    # Summary JSON
    # =========================
    print("Writing summary.json ...")
    summary = {
        "logreg_tfidf_accuracy": float(acc),
        "logreg_tfidf_macro_f1": safe_macro_f1(
            OUT / "classification_report_tfidf_only.csv"
        ),
        "kmeans_k": int(KMEANS_K),
        "kmeans_silhouette": float(sil_kmeans),
        "kmeans_ari_vs_labels": float(ari_kmeans),
        "n_recipes_used": int(len(df)),
        "n_cuisines": int(df["cuisine"].nunique()),
        "artifacts": [
            "classification_report_tfidf_only.csv",
            "clustering_quality_kmeans.csv",
            "top_features_per_cuisine.csv",
            "ingredient_counts.csv",
        ],
    }

    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done. Phase-2 artifacts written to:", OUT.resolve())


if __name__ == "__main__":
    main()
