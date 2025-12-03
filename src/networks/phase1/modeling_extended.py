
from pathlib import Path
import config as cfg
from collections import Counter
import json
import warnings

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



RANDOM_STATE = 42

# CHANGE THIS IF YOUR FILE IS DIFFERENT
INPUT_CSV = cfg.CLEANED_DATA_PATH

OUT = cfg.PHASE_1_REPORTS_PATH
OUT.mkdir(parents=True, exist_ok=True)

# Maximum number of rows used for modeling/clustering.
# Full dataset is ~80k; we cap at 40k for speed & stability.
MAX_RECIPES_FOR_MODELING = 40000

TFIDF_MAX_FEATURES = 15000
MIN_SAMPLES_PER_CLASS = 10  # drop extremely rare cuisines

# KMeans config
KMEANS_K = 40               # a reasonable number of clusters
MAX_SAMPLES_FOR_CLUSTERING = 40000  # can be <= MAX_RECIPES_FOR_MODELING

def choose_text_column(df: pd.DataFrame) -> str:
    if "ingredients_text" in df.columns:
        return "ingredients_text"
    elif "ingredients" in df.columns:
        return "ingredients"
    else:
        raise ValueError(
            "Expected a column named 'ingredients_text' or 'ingredients' in the input CSV."
        )


def compute_top_features_per_cuisine(
    X_tfidf, feature_names, cuisines, top_n: int = 25
) -> pd.DataFrame:
    cuisines = np.array(cuisines)
    rows = []
    unique_cuis = np.unique(cuisines)

    for c in unique_cuis:
        mask = cuisines == c
        if mask.sum() == 0:
            continue
        mean_vec = X_tfidf[mask].mean(axis=0)
        mean_vec = np.asarray(mean_vec).ravel()
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
    Handles different shapes (label as col or index).
    """
    if not report_path.exists():
        return None

    df = pd.read_csv(report_path)

    if "label" in df.columns:
        row = df.loc[df["label"] == "macro avg"]
        if not row.empty and "f1-score" in row.columns:
            return float(row["f1-score"].iloc[0])

    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "label"})
        row = df.loc[df["label"] == "macro avg"]
        if not row.empty and "f1-score" in row.columns:
            return float(row["f1-score"].iloc[0])

    df_idx = pd.read_csv(report_path, index_col=0)
    if "macro avg" in df_idx.index and "f1-score" in df_idx.columns:
        return float(df_idx.loc["macro avg", "f1-score"])

    return None


# =========================
# MAIN
# =========================
def main():
    print(f"[INFO] Loading data from {INPUT_CSV.resolve()}")
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Cannot find input CSV: {INPUT_CSV}")

    df = pd.read_parquet(INPUT_CSV)

    if "cuisine" not in df.columns:
        raise ValueError("Input CSV must contain a 'cuisine' column.")

    text_col = choose_text_column(df)
    print(f"[INFO] Using text column: {text_col}")

    # Drop rows with missing text or cuisine
    df = df.dropna(subset=[text_col, "cuisine"]).copy()

    # Normalize cuisine labels
    df["cuisine"] = df["cuisine"].astype(str).str.strip().str.lower()

    
    # Drop extremely rare cuisines so stratified split won't choke
    counts = df["cuisine"].value_counts()
    keep_cuis = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    dropped = set(counts.index) - set(keep_cuis)
    if dropped:
        dropped_rows = len(df) - len(df[df["cuisine"].isin(keep_cuis)])
        print(
            f"[INFO] Dropping {len(dropped)} cuisines with < {MIN_SAMPLES_PER_CLASS} samples "
            f"(total dropped rows: {dropped_rows})"
        )
        df = df[df["cuisine"].isin(keep_cuis)].copy()

    # If still very large, downsample to a manageable modeling set
    if len(df) > MAX_RECIPES_FOR_MODELING:
        df = df.sample(MAX_RECIPES_FOR_MODELING, random_state=RANDOM_STATE).reset_index(
            drop=True
        )
        print(f"[INFO] Downsampled to {len(df)} recipes for modeling.")

    print(
        f"[INFO] Final dataset for modeling: {len(df)} rows, "
        f"{df['cuisine'].nunique()} cuisines"
    )

    # =========================
    # TF-IDF
    # =========================
    print("[INFO] Vectorizing ingredients with TF-IDF ...")
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
    y_raw = df["cuisine"].values

    # Filter out rows with zero vectors (just in case)
    nonzero = X_tfidf.getnnz(axis=1) > 0
    X_tfidf = X_tfidf[nonzero]
    y_raw = y_raw[nonzero]
    df = df.loc[nonzero].reset_index(drop=True)

    print(
        f"[INFO] TF-IDF matrix shape: {X_tfidf.shape[0]} rows x {X_tfidf.shape[1]} features"
    )

    # Label encoding
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # =========================
    # Logistic Regression classifier
    # =========================
    print("[INFO] Training Logistic Regression (TF-IDF only) ...")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    clf = LogisticRegression(
        max_iter=150,
        n_jobs=-1,
        solver="saga",
        multi_class="multinomial",
        verbose=0,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = (y_pred == y_test).mean()
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"[RESULT] LogReg TF-IDF: accuracy={acc:.4f}, macro_F1={macro_f1:.4f}")

    # Save classification report
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T
    report_df.insert(0, "label", report_df.index)
    report_path = OUT / "classification_report_tfidf_only.csv"
    report_df.to_csv(report_path, index=False)
    print(f"[INFO] Saved classification report to {report_path}")

    # =========================
    # MiniBatchKMeans clustering on TF-IDF
    # =========================
    print("[INFO] Running MiniBatchKMeans clustering ...")
    n_samples = X_tfidf.shape[0]

    if n_samples > MAX_SAMPLES_FOR_CLUSTERING:
        rng = np.random.RandomState(RANDOM_STATE)
        idx_cluster = rng.choice(
            n_samples, size=MAX_SAMPLES_FOR_CLUSTERING, replace=False
        )
        X_cluster = X_tfidf[idx_cluster]
        y_cluster = y[idx_cluster]
        print(
            f"[INFO] Clustering on subset of {MAX_SAMPLES_FOR_CLUSTERING} / {n_samples} recipes."
        )
    else:
        idx_cluster = np.arange(n_samples)
        X_cluster = X_tfidf
        y_cluster = y

    # Normalize rows for cosine-like geometry
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

    print(
        f"[RESULT] KMeans (k={KMEANS_K}): silhouette={sil_kmeans:.4f}, "
        f"ARI vs cuisines={ari_kmeans:.4f}"
    )

    kmeans_qual_path = OUT / "clustering_quality_kmeans.csv"
    pd.DataFrame(
        [
            {
                "k": KMEANS_K,
                "n_samples_used": int(X_cluster.shape[0]),
                "silhouette": float(sil_kmeans),
                "ari_vs_labels": float(ari_kmeans),
            }
        ]
    ).to_csv(kmeans_qual_path, index=False)
    print(f"[INFO] Saved clustering quality to {kmeans_qual_path}")

    # =========================
    # Top features per cuisine
    # =========================
    print("[INFO] Computing top TF-IDF features per cuisine ...")
    top_feat_df = compute_top_features_per_cuisine(
        X_tfidf, feature_names, y_raw, top_n=25
    )
    top_feat_path = OUT / "top_features_per_cuisine.csv"
    top_feat_df.to_csv(top_feat_path, index=False)
    print(f"[INFO] Saved top features per cuisine to {top_feat_path}")

    # =========================
    # Ingredient frequency counts (simple string-based)
    # =========================
    print("[INFO] Computing ingredient frequency counts ...")
    counts_ing = Counter()
    for txt in df[text_col].astype(str):
        parts = [p.strip().lower() for p in txt.split(",") if p.strip()]
        counts_ing.update(parts)

    ing_rows = [
        {"ingredient": ing, "count": cnt} for ing, cnt in counts_ing.most_common()
    ]
    ing_counts_path = OUT / "ingredient_counts.csv"
    pd.DataFrame(ing_rows).to_csv(ing_counts_path, index=False)
    print(f"[INFO] Saved ingredient counts to {ing_counts_path}")

    # =========================
    # Summary JSON
    # =========================
    print("[INFO] Writing summary.json ...")
    summary = {
        "logreg_tfidf_accuracy": float(acc),
        "logreg_tfidf_macro_f1": safe_macro_f1(
            OUT / "classification_report_tfidf_only.csv"
        ),
        "kmeans_k": int(KMEANS_K),
        "kmeans_silhouette": float(sil_kmeans),
        "kmeans_ari_vs_labels": float(ari_kmeans),
        "n_recipes_modeled": int(len(df)),
        "n_cuisines_modeled": int(df["cuisine"].nunique()),
        "tfidf_max_features": int(TFIDF_MAX_FEATURES),
        "artifacts": [
            "classification_report_tfidf_only.csv",
            "clustering_quality_kmeans.csv",
            "top_features_per_cuisine.csv",
            "ingredient_counts.csv",
        ],
    }

    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[INFO] Phase-2 artifacts written to:", OUT.resolve())


if __name__ == "__main__":
    main()
