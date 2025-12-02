
import json
import config as cfg
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    silhouette_score,
    adjusted_rand_score,
)

DATA_PATH = cfg.CLEANED_DATA_PATH
OUT_DIR = cfg.PHASE_2_REPORTS_PATH

# TF-IDF settings
TFIDF_MAX_FEATURES = 15000

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_SAMPLES_PER_CLASS = 20  # drop very rare cuisines

# KMeans clustering
KMEANS_K = 40

# RandomForest 
MAX_RF_TRAIN_SAMPLES = 15000  # subsample training rows for RF


def ingredients_to_text(val):
    """
    Convert an 'ingredients' field into a single lowercase space-separated string.

    Handles:
        - list of strings
        - list of lists
        - plain string
        - anything else -> ""
    """
    if isinstance(val, list):
        flat = []
        for x in val:
            if isinstance(x, str):
                flat.append(x)
            elif isinstance(x, list):
                flat.extend(str(t) for t in x)
        return " ".join(flat).lower().strip()

    if isinstance(val, str):
        return val.lower().strip()

    return ""


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """
    Phase 2 modeling pipeline:
      1. Load cleaned data
      2. Ensure ingredients_text exists
      3. Clean and filter cuisines
      4. Build TF-IDF features
      5. Train Logistic Regression and Random Forest
      6. Run KMeans clustering
      7. Write predictions, reports, and summary
    """

    print(f"[INFO] Loading data from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"[INFO] Loaded data with shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")

    # -----------------------------------------------------------------
    # Choose label column
    # -----------------------------------------------------------------
    if "cuisine_cleaned_str" in df.columns:
        label_col = "cuisine_cleaned_str"
    elif "cuisine" in df.columns:
        label_col = "cuisine"
    else:
        raise ValueError("Expected 'cuisine_cleaned_str' or 'cuisine' column in data.")

    df[label_col] = df[label_col].astype(str).str.strip().str.lower()

    if "ingredients_text" not in df.columns:
        raise ValueError("Need either 'ingredients_text' or 'ingredients' in data.")
        

    df["ingredients_text"] = (
        df["ingredients_text"].astype(str).str.strip().str.lower()
    )

    # Drop rows with empty text or missing label
    df = df[(df["ingredients_text"].str.len() > 0) & df[label_col].notna()].copy()
    df.reset_index(drop=True, inplace=True)
    print(f"[INFO] After dropping empty text/labels: {df.shape}")

    counts = df[label_col].value_counts()
    keep_labels = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    df = df[df[label_col].isin(keep_labels)].copy()
    df.reset_index(drop=True, inplace=True)

    print(
        f"[INFO] After dropping cuisines with <{MIN_SAMPLES_PER_CLASS} samples: "
        f"{df.shape}"
    )
    print("[INFO] Top cuisines after filtering:")
    print(df[label_col].value_counts().head(10))

    le = LabelEncoder()
    y = le.fit_transform(df[label_col].values)
    texts = df["ingredients_text"].values

    print("[INFO] Building TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=2,
    )
    X = tfidf.fit_transform(texts)
    print(f"[INFO] TF-IDF shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"[INFO] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    print("[INFO] Training LogisticRegression (TF-IDF only)...")

    logreg = LogisticRegression(
        max_iter=200,
        n_jobs=1,
        multi_class="auto",
    )
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)

    acc_lr = accuracy_score(y_test, y_pred_lr)
    macro_f1_lr = f1_score(y_test, y_pred_lr, average="macro")

    print("\n=== Logistic Regression (Phase 2) ===")
    print(classification_report(y_test, y_pred_lr, target_names=le.classes_))

    # Save classification report
    report_lr = classification_report(
        y_test,
        y_pred_lr,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )
    df_report_lr = (
        pd.DataFrame(report_lr)
        .T.reset_index()
        .rename(columns={"index": "label"})
    )
    rep_path_lr = OUT_DIR / "classification_report_phase2_logreg.csv"
    df_report_lr.to_csv(rep_path_lr, index=False)
    print(f"[INFO] Wrote LR classification report to {rep_path_lr}")

    # Save predictions for visualizations
    df_preds_lr = pd.DataFrame({
        "y_test": y_test,
        "y_pred": y_pred_lr
    })
    preds_path_lr = OUT_DIR / "y_pred_phase2.csv"
    df_preds_lr.to_csv(preds_path_lr, index=False)
    print(f"[INFO] Wrote LR predictions to {preds_path_lr}")

    print("[INFO] Extracting top TF-IDF features per cuisine from LR coefficients...")
    feature_names = tfidf.get_feature_names_out()
    coefs = logreg.coef_  # shape: (n_classes, n_features)

    rows_top = []
    top_k = 20  # number of top features per cuisine

    for class_idx, cuisine_name in enumerate(le.classes_):
        weights = coefs[class_idx]
        top_indices = np.argsort(weights)[-top_k:][::-1]
        for rank, feat_idx in enumerate(top_indices):
            rows_top.append({
                "cuisine": cuisine_name,
                "rank": rank + 1,
                "ingredient": feature_names[feat_idx],
                "weight": float(weights[feat_idx]),
            })

    df_top_features = pd.DataFrame(rows_top)
    tf_path = OUT_DIR / "top_features_phase2_logreg.csv"
    df_top_features.to_csv(tf_path, index=False)
    print(f"[INFO] Wrote top features per cuisine to {tf_path}")

    print("[INFO] Computing simple ingredient token counts...")
    # naive tokenization: split on whitespace
    tokens = df["ingredients_text"].str.split()
    df_tokens = tokens.explode().dropna()
    df_counts = (
        df_tokens.value_counts()
        .reset_index()
        .rename(columns={"index": "ingredient", "ingredients_text": "count"})
    )
    counts_path = OUT_DIR / "ingredient_counts_phase2.csv"
    df_counts.to_csv(counts_path, index=False)
    print(f"[INFO] Wrote ingredient counts to {counts_path}")

    macro_f1_rf = None
    try:
        print("\n[INFO] Training RandomForestClassifier (reduced size)...")

        if X_train.shape[0] > MAX_RF_TRAIN_SAMPLES:
            rng = np.random.default_rng(seed=RANDOM_STATE)
            idx = rng.choice(
                X_train.shape[0],
                size=MAX_RF_TRAIN_SAMPLES,
                replace=False,
            )
            X_train_rf = X_train[idx]
            y_train_rf = y_train[idx]
            print(
                f"[INFO] RF using subsample of "
                f"{MAX_RF_TRAIN_SAMPLES} / {X_train.shape[0]} training samples"
            )
        else:
            X_train_rf = X_train
            y_train_rf = y_train
            print(f"[INFO] RF using all {X_train.shape[0]} training samples")

        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=20,
            max_features=0.5,
            min_samples_leaf=5,
            n_jobs=1,  # 1 core to keep peak memory lower
            random_state=RANDOM_STATE,
        )

        rf.fit(X_train_rf, y_train_rf)
        y_pred_rf = rf.predict(X_test)

        macro_f1_rf = f1_score(y_test, y_pred_rf, average="macro")

        print("\n=== Random Forest (Phase 2, reduced) ===")
        print(classification_report(y_test, y_pred_rf, target_names=le.classes_))
        print(f"RandomForest macro F1: {macro_f1_rf:.4f}")

        # classification report for RF
        report_rf = classification_report(
            y_test,
            y_pred_rf,
            target_names=le.classes_,
            output_dict=True,
            zero_division=0,
        )
        df_report_rf = (
            pd.DataFrame(report_rf)
            .T.reset_index()
            .rename(columns={"index": "label"})
        )
        rep_path_rf = OUT_DIR / "classification_report_phase2_rf.csv"
        df_report_rf.to_csv(rep_path_rf, index=False)
        print(f"[INFO] Wrote RF classification report to {rep_path_rf}")

        # save RF predictions
        df_preds_rf = pd.DataFrame({
            "y_test": y_test,
            "y_pred": y_pred_rf
        })
        preds_path_rf = OUT_DIR / "y_pred_rf_phase2.csv"
        df_preds_rf.to_csv(preds_path_rf, index=False)
        print(f"[INFO] Wrote RF predictions to {preds_path_rf}")

    except MemoryError:
        print(
            "[WARN] RandomForest ran out of memory even with reduced settings; "
            "skipping RF."
        )
        macro_f1_rf = None

    print(f"\n[INFO] Running KMeans (k={KMEANS_K}) on TF-IDF features...")
    kmeans = KMeans(n_clusters=KMEANS_K, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    sil = silhouette_score(X, cluster_labels, metric="cosine")
    ari = adjusted_rand_score(y, cluster_labels)

    print(f"[INFO] KMeans silhouette (cosine): {sil:.4f}")
    print(f"[INFO] KMeans ARI vs cuisine labels: {ari:.4f}")

    df_kmeans = pd.DataFrame(
        {"k": [KMEANS_K], "silhouette": [float(sil)], "ari_vs_labels": [float(ari)]}
    )
    kmeans_path = OUT_DIR / "clustering_quality_phase2_kmeans.csv"
    df_kmeans.to_csv(kmeans_path, index=False)
    print(f"[INFO] Wrote KMeans clustering quality to {kmeans_path}")

    # Save per-row cluster assignments for plots
    df_cluster_assign = pd.DataFrame({
        "row_index": np.arange(len(df)),
        "cuisine_label": df[label_col].values,
        "cluster": cluster_labels,
    })
    cluster_assign_path = OUT_DIR / "cluster_assignments_phase2.csv"
    df_cluster_assign.to_csv(cluster_assign_path, index=False)
    print(f"[INFO] Wrote cluster assignments to {cluster_assign_path}")

    summary = {
        "logreg_accuracy": float(acc_lr),
        "logreg_macro_f1": float(macro_f1_lr),
        "rf_macro_f1": float(macro_f1_rf) if macro_f1_rf is not None else None,
        "kmeans_k": int(KMEANS_K),
        "kmeans_silhouette": float(sil),
        "kmeans_ari_vs_labels": float(ari),
        "n_recipes_modeled": int(len(df)),
        "n_cuisines_modeled": int(df[label_col].nunique()),
        "tfidf_max_features": TFIDF_MAX_FEATURES,
    }

    summary_path1 = OUT_DIR / "summary_phase2_modeling.json"
    summary_path2 = OUT_DIR / "summary.json"  # alias for existing code
    with open(summary_path1, "w") as f:
        json.dump(summary, f, indent=2)
    with open(summary_path2, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Wrote phase 2 modeling summary to {summary_path1} and {summary_path2}")
    print("[INFO] Phase 2 modeling complete.")


if __name__ == "__main__":
    main()
