"""
Baseline Modeling Stage: TF-IDF, Logistic Regression, Random Forest, KMeans.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, f1_score, silhouette_score, adjusted_rand_score

from ..core import PipelineContext, StageResult
from ..utils import stage_logger

def extract_top_features(model, vectorizer, class_labels, top_n=20):
    """Extract top features from Logistic Regression coefficients."""
    feature_names = vectorizer.get_feature_names_out()
    rows = []
    for class_idx, label in enumerate(class_labels):
        coefs = model.coef_[class_idx]
        top_indices = np.argsort(coefs)[-top_n:][::-1]
        for rank, idx in enumerate(top_indices):
            rows.append({
                "cuisine": label,
                "rank": rank + 1,
                "feature": feature_names[idx],
                "weight": float(coefs[idx])
            })
    return pd.DataFrame(rows)

def run(context: PipelineContext, *, force: bool = False) -> StageResult:
    cfg = context.stage("analysis_baseline")
    logger = stage_logger(context, "analysis_baseline", force=force)
    
    # Config
    data_cfg = cfg.get("data", {})
    params = cfg.get("params", {})
    output_cfg = cfg.get("output", {})
    
    input_path = Path(data_cfg.get("input_path"))
    out_dir = Path(output_cfg.get("reports_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Data
    logger.info(f"Loading {input_path}...")
    df = pd.read_parquet(input_path)
    
    # Prepare text
    ing_col = data_cfg.get("ingredients_col", "inferred_ingredients")
    if not isinstance(df[ing_col].iloc[0], str):
        df["text"] = df[ing_col].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
    else:
        df["text"] = df[ing_col]
    
    cuisine_col = data_cfg.get("cuisine_col", "cuisine_deduped")
    min_label_freq = int(params.get("min_label_freq", 10))
    max_label_samples = params.get("max_label_samples")
    parent_map_path = params.get("parent_map_path")

    corrections = {
        "mediteranean": "Mediterranean",
        "mediterranean": "Mediterranean",
        "middle": "Middle Eastern",
        "middle eastern": "Middle Eastern",
        "east-asian": "East Asian",
        "east asian": "East Asian",
    }

    def _normalize_whitespace(s: str) -> str:
        return (
            s.replace("\xa0", " ")
             .replace("\u200b", "")
             .replace("\ufeff", "")
             .strip()
        )

    def _parse_list(val):
        if isinstance(val, (list, tuple, np.ndarray)):
            return [str(v) for v in val]
        if pd.isna(val):
            return []
        s = _normalize_whitespace(str(val))
        if not s:
            return []
        # Trim stray brackets/quotes
        s = s.strip("[]'\" ")
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, np.ndarray)):
                    return [_normalize_whitespace(str(x)) for x in parsed]
            except Exception:
                pass
        if "," in s:
            return [_normalize_whitespace(t) for t in s.split(",") if _normalize_whitespace(t)]
        return [s]

    generic_drop = {"kid friendly", "friendly"}

    def _load_parent_map(path: str | None):
        if not path:
            return {}
        p = Path(path)
        if not p.exists():
            logger.warning("Parent map not found at %s", p)
            return {}
        try:
            data = json.load(open(p, "r", encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items() if str(k).strip() and str(v).strip()}
        except Exception as e:
            logger.warning("Failed to load parent map %s: %s", p, e)
        return {}

    parent_map = _load_parent_map(parent_map_path)
    # Normalize parent map keys for flexible lookup
    def _norm(s: str) -> str:
        return str(s).strip().lower().replace("-", " ")
    parent_map_norm = {_norm(k): v for k, v in parent_map.items()}
    # Alias to collapse parent labels (e.g., East Asian -> Asian, Oriental -> Asian)
    parent_alias_norm = {
        "oriental": "Asian",
        "east asian": "Asian",
        "asian": "Asian",
        "indian": "Indian",
        "mediterranean": "Continental",
        "european": "Continental",
    }

    def _to_label(val):
        # Normalize label to a single string with simple de-biasing/cleanup
        labels = []
        for v in _parse_list(val):
            s = _normalize_whitespace(str(v))
            if not s:
                continue
            key = s.lower().replace("-", " ")
            s_norm = corrections.get(key, s)
            labels.append(s_norm)
        if not labels:
            return ""
        # Drop generic tags
        labels = [l for l in labels if l.lower() not in generic_drop] or labels
        # If American co-occurs with other labels, drop American to avoid bias
        if len(labels) > 1:
            labels_no_american = [l for l in labels if l.lower() != "american"]
            if labels_no_american:
                labels = labels_no_american
        return labels[0] if labels else ""

    df[cuisine_col] = df[cuisine_col].apply(_to_label)
    df = df[df[cuisine_col] != ""]

    # Filter to labels with sufficient support
    label_counts = df[cuisine_col].value_counts()
    keep_labels = label_counts[label_counts >= min_label_freq].index
    df = df[df[cuisine_col].isin(keep_labels)].reset_index(drop=True)
    if max_label_samples:
        try:
            max_label_samples_int = int(max_label_samples)
            df = (
                df.groupby(cuisine_col, group_keys=False)
                .apply(lambda g: g.sample(n=min(len(g), max_label_samples_int), random_state=42))
                .reset_index(drop=True)
            )
            logger.info("Downsampled each label to max %s samples (after min_freq filter)", max_label_samples_int)
        except Exception as e:
            logger.warning("Failed to downsample labels: %s", e)
    y = df[cuisine_col]
    logger.info("Label filtering: kept %s labels with at least %s examples", len(keep_labels), min_label_freq)
    
    # TF-IDF
    logger.info("Vectorizing...")
    tfidf = TfidfVectorizer(max_features=params.get("max_features", 5000), min_df=5)
    X = tfidf.fit_transform(df["text"])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Logistic Regression
    logger.info("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=500, n_jobs=-1, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Clean labels for reporting
    def _clean_label_scalar(val):
        # Reuse full cleaning/debiasing pipeline to collapse multi-values like "Chinese, Asian"
        return _to_label(val)

    y_true_clean = [_clean_label_scalar(v) for v in y_test]
    y_pred_clean = [_clean_label_scalar(v) for v in y_pred]

    # Parent-aware evaluation (do not replace original labels)
    def to_parent(v):
        if not parent_map_norm:
            return str(v)
        key = _norm(v)
        parent = parent_map_norm.get(key, str(v))
        parent_key = _norm(parent)
        return parent_alias_norm.get(parent_key, parent)
    y_true_parent = [to_parent(v) for v in y_true_clean]
    y_pred_parent = [to_parent(v) for v in y_pred_clean]

    # Save Preds with parent mapping info
    preds_df = pd.DataFrame({
        "y_true": y_true_clean,
        "y_pred": y_pred_clean,
        "y_true_parent": y_true_parent,
        "y_pred_parent": y_pred_parent,
        # Exact match wins; otherwise parent-level match counts
        "parent_match": [
            yt == yp or a == b
            for yt, yp, a, b in zip(y_true_clean, y_pred_clean, y_true_parent, y_pred_parent)
        ],
    })
    preds_df.to_csv(out_dir / "y_pred_logreg.csv", index=False)
    
    # Save Report
    report = classification_report(y_true_clean, y_pred_clean, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out_dir / "classification_report_logreg.csv")
    if parent_map:
        report_parent = classification_report(
            y_true_parent,
            y_pred_parent,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report_parent).T.to_csv(out_dir / "classification_report_logreg_parent.csv")
    
    # Top Features
    top_feat = extract_top_features(clf, tfidf, clf.classes_)
    top_feat.to_csv(out_dir / "top_features_logreg.csv", index=False)
    
    # 2. Random Forest (Subsampled)
    logger.info("Training Random Forest (Subsampled)...")
    max_samples = params.get("rf_max_samples", 15000)
    if X_train.shape[0] > max_samples:
        # Subsample for speed
        idx = np.random.choice(X_train.shape[0], max_samples, replace=False)
        X_rf, y_rf = X_train[idx], y_train.iloc[idx]
    else:
        X_rf, y_rf = X_train, y_train
        
    rf = RandomForestClassifier(n_estimators=50, max_depth=20, n_jobs=-1)
    rf.fit(X_rf, y_rf)
    
    # 3. Clustering (KMeans)
    logger.info("Running KMeans...")
    k = params.get("kmeans_k", 20)
    kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Save Cluster Assignments
    df_clusters = pd.DataFrame({
        "cuisine": y,
        "cluster": clusters
    })
    df_clusters.to_csv(out_dir / "cluster_assignments.csv", index=False)

    return StageResult(name="analysis_baseline", status="success", outputs={
        "report": str(out_dir / "classification_report_logreg.csv"),
        "preds": str(out_dir / "y_pred_logreg.csv"),
        "clusters": str(out_dir / "cluster_assignments.csv")
    })
