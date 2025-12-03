"""
Baseline analysis stage: Statistics, Modeling (LogReg/RF), and Clustering.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, f1_score, silhouette_score, adjusted_rand_score

from ..core import PipelineContext, StageResult
from ..utils import stage_logger

def compute_top_features_per_cuisine(X_tfidf, feature_names, cuisines, top_n=25):
    """Extract top TF-IDF terms per cuisine."""
    cuisines = np.array(cuisines)
    rows = []
    unique_cuis = np.unique(cuisines)

    for c in unique_cuis:
        mask = cuisines == c
        if mask.sum() == 0: continue
        
        # Average TF-IDF vector for this cuisine
        mean_vec = X_tfidf[mask].mean(axis=0)
        mean_vec = np.asarray(mean_vec).ravel()
        top_idx = np.argsort(mean_vec)[::-1][:top_n]
        
        for rank, j in enumerate(top_idx, start=1):
            rows.append({
                "cuisine": c,
                "feature": feature_names[j],
                "mean_tfidf": float(mean_vec[j]),
                "rank": rank,
            })
    return pd.DataFrame(rows)

"""
Baseline Modeling Stage: TF-IDF, Logistic Regression, Random Forest, KMeans.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, f1_score, silhouette_score, adjusted_rand_score

from ...core import PipelineContext, StageResult
from ...utils import stage_logger

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
        
    y = df[data_cfg.get("cuisine_col", "cuisine_deduped")]
    
    # TF-IDF
    logger.info("Vectorizing...")
    tfidf = TfidfVectorizer(max_features=params.get("max_features", 5000), min_df=5)
    X = tfidf.fit_transform(df["text"])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Logistic Regression
    logger.info("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=500, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Save Preds
    preds_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    preds_df.to_csv(out_dir / "y_pred_logreg.csv", index=False)
    
    # Save Report
    report = classification_report(y_test, y_pred, output_dict=True)
    pd.DataFrame(report).T.to_csv(out_dir / "classification_report_logreg.csv")
    
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