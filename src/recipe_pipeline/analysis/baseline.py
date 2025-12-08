"""
Baseline Modeling Stage: TF-IDF, Logistic Regression, Random Forest, KMeans.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
import ast
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, f1_score, silhouette_score, adjusted_rand_score
from sklearn.utils import check_random_state

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

def _svd_variance_frame(svd: TruncatedSVD) -> pd.DataFrame:
    """Return per-component and cumulative explained variance."""
    var = np.asarray(svd.explained_variance_ratio_, dtype=float)
    return pd.DataFrame({
        "component": np.arange(1, len(var) + 1),
        "explained_variance_ratio": var,
        "cumulative_explained_variance": np.cumsum(var),
    })

def _svd_top_terms(svd: TruncatedSVD, vectorizer: TfidfVectorizer, *, top_n: int = 15) -> pd.DataFrame:
    """Return top positive/negative terms per component for interpretability."""
    feature_names = vectorizer.get_feature_names_out()
    rows = []
    for comp_idx, comp in enumerate(svd.components_):
        order = np.argsort(comp)
        top_pos = order[-top_n:][::-1]
        top_neg = order[:top_n]
        for rank, idx in enumerate(top_pos, 1):
            rows.append({
                "component": comp_idx + 1,
                "direction": "positive",
                "rank": rank,
                "term": feature_names[idx],
                "weight": float(comp[idx]),
            })
        for rank, idx in enumerate(top_neg, 1):
            rows.append({
                "component": comp_idx + 1,
                "direction": "negative",
                "rank": rank,
                "term": feature_names[idx],
                "weight": float(comp[idx]),
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
    min_df_raw = params.get("min_df", 5)
    try:
        min_df = float(min_df_raw)
        if float(min_df_raw).is_integer():
            min_df = int(min_df)
    except Exception:
        min_df = 5
    max_df_raw = params.get("max_df", 1.0)
    max_df = float(max_df_raw) if max_df_raw is not None else 1.0
    ngram_min = int(params.get("ngram_min", 1))
    ngram_max = int(params.get("ngram_max", max(ngram_min, 1)))
    if ngram_max < ngram_min:
        ngram_max = ngram_min
    apply_parent_for_training = bool(params.get("apply_parent_for_training", False))
    cluster_use_parent_labels = bool(params.get("cluster_use_parent_labels", False))
    max_label_samples = params.get("max_label_samples")
    parent_map_path = params.get("parent_map_path")
    rng = check_random_state(42)

    corrections = {
        "mediteranean": "Mediterranean",
        "mediterranean": "Mediterranean",
        "middle": "Middle Eastern",
        "middle eastern": "Middle Eastern",
        "east-asian": "East Asian",
        "east asian": "East Asian",
        # Cajun synonyms
        "cajun & creole": "Cajun",
        "cajun and creole": "Cajun",
        # Indian subregions collapsed to Indian
        "karnataka": "Indian",
        "kerala": "Indian",
        "tamil nadu": "Indian",
        "maharashtrian": "Indian",
        "maharashtra": "Indian",
        "gujarati": "Indian",
        "goan": "Indian",
        "jharkhand": "Indian",
        # Cajun synonyms
        "cajun & creole": "Cajun",
        "cajun and creole": "Cajun",
        # Indian subregions collapsed to Indian
        "karnataka": "Indian",
        "kerala": "Indian",
        "tamil nadu": "Indian",
        "maharashtrian": "Indian",
        "maharashtra": "Indian",
        "gujarati": "Indian",
        "goan": "Indian",
        "jharkhand": "Indian",
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
    if apply_parent_for_training and not parent_map_norm:
        logger.warning("apply_parent_for_training enabled but no parent map found; training on original labels.")
    if cluster_use_parent_labels and not parent_map_norm:
        logger.warning("cluster_use_parent_labels enabled but no parent map found; clustering will use original labels.")

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

    def to_parent_label(v: str) -> str:
        """Map a cuisine to its parent using configured maps/aliases."""
        if not parent_map_norm:
            return str(v)
        key = _norm(v)
        parent = parent_map_norm.get(key, str(v))
        parent_key = _norm(parent)
        return parent_alias_norm.get(parent_key, parent)

    df[cuisine_col] = df[cuisine_col].apply(_to_label)
    df = df[df[cuisine_col] != ""]

    # Optional parent collapse before filtering/evaluation
    if apply_parent_for_training and parent_map_norm:
        df[cuisine_col] = df[cuisine_col].apply(to_parent_label)
    # Strip stray bracket/brace/paren characters and drop any remaining noisy labels
    df[cuisine_col] = df[cuisine_col].str.strip("[]{}() ")
    df = df[~df[cuisine_col].str.contains(r"[\\[\\]{}]", regex=True)]

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
    y_parent_full = df[cuisine_col].apply(to_parent_label) if parent_map_norm else y
    logger.info("Label filtering: kept %s labels with at least %s examples", len(keep_labels), min_label_freq)
    
    # TF-IDF
    logger.info("Vectorizing...")
    tfidf = TfidfVectorizer(
        max_features=params.get("max_features", 5000),
        min_df=min_df,
        max_df=max_df,
        ngram_range=(ngram_min, ngram_max),
    )
    X_tfidf = tfidf.fit_transform(df["text"])

    # Optional dimensionality reduction (sparse-friendly)
    use_svd = bool(params.get("use_svd", False))
    svd_components = int(params.get("svd_components", 300))
    svd_top_terms_n = int(params.get("svd_top_terms", 15))
    svd_scatter_samples = int(params.get("svd_scatter_samples", 5000))
    if use_svd:
        logger.info("Applying TruncatedSVD to %s components...", svd_components)
        svd = TruncatedSVD(n_components=svd_components, random_state=42)
        X_features = svd.fit_transform(X_tfidf)
        logger.info("TruncatedSVD explained variance (approx): %.4f", float(svd.explained_variance_ratio_.sum()))
        # Persist variance curve + component loadings for downstream visuals
        var_path = out_dir / "svd_variance.csv"
        _svd_variance_frame(svd).to_csv(var_path, index=False)
        terms_path = out_dir / "svd_components_top_terms.csv"
        _svd_top_terms(svd, tfidf, top_n=svd_top_terms_n).to_csv(terms_path, index=False)
        # Sample projections for interactive scatter plot
        sample_n = min(svd_scatter_samples, X_features.shape[0])
        proj_path = out_dir / "svd_projection_sample.csv"
        if sample_n > 0:
            sample_idx = rng.choice(X_features.shape[0], sample_n, replace=False)
            svd_proj = {
                "svd_1": X_features[sample_idx, 0],
                "svd_2": X_features[sample_idx, 1],
            }
            if svd_components >= 3:
                svd_proj["svd_3"] = X_features[sample_idx, 2]
            svd_proj[cuisine_col] = y.iloc[sample_idx].to_numpy()
            pd.DataFrame(svd_proj).to_csv(proj_path, index=False)
        logger.info("Saved SVD variance, component loadings, and sampled projections to %s", out_dir)
    else:
        X_features = X_tfidf
    
    # Split (same seed across feature spaces to keep splits aligned)
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    # Split (same seed across feature spaces to keep splits aligned)
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
    X_train_tf, X_test_tf, y_train_tf, y_test_tf = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    
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
    y_true_parent = [to_parent_label(v) for v in y_true_clean]
    y_pred_parent = [to_parent_label(v) for v in y_pred_clean]

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
    
    # Top Features (always from a TF-IDF model for interpretability)
    if use_svd:
        aux_clf = LogisticRegression(max_iter=500, n_jobs=-1, class_weight="balanced")
        aux_clf.fit(X_train_tf, y_train_tf)
        top_feat = extract_top_features(aux_clf, tfidf, aux_clf.classes_)
    else:
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
    clusters = kmeans.fit_predict(X_features)
    cluster_labels = y_parent_full if (cluster_use_parent_labels and parent_map_norm) else y
    # Cluster metrics (silhouette/ARI)
    sil = None
    ari = None
    try:
        sil = silhouette_score(X_features, clusters)
    except Exception as e:
        logger.warning("Silhouette score failed: %s", e)
    try:
        ari_labels = y_parent_full if (cluster_use_parent_labels and parent_map_norm) else y
        ari = adjusted_rand_score(ari_labels, clusters)
    except Exception as e:
        logger.warning("ARI failed: %s", e)
    
    # Save Cluster Assignments
    df_clusters = pd.DataFrame({
        "cuisine": cluster_labels,
        "cuisine_parent": y_parent_full if parent_map_norm else y,
        "cluster": clusters
    })
    df_clusters.to_csv(out_dir / "cluster_assignments.csv", index=False)
    pd.DataFrame([{
        "kmeans_k": k,
        "silhouette": sil,
        "ari": ari,
        "ari_label_space": "parent" if (cluster_use_parent_labels and parent_map_norm) else "child",
    }]).to_csv(out_dir / "cluster_metrics.csv", index=False)

    outputs = {
        "report": str(out_dir / "classification_report_logreg.csv"),
        "preds": str(out_dir / "y_pred_logreg.csv"),
        "clusters": str(out_dir / "cluster_assignments.csv"),
    }
    if use_svd:
        outputs.update({
            "svd_variance": str(var_path),
            "svd_components": str(terms_path),
            "svd_projection_sample": str(proj_path),
        })

    return StageResult(name="analysis_baseline", status="success", outputs=outputs)
