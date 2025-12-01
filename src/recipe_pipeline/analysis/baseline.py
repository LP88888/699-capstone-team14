import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.cluster import MiniBatchKMeans

from ..core import PipelineContext, StageResult
from ..utils import stage_logger

# ----------------------------
# Constants & Config Defaults
# ----------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
K_GRID = [8, 12, 16, 20, 25, 30]

BASIC_PARENT_MAP = {
    r".*\b(usa|american)\b.*": "american",
    r".*\brit(alian|alian recipes)\b.*": "italian",
    r".*\bind(ian|ian recipes)\b.*": "indian",
    r".*\bmex(ic|ican)\b.*": "mexican",
    r".*\bchin(ese)?\b.*": "chinese",
    r".*\bgreek\b.*": "greek",
    r".*\bfrench\b.*": "french",
    r".*\bkorean\b.*": "korean",
    r".*\bthai\b.*": "thai",
    r".*\bjapan(ese)?\b.*": "japanese",
    r".*\bgerman\b.*": "german",
    r".*\barab(ic)?\b.*": "arabic",
    r".*\bpakistan(i)?\b.*": "pakistani",
    r".*\bcar(r)?ibbean\b.*": "caribbean",
    r".*\bbrit(ish|ain|unitedkingdom)\b.*": "british",
    r".*\bcontinental\b.*": "continental",
    r".*\bmediterranean\b.*": "mediterranean",
}

DROP_LABEL_PATTERNS = [
    r"kid[-\s]?friendly",
    r"recipes?$",
    r"^other$",
    r"^fusion$",
    r"^asian$"
]

def normalize_ingredients(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\d+([\/\.\d]*)?", " ", s)
    s = re.sub(r"\b(oz|ounce|ounces|cup|cups|tsp|teaspoon|teaspoons|tbsp|tablespoon|tablespoons|g|kg|ml|l|pound|lb|lbs)\b", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def preprocess_cuisine(raw: str) -> str:
    c = str(raw).lower().strip()
    c = re.sub(r"\s*,\s*", ",", c)
    c = re.sub(r"\s+", " ", c).strip()
    if "," in c:
        c = c.split(",")[0]
    c = re.sub(r"\brecipes?\b", "", c).strip()
    for pat in DROP_LABEL_PATTERNS:
        if re.search(pat, c):
            return "drop_me"
    return c

def suggest_parent_label(cuisine: str) -> str:
    for pat, parent in BASIC_PARENT_MAP.items():
        if re.match(pat, cuisine):
            return parent
    return cuisine

def top_n_tokens(X, vectorizer, n=100):
    counts = np.asarray(X.sum(axis=0)).ravel()
    idx = np.argsort(counts)[::-1][:n]
    terms = vectorizer.get_feature_names_out()[idx]
    vals = counts[idx]
    return pd.DataFrame({"token": terms, "count": vals})

def class_report_table(y_true, y_pred, classes):
    rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    df_rep = pd.DataFrame(rep).T.reset_index().rename(columns={"index": "label"})
    return df_rep

def cm_long(cm, classes):
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_long = (cm_df.stack()
               .reset_index(name="count")
               .rename(columns={"level_0": "true", "level_1": "pred"}))
    cm_long = cm_long[cm_long["count"] > 0].sort_values("count", ascending=False)
    return cm_long

def cluster_majority_labels(df_with_clusters, cluster_col="cluster", label_col="cuisine", top_k=1):
    out = []
    for cid, sub in df_with_clusters.groupby(cluster_col):
        counts = sub[label_col].value_counts()
        top = counts.head(top_k)
        for lab, cnt in top.items():
            out.append({"cluster": cid, "label": lab, "count": int(cnt), "support": int(len(sub))})
    return pd.DataFrame(out).sort_values(["cluster", "count"], ascending=[True, False])

def run(context: PipelineContext, **kwargs) -> StageResult:
    logger = stage_logger(context, "analysis_baseline")
    config = context.stage("analysis").get("baseline", {})
    
    # Overrides
    input_path = kwargs.get("input_path", config.get("input_path"))
    output_dir = Path(kwargs.get("output_dir", config.get("output_dir", "./reports/baseline")))
    min_support = kwargs.get("min_support", config.get("min_support", 100))
    
    if not input_path:
        raise ValueError("Input path not specified for analysis_baseline")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading data from {input_path}")
    
    # Load data (Parquet)
    if str(input_path).endswith(".csv"):
         df = pd.read_csv(input_path) # Fallback if user provides csv
    else:
         df = pd.read_parquet(input_path)

    cols_needed = {"ingredients", "cuisine"}
    if not cols_needed.issubset(df.columns):
        # try fallback cols
        if "encoded_ingredients" in df.columns: # Assuming we might run on encoded data but we need text
            # If ingredients is encoded list of ints, this analysis (TFIDF) won't work directly on it
            # unless we decode it. But the config points to encoded_dataset which usually has text too?
            # Checking combine_raw_datasets_with_cuisine_encoded.parquet... usually it retains text columns.
            pass
            
        if not cols_needed.issubset(df.columns):
             logger.warning(f"Columns {cols_needed} not found. Available: {df.columns}")
             # Attempt to find alternates
             if "cuisine_clean" in df.columns:
                 df["cuisine"] = df["cuisine_clean"]
    
    if not cols_needed.issubset(df.columns):
         raise ValueError(f"Expected columns {cols_needed} in dataset")

    logger.info(f"Loaded {len(df)} rows. Cleaning...")
    
    df = df.dropna(subset=["ingredients", "cuisine"]).copy()
    
    # Check if ingredients is list or string
    if not df.empty and isinstance(df["ingredients"].iloc[0], (list, np.ndarray)):
         df["ingredients"] = df["ingredients"].apply(lambda x: " ".join(map(str, x)))

    df["ingredients_clean"] = df["ingredients"].map(normalize_ingredients)
    df["cuisine_raw"] = df["cuisine"]
    df["cuisine"] = df["cuisine"].map(preprocess_cuisine)
    df = df[df["cuisine"] != "drop_me"].copy()
    
    df["cuisine_suggested"] = df["cuisine"].map(suggest_parent_label)
    
    logger.info("Computing TF-IDF...")
    tfidf_config = config.get("tfidf", {})
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=tuple(tfidf_config.get("ngram_range", [1, 2])),
        min_df=5,
        max_df=0.7,
        stop_words="english",
        max_features=tfidf_config.get("max_features", 40000)
    )
    X = vec.fit_transform(df["ingredients_clean"])
    
    le = LabelEncoder()
    # Fit on all suggested cuisines first
    le.fit(df["cuisine_suggested"])
    
    label_series = df["cuisine_suggested"]
    vc = label_series.value_counts()
    keep_labels = vc[vc >= min_support].index
    
    if len(keep_labels) == 0:
        logger.warning(f"No classes met min_support={min_support}. Keeping top 20.")
        keep_labels = vc.head(20).index
        
    keep_mask = df["cuisine_suggested"].isin(keep_labels)
    df_kee = df.loc[keep_mask].copy()
    X_kee = X[keep_mask.to_numpy()]
    
    le = LabelEncoder()
    y_kee = le.fit_transform(df_kee["cuisine_suggested"])
    
    logger.info(f"Training baseline model on {len(df_kee)} samples, {len(le.classes_)} classes...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_kee, y_kee, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_kee
    )
    
    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        solver="saga",
        penalty="l2",
        class_weight="balanced",
        multi_class="multinomial",
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    classes = le.classes_
    rep_df = class_report_table(y_test, y_pred, classes)
    rep_df.to_csv(output_dir / "classification_report.csv", index=False)
    
    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(output_dir / "confusion_matrix_wide.csv")
    cm_long(cm, classes).to_csv(output_dir / "confusion_matrix_long.csv", index=False)
    
    top_tokens_df = top_n_tokens(X_kee, vec, n=200)
    top_tokens_df.to_csv(output_dir / "top_tokens.csv", index=False)
    
    summary = {
        "macro_f1": float(rep_df.loc[rep_df["label"] == "macro avg", "f1-score"].values[0]),
        "accuracy": float(rep_df.loc[rep_df["label"] == "accuracy", "precision"].values[0]),
        "classes_kept": classes.tolist(),
        "min_support_for_class": min_support,
    }
    json.dump(summary, open(output_dir / "summary.json", "w"), indent=2)
    
    logger.info("Running Clustering Sweep...")
    Xn = normalize(X_kee, norm="l2", copy=False)
    k_sweep = []
    for k in K_GRID:
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, batch_size=4096)
        lab = km.fit_predict(Xn)
        sil = silhouette_score(Xn, lab)
        k_sweep.append({"k": k, "silhouette": float(sil)})
    pd.DataFrame(k_sweep).to_csv(output_dir / "k_sweep.csv", index=False)
    
    best_k = sorted(k_sweep, key=lambda d: d["silhouette"], reverse=True)[0]["k"]
    logger.info(f"Best K: {best_k}")
    
    km_final = MiniBatchKMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10, batch_size=4096)
    clusters = km_final.fit_predict(Xn)
    
    df_kee = df_kee.copy()
    df_kee["cluster"] = clusters
    cluster_map = cluster_majority_labels(df_kee, "cluster", "cuisine_suggested", top_k=1)
    cluster_map.to_csv(output_dir / "cluster_label_map.csv", index=False)
    
    # Taxonomy suggestions
    raw_counts = df["cuisine"].value_counts()
    rows = []
    for lab, count in raw_counts.items():
        parent = suggest_parent_label(lab)
        if count >= min_support:
            suggested = parent
        else:
            suggested = parent if parent != lab else "other"
        rows.append({
            "raw_cuisine": lab,
            "count": int(count),
            "suggested_super_cuisine": suggested
        })
    taxo = pd.DataFrame(rows).sort_values(["suggested_super_cuisine", "count"], ascending=[True, False])
    taxo.to_csv(output_dir / "taxonomy_suggestions.csv", index=False)
    
    with open(output_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write(
            "Baseline artifacts generated.\n"
            "- classification_report.csv: per-class precision/recall/f1\n"
            "- confusion_matrix_wide.csv / confusion_matrix_long.csv\n"
            "- top_tokens.csv: most frequent tokens in TF-IDF\n"
            "- k_sweep.csv: silhouette by K\n"
            "- cluster_label_map.csv: majority label per cluster (editable)\n"
            "- taxonomy_suggestions.csv: suggested cuisine→super-cuisine mapping (editable)\n"
            "- summary.json: macro-F1, accuracy, kept classes, thresholds\n"
        )
        
    logger.info(f"Analysis baseline completed. Artifacts in {output_dir}")
    return StageResult(name="analysis_baseline", status="success", artifacts={"output_dir": str(output_dir)})

