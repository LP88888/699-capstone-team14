import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.cluster import MiniBatchKMeans

# ----------------------------
# Config
# ----------------------------
DATA_PATH = "combined_recipes.csv"
OUT_DIR = Path("reports/baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.2
RANDOM_STATE = 42

# for clustering sweep
K_GRID = [8, 12, 16, 20, 25, 30]

# minimum support to keep a cuisine as its own class in taxonomy suggestions
MIN_SUPPORT_FOR_OWN_CLASS = 100  # adjust later

BASIC_PARENT_MAP = {
    # quick heuristic to collapse obvious variants (edit freely later)
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
    r"^asian$"  # optional to drop "asian" if it's too broad
]

# ----------------------------
# Helpers
# ----------------------------
def normalize_ingredients(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\d+([\/\.\d]*)?", " ", s)  # numbers/fractions
    s = re.sub(r"\b(oz|ounce|ounces|cup|cups|tsp|teaspoon|teaspoons|tbsp|tablespoon|tablespoons|g|kg|ml|l|pound|lb|lbs)\b", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def preprocess_cuisine(raw: str) -> str:
    c = str(raw).lower().strip()
    c = re.sub(r"\s*,\s*", ",", c)    # normalize commas
    c = re.sub(r"\s+", " ", c).strip()
    # single-label: keep first token if commas are present
    if "," in c:
        c = c.split(",")[0]

    # remove “recipes”
    c = re.sub(r"\brecipes?\b", "", c).strip()

    # drop patterns
    for pat in DROP_LABEL_PATTERNS:
        if re.search(pat, c):
            return "drop_me"

    return c

def suggest_parent_label(cuisine: str) -> str:
    for pat, parent in BASIC_PARENT_MAP.items():
        if re.match(pat, cuisine):
            return parent
    return cuisine  # fallback: keep as-is

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

# ----------------------------
# Main
# ----------------------------
def main():
    # 1) Load
    df = pd.read_csv(DATA_PATH)
    if not {"ingredients", "cuisine"}.issubset(df.columns):
        raise ValueError("Expected columns 'ingredients' and 'cuisine' in combined_recipes.csv")

    # 2) Light cleanup to proceed
    df = df.dropna(subset=["ingredients", "cuisine"]).copy()
    df["ingredients_clean"] = df["ingredients"].map(normalize_ingredients)

    # Keep a raw cuisine copy for audits
    df["cuisine_raw"] = df["cuisine"]
    df["cuisine"] = df["cuisine"].map(preprocess_cuisine)
    df = df[df["cuisine"] != "drop_me"].copy()

    # 3) Suggest parent/normalized cuisine for taxonomy proposals
    df["cuisine_suggested"] = df["cuisine"].map(suggest_parent_label)

    # 4) TF-IDF
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1,2),
        min_df=5,
        max_df=0.7,
        stop_words="english",
        max_features=40000
    )
    X = vec.fit_transform(df["ingredients_clean"])

    # 5) Encode label (using suggested parent, but keep both)
    le = LabelEncoder()
    y = le.fit_transform(df["cuisine_suggested"])

    # 6) Optional: enforce minimum support for cleaner baseline report
    # Compute label counts aligned to df's index
    label_series = df["cuisine_suggested"]  # same length/index as df
    vc = label_series.value_counts()

    # Choose which labels to keep
    keep_labels = vc[vc >= MIN_SUPPORT_FOR_OWN_CLASS].index

    # Fallback: if nothing passes the threshold, keep top-N most common labels
    if len(keep_labels) == 0:
        TOP_N = 20
        keep_labels = vc.head(TOP_N).index

    # Build an aligned boolean mask from df, not from y
    keep_mask = df["cuisine_suggested"].isin(keep_labels)

    # Use the mask for both df and X (convert to numpy for the sparse matrix)
    df_kee = df.loc[keep_mask].copy()
    X_kee = X[keep_mask.to_numpy()]

    # Refit label encoder on the filtered frame
    le = LabelEncoder()
    y_kee = le.fit_transform(df_kee["cuisine_suggested"])


    # 7) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_kee, y_kee, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_kee
    )

    # 8) Baseline classifier (balanced, multinomial)
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
    # 9) Reports
    rep_df = class_report_table(y_test, y_pred, classes)
    rep_df.to_csv(OUT_DIR / "classification_report.csv", index=False)

    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(OUT_DIR / "confusion_matrix_wide.csv")
    cm_long(cm, classes).to_csv(OUT_DIR / "confusion_matrix_long.csv", index=False)

    top_tokens_df = top_n_tokens(X_kee, vec, n=200)
    top_tokens_df.to_csv(OUT_DIR / "top_tokens.csv", index=False)

    summary = {
        "macro_f1": float(rep_df.loc[rep_df["label"] == "macro avg", "f1-score"].values[0]),
        "accuracy": float(rep_df.loc[rep_df["label"] == "accuracy", "precision"].values[0]),
        "classes_kept": classes.tolist(),
        "min_support_for_class": MIN_SUPPORT_FOR_OWN_CLASS,
    }
    json.dump(summary, open(OUT_DIR / "summary.json", "w"), indent=2)

    # 10) K sweep (cosine-like with L2-normalized TF-IDF)
    Xn = normalize(X_kee, norm="l2", copy=False)
    k_sweep = []
    for k in K_GRID:
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, batch_size=4096)
        lab = km.fit_predict(Xn)
        sil = silhouette_score(Xn, lab)
        k_sweep.append({"k": k, "silhouette": float(sil)})
    pd.DataFrame(k_sweep).to_csv(OUT_DIR / "k_sweep.csv", index=False)

    # 11) Pick a k (best silhouette) and build cluster→label map by majority vote
    best_k = sorted(k_sweep, key=lambda d: d["silhouette"], reverse=True)[0]["k"]
    km_final = MiniBatchKMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10, batch_size=4096)
    clusters = km_final.fit_predict(Xn)

    df_kee = df_kee.copy()
    df_kee["cluster"] = clusters
    cluster_map = cluster_majority_labels(df_kee, "cluster", "cuisine_suggested", top_k=1)
    cluster_map.to_csv(OUT_DIR / "cluster_label_map.csv", index=False)

    # 12) Taxonomy suggestions (CSV to edit by hand)
    # Rule: any cuisine with support < MIN_SUPPORT_FOR_OWN_CLASS → suggest parent (via regex map) or "other"
    raw_counts = df["cuisine"].value_counts()
    sug_counts = df["cuisine_suggested"].value_counts()
    rows = []
    for lab, count in raw_counts.items():
        parent = suggest_parent_label(lab)
        if count >= MIN_SUPPORT_FOR_OWN_CLASS:
            suggested = parent  # keep as class
        else:
            suggested = parent if parent != lab else "other"
        rows.append({
            "raw_cuisine": lab,
            "count": int(count),
            "suggested_super_cuisine": suggested
        })
    taxo = pd.DataFrame(rows).sort_values(["suggested_super_cuisine", "count"], ascending=[True, False])
    taxo.to_csv(OUT_DIR / "taxonomy_suggestions.csv", index=False)

    # 13) Save a small README for teammates
    with open(OUT_DIR / "README.txt", "w", encoding="utf-8") as f:
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

    print(f"Done. Artifacts written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
