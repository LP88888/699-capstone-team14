import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from sklearn.decomposition import PCA

# -----------------------------
# Setup
# -----------------------------
OUT = Path("reports/phase2")  # update if your modeling script saves elsewhere
DATA_PATH = Path("data/encoded/combined_raw_datasets_with_cuisine_encoded.parquet")

# -----------------------------

# Load Data
# -----------------------------
df = pd.read_parquet(DATA_PATH)
summary = json.load(open(OUT / "summary.json"))

print("Loaded data with", len(df), "rows")
print("Summary metrics:")
for k, v in summary.items():
    print(f"  {k}: {v}")

# -----------------------------
# 1. Macro F1 Comparison
# -----------------------------
plt.figure(figsize=(5,4))
scores = {
    "TF-IDF Only": summary.get("tfidf_only_macro_f1"),
    "TF-IDF + Graph": summary.get("tfidf_plus_graph_macro_f1"),
}
sns.barplot(x=list(scores.keys()), y=list(scores.values()), palette="viridis")
plt.title("Macro F1 Comparison of Models")
plt.ylabel("Macro F1 Score")
plt.tight_layout()
plt.savefig(OUT / "viz_macro_f1_comparison.png")
plt.show()

# -----------------------------
# 2. Top Cuisines by Frequency
# -----------------------------
top_cuisines = df["cuisine"].value_counts().head(15)
plt.figure(figsize=(8,5))
sns.barplot(x=top_cuisines.values, y=top_cuisines.index, palette="crest")
plt.title("Most Common Cuisines in Dataset")
plt.xlabel("Recipe Count")
plt.tight_layout()
plt.savefig(OUT / "viz_top_cuisines.png")
plt.show()

# -----------------------------
# 3. PCA Projection (optional)
# -----------------------------
if (OUT / "X_tfidf.csv").exists():
    X_tfidf = pd.read_csv(OUT / "X_tfidf.csv")
    pca = PCA(n_components=2, random_state=42)
    X2d = pca.fit_transform(X_tfidf.values)
    pca_df = pd.DataFrame(X2d, columns=["PC1", "PC2"])
    pca_df["cuisine"] = df["cuisine"]
    plt.figure(figsize=(10,7))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2",
                    hue="cuisine", s=10, legend=False, palette="tab20")
    plt.title("PCA Projection of Recipes by Cuisine")
    plt.tight_layout()
    plt.savefig(OUT / "viz_pca_projection.png")
    plt.show()

# -----------------------------
# 4. Clustering Summary Heatmap (optional)
# -----------------------------
if (OUT / "clustering_quality_kmeans.csv").exists():
    cm = pd.read_csv(OUT / "clustering_quality_kmeans.csv")
    plt.figure(figsize=(8,6))
    sns.heatmap(cm.corr(), cmap="coolwarm", annot=False)
    plt.title("KMeans Clustering Quality Correlation")
    plt.tight_layout()
    plt.savefig(OUT / "viz_kmeans_correlation.png")
    plt.show()
