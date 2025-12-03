# flavor_saviors_modeling.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# 1. Load dataset
df = pd.read_parquet("data/encoded/combined_raw_datasets_with_cuisine_encoded.parquet")

# 2. Basic cleanup (enough to model)
# Keep only rows that have both cuisine and ingredients
df = df.dropna(subset=["cuisine", "ingredients"])

# Ensure strings
df["ingredients"] = df["ingredients"].astype(str)
df["cuisine"] = df["cuisine"].astype(str).str.strip().str.lower()

print(f"Loaded {len(df)} recipes across {df['cuisine'].nunique()} cuisines.")
print(df.head(3))

# 3. Vectorize ingredients (bag-of-words or TF-IDF)
tfidf = TfidfVectorizer(
    token_pattern=r"(?u)\b\w+\b",
    stop_words="english",
    max_features=5000
)
X = tfidf.fit_transform(df["ingredients"])



# Inspect class sizes
vc = df["cuisine"].value_counts()
print("Class counts (top 20):")
print(vc.head(20))

# Option A: collapse rare cuisines into 'other'
MIN_PER_CLASS = 5  # for test_size=0.2 this ensures at least 1 test sample
df["cuisine"] = df["cuisine"].where(df["cuisine"].map(vc) >= MIN_PER_CLASS, "other")

# Recompute counts after collapsing
vc2 = df["cuisine"].value_counts()
print("\nAfter collapsing rare classes:")
print(vc2)

# If 'other' is still too small (edge case), drop it
if vc2.get("other", 0) < 2:
    df = df[df["cuisine"] != "other"]

# Rebuild X, y if needed (if you already built them above)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words="english", max_features=5000)
X = tfidf.fit_transform(df["ingredients"].astype(str))


le = LabelEncoder()
y = le.fit_transform(df["cuisine"])

# Now the stratified split should work
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Logistic Regression baseline
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
print("\n=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr, target_names=le.classes_))
print("Macro F1:", f1_score(y_test, y_pred_lr, average="macro"))

# 6. Random Forest baseline
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\n=== Random Forest ===")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))
print("Macro F1:", f1_score(y_test, y_pred_rf, average="macro"))

# 7. Clustering exploration (unsupervised)
print("\n=== KMeans Clustering on Ingredient Vectors ===")
n_clusters = len(np.unique(y))
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

sil_score = silhouette_score(X, cluster_labels)
print(f"Silhouette Score: {sil_score:.4f}")

# Map clusters back to cuisines (rough exploratory mapping)
df["cluster"] = cluster_labels
cluster_summary = (
    df.groupby("cluster")["cuisine"]
      .apply(lambda x: x.value_counts().head(3))
)
print("\nTop cuisines per cluster:\n", cluster_summary)

# 8. Save model results
results = {
    "log_reg_macro_f1": f1_score(y_test, y_pred_lr, average="macro"),
    "rf_macro_f1": f1_score(y_test, y_pred_rf, average="macro"),
    "silhouette_score": sil_score
}
pd.DataFrame([results]).to_csv("model_results.csv", index=False)

print("\nResults saved to model_results.csv")
