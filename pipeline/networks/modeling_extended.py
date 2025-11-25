# modeling_ext.py
import re, os, json
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations
from collections import Counter, defaultdict

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, silhouette_score, adjusted_rand_score
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from scipy import sparse
    
import networkx as nx

OUT = Path("reports/phase2")
OUT.mkdir(parents=True, exist_ok=True)

DATA = "data/encoded/combined_raw_datasets_with_cuisine_encoded.parquet"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ----------------------
# Light normalization
# ----------------------
UNIT_RX = r"\b(oz|ounce|ounces|cup|cups|tsp|teaspoon|teaspoons|tbsp|tablespoon|tablespoons|g|kg|ml|l|lb|lbs|pound|pounds)\b"
def norm_ingredients(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\d+([\/\.\d]*)?", " ", s)
    s = re.sub(UNIT_RX, " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def preprocess_cuisine(c: str) -> str:
    c = str(c).lower().strip()
    c = re.sub(r"\s*,\s*", ",", c)
    c = re.sub(r"\brecipes?\b", "", c).strip()
    if "," in c:  # single-label baseline: keep first
        c = c.split(",")[0]
    return c

def main():
    df = pd.read_parquet(DATA)
    df = df.dropna(subset=["ingredients", "cuisine"]).copy()
    df["ingredients_raw"] = df["ingredients"].astype(str)
    df["ingredients_clean"] = df["ingredients_raw"].map(norm_ingredients)
    df["cuisine"] = df["cuisine"].map(preprocess_cuisine)

    # Keep only cuisines with minimal support to stabilize reports
    vc = df["cuisine"].value_counts()
    keep = vc[vc >= 100].index  # adjust as needed
    if len(keep) < 6:
        keep = vc.head(15).index
    df = df[df["cuisine"].isin(keep)].copy()
    df.reset_index(drop=True, inplace=True)

    # ----------------------
    # TF-IDF + baseline LR
    # ----------------------
    tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=5, max_df=0.7, stop_words="english", max_features=40000)
    X_tfidf = tfidf.fit_transform(df["ingredients_clean"])
    # ran into somezeros could be just from lousy data starting out
    nonzero_mask = (X_tfidf.getnnz(axis=1) > 0)

    # Keep only rows with at least one nonzero feature
    df = df.loc[nonzero_mask].reset_index(drop=True)
    X_tfidf = X_tfidf[nonzero_mask]


    le = LabelEncoder()
    y = le.fit_transform(df["cuisine"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    lr = LogisticRegression(
        max_iter=2000, solver="saga", n_jobs=-1,
        penalty="l2", class_weight="balanced", multi_class="multinomial",
        random_state=RANDOM_STATE
    )
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    rep = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(OUT/"classification_report_tfidf_only.csv")

    # ----------------------
    # Interpretability: top features per cuisine
    # ----------------------
    feat_names = np.array(tfidf.get_feature_names_out())
    top_rows = []
    coefs = lr.coef_  # [n_classes, n_features]
    for i, lab in enumerate(le.classes_):
        top_idx = np.argsort(coefs[i])[-25:][::-1]
        for rank, j in enumerate(top_idx, 1):
            top_rows.append({"cuisine": lab, "rank": rank, "feature": feat_names[j], "coef": float(coefs[i, j])})
    pd.DataFrame(top_rows).to_csv(OUT/"top_features_per_cuisine.csv", index=False)

    # ----------------------
    # Build Ingredient Co-occurrence Network
    # ----------------------
    # Tokenize per recipe (use cleaned)
    token_lists = [t.split() for t in df["ingredients_clean"].tolist()]
    # Count ingredient occurrences & co-occurrences
    ing_counts = Counter()
    co_counts = Counter()
    for toks in token_lists:
        uniq = sorted(set(toks))
        ing_counts.update(uniq)
        for a, b in combinations(uniq, 2):
            co_counts[(a, b)] += 1

    ing_df = pd.DataFrame(ing_counts.items(), columns=["ingredient", "count"]).sort_values("count", ascending=False)
    ing_df.to_csv(OUT/"ingredient_counts.csv", index=False)

    # keep edges above a small threshold to avoid hairball
    EDGE_MIN = max(3, int(0.0005 * len(df)))  # dynamic-ish
    edges = [(a, b, w) for (a, b), w in co_counts.items() if w >= EDGE_MIN]

    G = nx.Graph()
    G.add_nodes_from(ing_df["ingredient"].tolist())
    for a, b, w in edges:
        G.add_edge(a, b, weight=int(w))

    # Centralities & communities
    deg = dict(G.degree())
    btw = nx.betweenness_centrality(G, k=min(500, len(G)), seed=RANDOM_STATE)
    try:
        comms = list(nx.algorithms.community.louvain_communities(G, seed=RANDOM_STATE, weight="weight"))
    except Exception:
        # Fallback: greedy modularity if louvain not available
        comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))

    com_id = {}
    for idx, cset in enumerate(comms):
        for n in cset:
            com_id[n] = idx

    net_rows = []
    for n in G.nodes():
        net_rows.append({
            "ingredient": n,
            "degree": int(deg.get(n, 0)),
            "betweenness": float(btw.get(n, 0.0)),
            "community": int(com_id.get(n, -1)),
            "occurrences": int(ing_counts.get(n, 0))
        })
    net_df = pd.DataFrame(net_rows).sort_values(["community", "degree"], ascending=[True, False])
    net_df.to_csv(OUT/"network_nodes_centrality.csv", index=False)

    # Optional interactive HTML (pyvis) — safe to skip if not installed
    try:
        from pyvis.network import Network
        net = Network(height="720px", width="100%", bgcolor="#ffffff", notebook=False, directed=False)
        net.barnes_hut()
        # Add a subset of high-degree nodes to keep HTML manageable
        keep_nodes = set(net_df.sort_values("degree", ascending=False).head(600)["ingredient"])
        for a, b, w in edges:
            if a in keep_nodes and b in keep_nodes:
                net.add_node(a, title=a)
                net.add_node(b, title=b)
                net.add_edge(a, b, value=w)
        net.show(str(OUT/"ingredient_network.html"))
    except Exception as e:
        with open(OUT/"_pyvis_warning.txt", "w") as f:
            f.write(f"pyvis not available or render failed: {e}\n")

    # ----------------------
    # Graph-derived features per recipe
    #   Sum/mean/max centrality of recipe tokens, one-hot communities
    # ----------------------
    deg_s = net_df.set_index("ingredient")["degree"]
    btw_s = net_df.set_index("ingredient")["betweenness"]
    com_s = net_df.set_index("ingredient")["community"]

    graph_feat = []
    all_coms = sorted(net_df["community"].unique().tolist())
    com_index = {c:i for i, c in enumerate(all_coms)}

    for toks in token_lists:
        toks = [t for t in set(toks) if t in deg_s.index]
        if toks:
            deg_vals = deg_s.loc[toks].values
            btw_vals = btw_s.loc[toks].values
            com_vals = com_s.loc[toks].values
        else:
            deg_vals = np.array([0.0]); btw_vals = np.array([0.0]); com_vals = np.array([-1])

        row = {
            "deg_sum": float(deg_vals.sum()), "deg_mean": float(deg_vals.mean()), "deg_max": float(deg_vals.max()),
            "btw_sum": float(btw_vals.sum()), "btw_mean": float(btw_vals.mean()), "btw_max": float(btw_vals.max())
        }
        # simple community bag
        com_vec = np.zeros(len(all_coms), dtype=float)
        for c in com_vals:
            if c in com_index:
                com_vec[com_index[c]] += 1.0
        # store as ndarray; will hstack later
        graph_feat.append((row, com_vec))

    graph_tab = pd.DataFrame([r for r,_ in graph_feat])
    com_mat = np.vstack([v for _,v in graph_feat])  # [n_samples, n_coms]
    graph_tab.to_csv(OUT/"graph_features_table.csv", index=False)

    # Combine TF-IDF with graph features
    
    X_graph = sparse.hstack([X_tfidf, sparse.csr_matrix(graph_tab.values), sparse.csr_matrix(com_mat)], format="csr")

    Xg_train, Xg_test, yg_train, yg_test = train_test_split(
        X_graph, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    lr_g = LogisticRegression(
        max_iter=2000, solver="saga", n_jobs=-1,
        penalty="l2", class_weight="balanced", multi_class="multinomial",
        random_state=RANDOM_STATE
    )
    lr_g.fit(Xg_train, yg_train)
    ypred_g = lr_g.predict(Xg_test)
    rep_g = classification_report(yg_test, ypred_g, target_names=le.classes_, output_dict=True, zero_division=0)
    pd.DataFrame(rep_g).T.to_csv(OUT/"classification_report_tfidf_plus_graph.csv")

    # ----------------------
    # Improved clustering views
    # ----------------------
    # Cosine-like k-means (normalize vectors)
    Xn = normalize(X_tfidf, norm="l2", copy=True)
    k_vals = [8, 12, 16, 20, 25, 30]
    rows = []
    for k in k_vals:
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, batch_size=4096)
        lab = km.fit_predict(Xn)
        sil = silhouette_score(Xn, lab)
        ari = adjusted_rand_score(y, lab)  # uses your labels as a rough sanity check
        rows.append({"k": k, "silhouette": float(sil), "ari_vs_labels": float(ari)})
    pd.DataFrame(rows).to_csv(OUT/"clustering_quality_kmeans.csv", index=False)

    # Agglomerative clustering (cosine via precomputed 1-cos sim on reduced space)
    # To keep it light, project to 300 dims with TruncatedSVD
    # SVD reduction (as you had)
    
    
    svd = TruncatedSVD(n_components=300, random_state=RANDOM_STATE)
    X_svd = svd.fit_transform(X_tfidf)
    Xs = normalize(X_svd, norm="l2")

    # Drop any residual zero rows (very rare but safe)
    row_norms = np.linalg.norm(Xs, axis=1)
    nz_mask = row_norms > 0
    Xs_nz = Xs[nz_mask]
    y_nz  = y[nz_mask]


        # ----------------------
    # Agglomerative clustering (subset to avoid O(n^2) memory blowup)
    # ----------------------
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score, adjusted_rand_score

    # Current dataset size
    n_samples = Xs_nz.shape[0]

    # Max samples we allow for hierarchical clustering
    max_for_agg = 15000  # adjust based on RAM; 10k–15k is typical

    # Subsample if necessary
    if n_samples > max_for_agg:
        rng = np.random.RandomState(RANDOM_STATE)
        idx_sample = rng.choice(n_samples, size=max_for_agg, replace=False)
        X_agg = Xs_nz[idx_sample]
        y_agg = y_nz[idx_sample]
        print(f"[Agglomerative] Using subset of {max_for_agg} / {n_samples} samples")
    else:
        idx_sample = np.arange(n_samples)
        X_agg = Xs_nz
        y_agg = y_nz
        print(f"[Agglomerative] Using full dataset ({n_samples} samples)")

    # Run the agglomerative model (cosine metric is supported in modern sklearn)
    agg = AgglomerativeClustering(
        n_clusters=20,
        metric="cosine",
        linkage="average"
    )

    agg_lab = agg.fit_predict(X_agg)

    # Metrics computed only on the subset
    sil_agg = silhouette_score(X_agg, agg_lab, metric="cosine")
    ari_agg = adjusted_rand_score(y_agg, agg_lab)

    # Save metrics + cluster membership for subset only
    pd.DataFrame([{
        "k": 20,
        "n_samples_used": len(X_agg),
        "silhouette": float(sil_agg),
        "ari_vs_labels": float(ari_agg)
    }]).to_csv(OUT / "clustering_quality_agglomerative.csv", index=False)

    # Save subset assignments (for visualization)
    pd.DataFrame({
        "index_in_full": idx_sample,
        "cluster": agg_lab,
        "true_label": y_agg
    }).to_csv(OUT / "agg_clusters_subset.csv", index=False)

    print(f"Agglomerative clustering done on subset of size {len(X_agg)}")
    print(f"Silhouette: {sil_agg:.4f}, ARI: {ari_agg:.4f}")

    # ----------------------
    # Pairing recommender via PMI/Lift on co-occurrence
    # ----------------------
    N_recipes = len(df)
    # probabilities
    p_ing = {ing: cnt / N_recipes for ing, cnt in ing_counts.items() if cnt >= 3}
    pair_rows = []
    for (a,b), w in co_counts.items():
        if w < EDGE_MIN:
            continue
        pa = p_ing.get(a); pb = p_ing.get(b)
        if not pa or not pb:
            continue
        pab = w / N_recipes
        lift = pab / (pa * pb)
        pmi = np.log2(lift) if lift > 0 else 0.0
        pair_rows.append({"a": a, "b": b, "co_count": int(w), "lift": float(lift), "pmi": float(pmi)})

    pair_df = pd.DataFrame(pair_rows).sort_values(["pmi", "co_count"], ascending=[False, False])
    pair_df.to_csv(OUT/"pairings_pmi.csv", index=False)

    # Example: top suggested pairings per cuisine (based on ingredients present in that cuisine)
    # Build quick cuisine→ingredient frequency map
    cui_ing = defaultdict(Counter)
    for toks, lab in zip(token_lists, df["cuisine"].tolist()):
        for t in set(toks):
            cui_ing[lab][t] += 1
    # For each cuisine: top ingredients and their best partners by PMI
    sugg_rows = []
    pair_idx = {(row.a, row.b): (row.lift, row.pmi) for row in pair_df.itertuples(index=False)}
    pair_idx.update({(b,a):(l,p) for (a,b),(l,p) in pair_idx.items()})
    for lab, cnts in cui_ing.items():
        top_ings = [w for w,_ in cnts.most_common(50)]
        for ing in top_ings:
            best = []
            for other in top_ings:
                if other == ing: continue
                if (ing, other) in pair_idx:
                    lift, pmi = pair_idx[(ing, other)]
                    best.append((other, lift, pmi))
            best.sort(key=lambda x: (x[2], x[1]), reverse=True)
            for partner, lift, pmi in best[:5]:
                sugg_rows.append({"cuisine": lab, "ingredient": ing, "partner": partner, "lift": float(lift), "pmi": float(pmi)})
    pd.DataFrame(sugg_rows).to_csv(OUT/"pairings_by_cuisine.csv", index=False)

    # ----------------------
    # Summary
    # ----------------------
    def safe_macro_f1(report_path):
        path = OUT / report_path
        if not path.exists():
            return None
        df = pd.read_csv(path)
        # Handle either format (label as column or index)
        if "label" in df.columns:
            row = df.loc[df["label"] == "macro avg"]
            if not row.empty:
                return float(row["f1-score"].iloc[0])
        elif "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "label"})
            row = df.loc[df["label"] == "macro avg"]
            if not row.empty:
                return float(row["f1-score"].iloc[0])
        else:
            # fallback: try using index if CSV saved with index only
            df = pd.read_csv(path, index_col=0)
            if "macro avg" in df.index:
                return float(df.loc["macro avg", "f1-score"])
        return None

    
    summary = {
        "tfidf_only_macro_f1": safe_macro_f1("classification_report_tfidf_only.csv"),
        "tfidf_plus_graph_macro_f1": safe_macro_f1("classification_report_tfidf_plus_graph.csv"),
        "kmeans_grid": "see clustering_quality_kmeans.csv",
        "agglomerative_silhouette_subset": float(sil_agg),
        "agglomerative_ari_subset": float(ari_agg),
        "artifacts": [
            "top_features_per_cuisine.csv",
            "ingredient_counts.csv",
            "network_nodes_centrality.csv",
            "ingredient_network.html (if generated)",
            "graph_features_table.csv",
            "pairings_pmi.csv",
            "pairings_by_cuisine.csv",
            "agg_clusters_subset.csv"
        ]
    }


    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print("Phase-2 artifacts written to:", OUT.resolve())

if __name__ == "__main__":
    main()
