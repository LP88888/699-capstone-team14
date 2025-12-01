import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from collections import Counter, defaultdict
from scipy import sparse

import networkx as nx

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, silhouette_score, adjusted_rand_score
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering

from ..core import PipelineContext, StageResult
from ..utils import stage_logger
from .recommender import CuisineRecommender

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

def safe_macro_f1(report_path: Path) -> float:
    if not report_path.exists():
        return None
    df = pd.read_csv(report_path)
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
        try:
            df = pd.read_csv(report_path, index_col=0)
            if "macro avg" in df.index:
                return float(df.loc["macro avg", "f1-score"])
        except:
            pass
    return None

def run(context: PipelineContext, **kwargs) -> StageResult:
    logger = stage_logger(context, "analysis_graph")
    config = context.stage("analysis").get("graph", {})
    
    # Config/Overrides
    input_path = kwargs.get("input_path", config.get("input_path"))
    output_dir = Path(kwargs.get("output_dir", config.get("output_dir", "./reports/graph")))
    edge_min = kwargs.get("edge_min_support", config.get("edge_min_support", 3))
    
    if not input_path:
        raise ValueError("Input path not specified for analysis_graph")

    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading data from {input_path}")
    if str(input_path).endswith(".csv"):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_parquet(input_path)
    
    if "ingredients" not in df.columns or "cuisine" not in df.columns:
        # fallback for encoded dataset columns
        if "cuisine_clean" in df.columns:
            df["cuisine"] = df["cuisine_clean"]
    
    df = df.dropna(subset=["ingredients", "cuisine"]).copy()
    
    # Ensure ingredients are strings
    if not df.empty and isinstance(df["ingredients"].iloc[0], (list, np.ndarray)):
         df["ingredients"] = df["ingredients"].apply(lambda x: " ".join(map(str, x)))

    df["ingredients_raw"] = df["ingredients"].astype(str)
    df["ingredients_clean"] = df["ingredients_raw"].map(norm_ingredients)
    df["cuisine"] = df["cuisine"].map(preprocess_cuisine)

    # Keep only cuisines with minimal support
    vc = df["cuisine"].value_counts()
    keep = vc[vc >= 100].index
    if len(keep) < 6:
        keep = vc.head(15).index
    df = df[df["cuisine"].isin(keep)].copy()
    df.reset_index(drop=True, inplace=True)
    
    logger.info(f"Data ready: {len(df)} recipes, {df['cuisine'].nunique()} cuisines")
    
    # TF-IDF
    logger.info("Computing TF-IDF...")
    tfidf = TfidfVectorizer(
        analyzer="word", ngram_range=(1,2), min_df=5, max_df=0.7, 
        stop_words="english", max_features=40000
    )
    X_tfidf = tfidf.fit_transform(df["ingredients_clean"])
    nonzero_mask = (X_tfidf.getnnz(axis=1) > 0)
    df = df.loc[nonzero_mask].reset_index(drop=True)
    X_tfidf = X_tfidf[nonzero_mask]
    
    le = LabelEncoder()
    y = le.fit_transform(df["cuisine"])
    
    # Baseline LR
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    lr = LogisticRegression(
        max_iter=2000, solver="saga", n_jobs=-1,
        penalty="l2", class_weight="balanced", multi_class="multinomial",
        random_state=42
    )
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    rep = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(output_dir/"classification_report_tfidf_only.csv")
    
    # Graph Construction
    logger.info("Building Ingredient Co-occurrence Network...")
    token_lists = [t.split() for t in df["ingredients_clean"].tolist()]
    ing_counts = Counter()
    co_counts = Counter()
    for toks in token_lists:
        uniq = sorted(set(toks))
        ing_counts.update(uniq)
        for a, b in combinations(uniq, 2):
            co_counts[(a, b)] += 1
            
    ing_df = pd.DataFrame(ing_counts.items(), columns=["ingredient", "count"]).sort_values("count", ascending=False)
    ing_df.to_csv(output_dir/"ingredient_counts.csv", index=False)
    
    # Dynamic edge threshold if not strict
    if edge_min <= 3:
        computed_min = max(3, int(0.0005 * len(df)))
        edge_min = max(edge_min, computed_min)
        
    logger.info(f"Using edge weight threshold: {edge_min}")
    edges = [(a, b, w) for (a, b), w in co_counts.items() if w >= edge_min]
    
    G = nx.Graph()
    G.add_nodes_from(ing_counts.keys())
    for a, b, w in edges:
        G.add_edge(a, b, weight=int(w))
        
    # Centrality
    logger.info("Computing Centrality and Communities...")
    deg = dict(G.degree())
    btw = nx.betweenness_centrality(G, k=min(500, len(G)), seed=42)
    try:
        comms = list(nx.algorithms.community.louvain_communities(G, seed=42, weight="weight"))
    except Exception:
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
    net_df.to_csv(output_dir/"network_nodes_centrality.csv", index=False)
    
    # Graph Features for Classification (Integration)
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
        com_vec = np.zeros(len(all_coms), dtype=float)
        for c in com_vals:
            if c in com_index:
                com_vec[com_index[c]] += 1.0
        graph_feat.append((row, com_vec))
        
    graph_tab = pd.DataFrame([r for r,_ in graph_feat])
    com_mat = np.vstack([v for _,v in graph_feat])
    graph_tab.to_csv(output_dir/"graph_features_table.csv", index=False)
    
    X_graph = sparse.hstack([X_tfidf, sparse.csr_matrix(graph_tab.values), sparse.csr_matrix(com_mat)], format="csr")
    
    logger.info("Training Graph-Enhanced Classifier...")
    Xg_train, Xg_test, yg_train, yg_test = train_test_split(
        X_graph, y, test_size=0.2, random_state=42, stratify=y
    )
    lr_g = LogisticRegression(
        max_iter=2000, solver="saga", n_jobs=-1,
        penalty="l2", class_weight="balanced", multi_class="multinomial",
        random_state=42
    )
    lr_g.fit(Xg_train, yg_train)
    ypred_g = lr_g.predict(Xg_test)
    rep_g = classification_report(yg_test, ypred_g, target_names=le.classes_, output_dict=True, zero_division=0)
    pd.DataFrame(rep_g).T.to_csv(output_dir/"classification_report_tfidf_plus_graph.csv")
    
    # Save Summary
    summary = {
        "tfidf_only_macro_f1": safe_macro_f1(output_dir/"classification_report_tfidf_only.csv"),
        "tfidf_plus_graph_macro_f1": safe_macro_f1(output_dir/"classification_report_tfidf_plus_graph.csv"),
        "nodes": len(G.nodes()),
        "edges": len(G.edges()),
        "communities": len(all_coms)
    }
    json.dump(summary, open(output_dir / "summary.json", "w"), indent=2)
    
    # Initialize Recommender and save sample fusions
    cui_ing = defaultdict(Counter)
    for toks, lab in zip(token_lists, df["cuisine"].tolist()):
        for t in set(toks):
            cui_ing[lab][t] += 1
            
    recommender = CuisineRecommender(
        graph=G,
        tfidf_matrix=X_tfidf,
        tfidf_vectorizer=tfidf,
        cuisine_ingredient_map=dict(cui_ing),
        ingredient_counts=ing_counts,
        centrality_dict=dict(btw)
    )
    
    # Example fusion: Italian + Japanese
    try:
        fusion_out = output_dir / "fusion_samples"
        fusion_out.mkdir(exist_ok=True)
        recommender.save_fusion_report("italian", "japanese", output_path=fusion_out/"fusion_italian_japanese.json")
    except Exception as e:
        logger.warning(f"Could not generate sample fusion report: {e}")
        
    logger.info(f"Graph analysis completed. Artifacts in {output_dir}")
    return StageResult(name="analysis_graph", status="success", artifacts={"output_dir": str(output_dir)})

