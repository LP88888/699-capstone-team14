import numpy as np
import networkx as nx
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Dict, Any, Set
from pathlib import Path
import json

class CuisineRecommender:
    """
    Recommender for fusion cuisine suggestions based on ingredient co-occurrence graph
    and TF-IDF vectors.
    
    Identifies bridge ingredients (shared between cuisines) and novel ingredients
    (new flavor pairings) for fusion recipe development.
    """
    
    def __init__(
        self, 
        graph: nx.Graph, 
        tfidf_matrix, 
        tfidf_vectorizer: TfidfVectorizer,
        cuisine_ingredient_map: dict,
        ingredient_counts: Counter,
        centrality_dict: dict = None,
        random_state: int = 42
    ):
        """
        Initialize the CuisineRecommender.
        
        Args:
            graph: NetworkX graph of ingredient co-occurrences
            tfidf_matrix: Fitted TF-IDF sparse matrix
            tfidf_vectorizer: Fitted TfidfVectorizer
            cuisine_ingredient_map: Dict mapping cuisine -> Counter of ingredients
            ingredient_counts: Global Counter of ingredient occurrences
            centrality_dict: Optional dict of ingredient -> betweenness centrality
        """
        self.G = graph
        self.tfidf_matrix = tfidf_matrix
        self.tfidf_vectorizer = tfidf_vectorizer
        self.cuisine_ingredients = cuisine_ingredient_map
        self.ingredient_counts = ingredient_counts
        self.random_state = random_state
        
        # Compute centrality if not provided
        if centrality_dict is None:
            # Check if graph is too large for full betweenness
            k = min(500, len(self.G))
            self.centrality = nx.betweenness_centrality(
                self.G, k=k, seed=self.random_state
            )
        else:
            self.centrality = centrality_dict
        
        # Precompute degree for each node
        self.degree = dict(self.G.degree())
        
        # Get all cuisines
        self.cuisines = list(self.cuisine_ingredients.keys())
    
    def get_top_ingredients(self, cuisine: str, n: int = 50) -> List[Tuple[str, int]]:
        """Get top N ingredients for a cuisine based on frequency."""
        # Case-insensitive lookup
        c_map = {k.lower(): k for k in self.cuisines}
        key = c_map.get(cuisine.lower().strip())
        
        if not key:
            raise ValueError(f"Cuisine '{cuisine}' not found.")
                
        return self.cuisine_ingredients[key].most_common(n)
        
    def get_top_ingredients_by_centrality(self, cuisine: str, n: int = 50) -> List[Tuple[str, float]]:
        """
        Get top N ingredients for a cuisine ranked by centrality in the graph.
        """
        top_freq = self.get_top_ingredients(cuisine, n=n*2)  # Get more, then filter
        top_ingredients = [ing for ing, _ in top_freq]
        
        # Filter to those in graph and sort by centrality
        in_graph = [(ing, self.centrality.get(ing, 0)) for ing in top_ingredients if ing in self.G]
        in_graph.sort(key=lambda x: x[1], reverse=True)
        
        return in_graph[:n]
    
    def find_bridge_ingredients(
        self, 
        cuisine_a: str, 
        cuisine_b: str, 
        top_n: int = 50
    ) -> List[Tuple[str, float, List[str], List[str]]]:
        """
        Find bridge ingredients: nodes that have edges connecting to both 
        cuisine_a's and cuisine_b's top ingredients.
        """
        # Get top ingredients for each cuisine
        top_a_ings = {ing for ing, _ in self.get_top_ingredients(cuisine_a, top_n)}
        top_b_ings = {ing for ing, _ in self.get_top_ingredients(cuisine_b, top_n)}
        
        # 1. Get neighbors of Cuisine A's top ingredients
        # 2. Get neighbors of Cuisine B's top ingredients
        # 3. The intersection is the set of potential bridges
        neighbors_a = set()
        for ing in top_a_ings:
            if ing in self.G:
                neighbors_a.update(self.G.neighbors(ing))
                
        neighbors_b = set()
        for ing in top_b_ings:
            if ing in self.G:
                neighbors_b.update(self.G.neighbors(ing))
                
        candidates = neighbors_a.intersection(neighbors_b)
        
        bridges = []
        for node in candidates:
            # Re-verify connections to specific top lists for scoring
            node_neighbors = set(self.G.neighbors(node))
            edges_to_a = node_neighbors & top_a_ings
            edges_to_b = node_neighbors & top_b_ings
            
            if edges_to_a and edges_to_b:
                # Score = Connectivity * Centrality
                score = len(edges_to_a) * len(edges_to_b) * (1 + self.centrality.get(node, 0))
                bridges.append((node, score, list(edges_to_a), list(edges_to_b)))
        
        bridges.sort(key=lambda x: x[1], reverse=True)
        return bridges
    
    def find_novel_ingredients(
        self, 
        cuisine_a: str, 
        cuisine_b: str, 
        top_n: int = 50,
        centrality_threshold: float = 0.5
    ) -> List[Tuple[str, str, float, float]]:
        """
        Find novel ingredients: ingredients in cuisine_a that have never appeared 
        in cuisine_b but share an edge with a high-centrality ingredient in cuisine_b.
        """
        # Get all ingredients for each cuisine
        all_a = set(self.cuisine_ingredients.get(cuisine_a.lower(), {}).keys())
        all_b = set(self.cuisine_ingredients.get(cuisine_b.lower(), {}).keys())
        
        # Also try case-insensitive matching if direct lookup failed
        c_map = {k.lower(): k for k in self.cuisines}
        dict_a = self.cuisine_ingredients.get(c_map.get(cuisine_a.lower(), ""), {})
        dict_b = self.cuisine_ingredients.get(c_map.get(cuisine_b.lower(), ""), {})
        
        all_a = set(dict_a.keys())
        all_b = set(dict_b.keys())
        
        # Ingredients unique to A
        novel_candidates = all_a - all_b
        
        # High centrality items in B
        top_b_freq = [i[0] for i in self.cuisine_ingredients[c_map[cuisine_b.lower()]].most_common(top_n)]
        top_b_cent = [(i, self.centrality.get(i, 0)) for i in top_b_freq if i in self.G]
        
        if not top_b_cent: return []
        
        # Percentile threshold
        vals = [v for k, v in top_b_cent]
        thresh = np.percentile(vals, centrality_threshold * 100)
        high_cent_b = {k for k, v in top_b_cent if v >= thresh}
        
        suggestions = []
        for novel in novel_candidates:
            if novel not in self.G: continue
            
            # Fast neighbor check
            neighbors = set(self.G.neighbors(novel))
            matches = neighbors.intersection(high_cent_b)
            
            for partner in matches:
                weight = self.G[novel][partner].get("weight", 1)
                cent = self.centrality.get(partner, 0)
                suggestions.append((novel, partner, cent, weight))
                
        suggestions.sort(key=lambda x: x[2] * x[3], reverse=True)
        return suggestions
    
    def suggest_fusion(
        self, 
        cuisine_a: str, 
        cuisine_b: str, 
        strictness: float = 0.5
    ) -> dict:
        """
        Suggest fusion ingredients for combining two cuisines.
        """
        # Adjust top_n based on strictness (stricter = fewer ingredients considered)
        top_n = int(50 * (1 - strictness * 0.5))  # Range: 25-50
        top_n = max(25, int(50 * (1 - strictness * 0.5)))
        bridges = self.find_bridge_ingredients(cuisine_a, cuisine_b, top_n)
        novel_a = self.find_novel_ingredients(cuisine_a, cuisine_b, top_n)
        novel_b = self.find_novel_ingredients(cuisine_b, cuisine_a, top_n)
        
        return {
            "pair": f"{cuisine_a} + {cuisine_b}",
            "bridges": [
                {"name": b[0], "score": b[1], "connects_a": b[2][:3], "connects_b": b[3][:3]}
                for b in bridges[:10]
            ],
            "novel_from_a": [
                {"name": n[0], "pairs_with": n[1], "strength": n[3]}
                for n in novel_a[:10]
            ],
            "novel_from_b": [
                {"name": n[0], "pairs_with": n[1], "strength": n[3]}
                for n in novel_b[:10]
            ]
        }

    def save_fusion_report(
        self, 
        cuisine_a: str, 
        cuisine_b: str, 
        strictness: float = 0.5,
        output_path: Path = None
    ) -> Path:
        """
        Generate and save a fusion report to JSON.
        
        Args:
            cuisine_a: First cuisine
            cuisine_b: Second cuisine
            strictness: Filtering strictness (0-1)
            output_path: Optional output path
            
        Returns:
            Path to saved report
        """
        result = self.suggest_fusion(cuisine_a, cuisine_b, strictness)
        
        if output_path is None:
            # Fallback if no path provided
            output_path = Path(f"fusion_{cuisine_a}_{cuisine_b}.json")
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return output_path
