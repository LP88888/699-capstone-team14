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
        """
        Get top N ingredients for a cuisine based on frequency.
        """
        cuisine_lower = cuisine.lower().strip()
        
        # Find matching cuisine
        matched_cuisine = None
        for c in self.cuisines:
            if c.lower() == cuisine_lower:
                matched_cuisine = c
                break
        
        if matched_cuisine is None:
            available = ", ".join(self.cuisines[:10])
            raise ValueError(
                f"Cuisine '{cuisine}' not found. Available cuisines: {available}..."
            )
        
        counter = self.cuisine_ingredients[matched_cuisine]
        return counter.most_common(n)
    
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
        top_a = set(ing for ing, _ in self.get_top_ingredients(cuisine_a, top_n))
        top_b = set(ing for ing, _ in self.get_top_ingredients(cuisine_b, top_n))
        
        bridges = []
        all_nodes = set(self.G.nodes())
        
        for node in all_nodes:
            if node not in self.G:
                continue
            
            neighbors = set(self.G.neighbors(node))
            edges_to_a = neighbors & top_a
            edges_to_b = neighbors & top_b
            
            if edges_to_a and edges_to_b:
                # Bridge score: product of connection counts, weighted by centrality
                bridge_score = (
                    len(edges_to_a) * len(edges_to_b) * 
                    (1 + self.centrality.get(node, 0))
                )
                bridges.append((
                    node, 
                    bridge_score, 
                    list(edges_to_a), 
                    list(edges_to_b)
                ))
        
        # Sort by bridge score descending
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
        if not all_a:
            for c in self.cuisines:
                 if c.lower() == cuisine_a.lower():
                     all_a = set(self.cuisine_ingredients[c].keys())
                     break
        if not all_b:
            for c in self.cuisines:
                 if c.lower() == cuisine_b.lower():
                     all_b = set(self.cuisine_ingredients[c].keys())
                     break

        # Ingredients in A but not in B
        novel_candidates = all_a - all_b
        
        # Get high-centrality ingredients in B
        top_b_by_centrality = self.get_top_ingredients_by_centrality(cuisine_b, top_n)
        if not top_b_by_centrality:
            return []
        
        # Determine centrality threshold
        centrality_values = [c for _, c in top_b_by_centrality]
        if not centrality_values:
             return []
             
        threshold_value = np.percentile(centrality_values, centrality_threshold * 100)
        
        high_centrality_b = {
            ing for ing, cent in top_b_by_centrality 
            if cent >= threshold_value
        }
        
        # Find novel ingredients that share edges with high-centrality B ingredients
        novel_suggestions = []
        
        for novel_ing in novel_candidates:
            if novel_ing not in self.G:
                continue
            
            neighbors = set(self.G.neighbors(novel_ing))
            high_cent_partners = neighbors & high_centrality_b
            
            for partner in high_cent_partners:
                edge_weight = self.G[novel_ing][partner].get("weight", 1)
                partner_centrality = self.centrality.get(partner, 0)
                
                novel_suggestions.append((
                    novel_ing,
                    partner,
                    partner_centrality,
                    edge_weight
                ))
        
        # Sort by partner centrality * edge weight
        novel_suggestions.sort(
            key=lambda x: x[2] * x[3], 
            reverse=True
        )
        
        return novel_suggestions
    
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
        top_n = max(25, top_n)
        
        # Centrality threshold based on strictness
        centrality_threshold = strictness * 0.3 + 0.3  # Range: 0.3-0.6
        
        # Get top ingredients
        try:
            top_a = self.get_top_ingredients(cuisine_a, top_n)
            top_b = self.get_top_ingredients(cuisine_b, top_n)
        except ValueError as e:
            return {"error": str(e)}
        
        top_a_set = set(ing for ing, _ in top_a)
        top_b_set = set(ing for ing, _ in top_b)
        
        # Find shared ingredients
        shared = top_a_set & top_b_set
        
        # Find bridge ingredients
        bridges = self.find_bridge_ingredients(cuisine_a, cuisine_b, top_n)
        
        # Filter bridges based on strictness
        min_bridge_score = strictness * max(b[1] for b in bridges) if bridges else 0
        bridges_filtered = [
            b for b in bridges 
            if b[1] >= min_bridge_score * 0.1
        ]
        
        # Find novel ingredients in both directions
        novel_from_a = self.find_novel_ingredients(
            cuisine_a, cuisine_b, top_n, centrality_threshold
        )
        novel_from_b = self.find_novel_ingredients(
            cuisine_b, cuisine_a, top_n, centrality_threshold
        )
        
        # Limit results based on strictness
        max_results = int(20 * (1 - strictness * 0.5))  # Range: 10-20
        max_results = max(10, max_results)
        
        return {
            "cuisine_a": cuisine_a,
            "cuisine_b": cuisine_b,
            "top_a": top_a[:top_n],
            "top_b": top_b[:top_n],
            "bridge_ingredients": [
                {
                    "ingredient": b[0],
                    "bridge_score": b[1],
                    "connects_to_a": b[2][:5],  # Limit for readability
                    "connects_to_b": b[3][:5]
                }
                for b in bridges_filtered[:max_results]
            ],
            "novel_from_a": [
                {
                    "ingredient": n[0],
                    "pairs_well_with": n[1],
                    "partner_centrality": n[2],
                    "edge_strength": n[3]
                }
                for n in novel_from_a[:max_results]
            ],
            "novel_from_b": [
                {
                    "ingredient": n[0],
                    "pairs_well_with": n[1],
                    "partner_centrality": n[2],
                    "edge_strength": n[3]
                }
                for n in novel_from_b[:max_results]
            ],
            "shared_ingredients": list(shared),
            "parameters": {
                "strictness": strictness,
                "top_n_used": top_n,
                "centrality_threshold": centrality_threshold
            }
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
