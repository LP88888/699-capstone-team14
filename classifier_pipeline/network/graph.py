"""
Graph data structures and utilities for ingredient network.
"""

from typing import Dict, List, Set, Tuple, Optional
import networkx as nx
import logging

logger = logging.getLogger(__name__)


class IngredientGraph:
    """Wrapper around NetworkX graph with ingredient-specific utilities."""
    
    def __init__(self, graph: nx.Graph):
        """
        Initialize with a NetworkX graph.
        
        Args:
            graph: NetworkX graph with ingredients as nodes
        """
        self.graph = graph
        self._node_cache: Dict[int, Dict] = {}
        self._edge_cache: Dict[Tuple[int, int], Dict] = {}
    
    @property
    def num_nodes(self) -> int:
        """Number of ingredient nodes."""
        return self.graph.number_of_nodes()
    
    @property
    def num_edges(self) -> int:
        """Number of co-occurrence edges."""
        return self.graph.number_of_edges()
    
    def get_ingredient_neighbors(self, ingredient_id: int) -> List[int]:
        """
        Get neighboring ingredients (ingredients that co-occur).
        
        Args:
            ingredient_id: Ingredient ID
            
        Returns:
            List of neighboring ingredient IDs
        """
        if ingredient_id not in self.graph:
            return []
        return list(self.graph.neighbors(ingredient_id))
    
    def get_edge_weight(self, ing1: int, ing2: int) -> float:
        """
        Get edge weight between two ingredients.
        
        Args:
            ing1: First ingredient ID
            ing2: Second ingredient ID
            
        Returns:
            Edge weight (0.0 if no edge exists)
        """
        if not self.graph.has_edge(ing1, ing2):
            return 0.0
        return self.graph[ing1][ing2].get('weight', 0.0)
    
    def get_ingredient_degree(self, ingredient_id: int) -> int:
        """
        Get degree (number of connections) for an ingredient.
        
        Args:
            ingredient_id: Ingredient ID
            
        Returns:
            Degree (number of neighbors)
        """
        if ingredient_id not in self.graph:
            return 0
        return self.graph.degree(ingredient_id)
    
    def find_shortest_path(self, ing1: int, ing2: int) -> Optional[List[int]]:
        """
        Find shortest path between two ingredients.
        
        Args:
            ing1: Starting ingredient ID
            ing2: Target ingredient ID
            
        Returns:
            List of ingredient IDs forming the path, or None if no path exists
        """
        try:
            path = nx.shortest_path(self.graph, ing1, ing2)
            return path
        except nx.NetworkXNoPath:
            return None
    
    def get_ingredient_subgraph(self, ingredient_ids: Set[int]) -> nx.Graph:
        """
        Extract subgraph containing only specified ingredients.
        
        Args:
            ingredient_ids: Set of ingredient IDs to include
            
        Returns:
            Subgraph containing only specified ingredients and edges between them
        """
        return self.graph.subgraph(ingredient_ids).copy()
    
    def find_bridge_ingredients(
        self,
        ingredient_set1: Set[int],
        ingredient_set2: Set[int],
        max_distance: int = 2
    ) -> List[Tuple[int, float]]:
        """
        Find ingredients that bridge two ingredient sets.
        
        Optimized implementation using multi-source Dijkstra to compute
        distances from all nodes in each set in a single pass.
        
        Args:
            ingredient_set1: First set of ingredient IDs
            ingredient_set2: Second set of ingredient IDs
            max_distance: Maximum path length to consider
            
        Returns:
            List of (ingredient_id, bridge_score) tuples, sorted by score
        """
        # Filter sets to only include nodes that exist in the graph
        set1 = {ing for ing in ingredient_set1 if ing in self.graph}
        set2 = {ing for ing in ingredient_set2 if ing in self.graph}
        
        if not set1 or not set2:
            return []
        
        bridge_scores = {}
        
        # Step A: Calculate distances from all nodes in set1 to the rest of the graph
        # This is a single call that computes shortest paths from multiple sources
        dists_from_set1 = {}
        try:
            # multi_source_dijkstra_path_length returns a dict: {node: distance}
            # It computes shortest paths from all sources in set1 to all reachable nodes
            for node, dist in nx.multi_source_dijkstra_path_length(
                self.graph, set1, cutoff=max_distance
            ).items():
                # Only consider nodes not in the sets themselves
                if node not in set1 and node not in set2:
                    dists_from_set1[node] = dist
        except Exception as e:
            logger.warning(f"Error computing distances from set1: {e}")
            return []
        
        # Step B: Calculate distances from all nodes in set2 to the rest of the graph
        dists_from_set2 = {}
        try:
            for node, dist in nx.multi_source_dijkstra_path_length(
                self.graph, set2, cutoff=max_distance
            ).items():
                # Only consider nodes not in the sets themselves
                if node not in set1 and node not in set2:
                    dists_from_set2[node] = dist
        except Exception as e:
            logger.warning(f"Error computing distances from set2: {e}")
            return []
        
        # Step C: Iterate through nodes that appear in both dictionaries
        # These are the bridge ingredients (reachable from both sets)
        for bridge_ing in dists_from_set1.keys() & dists_from_set2.keys():
            dist_to_set1 = dists_from_set1[bridge_ing]
            dist_to_set2 = dists_from_set2[bridge_ing]
            
            # Bridge score: inverse of combined distance
            # Add 1 to avoid division by zero and to make score more meaningful
            score = 1.0 / (dist_to_set1 + dist_to_set2 + 1)
            bridge_scores[bridge_ing] = score
        
        # Sort by score (highest first)
        return sorted(bridge_scores.items(), key=lambda x: x[1], reverse=True)
    
    def compute_recipe_similarity(self, recipe1: List[int], recipe2: List[int]) -> float:
        """
        Compute similarity between two recipes based on network connectivity.
        
        Optimized implementation using subgraph extraction to avoid
        nested loops and individual edge lookups.
        
        Args:
            recipe1: List of ingredient IDs for first recipe
            recipe2: List of ingredient IDs for second recipe
            
        Returns:
            Similarity score [0, 1]
        """
        set1 = set(recipe1)
        set2 = set(recipe2)
        
        # Jaccard similarity on ingredient sets
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        jaccard = intersection / union if union > 0 else 0.0
        
        # Network-based similarity: average edge weights between recipes
        # Optimize by extracting subgraph first, then iterating over its edges
        combined_set = set1 | set2
        
        # Filter to only include nodes that exist in the graph
        combined_set = {ing for ing in combined_set if ing in self.graph}
        
        if not combined_set:
            # No valid ingredients, return Jaccard only
            return jaccard
        
        # Extract subgraph containing only ingredients from both recipes
        # This is much more efficient than checking every pair
        subgraph = self.graph.subgraph(combined_set)
        
        # Iterate over edges in the subgraph
        # Only count edges that connect ingredients from different recipes
        network_sim = 0.0
        edge_count = 0
        
        for ing1, ing2 in subgraph.edges():
            # Check if this edge connects ingredients from different recipes
            ing1_in_set1 = ing1 in set1
            ing1_in_set2 = ing1 in set2
            ing2_in_set1 = ing2 in set1
            ing2_in_set2 = ing2 in set2
            
            # Edge connects set1 and set2 if:
            # - ing1 in set1 and ing2 in set2, OR
            # - ing1 in set2 and ing2 in set1
            if (ing1_in_set1 and ing2_in_set2) or (ing1_in_set2 and ing2_in_set1):
                weight = subgraph[ing1][ing2].get('weight', 0.0)
                if weight > 0:
                    network_sim += weight
                    edge_count += 1
        
        network_sim = network_sim / edge_count if edge_count > 0 else 0.0
        
        # Combine Jaccard and network similarity
        combined = 0.5 * jaccard + 0.5 * network_sim
        return combined

