"""
Suggest fusion/infusion recipes by combining ingredients from different cuisines.
"""

from typing import List, Dict, Set, Tuple, Optional
import logging

from ..network.graph import IngredientGraph

logger = logging.getLogger(__name__)


class InfusionSuggestor:
    """Suggest fusion recipes using ingredient network."""
    
    def __init__(self, graph: IngredientGraph):
        """
        Initialize suggestor with ingredient graph.
        
        Args:
            graph: IngredientGraph instance
        """
        self.graph = graph
    
    def suggest_fusion_recipes(
        self,
        cuisine1_ingredients: Set[int],
        cuisine2_ingredients: Set[int],
        max_suggestions: int = 10,
        min_ingredients: int = 5,
        max_ingredients: int = 15,
    ) -> List[Dict]:
        """
        Suggest fusion recipes combining ingredients from two cuisines.
        
        Args:
            cuisine1_ingredients: Set of ingredient IDs from first cuisine
            cuisine2_ingredients: Set of ingredient IDs from second cuisine
            max_suggestions: Maximum number of suggestions to return
            min_ingredients: Minimum ingredients per suggestion
            max_ingredients: Maximum ingredients per suggestion
            
        Returns:
            List of suggestion dictionaries with:
            - ingredients: List of ingredient IDs
            - score: Fusion score
            - cuisine1_ratio: Ratio of ingredients from cuisine 1
            - cuisine2_ratio: Ratio of ingredients from cuisine 2
        """
        logger.info(f"Suggesting fusion recipes between {len(cuisine1_ingredients)} and {len(cuisine2_ingredients)} ingredients")
        
        # Find bridge ingredients
        bridges = self.graph.find_bridge_ingredients(cuisine1_ingredients, cuisine2_ingredients)
        
        suggestions = []
        
        # Strategy 1: Combine ingredients with bridge connections
        for bridge_ing, bridge_score in bridges[:max_suggestions]:
            # Get neighbors of bridge ingredient
            neighbors = set(self.graph.get_ingredient_neighbors(bridge_ing))
            
            # Select ingredients from both cuisines that connect to bridge
            from_cuisine1 = list((neighbors & cuisine1_ingredients))[:max_ingredients // 2]
            from_cuisine2 = list((neighbors & cuisine2_ingredients))[:max_ingredients // 2]
            
            if len(from_cuisine1) + len(from_cuisine2) >= min_ingredients:
                ingredients = from_cuisine1 + from_cuisine2 + [bridge_ing]
                score = self._compute_fusion_score(ingredients, cuisine1_ingredients, cuisine2_ingredients)
                
                suggestions.append({
                    'ingredients': ingredients,
                    'score': score,
                    'cuisine1_ratio': len(from_cuisine1) / len(ingredients),
                    'cuisine2_ratio': len(from_cuisine2) / len(ingredients),
                    'bridge_ingredient': bridge_ing,
                })
        
        # Strategy 2: Direct combination with network connectivity
        # Select ingredients that have strong connections across cuisines
        cross_cuisine_edges = []
        for ing1 in cuisine1_ingredients:
            for ing2 in cuisine2_ingredients:
                weight = self.graph.get_edge_weight(ing1, ing2)
                if weight > 0:
                    cross_cuisine_edges.append((ing1, ing2, weight))
        
        # Sort by edge weight
        cross_cuisine_edges.sort(key=lambda x: x[2], reverse=True)
        
        # Create suggestions from top cross-cuisine connections
        used_pairs = set()
        for ing1, ing2, weight in cross_cuisine_edges[:max_suggestions]:
            if (ing1, ing2) in used_pairs or (ing2, ing1) in used_pairs:
                continue
            used_pairs.add((ing1, ing2))
            
            # Get neighbors of both ingredients
            neighbors1 = set(self.graph.get_ingredient_neighbors(ing1))
            neighbors2 = set(self.graph.get_ingredient_neighbors(ing2))
            
            # Select balanced mix
            from_cuisine1 = list((neighbors1 & cuisine1_ingredients))[:max_ingredients // 2]
            from_cuisine2 = list((neighbors2 & cuisine2_ingredients))[:max_ingredients // 2]
            
            if len(from_cuisine1) + len(from_cuisine2) >= min_ingredients - 2:
                ingredients = from_cuisine1 + from_cuisine2 + [ing1, ing2]
                score = self._compute_fusion_score(ingredients, cuisine1_ingredients, cuisine2_ingredients)
                
                suggestions.append({
                    'ingredients': ingredients,
                    'score': score,
                    'cuisine1_ratio': len(from_cuisine1 + [ing1]) / len(ingredients),
                    'cuisine2_ratio': len(from_cuisine2 + [ing2]) / len(ingredients),
                    'bridge_ingredient': None,
                })
        
        # Sort by score and return top suggestions
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        return suggestions[:max_suggestions]
    
    def _compute_fusion_score(
        self,
        ingredients: List[int],
        cuisine1_ingredients: Set[int],
        cuisine2_ingredients: Set[int],
    ) -> float:
        """
        Compute fusion score for a recipe.
        
        Higher score = better fusion (balanced mix + good network connectivity).
        
        Args:
            ingredients: List of ingredient IDs
            cuisine1_ingredients: Set of cuisine 1 ingredient IDs
            cuisine2_ingredients: Set of cuisine 2 ingredient IDs
            
        Returns:
            Fusion score [0, 1]
        """
        if not ingredients:
            return 0.0
        
        ing_set = set(ingredients)
        
        # Balance score: how balanced is the mix?
        from_cuisine1 = len(ing_set & cuisine1_ingredients)
        from_cuisine2 = len(ing_set & cuisine2_ingredients)
        balance = 1.0 - abs(from_cuisine1 - from_cuisine2) / len(ingredients)
        
        # Connectivity score: average edge weight within recipe
        connectivity = 0.0
        edge_count = 0
        for i, ing1 in enumerate(ingredients):
            for ing2 in ingredients[i+1:]:
                weight = self.graph.get_edge_weight(ing1, ing2)
                if weight > 0:
                    connectivity += weight
                    edge_count += 1
        
        connectivity = connectivity / edge_count if edge_count > 0 else 0.0
        
        # Combined score
        score = 0.6 * balance + 0.4 * connectivity
        return score

