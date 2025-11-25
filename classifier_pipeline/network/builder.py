"""
Build ingredient co-occurrence network from encoded recipe data.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import ast  # Import ast at the top
import numpy as np  # Import numpy for type checking

import pandas as pd
import networkx as nx
from collections import Counter, defaultdict
# Import itertools for efficient iterators
from itertools import chain, combinations

logger = logging.getLogger(__name__)


class NetworkBuilder:
    """Builds ingredient co-occurrence network from recipe data."""
    
    def __init__(
        self,
        min_cooccurrence: int = 1,
        weight_method: str = "frequency",
        normalize_weights: bool = True,
        min_ingredient_freq: int = 2,
    ):
        self.min_cooccurrence = min_cooccurrence
        self.weight_method = weight_method
        self.normalize_weights = normalize_weights
        self.min_ingredient_freq = min_ingredient_freq
        
        self.ingredient_freq: Dict[int, int] = {}
        self.cooccurrence_matrix: Dict[Tuple[int, int], int] = defaultdict(int)
        self.graph: Optional[nx.Graph] = None

    def load_data(self, data_path: Path, ingredients_col: str = "encoded_ingredients") -> pd.DataFrame:
        """ (No changes here, this is fine) """
        logger.info(f"Loading data from {data_path}")
        df = pd.read_parquet(data_path)
        
        if ingredients_col not in df.columns:
            raise ValueError(f"Column '{ingredients_col}' not found in data. Available: {list(df.columns)}")
        
        logger.info(f"Loaded {len(df):,} recipes")
        return df

    def _clean_ingredient_row(self, ingredients: any) -> List[int]:
        """
        Safely parse a single row's ingredient data,
        handling lists, np.ndarrays, strings, and Nones/NaNs.
        """
        # 1. Handle list-like types (list, tuple, np.ndarray)
        if isinstance(ingredients, (list, tuple, np.ndarray)):
            # Use pd.notna to also filter out any Nones or NaNs *inside* the list
            try:
                return [int(i) for i in ingredients if pd.notna(i) and i != 0]
            except ValueError:
                logger.warning(f"Could not parse list-like value: {ingredients}")
                return []
        
        # 2. Handle string representations of lists
        if isinstance(ingredients, str):
            try:
                parsed = ast.literal_eval(ingredients)
                if isinstance(parsed, (list, tuple)):
                    return [int(i) for i in parsed if pd.notna(i) and i != 0]
            except:
                # Catches bad strings (not list-like) or syntax errors
                return []
        
        # 3. Handle None, np.nan, or anything else
        return []

    def compute_statistics(self, df: pd.DataFrame, ingredients_col: str = "encoded_ingredients") -> None:
        """
        Compute ingredient frequencies and co-occurrence statistics.
        This version is vectorized for much better performance.
        """
        logger.info("Computing ingredient frequencies and co-occurrences...")
        logger.info(f"Processing {len(df):,} recipes")
        
        # Reset statistics
        self.ingredient_freq = Counter()
        self.cooccurrence_matrix = defaultdict(int)

        # 1. Clean the entire column in one-shot using .apply()
        # This replaces the entire slow, row-by-row if/elif/else block
        logger.info(f"Cleaning and parsing '{ingredients_col}' column...")
        cleaned_lists = df[ingredients_col].apply(self._clean_ingredient_row)
        
        logger.info("Column cleaning complete. Computing statistics...")

        # 2. Compute ingredient frequencies (much faster)
        # We chain all sublists into one giant iterator and pass it to Counter
        all_ingredients = chain.from_iterable(cleaned_lists)
        self.ingredient_freq = Counter(all_ingredients)

        # 3. Compute co-occurrence matrix
        # We still loop, but over the *clean* Series, which is faster.
        empty_count = 0
        for ing_list in cleaned_lists:
            if not ing_list:
                empty_count += 1
                continue
                
            # Get unique, sorted ingredients for this recipe
            ing_list_sorted = sorted(set(ing_list))
            
            # Use itertools.combinations for efficient pairing
            # This is cleaner than a nested loop
            for ing1, ing2 in combinations(ing_list_sorted, 2):
                # We know ing1 < ing2 because the list is sorted
                self.cooccurrence_matrix[(ing1, ing2)] += 1
        
        logger.info(f"Processed {len(df):,} recipes")
        logger.info(f"   - Recipes with ingredients: {len(df) - empty_count:,}")
        logger.info(f"   - Empty recipes: {empty_count:,}")
        logger.info(f"Found {len(self.ingredient_freq):,} unique ingredients")
        logger.info(f"Found {len(self.cooccurrence_matrix):,} ingredient pairs")
        
        if len(self.ingredient_freq) == 0:
            logger.warning("No ingredients found! Check that the ingredients column contains valid data.")
            logger.warning(f"Column '{ingredients_col}' sample values:")
            for idx in range(min(5, len(df))):
                logger.warning(f"   Row {idx}: {df[ingredients_col].iloc[idx]} (type: {type(df[ingredients_col].iloc[idx])})")
        
        # Filter by minimum frequency
        if self.min_ingredient_freq > 1:
            filtered_ingredients = {
                ing_id: freq for ing_id, freq in self.ingredient_freq.items()
                if freq >= self.min_ingredient_freq
            }
            logger.info(f"After filtering (min_freq={self.min_ingredient_freq}): {len(filtered_ingredients):,} ingredients")
            self.ingredient_freq = filtered_ingredients

    def build_graph(self) -> nx.Graph:
        """
        Build NetworkX graph from co-occurrence statistics.
        (No changes needed)
        """
        logger.info("Building network graph...")
        
        G = nx.Graph()
        
        # Add nodes (ingredients)
        valid_ingredients = set(self.ingredient_freq.keys())
        for ing_id in valid_ingredients:
            G.add_node(ing_id, frequency=self.ingredient_freq[ing_id])
        
        logger.info(f"Added {G.number_of_nodes():,} nodes")
        
        # Add edges (co-occurrences)
        edge_count = 0
        for (ing1, ing2), count in self.cooccurrence_matrix.items():
            # Only add edge if both ingredients are valid and meet minimum co-occurrence
            if ing1 in valid_ingredients and ing2 in valid_ingredients:
                if count >= self.min_cooccurrence:
                    weight = self._compute_weight(ing1, ing2, count)
                    G.add_edge(ing1, ing2, weight=weight, count=count)
                    edge_count += 1
        
        logger.info(f"Added {edge_count:,} edges")
        
        self.graph = G
        return G
    
    def _compute_weight(self, ing1: int, ing2: int, cooccurrence_count: int) -> float:
        """
        Compute edge weight based on co-occurrence.
        (No changes needed)
        """
        if self.weight_method == "frequency":
            weight = float(cooccurrence_count)
        elif self.weight_method == "jaccard":
            # Jaccard similarity: |A ∩ B| / |A ∪ B|
            freq1 = self.ingredient_freq.get(ing1, 0)
            freq2 = self.ingredient_freq.get(ing2, 0)
            union = freq1 + freq2 - cooccurrence_count
            weight = cooccurrence_count / union if union > 0 else 0.0
        elif self.weight_method == "cosine":
            # Cosine similarity: |A ∩ B| / sqrt(|A| * |B|)
            freq1 = self.ingredient_freq.get(ing1, 0)
            freq2 = self.ingredient_freq.get(ing2, 0)
            weight = cooccurrence_count / ((freq1 * freq2) ** 0.5) if (freq1 * freq2) > 0 else 0.0
        else:
            weight = float(cooccurrence_count)
        
        return weight
    
    def normalize_graph_weights(self, G: nx.Graph) -> nx.Graph:
        """
        Normalize edge weights to [0, 1] range.
        (No changes needed)
        """
        if not G.edges():
            return G
        
        weights = [data['weight'] for _, _, data in G.edges(data=True)]
        max_weight = max(weights)
        min_weight = min(weights)
        
        if max_weight == min_weight:
            # All weights are the same, set to 1.0
            for u, v, data in G.edges(data=True):
                data['weight'] = 1.0
        else:
            # Normalize to [0, 1]
            weight_range = max_weight - min_weight
            for u, v, data in G.edges(data=True):
                data['weight'] = (data['weight'] - min_weight) / weight_range
        
        return G
    
    def save_graph(self, output_path: Path, format: str = "graphml") -> None:
        """
        Save graph to file.
        (No changes needed)
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_graph() first.")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving graph to {output_path} (format: {format})")
        
        if format == "graphml":
            nx.write_graphml(self.graph, output_path)
        elif format == "gexf":
            nx.write_gexf(self.graph, output_path)
        elif format == "edgelist":
            nx.write_edgelist(self.graph, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved graph with {self.graph.number_of_nodes():,} nodes and {self.graph.number_of_edges():,} edges")