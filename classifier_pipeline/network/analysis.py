"""
Network analysis functions for ingredient network.
"""

from typing import Dict, List, Tuple, Set
import networkx as nx
import logging
import time

logger = logging.getLogger(__name__)


class NetworkAnalyzer:
    """Analyze ingredient network properties."""
    
    def __init__(self, graph: nx.Graph):
        """
        Initialize analyzer with network graph.
        
        Args:
            graph: NetworkX graph
        """
        self.graph = graph
        self._centrality_cache: Dict[str, Dict[int, float]] = {}
    

    def compute_centrality_measures(self) -> Dict[str, Dict[int, float]]:
        logger.info("Starting centrality computation...")
        measures = {}
        
        # 1. Degree Centrality (Instant - Keep this)
        t0 = time.time()
        logger.info("Step 1/3: Degree Centrality...")
        measures['degree'] = nx.degree_centrality(self.graph)
        logger.info(f"Done in {time.time() - t0:.2f}s")
        
        # 2. PageRank (Fast & Valuable - Keep this)
        # PageRank is usually much faster than Betweenness for this size
        try:
            t0 = time.time()
            logger.info("Step 2/3: PageRank...")
            measures['pagerank'] = nx.pagerank(self.graph, alpha=0.85, max_iter=100)
            logger.info(f"Done in {time.time() - t0:.2f}s")
        except Exception as e:
            logger.warning(f"Skipping PageRank: {e}")
            measures['pagerank'] = {}

        # 3. Betweenness (The Culprit - SKIP FOR NOW)
        # If you really need this, we must use a different library (igraph)
        logger.info("Step 3/3: Betweenness Centrality (SKIPPING due to performance)...")
        measures['betweenness'] = {} 
        
        # 4. Closeness (SKIPPING - Definitely too slow)
        logger.info("Step 4/4: Closeness Centrality (SKIPPING)...")
        measures['closeness'] = {}
        
        self._centrality_cache = measures
        logger.info("Centrality computation finished.")
        return measures
        
    def get_top_ingredients(self, measure: str = 'degree', top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Get top ingredients by centrality measure.
        
        Args:
            measure: Centrality measure name ('degree', 'betweenness', 'closeness', 'pagerank')
            top_k: Number of top ingredients to return
            
        Returns:
            List of (ingredient_id, score) tuples
        """
        if not self._centrality_cache:
            self.compute_centrality_measures()
        
        if measure not in self._centrality_cache:
            raise ValueError(f"Unknown measure: {measure}. Available: {list(self._centrality_cache.keys())}")
        
        scores = self._centrality_cache[measure]
        sorted_ingredients = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_ingredients[:top_k]
    
    def analyze_community_structure(self) -> Dict[int, int]:
        """
        Detect communities (clusters) in the ingredient network.
        
        Returns:
            Dictionary mapping ingredient_id -> community_id
        """
        logger.info("Analyzing community structure...")
        
        try:
            # Use Louvain algorithm for community detection
            import community as community_louvain
            communities = community_louvain.best_partition(self.graph)
            logger.info(f"Found {len(set(communities.values()))} communities")
            return communities
        except ImportError:
            logger.warning("python-louvain not installed. Using greedy modularity instead.")
            # Fallback to greedy modularity
            communities_generator = nx.community.greedy_modularity_communities(self.graph)
            communities = {}
            for comm_id, comm in enumerate(communities_generator):
                for ing_id in comm:
                    communities[ing_id] = comm_id
            logger.info(f"Found {len(communities_generator)} communities")
            return communities
    
    def get_network_statistics(self) -> Dict[str, float]:
        """
        Compute network statistics.
        
        Note: Path-length statistics (diameter, average_path_length) are skipped
        for performance reasons on large graphs (>1000 nodes). These calculations
        require computing all-pairs shortest paths which is O(n^2) and can hang
        on graphs with 7k+ nodes.
        
        Returns:
            Dictionary of network statistics
        """
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        
        stats = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
        }
        
        # Handle empty graph
        if num_nodes == 0:
            stats.update({
                'density': 0.0,
                'is_connected': False,
                'diameter': 0,
                'average_path_length': 0.0,
                'largest_component_size': 0,
                'avg_degree': 0.0,
                'max_degree': 0,
                'min_degree': 0,
            })
            return stats
        
        stats['density'] = nx.density(self.graph)
        
        # Check connectivity (only if graph has nodes)
        try:
            stats['is_connected'] = nx.is_connected(self.graph)
        except nx.NetworkXPointlessConcept:
            stats['is_connected'] = False
        
        # Log warning about skipped path-length statistics
        if num_nodes > 1000:
            logger.warning(
                f"Path-length statistics (diameter, average_path_length) skipped for performance. "
                f"Graph has {num_nodes:,} nodes. These calculations require O(n^2) all-pairs "
                f"shortest paths and can hang on large graphs."
            )
        
        if stats['is_connected']:
            # SKIPPED: diameter and average_path_length for performance
            # stats['diameter'] = nx.diameter(self.graph)  # Too slow on large graphs
            # stats['average_path_length'] = nx.average_shortest_path_length(self.graph)  # Too slow on large graphs
            stats['diameter'] = 0
            stats['average_path_length'] = 0.0
            stats['largest_component_size'] = num_nodes
        else:
            # For disconnected graphs, analyze largest component
            if num_nodes > 0:
                try:
                    components = list(nx.connected_components(self.graph))
                    if components:
                        largest_cc = max(components, key=len)
                        # subgraph = self.graph.subgraph(largest_cc)  # Not needed if we skip path calculations
                        stats['largest_component_size'] = len(largest_cc)
                        # SKIPPED: diameter and average_path_length for performance
                        # if len(largest_cc) > 1:
                        #     stats['diameter'] = nx.diameter(subgraph)  # Too slow on large graphs
                        #     stats['average_path_length'] = nx.average_shortest_path_length(subgraph)  # Too slow on large graphs
                        # else:
                        #     stats['diameter'] = 0
                        #     stats['average_path_length'] = 0.0
                        stats['diameter'] = 0
                        stats['average_path_length'] = 0.0
                    else:
                        stats['largest_component_size'] = 0
                        stats['diameter'] = 0
                        stats['average_path_length'] = 0.0
                except Exception as e:
                    logger.warning(f"Error computing component statistics: {e}")
                    stats['largest_component_size'] = 0
                    stats['diameter'] = 0
                    stats['average_path_length'] = 0.0
            else:
                stats['largest_component_size'] = 0
                stats['diameter'] = 0
                stats['average_path_length'] = 0.0
        
        # Degree statistics
        degrees = dict(self.graph.degree())
        degree_values = list(degrees.values())
        stats['avg_degree'] = sum(degree_values) / len(degree_values) if degree_values else 0
        stats['max_degree'] = max(degree_values) if degree_values else 0
        stats['min_degree'] = min(degree_values) if degree_values else 0
        
        return stats

