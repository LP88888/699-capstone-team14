"""
Classify cuisine type from ingredient lists using network features.
"""

from typing import List, Dict, Set, Tuple, Optional
import logging
import numpy as np
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from ..network.graph import IngredientGraph

logger = logging.getLogger(__name__)


class CuisineClassifier:
    """Classify cuisine from ingredients using network-based features."""
    
    def __init__(self, graph: IngredientGraph):
        """
        Initialize classifier with ingredient graph.
        
        Args:
            graph: IngredientGraph instance
        """
        self.graph = graph
        self.model: Optional[RandomForestClassifier] = None
        self.cuisine_to_id: Dict[str, int] = {}
        self.id_to_cuisine: Dict[int, str] = {}
    
    def extract_features(self, ingredient_list: List[int]) -> np.ndarray:
        """
        Extract network-based features from ingredient list.
        
        Args:
            ingredient_list: List of ingredient IDs
            
        Returns:
            Feature vector
        """
        features = []
        ing_set = set(ingredient_list)
        
        if not ing_set:
            # Return zero vector if no ingredients
            return np.zeros(10)
        
        # Feature 1: Average degree of ingredients
        degrees = [self.graph.get_ingredient_degree(ing_id) for ing_id in ing_set]
        features.append(np.mean(degrees) if degrees else 0.0)
        features.append(np.max(degrees) if degrees else 0.0)
        features.append(np.min(degrees) if degrees else 0.0)
        
        # Feature 2: Internal connectivity (average edge weight within recipe)
        internal_weights = []
        ing_list = list(ing_set)
        for i, ing1 in enumerate(ing_list):
            for ing2 in ing_list[i+1:]:
                weight = self.graph.get_edge_weight(ing1, ing2)
                if weight > 0:
                    internal_weights.append(weight)
        features.append(np.mean(internal_weights) if internal_weights else 0.0)
        features.append(len(internal_weights))  # Number of internal connections
        
        # Feature 3: External connectivity (connections to ingredients not in recipe)
        external_weights = []
        all_neighbors = set()
        for ing_id in ing_set:
            neighbors = self.graph.get_ingredient_neighbors(ing_id)
            for neighbor in neighbors:
                if neighbor not in ing_set:
                    all_neighbors.add(neighbor)
                    weight = self.graph.get_edge_weight(ing_id, neighbor)
                    external_weights.append(weight)
        features.append(np.mean(external_weights) if external_weights else 0.0)
        features.append(len(all_neighbors))  # Number of external neighbors
        
        # Feature 4: Recipe size
        features.append(len(ing_set))
        
        # Feature 5: Network density of recipe subgraph
        if len(ing_set) > 1:
            subgraph = self.graph.get_ingredient_subgraph(ing_set)
            max_possible_edges = len(ing_set) * (len(ing_set) - 1) / 2
            actual_edges = subgraph.number_of_edges()
            density = actual_edges / max_possible_edges if max_possible_edges > 0 else 0.0
        else:
            density = 0.0
        features.append(density)
        
        return np.array(features)
    
    def prepare_training_data(
        self,
        recipes: List[List[int]],
        cuisines: List[int],
        test_size: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data from recipes and cuisine labels.
        
        Args:
            recipes: List of recipe ingredient lists
            cuisines: List of cuisine IDs (one per recipe)
            test_size: Fraction of data for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Preparing training data from {len(recipes)} recipes")
        
        # Extract features
        X = np.array([self.extract_features(recipe) for recipe in recipes])
        y = np.array(cuisines)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train classifier model.
        
        Args:
            X_train: Training feature vectors
            y_train: Training labels
        """
        logger.info("Training cuisine classifier...")
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
        )
        
        self.model.fit(X_train, y_train)
        logger.info("Training complete")
    
    def predict(self, ingredient_list: List[int]) -> Tuple[int, float]:
        """
        Predict cuisine for a recipe.
        
        Args:
            ingredient_list: List of ingredient IDs
            
        Returns:
            (predicted_cuisine_id, confidence_score)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        features = self.extract_features(ingredient_list).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = np.max(probabilities)
        
        return int(prediction), float(confidence)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate classifier on test data.
        
        Args:
            X_test: Test feature vectors
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
        }

