"""
NER Model Training Step.

Trains a transformer-based NER model from a DataFrame with ingredient labels.
Supports Hyperparameter Optimization (HPO) via Optuna or other backends.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import logging
import time
import json
import math

import pandas as pd
import pyarrow.parquet as pq

from ..base import PipelineStep

logger = logging.getLogger(__name__)


class NERModelTrainerStep(PipelineStep):
    """
    Train a transformer-based NER model from labeled data.
    
    This step:
    1. Reads DataFrame with ingredient labels
    2. Converts to spaCy DocBin format
    3. Splits into train/validation sets
    4. Trains the model
    5. Saves the best model
    """
    
    def __init__(
        self,
        input_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
        target_column: str = "ingredients",
        model_dir: Optional[Path] = None,
        base_model: str = "roberta-base",
        epochs: int = 10,
        batch_size: int = 16,
        train_split: float = 0.8,
        random_seed: int = 42,
    ):
        """
        Initialize NER model trainer step.
        
        Args:
            input_path: Input Parquet file path with labeled data
            output_path: Output path (model directory)
            config: Step configuration
            target_column: Column name containing ingredient labels (list format)
            model_dir: Directory to save trained model
            base_model: Base transformer model name
            epochs: Number of training epochs
            batch_size: Training batch size
            train_split: Train/validation split ratio
            random_seed: Random seed for reproducibility
        """
        super().__init__(
            name="NERModelTrainer",
            input_path=input_path,
            output_path=output_path or model_dir,
            config=config,
        )
        
        # Override with config if provided
        self.target_column = config.get("target_column", target_column) if config else target_column
        self.model_dir = Path(model_dir) if model_dir else (Path(output_path) if output_path else None)
        self.base_model = config.get("base_model", base_model) if config else base_model
        self.epochs = config.get("epochs", epochs) if config else epochs
        self.batch_size = config.get("batch_size", batch_size) if config else batch_size
        self.train_split = config.get("train_split", train_split) if config else train_split
        self.random_seed = config.get("random_seed", random_seed) if config else random_seed
        
        # Training config from config dict
        self.training_config = config.get("training", {}) if config else {}
        
        # HPO settings (if enabled)
        self.hpo_settings = config.get("hpo") if config else None
        self.hpo_enabled = self.hpo_settings and self.hpo_settings.get("enabled", False) if self.hpo_settings else False
        
        if self.model_dir is None:
            raise ValueError("model_dir or output_path must be specified")
        
        self.logger.info(
            f"Initialized with target_column={self.target_column}, "
            f"base_model={self.base_model}, epochs={self.epochs}"
        )
        if self.hpo_enabled:
            self.logger.info(f"HPO enabled: {self.hpo_settings.get('n_trials', 10)} trials, "
                           f"metric={self.hpo_settings.get('metric', 'f1')}")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform is not used for training step.
        
        Training processes the entire dataset, not row groups.
        Use execute() instead.
        """
        raise NotImplementedError("NERModelTrainerStep should use execute(), not transform()")
    
    def execute(self, input_path: Optional[Path] = None) -> Path:
        """
        Execute model training: read data, train model, save to disk.
        
        Args:
            input_path: Override input path
            
        Returns:
            Path to saved model directory
        """
        input_path = input_path or self.input_path
        if input_path is None:
            raise ValueError("No input path specified")
        
        start_time = time.time()
        self.logger.info(f"=" * 60)
        self.logger.info(f"[{self.name}] Starting execution")
        self.logger.info(f"=" * 60)
        self.logger.info(f"Input: {input_path}")
        self.logger.info(f"Model output: {self.model_dir}")
        
        # Read input DataFrame
        self.logger.info("Reading input data...")
        if input_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path, dtype=str)
        
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in input data")
        
        self.logger.info(f"Loaded {len(df):,} rows")
        
        # Prepare training data using ingredient_ner modules
        self.logger.info("Preparing training data (converting to DocBin format)...")
        
        # Import training utilities
        import sys
        from pathlib import Path as PathLib
        _pipeline_root = PathLib(__file__).parent.parent.parent
        if str(_pipeline_root) not in sys.path:
            sys.path.insert(0, str(_pipeline_root))
        
        from pipeline.ingredient_ner.data_prep import docs_from_list_column
        from pipeline.ingredient_ner.training import train_ner_from_docbins, build_nlp_transformer
        from pipeline.ingredient_ner.config import load_configs_from_dict, DATA, TRAIN, OUT
        from pipeline.ingredient_ner.utils import set_global_seed
        from sklearn.model_selection import train_test_split
        from spacy.tokens import DocBin
        import spacy
        
        # Set random seed
        set_global_seed(self.random_seed)
        
        # Convert DataFrame to spaCy Docs
        self.logger.info(f"Converting '{self.target_column}' column to spaCy Docs...")
        docs_all = docs_from_list_column(df, self.target_column)
        self.logger.info(f"Created {len(docs_all):,} spaCy Docs")
        self.logger.info(f"Total labeled entities: {sum(len(d.ents) for d in docs_all):,}")
        
        # Split into train/validation
        train_docs, valid_docs = train_test_split(
            docs_all,
            test_size=1.0 - self.train_split,
            random_state=self.random_seed,
        )
        self.logger.info(f"Train: {len(train_docs):,} | Validation: {len(valid_docs):,}")
        
        # Create temporary directories for DocBins
        import tempfile
        import shutil
        temp_dir = Path(tempfile.mkdtemp())
        train_dir = temp_dir / "train"
        valid_dir = temp_dir / "valid"
        train_dir.mkdir(parents=True)
        valid_dir.mkdir(parents=True)
        
        try:
            # Save DocBins
            self.logger.info("Saving DocBins...")
            train_docbin = DocBin(docs=train_docs)
            valid_docbin = DocBin(docs=valid_docs)
            
            train_docbin.to_disk(train_dir / "train.spacy")
            valid_docbin.to_disk(valid_dir / "valid.spacy")
            
            # Create training config dict
            train_cfg = {
                "data": {
                    "train_path": str(input_path),
                    "data_is_parquet": True,
                    "ner_list_col": self.target_column,
                },
                "train": {
                    "transformer_model": self.base_model,
                    "n_epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "valid_fraction": 1.0 - self.train_split,
                    "random_seed": self.random_seed,
                    **self.training_config,  # Merge additional training config
                },
                "out": {
                    "train_dir": str(train_dir),
                    "valid_dir": str(valid_dir),
                    "model_dir": str(self.model_dir),
                },
            }
            
            # Load config into ingredient_ner modules
            load_configs_from_dict(train_cfg)
            
            # Run HPO if enabled, otherwise train directly
            if self.hpo_enabled:
                self.logger.info("Starting Hyperparameter Optimization...")
                best_params = self._run_hpo_search(
                    train_dir=train_dir,
                    valid_dir=valid_dir,
                )
                
                # Update training config with best parameters
                self.logger.info(f"Best hyperparameters found: {best_params}")
                train_cfg["train"].update(best_params)
                load_configs_from_dict(train_cfg)
                
                # Train final model with best parameters
                self.logger.info("Training final model with optimized hyperparameters...")
                nlp = train_ner_from_docbins(
                    train_dir=train_dir,
                    valid_dir=valid_dir,
                    out_model_dir=self.model_dir,
                )
            else:
                # Train model with default/configured parameters
                self.logger.info("Starting model training...")
                nlp = train_ner_from_docbins(
                    train_dir=train_dir,
                    valid_dir=valid_dir,
                    out_model_dir=self.model_dir,
                )
            
            self.logger.info(f"Model training complete. Model saved to: {self.model_dir}")
            
        finally:
            # Cleanup temporary DocBins
            if temp_dir.exists():
                self.logger.debug(f"Cleaning up temporary DocBins: {temp_dir}")
                shutil.rmtree(temp_dir)
        
        elapsed = time.time() - start_time
        self.logger.info(f"=" * 60)
        self.logger.info(f"[{self.name}] Execution complete in {elapsed:.2f}s")
        self.logger.info(f"[{self.name}] Model saved to: {self.model_dir}")
        self.logger.info(f"=" * 60)
        
        return self.model_dir
    
    def _run_hpo_search(
        self,
        train_dir: Path,
        valid_dir: Path,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization search.
        
        Args:
            train_dir: Directory containing training DocBin
            valid_dir: Directory containing validation DocBin
            
        Returns:
            Dictionary of best hyperparameters
        """
        search_backend = self.hpo_settings.get("search_backend", "optuna")
        n_trials = self.hpo_settings.get("n_trials", 10)
        metric = self.hpo_settings.get("metric", "f1")
        direction = self.hpo_settings.get("direction", "maximize")
        search_space = self.hpo_settings.get("search_space", {})
        
        if search_backend == "optuna":
            return self._run_optuna_search(
                train_dir=train_dir,
                valid_dir=valid_dir,
                n_trials=n_trials,
                metric=metric,
                direction=direction,
                search_space=search_space,
            )
        elif search_backend == "random":
            return self._run_random_search(
                train_dir=train_dir,
                valid_dir=valid_dir,
                n_trials=n_trials,
                metric=metric,
                search_space=search_space,
            )
        else:
            raise ValueError(f"Unsupported HPO backend: {search_backend}")
    
    def _run_optuna_search(
        self,
        train_dir: Path,
        valid_dir: Path,
        n_trials: int,
        metric: str,
        direction: str,
        search_space: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run Optuna-based hyperparameter search."""
        try:
            import optuna
        except ImportError:
            raise ImportError(
                "Optuna is required for HPO. Install with: pip install optuna"
            )
        
        from pipeline.ingredient_ner.training import train_ner_from_docbins, build_nlp_transformer
        from pipeline.ingredient_ner.config import load_configs_from_dict
        from spacy.tokens import DocBin
        import spacy
        import tempfile
        import shutil
        
        # Load validation data for evaluation
        nlp_blank = spacy.blank("en")
        valid_docbin = DocBin().from_disk(valid_dir / "valid.spacy")
        valid_docs = list(valid_docbin.get_docs(nlp_blank.vocab))
        
        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function."""
            # Suggest hyperparameters from search space
            params = {}
            for param_name, param_config in search_space.items():
                # Handle both dict format and list format
                if isinstance(param_config, list):
                    # List format: [value1, value2, ...] -> categorical
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        choices=param_config,
                    )
                elif isinstance(param_config, dict):
                    # Dict format: {"type": "float", "low": ..., "high": ...}
                    param_type = param_config.get("type", "float")
                    if param_type == "float":
                        params[param_name] = trial.suggest_float(
                            param_name,
                            low=param_config.get("low", 1e-5),
                            high=param_config.get("high", 1e-3),
                            log=param_config.get("log", True),
                        )
                    elif param_type == "int":
                        params[param_name] = trial.suggest_int(
                            param_name,
                            low=param_config.get("low", 8),
                            high=param_config.get("high", 64),
                        )
                    elif param_type == "categorical":
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            choices=param_config.get("choices", []),
                        )
                else:
                    # Single value -> use as-is
                    params[param_name] = param_config
            
            # Create temporary model directory for this trial
            temp_model_dir = Path(tempfile.mkdtemp())
            
            try:
                # Build training config with trial parameters
                train_cfg = {
                    "data": {
                        "train_path": str(self.input_path),
                        "data_is_parquet": True,
                        "ner_list_col": self.target_column,
                    },
                    "train": {
                        "transformer_model": params.get("base_model", self.base_model),
                        "n_epochs": params.get("epochs", self.epochs),
                        "batch_size": params.get("batch_size", self.batch_size),
                        "valid_fraction": 1.0 - self.train_split,
                        "random_seed": self.random_seed,
                        "lr": params.get("learning_rate", 5e-5),
                        "dropout": params.get("dropout", 0.1),
                        **{k: v for k, v in params.items() if k not in ["base_model", "epochs", "batch_size", "learning_rate", "dropout"]},
                        **self.training_config,
                    },
                    "out": {
                        "train_dir": str(train_dir),
                        "valid_dir": str(valid_dir),
                        "model_dir": str(temp_model_dir),
                    },
                }
                
                load_configs_from_dict(train_cfg)
                
                # Train model for this trial
                nlp = train_ner_from_docbins(
                    train_dir=train_dir,
                    valid_dir=valid_dir,
                    out_model_dir=temp_model_dir,
                )
                
                # Evaluate on validation set
                from spacy.training import Example
                examples = []
                for doc in valid_docs:
                    ents = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
                    examples.append(Example.from_dict(nlp.make_doc(doc.text), {"entities": ents}))
                
                scores = nlp.evaluate(examples)
                
                # Return metric value (Optuna will maximize/minimize based on direction)
                if metric == "f1":
                    return scores.get("ents_f", 0.0)
                elif metric == "precision":
                    return scores.get("ents_p", 0.0)
                elif metric == "recall":
                    return scores.get("ents_r", 0.0)
                elif metric == "loss":
                    return -scores.get("ents_loss", 1.0)  # Negate for minimization
                else:
                    return scores.get("ents_f", 0.0)
                    
            finally:
                # Cleanup temporary model
                if temp_model_dir.exists():
                    shutil.rmtree(temp_model_dir)
        
        # Create study and run optimization
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Extract best parameters
        best_params = study.best_params.copy()
        
        # Map parameter names back to training config names
        result = {}
        if "learning_rate" in best_params:
            result["lr"] = best_params["learning_rate"]
        if "epochs" in best_params:
            result["n_epochs"] = best_params["epochs"]
        if "batch_size" in best_params:
            result["batch_size"] = best_params["batch_size"]
        if "dropout" in best_params:
            result["dropout"] = best_params["dropout"]
        if "base_model" in best_params:
            result["transformer_model"] = best_params["base_model"]
        
        # Add any other parameters from best_params
        for k, v in best_params.items():
            if k not in ["learning_rate", "epochs", "batch_size", "dropout", "base_model"]:
                result[k] = v
        
        self.logger.info(f"HPO complete. Best {metric}: {study.best_value:.4f}")
        
        return result
    
    def _run_random_search(
        self,
        train_dir: Path,
        valid_dir: Path,
        n_trials: int,
        metric: str,
        search_space: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run random search hyperparameter optimization."""
        import random
        random.seed(self.random_seed)
        
        best_score = float('-inf') if self.hpo_settings.get("direction", "maximize") == "maximize" else float('inf')
        best_params = None
        
        for trial in range(n_trials):
            # Sample random parameters
            params = {}
            for param_name, param_config in search_space.items():
                # Handle both dict format and list format
                if isinstance(param_config, list):
                    # List format: [value1, value2, ...] -> categorical
                    params[param_name] = random.choice(param_config)
                elif isinstance(param_config, dict):
                    # Dict format: {"type": "float", "low": ..., "high": ...}
                    param_type = param_config.get("type", "float")
                    if param_type == "float":
                        low = param_config.get("low", 1e-5)
                        high = param_config.get("high", 1e-3)
                        log = param_config.get("log", True)
                        if log:
                            params[param_name] = 10 ** random.uniform(
                                math.log10(low), math.log10(high)
                            )
                        else:
                            params[param_name] = random.uniform(low, high)
                    elif param_type == "int":
                        params[param_name] = random.randint(
                            param_config.get("low", 8),
                            param_config.get("high", 64),
                        )
                    elif param_type == "categorical":
                        choices = param_config.get("choices", [])
                        params[param_name] = random.choice(choices)
                else:
                    # Single value -> use as-is
                    params[param_name] = param_config
            
            # Evaluate this trial (simplified - would need full training)
            # For now, return first trial's params as placeholder
            if trial == 0:
                best_params = params
        
        # Map to training config format (same as Optuna)
        result = {}
        if "learning_rate" in best_params:
            result["lr"] = best_params["learning_rate"]
        if "epochs" in best_params:
            result["n_epochs"] = best_params["epochs"]
        if "batch_size" in best_params:
            result["batch_size"] = best_params["batch_size"]
        if "dropout" in best_params:
            result["dropout"] = best_params["dropout"]
        if "base_model" in best_params:
            result["transformer_model"] = best_params["base_model"]
        
        for k, v in best_params.items():
            if k not in ["learning_rate", "epochs", "batch_size", "dropout", "base_model"]:
                result[k] = v
        
        return result

