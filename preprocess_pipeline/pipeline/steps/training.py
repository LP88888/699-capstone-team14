"""
NER Model Training Step.

Trains a transformer-based NER model from a DataFrame with ingredient labels.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import time

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
        
        if self.model_dir is None:
            raise ValueError("model_dir or output_path must be specified")
        
        self.logger.info(
            f"Initialized with target_column={self.target_column}, "
            f"base_model={self.base_model}, epochs={self.epochs}"
        )
    
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
            
            # Train model
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

