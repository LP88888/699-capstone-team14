"""
Encoding step: Convert normalized ingredients to integer IDs.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Use relative imports within the pipeline package
from ..base import PipelineStep
from ..ingrnorm.encoder import IngredientEncoder

logger = logging.getLogger(__name__)


class EncodingStep(PipelineStep):
    """
    Encode normalized ingredients to integer IDs.
    
    This step:
    1. Fits an encoder on the normalized ingredient vocabulary
    2. Encodes all ingredients to integer IDs
    3. Saves encoder maps (token<->ID) to JSON files
    """
    
    def __init__(
        self,
        input_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
        id_to_token_path: Optional[Path] = None,
        token_to_id_path: Optional[Path] = None,
        ingredients_col: str = "NER_clean",
        min_freq: int = 1,
        dataset_id: int = 1,
    ):
        """
        Initialize encoding step.
        
        Args:
            input_path: Input Parquet file path
            output_path: Output Parquet file path
            config: Step configuration
            id_to_token_path: Path to save ID->token mapping JSON
            token_to_id_path: Path to save token->ID mapping JSON
            ingredients_col: Column name containing normalized ingredient lists
            min_freq: Minimum frequency for ingredient to be included in vocabulary
            dataset_id: Dataset ID to assign to encoded recipes
        """
        super().__init__(
            name="EncodingStep",
            input_path=input_path,
            output_path=output_path,
            config=config,
        )
        
        self.ingredients_col = config.get("ingredients_col", ingredients_col) if config else ingredients_col
        self.min_freq = config.get("min_freq", min_freq) if config else min_freq
        self.dataset_id = config.get("dataset_id", dataset_id) if config else dataset_id
        
        self.id_to_token_path = Path(id_to_token_path) if id_to_token_path else None
        self.token_to_id_path = Path(token_to_id_path) if token_to_id_path else None
        
        self.encoder: Optional[IngredientEncoder] = None
        
        self.logger.info(
            f"Initialized with min_freq={self.min_freq}, "
            f"dataset_id={self.dataset_id}, col={self.ingredients_col}"
        )
    
    def fit_encoder(self) -> IngredientEncoder:
        """
        Fit encoder on input data.
        
        Returns:
            Fitted IngredientEncoder
        """
        if self.input_path is None:
            raise ValueError("No input path specified for fitting encoder")
        
        self.logger.info(f"Fitting encoder from {self.input_path}...")
        encoder = IngredientEncoder(min_freq=self.min_freq)
        encoder.fit_from_parquet_streaming(
            parquet_path=self.input_path,
            col=self.ingredients_col,
            min_freq=self.min_freq,
        )
        encoder.freeze()
        
        self.encoder = encoder
        self.logger.info(f"Encoder fitted: {len(encoder.token_to_id):,} tokens")
        
        # Save encoder maps
        if self.id_to_token_path and self.token_to_id_path:
            encoder.save_maps(
                id_to_token_path=self.id_to_token_path,
                token_to_id_path=self.token_to_id_path,
            )
            self.logger.info(f"Saved encoder maps:")
            self.logger.info(f"  - {self.id_to_token_path}")
            self.logger.info(f"  - {self.token_to_id_path}")
        
        return encoder
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode ingredients in DataFrame to integer IDs.
        
        Note: Encoder must be fitted first using fit_encoder().
        
        Args:
            df: Input DataFrame with normalized ingredient lists
            
        Returns:
            DataFrame with encoded ingredients
        """
        self.validate_input(df)
        
        if self.encoder is None:
            raise ValueError("Encoder not fitted. Call fit_encoder() first.")
        
        if self.ingredients_col not in df.columns:
            self.logger.warning(f"Column '{self.ingredients_col}' not found")
            return df
        
        # Encode ingredients
        id_lists = self.encoder.transform_series_to_idlists(df[self.ingredients_col])
        
        # Create output DataFrame with encoded format
        import numpy as np
        output_df = pd.DataFrame({
            "Dataset ID": np.full(len(df), self.dataset_id, dtype=np.int32),
            "Index": np.arange(len(df), dtype=np.int64),
            "Ingredients": id_lists,
        })
        
        self.validate_output(output_df)
        return output_df
    
    def execute(self, input_path: Optional[Path] = None, fit_encoder: bool = True) -> Path:
        """
        Execute encoding step.
        
        Args:
            input_path: Override input path
            fit_encoder: If True, fit encoder first; if False, use existing encoder
            
        Returns:
            Path to output file
        """
        input_path = input_path or self.input_path
        if input_path is None:
            raise ValueError("No input path specified")
        
        start_time = time.time()
        self.logger.info(f"=" * 60)
        self.logger.info(f"[{self.name}] Starting execution")
        self.logger.info(f"=" * 60)
        
        # Fit encoder if needed
        if fit_encoder or self.encoder is None:
            self.fit_encoder()
        
        # Use encoder's built-in streaming method for efficiency
        output_path = self.encoder.encode_parquet_streaming(
            parquet_path=input_path,
            out_parquet_path=self.output_path,
            dataset_id=self.dataset_id,
            col=self.ingredients_col,
            compression=self.compression,
        )
        
        elapsed = time.time() - start_time
        self.logger.info(f"=" * 60)
        self.logger.info(f"[{self.name}] Execution complete in {elapsed:.2f}s")
        self.logger.info(f"=" * 60)
        
        return output_path
