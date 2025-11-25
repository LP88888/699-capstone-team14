"""
Ingredient normalization step using spaCy.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import time

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Use relative imports within the pipeline package
from ..base import PipelineStep
from ..ingrnorm.spacy_normalizer import SpacyIngredientNormalizer

logger = logging.getLogger(__name__)


class IngredientNormalizerStep(PipelineStep):
    """
    Normalize ingredients using spaCy NLP.
    
    This step applies spaCy-based normalization to ingredient lists,
    extracting canonical forms and removing adjectives/units.
    """
    
    def __init__(
        self,
        input_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
        list_col: str = "NER",
        out_col: str = "NER_clean",
        spacy_model: str = "en_core_web_sm",
        batch_size: int = 512,
        n_process: int = 1,
    ):
        """
        Initialize normalization step.
        
        Args:
            input_path: Input Parquet file path
            output_path: Output Parquet file path
            config: Step configuration (can override other parameters)
            list_col: Input column name containing ingredient lists
            out_col: Output column name for normalized ingredients
            spacy_model: spaCy model name
            batch_size: Batch size for spaCy processing
            n_process: Number of processes (1=single-threaded)
        """
        super().__init__(
            name="IngredientNormalizer",
            input_path=input_path,
            output_path=output_path,
            config=config,
        )
        
        # Override with config if provided
        self.list_col = config.get("list_col", list_col) if config else list_col
        self.out_col = config.get("out_col", out_col) if config else out_col
        self.spacy_model = config.get("spacy_model", spacy_model) if config else spacy_model
        self.batch_size = config.get("batch_size", batch_size) if config else batch_size
        self.n_process = config.get("n_process", n_process) if config else n_process
        
        # Initialize normalizer
        self.normalizer = SpacyIngredientNormalizer(
            model=self.spacy_model,
            batch_size=self.batch_size,
            n_process=self.n_process,
        )
        
        self.logger.info(
            f"Initialized with model={self.spacy_model}, "
            f"batch_size={self.batch_size}, n_process={self.n_process}"
        )
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize ingredient lists in the DataFrame.
        
        Args:
            df: Input DataFrame with ingredient lists
            
        Returns:
            DataFrame with normalized ingredients in out_col
        """
        self.validate_input(df)
        
        # Ensure list_col exists
        if self.list_col not in df.columns:
            self.logger.warning(f"Column '{self.list_col}' not found, creating empty lists")
            df[self.list_col] = [[] for _ in range(len(df))]
        
        # Convert to list format
        lists = [
            list(x) if isinstance(x, (list, tuple, np.ndarray)) else []
            for x in df[self.list_col]
        ]
        
        # Normalize
        self.logger.debug(f"Normalizing {len(lists)} ingredient lists...")
        cleaned_lists = self.normalizer.normalize_batch(lists)
        
        # Add normalized column
        df[self.out_col] = cleaned_lists
        
        self.validate_output(df)
        return df
    
    def execute(self, input_path: Optional[Path] = None) -> Path:
        """
        Execute normalization step with streaming support.
        
        Overrides base execute to use the specialized streaming pattern
        that preserves the input schema while adding the output column.
        """
        input_path = input_path or self.input_path
        if input_path is None:
            raise ValueError("No input path specified")
        
        start_time = time.time()
        self.logger.info(f"=" * 60)
        self.logger.info(f"[{self.name}] Starting execution")
        self.logger.info(f"=" * 60)
        self.logger.info(f"Input: {input_path}")
        self.logger.info(f"Output: {self.output_path}")
        
        # Read input
        pf = self.read_parquet(input_path)
        
        # Process row groups
        output_path = self.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        writer = None
        schema = None
        total_rows = 0
        
        num_row_groups = pf.num_row_groups
        self.logger.info(f"Processing {num_row_groups} row group(s)")
        
        for rg_idx in range(num_row_groups):
            rg_start = time.time()
            self.logger.info(f"[{self.name}] Row group {rg_idx + 1}/{num_row_groups}")
            
            # Read row group
            df = pf.read_row_group(rg_idx).to_pandas()
            self.logger.debug(f"  Read {len(df):,} rows")
            
            # Transform
            df_transformed = self.transform(df)
            
            if df_transformed is None or len(df_transformed) == 0:
                self.logger.warning(f"  Transformation returned empty DataFrame, skipping")
                continue
            
            # Determine schema (preserve all columns, add out_col)
            table = pa.Table.from_pandas(df_transformed, preserve_index=False)
            
            # Ensure out_col is list<string>
            if schema is None:
                # Build schema preserving existing columns
                fields = []
                for col in df_transformed.columns:
                    if col == self.out_col:
                        fields.append(pa.field(self.out_col, pa.list_(pa.string())))
                    else:
                        # Infer type from first row group
                        fields.append(table.schema.field(col))
                schema = pa.schema(fields)
                writer = pq.ParquetWriter(str(output_path), schema, compression=self.compression)
            else:
                # Cast to match schema
                if table.schema != schema:
                    table = table.cast(schema, safe=False)
            
            writer.write_table(table)
            total_rows += len(df_transformed)
            
            rg_elapsed = time.time() - rg_start
            self.logger.info(f"  Processed in {rg_elapsed:.2f}s ({len(df_transformed)/rg_elapsed:.0f} rows/sec)")
        
        if writer:
            writer.close()
        
        elapsed = time.time() - start_time
        self.logger.info(
            f"[{self.name}] Completed: {total_rows:,} rows in {elapsed:.2f}s "
            f"({total_rows/elapsed:.0f} rows/sec)"
        )
        self.logger.info(f"[{self.name}] Output: {output_path}")
        
        return output_path
