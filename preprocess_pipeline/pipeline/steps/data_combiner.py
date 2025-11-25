"""
Raw data combination step.

Ingests CSVs/Parquets from a directory and outputs a unified DataFrame.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import time
import ast
import json
import re

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..base import PipelineStep

logger = logging.getLogger(__name__)


def read_csv_with_fallback(path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Try multiple encodings to read CSV."""
    encodings = ["utf-8", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            logger.debug(f"Attempting to read {path.name} with encoding={enc}")
            df = pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
            logger.info(f"Successfully read {path.name} with encoding={enc} ({len(df):,} rows)")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.exception(f"Unexpected error reading {path.name}: {e}")
            raise
    
    # Last resort
    logger.warning(f"All standard encodings failed for {path.name}. Trying utf-8 with errors='replace'.")
    df = pd.read_csv(path, encoding="utf-8", errors='replace', dtype=str, low_memory=False)
    logger.info(f"Read {path.name} with utf-8 (errors=replace) ({len(df):,} rows)")
    return df


def find_ingredients_column(df: pd.DataFrame, logger: logging.Logger) -> Optional[str]:
    """Find ingredients column (case-insensitive)."""
    cols_lower = {col.lower(): col for col in df.columns}
    candidates = ["ingredients", "ingredient", "ing", "ingr", "ingredient_list", "ingredients_list"]
    for candidate in candidates:
        if candidate in cols_lower:
            actual_col = cols_lower[candidate]
            logger.info(f"Found ingredients column: '{actual_col}' (matched '{candidate}')")
            return actual_col
    return None


def find_cuisine_column(df: pd.DataFrame, logger: logging.Logger) -> Optional[str]:
    """Find cuisine column (case-insensitive)."""
    cols_lower = {col.lower(): col for col in df.columns}
    candidates = ["cuisine", "cuisines", "cuisine_type", "type", "country", "countries", "region", "regions"]
    for candidate in candidates:
        if candidate in cols_lower:
            actual_col = cols_lower[candidate]
            logger.info(f"Found cuisine column: '{actual_col}' (matched '{candidate}')")
            return actual_col
    return None


def extract_cuisine_from_text(text: str, cuisine_default: str = "unknown") -> str:
    """Extract cuisine from text using heuristics."""
    if pd.isna(text) or not str(text).strip():
        return cuisine_default
    
    text_lower = str(text).lower()
    
    # Common cuisine patterns
    cuisine_patterns = {
        "american", "italian", "mexican", "chinese", "japanese", "indian", "thai",
        "french", "greek", "spanish", "mediterranean", "middle eastern", "lebanese",
        "korean", "vietnamese", "german", "british", "caribbean", "african",
    }
    
    for cuisine in cuisine_patterns:
        if cuisine in text_lower:
            return cuisine.title()
    
    return cuisine_default


class RawDataCombinerStep(PipelineStep):
    """
    Combine raw datasets from a directory into a unified format.
    
    Processes CSV/Parquet files, extracts ingredients/cuisine columns,
    and combines them into a single DataFrame.
    """
    
    def __init__(
        self,
        input_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
        data_dir: Optional[Path] = None,
        ingredients_col: Optional[str] = None,
        cuisine_col: Optional[str] = None,
        cuisine_default: str = "unknown",
        run_ner_inference: bool = False,
        ner_model_path: Optional[Path] = None,
    ):
        """
        Initialize data combiner step.
        
        Args:
            input_path: Input directory path (or single file)
            output_path: Output Parquet file path
            config: Step configuration
            data_dir: Directory containing raw CSV/Parquet files
            ingredients_col: Name of ingredients column (auto-detect if None)
            cuisine_col: Name of cuisine column (auto-detect if None)
            cuisine_default: Default cuisine value
            run_ner_inference: Whether to run NER inference on text columns
            ner_model_path: Path to NER model for inference
        """
        super().__init__(
            name="RawDataCombiner",
            input_path=input_path,
            output_path=output_path,
            config=config,
        )
        
        self.data_dir = Path(data_dir) if data_dir else (Path(input_path) if input_path else None)
        self.ingredients_col = config.get("ingredients_col", ingredients_col) if config else ingredients_col
        self.cuisine_col = config.get("cuisine_col", cuisine_col) if config else cuisine_col
        self.cuisine_default = config.get("cuisine_default", cuisine_default) if config else cuisine_default
        self.run_ner_inference = config.get("run_ner_inference", run_ner_inference) if config else run_ner_inference
        self.ner_model_path = Path(ner_model_path) if ner_model_path else None
        
        if self.data_dir is None:
            raise ValueError("data_dir or input_path must be specified")
        
        self.logger.info(f"Initialized with data_dir={self.data_dir}")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform is not used for this step.
        
        This step processes entire files, not row groups.
        Use execute() instead.
        """
        raise NotImplementedError("RawDataCombinerStep should use execute(), not transform()")
    
    def execute(self, input_path: Optional[Path] = None) -> Path:
        """
        Execute data combination: read files, combine, output unified Parquet.
        
        Args:
            input_path: Override input path (directory or file)
            
        Returns:
            Path to output file
        """
        data_dir = Path(input_path) if input_path else self.data_dir
        if data_dir is None:
            raise ValueError("No input path specified")
        
        start_time = time.time()
        self.logger.info(f"=" * 60)
        self.logger.info(f"[{self.name}] Starting execution")
        self.logger.info(f"=" * 60)
        self.logger.info(f"Input directory: {data_dir}")
        self.logger.info(f"Output: {self.output_path}")
        
        # Find all data files
        if data_dir.is_file():
            data_files = [data_dir]
        else:
            csv_files = list(data_dir.glob("*.csv"))
            parquet_files = list(data_dir.glob("*.parquet"))
            data_files = csv_files + parquet_files
        
        if not data_files:
            raise ValueError(f"No CSV or Parquet files found in {data_dir}")
        
        self.logger.info(f"Found {len(data_files)} data file(s)")
        
        # Process each file
        all_dfs = []
        dataset_id = 1
        
        for file_path in data_files:
            self.logger.info(f"Processing {file_path.name}...")
            
            try:
                # Read file
                if file_path.suffix.lower() == ".parquet":
                    df = pd.read_parquet(file_path, dtype=str)
                else:
                    df = read_csv_with_fallback(file_path, self.logger)
                
                if len(df) == 0:
                    self.logger.warning(f"Skipping empty file: {file_path.name}")
                    continue
                
                def auto_detect(col_name: str) -> str:
                    if self.ingredients_col not in df.columns:
                        self.ingredients_col = find_ingredients_column(df, self.logger)
                    return self.ingredients_col

                self.ingredients_col = auto_detect(self.ingredients_col)
                ing_col = self.ingredients_col
                self.cuisine_col = auto_detect(self.cuisine_col)
                cui_col = self.cuisine_col

                # Extract ingredients and cuisine
                ingredients_series = df[ing_col].astype(str)
                cuisine_series = df[cui_col].astype(str)
                
                # Create result DataFrame
                result = pd.DataFrame({
                    "Dataset_ID": dataset_id,
                    "index": df.index,
                    "ingredients": ingredients_series,
                    "cuisine": cuisine_series,
                })
                
                # Drop rows with empty ingredients
                before_drop = len(result)
                result = result[result["ingredients"].astype(str).str.strip() != ""].copy()
                dropped = before_drop - len(result)
                if dropped > 0:
                    self.logger.info(f"Dropped {dropped:,} rows with empty ingredients")
                
                all_dfs.append(result)
                dataset_id += 1
                
            except Exception as e:
                self.logger.exception(f"Error processing {file_path.name}: {e}")
                continue
        
        if not all_dfs:
            raise ValueError("No valid data processed from any files")
        
        # Combine all DataFrames
        combined_df = pd.concat(all_dfs, ignore_index=True)
        self.logger.info(f"Combined {len(combined_df):,} rows from {len(all_dfs)} dataset(s)")
        
        # Run NER inference if requested
        if self.run_ner_inference and self.ner_model_path:
            self.logger.info("Running NER inference on ingredients...")
            try:
                from ..ingredient_ner.inference import run_full_inference_from_config
                # This would need to be adapted to work with the DataFrame
                # For now, we'll skip this and handle it in a separate step
                self.logger.warning("NER inference integration pending - skipping for now")
            except ImportError:
                self.logger.warning("NER inference modules not available")
        
        # Write output
        output_path = self.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to Parquet
        table = pa.Table.from_pandas(combined_df, preserve_index=False)
        pq.write_table(table, str(output_path), compression=self.compression)
        
        elapsed = time.time() - start_time
        self.logger.info(f"=" * 60)
        self.logger.info(f"[{self.name}] Execution complete in {elapsed:.2f}s")
        self.logger.info(f"[{self.name}] Output: {output_path}")
        self.logger.info(f"=" * 60)
        
        return output_path

