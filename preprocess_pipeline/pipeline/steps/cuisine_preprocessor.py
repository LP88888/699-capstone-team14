"""
Cuisine preprocessing step: Split multi-cuisine entries into lists.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import time
import ast
import json

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..base import PipelineStep

logger = logging.getLogger(__name__)


def split_cuisine_entries(cuisine_value: str) -> list[str]:
    """
    Split cuisine entries that may contain multiple cuisines.
    Handles formats like:
    - "[American, Italian]" -> ["American", "Italian"]
    - "American, Italian" -> ["American", "Italian"]
    - "American & Italian" -> ["American", "Italian"]
    - "American" -> ["American"]
    - "Goan recipes" -> ["Goan"] (removes "recipes" suffix)
    """
    if pd.isna(cuisine_value) or not str(cuisine_value).strip():
        return []
    
    s = str(cuisine_value).strip()
    
    # Try parsing as list/JSON first
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                cuisines = [str(x).strip() for x in parsed if str(x).strip()]
            else:
                cuisines = [s]
        except:
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    cuisines = [str(x).strip() for x in parsed if str(x).strip()]
                else:
                    cuisines = [s]
            except:
                cuisines = [s]
    else:
        # Try splitting on comma
        if "," in s:
            parts = [x.strip() for x in s.split(",") if x.strip()]
            if len(parts) > 1:
                cuisines = parts
            else:
                cuisines = [s]
        else:
            # Try splitting on " & " or " and "
            found_split = False
            for sep in [" & ", " and ", " AND "]:
                if sep in s:
                    parts = [x.strip() for x in s.split(sep) if x.strip()]
                    if len(parts) > 1:
                        cuisines = parts
                        found_split = True
                        break
            if not found_split:
                # Single value
                cuisines = [s] if s else []
    
    # Clean up each cuisine: remove "recipes" suffix (case-insensitive)
    cleaned = []
    for c in cuisines:
        c_clean = c.strip()
        # Remove "recipes" suffix
        if c_clean.lower().endswith(" recipes"):
            c_clean = c_clean[:-8].strip()
        elif c_clean.lower().endswith(" recipe"):
            c_clean = c_clean[:-7].strip()
        if c_clean:
            cleaned.append(c_clean)
    
    return cleaned


class CuisinePreprocessingStep(PipelineStep):
    """
    Preprocess cuisine column by splitting multi-cuisine entries into lists.
    
    This step handles the unique requirement for cuisine normalization:
    splitting entries that contain multiple cuisines (e.g., "American, Italian")
    into proper list format for downstream normalization.
    """
    
    def __init__(
        self,
        input_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
        cuisine_col: str = "cuisine",
    ):
        """
        Initialize cuisine preprocessing step.
        
        Args:
            input_path: Input Parquet file path
            output_path: Output Parquet file path
            config: Step configuration
            cuisine_col: Column name containing cuisine data
        """
        super().__init__(
            name="CuisinePreprocessing",
            input_path=input_path,
            output_path=output_path,
            config=config,
        )
        
        self.cuisine_col = config.get("cuisine_col", cuisine_col) if config else cuisine_col
        self.logger.info(f"Initialized with cuisine_col={self.cuisine_col}")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Split cuisine entries into lists.
        
        Args:
            df: Input DataFrame with cuisine column
            
        Returns:
            DataFrame with cuisine column converted to list format
        """
        self.validate_input(df)
        
        if self.cuisine_col not in df.columns:
            self.logger.error(f"Column '{self.cuisine_col}' not found in input")
            self.logger.error(f"Available columns: {list(df.columns)}")
            raise KeyError(f"Column '{self.cuisine_col}' not found")
        
        # Split cuisine entries into lists
        self.logger.debug(f"Splitting cuisine entries...")
        cuisine_lists = df[self.cuisine_col].apply(split_cuisine_entries)
        
        # Count splits for statistics
        splits = sum(1 for lst in cuisine_lists if len(lst) > 1)
        if splits > 0:
            self.logger.info(f"  Found {splits:,} entries with multiple cuisines (out of {len(cuisine_lists):,})")
        
        # Preserve all columns, but update cuisine column to list format
        df_output = df.copy()
        df_output[self.cuisine_col] = cuisine_lists
        
        self.validate_output(df_output)
        return df_output
    
    def execute(self, input_path: Optional[Path] = None) -> Path:
        """
        Execute cuisine preprocessing with streaming support.
        
        Overrides base execute to ensure proper schema (list<string>).
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
            
            # Convert to PyArrow table
            table = pa.Table.from_pandas(df_transformed, preserve_index=False)
            
            # Determine schema from first row group
            if schema is None:
                # Build schema preserving all columns, ensuring cuisine_col is list<string>
                fields = []
                for col in df_transformed.columns:
                    if col == self.cuisine_col:
                        fields.append(pa.field(self.cuisine_col, pa.list_(pa.string())))
                    else:
                        # Preserve original column type
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
        
        writer.close()
        
        elapsed = time.time() - start_time
        self.logger.info(
            f"[{self.name}] Completed: {total_rows:,} rows in {elapsed:.2f}s "
            f"({total_rows/elapsed:.0f} rows/sec)"
        )
        self.logger.info(f"[{self.name}] Output: {output_path}")
        
        return output_path

