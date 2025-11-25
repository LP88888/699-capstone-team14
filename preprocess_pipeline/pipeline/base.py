"""
Abstract base class for pipeline steps.

Provides common functionality for:
- Reading/writing Parquet files
- Logging setup
- Execution time measurement
- Configuration handling
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
import time
import logging

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

logger = logging.getLogger(__name__)


class PipelineStep(ABC):
    """
    Abstract base class for pipeline transformation steps.
    
    Subclasses must implement:
    - transform(df: pd.DataFrame) -> pd.DataFrame: The core transformation logic
    
    The base class handles:
    - Reading input Parquet files (streaming by row groups)
    - Writing output Parquet files
    - Logging setup
    - Execution time measurement
    """
    
    def __init__(
        self,
        name: str,
        input_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
        compression: str = "zstd",
    ):
        """
        Initialize pipeline step.
        
        Args:
            name: Step name (for logging)
            input_path: Input Parquet file path
            output_path: Output Parquet file path
            config: Step-specific configuration dictionary
            compression: Parquet compression format (default: zstd)
        """
        self.name = name
        self.input_path = Path(input_path) if input_path else None
        self.output_path = Path(output_path) if output_path else None
        self.config = config or {}
        self.compression = compression
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input DataFrame.
        
        This is the core logic that subclasses must implement.
        The method receives a DataFrame (from a single row group) and
        returns a transformed DataFrame.
        
        For steps that generate models (e.g., training), they should
        override execute() instead and return the model path.
        
        Args:
            df: Input DataFrame (from one row group)
            
        Returns:
            Transformed DataFrame
        """
        pass
    
    def generate_model(self, df: pd.DataFrame) -> Optional[Path]:
        """
        Generate a model artifact from the input DataFrame.
        
        Override this method for steps that produce models (e.g., training).
        By default, returns None (not a model-generating step).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Path to saved model, or None if not a model-generating step
        """
        return None
    
    def read_parquet(self, path: Optional[Path] = None) -> pq.ParquetFile:
        """
        Read Parquet file and return ParquetFile object for streaming access.
        
        Args:
            path: Path to Parquet file (defaults to self.input_path)
            
        Returns:
            ParquetFile object
        """
        path = path or self.input_path
        if path is None:
            raise ValueError("No input path specified")
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        
        self.logger.info(f"Reading Parquet file: {path}")
        return pq.ParquetFile(str(path))
    
    def write_parquet(
        self,
        df: pd.DataFrame,
        path: Optional[Path] = None,
        schema: Optional[pa.Schema] = None,
        append: bool = False,
    ) -> Path:
        """
        Write DataFrame to Parquet file.
        
        Args:
            df: DataFrame to write
            path: Output path (defaults to self.output_path)
            schema: PyArrow schema (if None, inferred from DataFrame)
            append: If True, append to existing file (requires same schema)
            
        Returns:
            Path to written file
        """
        path = path or self.output_path
        if path is None:
            raise ValueError("No output path specified")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if schema is None:
            table = pa.Table.from_pandas(df, preserve_index=False)
        else:
            table = pa.Table.from_pandas(df, preserve_index=False, schema=schema)
            # Cast to match schema if needed
            if table.schema != schema:
                table = table.cast(schema, safe=False)
        
        if append and path.exists():
            # Append mode: read existing schema
            existing_pf = pq.ParquetFile(str(path))
            existing_schema = existing_pf.schema_arrow
            if schema is None:
                schema = existing_schema
            # Write with existing writer
            writer = pq.ParquetWriter(str(path), schema, compression=self.compression)
            writer.write_table(table)
            writer.close()
        else:
            # Write new file
            pq.write_table(table, str(path), compression=self.compression)
        
        self.logger.debug(f"Wrote {len(df):,} rows to {path}")
        return path
    
    def write_parquet_streaming(
        self,
        parquet_file: pq.ParquetFile,
        output_path: Optional[Path] = None,
        columns: Optional[list[str]] = None,
        schema: Optional[pa.Schema] = None,
    ) -> Path:
        """
        Stream Parquet file row groups, transform each, and write to output.
        
        This method handles the streaming pattern used throughout the pipeline:
        - Read row groups one at a time
        - Transform each row group
        - Write to output Parquet file
        
        Args:
            parquet_file: Input ParquetFile object
            output_path: Output path (defaults to self.output_path)
            columns: Columns to read (None = all columns)
            schema: Output schema (if None, inferred from first transformed row group)
            
        Returns:
            Path to written file
        """
        output_path = output_path or self.output_path
        if output_path is None:
            raise ValueError("No output path specified")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        writer = None
        total_rows = 0
        start_time = time.time()
        
        num_row_groups = parquet_file.num_row_groups
        self.logger.info(f"Processing {num_row_groups} row group(s)")
        
        for rg_idx in range(num_row_groups):
            rg_start = time.time()
            self.logger.info(f"[{self.name}] Row group {rg_idx + 1}/{num_row_groups}")
            
            # Read row group
            if columns:
                df = parquet_file.read_row_group(rg_idx, columns=columns).to_pandas()
            else:
                df = parquet_file.read_row_group(rg_idx).to_pandas()
            
            self.logger.debug(f"  Read {len(df):,} rows")
            
            # Transform
            df_transformed = self.transform(df)
            
            if df_transformed is None or len(df_transformed) == 0:
                self.logger.warning(f"  Transformation returned empty DataFrame, skipping row group {rg_idx}")
                continue
            
            # Determine schema from first row group
            if writer is None:
                table = pa.Table.from_pandas(df_transformed, preserve_index=False)
                if schema is None:
                    schema = table.schema
                else:
                    # Cast to match provided schema
                    if table.schema != schema:
                        table = table.cast(schema, safe=False)
                writer = pq.ParquetWriter(str(output_path), schema, compression=self.compression)
            else:
                table = pa.Table.from_pandas(df_transformed, preserve_index=False)
                if table.schema != schema:
                    table = table.cast(schema, safe=False)
            
            # Write row group
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
    
    def execute(self, input_path: Optional[Path] = None) -> Path:
        """
        Execute the pipeline step: read, transform, write.
        
        This is the main entry point for running a step.
        
        Args:
            input_path: Override input path (defaults to self.input_path)
            
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
        self.logger.info(f"Input: {input_path}")
        self.logger.info(f"Output: {self.output_path}")
        
        # Read input
        pf = self.read_parquet(input_path)
        
        # Transform and write (streaming)
        output_path = self.write_parquet_streaming(pf)
        
        elapsed = time.time() - start_time
        self.logger.info(f"=" * 60)
        self.logger.info(f"[{self.name}] Execution complete in {elapsed:.2f}s")
        self.logger.info(f"=" * 60)
        
        return output_path
    
    def validate_input(self, df: pd.DataFrame) -> None:
        """
        Validate input DataFrame (optional override for subclasses).
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If validation fails
        """
        if df is None:
            raise ValueError("Input DataFrame is None")
        if len(df) == 0:
            self.logger.warning("Input DataFrame is empty")
    
    def validate_output(self, df: pd.DataFrame) -> None:
        """
        Validate output DataFrame (optional override for subclasses).
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If validation fails
        """
        if df is None:
            raise ValueError("Output DataFrame is None")

