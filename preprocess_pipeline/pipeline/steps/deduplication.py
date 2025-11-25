"""
Deduplication step using SBERT or Word2Vec.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import time

import pandas as pd
import pyarrow.parquet as pq

# Use relative imports within the pipeline package
from ..base import PipelineStep
from ..ingrnorm.sbert_dedupe import sbert_dedupe
from ..ingrnorm.w2v_dedupe import w2v_dedupe
from ..ingrnorm.dedupe_map import apply_map_to_parquet_streaming, load_jsonl_map
from ..ingrnorm.parquet_utils import vocab_from_parquet_listcol

logger = logging.getLogger(__name__)


class DeduplicationStep(PipelineStep):
    """
    Deduplicate ingredients using SBERT or Word2Vec similarity.
    
    This step:
    1. Builds vocabulary from normalized ingredients
    2. Creates deduplication mapping using semantic similarity
    3. Applies mapping to create canonical ingredient forms
    """
    
    def __init__(
        self,
        input_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
        dedupe_map_path: Optional[Path] = None,
        method: str = "sbert",  # "sbert" or "w2v"
        list_col: str = "NER_clean",
    ):
        """
        Initialize deduplication step.
        
        Args:
            input_path: Input Parquet file path
            output_path: Output Parquet file path
            config: Step configuration
            dedupe_map_path: Path to save/load dedupe mapping JSONL
            method: Deduplication method ("sbert" or "w2v")
            list_col: Column name containing normalized ingredient lists
        """
        super().__init__(
            name=f"DeduplicationStep_{method.upper()}",
            input_path=input_path,
            output_path=output_path,
            config=config,
        )
        
        self.method = method.lower()
        if self.method not in ("sbert", "w2v"):
            raise ValueError(f"Unknown deduplication method: {method}")
        
        self.list_col = config.get("list_col", list_col) if config else list_col
        self.dedupe_map_path = Path(dedupe_map_path) if dedupe_map_path else None
        
        # Get method-specific config
        if self.method == "sbert":
            self.method_config = config.get("sbert", {}) if config else {}
        else:
            self.method_config = config.get("w2v", {}) if config else {}
        
        self.logger.info(f"Initialized with method={self.method}")
    
    def build_dedupe_map(self, vocab: Dict[str, int]) -> Dict[str, str]:
        """
        Build deduplication mapping from vocabulary.
        
        Args:
            vocab: Vocabulary counter (token -> frequency)
            
        Returns:
            Mapping dictionary (from -> to)
        """
        self.logger.info(f"Building dedupe map using {self.method.upper()}...")
        
        if self.dedupe_map_path is None:
            raise ValueError("dedupe_map_path must be specified")
        
        self.dedupe_map_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.method == "sbert":
            mapping = sbert_dedupe(
                vocab_counter=vocab,
                out_path=str(self.dedupe_map_path),
                model_name=self.method_config.get("model", "all-MiniLM-L6-v2"),
                threshold=float(self.method_config.get("threshold", 0.88)),
                topk=int(self.method_config.get("topk", 25)),
                min_len=int(self.method_config.get("min_len", 2)),
                require_token_overlap=bool(self.method_config.get("require_token_overlap", True)),
                block_generic_as_canon=bool(self.method_config.get("block_generic_as_canon", True)),
            )
        else:  # w2v
            # For w2v, we need the corpus parquet path
            corpus_path = self.config.get("corpus_parquet") if self.config else None
            if corpus_path is None:
                corpus_path = self.input_path
            
            mapping = w2v_dedupe(
                vocab_counter=vocab,
                corpus_parquet=str(corpus_path),
                list_col=self.list_col,
                model_cache_path=self.dedupe_map_path.with_suffix(".w2v"),
                vector_size=int(self.method_config.get("vector_size", 100)),
                window=int(self.method_config.get("window", 5)),
                min_count=int(self.method_config.get("min_count", 1)),
                workers=int(self.method_config.get("workers", 4)),
                sg=int(self.method_config.get("sg", 1)),
                epochs=int(self.method_config.get("epochs", 8)),
                threshold=float(self.method_config.get("threshold", 0.85)),
                topk=int(self.method_config.get("topk", 25)),
                out_path=str(self.dedupe_map_path),
            )
        
        self.logger.info(f"Built dedupe map with {len(mapping):,} mappings")
        return mapping
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply deduplication mapping to DataFrame.
        
        Note: This assumes the dedupe map has already been built.
        Use build_dedupe_map() first, or load an existing map.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with deduplicated ingredients
        """
        self.validate_input(df)
        
        if self.list_col not in df.columns:
            self.logger.warning(f"Column '{self.list_col}' not found, skipping deduplication")
            return df
        
        # Load dedupe map
        if self.dedupe_map_path is None or not self.dedupe_map_path.exists():
            self.logger.warning(f"Dedupe map not found at {self.dedupe_map_path}, skipping")
            return df
        
        mapping = load_jsonl_map(self.dedupe_map_path)
        self.logger.debug(f"Loaded {len(mapping):,} dedupe mappings")
        
        # Apply mapping - modify the column in place (preserves other columns)
        import numpy as np
        df[self.list_col] = [
            [mapping.get(str(tok), str(tok)) for tok in (lst if isinstance(lst, (list, tuple)) else list(lst))]
            if isinstance(lst, (list, tuple, np.ndarray)) and len(lst) > 0
            else (lst if isinstance(lst, (list, tuple)) else [])
            for lst in df[self.list_col]
        ]
        
        self.validate_output(df)
        return df
    
    def execute(self, input_path: Optional[Path] = None, build_map: bool = True) -> Path:
        """
        Execute deduplication step.
        
        Args:
            input_path: Override input path
            build_map: If True, build dedupe map first; if False, use existing map
            
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
        
        # Build dedupe map if needed
        if build_map:
            min_freq = self.method_config.get("min_freq_for_vocab", 1)
            vocab = vocab_from_parquet_listcol(
                str(input_path),
                col=self.list_col,
                min_freq=min_freq,
            )
            self.logger.info(f"Built vocabulary: {len(vocab):,} tokens (min_freq={min_freq})")
            
            if len(vocab) == 0:
                raise ValueError(f"Vocabulary is empty. Check column '{self.list_col}' in {input_path}")
            
            self.build_dedupe_map(vocab)
        
        # Apply dedupe map using streaming approach
        if self.dedupe_map_path is None or not self.dedupe_map_path.exists():
            raise FileNotFoundError(f"Dedupe map not found: {self.dedupe_map_path}")
        
        mapping = load_jsonl_map(self.dedupe_map_path)
        self.logger.info(f"Applying dedupe map with {len(mapping):,} mappings...")
        
        apply_map_to_parquet_streaming(
            in_path=str(input_path),
            out_path=str(self.output_path),
            mapping=mapping,
            list_col=self.list_col,
            compression=self.compression,
        )
        
        elapsed = time.time() - start_time
        self.logger.info(f"=" * 60)
        self.logger.info(f"[{self.name}] Execution complete in {elapsed:.2f}s")
        self.logger.info(f"=" * 60)
        
        return self.output_path
