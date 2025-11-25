"""
Cuisine normalization pipeline orchestrator.

Loads configuration and runs cuisine normalization steps in sequence.
Similar to run_pipeline.py but specifically for cuisine normalization.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import List, Optional

from .config import PipelineConfig
from .base import PipelineStep
from .steps import (
    CuisinePreprocessingStep,
    IngredientNormalizerStep,
    DeduplicationStep,
    EncodingStep,
)
from .common.logging_setup import setup_logging

logger = logging.getLogger(__name__)


class CuisinePipelineOrchestrator:
    """
    Orchestrates the execution of cuisine normalization pipeline steps.
    
    Handles:
    - Loading configuration
    - Instantiating steps
    - Running steps in sequence
    - Passing output of one step as input to the next
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize orchestrator with configuration.
        
        Args:
            config: PipelineConfig instance
        """
        self.config = config
        self.steps: List[PipelineStep] = []
        self.logger = logging.getLogger(f"{__name__}.Orchestrator")
    
    def setup_steps(self) -> None:
        """Instantiate pipeline steps based on configuration."""
        self.logger.info("Setting up cuisine normalization pipeline steps...")
        
        data_cfg = self.config.data
        output_cfg = self.config.output
        stages_cfg = self.config.stages
        
        # Input path
        input_path = Path(data_cfg.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        current_input = input_path
        cuisine_col = data_cfg.cuisine_col
        
        # Step 0: Preprocess cuisine column (split multi-cuisine entries)
        # This creates a parquet with cuisine column as a list
        # We'll use a temporary file, then normalize it
        temp_preprocessed = Path(output_cfg.baseline_parquet).parent / "_cuisine_preprocessed.parquet"
        
        if stages_cfg.write_parquet:
            preprocess_step = CuisinePreprocessingStep(
                input_path=current_input,
                output_path=temp_preprocessed,
                config={
                    "cuisine_col": cuisine_col,
                },
            )
            self.steps.append(preprocess_step)
            current_input = temp_preprocessed
        
        # Step 1: Cuisine Normalization (spaCy)
        # Reuse IngredientNormalizerStep but with cuisine column
        if stages_cfg.write_parquet:
            normalizer = IngredientNormalizerStep(
                input_path=current_input,
                output_path=Path(output_cfg.baseline_parquet),
                config={
                    "list_col": cuisine_col,
                    "out_col": output_cfg.list_col_for_vocab,
                    "spacy_model": self.config.sbert.spacy_model,
                    "batch_size": self.config.sbert.spacy_batch_size,
                    "n_process": self.config.sbert.spacy_n_process,
                },
            )
            self.steps.append(normalizer)
            current_input = Path(output_cfg.baseline_parquet)
        
        # Step 2: Deduplication
        if stages_cfg.sbert_dedupe or stages_cfg.w2v_dedupe:
            method = "sbert" if stages_cfg.sbert_dedupe else "w2v"
            method_config = self.config.sbert.model_dump() if method == "sbert" else self.config.w2v.model_dump()
            
            dedupe_step = DeduplicationStep(
                input_path=current_input,
                output_path=Path(output_cfg.dedup_parquet),
                config={
                    "list_col": output_cfg.list_col_for_vocab,
                    "method": method,
                    method: method_config,
                    "corpus_parquet": str(current_input),
                },
                dedupe_map_path=Path(output_cfg.cosine_map_path),
                method=method,
                list_col=output_cfg.list_col_for_vocab,
            )
            self.steps.append(dedupe_step)
            
            # Update current input to deduped version
            if stages_cfg.apply_cosine_map:
                current_input = Path(output_cfg.dedup_parquet)
        
        # Step 3: Encoding
        if stages_cfg.encode_ids:
            encoder_step = EncodingStep(
                input_path=current_input,
                output_path=Path(output_cfg.unified_parquet),
                config={
                    "ingredients_col": self.config.encoder.ingredients_col,
                    "min_freq": self.config.encoder.min_freq,
                    "dataset_id": self.config.encoder.dataset_id,
                },
                id_to_token_path=Path(output_cfg.ingredient_id_to_token),
                token_to_id_path=Path(output_cfg.ingredient_token_to_id),
                ingredients_col=self.config.encoder.ingredients_col,
                min_freq=self.config.encoder.min_freq,
                dataset_id=self.config.encoder.dataset_id,
            )
            self.steps.append(encoder_step)
        
        self.logger.info(f"Setup complete: {len(self.steps)} step(s) configured")
    
    def run(self, force: bool = False) -> None:
        """
        Execute all pipeline steps in sequence.
        
        Args:
            force: If True, rebuild artifacts even if they exist
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Cuisine Normalization Pipeline Execution")
        self.logger.info("=" * 60)
        
        if not self.steps:
            self.logger.warning("No steps configured. Call setup_steps() first.")
            return
        
        for step_idx, step in enumerate(self.steps, 1):
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Step {step_idx}/{len(self.steps)}: {step.name}")
            self.logger.info(f"{'=' * 60}")
            
            # Check if output exists (unless force)
            if not force and step.output_path and step.output_path.exists():
                self.logger.info(f"Output exists: {step.output_path}")
                self.logger.info("Skipping step (use --force to rebuild)")
                continue
            
            try:
                # Execute step
                output_path = step.execute()
                self.logger.info(f"Step {step_idx} completed: {output_path}")
                
                # Update input for next step
                if step_idx < len(self.steps):
                    next_step = self.steps[step_idx]
                    next_step.input_path = output_path
                    self.logger.debug(f"Updated next step input: {next_step.input_path}")
            
            except Exception as e:
                self.logger.error(f"Step {step_idx} failed: {e}", exc_info=True)
                raise
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Cuisine Normalization Pipeline Execution Complete")
        self.logger.info("=" * 60)


def main():
    """Main entry point for cuisine normalization pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Run cuisine normalization pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pipeline/config/cuisnorm.yaml",
        help="Path to cuisine normalization configuration YAML file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild all artifacts even if they exist",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = PipelineConfig.from_yaml(config_path)
    
    # Setup logging
    setup_logging(config.to_dict())
    logger = logging.getLogger(__name__)
    
    # Create and run orchestrator
    orchestrator = CuisinePipelineOrchestrator(config)
    orchestrator.setup_steps()
    orchestrator.run(force=args.force)
    
    logger.info("Cuisine normalization pipeline execution finished successfully")


if __name__ == "__main__":
    main()

