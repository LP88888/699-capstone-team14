"""
Unified pipeline orchestrator.

Supports multiple modes:
- train: Train NER model from raw data
- inference: Run inference and normalization pipeline
- full: Train model then run inference pipeline
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import List, Optional

from .config import PipelineConfig, LegacyPipelineConfig
from .base import PipelineStep
from .steps import (
    RawDataCombinerStep,
    NERModelTrainerStep,
    IngredientNormalizerStep,
    DeduplicationStep,
    EncodingStep,
)
from .task_orchestrator import TaskBasedOrchestrator
from .common.logging_setup import setup_logging

logger = logging.getLogger(__name__)


class UnifiedPipelineOrchestrator:
    """
    Unified orchestrator for the complete preprocessing pipeline.
    
    Handles:
    - Raw data ingestion
    - Model training
    - Inference and normalization
    - Column-agnostic processing
    """
    
    def __init__(self, config: LegacyPipelineConfig):
        """
        Initialize orchestrator with configuration.
        
        Args:
            config: LegacyPipelineConfig instance
        """
        self.config = config
        self.steps: List[PipelineStep] = []
        self.logger = logging.getLogger(f"{__name__}.Orchestrator")
        self.mode = config.pipeline.mode if config.pipeline else "full"
    
    def _get_column_input_path(self, column_name: str, mode: Optional[str] = None) -> Path:
        """
        Get input path for a column, handling both single files and directories.
        
        Args:
            column_name: Name of the column
            mode: Pipeline mode (if None, uses self.mode)
            
        Returns:
            Path object (file or directory)
        """
        if not self.config.pipeline:
            return Path(self.config.data.input_path)
        
        input_path = self.config.pipeline.get_input_path_for_column(column_name, mode)
        
        # Handle list of paths (for combining multiple sources)
        if isinstance(input_path, list):
            # If list provided, use first path as base (or create combined path)
            # The combiner will handle multiple paths
            if len(input_path) > 0:
                return Path(input_path[0])
            else:
                return Path(self.config.pipeline.input_path)
        
        return Path(input_path)
    
    def _needs_combining(self, input_path: Path) -> bool:
        """
        Check if input path needs to be combined (directory or list of files).
        
        Args:
            input_path: Path to check
            
        Returns:
            True if needs combining, False if single file/parquet
        """
        if input_path.is_dir():
            return True
        
        # Check if it's a parquet file (already combined)
        if input_path.suffix.lower() == ".parquet" and input_path.exists():
            return False
        
        # Check if it's a single CSV file
        if input_path.suffix.lower() == ".csv" and input_path.exists():
            return False
        
        # Default: assume it's a directory that needs combining
        return True
    
    def setup_steps(self) -> None:
        """Instantiate pipeline steps based on configuration and mode."""
        self.logger.info(f"Setting up pipeline steps for mode: {self.mode}")
        
        pipeline_cfg = self.config.pipeline
        data_cfg = self.config.data
        output_cfg = self.config.output
        stages_cfg = self.config.stages
        
        # Track combined outputs per column
        column_combined_outputs: Dict[str, Path] = {}
        
        # Step 0: Raw Data Combination (per column if needed)
        if self.mode in ("train", "full"):
            columns_to_combine = pipeline_cfg.columns if pipeline_cfg else [data_cfg.ner_col]
            
            for column_name in columns_to_combine:
                # Get input path for this column in train mode
                column_input = self._get_column_input_path(column_name, mode="train")
                
                # Check if this column needs combining
                if self._needs_combining(column_input):
                    self.logger.info(f"Combining data for column '{column_name}' from {column_input}")
                    
                    # Create column-specific output
                    combined_output = Path(output_cfg.baseline_parquet).parent / f"combined_{column_name}_data.parquet"
                    
                    # Determine which columns to extract
                    if column_name == "ingredients":
                        extract_cols = {"ingredients": data_cfg.ner_col}
                    elif column_name == "cuisine":
                        extract_cols = {"cuisine": data_cfg.cuisine_col}
                    else:
                        extract_cols = {column_name: column_name}
                    
                    combiner = RawDataCombinerStep(
                        input_path=column_input,
                        output_path=combined_output,
                        config={
                            "ingredients_col": extract_cols.get("ingredients"),
                            "cuisine_col": extract_cols.get("cuisine"),
                            "run_ner_inference": False,  # Will run after training
                        },
                        data_dir=column_input if column_input.is_dir() else None,
                    )
                    self.steps.append(combiner)
                    column_combined_outputs[column_name] = combined_output
                else:
                    # Single file, use directly
                    self.logger.info(f"Using single file for column '{column_name}': {column_input}")
                    column_combined_outputs[column_name] = column_input
        
        # Step 1: Model Training (if mode includes training)
        if self.mode in ("train", "full"):
            if self.config.training:
                model_dir = Path(pipeline_cfg.model_dir) if pipeline_cfg else Path("models/ingredient_ner/")
                
                # Use ingredients column input for training
                training_input = column_combined_outputs.get("ingredients", 
                    self._get_column_input_path("ingredients", mode="train"))
                
                trainer = NERModelTrainerStep(
                    input_path=training_input,
                    output_path=model_dir,
                    config={
                        "target_column": data_cfg.ner_col,
                        "base_model": self.config.training.base_model,
                        "epochs": self.config.training.epochs,
                        "batch_size": self.config.training.batch_size,
                        "train_split": self.config.training.train_split,
                        "random_seed": self.config.training.random_seed,
                        "training": self.config.training.model_dump(),
                    },
                    model_dir=model_dir,
                )
                self.steps.append(trainer)
                self.trained_model_path = model_dir
            else:
                self.logger.warning("Training mode requested but no training config provided")
                self.trained_model_path = None
        
        # Step 2: Inference and Normalization (if mode includes inference)
        if self.mode in ("inference", "full"):
            # Process each column specified in pipeline config
            columns_to_process = pipeline_cfg.columns if pipeline_cfg else [data_cfg.ner_col]
            
            for target_column in columns_to_process:
                self.logger.info(f"Processing column: {target_column}")
                
                # Get input path for this column in inference mode
                if self.mode == "inference":
                    # In inference mode, get column-specific input
                    column_input = self._get_column_input_path(target_column, mode="inference")
                    
                    # Check if needs combining
                    if self._needs_combining(column_input):
                        # Combine data for this column
                        combined_output = Path(output_cfg.baseline_parquet).parent / f"combined_{target_column}_inference.parquet"
                        
                        if target_column == "ingredients":
                            extract_cols = {"ingredients": data_cfg.ner_col}
                        elif target_column == "cuisine":
                            extract_cols = {"cuisine": data_cfg.cuisine_col}
                        else:
                            extract_cols = {target_column: target_column}
                        
                        combiner = RawDataCombinerStep(
                            input_path=column_input,
                            output_path=combined_output,
                            config={
                                "ingredients_col": extract_cols.get("ingredients"),
                                "cuisine_col": extract_cols.get("cuisine"),
                                "run_ner_inference": False,
                            },
                            data_dir=column_input if column_input.is_dir() else None,
                        )
                        self.steps.append(combiner)
                        col_current_input = combined_output
                    else:
                        # Single file, use directly
                        col_current_input = column_input
                else:
                    # In full mode, use combined data from training step
                    col_current_input = column_combined_outputs.get(target_column,
                        self._get_column_input_path(target_column, mode="train"))
                
                # Determine column-specific paths
                if target_column == "ingredients":
                    list_col = data_cfg.ner_col
                    out_col = output_cfg.list_col_for_vocab
                    baseline_path = Path(output_cfg.baseline_parquet)
                    dedup_path = Path(output_cfg.dedup_parquet)
                    unified_path = Path(output_cfg.unified_parquet)
                elif target_column == "cuisine":
                    list_col = data_cfg.cuisine_col
                    out_col = "cuisine_clean"
                    baseline_path = Path(output_cfg.baseline_parquet).parent / "cuisine_baseline.parquet"
                    dedup_path = Path(output_cfg.dedup_parquet).parent / "cuisine_deduped.parquet"
                    unified_path = Path(output_cfg.unified_parquet).parent / "cuisine_unified.parquet"
                else:
                    # Generic column processing
                    list_col = target_column
                    out_col = f"{target_column}_clean"
                    baseline_path = Path(output_cfg.baseline_parquet).parent / f"{target_column}_baseline.parquet"
                    dedup_path = Path(output_cfg.dedup_parquet).parent / f"{target_column}_deduped.parquet"
                    unified_path = Path(output_cfg.unified_parquet).parent / f"{target_column}_unified.parquet"
                
                # Step 2a: Normalization
                if stages_cfg.write_parquet:
                    normalizer = IngredientNormalizerStep(
                        input_path=col_current_input,
                        output_path=baseline_path,
                        config={
                            "list_col": list_col,
                            "out_col": out_col,
                            "spacy_model": self.config.sbert.spacy_model,
                            "batch_size": self.config.sbert.spacy_batch_size,
                            "n_process": self.config.sbert.spacy_n_process,
                        },
                    )
                    self.steps.append(normalizer)
                    col_current_input = baseline_path
                
                # Step 2b: Deduplication
                if stages_cfg.sbert_dedupe or stages_cfg.w2v_dedupe:
                    method = "sbert" if stages_cfg.sbert_dedupe else "w2v"
                    method_config = self.config.sbert.model_dump() if method == "sbert" else self.config.w2v.model_dump()
                    
                    dedupe_map_path = Path(output_cfg.cosine_map_path)
                    if target_column != "ingredients":
                        dedupe_map_path = dedupe_map_path.parent / f"{target_column}_dedupe_map.jsonl"
                    
                    dedupe_step = DeduplicationStep(
                        input_path=col_current_input,
                        output_path=dedup_path,
                        config={
                            "list_col": out_col,
                            "method": method,
                            method: method_config,
                            "corpus_parquet": str(col_current_input),
                        },
                        dedupe_map_path=dedupe_map_path,
                        method=method,
                        list_col=out_col,
                    )
                    self.steps.append(dedupe_step)
                    
                    if stages_cfg.apply_cosine_map:
                        col_current_input = dedup_path
                
                # Step 2c: Encoding
                if stages_cfg.encode_ids:
                    id_to_token_path = Path(output_cfg.ingredient_id_to_token)
                    token_to_id_path = Path(output_cfg.ingredient_token_to_id)
                    if target_column != "ingredients":
                        id_to_token_path = id_to_token_path.parent / f"{target_column}_id_to_token.json"
                        token_to_id_path = token_to_id_path.parent / f"{target_column}_token_to_id.json"
                    
                    encoder_step = EncodingStep(
                        input_path=col_current_input,
                        output_path=unified_path,
                        config={
                            "ingredients_col": out_col,
                            "min_freq": self.config.encoder.min_freq,
                            "dataset_id": self.config.encoder.dataset_id,
                        },
                        id_to_token_path=id_to_token_path,
                        token_to_id_path=token_to_id_path,
                        ingredients_col=out_col,
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
        self.logger.info(f"Starting Unified Pipeline Execution (mode: {self.mode})")
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
        self.logger.info("Unified Pipeline Execution Complete")
        self.logger.info("=" * 60)


def main():
    """Main entry point for unified pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Run unified preprocessing pipeline (task-based or legacy mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pipeline/config/pipeline_config.yaml",
        help="Path to pipeline configuration YAML file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference", "full"],
        default=None,
        help="Pipeline mode (for legacy configs, overrides config.pipeline.mode)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild all artifacts even if they exist",
    )
    parser.add_argument(
        "--use-task-config",
        action="store_true",
        help="Force use of task-based configuration (UnifiedConfig)",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Try to detect config type by checking for 'tasks' key
    with open(config_path, "r", encoding="utf-8") as f:
        import yaml
        cfg_dict = yaml.safe_load(f)
    
    # Determine which config type to use
    use_task_config = args.use_task_config or "tasks" in cfg_dict
    
    if use_task_config:
        # Use task-based PipelineConfig
        config = PipelineConfig.from_yaml(config_path)
        setup_logging(config.to_dict())
        logger = logging.getLogger(__name__)
        
        # Create and run task-based orchestrator
        orchestrator = TaskBasedOrchestrator(config)
        orchestrator.setup_steps()
        orchestrator.run(force=args.force)
        
        logger.info("Task-based pipeline execution finished successfully")
    else:
        # Use legacy LegacyPipelineConfig
        config = LegacyPipelineConfig.from_yaml(config_path)
        
        # Override mode if specified
        if args.mode:
            if config.pipeline is None:
                from .config import PipelineModeConfig
                config.pipeline = PipelineModeConfig()
            config.pipeline.mode = args.mode
        
        # Setup logging
        setup_logging(config.to_dict())
        logger = logging.getLogger(__name__)
        
        # Create and run legacy orchestrator
        orchestrator = UnifiedPipelineOrchestrator(config)
        orchestrator.setup_steps()
        orchestrator.run(force=args.force)
        
        logger.info("Legacy pipeline execution finished successfully")


if __name__ == "__main__":
    main()
