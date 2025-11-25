"""
Task-based pipeline orchestrator.

Processes tasks dynamically based on UnifiedConfig, allowing different
steps and parameters for each column (e.g., ingredients vs cuisine).
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from .config import PipelineConfig, ProcessingTaskConfig, TaskStepConfig, TrainingTaskConfig
from .base import PipelineStep
from .steps import (
    RawDataCombinerStep,
    NERModelTrainerStep,
    IngredientNormalizerStep,
    DeduplicationStep,
    EncodingStep,
    CuisinePreprocessingStep,
)

logger = logging.getLogger(__name__)


class TaskBasedOrchestrator:
    """
    Task-based orchestrator for the preprocessing pipeline.
    
    Processes tasks dynamically, allowing:
    - Different input/output paths per task
    - Different step configurations per task
    - Different parameters for different columns
    - Column renaming (e.g., 'NER' -> 'NER_clean')
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize orchestrator with pipeline configuration.
        
        Args:
            config: PipelineConfig instance
        """
        self.config = config
        self.steps: List[PipelineStep] = []
        self.logger = logging.getLogger(f"{__name__}.TaskOrchestrator")
    
    def _create_step_from_config(
        self,
        step_config: TaskStepConfig,
        input_path: Path,
        output_path: Path,
        target_column: str,
        output_column: str,
        task_name: str,
    ) -> Optional[PipelineStep]:
        """
        Create a pipeline step from step configuration.
        
        Args:
            step_config: TaskStepConfig for the step
            input_path: Input file path
            output_path: Output file path
            target_column: Input column name
            output_column: Output column name
            task_name: Name of the task (for logging)
            
        Returns:
            PipelineStep instance, or None if step type is unknown
        """
        step_type = step_config.type.lower()
        params = step_config.params or {}
        
        self.logger.info(f"Creating step '{step_config.name}' (type: {step_type})")
        
        if step_type == "spacy" or step_type == "normalization":
            return IngredientNormalizerStep(
                input_path=input_path,
                output_path=output_path,
                config={
                    "list_col": target_column,
                    "out_col": output_column,
                    "spacy_model": params.get("model", "en_core_web_sm"),
                    "batch_size": params.get("batch_size", 512),
                    "n_process": params.get("n_process", 1),
                },
            )
        
        elif step_type == "sbert":
            # Determine dedupe map path
            dedupe_map_path = output_path.parent / f"{task_name}_dedupe_map.jsonl"
            
            # Deduplication reads from target_column and writes to output_column
            # The apply_map_to_parquet_streaming modifies the column in place,
            # but we control which column via list_col parameter
            return DeduplicationStep(
                input_path=input_path,
                output_path=output_path,
                config={
                    "list_col": target_column,  # Read from input column
                    "method": "sbert",
                    "sbert": {
                        "model": params.get("model", "all-MiniLM-L6-v2"),
                        "threshold": params.get("threshold", 0.88),
                        "topk": params.get("topk", 25),
                        "min_len": params.get("min_len", 2),
                        "require_token_overlap": params.get("require_token_overlap", True),
                        "block_generic_as_canon": params.get("block_generic_as_canon", True),
                        "min_freq_for_vocab": params.get("min_freq_for_vocab", 2),
                        "spacy_model": params.get("spacy_model", "en_core_web_sm"),
                        "spacy_batch_size": params.get("spacy_batch_size", 512),
                        "spacy_n_process": params.get("spacy_n_process", 1),
                    },
                    "corpus_parquet": str(input_path),
                },
                dedupe_map_path=dedupe_map_path,
                method="sbert",
                list_col=target_column,  # Process this column
            )
        
        elif step_type == "w2v":
            dedupe_map_path = output_path.parent / f"{task_name}_dedupe_map.jsonl"
            
            return DeduplicationStep(
                input_path=input_path,
                output_path=output_path,
                config={
                    "list_col": target_column,  # Read from input column
                    "method": "w2v",
                    "w2v": {
                        "vector_size": params.get("vector_size", 100),
                        "window": params.get("window", 5),
                        "min_count": params.get("min_count", 1),
                        "workers": params.get("workers", 4),
                        "sg": params.get("sg", 1),
                        "epochs": params.get("epochs", 8),
                        "threshold": params.get("threshold", 0.85),
                        "topk": params.get("topk", 25),
                        "min_freq_for_vocab": params.get("min_freq_for_vocab", 2),
                    },
                    "corpus_parquet": str(input_path),
                },
                dedupe_map_path=dedupe_map_path,
                method="w2v",
                list_col=target_column,  # Process this column
            )
        
        elif step_type == "encoder" or step_type == "token_to_id":
            # Determine encoder map paths
            id_to_token_path = Path(params.get("vocab_file", str(output_path.parent / f"{task_name}_id_to_token.json")))
            token_to_id_path = id_to_token_path.parent / f"{task_name}_token_to_id.json"
            
            return EncodingStep(
                input_path=input_path,
                output_path=output_path,
                config={
                    "ingredients_col": target_column if input_path == output_path else output_column,
                    "min_freq": params.get("min_freq", 1),
                    "dataset_id": params.get("dataset_id", 1),
                },
                id_to_token_path=id_to_token_path,
                token_to_id_path=token_to_id_path,
                ingredients_col=target_column if input_path == output_path else output_column,
                min_freq=params.get("min_freq", 1),
                dataset_id=params.get("dataset_id", 1),
            )
        
        elif step_type == "list_splitter" or step_type == "preprocessing":
            # For cuisine preprocessing (splitting multi-cuisine entries)
            return CuisinePreprocessingStep(
                input_path=input_path,
                output_path=output_path,
                config={
                    "cuisine_col": target_column,
                },
            )
        
        else:
            self.logger.warning(f"Unknown step type: {step_type}. Skipping step '{step_config.name}'")
            return None
    
    def setup_steps(self) -> None:
        """Instantiate pipeline steps based on task configuration."""
        self.logger.info("Setting up task-based pipeline steps...")
        
        # Step 0: Data Ingestion (if enabled)
        if self.config.ingestion and self.config.ingestion.enabled:
            ingestion = self.config.ingestion
            self.logger.info(f"Setting up ingestion: {ingestion.input_dir} -> {ingestion.output_file}")
            
            combiner = RawDataCombinerStep(
                input_path=Path(ingestion.input_dir),
                output_path=Path(ingestion.output_file),
                config={
                    "ingredients_col": ingestion.column_mapping.get("ingredients"),
                    "cuisine_col": ingestion.column_mapping.get("cuisine"),
                    "run_ner_inference": False,
                },
                data_dir=Path(ingestion.input_dir),
            )
            self.steps.append(combiner)
        
        # Step 1: Process each task
        for task in self.config.tasks:
            if not task.enabled:
                self.logger.info(f"Skipping disabled task: {task.name}")
                continue
            
            self.logger.info(f"Setting up task: {task.name}")
            self.logger.info(f"  Input: {task.input_path}")
            self.logger.info(f"  Output: {task.output_path}")
            self.logger.info(f"  Target column: {task.target_column} -> {task.output_column}")
            
            # Track current input/output for this task
            current_input = Path(task.input_path)
            current_output_column = task.target_column  # Starts with target, updates as steps process
            
            # Process each step in the task
            for step_idx, step_config in enumerate(task.steps):
                # Determine intermediate output path
                if step_idx == len(task.steps) - 1:
                    # Last step: use final output path
                    step_output = Path(task.output_path)
                else:
                    # Intermediate step: create temporary output
                    step_output = Path(task.output_path).parent / f"{task.name}_step_{step_idx}_{step_config.name}.parquet"
                
                # Determine input/output columns
                # First step reads target_column, subsequent steps read previous output_column
                # Important: We preserve the original column and add new columns
                step_input_col = task.target_column if step_idx == 0 else current_output_column
                
                # Determine output column based on step type
                # For normalization: adds new column (preserves original)
                # For deduplication: modifies column in place (preserves original, modifies processed column)
                # For encoding: creates new format (typically final)
                if step_idx == len(task.steps) - 1:
                    # Final step: use the task's output_column
                    step_output_col = task.output_column
                else:
                    # Intermediate step: create temporary column name
                    if step_config.type in ("spacy", "normalization"):
                        # Normalization adds a new column (e.g., 'NER' -> 'NER_clean')
                        step_output_col = f"{step_input_col}_normalized"
                    elif step_config.type in ("sbert", "w2v"):
                        # Deduplication modifies the column in place
                        # But we want to track it, so use a descriptive name
                        step_output_col = f"{step_input_col}_deduped"
                    elif step_config.type in ("list_splitter", "preprocessing"):
                        step_output_col = f"{step_input_col}_preprocessed"
                    else:
                        step_output_col = f"{step_input_col}_processed"
                
                # Special handling: For deduplication, if we want to write to a new column,
                # we need to adjust. But currently deduplication modifies in place.
                # For now, we'll use the same column name for deduplication output
                # (it modifies the input column in place, preserving other columns)
                if step_config.type in ("sbert", "w2v"):
                    # Deduplication modifies the column in place, so output_col = input_col
                    # But we track it as _deduped for clarity in the pipeline
                    actual_dedup_col = step_input_col  # The column that gets modified
                    step_output_col = step_output_col  # The name we use for tracking
                
                # Create step
                # For deduplication, we need to handle the column name correctly
                # since it modifies the column in place (via apply_map_to_parquet_streaming)
                if step_config.type in ("sbert", "w2v"):
                    # Deduplication modifies the input column in place
                    # So the output column is the same as input (but modified)
                    dedup_col = step_input_col
                    step = self._create_step_from_config(
                        step_config=step_config,
                        input_path=current_input,
                        output_path=step_output,
                        target_column=dedup_col,
                        output_column=dedup_col,  # Modified in place, same name
                        task_name=task.name,
                    )
                    # After deduplication, the column name is unchanged (but content is modified)
                    # For next step, we continue using the same column name
                    current_output_column = step_input_col
                else:
                    # Other steps (normalization, preprocessing, encoding) add new columns
                    step = self._create_step_from_config(
                        step_config=step_config,
                        input_path=current_input,
                        output_path=step_output,
                        target_column=step_input_col,
                        output_column=step_output_col,
                        task_name=task.name,
                    )
                    # Update to use the new output column for next step
                    current_output_column = step_output_col
                
                if step is None:
                    self.logger.warning(f"Step '{step_config.name}' could not be created, skipping")
                    continue
                
                self.steps.append(step)
                
                # Update for next step
                current_input = step_output
                # current_output_column already updated above based on step type
        
        # Step 2: Training Tasks (iterate through training_tasks list)
        for training_task in self.config.training_tasks:
            if not training_task.enabled:
                self.logger.info(f"Skipping disabled training task: {training_task.name}")
                continue
            
            self.logger.info(f"Setting up training task: {training_task.name}")
            self.logger.info(f"  Task type: {training_task.task_type}")
            self.logger.info(f"  Input: {training_task.input_path}")
            self.logger.info(f"  Model output: {training_task.model_dir}")
            
            # Determine target column based on task_type
            # NERModelTrainerStep expects the column with labeled entities (list format)
            if training_task.task_type == "token_classification":
                # For NER: use label_column (the list of entities/labels)
                target_col = training_task.label_column
                if target_col is None:
                    self.logger.warning(f"Training task '{training_task.name}' missing label_column for token_classification, skipping")
                    continue
            elif training_task.task_type == "text_classification":
                # For classification: use label_column (the target labels)
                # Note: NERModelTrainerStep is designed for NER, so text_classification
                # would need a different trainer step (not yet implemented)
                self.logger.warning(f"Training task '{training_task.name}' uses text_classification which is not yet supported by NERModelTrainerStep, skipping")
                continue
            else:
                self.logger.warning(f"Unknown task_type '{training_task.task_type}' for training task '{training_task.name}', skipping")
                continue
            
            training_input = Path(training_task.input_path)
            model_dir = Path(training_task.model_dir)
            
            # Extract training parameters from params dict
            training_params = training_task.params or {}
            
            # Create trainer step
            trainer = NERModelTrainerStep(
                input_path=training_input,
                output_path=model_dir,
                config={
                    "target_column": target_col,
                    "base_model": training_params.get("base_model", "roberta-base"),
                    "epochs": training_params.get("epochs", 10),
                    "batch_size": training_params.get("batch_size", 16),
                    "train_split": training_params.get("train_split", 0.8),
                    "random_seed": training_params.get("random_seed", 42),
                    "training": training_params,
                },
                model_dir=model_dir,
            )
            self.steps.append(trainer)
        
        self.logger.info(f"Setup complete: {len(self.steps)} step(s) configured")
    
    def run(self, force: bool = False) -> None:
        """
        Execute all pipeline steps in sequence.
        
        Args:
            force: If True, rebuild artifacts even if they exist
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Task-Based Pipeline Execution")
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
        self.logger.info("Task-Based Pipeline Execution Complete")
        self.logger.info("=" * 60)

