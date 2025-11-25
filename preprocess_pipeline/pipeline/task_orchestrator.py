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
        
        # Separate tasks into training and inference phases
        # Training phase: process ingredients for training -> train ingredient model
        # Inference phase: process ingredients with NER -> process cuisine -> train cuisine -> encode
        
        # Find ingredient training task (if any)
        ingredient_training_task = None
        for training_task in self.config.training_tasks:
            if training_task.enabled and training_task.task_type == "token_classification":
                # Check if it's for ingredients (by name or label_column)
                if "ingredient" in training_task.name.lower() or training_task.label_column == "ingredients_raw":
                    ingredient_training_task = training_task
                    break
        
        # Find cuisine training task (if any)
        cuisine_training_task = None
        for training_task in self.config.training_tasks:
            if training_task.enabled and training_task.task_type == "text_classification":
                if "cuisine" in training_task.name.lower():
                    cuisine_training_task = training_task
                    break
        
        # Separate processing tasks by type
        ingredient_task = None
        cuisine_task = None
        for task in self.config.tasks:
            if "ingredient" in task.name.lower():
                ingredient_task = task
            elif "cuisine" in task.name.lower():
                cuisine_task = task
        
        # ============================================
        # PHASE 1: TRAINING DATA PREPARATION
        # ============================================
        if ingredient_training_task:
            self.logger.info("=" * 60)
            self.logger.info("PHASE 1: Training Data Preparation")
            self.logger.info("=" * 60)
            
            # Step 1.1: Load training data (if training task has different input)
            training_input = Path(ingredient_training_task.input_path)
            if not training_input.exists():
                # If training input doesn't exist, use ingestion
                if self.config.ingestion and self.config.ingestion.enabled:
                    ingestion = self.config.ingestion
                    self.logger.info(f"Loading training data via ingestion: {ingestion.input_dir}")
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
                    training_input = Path(ingestion.output_file)
            
            # Step 1.2: Process ingredients for training (normalization + deduplication, NO encoding)
            if ingredient_task:
                self.logger.info("Processing ingredients for training (no encoding)...")
                training_steps = [s for s in ingredient_task.steps if s.type != "encoder"]
                if training_steps:
                    current_input = training_input
                    current_output_column = ingredient_task.target_column
                    
                    for step_idx, step_config in enumerate(training_steps):
                        # Use temporary output path for training processing
                        step_output = training_input.parent / f"ingredients_training_step_{step_idx}_{step_config.name}.parquet"
                        
                        step_input_col = ingredient_task.target_column if step_idx == 0 else current_output_column
                        
                        if step_idx == len(training_steps) - 1:
                            step_output_col = f"{ingredient_task.output_column}_training"
                        else:
                            if step_config.type in ("spacy", "normalization"):
                                step_output_col = f"{step_input_col}_normalized"
                            elif step_config.type in ("sbert", "w2v"):
                                step_output_col = f"{step_input_col}_deduped"
                            else:
                                step_output_col = f"{step_input_col}_processed"
                        
                        if step_config.type in ("sbert", "w2v"):
                            dedup_col = step_input_col
                            step = self._create_step_from_config(
                                step_config=step_config,
                                input_path=current_input,
                                output_path=step_output,
                                target_column=dedup_col,
                                output_column=dedup_col,
                                task_name=f"{ingredient_task.name}_training",
                            )
                            current_output_column = step_input_col
                        else:
                            step = self._create_step_from_config(
                                step_config=step_config,
                                input_path=current_input,
                                output_path=step_output,
                                target_column=step_input_col,
                                output_column=step_output_col,
                                task_name=f"{ingredient_task.name}_training",
                            )
                            current_output_column = step_output_col
                        
                        if step is not None:
                            self.steps.append(step)
                            current_input = step_output
                    
                    # Update training task input to use processed training data
                    training_input = current_input
            
            # Step 1.3: Train ingredient transformer
            self.logger.info("Training ingredient NER model...")
            target_col = ingredient_training_task.label_column
            training_params = ingredient_training_task.params or {}
            hpo_settings = training_params.pop("hpo", None) if "hpo" in training_params else None
            
            trainer_config = {
                "target_column": target_col,
                "base_model": training_params.get("base_model", "roberta-base"),
                "epochs": training_params.get("epochs", 10),
                "batch_size": training_params.get("batch_size", 16),
                "train_split": training_params.get("train_split", 0.8),
                "random_seed": training_params.get("random_seed", 42),
                "training": training_params,
            }
            if hpo_settings:
                trainer_config["hpo"] = hpo_settings
            
            trainer = NERModelTrainerStep(
                input_path=training_input,
                output_path=Path(ingredient_training_task.model_dir),
                config=trainer_config,
                model_dir=Path(ingredient_training_task.model_dir),
            )
            self.steps.append(trainer)
        
        # ============================================
        # PHASE 2: INFERENCE DATA PROCESSING
        # ============================================
        self.logger.info("=" * 60)
        self.logger.info("PHASE 2: Inference Data Processing")
        self.logger.info("=" * 60)
        
        # Step 2.1: Load combined_raw for inference
        if self.config.ingestion and self.config.ingestion.enabled:
            ingestion = self.config.ingestion
            inference_input = Path(ingestion.output_file)
            
            # Only add ingestion step if it wasn't already added in training phase
            if not ingredient_training_task or Path(ingredient_training_task.input_path).exists():
                self.logger.info(f"Loading inference data via ingestion: {ingestion.input_dir}")
                combiner = RawDataCombinerStep(
                    input_path=Path(ingestion.input_dir),
                    output_path=Path(ingestion.output_file),
                    config={
                        "ingredients_col": ingestion.column_mapping.get("ingredients"),
                        "cuisine_col": ingestion.column_mapping.get("cuisine"),
                        "run_ner_inference": True,  # Enable NER inference now that model is trained
                        "ner_model_path": str(Path(ingredient_training_task.model_dir)) if ingredient_training_task else None,
                    },
                    data_dir=Path(ingestion.input_dir),
                )
                self.steps.append(combiner)
        
        # Step 2.2: Process ingredients with inference (normalization with NER + deduplication, NO encoding)
        if ingredient_task:
            self.logger.info("Processing ingredients with NER inference (no encoding)...")
            inference_steps = [s for s in ingredient_task.steps if s.type != "encoder"]
            if inference_steps:
                current_input = inference_input
                current_output_column = ingredient_task.target_column
                
                for step_idx, step_config in enumerate(inference_steps):
                    step_output = Path(ingredient_task.output_path).parent / f"ingredients_inference_step_{step_idx}_{step_config.name}.parquet"
                    
                    step_input_col = ingredient_task.target_column if step_idx == 0 else current_output_column
                    
                    if step_idx == len(inference_steps) - 1:
                        step_output_col = f"{ingredient_task.output_column}_inference"
                    else:
                        if step_config.type in ("spacy", "normalization"):
                            step_output_col = f"{step_input_col}_normalized"
                        elif step_config.type in ("sbert", "w2v"):
                            step_output_col = f"{step_input_col}_deduped"
                        else:
                            step_output_col = f"{step_input_col}_processed"
                    
                    if step_config.type in ("sbert", "w2v"):
                        dedup_col = step_input_col
                        step = self._create_step_from_config(
                            step_config=step_config,
                            input_path=current_input,
                            output_path=step_output,
                            target_column=dedup_col,
                            output_column=dedup_col,
                            task_name=f"{ingredient_task.name}_inference",
                        )
                        current_output_column = step_input_col
                    else:
                        step = self._create_step_from_config(
                            step_config=step_config,
                            input_path=current_input,
                            output_path=step_output,
                            target_column=step_input_col,
                            output_column=step_output_col,
                            task_name=f"{ingredient_task.name}_inference",
                        )
                        current_output_column = step_output_col
                    
                    if step is not None:
                        self.steps.append(step)
                        current_input = step_output
                
                # Store inference output for later encoding
                ingredient_inference_output = current_input
                ingredient_inference_column = current_output_column
        
        # Step 2.3: Process cuisine (normalization + deduplication, NO encoding)
        if cuisine_task:
            self.logger.info("Processing cuisine (no encoding)...")
            cuisine_steps = [s for s in cuisine_task.steps if s.type != "encoder"]
            if cuisine_steps:
                # Use combined_raw as input for cuisine
                current_input = inference_input
                current_output_column = cuisine_task.target_column
                
                for step_idx, step_config in enumerate(cuisine_steps):
                    step_output = Path(cuisine_task.output_path).parent / f"cuisine_step_{step_idx}_{step_config.name}.parquet"
                    
                    step_input_col = cuisine_task.target_column if step_idx == 0 else current_output_column
                    
                    if step_idx == len(cuisine_steps) - 1:
                        step_output_col = f"{cuisine_task.output_column}_no_encode"
                    else:
                        if step_config.type in ("spacy", "normalization"):
                            step_output_col = f"{step_input_col}_normalized"
                        elif step_config.type in ("sbert", "w2v"):
                            step_output_col = f"{step_input_col}_deduped"
                        elif step_config.type in ("list_splitter", "preprocessing"):
                            step_output_col = f"{step_input_col}_preprocessed"
                        else:
                            step_output_col = f"{step_input_col}_processed"
                    
                    if step_config.type in ("sbert", "w2v"):
                        dedup_col = step_input_col
                        step = self._create_step_from_config(
                            step_config=step_config,
                            input_path=current_input,
                            output_path=step_output,
                            target_column=dedup_col,
                            output_column=dedup_col,
                            task_name=cuisine_task.name,
                        )
                        current_output_column = step_input_col
                    else:
                        step = self._create_step_from_config(
                            step_config=step_config,
                            input_path=current_input,
                            output_path=step_output,
                            target_column=step_input_col,
                            output_column=step_output_col,
                            task_name=cuisine_task.name,
                        )
                        current_output_column = step_output_col
                    
                    if step is not None:
                        self.steps.append(step)
                        current_input = step_output
                
                # Store processed output for later encoding
                cuisine_processed_output = current_input
                cuisine_processed_column = current_output_column
        
        # Step 2.4: Train cuisine transformer (if enabled)
        if cuisine_training_task:
            self.logger.info("Training cuisine classification model...")
            # Use processed cuisine data as input
            cuisine_training_input = cuisine_processed_output if cuisine_task else Path(cuisine_training_task.input_path)
            target_col = cuisine_training_task.label_column
            training_params = cuisine_training_task.params or {}
            hpo_settings = training_params.pop("hpo", None) if "hpo" in training_params else None
            
            trainer_config = {
                "target_column": target_col,
                "base_model": training_params.get("base_model", "roberta-base"),
                "epochs": training_params.get("epochs", 10),
                "batch_size": training_params.get("batch_size", 16),
                "train_split": training_params.get("train_split", 0.8),
                "random_seed": training_params.get("random_seed", 42),
                "training": training_params,
            }
            if hpo_settings:
                trainer_config["hpo"] = hpo_settings
            
            trainer = NERModelTrainerStep(
                input_path=cuisine_training_input,
                output_path=Path(cuisine_training_task.model_dir),
                config=trainer_config,
                model_dir=Path(cuisine_training_task.model_dir),
            )
            self.steps.append(trainer)
        
        # ============================================
        # PHASE 3: ENCODING (Final step)
        # ============================================
        self.logger.info("=" * 60)
        self.logger.info("PHASE 3: Encoding")
        self.logger.info("=" * 60)
        
        # Step 3.1: Encode ingredients
        if ingredient_task:
            encoding_step = [s for s in ingredient_task.steps if s.type == "encoder"]
            if encoding_step:
                encoding_config = encoding_step[0]
                # Use inference output as input for encoding
                if 'ingredient_inference_output' in locals():
                    encoding_input = ingredient_inference_output
                    encoding_column = ingredient_inference_column
                else:
                    encoding_input = inference_input
                    encoding_column = ingredient_task.target_column
                
                step = self._create_step_from_config(
                    step_config=encoding_config,
                    input_path=encoding_input,
                    output_path=Path(ingredient_task.output_path),
                    target_column=encoding_column,
                    output_column=ingredient_task.output_column,
                    task_name=ingredient_task.name,
                )
                if step is not None:
                    self.steps.append(step)
        
        # Step 3.2: Encode cuisine
        if cuisine_task:
            encoding_step = [s for s in cuisine_task.steps if s.type == "encoder"]
            if encoding_step:
                encoding_config = encoding_step[0]
                # Use processed cuisine output as input for encoding
                if 'cuisine_processed_output' in locals():
                    encoding_input = cuisine_processed_output
                    encoding_column = cuisine_processed_column
                else:
                    encoding_input = inference_input
                    encoding_column = cuisine_task.target_column
                
                step = self._create_step_from_config(
                    step_config=encoding_config,
                    input_path=encoding_input,
                    output_path=Path(cuisine_task.output_path),
                    target_column=encoding_column,
                    output_column=cuisine_task.output_column,
                    task_name=cuisine_task.name,
                )
                if step is not None:
                    self.steps.append(step)
        
        # All steps have been configured in the phases above
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

