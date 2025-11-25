"""
Unified Pydantic configuration models for the preprocessing pipeline.

Replaces scattered YAML loading with a single, validated configuration schema.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, field_validator
import yaml
import logging

logger = logging.getLogger(__name__)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    console: bool = True
    file: Optional[str] = None
    rotate: Dict[str, int] = Field(default_factory=lambda: {"max_bytes": 10485760, "backup_count": 5})
    fmt: str = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"


class DataConfig(BaseModel):
    """Data input/output configuration."""
    input_path: str
    ner_col: str = "NER"
    chunksize: int = 200000
    cuisine_col: str = "cuisine"
    
    @field_validator("input_path")
    @classmethod
    def validate_input_path(cls, v: str) -> str:
        path = Path(v)
        if not path.exists() and not path.is_absolute():
            # Relative path might be resolved later
            pass
        return v


class CleanupConfig(BaseModel):
    """Cleanup configuration for intermediate files."""
    enabled: bool = False
    paths: List[str] = Field(default_factory=list)


class OutputConfig(BaseModel):
    """Output paths configuration."""
    baseline_parquet: str
    dedup_parquet: str
    cosine_map_path: str
    list_col_for_vocab: str = "NER_clean"
    unified_parquet: str
    ingredient_id_to_token: str
    ingredient_token_to_id: str


class StagesConfig(BaseModel):
    """Pipeline stages enable/disable flags."""
    write_parquet: bool = True
    sbert_dedupe: bool = True
    w2v_dedupe: bool = False
    apply_cosine_map: bool = True
    encode_ids: bool = True


class SBERTConfig(BaseModel):
    """SBERT deduplication configuration."""
    model: str = "all-MiniLM-L6-v2"
    threshold: float = 0.88
    topk: int = 25
    min_len: int = 2
    require_token_overlap: bool = True
    block_generic_as_canon: bool = True
    min_freq_for_vocab: int = 2
    spacy_model: str = "en_core_web_sm"
    spacy_batch_size: int = 512
    spacy_n_process: int = 1


class W2VConfig(BaseModel):
    """Word2Vec deduplication configuration."""
    vector_size: int = 100
    window: int = 5
    min_count: int = 1
    workers: int = 4
    sg: int = 1
    epochs: int = 8
    threshold: float = 0.85
    topk: int = 25
    min_freq_for_vocab: int = 2


class EncoderConfig(BaseModel):
    """Encoder configuration."""
    min_freq: int = 1
    dataset_id: int = 1
    ingredients_col: str = "NER_clean"


class TrainingConfig(BaseModel):
    """NER model training configuration."""
    enabled: bool = False
    input_path: Optional[str] = None
    model_output: Optional[str] = None
    # Support for params dict (from YAML) - these override defaults
    params: Dict[str, Any] = Field(default_factory=dict)
    
    # Default values (can be overridden by params)
    base_model: str = "roberta-base"
    epochs: int = 10
    batch_size: int = 16
    train_split: float = 0.8
    random_seed: int = 42
    valid_fraction: Optional[float] = None  # If None, calculated from train_split
    lr: float = 5e-5
    dropout: float = 0.1
    window: int = 64
    stride: int = 48
    freeze_layers: int = 2
    use_amp: bool = True
    early_stopping_patience: int = 3
    use_tok2vec_debug: bool = False
    max_train_docs: Optional[int] = None
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """Get parameter value, checking params dict first, then attribute."""
        if key in self.params:
            return self.params[key]
        return getattr(self, key, default)
    
    def model_dump(self) -> Dict[str, Any]:
        """Convert to dict, merging params if present."""
        base_dict = super().model_dump(exclude={"params"})
        # Merge params into base dict (params override defaults)
        if self.params:
            base_dict.update(self.params)
        return base_dict


class PipelineModeConfig(BaseModel):
    """
    Pipeline execution mode configuration.
    
    Supports per-column input paths for different data sources.
    """
    mode: str = "full"  # Options: "train", "inference", "full"
    input_path: str = "data/raw/"  # Default/fallback input path
    model_dir: str = "models/ingredient_ner/"
    columns: List[str] = Field(default_factory=lambda: ["ingredients", "cuisine"])
    run_ner_inference: bool = True
    
    # Per-column input paths (overrides input_path for specific columns)
    # Format: {column_name: input_path}
    # - For single file: provide file path
    # - For multiple files: provide directory path (will be combined)
    # - For combined dataset: provide path to existing combined parquet
    column_input_paths: Dict[str, Union[str, List[str]]] = Field(default_factory=dict)
    
    # Mode-specific input path overrides
    # Format: {mode: {column_name: input_path}}
    # Allows different input sources for train vs inference modes
    mode_specific_inputs: Dict[str, Dict[str, Union[str, List[str]]]] = Field(default_factory=dict)
    
    def get_input_path_for_column(self, column_name: str, mode: Optional[str] = None) -> Union[str, List[str]]:
        """
        Get input path for a specific column, respecting mode-specific overrides.
        
        Priority:
        1. mode_specific_inputs[mode][column_name] (if mode is specified)
        2. column_input_paths[column_name]
        3. input_path (default)
        
        Args:
            column_name: Name of the column
            mode: Pipeline mode (if None, uses self.mode)
            
        Returns:
            Input path (string) or list of paths for combining
        """
        mode = mode or self.mode
        
        # Check mode-specific inputs first
        if mode in self.mode_specific_inputs:
            if column_name in self.mode_specific_inputs[mode]:
                return self.mode_specific_inputs[mode][column_name]
        
        # Check column-specific inputs
        if column_name in self.column_input_paths:
            return self.column_input_paths[column_name]
        
        # Fall back to default
        return self.input_path


class GlobalConfig(BaseModel):
    """Global settings for the pipeline."""
    base_dir: str = "./data"
    logging_level: str = "INFO"


class IngestionConfig(BaseModel):
    """Data ingestion configuration."""
    enabled: bool = True
    input_dir: str = "./data/raw"
    output_file: str = "./data/intermediate/combined_raw.parquet"
    column_mapping: Dict[str, str] = Field(default_factory=dict)


class TaskStepConfig(BaseModel):
    """Configuration for a single processing step within a task."""
    name: str
    type: str  # e.g., 'spacy', 'sbert', 'w2v', 'encoder', 'list_splitter'
    params: Dict[str, Any] = Field(default_factory=dict)


class ProcessingTaskConfig(BaseModel):
    """Configuration for a processing task (e.g., process_ingredients, process_cuisine)."""
    name: str
    enabled: bool = True
    input_path: str
    output_path: str
    target_column: str  # The input column (e.g., 'NER', 'cuisine_raw')
    output_column: str  # The output column (e.g., 'NER_clean', 'cuisine_clean')
    steps: List[TaskStepConfig] = Field(default_factory=list)
    
    @field_validator("input_path", "output_path")
    @classmethod
    def validate_paths(cls, v: str) -> str:
        """Validate that paths are strings (existence checked at runtime)."""
        return str(v)


class TrainingTaskConfig(BaseModel):
    """Configuration for a training task (e.g., ingredient_ner_model, cuisine_classification_model)."""
    name: str
    enabled: bool = True
    task_type: str  # e.g., "token_classification", "text_classification"
    input_path: str
    model_dir: str
    params: Dict[str, Any] = Field(default_factory=dict)
    
    # Column specifications (varies by task_type)
    target_column: Optional[str] = None  # For token_classification: input text column
    label_column: Optional[str] = None  # For token_classification: label/NER list column
    text_column: Optional[str] = None    # For text_classification: feature text column
    # label_column is also used for text_classification: target label column
    
    @field_validator("input_path", "model_dir")
    @classmethod
    def validate_paths(cls, v: str) -> str:
        """Validate that paths are strings (existence checked at runtime)."""
        return str(v)


class LegacyPipelineConfig(BaseModel):
    """
    Legacy pipeline configuration (for backward compatibility).
    
    This model validates and holds all configuration for the preprocessing pipeline.
    Maintained for backward compatibility with old config files.
    """
    pipeline: Optional[PipelineModeConfig] = Field(default_factory=PipelineModeConfig)
    data: Optional[DataConfig] = None
    output: Optional[OutputConfig] = None
    stages: Optional[StagesConfig] = None
    sbert: Optional[SBERTConfig] = None
    w2v: Optional[W2VConfig] = None
    encoder: Optional[EncoderConfig] = None
    training: Optional[TrainingConfig] = Field(default_factory=TrainingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cleanup: CleanupConfig = Field(default_factory=CleanupConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> "LegacyPipelineConfig":
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            LegacyPipelineConfig instance
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
        
        if not isinstance(cfg_dict, dict):
            raise ValueError(f"YAML file did not parse to a dict: {yaml_path}")
        
        return cls(**cfg_dict)
    
    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any]) -> "LegacyPipelineConfig":
        """
        Create configuration from dictionary.
        
        Args:
            cfg_dict: Configuration dictionary
            
        Returns:
            LegacyPipelineConfig instance
        """
        return cls(**cfg_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
    
    def get_step_config(self, step_name: str) -> Dict[str, Any]:
        """
        Get step-specific configuration.
        
        Args:
            step_name: Name of the step (e.g., "sbert", "w2v", "encoder")
            
        Returns:
            Configuration dictionary for the step
        """
        if hasattr(self, step_name):
            return getattr(self, step_name).model_dump()
        return {}


class PipelineConfig(BaseModel):
    """
    Task-based pipeline configuration.
    
    This is the main configuration format that supports:
    - Multiple independent processing tasks
    - Per-task step configurations
    - Different parameters for different columns (e.g., ingredients vs cuisine)
    - Multiple training tasks
    """
    global_settings: Optional[GlobalConfig] = Field(default_factory=GlobalConfig)
    ingestion: Optional[IngestionConfig] = Field(default_factory=IngestionConfig)
    tasks: List[ProcessingTaskConfig] = Field(default_factory=list)
    training_tasks: List[TrainingTaskConfig] = Field(default_factory=list)
    logging: Optional[LoggingConfig] = Field(default_factory=LoggingConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> "PipelineConfig":
        """
        Load pipeline configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            PipelineConfig instance
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
        
        if not isinstance(cfg_dict, dict):
            raise ValueError(f"YAML file did not parse to a dict: {yaml_path}")
        
        # Handle 'global' key (rename to 'global_settings' for internal use)
        if 'global' in cfg_dict and 'global_settings' not in cfg_dict:
            cfg_dict['global_settings'] = cfg_dict.pop('global')
        
        return cls(**cfg_dict)
    
    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any]) -> "PipelineConfig":
        """
        Create pipeline configuration from dictionary.
        
        Args:
            cfg_dict: Configuration dictionary
            
        Returns:
            PipelineConfig instance
        """
        # Handle 'global' key
        if 'global' in cfg_dict and 'global_settings' not in cfg_dict:
            cfg_dict['global_settings'] = cfg_dict.pop('global')
        return cls(**cfg_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = self.model_dump()
        # Rename global_settings back to global for YAML compatibility
        if 'global_settings' in result and 'global' not in result:
            result['global'] = result.pop('global_settings')
        return result

