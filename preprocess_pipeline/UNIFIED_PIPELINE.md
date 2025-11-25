# Unified Pipeline Architecture

## Overview

The preprocessing pipeline has been expanded to support a unified, object-oriented architecture that integrates:
- **Raw Data Ingestion**: Combine multiple CSV/Parquet files
- **Model Training**: Train transformer-based NER models
- **Inference & Normalization**: Process data through normalization, deduplication, and encoding

## Architecture Components

### 1. Base Class (`pipeline/base.py`)

The `PipelineStep` base class now supports:
- **Data Transformation**: `transform(df) -> df` (DataFrame to DataFrame)
- **Model Generation**: `generate_model(df) -> Path` (DataFrame to Model Artifact)
- Common I/O, logging, and timing functionality

### 2. New Step Classes

#### `RawDataCombinerStep` (`pipeline/steps/data_combiner.py`)
- Ingests CSVs/Parquets from a directory
- Auto-detects ingredients and cuisine columns
- Combines multiple datasets into unified format
- Outputs Parquet file ready for processing

#### `NERModelTrainerStep` (`pipeline/steps/training.py`)
- Reads DataFrame with labeled ingredient data
- Converts to spaCy DocBin format
- Splits into train/validation sets
- Trains transformer-based NER model
- Saves best model checkpoint

### 3. Updated Configuration (`pipeline/config.py`)

New configuration models:

#### `PipelineModeConfig`
```yaml
pipeline:
  mode: "full"  # Options: "train", "inference", "full"
  input_path: "data/raw/"  # Default/fallback path
  model_dir: "models/ingredient_ner/"
  columns: ["ingredients", "cuisine"]
  run_ner_inference: true
  
  # Per-column input paths (optional)
  column_input_paths:
    ingredients: "data/raw/single_file.csv"  # Single file
    cuisine: "data/raw/"  # Directory (will be combined)
  
  # Mode-specific input path overrides (optional)
  # Allows different sources for train vs inference
  mode_specific_inputs:
    train:
      ingredients: "data/raw/training_data.csv"  # Single CSV for training
      cuisine: "data/combined_datasets.parquet"  # Pre-combined for training
    inference:
      ingredients: "data/combined_datasets.parquet"  # Combined for inference
      cuisine: "data/raw/single_file.csv"  # Single CSV for inference
```

**Input Path Resolution Priority:**
1. `mode_specific_inputs[mode][column]` (most specific)
2. `column_input_paths[column]` (column-specific)
3. `input_path` (default fallback)

**Input Path Types:**
- **Single file**: Path to a CSV or Parquet file (used directly)
- **Directory**: Path to directory containing multiple CSVs (will be combined)
- **Combined dataset**: Path to existing combined Parquet file (used directly)

#### `TrainingConfig`
```yaml
training:
  base_model: "roberta-base"
  epochs: 10
  batch_size: 16
  train_split: 0.8
  random_seed: 42
  lr: 5e-5
  dropout: 0.1
  window: 64
  stride: 48
  freeze_layers: 2
  use_amp: true
  early_stopping_patience: 3
```

### 4. Unified Orchestrator (`pipeline/run_pipeline.py`)

The `UnifiedPipelineOrchestrator` supports three modes:

#### Train Mode
```
RawDataCombinerStep → NERModelTrainerStep
```
- Combines raw datasets
- Trains NER model
- Saves model to disk

#### Inference Mode
```
Input Data → Normalization → Deduplication → Encoding
```
- Processes each column specified in `config.pipeline.columns`
- Applies normalization, deduplication, encoding
- Column-agnostic processing

#### Full Mode
```
RawDataCombinerStep → NERModelTrainerStep → Normalization → Deduplication → Encoding
```
- Complete end-to-end pipeline
- Trains model, then runs inference

## Usage

### Command Line

```bash
# Train mode
python -m pipeline.run_pipeline --config pipeline/config/ingrnorm.yaml --mode train

# Inference mode
python -m pipeline.run_pipeline --config pipeline/config/ingrnorm.yaml --mode inference

# Full mode (default)
python -m pipeline.run_pipeline --config pipeline/config/ingrnorm.yaml --mode full

# Force rebuild
python -m pipeline.run_pipeline --config pipeline/config/ingrnorm.yaml --force
```

### Python API

```python
from pipeline.config import PipelineConfig
from pipeline.run_pipeline import UnifiedPipelineOrchestrator
from pipeline.common.logging_setup import setup_logging

# Load config
config = PipelineConfig.from_yaml("pipeline/config/ingrnorm.yaml")

# Set mode
config.pipeline.mode = "full"  # or "train", "inference"

# Setup logging
setup_logging(config.to_dict())

# Create orchestrator
orchestrator = UnifiedPipelineOrchestrator(config)
orchestrator.setup_steps()

# Execute
orchestrator.run(force=False)
```

### Jupyter Notebook

See `notebooks/00_unified_pipeline.ipynb` for interactive examples.

## Configuration Example

### Basic Configuration

```yaml
pipeline:
  mode: "full"
  input_path: "data/raw/"
  model_dir: "models/ingredient_ner/"
  columns: ["ingredients", "cuisine"]
  run_ner_inference: true

data:
```

### Advanced: Per-Column Input Paths

This example shows how to use different input sources for different columns and modes:

```yaml
pipeline:
  mode: "full"
  input_path: "data/raw/"  # Default fallback
  model_dir: "models/ingredient_ner/"
  columns: ["ingredients", "cuisine"]
  
  # Mode-specific inputs allow different sources for train vs inference
  mode_specific_inputs:
    train:
      # Training: ingredients from single CSV, cuisines from combined dataset
      ingredients: "preprocess_pipeline/data/raw/wilmerarltstrmberg_data.csv"
      cuisine: "preprocess_pipeline/data/combined_raw_datasets.parquet"
    inference:
      # Inference: ingredients from combined dataset, cuisines from single CSV
      ingredients: "preprocess_pipeline/data/combined_raw_datasets.parquet"
      cuisine: "preprocess_pipeline/data/raw/wilmerarltstrmberg_data.csv"

data:
  input_path: "./data/raw/wilmerarltstrmberg_data.csv"
  ner_col: "NER"
  cuisine_col: "cuisine"
  chunksize: 200000

training:
  base_model: "roberta-base"
  epochs: 10
  batch_size: 16
  train_split: 0.8
  random_seed: 42

output:
  baseline_parquet: "./data/normalized/recipes_data_clean.parquet"
  dedup_parquet: "./data/normalized/recipes_data_clean_spell_dedup.parquet"
  cosine_map_path: "./data/normalized/cosine_dedupe_map.jsonl"
  list_col_for_vocab: "NER_clean"
  unified_parquet: "./data/encoded/datasets_unified.parquet"
  ingredient_id_to_token: "./data/encoded/ingredient_id_to_token.json"
  ingredient_token_to_id: "./data/encoded/ingredient_token_to_id.json"

stages:
  write_parquet: true
  sbert_dedupe: true
  w2v_dedupe: false
  apply_cosine_map: true
  encode_ids: true

# ... (sbert, w2v, encoder configs)
```

## Column-Agnostic Processing

The pipeline now processes any column specified in `config.pipeline.columns`:
- **ingredients**: Standard ingredient normalization
- **cuisine**: Cuisine normalization (with multi-cuisine splitting)
- **Any other column**: Generic normalization pipeline

Each column gets its own:
- Baseline normalized parquet
- Deduplication map
- Encoded output with ID mappings

## Benefits

1. **Unified Interface**: Single entry point for all preprocessing tasks
2. **Flexible Modes**: Train, inference, or full pipeline
3. **Column-Agnostic**: Process any column through the same pipeline
4. **Extensible**: Easy to add new steps by inheriting `PipelineStep`
5. **Testable**: Notebooks provide interactive testing and debugging

## Migration from Old Scripts

| Old Script | New Approach |
|------------|--------------|
| `combine_raw_datasets.py` | `RawDataCombinerStep` |
| `run_ingredient_ner.py` | `NERModelTrainerStep` |
| `run_ingrnorm.py` | `UnifiedPipelineOrchestrator` (inference mode) |
| `run_cuisine_norm.py` | `UnifiedPipelineOrchestrator` (with cuisine column) |

Old scripts remain for backward compatibility but are superseded by the unified pipeline.

## Testing and Debugging

1. **Notebooks**: Use `notebooks/00_unified_pipeline.ipynb` for interactive testing
2. **Logs**: Check `pipeline/logs/` for detailed execution logs
3. **Force Rebuild**: Use `--force` flag to rebuild all artifacts
4. **Step-by-Step**: Access individual steps via `orchestrator.steps[i]`

## Next Steps

- Add NER inference integration to `RawDataCombinerStep`
- Support for multiple model training (ingredients, cuisines, etc.)
- Parallel column processing
- Progress tracking and checkpointing

