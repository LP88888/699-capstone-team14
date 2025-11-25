# Task-Based Pipeline Architecture

## Overview

The pipeline has been refactored to support a **task-based architecture** that allows:
- **Different input/output paths per task**
- **Different step configurations per task**
- **Different parameters for different columns** (e.g., ingredients vs cuisine)
- **Column renaming** (e.g., read 'NER' → write 'NER_clean', preserving original)

## Architecture

### Configuration Structure

The new `UnifiedConfig` supports:

```yaml
global_settings:
  base_dir: "./data"
  model_dir: "./models"
  logging_level: "INFO"
  device: "cpu"

ingestion:
  enabled: true
  input_dir: "./data/raw"
  output_file: "./data/intermediate/combined_raw.parquet"
  column_mapping:
    "ingredients": "ingredients_raw"
    "NER": "ingredients_list"
    "cuisine": "cuisine_raw"

tasks:
  - name: "process_ingredients"
    enabled: true
    input_path: "./data/intermediate/combined_raw.parquet"
    output_path: "./data/processed/ingredients_encoded.parquet"
    target_column: "NER"           # Read this column
    output_column: "NER_clean"      # Write to this column (original preserved)
    steps:
      - name: "normalization"
        type: "spacy"
        params:
          model: "en_core_web_sm"
          batch_size: 1024
      
      - name: "deduplication"
        type: "sbert"
        params:
          model: "all-MiniLM-L6-v2"
          threshold: 0.88  # Strict for ingredients
      
      - name: "encoding"
        type: "encoder"
        params:
          min_freq: 1

  - name: "process_cuisine"
    enabled: true
    input_path: "./data/intermediate/combined_raw.parquet"
    output_path: "./data/processed/cuisine_encoded.parquet"
    target_column: "cuisine_raw"
    output_column: "cuisine_clean"
    steps:
      - name: "preprocessing"
        type: "list_splitter"  # Splits "[Italian, American]" strings
      
      - name: "deduplication"
        type: "sbert"
        params:
          threshold: 0.92  # Different threshold for cuisine!
      
      - name: "encoding"
        type: "encoder"
        params:
          min_freq: 1

training:
  enabled: false
  input_path: "./data/processed/ingredients_encoded.parquet"
  model_output: "./models/ner_model"
  params:
    epochs: 10
    base_model: "roberta-base"
```

### Key Features

#### 1. **Per-Task Configuration**
Each task has its own:
- `input_path`: Where to read data from
- `output_path`: Where to write results
- `target_column`: Column to read from
- `output_column`: Column to write to (original preserved)
- `steps`: List of processing steps with their own parameters

#### 2. **Column Renaming**
- **Normalization**: Reads `target_column` (e.g., 'NER'), adds `output_column` (e.g., 'NER_clean')
- **Deduplication**: Modifies the processed column in place (preserves original)
- **Encoding**: Creates encoded format from processed column
- **Original columns are always preserved** unless explicitly overwritten

#### 3. **Step Types**

| Step Type | Description | Parameters |
|-----------|-------------|------------|
| `spacy` / `normalization` | spaCy-based normalization | `model`, `batch_size`, `n_process` |
| `sbert` | SBERT deduplication | `model`, `threshold`, `topk`, etc. |
| `w2v` | Word2Vec deduplication | `vector_size`, `threshold`, etc. |
| `encoder` / `token_to_id` | Token-to-ID encoding | `min_freq`, `dataset_id` |
| `list_splitter` / `preprocessing` | Split multi-value strings | (cuisine-specific) |

#### 4. **Different Parameters Per Task**
- Ingredients: `threshold: 0.88` (stricter)
- Cuisine: `threshold: 0.92` (even stricter)
- Each task can have completely different step configurations

## Usage

### Command Line

```bash
# Run task-based pipeline
python -m pipeline.run_pipeline --config pipeline/config/pipeline_config.yaml

# Force rebuild
python -m pipeline.run_pipeline --config pipeline/config/pipeline_config.yaml --force

# Use legacy config (auto-detected)
python -m pipeline.run_pipeline --config pipeline/config/ingrnorm.yaml
```

### Python API

```python
from pipeline.config import UnifiedConfig
from pipeline.task_orchestrator import TaskBasedOrchestrator
from pipeline.common.logging_setup import setup_logging

# Load unified config
config = UnifiedConfig.from_yaml("pipeline/config/pipeline_config.yaml")

# Setup logging
setup_logging(config.to_dict())

# Create orchestrator
orchestrator = TaskBasedOrchestrator(config)
orchestrator.setup_steps()

# Execute
orchestrator.run(force=False)
```

## Column Flow Example

For a task with `target_column: "NER"` and `output_column: "NER_clean"`:

1. **Input**: DataFrame with column `NER`
2. **Normalization Step**: 
   - Reads: `NER`
   - Writes: `NER_clean` (new column, `NER` preserved)
3. **Deduplication Step**:
   - Reads: `NER_clean`
   - Modifies: `NER_clean` in place (preserves `NER`)
4. **Encoding Step**:
   - Reads: `NER_clean`
   - Writes: Encoded format

**Result**: DataFrame contains both `NER` (original) and `NER_clean` (processed)

## Migration from Legacy Config

The pipeline automatically detects config type:
- **Task-based**: Config contains `tasks` key → Uses `TaskBasedOrchestrator`
- **Legacy**: Config contains `pipeline`/`data`/`output` → Uses `UnifiedPipelineOrchestrator`

Both formats are supported for backward compatibility.

## Benefits

1. **Flexibility**: Different parameters for different columns
2. **Clarity**: Each task is self-contained with clear input/output
3. **Maintainability**: Easy to add new tasks or modify existing ones
4. **Column Safety**: Original columns preserved by default
5. **Reusability**: Same step types can be used with different parameters

