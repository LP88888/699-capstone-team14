# Pipeline Refactoring Summary

## Overview

The preprocessing pipeline has been refactored from disconnected scripts into a cohesive, object-oriented architecture using:
- **Abstract Base Class**: `PipelineStep` for common I/O, logging, and timing
- **Pydantic Configuration**: Unified, validated configuration models
- **Concrete Step Classes**: Reusable transformation steps
- **Orchestrators**: Main entry points that run steps in sequence

## Architecture

### Base Classes

**`pipeline/base.py`**
- `PipelineStep`: Abstract base class handling:
  - Parquet file I/O (streaming by row groups)
  - Logging setup
  - Execution time measurement
  - Abstract `transform()` method for subclasses

### Configuration

**`pipeline/config.py`**
- `PipelineConfig`: Main configuration model with Pydantic validation
- Sub-configs: `DataConfig`, `OutputConfig`, `StagesConfig`, `SBERTConfig`, `W2VConfig`, `EncoderConfig`, `LoggingConfig`, `CleanupConfig`
- Methods: `from_yaml()`, `from_dict()`, `to_dict()`, `get_step_config()`

### Step Classes

**`pipeline/steps/`**
- `IngredientNormalizerStep`: spaCy-based ingredient normalization
- `DeduplicationStep`: SBERT/Word2Vec deduplication
- `EncodingStep`: Token-to-ID encoding
- `CuisinePreprocessingStep`: Splits multi-cuisine entries into lists

### Orchestrators

**`pipeline/run_pipeline.py`**
- `PipelineOrchestrator`: Runs ingredient normalization pipeline
- Loads config, instantiates steps, executes in sequence

**`pipeline/run_cuisine_pipeline.py`**
- `CuisinePipelineOrchestrator`: Runs cuisine normalization pipeline
- Similar structure but for cuisine normalization

## Usage

### Ingredient Normalization

```bash
# Run ingredient normalization pipeline
python -m pipeline.run_pipeline --config pipeline/config/ingrnorm.yaml

# Force rebuild all artifacts
python -m pipeline.run_pipeline --config pipeline/config/ingrnorm.yaml --force
```

### Cuisine Normalization

```bash
# Run cuisine normalization pipeline
python -m pipeline.run_cuisine_pipeline --config pipeline/config/cuisnorm.yaml

# Force rebuild all artifacts
python -m pipeline.run_cuisine_pipeline --config pipeline/config/cuisnorm.yaml --force
```

## Script Coverage

### ✅ Covered by New Pipeline Structure

1. **`run_ingrnorm.py`** → **`run_pipeline.py`**
   - Ingredient normalization (spaCy)
   - Deduplication (SBERT/W2V)
   - Encoding to IDs
   - All stages integrated into `PipelineOrchestrator`

2. **`run_cuisine_norm.py`** → **`run_cuisine_pipeline.py`**
   - Cuisine preprocessing (split multi-cuisine entries)
   - Cuisine normalization (spaCy)
   - Cuisine deduplication (SBERT/W2V)
   - Cuisine encoding to IDs
   - All stages integrated into `CuisinePipelineOrchestrator`

### ❌ Not Covered (Separate Preprocessing Steps)

These scripts handle preprocessing steps that occur **before** normalization and are not part of the normalization pipeline:

1. **`combine_raw_datasets.py`**
   - Combines multiple raw CSV files
   - Extracts ingredients/cuisine columns
   - Runs NER inference
   - Outputs combined dataset
   - **Status**: Standalone script, runs before normalization pipeline

2. **`apply_ingredient_ner.py`**
   - Applies trained NER model to extract ingredients
   - Normalizes and canonicalizes ingredients
   - Maps to ingredient IDs
   - **Status**: Standalone script, can be used independently

3. **`run_ingredient_ner.py`**
   - Trains ingredient NER model
   - **Status**: Training script, separate from normalization pipeline

4. **`encode_ingredients.py`**
   - Standalone encoding script
   - **Status**: Functionality now in `EncodingStep`, but script remains for backward compatibility

## Pipeline Flow

### Ingredient Normalization Pipeline

```
Input (Parquet with NER column)
  ↓
[IngredientNormalizerStep] - spaCy normalization
  ↓
[DeduplicationStep] - SBERT/W2V deduplication
  ↓
[EncodingStep] - Token-to-ID encoding
  ↓
Output (Encoded ingredients with IDs)
```

### Cuisine Normalization Pipeline

```
Input (Parquet with cuisine column)
  ↓
[CuisinePreprocessingStep] - Split multi-cuisine entries
  ↓
[IngredientNormalizerStep] - spaCy normalization (reused)
  ↓
[DeduplicationStep] - SBERT/W2V deduplication (reused)
  ↓
[EncodingStep] - Token-to-ID encoding (reused)
  ↓
Output (Encoded cuisines with IDs)
```

## Benefits

1. **Eliminates Boilerplate**: Common I/O, logging, timing handled by base class
2. **Unified Configuration**: Single Pydantic model with validation
3. **Extensibility**: Easy to add new steps by inheriting `PipelineStep`
4. **Maintainability**: Clear separation of concerns, reusable components
5. **Backward Compatibility**: Existing transformation logic preserved

## Migration Notes

- Old scripts (`run_ingrnorm.py`, `run_cuisine_norm.py`) still exist for reference
- New orchestrators (`run_pipeline.py`, `run_cuisine_pipeline.py`) are the recommended entry points
- Configuration format remains the same (YAML), but now validated with Pydantic
- All step classes can be used independently or as part of the orchestrator

