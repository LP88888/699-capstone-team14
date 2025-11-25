# Classifier Pipeline

This pipeline builds a network of recipes based on ingredients and uses it for:
1. **Infusion Recipe Suggestions**: Suggest fusion recipes by combining ingredients from different cuisines
2. **Cuisine Classification**: Predict cuisine type from ingredient lists

## Overview

The pipeline constructs a graph/network where:
- **Nodes**: Ingredients (using ingredient IDs from preprocessing)
- **Edges**: Co-occurrence relationships between ingredients (weighted by frequency)
- **Recipe Nodes**: Optional - recipes can be represented as subgraphs or feature vectors

## Pipeline Structure

```
classifier_pipeline/
├── README.md
├── requirements.txt
├── config/
│   └── network_config.yaml      # Network construction parameters
├── network/
│   ├── __init__.py
│   ├── builder.py               # Build ingredient co-occurrence network
│   ├── graph.py                 # Graph data structures and utilities
│   └── analysis.py              # Network analysis functions
├── inference/
│   ├── __init__.py
│   ├── infusion_suggestor.py    # Suggest fusion recipes
│   └── cuisine_classifier.py    # Classify cuisine from ingredients
├── data/
│   └── (network artifacts will be saved here)
└── notebooks/
    ├── 01_build_network.ipynb
    ├── 02_infusion_suggestions.ipynb
    └── 03_cuisine_classification.ipynb
```

## Input Data

The pipeline expects the final output from preprocessing:
- **File**: `preprocess_pipeline/data/encoded_combined_datasets_with_cuisine_encoded.parquet`
- **Required Columns**:
  - `encoded_ingredients`: List of ingredient IDs per recipe
  - `cuisine_encoded`: List of cuisine IDs per recipe
  - `Dataset_ID`, `index`: Recipe identifiers

## Workflow

1. **Network Construction** (`01_build_network.ipynb`)
   - Load encoded recipe data
   - Build ingredient co-occurrence graph
   - Compute edge weights (co-occurrence frequencies)
   - Save network artifacts

2. **Infusion Recipe Suggestions** (`02_infusion_suggestions.ipynb`)
   - Use network to find ingredient bridges between cuisines
   - Suggest fusion recipes combining ingredients from different cuisines
   - Rank suggestions by network connectivity

3. **Cuisine Classification** (`03_cuisine_classification.ipynb`)
   - Extract ingredient features from recipes
   - Use network-based features for classification
   - Train/validate classifier model

