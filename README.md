## Repository Structure

```
.
├── data/
│   ├── encoded/
│   ├── normalized/
│   └── raw/
├── notebooks/
├── src/
│   └── recipe_pipeline/
│       ├── common/
│       ├── config/          # All YAML lives here (see pipeline.yaml)
│       ├── pipeline/
│       │   ├── core.py
│       │   ├── runner.py    # Orchestration entry point
│       │   └── stages/      # Stage implementations
│       └── ...
├── README.md
├── requirements.txt
└── ...
```


## Quickstart

1) Create/activate a virtualenv (or your preferred env).
2) Install the project in editable mode:
   ```sh
   pip install -e .
   ```
3) Run the pipeline:
   ```sh
   python -m recipe_pipeline
   ```


## Environment Variables (.env)

For sensitive information (API keys, database credentials, etc.), create a `.env` file in the root directory.


4. **Add new libraries:**
	 - When introducing new libraries, add them to `requirements.txt`.
	 - To list installed packages:
		 ```sh
		 pip freeze
		 ```
	 - Copy any new packages to `requirements.txt`.

## Unified Preprocess Pipeline

All stage configuration now lives in `src/recipe_pipeline/config/pipeline.yaml`.  
The new runner wires those settings into the individual stage modules, so you no longer have to juggle per‑script YAMLs.

### Discover available stages

```sh
python -m recipe_pipeline.pipeline --list
```

Default execution order (can be overridden):

1. `combine_raw`
2. `ingredient_normalization`
3. `ingredient_ner_train`
4. `ingredient_ner_infer`
5. `ingredient_encoding`
6. `cuisine_normalization`
7. `cuisine_classifier`

### Run the whole pipeline

```sh
python -m recipe_pipeline.pipeline
```

Use `--keep-going` if you want later stages to run even when one fails:

```sh
python -m recipe_pipeline.pipeline --keep-going
```

### Run pipeline via CLI

If you have installed the package and have the `recipe-pipeline` command available, you can run the pipeline directly:

```sh
recipe-pipeline --config recipe_pipeline/config/pipeline.yaml
```

This is equivalent to:

```sh
python -m recipe_pipeline.pipeline --config recipe_pipeline/config/pipeline.yaml
```

### Run specific stages

```sh
# Re-run just ingestion and normalization
python -m recipe_pipeline.pipeline --stages combine_raw ingredient_normalization

# Train the classifiers only
python -m recipe_pipeline.pipeline --stages ingredient_ner_train cuisine_classifier
```

### Use an alternate config or working directory

```sh
python -m recipe_pipeline.pipeline \
  --config my_configs/pipeline.local.yaml \
  --workdir /tmp/preprocess-run
```

### Per-stage overrides

Stage functions accept keyword overrides (see `pipeline/runner.py`). Example: reusing an exported dataset for encoding only.

```py
from pathlib import Path
from recipe_pipeline.runner import PipelineRunner

runner = PipelineRunner.from_file()
runner.run(
    stages=["ingredient_encoding"],
    overrides={
        "ingredient_encoding": {
            "input_path": Path("data/combined_raw_datasets_with_inference.parquet"),
            "output_path": Path("data/combined_raw_datasets_with_cuisine_encoded.parquet"),
        }
    },
)
```

### Logging & artifacts

Logging targets (files + levels) are configured under the `logging` section of `pipeline.yaml`.  
Each stage writes intermediate and final artifacts to the locations defined in the same file (e.g., encoded parquet, dedupe maps, model directories).

> **Tip:** If you want to keep old artifacts for debugging, disable the relevant cleanup flags in `pipeline.yaml` (`cleanup.enabled`, `stages.apply_cosine_map`, etc.).
