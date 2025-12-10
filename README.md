## Repository Structure

```
.
├── data/
│   ├── combined_raw_datasets_with_inference_encoded_with_cuisine_encoded.parquet
├── public/
│   └── fusion/
│       └── fusion_African_American.json
│       └── ...
│   ├── cuisine_network.html
│   ├── index.html
│   ├── ingredient_network.html
├── src/
│   └── networks/
│       └── phase1/
│       └── phase2/
│   └── recipe_pipeline/
│       ├── analysis/
│       ├── common/
│       ├── config/          # All YAML lives here (see pipeline.yaml)
│       ├── cuisine_classifier/
│       ├── ingredient_ner/
│       ├── ingrnorm/
│       ├── notebooks/
│       │   └── preprocess/
│       │   └── training/
│       ├── preprocessing/
│       └── ...
├── LICENSE
├── README.md
├── requirements.txt
└── ...
```


## Quickstart - do this first

1) Create/activate a virtualenv (or your preferred env).
   For virtualenv, after cloning this repo, in the same directory, run:
   ```sh
   python -m venv venv
   venv\Scripts\Activate
   ```
2) Install the project in editable mode:
   ```sh
   pip install -r requirements.txt
   pip install -e .
   ```
3) Run the pipeline:
   ```sh
   python -m recipe_pipeline
   ```

4. **Add new libraries:**
	 - When introducing new libraries, add them to `requirements.txt`.
	 - To list installed packages:
		 ```sh
		 pip freeze
		 ```
	 - Copy any new packages to `requirements.txt`.

## Unified Preprocess Pipeline

All stage configuration lives in `src/recipe_pipeline/config/pipeline.yaml`.  
The runner wires those settings into the individual stage modules, so you do not have to juggle per‑script YAMLs. Please ensure you configure these setttings according to your environment.

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

## Ingredient normalization & dedupe map

- Canonical ingredient mappings live at `data/ingr_normalized/dedupe_map.jsonl` and are applied via `normalize_token_with_map` in `ingrnorm/dedupe_map.py`.
- Built-in rules: collapse duplicate leading tokens (`salt salt` → `salt`), strip non-food form/measure tokens (`slice/sliced/slices`, `chunk/chunky`, `piece/pieces`, units/counts/numerics), and drop noisy tokens in `_DROP_TOKENS`.
- Multi-word phrases keep the dominant ingredient token(s) using unigram frequencies from map targets after stripping noise (e.g., `chicken slices` → `chicken`, `flour cup` → `flour`).
- After changing the map or rules, regenerate deduped ingredients (e.g., `recipes_data_clean.parquet`) by rerunning the apply-map stage:
  ```sh
  python -m recipe_pipeline.pipeline --stages ingredient_normalization --force
  ```
  Then rerun downstream stages that consume cleaned ingredients (encoding, PMI/graph, recommenders) so artifacts stay in sync.

## GPU / CPU setup

- GPU toggles live in `src/recipe_pipeline/config/pipeline.yaml`:
  - `combine_raw.inference.use_gpu` (ingredient NER inference during ingestion)
  - `ingredient_ner.inference.use_gpu` (second-pass inference)
  Set these to `false` for CPU-only runs and consider lowering batch sizes if memory is tight.
- GPU dependencies in `requirements.txt`: `torch==2.9.1` and `cupy-cuda12x==12.3.0` are CUDA builds. On CPU-only machines, skip those and install a CPU torch wheel instead:
  ```sh
  # install everything except the CUDA libs
  pip install -r requirements.txt --no-deps
  pip install torch==2.9.1+cpu -f https://download.pytorch.org/whl/cpu
  ```
  or remove/comment the `cupy-cuda12x` line and reinstall. Cupy is optional; if you skip it, ensure any code paths using cupy are disabled.
-
