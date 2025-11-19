# pipeline/scripts/apply_ingredient_ner.py
"""
Apply trained ingredient NER model to a new dataset.

This script:
1. Loads a trained spaCy NER model
2. Runs NER on a text column to extract ingredient entities
3. Normalizes and canonicalizes ingredients using dedupe map
4. Optionally maps to ingredient IDs using encoder maps
5. Outputs wide and tall format Parquet files

Example usage:
    python pipeline/scripts/apply_ingredient_ner.py \
        --config pipeline/config/ingredient_ner.yaml \
        --in-path data/new_dataset.parquet \
        --text-col ingredients_raw \
        --out-base data/new_dataset_ner
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys as _sys
import logging

# Make `pipeline/` importable when running from repo root
_SYS_PATH_ROOT = Path.cwd() / "pipeline"
if str(_SYS_PATH_ROOT) not in _sys.path:
    _sys.path.append(str(_SYS_PATH_ROOT))

import yaml

from common.logging_setup import setup_logging
from ingredient_ner.config import load_inference_configs_from_dict, DATA, OUT
from ingredient_ner.inference import run_full_inference_from_config
from ingredient_ner.utils import configure_device

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Apply trained ingredient NER model to a new dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with config defaults
  python pipeline/scripts/apply_ingredient_ner.py --text-col ingredients_raw --out-base data/output_ner

  # Override input path and sample
  python pipeline/scripts/apply_ingredient_ner.py \\
      --in-path data/new_dataset.parquet \\
      --text-col ingredients_raw \\
      --out-base data/new_dataset_ner \\
      --sample-n 1000

  # Process first 100 rows
  python pipeline/scripts/apply_ingredient_ner.py \\
      --text-col ingredients_raw \\
      --out-base data/output_ner \\
      --head-n 100
        """,
    )
    
    ap.add_argument(
        "--config",
        type=str,
        default="pipeline/config/ingredient_ner_inference.yaml",
        help="Path to ingredient NER inference config YAML",
    )
    ap.add_argument(
        "--in-path",
        type=str,
        default=None,
        help="Input dataset path (CSV or Parquet). Overrides config data.input_path.",
    )
    ap.add_argument(
        "--text-col",
        type=str,
        default=None,
        help="Column name containing raw ingredient text to process. Overrides config inference.text_col.",
    )
    ap.add_argument(
        "--out-base",
        type=str,
        default=None,
        help="Base path for output files (will write <base>_wide.parquet and <base>_tall.parquet). Overrides config output.out_base.",
    )
    
    # Sampling options (mutually exclusive)
    sampling_group = ap.add_mutually_exclusive_group()
    sampling_group.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="Randomly sample N rows",
    )
    sampling_group.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Randomly sample fraction of rows (0.0-1.0)",
    )
    sampling_group.add_argument(
        "--head-n",
        type=int,
        default=None,
        help="Take first N rows",
    )
    
    # Performance options
    ap.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for spaCy processing (default: 256)",
    )
    ap.add_argument(
        "--n-process",
        type=int,
        default=1,
        help="Number of processes for spaCy (default: 1). "
             "Note: >1 may not work on Windows with transformers. Use at your own risk.",
    )
    ap.add_argument(
        "--use-gpu",
        action="store_true",
        help="Attempt to use GPU (default: CPU). May not work on Windows.",
    )
    
    args = ap.parse_args()

    # Load config first to get defaults
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML at {config_path} did not parse to a dict.")

    # Setup logging
    setup_logging(cfg)
    logger.info("=" * 60)
    logger.info("Applying Ingredient NER Model")
    logger.info("=" * 60)

    # Populate DATA / OUT globals from inference config
    load_inference_configs_from_dict(cfg)
    
    # Get inference settings from config
    inference_cfg = cfg.get("inference") or {}
    
    # Determine text column (CLI overrides config)
    text_col = args.text_col or inference_cfg.get("text_col")
    if not text_col:
        raise ValueError(
            "text_col must be specified either via --text-col argument "
            "or in config as inference.text_col"
        )
    
    # Determine output base (CLI overrides config)
    out_base = args.out_base or str(OUT.PRED_OUT)
    
    # Determine input path (CLI overrides config)
    if args.in_path:
        in_path = Path(args.in_path)
    elif DATA.TRAIN_PATH:
        in_path = DATA.TRAIN_PATH
        logger.info(f"Using input path from config: {in_path}")
    else:
        raise ValueError(
            "Input path must be specified either via --in-path argument "
            "or in config as data.input_path"
        )
    
    if not in_path.exists():
        raise FileNotFoundError(f"Input data not found: {in_path}")
    
    # Get performance settings (CLI overrides config)
    # Use CLI args if provided, otherwise use config values
    batch_size = args.batch_size if args.batch_size != 256 else inference_cfg.get("batch_size", 256)
    n_process = args.n_process if args.n_process != 1 else inference_cfg.get("n_process", 1)
    use_gpu = args.use_gpu or inference_cfg.get("use_gpu", False)
    
    # Get sampling settings (CLI overrides config)
    sample_n = args.sample_n or inference_cfg.get("sample_n")
    sample_frac = args.sample_frac or inference_cfg.get("sample_frac")
    head_n = args.head_n or inference_cfg.get("head_n")
    
    # Configure device (CPU by default, GPU if requested)
    if use_gpu:
        configure_device()
    else:
        import spacy
        spacy.prefer_cpu()
        logger.info("Using CPU (set inference.use_gpu: true in config or use --use-gpu to attempt GPU acceleration)")
    
    logger.info(f"Input: {in_path}")
    logger.info(f"Text column: {text_col}")
    logger.info(f"Output base: {out_base}")
    logger.info(f"Model: {OUT.MODEL_DIR}")
    logger.info(f"Batch size: {batch_size}, Processes: {n_process}")
    
    if not OUT.MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {OUT.MODEL_DIR}. "
            f"Train a model first using: python pipeline/scripts/run_ingredient_ner.py"
        )

    # Validate sampling options
    sampling_count = sum([
        sample_n is not None,
        sample_frac is not None,
        head_n is not None,
    ])
    if sampling_count > 1:
        raise ValueError("Only one sampling option (--sample-n, --sample-frac, --head-n) can be used at a time")

    # Run inference
    logger.info("Starting inference...")
    try:
        df_wide, df_tall = run_full_inference_from_config(
            text_col=text_col,
            out_base=Path(out_base),
            data_path=in_path,
            sample_n=sample_n,
            sample_frac=sample_frac,
            head_n=head_n,
            batch_size=batch_size,
            n_process=n_process,
        )
        
        logger.info("=" * 60)
        logger.info("Inference Complete")
        logger.info("=" * 60)
        logger.info(f"Processed {len(df_wide)} rows")
        logger.info(f"Extracted {len(df_tall)} ingredient entities")
        logger.info(f"Output files:")
        logger.info(f"  - {Path(out_base).with_name(Path(out_base).stem + '_wide.parquet')}")
        logger.info(f"  - {Path(out_base).with_name(Path(out_base).stem + '_tall.parquet')}")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

