# preprocess_pipeline/scripts/train_cuisine_classifier.py
from __future__ import annotations

from pathlib import Path
import argparse
import sys as _sys

# Make `preprocess_pipeline/` importable when running from repo root
_SYS_PATH_ROOT = Path.cwd() / "preprocess_pipeline"
if str(_SYS_PATH_ROOT) not in _sys.path:
    _sys.path.append(str(_SYS_PATH_ROOT))

import yaml

from common.logging_setup import setup_logging
from cuisine_classifier.config import load_configs_from_dict, DATA, TRAIN, OUT
from cuisine_classifier.utils import set_global_seed
from cuisine_classifier.data_prep import prepare_docbins_from_config
from cuisine_classifier.training import train_classifier_from_docbins


def main() -> None:
    ap = argparse.ArgumentParser(description="Train cuisine classification model")
    ap.add_argument(
        "--config",
        type=str,
        default="preprocess_pipeline/config/cuisine_classifier.yaml",
        help="Path to cuisine classifier config YAML",
    )
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Load full pipeline config once
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML at {config_path} did not parse to a dict.")

    # Logging first
    setup_logging(cfg)

    # Populate DATA / TRAIN / OUT globals for cuisine_classifier modules
    load_configs_from_dict(cfg)

    # Seed + run training pipeline
    set_global_seed(TRAIN.RANDOM_SEED)
    prepare_docbins_from_config()
    train_classifier_from_docbins()
    
    # Cleanup: Remove intermediate docbins after training (keep only the final model)
    import logging
    import shutil
    logger = logging.getLogger(__name__)
    
    cleanup_docbins = cfg.get("cuisine_classifier", {}).get("cleanup_docbins", True)
    if cleanup_docbins:
        logger.info("=" * 60)
        logger.info("Cleaning up intermediate training artifacts (docbins)")
        logger.info("=" * 60)
        
        docbin_dirs = [OUT.TRAIN_DIR, OUT.VALID_DIR]
        for docbin_dir in docbin_dirs:
            if docbin_dir.exists():
                try:
                    # Count files before deletion
                    spacy_files = list(docbin_dir.glob("*.spacy"))
                    if spacy_files:
                        logger.info(f"Removing {len(spacy_files)} docbin shard(s) from {docbin_dir}")
                        shutil.rmtree(docbin_dir)
                        logger.info(f"Deleted {docbin_dir}")
                    else:
                        logger.debug(f"No docbin files found in {docbin_dir}")
                except Exception as e:
                    logger.warning(f"Failed to delete {docbin_dir}: {e}")
        
        logger.info("Training complete. Final model saved at:")
        logger.info(f"  - {OUT.MODEL_DIR}")
        logger.info("Intermediate docbins have been cleaned up.")
    else:
        logger.info("Training complete. Final model saved at:")
        logger.info(f"  - {OUT.MODEL_DIR}")
        logger.info("Docbins preserved (cleanup_docbins=false in config)")


if __name__ == "__main__":
    main()

