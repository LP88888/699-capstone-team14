# pipeline/scripts/ingredient_ner_train.py
from __future__ import annotations

from pathlib import Path
import argparse
import sys as _sys

# Make `pipeline/` importable when running from repo root
_SYS_PATH_ROOT = Path.cwd() / "pipeline"
if str(_SYS_PATH_ROOT) not in _sys.path:
    _sys.path.append(str(_SYS_PATH_ROOT))

import yaml

from common.logging_setup import setup_logging
from ingredient_ner.config import load_configs_from_dict, DATA, TRAIN, OUT
from ingredient_ner.utils import set_global_seed
from ingredient_ner.data_prep import prepare_docbins_from_config
from ingredient_ner.training import train_ner_from_docbins


def main() -> None:
    ap = argparse.ArgumentParser(description="Train ingredient NER model")
    ap.add_argument(
        "--config",
        type=str,
        default="pipeline/config/ingredient_ner.yaml",
        help="Path to ingredient NER config YAML",
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

    # Populate DATA / TRAIN / OUT globals for ingredient_ner modules
    load_configs_from_dict(cfg)

    # Seed + run training pipeline
    set_global_seed(TRAIN.RANDOM_SEED)
    prepare_docbins_from_config()
    train_ner_from_docbins()


if __name__ == "__main__":
    main()
