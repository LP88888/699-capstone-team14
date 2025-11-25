# cuisine_classifier/config.py
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import yaml

# Public globals other modules import
DATA = SimpleNamespace()
TRAIN = SimpleNamespace()
OUT = SimpleNamespace()


def _to_path(v: Any) -> Path | None:
    if v is None:
        return None
    return Path(str(v))


def load_full_yaml(path: str | Path) -> Dict[str, Any]:
    """Load the full pipeline YAML."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"YAML at {path} did not parse to a dict.")
    return cfg


def _build_data_ns(cfg: Dict[str, Any]) -> SimpleNamespace:
    data_cfg = cfg.get("data") or {}
    classifier_cfg = cfg.get("cuisine_classifier") or {}
    out_cfg = cfg.get("output") or {}

    # 1) training file path
    # Prefer cuisine_classifier.train_path, fallback to data.input_path, then default
    train_path_str = classifier_cfg.get("train_path")
    print(train_path_str)
    if not train_path_str:
        train_path_str = data_cfg.get("input_path")
    if not train_path_str:
        print(data_cfg)
        train_path_str = "./data/combined_raw_datasets_with_cuisine_encoded.parquet"
    train_path = Path(train_path_str)

    # 2) parquet vs csv
    if "data_is_parquet" in classifier_cfg:
        data_is_parquet = bool(classifier_cfg["data_is_parquet"])
    else:
        data_is_parquet = train_path.suffix.lower() == ".parquet"

    # 3) max rows
    max_rows_raw = classifier_cfg.get("max_rows")
    max_rows = int(max_rows_raw) if max_rows_raw is not None else None

    # 4) Column names
    text_col = classifier_cfg.get("text_col") or data_cfg.get("text_col") or "ingredients"
    cuisine_col = classifier_cfg.get("cuisine_col") or data_cfg.get("cuisine_col") or "cuisine"

    return SimpleNamespace(
        DATA_IS_PARQUET=data_is_parquet,
        TRAIN_PATH=train_path,
        MAX_ROWS=max_rows,
        TEXT_COL=text_col,
        CUISINE_COL=cuisine_col,
    )


def _build_train_ns(cfg: Dict[str, Any]) -> SimpleNamespace:
    """Build TRAIN namespace from cuisine_classifier config. Uses UPPERCASE keys to match existing code."""
    classifier_cfg = cfg.get("cuisine_classifier") or {}

    # Debug mode: if use_tok2vec_debug is True, we'll use tok2vec instead of transformers
    use_tok2vec_debug = bool(classifier_cfg.get("use_tok2vec_debug", False))
    
    # If debug mode, reduce epochs for faster iteration
    n_epochs = int(classifier_cfg.get("n_epochs", 20))
    if use_tok2vec_debug:
        n_epochs = min(n_epochs, 5)  # Cap at 5 epochs in debug mode

    return SimpleNamespace(
        RANDOM_SEED=int(classifier_cfg.get("random_seed", 42)),
        VALID_FRACTION=float(classifier_cfg.get("valid_fraction", 0.2)),
        SHARD_SIZE=int(classifier_cfg.get("shard_size", 2000)),
        BATCH_SIZE=int(classifier_cfg.get("batch_size", 256)),
        TRANSFORMER_MODEL=str(classifier_cfg.get("transformer_model", "distilbert-base-uncased")),
        WINDOW=int(classifier_cfg.get("window", 64)),
        STRIDE=int(classifier_cfg.get("stride", 48)),
        LR=float(classifier_cfg.get("lr", 5e-5)),
        DROPOUT=float(classifier_cfg.get("dropout", 0.1)),
        N_EPOCHS=n_epochs,
        FREEZE_LAYERS=int(classifier_cfg.get("freeze_layers", 2)),
        USE_AMP=bool(classifier_cfg.get("use_amp", True)),
        EARLY_STOPPING_PATIENCE=int(classifier_cfg.get("early_stopping_patience", 3)),
        EVAL_SNAPSHOT_MAX=int(classifier_cfg.get("eval_snapshot_max", 1500)),
        CLEAR_CACHE_EVERY=int(classifier_cfg.get("clear_cache_every", 200)),
        USE_TOK2VEC_DEBUG=use_tok2vec_debug,
        MAX_TRAIN_DOCS=int(classifier_cfg.get("max_train_docs")) if classifier_cfg.get("max_train_docs") is not None else None,
    )


def _build_out_ns(cfg: Dict[str, Any]) -> SimpleNamespace:
    classifier_cfg = cfg.get("cuisine_classifier") or {}
    out_cfg = cfg.get("output") or {}

    out_dir = _to_path(classifier_cfg.get("out_dir")) or Path("./models/cuisine_classifier_trf")
    model_dir = _to_path(classifier_cfg.get("model_dir")) or (out_dir / "model-best")

    boot_dir = out_dir / "bootstrapped"
    train_dir = boot_dir / "train"
    valid_dir = boot_dir / "valid"

    pred_out = _to_path(out_cfg.get("preds_base")) or Path("./data/training/cuisine_predictions.parquet")

    return SimpleNamespace(
        OUT_DIR=out_dir,
        MODEL_DIR=model_dir,
        BOOT_DIR=boot_dir,
        TRAIN_DIR=train_dir,
        VALID_DIR=valid_dir,
        PRED_OUT=pred_out,
    )


def _update_ns(target: SimpleNamespace, src: SimpleNamespace) -> None:
    """Mutate an existing SimpleNamespace to have src's attributes."""
    target.__dict__.clear()
    target.__dict__.update(src.__dict__)


# ---------------------- public API ---------------------- #

def load_configs_from_dict(cfg: Dict[str, Any]) -> Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    """
    Populate DATA / TRAIN / OUT from an in-memory YAML dict.

    IMPORTANT: we mutate the existing SimpleNamespace objects so
    any 'from cuisine_classifier.config import TRAIN' references see updates.
    """
    global DATA, TRAIN, OUT
    _update_ns(DATA, _build_data_ns(cfg))
    _update_ns(TRAIN, _build_train_ns(cfg))
    _update_ns(OUT, _build_out_ns(cfg))
    return DATA, TRAIN, OUT


def load_configs_from_yaml(path: str | Path) -> Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    """Convenience that reads YAML then delegates to load_configs_from_dict."""
    cfg = load_full_yaml(path)
    return load_configs_from_dict(cfg)


def print_configs() -> None:
    from pprint import pprint
    print("DATA:")
    pprint(vars(DATA))
    print("\nTRAIN:")
    pprint(vars(TRAIN))
    print("\nOUT:")
    pprint(vars(OUT))

