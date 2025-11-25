# pipeline/ingredient_ner/config.py
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
    ner_cfg = cfg.get("ner") or {}
    out_cfg = cfg.get("output") or {}

    # 1) training file path
    # Prefer ner.train_path, fallback to data.input_path, then default
    train_path_str = ner_cfg.get("train_path")
    if not train_path_str:
        train_path_str = data_cfg.get("input_path")
    if not train_path_str:
        train_path_str = "./data/raw/wilmerarltstrmberg_data.csv"
    train_path = Path(train_path_str)

    # 2) parquet vs csv
    if "data_is_parquet" in ner_cfg:
        data_is_parquet = bool(ner_cfg["data_is_parquet"])
    else:
        data_is_parquet = train_path.suffix.lower() == ".parquet"

    # 3) max rows
    max_rows_raw = ner_cfg.get("max_rows")
    max_rows = int(max_rows_raw) if max_rows_raw is not None else None

    # 4) NER columns
    ner_list_col = ner_cfg.get("ner_list_col") or data_cfg.get("ner_col") or None
    text_col = ner_cfg.get("text_col") or None
    lexicon_json = _to_path(ner_cfg.get("lexicon_json")) if ner_cfg.get("lexicon_json") else None

    # 5) other artifacts
    dedupe_jsonl = _to_path(out_cfg.get("cosine_map_path"))
    ing_id2tok = _to_path(out_cfg.get("ingredient_id_to_token"))
    ing_tok2id = _to_path(out_cfg.get("ingredient_token_to_id"))

    return SimpleNamespace(
        DATA_IS_PARQUET=data_is_parquet,
        TRAIN_PATH=train_path,
        MAX_ROWS=max_rows,
        NER_LIST_COL=ner_list_col,
        TEXT_COL=text_col,
        LEXICON_JSON=lexicon_json,
        DEDUPE_JSONL=dedupe_jsonl,
        ING_ID2TOK_JSON=ing_id2tok,
        ING_TOK2ID_JSON=ing_tok2id,
    )


def _build_train_ns(cfg: Dict[str, Any]) -> SimpleNamespace:
    """Build TRAIN namespace from ner config. Uses UPPERCASE keys to match existing code."""
    ner_cfg = cfg.get("ner") or {}

    # Debug mode: if use_tok2vec_debug is True, we'll use tok2vec instead of transformers
    use_tok2vec_debug = bool(ner_cfg.get("use_tok2vec_debug", False))
    
    # If debug mode, reduce epochs for faster iteration
    n_epochs = int(ner_cfg.get("n_epochs", 20))
    if use_tok2vec_debug:
        n_epochs = min(n_epochs, 5)  # Cap at 5 epochs in debug mode

    return SimpleNamespace(
        RANDOM_SEED=int(ner_cfg.get("random_seed", 42)),
        VALID_FRACTION=float(ner_cfg.get("valid_fraction", 0.2)),
        SHARD_SIZE=int(ner_cfg.get("shard_size", 2000)),
        BATCH_SIZE=int(ner_cfg.get("batch_size", 256)),
        TRANSFORMER_MODEL=str(ner_cfg.get("transformer_model", "distilbert-base-uncased")),
        WINDOW=int(ner_cfg.get("window", 64)),
        STRIDE=int(ner_cfg.get("stride", 48)),
        LR=float(ner_cfg.get("lr", 5e-5)),
        DROPOUT=float(ner_cfg.get("dropout", 0.1)),
        N_EPOCHS=n_epochs,
        FREEZE_LAYERS=int(ner_cfg.get("freeze_layers", 2)),
        USE_AMP=bool(ner_cfg.get("use_amp", True)),
        EARLY_STOPPING_PATIENCE=int(ner_cfg.get("early_stopping_patience", 3)),
        EVAL_SNAPSHOT_MAX=int(ner_cfg.get("eval_snapshot_max", 1500)),
        CLEAR_CACHE_EVERY=int(ner_cfg.get("clear_cache_every", 200)),
        USE_TOK2VEC_DEBUG=use_tok2vec_debug,
        MAX_TRAIN_DOCS=int(ner_cfg.get("max_train_docs")) if ner_cfg.get("max_train_docs") is not None else None,
    )


def _build_out_ns(cfg: Dict[str, Any]) -> SimpleNamespace:
    ner_cfg = cfg.get("ner") or {}
    out_cfg = cfg.get("output") or {}

    out_dir = _to_path(ner_cfg.get("out_dir")) or Path("./models/ingredient_ner_trf")
    model_dir = _to_path(ner_cfg.get("model_dir")) or (out_dir / "model-best")

    boot_dir = out_dir / "bootstrapped"
    train_dir = boot_dir / "train"
    valid_dir = boot_dir / "valid"

    pred_out = _to_path(out_cfg.get("ner_preds_base")) or Path("./data/training/predictions.parquet")

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
    any 'from ingredient_ner.config import TRAIN' references see updates.
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


# ---------------------- Inference-specific config ---------------------- #

def _build_inference_data_ns(cfg: Dict[str, Any]) -> SimpleNamespace:
    """Build DATA namespace for inference from inference config YAML."""
    data_cfg = cfg.get("data") or {}
    artifacts_cfg = cfg.get("artifacts") or {}
    
    input_path_str = data_cfg.get("input_path")
    input_path = _to_path(input_path_str) if input_path_str else None
    
    # Determine if parquet or CSV
    if "data_is_parquet" in data_cfg:
        data_is_parquet = bool(data_cfg["data_is_parquet"])
    elif input_path:
        data_is_parquet = input_path.suffix.lower() == ".parquet"
    else:
        data_is_parquet = True  # Default to parquet
    
    return SimpleNamespace(
        DATA_IS_PARQUET=data_is_parquet,
        TRAIN_PATH=input_path,  # Reusing TRAIN_PATH name for consistency
        MAX_ROWS=None,
        NER_LIST_COL=None,
        TEXT_COL=None,
        LEXICON_JSON=None,
        DEDUPE_JSONL=_to_path(artifacts_cfg.get("cosine_map_path")),
        ING_ID2TOK_JSON=_to_path(artifacts_cfg.get("ingredient_id_to_token")),
        ING_TOK2ID_JSON=_to_path(artifacts_cfg.get("ingredient_token_to_id")),
    )


def _build_inference_out_ns(cfg: Dict[str, Any]) -> SimpleNamespace:
    """Build OUT namespace for inference from inference config YAML."""
    model_cfg = cfg.get("model") or {}
    out_cfg = cfg.get("output") or {}
    
    model_dir = _to_path(model_cfg.get("model_dir")) or Path("./models/ingredient_ner_trf/model-best")
    out_base = _to_path(out_cfg.get("out_base")) or Path("./data/inference_output")
    
    return SimpleNamespace(
        OUT_DIR=model_dir.parent,  # For consistency
        MODEL_DIR=model_dir,
        BOOT_DIR=None,
        TRAIN_DIR=None,
        VALID_DIR=None,
        PRED_OUT=out_base,
    )


def load_inference_configs_from_dict(cfg: Dict[str, Any]) -> Tuple[SimpleNamespace, SimpleNamespace]:
    """
    Populate DATA / OUT from an inference config dict.
    
    Returns only DATA and OUT (no TRAIN needed for inference).
    """
    global DATA, OUT
    _update_ns(DATA, _build_inference_data_ns(cfg))
    _update_ns(OUT, _build_inference_out_ns(cfg))
    return DATA, OUT


def load_inference_configs_from_yaml(path: str | Path) -> Tuple[SimpleNamespace, SimpleNamespace]:
    """Convenience that reads inference YAML then delegates to load_inference_configs_from_dict."""
    cfg = load_full_yaml(path)
    return load_inference_configs_from_dict(cfg)