"""Apply the trained ingredient NER model to a dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from recipe_pipeline.ingredient_ner.config import DATA, OUT, load_inference_configs_from_dict
from recipe_pipeline.ingredient_ner.inference import run_full_inference_from_config
from recipe_pipeline.ingredient_ner.utils import configure_device

from ..core import PipelineContext, StageResult
from ..utils import stage_logger


INFERRED_COL = "inferred_ingredients"
ENCODED_COL = "encoded_ingredients"


def _load_full_dataset(path: Path, *, logger: logging.Logger) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        logger.debug("Reading parquet dataset: %s", path)
        return pd.read_parquet(path)
    logger.debug("Reading CSV dataset: %s", path)
    return pd.read_csv(path)


def _write_combined_dataset(
    source_path: Path,
    combined_path: Path,
    df_wide: pd.DataFrame,
    *,
    logger: logging.Logger,
) -> str:
    if "NER_clean" not in df_wide.columns:
        raise KeyError("NER_clean column missing from inference output; cannot map results back to dataset")

    logger.info("Loading source dataset to attach inference columns")
    df_source = _load_full_dataset(source_path, logger=logger)
    source_rows = len(df_source)
    infer_rows = len(df_wide)

    if source_rows != infer_rows:
        logger.warning(
            "Row count mismatch between source (%s) and inference results (%s). Using min rows for alignment.",
            source_rows,
            infer_rows,
        )

    rows_to_assign = min(source_rows, infer_rows)
    if rows_to_assign == 0:
        raise ValueError("No rows available to merge inference outputs into dataset")

    inferred_lists = df_wide["NER_clean"].tolist()
    encoded_lists = df_wide["Ingredients"].tolist() if "Ingredients" in df_wide.columns else None

    df_source = df_source.copy()
    # ensure object dtype to safely hold lists
    df_source[INFERRED_COL] = pd.Series([None] * source_rows, dtype=object)
    df_source[ENCODED_COL] = pd.Series([None] * source_rows, dtype=object)
    df_source.loc[: rows_to_assign - 1, INFERRED_COL] = pd.Series(
        inferred_lists[:rows_to_assign], dtype=object
    )
    if encoded_lists is not None:
        df_source.loc[: rows_to_assign - 1, ENCODED_COL] = pd.Series(
            encoded_lists[:rows_to_assign], dtype=object
        )

    combined_path.parent.mkdir(parents=True, exist_ok=True)
    if combined_path.suffix.lower() == ".parquet":
        df_source.to_parquet(combined_path, index=False, compression="zstd")
    else:
        df_source.to_csv(combined_path, index=False)

    logger.info(
        "Saved combined dataset with inference columns to %s (%s rows mapped)",
        combined_path,
        rows_to_assign,
    )
    return str(combined_path)


def run(
    context: PipelineContext,
    *,
    data_path: Optional[Path] = None,
    text_col: Optional[str] = None,
    out_base: Optional[Path] = None,
    force: bool = False,
) -> StageResult:
    cfg = context.stage("ingredient_ner")
    inference_cfg = cfg.get("inference") or {}
    if not inference_cfg:
        raise KeyError("ingredient_ner.inference section missing from pipeline config.")

    logger = stage_logger(context, "ingredient_ner_inference", force=force)
    load_inference_configs_from_dict(inference_cfg)

    text_col = text_col or inference_cfg.get("inference", {}).get("text_col")
    if not text_col:
        raise ValueError("inference.text_col must be provided in the config or via run() override.")

    outputs_cfg = inference_cfg.get("output", {})
    out_base = Path(out_base or outputs_cfg.get("out_base") or OUT.PRED_OUT)
    combined_dataset_path = outputs_cfg.get("combined_dataset")
    combined_dataset = Path(combined_dataset_path) if combined_dataset_path else None

    if data_path:
        input_path = Path(data_path)
    elif DATA.TRAIN_PATH:
        input_path = DATA.TRAIN_PATH
        logger.info("Using inference input from config: %s", input_path)
    else:
        raise ValueError("Provide data_path or set data.input_path in the inference config.")

    if not input_path.exists():
        raise FileNotFoundError(f"Input data not found: {input_path}")

    inference_settings = inference_cfg.get("inference", {})
    batch_size = int(inference_settings.get("batch_size", 256))
    n_process = int(inference_settings.get("n_process", 1))
    use_spacy_normalizer = bool(inference_settings.get("use_spacy_normalizer", True))
    spacy_model = inference_settings.get("spacy_model", "en_core_web_sm")
    use_gpu = bool(inference_settings.get("use_gpu", False))

    sample_n = inference_settings.get("sample_n")
    sample_frac = inference_settings.get("sample_frac")
    head_n = inference_settings.get("head_n")

    sampling_count = sum(opt is not None for opt in (sample_n, sample_frac, head_n))
    if sampling_count > 1:
        raise ValueError("Only one sampling option (sample_n, sample_frac, head_n) can be used at a time.")

    if use_gpu:
        configure_device()
    else:
        logger.info("Using CPU for inference (set inference.use_gpu to true to attempt GPU acceleration).")

    if not OUT.MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Model directory not found: {OUT.MODEL_DIR}. "
            "Train a model first using the ingredient_ner_train stage."
        )

    logger.info("Starting inference on %s", input_path)
    logger.info("Text column: %s", text_col)
    logger.info("Output base: %s", out_base)
    logger.info("Batch size: %s  | Processes: %s", batch_size, n_process)

    try:
        # df_wide, df_tall = run_full_inference_from_config(
        #     text_col=text_col,
        #     out_base=out_base,
        #     data_path=input_path,
        #     sample_n=sample_n,
        #     sample_frac=sample_frac,
        #     head_n=head_n,
        #     batch_size=batch_size,
        #     n_process=n_process,
        #     use_spacy_normalizer=use_spacy_normalizer,
        #     spacy_model=spacy_model,
        # )
        wide_path = Path(out_base).with_name(Path(out_base).stem + "_wide.parquet")
        tall_path = Path(out_base).with_name(Path(out_base).stem + "_tall.parquet")

        # Load wide_path
        df_wide = pd.read_parquet(wide_path)
        df_tall = pd.read_parquet(tall_path)
        # logger.info("Inference complete. Processed %s rows / %s entities.", len(df_wide), len(df_tall))
        # logger.info("Wide output: %s", wide_path)
        # logger.info("Tall output: %s", tall_path)

        combined_written: Optional[str] = None
        if combined_dataset:
            if sampling_count > 0:
                logger.warning(
                    "Skipping combined dataset write because sampling options were used (sample_n/sample_frac/head_n)."
                )
            else:
                combined_written = _write_combined_dataset(
                    input_path,
                    combined_dataset,
                    df_wide,
                    logger=logger,
                )

        return StageResult(
            name="ingredient_ner_infer",
            status="success",
            outputs={
                "wide_path": str(wide_path),
                "tall_path": str(tall_path),
                "rows": len(df_wide),
                "entities": len(df_tall),
                "combined_dataset": combined_written,
            },
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("Inference failed: %s", exc)
        return StageResult(name="ingredient_ner_infer", status="failed", details=str(exc))
