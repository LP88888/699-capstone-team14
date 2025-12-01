"""Apply the trained ingredient NER model to a dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.preprocess_pipeline.ingredient_ner.config import load_inference_configs_from_dict, DATA, OUT
from src.preprocess_pipeline.ingredient_ner.inference import run_full_inference_from_config
from src.preprocess_pipeline.ingredient_ner.utils import configure_device

from ..core import PipelineContext, StageResult
from ..utils import stage_logger


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

    out_base = Path(out_base or inference_cfg.get("output", {}).get("out_base") or OUT.PRED_OUT)

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
        df_wide, df_tall = run_full_inference_from_config(
            text_col=text_col,
            out_base=out_base,
            data_path=input_path,
            sample_n=sample_n,
            sample_frac=sample_frac,
            head_n=head_n,
            batch_size=batch_size,
            n_process=n_process,
            use_spacy_normalizer=use_spacy_normalizer,
            spacy_model=spacy_model,
        )
        wide_path = Path(out_base).with_name(Path(out_base).stem + "_wide.parquet")
        tall_path = Path(out_base).with_name(Path(out_base).stem + "_tall.parquet")

        logger.info("Inference complete. Processed %s rows / %s entities.", len(df_wide), len(df_tall))
        logger.info("Wide output: %s", wide_path)
        logger.info("Tall output: %s", tall_path)

        return StageResult(
            name="ingredient_ner_infer",
            status="success",
            outputs={
                "wide_path": str(wide_path),
                "tall_path": str(tall_path),
                "rows": len(df_wide),
                "entities": len(df_tall),
            },
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("Inference failed: %s", exc)
        return StageResult(name="ingredient_ner_infer", status="failed", details=str(exc))

