from __future__ import annotations

from pathlib import Path
from typing import Optional

from recipe_pipeline.cuisine_classifier.config import load_configs_from_dict, OUT, TRAIN
from recipe_pipeline.cuisine_classifier.utils import set_global_seed
from recipe_pipeline.cuisine_classifier.data_prep import prepare_docbins_from_config
from recipe_pipeline.cuisine_classifier.training import train_classifier_from_docbins

from ..core import PipelineContext, StageResult
from ..utils import stage_logger


def run(
    context: PipelineContext,
    *,
    config_override: Optional[dict] = None,
    force: bool = False,
) -> StageResult:
    cfg = context.stage("cuisine_classifier")
    training_cfg = config_override or cfg.get("training") or cfg
    if not training_cfg:
        raise KeyError("cuisine_classifier.training section missing from pipeline config.")

    logger = stage_logger(context, "cuisine_classifier", force=force)

    try:
        load_configs_from_dict(training_cfg)
        set_global_seed(TRAIN.RANDOM_SEED)
        prepare_docbins_from_config()
        train_classifier_from_docbins()

        cleanup_docbins = training_cfg.get("cuisine_classifier", {}).get("cleanup_docbins", True)
        if cleanup_docbins:
            _cleanup_docbins(logger)
        else:
            logger.info("Docbins preserved (cleanup_docbins=false).")

        logger.info("Training complete. Final model saved at %s", OUT.MODEL_DIR)
        return StageResult(
            name="cuisine_classifier",
            status="success",
            outputs={"model_dir": str(OUT.MODEL_DIR)},
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("Cuisine classifier training failed: %s", exc)
        return StageResult(name="cuisine_classifier", status="failed", details=str(exc))


def _cleanup_docbins(logger) -> None:
    import shutil

    logger.info("=" * 60)
    logger.info("Cleaning up intermediate training artifacts (docbins)")
    logger.info("=" * 60)

    for docbin_dir in [OUT.TRAIN_DIR, OUT.VALID_DIR]:
        if docbin_dir.exists():
            try:
                spacy_files = list(docbin_dir.glob("*.spacy"))
                if spacy_files:
                    logger.info("Removing %s docbin shard(s) from %s", len(spacy_files), docbin_dir)
                    shutil.rmtree(docbin_dir)
                    logger.info("Deleted %s", docbin_dir)
                else:
                    logger.debug("No docbin files found in %s", docbin_dir)
            except Exception as exc:
                logger.warning("Failed to delete %s: %s", docbin_dir, exc)