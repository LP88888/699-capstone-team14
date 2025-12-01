from __future__ import annotations

import shutil

from ingredient_ner.config import load_configs_from_dict, TRAIN, OUT
from ingredient_ner.utils import set_global_seed
from ingredient_ner.data_prep import prepare_docbins_from_config
from ingredient_ner.training import train_ner_from_docbins

from ..core import PipelineContext, StageResult
from ..utils import stage_logger


def run(context: PipelineContext, *, force: bool = False) -> StageResult:
    cfg = context.stage("ingredient_ner")
    training_cfg = cfg.get("training") or cfg
    if not training_cfg:
        raise KeyError("ingredient_ner training config missing under pipeline.ingredient_ner.training")

    logger = stage_logger(context, "ingredient_ner", force=force)

    try:
        load_configs_from_dict(training_cfg)
        set_global_seed(TRAIN.RANDOM_SEED)

        logger.info("Step 1: Preparing Data (Streaming Shards)...")
        prepare_docbins_from_config()

        logger.info("Step 2: Training Model...")
        train_ner_from_docbins()

        cleanup_docbins = training_cfg.get("ner", {}).get("cleanup_docbins", True)
        if cleanup_docbins:
            logger.info("Step 3: Cleanup...")
            for docbin_dir in [OUT.TRAIN_DIR, OUT.VALID_DIR]:
                if docbin_dir.exists():
                    try:
                        count = len(list(docbin_dir.glob("*.spacy")))
                        shutil.rmtree(docbin_dir)
                        logger.info("  - Removed %s (%s shards)", docbin_dir, count)
                    except Exception as exc:
                        logger.warning("  - Failed to remove %s: %s", docbin_dir, exc)
        else:
            logger.info("Skipping cleanup (configured to keep intermediate files).")

        logger.info("Done. Model saved to: %s", OUT.MODEL_DIR)
        return StageResult(
            name="ingredient_ner_train",
            status="success",
            outputs={"model_dir": str(OUT.MODEL_DIR)},
        )
    except Exception as exc:  # pragma: no cover - pipeline runner logs
        logger.exception("Ingredient NER training failed: %s", exc)
        return StageResult(name="ingredient_ner_train", status="failed", details=str(exc))