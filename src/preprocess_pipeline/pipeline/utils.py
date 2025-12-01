from __future__ import annotations

import logging
from typing import Optional

from preprocess_pipeline.common.logging_setup import setup_logging
from .core import PipelineContext


def stage_logger(context: PipelineContext, stage_name: str, *, force: bool = False) -> logging.Logger:
    """
    Configure logging for a stage and return a namespaced logger.
    """
    setup_logging(context.logging(stage_name), force=force)
    return logging.getLogger(f"preprocess_pipeline.{stage_name}")


def bool_from_cfg(value: Optional[bool], default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)

