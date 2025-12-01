"""Individual pipeline stage implementations."""

from __future__ import annotations

from typing import Callable, Dict

from . import (
    combine_raw,
    cuisine_classifier,
    cuisine_normalization,
    ingredient_encoding,
    ingredient_ner_infer,
    ingredient_ner_train,
    ingredient_normalization,
)
from ..core import PipelineContext, StageResult

StageFn = Callable[[PipelineContext], StageResult]

STAGES: Dict[str, StageFn] = {
    "combine_raw": combine_raw.run,
    "ingredient_normalization": ingredient_normalization.run,
    "ingredient_ner_train": ingredient_ner_train.run,
    "ingredient_ner_infer": ingredient_ner_infer.run,
    "ingredient_encoding": ingredient_encoding.run,
    "cuisine_normalization": cuisine_normalization.run,
    "cuisine_classifier": cuisine_classifier.run,
}

__all__ = ["STAGES", "StageFn"]

