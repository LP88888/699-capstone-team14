"""Pipeline stage registry."""

from __future__ import annotations

from typing import Callable, Dict

from recipe_pipeline.analysis import pmi, recommender_stage, viz, ingredients_summary

from .core import PipelineContext, StageResult
from .preprocessing import (
    combine_raw,
    cuisine_classifier,
    cuisine_normalization,
    ingredient_encoding,
    ingredient_post_map,
    ingredient_ner_infer,
    ingredient_ner_train,
    ingredient_normalization,
)
from .analysis import baseline, graph

StageFn = Callable[[PipelineContext], StageResult]

STAGES: Dict[str, StageFn] = {
    "combine_raw": combine_raw.run,
    "ingredient_normalization": ingredient_normalization.run,
    "ingredient_post_map": ingredient_post_map.run,
    "ingredient_ner_train": ingredient_ner_train.run,
    "ingredient_ner_infer": ingredient_ner_infer.run,
    "ingredient_encoding": ingredient_encoding.run,
    "cuisine_normalization": cuisine_normalization.run,
    "cuisine_classifier": cuisine_classifier.run,
    "analysis_baseline": baseline.run,
    "analysis_pmi": pmi.run,           # <--- NEW
    "analysis_viz": viz.run,           # <--- REPLACES analysis_graph
    "analysis_graph": graph.run,
    "ingredients_summary": ingredients_summary.run,
    "analysis_recommender": recommender_stage.run,
}

__all__ = ["STAGES", "StageFn"]
