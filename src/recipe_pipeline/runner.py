from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableSequence, Optional, Sequence, TypedDict

from recipe_pipeline.config import PipelineConfig, load_config

from .core import PipelineContext, StageResult
from .registry import STAGES, StageFn

StageName = str

PIPELINE_ORDER: List[StageName] = [
    # "ingredient_normalization",
    # "ingredient_ner_train",
    # "combine_raw",
    # "ingredient_ner_infer",
    # "ingredient_post_map",
    # "ingredient_encoding",
    # "cuisine_normalization",
    # "ingredients_summary",
    # "cuisine_classifier",
    "analysis_baseline",
    "analysis_pmi",
    "analysis_graph",
    "analysis_viz",
    "analysis_recommender",
]


class StageOverrides(TypedDict, total=False):
    kwargs: Mapping[str, object]


class PipelineRunner:
    """High-level orchestrator for preprocess pipeline stages."""

    def __init__(self, config: PipelineConfig, *, workdir: Optional[Path] = None) -> None:
        self.config = config
        self.context = PipelineContext(config=config, workdir=workdir or Path.cwd())

    @classmethod
    def from_file(
        cls,
        path: str | Path | None = None,
        *,
        workdir: Optional[Path] = None,
    ) -> "PipelineRunner":
        return cls(load_config(path), workdir=workdir)

    def available_stages(self) -> List[StageName]:
        return list(STAGES.keys())

    def run(
        self,
        stages: Optional[Sequence[StageName]] = None,
        *,
        overrides: Optional[Mapping[StageName, Mapping[str, object]]] = None,
        stop_on_failure: bool = True,
    ) -> List[StageResult]:
        
        # if stages is None, run all stages in default order
        stages = stages or PIPELINE_ORDER 
        
        order = self._resolve_order(stages)
        overrides = overrides or {}

        results: List[StageResult] = []
        for name in order:
            stage_fn = STAGES.get(name) 
            if not stage_fn:
                results.append(
                    StageResult(
                        name=name,
                        status="skipped",
                        details=f"Stage '{name}' is not registered.",
                    )
                )
                continue

            kwargs = dict(overrides.get(name, {}))
            result = stage_fn(self.context, **kwargs)
            results.append(result)

            if stop_on_failure and result.status == "failed":
                break
        return results

    def _resolve_order(self, stages: Optional[Sequence[StageName]]) -> List[StageName]:
        if stages:
            return list(stages)
        return PIPELINE_ORDER.copy()


__all__ = ["PipelineRunner", "PIPELINE_ORDER", "StageName"]
