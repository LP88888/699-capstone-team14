from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from preprocess_pipeline.config import PipelineConfig


@dataclass
class StageResult:
    name: str
    status: str
    outputs: Dict[str, Any] = field(default_factory=dict)
    details: Optional[str] = None


@dataclass
class PipelineContext:
    config: PipelineConfig
    workdir: Path = Path.cwd()

    def stage(self, name: str, *, required: bool = True) -> Dict[str, Any]:
        return self.config.stage(name, required=required)

    def logging(self, name: str) -> Dict[str, Any]:
        return self.config.logging(name)

