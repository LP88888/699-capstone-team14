from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from recipe_pipeline.config import PipelineConfig


@dataclass
class StageResult:
    name: str = ""
    status: str = "success"
    outputs: Dict[str, Any] = field(default_factory=dict)
    details: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineContext:
    config: PipelineConfig
    workdir: Path = Path.cwd()
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("recipe_pipeline"))

    def stage(self, name: str, *, required: bool = True) -> Dict[str, Any]:
        return self.config.stage(name, required=required)

    def logging(self, name: str) -> Dict[str, Any]:
        return self.config.logging(name)

