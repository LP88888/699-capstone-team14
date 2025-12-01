from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("pipeline.yaml")


class PipelineConfig:
    """Helper that wraps the consolidated pipeline YAML."""

    def __init__(self, data: Mapping[str, Any], path: Path) -> None:
        self._data = dict(data)
        self.path = path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PipelineConfig":
        resolved = Path(path or DEFAULT_CONFIG_PATH)
        if not resolved.exists():
            raise FileNotFoundError(f"Pipeline config not found at {resolved}")
        with resolved.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
            if not isinstance(raw, MutableMapping):
                raise ValueError(f"Pipeline config at {resolved} must be a mapping.")
        return cls(raw, resolved)

    @property
    def raw(self) -> Dict[str, Any]:
        """Return a deep copy of the raw config."""
        return deepcopy(self._data)

    @property
    def artifacts(self) -> Dict[str, Any]:
        return deepcopy(self._data.get("artifacts", {}))

    def stage(self, name: str, *, required: bool = True) -> Dict[str, Any]:
        pipeline_cfg = self._data.get("pipeline", {})
        stage_cfg = pipeline_cfg.get(name)
        if stage_cfg is None:
            if required:
                raise KeyError(f"Stage '{name}' not found in pipeline config {self.path}")
            return {}
        return deepcopy(stage_cfg)

    def logging(self, name: str, *, fallback: str | None = "pipeline") -> Dict[str, Any]:
        logging_cfg = self._data.get("logging", {})
        target = logging_cfg.get(name)
        if not target and fallback:
            target = logging_cfg.get(fallback)
        if target:
            return {"logging": deepcopy(target)}

        defaults = self._data.get("logging_defaults")
        if defaults:
            return {"logging": deepcopy(defaults)}
        return {}


def load_config(path: str | Path | None = None) -> PipelineConfig:
    return PipelineConfig.load(path)


__all__ = ["PipelineConfig", "load_config", "DEFAULT_CONFIG_PATH"]

