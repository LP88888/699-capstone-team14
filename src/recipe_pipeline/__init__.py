"""Pipeline orchestration utilities."""

# Silence noisy CuPy CUDA-path warnings early (before any CuPy/Thinc import).
import os
import warnings

os.environ.setdefault("CUPY_DISABLE_CUDA_ENV_CHECK", "1")
warnings.filterwarnings(
    "ignore",
    message="CUDA path could not be detected. Set CUDA_PATH environment variable if CuPy fails to load.",
    category=UserWarning,
)

from .runner import PIPELINE_ORDER, PipelineRunner, StageName

__all__ = ["PIPELINE_ORDER", "PipelineRunner", "StageName"]

