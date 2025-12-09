"""
Cleanup Stage: remove intermediate parquet artifacts while preserving a single canonical dataset.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable, List, Set

from recipe_pipeline.core import PipelineContext, StageResult
from recipe_pipeline.utils import stage_logger


def _resolve_paths(paths: Iterable[str]) -> Set[Path]:
    resolved = set()
    for p in paths:
        try:
            resolved.add(Path(p).resolve())
        except Exception:
            # Fall back to raw Path (e.g., if it does not exist yet)
            resolved.add(Path(p))
    return resolved


def _is_kept(path: Path, keep: Set[Path]) -> bool:
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    for k in keep:
        try:
            if resolved.samefile(k):
                return True
        except Exception:
            # Fallback to string comparison
            if str(resolved) == str(k):
                return True
    return False


def run(context: PipelineContext, *, force: bool = False) -> StageResult:
    cfg = context.stage("cleanup_artifacts", required=False) or {}
    logger = stage_logger(context, "cleanup_artifacts", force=force)

    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return StageResult(name="cleanup_artifacts", status="skipped", details="disabled")

    patterns: List[str] = cfg.get("patterns", ["./data/**/*.parquet"])
    dry_run = bool(cfg.get("dry_run", True))
    keep_paths_cfg: List[str] = cfg.get("keep", [])

    # Add artifacts by key if configured
    keep_artifact_keys: List[str] = cfg.get("keep_artifacts", ["cuisine_combined_encoded"])
    artifacts = context.config.artifacts if hasattr(context, "config") else {}
    for key in keep_artifact_keys:
        path = artifacts.get(key)
        if path:
            keep_paths_cfg.append(path)

    keep_paths = _resolve_paths(keep_paths_cfg)

    to_delete: List[Path] = []
    for pattern in patterns:
        for match in glob.glob(pattern, recursive=True):
            p = Path(match)
            if p.is_dir():
                continue
            if _is_kept(p, keep_paths):
                continue
            to_delete.append(p)

    deleted = []
    skipped = []
    for p in to_delete:
        if dry_run:
            skipped.append(str(p))
            continue
        try:
            p.unlink()
            deleted.append(str(p))
        except FileNotFoundError:
            continue
        except Exception as exc:
            logger.warning("Failed to delete %s: %s", p, exc)
            skipped.append(str(p))

    if dry_run:
        logger.info("Dry run enabled; would delete %s parquet(s)", len(to_delete))
    else:
        logger.info("Deleted %s parquet(s); kept %s", len(deleted), len(keep_paths))

    return StageResult(
        name="cleanup_artifacts",
        status="success",
        outputs={
            "deleted": deleted if not dry_run else [],
            "kept": sorted(str(p) for p in keep_paths),
            "dry_run": dry_run,
            "candidate_deletions": skipped if dry_run else deleted,
        },
    )


__all__ = ["run"]
