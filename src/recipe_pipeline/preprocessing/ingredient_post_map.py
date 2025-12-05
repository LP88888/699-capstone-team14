from __future__ import annotations

from pathlib import Path
from typing import Optional

from recipe_pipeline.ingrnorm.dedupe_map import apply_map_to_parquet_streaming, load_jsonl_map
from recipe_pipeline.core import PipelineContext, StageResult
from recipe_pipeline.utils import stage_logger


_STOP_SUFFIXES = {
    "powder",
    "sauce",
    "paste",
    "broth",
    "stock",
    "soup",
    "granules",
    "mix",
    "dressing",
    "rings",
    "juice",
    "puree",
}

_CANONICAL_SETS = {
    "onion": {
        "onion",
        "onions",
        "baby onions",
        "garden onions",
        "green onions",
        "salad onions",
    },
    "chicken": {"chicken", "chickens"},
}


def _canonicalize_token(token: str, mapping: dict) -> str:
    """Lightweight canonicalizer for obvious plurals/variants without touching other forms."""
    if not token:
        return token
    t = str(token).strip()
    tl = t.lower()
    words = tl.split()
    # Skip if token also names a preparation/suffix we don't want to collapse
    if any(w in _STOP_SUFFIXES for w in words[1:]):
        return t
    for canon, variants in _CANONICAL_SETS.items():
        if tl in variants:
            return canon
    # Simple plural-to-singular on the last token if we know the singular
    if words:
        last = words[-1]
        prefixes = words[:-1]
        cand = None
        if last.endswith("es") and len(last) > 2:
            cand = last[:-2]
        elif last.endswith("s") and len(last) > 1:
            cand = last[:-1]
        if cand:
            candidate_token = " ".join([*prefixes, cand]).strip()
            # Only collapse if the candidate is known to us (in canonical sets or mapping keys/values)
            in_sets = any(candidate_token in vs for vs in _CANONICAL_SETS.values())
            in_map = candidate_token in mapping or candidate_token in mapping.values()
            if in_sets or in_map:
                return candidate_token
    return t


def _apply_target(
    *,
    logger,
    target_cfg: dict,
    default_map: Optional[Path],
    default_list_col: str,
    default_dedupe: bool,
    default_collapse_dupes: bool,
):
    input_path = Path(target_cfg.get("input_path"))
    output_path = Path(target_cfg.get("output_path"))
    list_col = target_cfg.get("list_col", default_list_col)
    map_path = Path(target_cfg.get("map_path", default_map or ""))
    dedupe_tokens = target_cfg.get("dedupe_tokens", default_dedupe)
    collapse_duplicate_words = target_cfg.get("collapse_duplicate_words", default_collapse_dupes)

    mapping_obj: dict | str = {}
    if map_path and map_path.exists():
        mapping_obj = str(map_path)
        logger.info("Using post map file: %s", map_path)
    else:
        logger.warning("Map %s not found; proceeding with empty mapping", map_path or "<none>")

    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        return {"status": "failed", "input": str(input_path)}

    logger.info(
        "Applying post-dedupe map to %s -> %s (col=%s, dedupe_tokens=%s, collapse_duplicate_words=%s)",
        input_path,
        output_path,
        list_col,
        dedupe_tokens,
        collapse_duplicate_words,
    )
    apply_map_to_parquet_streaming(
        in_path=str(input_path),
        out_path=str(output_path),
        mapping=mapping_obj,
        list_col=list_col,
        dedupe_tokens=dedupe_tokens,
        collapse_duplicate_words=collapse_duplicate_words,
        canonicalizer=_canonicalize_token,
    )
    return {"status": "success", "input": str(input_path), "output": str(output_path), "map": str(map_path)}


def run(
    context: PipelineContext,
    *,
    force: bool = False,
) -> StageResult:
    cfg = context.stage("ingredient_post_map")
    logger = stage_logger(context, "ingredient_post_map", force=force)

    enabled = cfg.get("enabled", True)
    if not enabled:
        logger.info("ingredient_post_map disabled; skipping.")
        return StageResult(name="ingredient_post_map", status="skipped", details="disabled")

    default_map = Path(cfg.get("map_path", "data/ingr_normalized/ingredient_second_pass_map.jsonl"))
    default_list_col = cfg.get("list_col", "NER_clean")
    default_dedupe_tokens = cfg.get("dedupe_tokens", True)
    default_collapse_duplicate_words = cfg.get("collapse_duplicate_words", True)

    targets = cfg.get("targets")
    results = []

    if targets:
        for target_cfg in targets:
            results.append(
                _apply_target(
                    logger=logger,
                    target_cfg=target_cfg,
                    default_map=default_map,
                    default_list_col=default_list_col,
                    default_dedupe=default_dedupe_tokens,
                    default_collapse_dupes=default_collapse_duplicate_words,
                )
            )
    else:
        target_cfg = {
            "input_path": cfg.get("input_path", "data/ingr_normalized/recipes_data_clean_spell_dedup.parquet"),
            "output_path": cfg.get(
                "output_path", "data/ingr_normalized/recipes_data_clean_spell_dedup_post.parquet"
            ),
        }
        results.append(
            _apply_target(
                logger=logger,
                target_cfg=target_cfg,
                default_map=default_map,
                default_list_col=default_list_col,
                default_dedupe=default_dedupe_tokens,
                default_collapse_dupes=default_collapse_duplicate_words,
            )
        )

    failures = [r for r in results if r.get("status") == "failed"]
    if failures:
        return StageResult(
            name="ingredient_post_map",
            status="failed",
            details=f"failed for {len(failures)} target(s)",
            outputs={"results": results},
        )

    return StageResult(
        name="ingredient_post_map",
        status="success",
        outputs={"results": results},
    )


__all__ = ["run"]
