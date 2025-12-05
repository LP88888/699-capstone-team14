"""
Summarize ingredient dedupe map and generate simple visuals.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from ..core import PipelineContext, StageResult
from ..utils import stage_logger


def run(context: PipelineContext, *, force: bool = False) -> StageResult:
    cfg = context.stage("ingredients_summary")
    logger = stage_logger(context, "ingredients_summary", force=force)

    map_path = Path(cfg.get("data", {}).get("map_path", "data/ingr_normalized/dedupe_map.jsonl"))
    out_dir = Path(cfg.get("output", {}).get("reports_dir", "./reports/ingredients"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not map_path.exists():
        logger.warning("Map not found: %s", map_path)
        return StageResult(name="ingredients_summary", status="skipped", details=f"Missing map {map_path}")

    identities = 0
    merges = 0
    counter = Counter()
    examples = []
    for line in map_path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        src = obj.get("from")
        dst = obj.get("to")
        if not src or not dst:
            continue
        if src == dst:
            identities += 1
        else:
            merges += 1
            counter[dst] += 1
            if len(examples) < 20:
                examples.append((src, dst))

    summary = {
        "map_path": str(map_path),
        "identity": identities,
        "non_identity": merges,
        "top_targets": counter.most_common(50),
        "examples": examples,
    }
    with open(out_dir / "dedupe_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Bar plot of top merge targets
    if counter:
        top_df = pd.DataFrame(counter.most_common(20), columns=["target", "count"])
        plt.figure(figsize=(10, 6))
        top_df.plot(kind="barh", x="target", y="count", legend=False, colormap="viridis")
        plt.title("Top Ingredient Merge Targets")
        plt.xlabel("Count")
        plt.ylabel("Canonical Ingredient")
        plt.tight_layout()
        plt.savefig(out_dir / "dedupe_top_targets.png")
        plt.close()

    logger.info("Ingredient dedupe summary written to %s", out_dir)
    return StageResult(name="ingredients_summary", status="success", outputs={"reports_dir": str(out_dir)})
