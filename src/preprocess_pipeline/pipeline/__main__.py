from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .runner import PIPELINE_ORDER, PipelineRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preprocess pipeline stages")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to consolidated pipeline YAML (default: preprocess_pipeline/config/pipeline.yaml)",
    )
    parser.add_argument(
        "--stages",
        type=str,
        nargs="*",
        default=None,
        help="Specific stages to run (default: full pipeline order)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available stages and exit",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Working directory for stage execution (default: current directory)",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue running remaining stages even if a stage fails",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = PipelineRunner.from_file(args.config, workdir=args.workdir)

    if args.list:
        print("Available stages:")
        for name in runner.available_stages():
            prefix = " *" if name in PIPELINE_ORDER else "  "
            print(f"{prefix} {name}")
        return 0

    results = runner.run(
        stages=args.stages,
        stop_on_failure=not args.keep_going,
    )

    failed = False
    print("=" * 60)
    for result in results:
        status = result.status.upper()
        print(f"{status:>8}  {result.name}")
        if result.details:
            print(f"    {result.details}")
        if result.outputs:
            for key, value in result.outputs.items():
                print(f"    - {key}: {value}")
        failed = failed or result.status == "failed"

    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

