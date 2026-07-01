"""Thin CLI entrypoints (wired in pyproject [project.scripts]).

``automl-run`` drives the same ``pipeline.run_pipeline`` the Dagster job and tests use, so it
works with zero servers (SQLite + SQLite-backed MLflow). ``automl-seed`` runs an example loader.
"""

from __future__ import annotations

import subprocess
import sys


def seed() -> None:
    """`automl-seed [example]` -> run the example's seed loader (default: buyer_stage)."""
    example = sys.argv[1] if len(sys.argv) > 1 else "buyer_stage"
    raise SystemExit(subprocess.call([sys.executable, "-m", f"examples.{example}.seed_raw"]))


def run() -> None:
    """`automl-run` -> execute the full pipeline once and print a summary."""
    from automl_template.pipeline import run_pipeline

    result = run_pipeline()
    print(
        f"[automl-run] run_date={result.run_date} retrained={result.retrained} "
        f"model={result.model_version} rows_scored={result.rows_scored} "
        f"macro_f1={result.macro_f1} problems={result.consistency_problems}"
    )
    raise SystemExit(1 if result.consistency_problems else 0)
