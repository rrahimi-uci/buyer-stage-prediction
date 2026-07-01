"""Scaffold a new example from the buyer_stage template.

    python scripts/new_example.py --name churn --target churned

Creates ``examples/<name>/`` with a pipeline.yaml + feature_spec.yaml stub, a seed_raw.py
shell, and an empty golden/. The framework needs NO changes — adapting is config + data.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
CONFIG = Path(__file__).resolve().parent.parent / "config" / "pipeline.example.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--target", default="<TARGET_COLUMN>")
    args = ap.parse_args()

    dest = EXAMPLES / args.name
    if dest.exists():
        raise SystemExit(f"{dest} already exists")
    (dest / "golden").mkdir(parents=True)
    (dest / "sample_data").mkdir()
    (dest / "__init__.py").write_text("")

    pipeline = CONFIG.read_text().replace("<TARGET_COLUMN>", args.target)
    (dest / "pipeline.yaml").write_text(pipeline)
    (dest / "feature_spec.yaml").write_text(
        "one_hot: {}\nclip: {}\ncoalesce_default: {}\ndrop: []\n"
        "column_order_artifact: golden/column_order.txt\n"
    )
    shutil.copy(EXAMPLES / "buyer_stage" / "seed_raw.py", dest / "seed_raw.py")
    print(f"created {dest}")
    print(f"  -> edit pipeline.yaml + feature_spec.yaml, then `make seed EXAMPLE={args.name}`")


if __name__ == "__main__":
    main()
