"""Regenerate the checked-in golden parity fixtures for an example.

The golden (column_order.txt + feature_matrix.parquet) is the trusted reference that
``test_parity`` compares against. Regenerating it is a deliberate, reviewed act — the diff
MUST be inspected in a PR (the test is otherwise circular). Hence this is a separate,
explicit command, never run automatically.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", default="buyer_stage")
    args = ap.parse_args()
    golden_dir = Path("examples") / args.example / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    # TODO(phase-1): run the reshape over sample_data and write:
    #   golden/column_order.txt   (canonical 228-feature order)
    #   golden/feature_matrix.parquet
    print(f"[regen_golden] would regenerate {golden_dir} — REVIEW THE DIFF IN A PR.")


if __name__ == "__main__":
    main()
