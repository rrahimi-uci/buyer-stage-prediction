"""Seed the buyer-stage example into the offline store.

Writes ``train``/``test`` feature matrices into the offline partition for the active run_date,
which ``pipeline.run_pipeline`` then reads. Two modes:

* **Real data** (if the CSVs are present in ``sample_data/``): reconciles the two source files
  (verified shapes, ARCHITECTURE §5) — ``tr_val`` is 229 cols (228 features + ``buyer_stage``,
  no member_id); ``test`` is 230 cols (``member_id`` first, label ``stage_label_1``). Both are
  aligned to ``[member_id, <228 features>, buyer_stage]``; the train file gets a surrogate
  ``member_id``; the test label is collapsed from ``stage_label_1``.
* **Synthetic fallback** (CSVs absent): generates buyer-stage-shaped data so the demo runs with
  zero proprietary data. The CSVs are intentionally not committed (provenance).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from automl_template.config import get_settings, load_pipeline
from automl_template.store import offline
from examples.buyer_stage.label_collapse import collapse_labels
from examples.buyer_stage.synth import make_synthetic

DEFAULT_SOURCE = Path(__file__).parent / "sample_data"
ENTITY_KEY = "member_id"
TARGET = "buyer_stage"


def _load_real(source: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    train = pl.read_csv(source / "buyer_stage_tr_val_set.csv")
    if ENTITY_KEY not in train.columns:
        train = (
            train.with_row_index(name="_i")
            .with_columns((pl.lit("train_") + pl.col("_i").cast(pl.Utf8)).alias(ENTITY_KEY))
            .drop("_i")
        )

    test = collapse_labels(pl.read_csv(source / "buyer_stage_test_set.csv"))
    drop = [c for c in ("stage_label_1", "stage_label_2", "stage_label_3") if c in test.columns]
    test = test.drop(drop)

    feature_cols = [c for c in train.columns if c not in (ENTITY_KEY, TARGET)]
    keep = [ENTITY_KEY, *feature_cols, TARGET]
    return train.select(keep), test.select([c for c in keep if c in test.columns])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--synthetic", action="store_true", help="force synthetic data")
    ap.add_argument("--run-date", default=None)
    args = ap.parse_args()

    settings = get_settings()
    cfg = load_pipeline(settings)
    run_date = args.run_date or settings.resolved_run_date()

    csv = args.source_dir / "buyer_stage_tr_val_set.csv"
    if args.synthetic or not csv.exists():
        if not args.synthetic:
            print(f"[seed] real CSVs not found in {args.source_dir} -> using synthetic data.")
        train, test = make_synthetic(labels=cfg.labels, entity_key=ENTITY_KEY, target=TARGET)
    else:
        train, test = _load_real(args.source_dir)

    offline.write_matrix(train, run_date, settings.data_dir, name="train")
    offline.write_matrix(test, run_date, settings.data_dir, name="test")
    print(
        f"[seed] run_date={run_date} train={train.shape} test={test.shape} "
        f"-> {settings.data_dir}/feature_matrix/run_date={run_date}/"
    )


if __name__ == "__main__":
    main()
