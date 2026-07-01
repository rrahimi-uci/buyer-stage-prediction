"""Collapse the three Qualtrics labelings into the single 4-class ``buyer_stage`` target.

``buyer_stage`` is NOT a raw survey column. The survey provides ``stage_label_1/2/3``;
this step (porting the selection logic from the original ``bi_select_best_labeling.ipynb``)
resolves them into one canonical label. The shipped ``buyer_stage_tr_val_set.csv`` is already
collapsed (its last column IS ``buyer_stage``); the ``buyer_stage_test_set.csv`` carries
``stage_label_1`` and is collapsed here so train/test share the target definition.
"""

from __future__ import annotations

import polars as pl

CANONICAL_LABELS = ["Active Searcher", "Ready to Transact", "Dreamer", "Casual Explorer"]


def collapse_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Return ``df`` with a single ``buyer_stage`` column.

    Strategy (port of bi_select_best_labeling): prefer ``stage_label_1`` as the primary
    labeling, falling back to 2 then 3. Replace with the notebook's exact selection rule
    if it differed (TODO: confirm against the notebook's chosen labeling).
    """
    if "buyer_stage" in df.columns:
        return df  # already collapsed (tr_val set)

    label_cols = [c for c in ("stage_label_1", "stage_label_2", "stage_label_3") if c in df.columns]
    if not label_cols:
        raise ValueError("no stage_label_* columns to collapse")

    return df.with_columns(pl.coalesce([pl.col(c) for c in label_cols]).alias("buyer_stage"))


__all__ = ["collapse_labels", "CANONICAL_LABELS"]
