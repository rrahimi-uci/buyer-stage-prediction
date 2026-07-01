"""Polars finalization: enforce the canonical column order and the model feature set.

The canonical 228-column order is a DERIVED golden artifact (checked in under the
example's ``golden/``), never hand-authored framework code. This module reads it and
guarantees train and score produce identically-ordered matrices.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from automl_template.config import PipelineConfig


def load_column_order(artifact_path: str | Path) -> list[str]:
    """Read the frozen canonical column order (one column name per line)."""
    return [c.strip() for c in Path(artifact_path).read_text().splitlines() if c.strip()]


def apply_canonical_order(
    df: pl.DataFrame, column_order: list[str], preserve: tuple[str, ...] = ()
) -> pl.DataFrame:
    """Reindex ``df`` to the canonical feature order, filling absent columns with 0.0.

    Missing columns (e.g. a one-hot value unseen in this batch) are added as a Float64 0.0
    so the model always sees the exact training-time schema AND dtype — a core anti-skew
    guard. A float zero (not ``pl.lit(0)``, which is Int32) avoids a subtle dtype skew with
    the float feature columns produced by the coalesce/clip reshape.

    ``preserve`` lists extra columns (e.g. the entity key and/or target) to keep at the end
    if present, so callers that order before splitting off ``y`` don't silently drop it.
    """
    out = df
    for col in column_order:
        if col not in out.columns:
            out = out.with_columns(pl.lit(0.0).cast(pl.Float64).alias(col))
    keep = column_order + [c for c in preserve if c in out.columns and c not in column_order]
    return out.select(keep)


def split_features_and_key(df: pl.DataFrame, cfg: PipelineConfig) -> tuple[pl.DataFrame, pl.Series]:
    """Return (feature_frame, entity_key_series).

    The entity key is carried through the matrix but EXCLUDED from the model feature set
    (ARCHITECTURE §5 / MustFix #7) — the single most common train/serve bug.
    """
    key = df.get_column(cfg.entity_key)
    features = df.drop([cfg.entity_key]) if cfg.entity_key in df.columns else df
    if cfg.target in features.columns:
        features = features.drop([cfg.target])
    return features, key


__all__ = ["load_column_order", "apply_canonical_order", "split_features_and_key"]
