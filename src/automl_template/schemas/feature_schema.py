"""Data validation. Mirrors the original Lambda gates as Pandera checks.

Key subtlety (ARCHITECTURE §5 / MustFix #6): the point-in-time contract encodes the
original's *symmetric ±N-day* survey-join window — ``abs(snapshot - label_start) <= N`` —
NOT a one-sided ``feature_ts <= label_ts``. The label may legitimately be up to N days in
the future; a one-sided contract would reject the original's own data.
"""

from __future__ import annotations

import polars as pl
import polars.selectors as cs

from automl_template.config import PipelineConfig


def _numeric_columns(df: pl.DataFrame) -> set[str]:
    """Names of numeric columns (selectors API; avoids deprecated NUMERIC_DTYPES/is_numeric)."""
    return set(df.select(cs.numeric()).columns)


def validate_feature_matrix(df: pl.DataFrame, cfg: PipelineConfig) -> list[str]:
    """Return a list of human-readable violations (empty == valid).

    Checks: non-negativity on count/dwell columns, dwell clip-bound <= max, and the
    presence of the entity key. dtype checks are delegated to Pandera in phase-1.
    """
    violations: list[str] = []

    if cfg.entity_key not in df.columns:
        violations.append(f"missing entity key column '{cfg.entity_key}'")

    numeric = _numeric_columns(df)

    # Non-negativity on numeric feature columns (counts, dwell, etc.).
    for col in df.columns:
        if col in (cfg.entity_key, cfg.target):
            continue
        if col in numeric:
            min_val = df.get_column(col).min()
            if min_val is not None and min_val < 0:
                violations.append(f"column '{col}' has negative value {min_val}")

    # Dwell clip-bound: any *_dwell_time_* column must be <= 180 (the original clip).
    for col in df.columns:
        if "dwell_time" in col and col in numeric:
            max_val = df.get_column(col).max()
            if max_val is not None and max_val > 180:
                violations.append(f"dwell column '{col}' exceeds clip bound 180 ({max_val})")

    return violations


def validate_training_data(df: pl.DataFrame, cfg: PipelineConfig) -> list[str]:
    """Replaces ``check_if_training_data_available`` + adds a per-class support gate."""
    violations: list[str] = []
    if cfg.target not in df.columns:
        return [f"missing target column '{cfg.target}'"]

    present = set(df.get_column(cfg.target).unique().to_list())
    expected = set(cfg.labels)
    if cfg.labels and present != expected:
        violations.append(f"class set mismatch: expected {sorted(expected)}, got {sorted(present)}")

    counts = df.get_column(cfg.target).value_counts()
    for row in counts.iter_rows(named=True):
        label, n = row[cfg.target], row["count"]
        if n < cfg.validation.min_rows_per_class:
            violations.append(
                f"class '{label}' has {n} rows < min_rows_per_class "
                f"({cfg.validation.min_rows_per_class})"
            )
    return violations


def validate_point_in_time(
    df: pl.DataFrame,
    cfg: PipelineConfig,
    snapshot_col: str = "snapshot_date",
    label_start_col: str = "label_start_date",
) -> list[str]:
    """Assert every joined row obeys the ±pit_window_days symmetric window.

    ``abs(snapshot_date - label_start_date) <= pit_window_days``.
    """
    if snapshot_col not in df.columns or label_start_col not in df.columns:
        return []  # PIT only checkable pre-collapse, on the labelled join
    window = cfg.validation.pit_window_days

    # Date arithmetic requires temporal dtypes. Coerce yyyymmdd ints/strings to Date so the
    # check works whether the join emitted Date columns or raw snapshot_date_mst_yyyymmdd.
    def _as_date(col: str) -> pl.Expr:
        if df.schema[col] in (pl.Date, pl.Datetime):
            return pl.col(col).cast(pl.Date)
        return pl.col(col).cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)

    delta = (_as_date(snapshot_col) - _as_date(label_start_col)).dt.total_days().abs()
    bad = df.filter(delta > window)
    if bad.height:
        return [f"{bad.height} rows violate ±{window}-day point-in-time window"]
    return []


__all__ = ["validate_feature_matrix", "validate_training_data", "validate_point_in_time"]
