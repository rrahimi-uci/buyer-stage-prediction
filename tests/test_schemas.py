"""Validation checks: non-negativity, dwell clip bound, per-class support."""

from __future__ import annotations

import polars as pl

from automl_template.config import PipelineConfig
from automl_template.schemas.feature_schema import (
    validate_feature_matrix,
    validate_training_data,
)


def _cfg() -> PipelineConfig:
    return PipelineConfig(
        target="buyer_stage",
        entity_key="member_id",
        labels=["A", "B"],
    )


def test_feature_matrix_flags_negative_and_overclip() -> None:
    df = pl.DataFrame(
        {
            "member_id": ["m1", "m2"],
            "total_searches_001": [1, -5],  # negative -> violation
            "total_ldp_dwell_time_seconds_001": [10, 999],  # > 180 -> violation
        }
    )
    violations = validate_feature_matrix(df, _cfg())
    assert any("negative" in v for v in violations)
    assert any("clip bound 180" in v for v in violations)


def test_training_data_class_set_and_support() -> None:
    cfg = _cfg()
    cfg.validation.min_rows_per_class = 2
    df = pl.DataFrame({"buyer_stage": ["A", "A", "B"]})  # B has 1 row < 2
    violations = validate_training_data(df, cfg)
    assert any("min_rows_per_class" in v for v in violations)
