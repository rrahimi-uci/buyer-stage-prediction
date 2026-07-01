"""Point-in-time contract: the ±N-day SYMMETRIC window (not feature_ts <= label_ts).

A row whose label is 1 day in the FUTURE must PASS (the original allowed ±2 days); a row
5 days out must FAIL.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from automl_template.config import PipelineConfig
from automl_template.schemas.feature_schema import validate_point_in_time


def _cfg(window: int = 2) -> PipelineConfig:
    cfg = PipelineConfig(target="buyer_stage", entity_key="member_id")
    cfg.validation.pit_window_days = window
    return cfg


def test_future_label_within_window_passes() -> None:
    df = pl.DataFrame(
        {
            "snapshot_date": [date(2026, 1, 10)],
            "label_start_date": [date(2026, 1, 11)],  # +1 day, within ±2
        }
    )
    assert validate_point_in_time(df, _cfg()) == []


def test_label_outside_window_fails() -> None:
    df = pl.DataFrame(
        {
            "snapshot_date": [date(2026, 1, 10)],
            "label_start_date": [date(2026, 1, 16)],  # +6 days, outside ±2
        }
    )
    assert validate_point_in_time(df, _cfg()) != []
