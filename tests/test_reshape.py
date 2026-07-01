"""Reshape correctness on a tiny in-memory fixture (no docker needed)."""

from __future__ import annotations

import polars as pl

from automl_template.compute.duckdb_backend import DuckDBBackend
from automl_template.config import FeatureSpec, PipelineConfig
from automl_template.etl.reshape import build_feature_matrix


def _cfg() -> PipelineConfig:
    return PipelineConfig(target="buyer_stage", entity_key="member_id", windows=["001", "007"])


def _spec() -> FeatureSpec:
    return FeatureSpec(
        one_hot={"ldp_dominant_segment": ["for sale", "for rent"]},
        clip={"total_ldp_dwell_time_seconds": 180},
        coalesce_default={"total_searches": 0, "total_ldp_dwell_time_seconds": 0},
    )


def test_dwell_clip_and_one_hot() -> None:
    backend = DuckDBBackend()
    for w in ("001", "007"):
        backend.register_dataframe(
            f"t{w}",
            pl.DataFrame(
                {
                    "member_id": ["m1", "m2"],
                    "snapshot_date_mst_yyyymmdd": ["20260101", "20260101"],
                    "ldp_dominant_segment": ["for sale", "for rent"],
                    "total_ldp_dwell_time_seconds": [999, None],  # 999 must clip to 180
                    "total_searches": [3, None],  # None must coalesce to 0
                }
            ),
        )

    out = build_feature_matrix(backend, _cfg(), _spec(), {"001": "t001", "007": "t007"})

    assert "member_id" in out.columns
    assert out.get_column("total_ldp_dwell_time_seconds_001").max() <= 180
    assert out.get_column("total_searches_001").null_count() == 0
    assert "ldp_dominant_segment_for_sale_001" in out.columns
