"""Batch scoring. Replaces SageMaker Batch Transform (start_batch_transformation + poll).

Loads the champion via the registry and scores a feature matrix with Polars. Incoming features
are realigned to the champion's exact training feature-column contract (missing columns filled
with ``0.0``, extra columns dropped, canonical order enforced) — the anti-skew guarantee.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from automl_template.config import PipelineConfig
from automl_template.ml.registry import load_champion


def align_features(features: pl.DataFrame, feature_columns: list[str]) -> pl.DataFrame:
    """Return ``features`` projected to exactly ``feature_columns`` (fill absent with 0.0)."""
    out = features
    for col in feature_columns:
        if col not in out.columns:
            out = out.with_columns(pl.lit(0.0).cast(pl.Float64).alias(col))
    return out.select(feature_columns)


def score_batch(
    predictor: Any,
    feature_columns: list[str],
    matrix: pl.DataFrame,
    model_version: str,
    cfg: PipelineConfig,
) -> pl.DataFrame:
    """Score ``matrix`` and return an online-store-ready frame.

    ``matrix`` carries the entity key plus feature columns (and possibly the target, which is
    ignored). Returns columns: ``<entity_key>``, ``predicted_class``, ``model_version``.
    """
    entity = matrix.get_column(cfg.entity_key)
    X = align_features(matrix, feature_columns).to_pandas()
    preds = predictor.predict(X)  # pyfunc / sklearn predict -> original labels
    return pl.DataFrame(
        {
            cfg.entity_key: entity,
            "predicted_class": pl.Series([str(p) for p in preds]),
        }
    ).with_columns(pl.lit(model_version).alias("model_version"))


def score_with_champion(
    matrix: pl.DataFrame, model_name: str, cfg: PipelineConfig
) -> tuple[pl.DataFrame, str] | None:
    """Convenience: load the champion and score. Returns (predictions, version) or None."""
    loaded = load_champion(model_name)
    if loaded is None:
        return None
    predictor, feature_columns, version = loaded
    return score_batch(predictor, feature_columns, matrix, version, cfg), version


__all__ = ["align_features", "score_batch", "score_with_champion"]
