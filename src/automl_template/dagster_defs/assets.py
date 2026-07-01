"""Software-defined assets — one per stage of the original Step Functions DAG.

Each asset is a THIN wrapper over a step function in ``automl_template.pipeline`` (the single
source of the flow, also driven by the CLI and the end-to-end tests). State lives in MLflow
(registry) and the serving DB, so assets pass only small handles between each other.

  raw_inputs -> feature_matrix -> drift_report -> train_decision
              -> model_version (train+register+promote)? -> batch_predictions
              -> load_online_store

NOTE: no ``from __future__ import annotations`` — dagster (1.13+) inspects the real
``AssetExecutionContext`` parameter annotation and rejects a stringized one.
"""

import polars as pl
from dagster import AssetExecutionContext, Failure, asset

from automl_template import pipeline as P
from automl_template.constants import MODEL_NAME, SCORE_MATRIX, TRAIN_MATRIX
from automl_template.ml import registry, scoring
from automl_template.runtime import RunContext
from automl_template.store import offline
from automl_template.store.online import OnlineStore


@asset
def raw_inputs(context: AssetExecutionContext) -> dict:
    """Resolve the run_date partition (replaces the daily-data gate)."""
    ctx = RunContext.build(configure=False)
    context.log.info("raw_inputs for run_date=%s", ctx.run_date)
    return {"run_date": ctx.run_date}


@asset(deps=[raw_inputs])
def feature_matrix(context: AssetExecutionContext) -> dict:
    """Materialize/locate the feature matrix for the run_date partition.

    The buyer_stage example seeds a pre-shaped matrix (the original CSVs are post-ETL); a domain
    with raw windowed inputs would run ``etl.reshape`` here. We assert the partition is present.
    """
    ctx = RunContext.build(configure=False)
    if not offline.has_matrix(ctx.run_date, ctx.data_dir, TRAIN_MATRIX):
        raise Failure(
            f"no seeded matrix for run_date={ctx.run_date} under {ctx.data_dir}. "
            "Run `make seed EXAMPLE=<name>` (or the example's seed_raw) first."
        )
    return {"run_date": ctx.run_date}


@asset(deps=[feature_matrix])
def drift_report(context: AssetExecutionContext) -> float:
    """Evidently drift share vs the champion's reference partition (skeleton -> 0.0)."""
    # TODO(phase-4): compute_drift(current, reference) once a reference partition is pinned.
    return 0.0


# `drift_report` is a function parameter (a loaded input) so it must NOT also appear in `deps=`.
@asset(deps=[feature_matrix])
def train_decision(context: AssetExecutionContext, drift_report: float) -> bool:
    """Conditional-retrain POLICY (fixes the original stale-model bug)."""
    ctx = RunContext.build()
    decision, reason = P.decide_retrain(ctx.cfg, drift_share=drift_report)
    context.log.info("train_decision=%s (%s)", decision, reason)
    return decision


@asset
def model_version(context: AssetExecutionContext, train_decision: bool) -> str:
    """Train+register+promote when decided; return the serving champion version."""
    ctx = RunContext.build()
    if train_decision:
        train_df = offline.read_matrix(ctx.run_date, ctx.data_dir, TRAIN_MATRIX)
        P.train_register_promote(train_df, ctx.cfg)
    champ = registry.get_champion(MODEL_NAME)
    version = champ.version if champ else "none"
    context.log.info("serving model_version=%s", version)
    return version


@asset
def batch_predictions(context: AssetExecutionContext, model_version: str) -> pl.DataFrame:
    """Load the champion and score the run's matrix (score split if present, else train)."""
    ctx = RunContext.build()
    name = (
        SCORE_MATRIX
        if offline.has_matrix(ctx.run_date, ctx.data_dir, SCORE_MATRIX)
        else TRAIN_MATRIX
    )
    matrix = offline.read_matrix(ctx.run_date, ctx.data_dir, name)
    scored = scoring.score_with_champion(matrix, MODEL_NAME, ctx.cfg)
    if scored is None:
        raise Failure("no champion available to score with")
    predictions, _ = scored
    context.log.info("scored %d rows", predictions.height)
    return predictions


@asset(deps=[batch_predictions])
def load_online_store(
    context: AssetExecutionContext, batch_predictions: pl.DataFrame, model_version: str
) -> int:
    """Atomic swap of the run's predictions into the serving store."""
    ctx = RunContext.build(configure=False)
    rows = OnlineStore(ctx.settings).load_predictions(batch_predictions, model_version, ctx.cfg)
    context.log.info("loaded %d predictions (model_version=%s)", rows, model_version)
    return rows


ALL_ASSETS = [
    raw_inputs,
    feature_matrix,
    drift_report,
    train_decision,
    model_version,
    batch_predictions,
    load_online_store,
]
