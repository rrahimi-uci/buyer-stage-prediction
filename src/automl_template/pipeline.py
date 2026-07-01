"""End-to-end pipeline orchestration — the single source of the flow.

This imperative composition is what the CLI (`automl-run`), the end-to-end tests, AND the
Dagster assets all drive, so there is one definition of the pipeline and it is testable with
zero external services (SQLite + SQLite-backed MLflow). The Dagster layer (``dagster_defs``)
wraps these same step functions as software-defined assets for scheduling/observability.

Flow (ARCHITECTURE §4):
  load training matrix -> validate -> train_decision -> (train + register + promote)?
    -> score with champion -> swap into online store -> consistency check -> notify
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import polars as pl

from automl_template.config import PipelineConfig, Settings
from automl_template.constants import MODEL_NAME, SCORE_MATRIX, TRAIN_MATRIX
from automl_template.errors import ChampionUnavailableError, DataValidationError
from automl_template.ml import registry, scoring
from automl_template.ml.automl import get_trainer
from automl_template.notify.apprise_client import notify
from automl_template.runtime import RunContext
from automl_template.schemas import feature_schema
from automl_template.store import offline
from automl_template.store.online import OnlineStore

logger = logging.getLogger("automl_template.pipeline")


@dataclass
class PipelineResult:
    run_date: str
    retrained: bool
    promoted: bool
    model_version: str | None
    rows_scored: int
    macro_f1: float | None
    consistency_problems: list[str]


def decide_retrain(cfg: PipelineConfig, drift_share: float = 0.0) -> tuple[bool, str]:
    """Conditional-retrain POLICY (fixes the original stale-model bug)."""
    champ = registry.get_champion(MODEL_NAME)
    if champ is None:
        return True, "cold-start (no champion)"
    if champ.age_days > cfg.retrain.max_model_age_days:
        return True, f"age {champ.age_days:.1f}d > {cfg.retrain.max_model_age_days}d"
    if drift_share > cfg.retrain.drift_threshold:
        return True, f"drift {drift_share:.2f} > {cfg.retrain.drift_threshold}"
    return False, "champion fresh and no drift"


def train_register_promote(train_df: pl.DataFrame, cfg: PipelineConfig) -> tuple[str, bool, float]:
    """Train, log to MLflow, register, and maybe promote. Returns (run_id, promoted, macro_f1)."""
    features = train_df.drop([c for c in (cfg.entity_key, cfg.target) if c in train_df.columns])
    y = train_df.get_column(cfg.target).to_pandas()
    result = get_trainer(cfg).train(features.to_pandas(), y, cfg)
    run_id = registry.log_model_run(result, cfg)
    promoted = registry.register_and_maybe_promote(MODEL_NAME, run_id, result.macro_f1, cfg)
    logger.info("trained macro_f1=%.4f promoted=%s", result.macro_f1, promoted)
    return run_id, promoted, result.macro_f1


def run_pipeline(
    *,
    settings: Settings | None = None,
    cfg: PipelineConfig | None = None,
    run_date: str | None = None,
    train_name: str = TRAIN_MATRIX,
    score_name: str = SCORE_MATRIX,
) -> PipelineResult:
    """Execute the full pipeline against a seeded offline partition. Returns a summary."""
    ctx = RunContext.build(settings=settings, cfg=cfg, run_date=run_date)
    cfg = ctx.cfg

    train_df = offline.read_matrix(ctx.run_date, ctx.data_dir, name=train_name)
    score_df = offline.read_matrix(ctx.run_date, ctx.data_dir, name=score_name)

    violations = feature_schema.validate_training_data(train_df, cfg)
    if violations:
        notify("error", f"training data invalid: {violations}", ctx.settings)
        raise DataValidationError(violations)

    macro_f1: float | None = None
    retrain, reason = decide_retrain(cfg)
    logger.info("train_decision=%s (%s)", retrain, reason)
    if retrain:
        _, _, macro_f1 = train_register_promote(train_df, cfg)

    scored = scoring.score_with_champion(score_df, MODEL_NAME, cfg)
    if scored is None:
        notify("error", "no champion available to score with", ctx.settings)
        raise ChampionUnavailableError("no champion model available after train_decision")
    predictions, version = scored

    store = OnlineStore(ctx.settings)
    rows = store.load_predictions(predictions, version, cfg)
    problems = store.check_consistency()
    if problems:
        notify("error", f"online store inconsistent: {problems}", ctx.settings)
    else:
        notify("info", f"run {ctx.run_date}: scored {rows} rows with model {version}", ctx.settings)

    return PipelineResult(
        run_date=ctx.run_date,
        retrained=retrain,
        promoted=retrain,
        model_version=version,
        rows_scored=rows,
        macro_f1=macro_f1,
        consistency_problems=problems,
    )


__all__ = [
    "MODEL_NAME",
    "PipelineResult",
    "decide_retrain",
    "train_register_promote",
    "run_pipeline",
]
