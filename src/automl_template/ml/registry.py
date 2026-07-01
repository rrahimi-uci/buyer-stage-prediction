"""MLflow experiment logging + Model Registry champion/challenger promotion.

Replaces ``create_and_save_model`` (which had NO incumbent comparison — see ARCHITECTURE §3).
``log_model_run`` logs the fitted predictor (sklearn flavor) + the feature-column contract;
``register_and_maybe_promote`` promotes the new version to the ``champion`` alias ONLY if it
beats the incumbent on macro-F1 by ``promotion.margin``. Cold-start promotes unconditionally.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlflow  # module-level so tests can patch automl_template.ml.registry.mlflow
import mlflow.artifacts  # noqa: F401  (ensure submodules are bound on the mlflow package)
import mlflow.pyfunc  # noqa: F401
import mlflow.sklearn  # noqa: F401
from mlflow.exceptions import MlflowException

from automl_template.config import PipelineConfig
from automl_template.constants import (
    CHAMPION_ALIAS,
    FEATURE_COLUMNS_ARTIFACT,
    MODEL_ARTIFACT_PATH,
)
from automl_template.ml.automl import TrainResult

_MS_PER_DAY = 86_400_000


def log_model_run(result: TrainResult, cfg: PipelineConfig) -> str:
    """Log a training run (params, macro-F1, model, feature-column contract). Returns run_id.

    The feature-column list is logged as an artifact so scoring can realign incoming columns
    to the exact training schema — the contract that keeps train and serve in lock-step.
    """
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "engine": cfg.automl.engine,
                "metric": cfg.metric,
                "time_budget_seconds": cfg.automl.time_budget_seconds,
                "max_candidates": cfg.automl.max_candidates,
                "n_features": len(result.feature_columns),
                **{f"best.{k}": v for k, v in result.best_config.items()},
            }
        )
        mlflow.log_metric("macro_f1", result.macro_f1)
        mlflow.log_dict({"feature_columns": result.feature_columns}, FEATURE_COLUMNS_ARTIFACT)
        # cloudpickle handles FLAML's AutoML object (skops, the new default, rejects custom
        # classes not defined at top level).
        mlflow.sklearn.log_model(
            result.predictor,
            artifact_path=MODEL_ARTIFACT_PATH,
            serialization_format="cloudpickle",
        )
        return run.info.run_id


@dataclass
class ChampionInfo:
    version: str
    macro_f1: float
    age_days: float


def _alias_missing(exc: MlflowException) -> bool:
    """True iff the error means the champion alias / model simply doesn't exist yet."""
    code = getattr(exc, "error_code", "") or ""
    if code in ("RESOURCE_DOES_NOT_EXIST", "NOT_FOUND"):
        return True
    msg = str(exc).lower()
    return "does not exist" in msg or "not found" in msg or "no model version" in msg


def get_champion(model_name: str) -> ChampionInfo | None:
    """Return the current champion, or None on a genuine cold-start.

    Only a "does-not-exist" error is treated as cold-start; any other failure (network,
    auth, 5xx) is re-raised so a transient outage can never masquerade as "no champion"
    and trigger an unconditional promotion.
    """
    client = mlflow.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except MlflowException as exc:
        if _alias_missing(exc):
            return None
        raise
    metric = client.get_run(mv.run_id).data.metrics.get("macro_f1", 0.0)
    age_days = max(0.0, (time.time() * 1000 - mv.creation_timestamp) / _MS_PER_DAY)
    # mlflow returns version as int in some builds; normalize to str everywhere downstream.
    return ChampionInfo(version=str(mv.version), macro_f1=float(metric), age_days=age_days)


def register_and_maybe_promote(
    model_name: str,
    run_id: str,
    challenger_macro_f1: float,
    cfg: PipelineConfig,
) -> bool:
    """Register the just-trained model version and promote iff it wins. Returns promoted?."""
    client = mlflow.MlflowClient()
    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}"
    mv = mlflow.register_model(model_uri, model_name)

    champ = get_champion(model_name)
    if champ is None:
        client.set_registered_model_alias(model_name, CHAMPION_ALIAS, mv.version)
        return True  # cold-start

    if cfg.promotion.require_manual_promotion:
        return False  # leave for a human to flip the alias in the MLflow UI

    if challenger_macro_f1 > champ.macro_f1 + cfg.promotion.margin:
        client.set_registered_model_alias(model_name, CHAMPION_ALIAS, mv.version)
        return True
    return False


def load_champion(model_name: str) -> tuple[object, list[str], str] | None:
    """Load (predictor, feature_columns, version) for the champion, or None on cold-start."""
    client = mlflow.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except MlflowException as exc:
        if _alias_missing(exc):
            return None
        raise
    model = mlflow.pyfunc.load_model(f"models:/{model_name}@{CHAMPION_ALIAS}")
    cols = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/{FEATURE_COLUMNS_ARTIFACT}")[
        "feature_columns"
    ]
    return model, cols, str(mv.version)


__all__ = [
    "ChampionInfo",
    "get_champion",
    "log_model_run",
    "register_and_maybe_promote",
    "load_champion",
    "CHAMPION_ALIAS",
]
