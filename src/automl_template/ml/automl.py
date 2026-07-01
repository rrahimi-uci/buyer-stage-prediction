"""AutoML training. Replaces SageMaker Autopilot (start_automl + check_automl_status).

FLAML is the default (MIT, fast, synchronous): the launch+poll+wait of the original two Lambdas
collapses into one synchronous call. ``metric="macro_f1"`` matches the original ``F1macro``
objective; ``max_iter``/``time_budget`` mirror AUTOML_MAX_CANDIDATE / runtime.

The returned ``predictor`` is the fitted FLAML ``AutoML`` object — it implements scikit-learn's
``predict``/``predict_proba`` and inverse-maps to the ORIGINAL class labels, so it round-trips
string targets cleanly and is logged to MLflow via the sklearn flavor.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from automl_template.config import PipelineConfig
from automl_template.errors import ConfigurationError


@dataclass
class TrainResult:
    predictor: Any  # fitted, sklearn-compatible predictor (FLAML AutoML)
    best_config: dict  # winning hyperparameters
    macro_f1: float  # macro-F1 on FLAML's internal validation
    feature_columns: list[str]


class AutoMLTrainer(Protocol):
    def train(self, X: pd.DataFrame, y: pd.Series, cfg: PipelineConfig) -> TrainResult: ...


def _numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    """Keep only numeric feature columns (the reshape emits numeric features)."""
    num = X.select_dtypes(include="number")
    return num


class FlamlTrainer:
    """Default trainer."""

    def train(self, X: pd.DataFrame, y: pd.Series, cfg: PipelineConfig) -> TrainResult:
        from flaml import AutoML

        Xn = _numeric_features(X)
        automl = AutoML()
        automl.fit(
            X_train=Xn,
            y_train=y,
            task="classification",
            metric="macro_f1",
            time_budget=cfg.automl.time_budget_seconds,
            max_iter=cfg.automl.max_candidates,  # honor the original candidate cap
            estimator_list=["lgbm", "xgboost", "rf", "extra_tree"],
            seed=cfg.automl.seed,
            verbose=0,
        )
        # FLAML reports loss = 1 - macro_f1 for this metric.
        macro_f1 = max(0.0, 1.0 - float(automl.best_loss))
        return TrainResult(
            predictor=automl,
            best_config=dict(automl.best_config or {}),
            macro_f1=macro_f1,
            feature_columns=list(Xn.columns),
        )


TrainerFactory = Callable[[], AutoMLTrainer]
_TRAINERS: dict[str, TrainerFactory] = {}


def register_trainer(name: str, factory: TrainerFactory) -> None:
    """Register an AutoML engine factory under ``name`` (extensibility seam for new backends)."""
    _TRAINERS[name] = factory


def get_trainer(cfg: PipelineConfig) -> AutoMLTrainer:
    try:
        factory = _TRAINERS[cfg.automl.engine]
    except KeyError:
        raise ConfigurationError(
            f"unknown automl engine {cfg.automl.engine!r}; registered: {sorted(_TRAINERS)}"
        ) from None
    return factory()


def _autogluon_trainer() -> AutoMLTrainer:
    # TODO(opt): from automl_template.ml.autogluon_trainer import AutoGluonTrainer; return it.
    raise ConfigurationError("AutoGluon trainer ships in the `autogluon` extra; not yet wired")


register_trainer("flaml", FlamlTrainer)
register_trainer("autogluon", _autogluon_trainer)

__all__ = ["AutoMLTrainer", "FlamlTrainer", "TrainResult", "get_trainer", "register_trainer"]
