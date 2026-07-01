"""Configuration layer.

Two layers, mirroring docs/ARCHITECTURE.md §9:

1. ``Settings`` — runtime/secrets from env / ``.env`` (DSNs, MLflow URI, Apprise URL).
   Replaces AWS CloudFormation Mappings + Lambda env vars.
2. ``PipelineConfig`` + ``FeatureSpec`` — typed YAML describing the *domain problem*.
   Lives under ``examples/<domain>/`` and is the only thing a new use case edits.
"""

from __future__ import annotations

from datetime import UTC
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# --------------------------------------------------------------------------- #
# Layer 1: runtime / secrets                                                  #
# --------------------------------------------------------------------------- #
class Settings(BaseSettings):
    """Runtime configuration sourced from environment / ``.env``.

    Defaults are **zero-server**: SQLite for the online store and a SQLite-backed MLflow
    registry, with file artifacts under ``data/``. This lets ``pytest`` and a bare
    ``automl-run`` work with no Docker. ``docker compose`` overrides them via env to Postgres
    + the MLflow server (see ``.env.example`` / ``docker-compose.yml``).
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    pipeline_config: Path = Field(default=Path("examples/buyer_stage/pipeline.yaml"))
    data_dir: Path = Field(default=Path("data"))
    # Any SQLAlchemy URL. SQLite default works everywhere; compose sets a Postgres URL.
    serving_db_url: str = "sqlite:///./data/serving.db"
    # A SQLite-backed tracking URI enables the MLflow Model Registry locally (the bare file
    # store cannot); compose points this at the MLflow server (http://mlflow:5000).
    mlflow_tracking_uri: str = "sqlite:///./data/mlflow.db"
    mlflow_experiment: str = "tabular-automl-template"
    apprise_url: str = ""  # empty -> log-only
    run_date: str | None = None  # ISO date; defaults to "today" at run time

    def resolved_run_date(self) -> str:
        """Run date as YYYY-MM-DD; falls back to today (UTC) when unset."""
        if self.run_date:
            return self.run_date
        from datetime import datetime

        return datetime.now(UTC).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- #
# Layer 2: domain problem definition (typed YAML)                             #
# --------------------------------------------------------------------------- #
class AutoMLConfig(BaseModel):
    engine: str = "flaml"
    time_budget_seconds: int = 120
    max_candidates: int = 20
    seed: int = 42


class RetrainConfig(BaseModel):
    max_model_age_days: int = 14
    drift_threshold: float = 0.30


class PromotionConfig(BaseModel):
    margin: float = 0.0
    require_manual_promotion: bool = False


class ValidationConfig(BaseModel):
    min_rows_per_class: int = 50
    pit_window_days: int = 2


class OnlineStoreConfig(BaseModel):
    ttl_days: int = 7


class DriftConfig(BaseModel):
    reference_run_date: str = "champion"


class PipelineConfig(BaseModel):
    """The domain problem, loaded from ``examples/<domain>/pipeline.yaml``."""

    target: str
    entity_key: str
    problem_type: str = "multiclass"
    labels: list[str] = Field(default_factory=list)
    metric: str = "macro_f1"
    windows: list[str] = Field(default_factory=lambda: ["001", "007", "030", "060", "090"])
    automl: AutoMLConfig = AutoMLConfig()
    retrain: RetrainConfig = RetrainConfig()
    promotion: PromotionConfig = PromotionConfig()
    validation: ValidationConfig = ValidationConfig()
    online_store: OnlineStoreConfig = OnlineStoreConfig()
    drift: DriftConfig = DriftConfig()

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)


class FeatureSpec(BaseModel):
    """Per-feature handling, loaded from ``examples/<domain>/feature_spec.yaml``.

    Drives the reshape: which columns to one-hot (and their value sets), which to
    clip, which to coalesce-to-default, and the canonical output column order.
    """

    one_hot: dict[str, list[str]] = Field(default_factory=dict)
    clip: dict[str, float] = Field(default_factory=dict)  # column -> max (e.g. dwell -> 180)
    coalesce_default: dict[str, float] = Field(default_factory=dict)
    drop: list[str] = Field(default_factory=list)
    # Canonical column order is a DERIVED golden artifact, referenced not inlined.
    column_order_artifact: str = "golden/column_order.txt"

    @classmethod
    def load(cls, path: str | Path) -> FeatureSpec:
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_pipeline(settings: Settings | None = None) -> PipelineConfig:
    settings = settings or get_settings()
    return PipelineConfig.load(settings.pipeline_config)


def load_feature_spec(settings: Settings | None = None) -> FeatureSpec:
    """Load the active example's feature_spec.yaml (sibling of its pipeline.yaml)."""
    settings = settings or get_settings()
    spec_path = settings.pipeline_config.parent / "feature_spec.yaml"
    return FeatureSpec.load(spec_path)


def configure_mlflow(settings: Settings | None = None) -> None:
    """Point MLflow at the configured tracking URI, ensuring local dirs exist first."""
    import mlflow

    settings = settings or get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
