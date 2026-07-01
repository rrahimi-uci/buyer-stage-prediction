"""Run context — the resolved inputs a single pipeline run needs.

Bundles the runtime ``Settings``, the domain ``PipelineConfig``, and the resolved ``run_date``
into one immutable object built in a single place, replacing the repeated
``get_settings()`` / ``load_pipeline()`` / ``configure_mlflow()`` / ``resolved_run_date()``
preamble that otherwise appears in every asset and entrypoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from automl_template.config import (
    PipelineConfig,
    Settings,
    configure_mlflow,
    get_settings,
    load_pipeline,
)


@dataclass(frozen=True)
class RunContext:
    settings: Settings
    cfg: PipelineConfig
    run_date: str

    @classmethod
    def build(
        cls,
        *,
        settings: Settings | None = None,
        cfg: PipelineConfig | None = None,
        run_date: str | None = None,
        configure: bool = True,
    ) -> RunContext:
        """Resolve settings + config + run_date (and point MLflow at its tracking URI)."""
        settings = settings or get_settings()
        cfg = cfg or load_pipeline(settings)
        if configure:
            configure_mlflow(settings)
        return cls(settings=settings, cfg=cfg, run_date=run_date or settings.resolved_run_date())

    @property
    def data_dir(self) -> Path:
        return self.settings.data_dir


__all__ = ["RunContext"]
