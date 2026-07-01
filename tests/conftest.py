"""Shared test fixtures.

``isolated_env`` points the whole stack at a throwaway temp dir with SQLite for both the online
store and the MLflow registry — so every test runs end-to-end with ZERO external services.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SERVING_DB_URL", f"sqlite:///{tmp_path}/serving.db")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setenv("PIPELINE_CONFIG", "examples/buyer_stage/pipeline.yaml")
    monkeypatch.setenv("RUN_DATE", "2026-06-29")

    from automl_template.config import get_settings

    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


@pytest.fixture
def seeded(isolated_env):
    """isolated_env + a small synthetic train/test partition written to the offline store."""
    from examples.buyer_stage.synth import make_synthetic

    from automl_template.config import get_settings, load_pipeline
    from automl_template.store import offline

    settings = get_settings()
    cfg = load_pipeline(settings)
    train, test = make_synthetic(labels=cfg.labels, seed=7)
    offline.write_matrix(train, "2026-06-29", settings.data_dir, name="train")
    offline.write_matrix(test, "2026-06-29", settings.data_dir, name="test")
    return {"settings": settings, "cfg": cfg, "n_test": test.height}
