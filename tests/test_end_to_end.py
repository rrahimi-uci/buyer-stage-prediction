"""Full end-to-end test: seed -> train -> register/promote -> score -> online store -> serve.

Runs the real FLAML/MLflow/DuckDB stack against SQLite with zero external services.
"""

from __future__ import annotations


def test_pipeline_runs_and_serves(seeded):
    from automl_template.pipeline import run_pipeline
    from automl_template.store.online import OnlineStore

    settings, cfg, n_test = seeded["settings"], seeded["cfg"], seeded["n_test"]

    result = run_pipeline(settings=settings, cfg=cfg, run_date="2026-06-29")

    assert result.retrained is True  # cold-start -> trains
    assert result.promoted is True
    assert result.model_version is not None
    assert result.rows_scored == n_test
    assert result.consistency_problems == []
    assert result.macro_f1 is not None

    # Online store holds exactly the scored rows under one uniform model version.
    store = OnlineStore(settings)
    assert store.check_consistency() == []
    row = store.get_prediction("test_0")
    assert row is not None
    assert row["predicted_class"] in cfg.labels
    assert row["model_version"] == result.model_version


def test_fastapi_serves_predictions(seeded):
    from fastapi.testclient import TestClient

    from automl_template.pipeline import run_pipeline

    settings, cfg = seeded["settings"], seeded["cfg"]
    run_pipeline(settings=settings, cfg=cfg, run_date="2026-06-29")

    import api.main as api_main

    with TestClient(api_main.app) as client:
        ok = client.get("/predict/test_0")
        assert ok.status_code == 200
        assert ok.json()["predicted_class"] in cfg.labels

        missing = client.get("/predict/__nope__")
        assert missing.status_code == 404

        assert client.get("/healthz").json()["status"] == "ok"
        assert client.get("/model/active").json()["model_versions"]


def test_second_run_keeps_store_uniform(seeded):
    """A second run swaps the store atomically — never a mix of model versions."""
    from automl_template.pipeline import run_pipeline
    from automl_template.store.online import OnlineStore

    settings, cfg = seeded["settings"], seeded["cfg"]
    run_pipeline(settings=settings, cfg=cfg, run_date="2026-06-29")
    run_pipeline(settings=settings, cfg=cfg, run_date="2026-06-29")

    store = OnlineStore(settings)
    assert store.check_consistency() == []
    assert len(store.active_model_versions()) == 1  # uniform after the swap
