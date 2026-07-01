"""Exercise the Dagster path: materialize the whole job in-process on synthetic data.

Confirms the asset graph + asset checks run and that a prediction lands in the serving store
via the Dagster orchestration (not just the imperative pipeline()).
"""

from __future__ import annotations


def test_definitions_load():
    from automl_template.dagster_defs.definitions import daily_pipeline, defs

    assert defs is not None
    graph = (
        defs.resolve_asset_graph()
        if hasattr(defs, "resolve_asset_graph")
        else defs.get_asset_graph()
    )
    keys = {k.to_user_string() for k in graph.get_all_asset_keys()}
    assert {"raw_inputs", "feature_matrix", "model_version", "load_online_store"} <= keys
    assert daily_pipeline.name == "daily_pipeline"


def test_job_materializes_end_to_end(seeded):
    from automl_template.config import get_settings
    from automl_template.dagster_defs.definitions import defs
    from automl_template.store.online import OnlineStore

    job = (
        defs.resolve_job_def("daily_pipeline")
        if hasattr(defs, "resolve_job_def")
        else defs.get_job_def("daily_pipeline")
    )
    result = job.execute_in_process()
    assert result.success

    # The Dagster run wrote predictions to the (SQLite) serving store.
    row = OnlineStore(get_settings()).get_prediction("test_0")
    assert row is not None
    assert row["predicted_class"] in seeded["cfg"].labels
