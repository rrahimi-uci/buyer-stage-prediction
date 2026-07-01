"""Top-level Dagster ``Definitions`` — the code location loaded by ``dagster dev``."""

from __future__ import annotations

from dagster import Definitions, define_asset_job

from automl_template.dagster_defs.asset_checks import ALL_CHECKS
from automl_template.dagster_defs.assets import ALL_ASSETS
from automl_template.dagster_defs.schedules import ALL_SCHEDULES
from automl_template.dagster_defs.sensors import ALL_SENSORS

# One job materializing the whole asset graph (the Step Functions state machine analog).
daily_pipeline = define_asset_job(name="daily_pipeline", selection="*")

defs = Definitions(
    assets=ALL_ASSETS,
    asset_checks=ALL_CHECKS,
    jobs=[daily_pipeline],
    schedules=ALL_SCHEDULES,
    sensors=ALL_SENSORS,
)
