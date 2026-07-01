"""Sensors. Replace EventBridge triggers + Step Functions Catch.

daily_data_sensor    -> fire the job when the run_date partition lands
run_failure_sensor   -> Apprise on any run failure (replaces SNS error path)
"""

from __future__ import annotations

from dagster import (
    DefaultSensorStatus,
    RunFailureSensorContext,
    SkipReason,
    run_failure_sensor,
    sensor,
)

from automl_template.notify.apprise_client import notify


@sensor(job_name="daily_pipeline", default_status=DefaultSensorStatus.STOPPED)
def daily_data_sensor():
    """Fire when a new run_date partition is available.

    TODO(phase-4): query raw_ingest for an unprocessed available run_date and, when found,
        ``yield RunRequest(run_key=run_date, run_config=...)`` instead of skipping.
    """
    yield SkipReason("skeleton: no new partition signal wired yet")


@run_failure_sensor
def run_failure_notification(context: RunFailureSensorContext) -> None:
    notify("error", f"run {context.dagster_run.run_id} failed: {context.failure_event.message}")


ALL_SENSORS = [daily_data_sensor, run_failure_notification]
