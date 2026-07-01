"""Schedules. Replaces the EventBridge cron that kicked off the daily pipeline."""

from __future__ import annotations

from dagster import DefaultScheduleStatus, ScheduleDefinition

daily_schedule = ScheduleDefinition(
    name="daily_pipeline_schedule",
    job_name="daily_pipeline",
    cron_schedule="0 6 * * *",  # 06:00 daily; override per-env
    default_status=DefaultScheduleStatus.STOPPED,
)

ALL_SCHEDULES = [daily_schedule]
