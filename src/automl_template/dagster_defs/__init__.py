"""Dagster definitions: the single job that replaces the Step Functions state machine.

Each original Lambda/gate maps to a Dagster asset or asset check:
  Choice  -> branching asset (train_decision)
  Wait/poll -> synchronous in-process asset
  Retry/Catch -> Dagster retry policy + run_failure_sensor
"""
