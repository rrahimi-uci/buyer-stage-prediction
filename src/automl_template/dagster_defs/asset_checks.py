"""Asset checks — the OSS analogs of the original data-gate Lambdas.

raw_inputs_present          <- check_if_daily_data_available
feature_matrix_valid        <- Pandera-style dtype/range/clip checks
training_data_valid         <- check_if_training_data_available + per-class support
online_store_consistent     <- check_uploading_to_dynamo_db_status
"""

from dagster import AssetCheckResult, asset_check

from automl_template.constants import TRAIN_MATRIX
from automl_template.dagster_defs.assets import (
    feature_matrix,
    load_online_store,
    model_version,
    raw_inputs,
)
from automl_template.runtime import RunContext
from automl_template.schemas import feature_schema
from automl_template.store import offline
from automl_template.store.online import OnlineStore


@asset_check(asset=raw_inputs)
def raw_inputs_present() -> AssetCheckResult:
    ctx = RunContext.build(configure=False)
    present = offline.has_matrix(ctx.run_date, ctx.data_dir, TRAIN_MATRIX)
    return AssetCheckResult(passed=present, metadata={"run_date": ctx.run_date})


@asset_check(asset=feature_matrix)
def feature_matrix_valid() -> AssetCheckResult:
    ctx = RunContext.build(configure=False)
    df = offline.read_matrix(ctx.run_date, ctx.data_dir, TRAIN_MATRIX)
    violations = feature_schema.validate_feature_matrix(df, ctx.cfg)
    return AssetCheckResult(passed=not violations, metadata={"violations": violations[:20]})


@asset_check(asset=model_version)
def training_data_valid() -> AssetCheckResult:
    ctx = RunContext.build(configure=False)
    df = offline.read_matrix(ctx.run_date, ctx.data_dir, TRAIN_MATRIX)
    violations = feature_schema.validate_training_data(df, ctx.cfg)
    return AssetCheckResult(passed=not violations, metadata={"violations": violations[:20]})


@asset_check(asset=load_online_store)
def online_store_consistent() -> AssetCheckResult:
    store = OnlineStore(RunContext.build(configure=False).settings)
    versions = store.active_model_versions()
    problems = store.check_consistency()
    return AssetCheckResult(
        passed=not problems, metadata={"problems": problems, "versions": versions}
    )


ALL_CHECKS = [
    raw_inputs_present,
    feature_matrix_valid,
    training_data_valid,
    online_store_consistent,
]
