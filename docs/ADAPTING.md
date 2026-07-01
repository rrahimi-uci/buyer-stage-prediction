# Adapting to your own tabular problem

The framework (`src/automl_template/`) has **zero domain knowledge**. To target a new problem
you create an example — config + data — and never touch framework code.

## 1. Scaffold

```bash
make new-example NAME=churn TARGET=churned
```

This creates `examples/churn/` with `pipeline.yaml`, `feature_spec.yaml`, a `seed_raw.py` shell,
and `golden/`.

## 2. Define the problem — `examples/churn/pipeline.yaml`

```yaml
target: churned
entity_key: customer_id          # carried, excluded from features, online-store PK
problem_type: binary             # multiclass | binary | regression
labels: ["churn", "retain"]      # classification only; order is canonical
metric: macro_f1
windows: ["007", "030", "090"]   # your feature windows (suffix _<window>); [] if none
```

## 3. Define feature handling — `examples/churn/feature_spec.yaml`

```yaml
one_hot:
  plan_type: ["free", "pro", "enterprise"]
clip:
  session_seconds: 3600          # cap outliers (the buyer-stage example clips dwell at 180)
coalesce_default:
  total_logins: 0
drop: ["signup_ip"]
column_order_artifact: golden/column_order.txt
```

## 4. Wire the data — `examples/churn/seed_raw.py`

Adapt the buyer-stage seeder: read your sources, map your entity column to `entity_key`, align
train/score to the same feature schema, write Parquet into the offline store.

## 5. Freeze the golden, then run

```bash
make seed EXAMPLE=churn
make regen-golden EXAMPLE=churn   # review the column_order diff in a PR — it's the parity baseline
PIPELINE_CONFIG=examples/churn/pipeline.yaml make run
```

## Swaps & scale-out

| Want | Do |
|---|---|
| Higher-accuracy AutoML | `pip install -e ".[autogluon]"`, set `automl.engine: autogluon` |
| Bigger-than-laptop data | `pip install -e ".[spark]"`, implement `compute/spark_backend.py` behind `ComputeBackend` |
| High-QPS online reads | swap Postgres serving for **Valkey** (BSD-3) — `store/online.py` is the only seam |
| dbt-managed features | lift `reshape.py`'s SQL into `dbt-duckdb` models (Apache-2.0); behavior unchanged |
| Cloud VM deploy | `infra/opentofu` single-VM module (MPL-2.0) |
| Object storage | point Parquet paths at `s3://…` via `s3fs`; use the MinIO **client** (Apache-2.0), never the AGPL server |

> Avoid Redis 7.4+ (SSPL/RSAL), Grafana (AGPL), and Terraform (BSL) in a redistributed product.
> See [LICENSES.md](LICENSES.md).
