# tabular-automl-template — Deep Plan & Architecture

> **North star:** *Copy two YAML files, drop in your tabular data, `make demo` — and you
> have a daily, drift-aware, batch-scoring-to-online-store ML pipeline running on a laptop
> with a 100%-permissive OSS stack.*

This is the open-source, single-node, `docker compose`-portable rewrite of the proprietary
AWS *buyer-stage-ml* pipeline (Step Functions + 11 Lambdas + Athena/EMR + SageMaker Autopilot
+ Batch Transform + DynamoDB + Glue + CloudFormation + Jenkins). It collapses that cloud estate
into one Dagster job of software-defined assets and asset checks; runs the feature reshape
in-process with DuckDB + Polars; trains a multiclass classifier with FLAML pinned to macro-F1;
versions the champion in MLflow; batch-scores to a Postgres online store via a transactional
staging-swap; serves point lookups over FastAPI; watches drift with Evidently; and fans out
failures through Apprise. **Domain knowledge lives entirely in data** (`pipeline.yaml` +
`feature_spec.yaml` + a seed loader under `examples/<domain>/`).

The design was selected by a 4-candidate architecture panel ("Minimal-Core" won, 8.33/10) and
hardened against an adversarial critique; every old→new mapping below was verified against the
source repo's actual SQL/lambda/CSV artifacts.

> **Implementation status (read first).** The pipeline is **wired and runs end-to-end**, with
> a passing test suite (unit + full end-to-end + a Dagster in-process materialization) that uses
> **zero external services** — SQLite for the online store and a SQLite-backed MLflow registry.
> A single `automl_template.pipeline.run_pipeline` defines the flow; the CLI (`automl-run`), the
> Dagster job, and the tests all drive it. `make demo` (or `seed → run`) trains a real FLAML
> model, registers/promotes a champion in MLflow, batch-scores, swaps the predictions into the
> serving store, and serves them over FastAPI. **Still skeleton / opt-in:** `drift_report`
> returns `0.0` (Evidently wiring + resolving `drift.reference_run_date: champion` to a concrete
> partition is phase-4); the `daily_data_sensor` skips (no partition signal wired); and the
> DuckDB `etl.reshape` path is exercised by its unit test but the buyer-stage example feeds a
> **pre-shaped** matrix (the original CSVs are post-ETL), so reshape is only invoked for a domain
> that supplies raw windowed inputs. The buyer-stage `golden/column_order.txt` is a committed
> placeholder. In code, the diagram's `automl_train` + `register_model` are **folded into the
> single `model_version` asset** (FLAML in `ml/automl.py`, registry in `ml/registry.py`).

## 1. Chosen OSS stack

| Layer | Tool (pinned) | License | Why |
|---|---|---|---|
| Orchestration | Dagster ~1.9 | Apache-2.0 | Assets + asset-checks map each Lambda/gate 1:1; schedules+sensors replace EventBridge + Step Functions Choice/Wait/Retry |
| ETL compute | DuckDB ~1.1 | MIT | In-process SQL join/reshape; no cluster at ~5k rows |
| Data shaping | Polars ~1.x | MIT | One-hot, 180s dwell-clip, coalesce-default, column ordering |
| AutoML (default) | FLAML ~2.3 | MIT | Fast synchronous multiclass over LightGBM/XGBoost/RF; `metric="macro_f1"` = original `F1macro` |
| AutoML (optional) | AutoGluon ~1.1 | Apache-2.0 | Stacked-ensemble accuracy upgrade behind same interface; off by default |
| Tracking + registry | MLflow ~2.18 | Apache-2.0 | Experiments, Model Registry, champion alias. SQLite-backed locally (registry works); MLflow server in compose |
| Online store | SQLAlchemy → SQLite (local) / PostgreSQL 16 (compose) | MIT / PostgreSQL | One portable schema; SQLite needs no server, Postgres scales. Valkey ~8 (BSD-3) documented swap, **not** Redis 7.4+/SSPL |
| Serving API | FastAPI + Uvicorn | MIT / BSD-3 | `/predict/{entity_id}`, `/healthz`, `/metrics`, `/model/active` |
| Validation | Pandera ~0.20 | MIT | dtype/range/non-neg + class-set + survey-window PIT |
| Monitoring | Evidently ==0.4.x | Apache-2.0 | Drift report → MLflow score → feeds retrain policy |
| Metrics (opt) | Prometheus; Grafana OFF | Apache-2.0 / AGPL | Grafana AGPL-flagged in compose, never auto-started |
| Notifications | Apprise ~1.9 | BSD-2 | One `APPRISE_URL` fans out (replaces SNS) |
| DB driver | pg8000 ~1.31 | BSD-3 | Pure-Python; avoids psycopg LGPL |
| Config | pydantic-settings ~2.x + YAML | MIT | `.env` replaces CFN Mappings |
| Packaging | Docker + compose profiles | Apache-2.0 | `core` / `monitoring` / `spark` / `autogluon` |
| IaC (opt) | OpenTofu ~1.8 | MPL-2.0 | VM module; avoids Terraform-BSL |
| CI | GitHub Actions + Trivy | — / Apache-2.0 | ruff, mypy, pytest, PIT, golden-parity, `make demo` |

**License posture:** default `core` profile is 100% permissive. AGPL (Grafana, MinIO *server*),
SSPL/RSAL (Redis 7.4+), BSL (Terraform) are quarantined to optional profiles/documented swaps
and flagged inline. CI audits the full resolved graph including optional profiles.

## 2. Old AWS → New OSS mapping (verified)

| Old AWS (verified artifact) | New OSS | Notes |
|---|---|---|
| 5× `consp_member_id_summary_t0NN.sql` = `CREATE EXTERNAL TABLE` DDL over pre-aggregated Parquet | DuckDB `read_parquet` views over 5 pre-windowed tables | **No aggregation here or in original** — windows arrive materialized |
| `buyer_intent_training_set_..._outlier_filter.sql` (401-line join+reshape) | `etl/reshape.py` (DuckDB SQL + Polars: one-hot/coalesce/180s-clip/`_00N` suffix); column order = checked-in golden | Re-scoped to a join/reshape generator, not a window generator |
| `buyer_intent_label_data_set.sql` (±2-day `member_id` join) | `examples/buyer_stage/label_join.sql` + `label_collapse.py` | PIT contract = ±2-day window, not `feature_ts <= label_ts` |
| `bs_ml_pipeline_state_machine.py` (Step Functions + 11 Lambdas) | one Dagster job: assets + checks + schedule + failure sensor | Choice→branching asset; Wait/poll→synchronous; Retry/Catch→retry policy + sensor |
| `check_if_daily_data_available` (`head_object`) | `daily_data_sensor` + `raw_inputs_present` check on run_date partition | |
| `check_if_model_available` (single-key `head_object`, **stale-model bug**) | `train_decision` = no-champion OR age>14d OR drift>0.30 | Bug confirmed; policy fix is a deliberate improvement |
| `check_if_training_data_available` | Pandera `training_data_valid` (rows, target, class-set==, min-rows-per-class) | |
| `start_automl` + `check_automl_status` (Autopilot, F1macro) | `automl_train` → FLAML in-process, macro_f1, trials→MLflow | launch+poll+wait collapse to one synchronous asset |
| `create_and_save_model` (no incumbent compare) | `register_model`: sklearn Pipeline→pyfunc; promote champion iff beats incumbent | champion/challenger is an improvement |
| `start_batch_transformation` + poll | `batch_predictions`: pyfunc load + Polars predict | |
| `bs_push2ddb_glue.py` (`{member_id, value, ttl}`) | `load_online_store`: staging + transactional UPSERT/swap | schema preserved: `predicted_class`+`ttl`, probs nullable |
| `check_uploading_to_dynamo_db_status` | asset check: rowcount + uniform `model_version` | blocks success notify |
| DynamoDB | serving Postgres `predictions_online` read by FastAPI; Valkey swap | PK `entity_id` == DDB hash key |
| `send_sns_notification` | Apprise from failure sensor + success hook | one URL, fan-out |
| CloudFormation + CFN Mappings | docker-compose profiles + per-env `.env` + YAML; optional OpenTofu | |
| Jenkins | GitHub Actions (+ Trivy + `make demo` regression gate) | |
| EventBridge schedule | Dagster `ScheduleDefinition` (cron) | |
| S3 (CSV matrix + model) | local Parquet volume + `./mlruns`; `s3fs` URIs to swap to object store | |

## 3. Architecture (one Dagster job)

```
 daily_data_sensor ─► [check] raw_inputs_present ─► reshape ─► feature_matrix
 (run_date partition)                                  │ (DuckDB join+reshape,
                                                       │  Polars one-hot/180s-clip)
                                                       ▼
                                          [check] feature_matrix_valid (Pandera)
   drift_report (Evidently) ───────────────────────────┤
   (HTML + score→MLflow)                                ▼
                              train_decision  (no-champion OR age>14 OR drift>0.30)
                                  │ retrain                      │ skip
                                  ▼                              │
                          [check] training_data_valid            │
                                  ▼                              │
                          automl_train (FLAML, macro_f1)         │
                                  ▼   trials→MLflow              │
                          register_model (sklearn Pipeline→pyfunc;│
                                  │     promote champion iff      │
                                  ▼     beats incumbent on test)  │
                              MLflow ◄── champion alias           │
                                  ▼                              ▼
                          batch_predictions (pyfunc.predict over feature_matrix)
                                  ▼
                          load_online_store (staging → transactional UPSERT/swap)
                                  ▼
                          [check] online_store_consistent (rowcount + uniform version)
                                  ▼
                          notify_success (Apprise)   run_failure_sensor ─► Apprise
                                  │ writes
                    postgres-serving: predictions_online (PK entity_id, predicted_class,
                                  │   class_probabilities JSONB null, model_version, ttl)
                                  ▼ reads
                          FastAPI: GET /predict/{entity_id} · /model/active · /metrics · /healthz
```

**Runtime walkthrough:** (1) daily-data gate fires on partition landing; (2) DuckDB reshape +
Polars shaping → `feature_matrix`; (3) Pandera validates dtype/non-neg/180-clip + ±2-day PIT;
(4) Evidently drift vs the champion's reference partition → MLflow; (5) `train_decision` policy
(no-champion ∨ age>`max_model_age_days` ∨ drift>`drift_threshold`) — fixes the stale-model bug;
(6) on retrain, `training_data_valid` gates (class-set + `min_rows_per_class`), then FLAML
(`macro_f1`); (7) promotion: challenger evaluated on the held-out test set, promote champion alias
iff `macro_f1 > champion + margin` (cold-start unconditional; prod may require manual approval);
(8) batch score via pyfunc; (9) staging write + single transactional swap into serving Postgres
(`ttl` preserves DynamoDB eviction); (10) consistency check blocks notify on failure; (11) FastAPI
serves; (12) Apprise on success/failure.

## 4. Feature handling (DuckDB/Polars replace Athena SQL)

**Critical (verified):** the 5 `consp_member_id_summary_t0NN.sql` files are `CREATE EXTERNAL
TABLE` DDL — windows arrive **pre-materialized**, there is no raw-event windowing in this repo.
The only real transformation is the 401-line join+reshape CTAS. So `etl/reshape.py` emits a
concrete, spec-driven reshape (per-window one-hot CASE, COALESCE-to-default, 180s dwell-clip,
`_00N` suffixing) and `etl/shaping.py` finalizes the canonical column order from a
**derived golden** artifact (to be generated by `make regen-golden`; a placeholder is
committed), not hand-authored framework code. A generic feature-spec→SQL generator is
**deferred** until a second example needs it.

- **No train/serve skew (by design):** `build_feature_matrix` is intended as the *single
  shared reshape entrypoint* for both the training and batch-scoring assets; wiring both call
  sites to it is a phase-1 task. The concrete skew guard that exists today is
  `shaping.apply_canonical_order`, which fills any absent column with a typed `0.0` in a frozen
  order so train and score see an identical schema + dtype. The sklearn-Pipeline-inside-pyfunc
  owns only model-adjacent transforms.
- **Point-in-time = ±2-day symmetric window** (`abs(snapshot − label_start) ≤ pit_window_days`).
  A naive one-sided contract would reject the original's own data.
- **Label derivation:** `buyer_stage` is not a raw survey column; `label_collapse.py` resolves
  `stage_label_{1,2,3}` (±2-day join) into the 4-class target.
- **Entity key:** `member_id` → `entity_id`, carried in `feature_matrix`, **excluded from the
  feature set**, used as the `predictions_online` PK. (`tr_val`=229 cols/no member_id;
  `test`=230 cols/`member_id` first + label `stage_label_1` — `seed_raw.py` reconciles both.)

## 5. Repo layout (FRAMEWORK vs EXAMPLE)

```text
tabular-automl-template/
├── docker-compose.yml                             # core/monitoring/spark/autogluon profiles
├── Dockerfile  Makefile  pyproject.toml  .env.example  LICENSE
├── config/pipeline.example.yaml                   # generic template config
├── src/automl_template/                           # ===== FRAMEWORK (zero domain knowledge) =====
│   ├── config.py                                  # Settings (zero-server defaults) + PipelineConfig + FeatureSpec
│   ├── constants.py · errors.py · runtime.py      # shared names · typed error hierarchy · RunContext
│   ├── py.typed                                   # PEP 561 — ships type information
│   ├── pipeline.py                                # run_pipeline() — the single source of the flow
│   ├── compute/{base,duckdb_backend}.py           # default DuckDB; Spark pluggable
│   ├── etl/{reshape,shaping}.py                   # reshape (raw windowed inputs) + canonical order
│   ├── schemas/feature_schema.py                  # validation + ±2-day PIT contract
│   ├── ml/{automl,registry,scoring}.py            # FLAML; MLflow champion/challenger; batch score
│   ├── store/{models,offline,online}.py           # portable SQLAlchemy schema; Parquet matrix; atomic-swap store
│   ├── monitoring/drift.py                         # Evidently (phase-4)
│   ├── notify/apprise_client.py
│   └── dagster_defs/{assets,asset_checks,sensors,schedules,definitions}.py  # thin wrappers over pipeline.py
├── api/main.py                                    # FastAPI serving (lazy settings; lifespan ensures schema)
├── examples/buyer_stage/                          # ===== EXAMPLE (data + config only) =====
│   ├── pipeline.yaml  feature_spec.yaml           # target/labels/windows + one-hot/clip/coalesce
│   ├── label_join.sql  label_collapse.py  seed_raw.py  synth.py  # synth.py = zero-data fallback
│   ├── golden/column_order.txt                    # derived parity fixture (placeholder)
│   └── sample_data/                               # NOT committed (provenance)
├── tests/                                         # unit + full end-to-end + Dagster materialize (SQLite, no servers)
├── scripts/                                       # smoke_predict, new_example, regen_golden
├── infra/{opentofu,helm}/                         # optional single-VM / K8s stubs
└── .github/                                       # ci.yml + issue/PR templates + dependabot
```

Adapting to a new problem: `make new-example NAME=foo TARGET=bar`, edit two YAMLs, drop seed data,
rerun — **no framework edits**.

## 6. Local dev

**Zero-server (no Docker)** — the fastest loop, used by CI and contributors:

```bash
pip install -e ".[dev]"
pytest                          # full unit + end-to-end suite on SQLite + SQLite-backed MLflow
python -m examples.buyer_stage.seed_raw   # seed (synthetic fallback if the CSVs are absent)
automl-run                      # train -> score -> load the SQLite serving store
```

**Full stack (Docker)** — `core` profile = 5 services: `postgres-platform` (MLflow backend),
`postgres-serving` (`predictions_online`), `mlflow` (:5000), `dagster` UI (:3000), `api` (:8000):

```bash
cp .env.example .env
make demo                       # up + seed + run pipeline + smoke-test GET /predict
make materialize                # same flow through the Dagster job (graph + asset checks)
```

The serving schema is created by the app (`store.models.ensure_schema`), so no `init.sql` is
needed. Two `postgres:16` containers are lightweight; to use one DB, point `SERVING_DB_URL` at
the platform instance and start only `postgres-platform mlflow dagster api`.

## 7. Phased build plan (~25–32 engineer-days)

| Phase | Goal | Effort |
|---|---|---|
| **0 — Skeleton & platform** | `docker compose up` brings up empty platform (5 services) | 3–4 d |
| **1 — Thin vertical slice** | seed→reshape→validate→FLAML cold-start→register→score→UPSERT→`/predict` 200 | 5–6 d |
| **2 — Retrain policy & promotion** | stale-model fix; champion/challenger; min-rows-per-class | 4–5 d |
| **3 — Online-store consistency & serving** | staging swap + consistency check; ttl; full API; testcontainer | 3–4 d |
| **4 — Orchestration, drift, notifications** | sensor + schedule + Evidently + Apprise | 4–5 d |
| **5 — Generality proof (fast-follow)** | `make new-example` + `examples/churn/` after buyer_stage parity green | 3–4 d |
| **6 — CI/CD, deploy, license audit** | GH Actions, OpenTofu VM, profiles, full-graph license audit | 3–4 d |

Phase 1 is the runnable thin vertical slice; later phases are incremental and (for
monitoring/drift/promotion) optional-gated so the core demo is never blocked.

## 8. Config, secrets, observability, CI/CD

- **Config:** `pydantic-settings` for runtime/secrets (DSNs, MLflow URI, Apprise URL); typed YAML
  `PipelineConfig` + `FeatureSpec` for the domain. Per-env `.env` replaces CFN Mappings.
- **Secrets:** `.env` git-ignored (CI asserts absence); least privilege — `api` sees only the
  serving DSN + MLflow URI, never the platform DB or Apprise URL. Prod: Docker/Swarm `*_FILE`
  secrets; no secret in an image layer.
- **Observability:** Dagster run logs + asset graph; MLflow lineage incl. drift score/HTML;
  FastAPI Prometheus `/metrics`. Optional `monitoring` profile (Grafana OFF/AGPL-flagged).
- **CI:** ruff + mypy + pytest; Pandera + PIT; golden-parity (golden generated once, PR-reviewed,
  so non-circular); `make demo` regression gate; docker build; Trivy; full-graph license audit.

## 9. Risks & open questions

| Risk | Mitigation |
|---|---|
| Reshape fidelity vs the 401-line CTAS | concrete reshape + checked-in golden; `test_parity` row-for-row in CI |
| DuckDB vs Athena/Presto dialect drift | golden-parity gate; `date_diff`/`strptime` ported and unit-tested |
| ±2-day window allows future labels; bad timestamp → leakage | Pandera window contract + `test_point_in_time` as a hard gate |
| FLAML minority-class recall vs Autopilot ensembling | pin macro_f1 + seeds + class weighting; AutoGluon pluggable |
| Two-Postgres footprint | both are lightweight `postgres:16`; collapse to one DB by pointing both DSNs at one instance (separate schemas) if RAM-constrained |
| Mixed `model_version` on bad swap | transactional staging-swap + `online_store_consistent` blocks notify |
| Retrain thresholds (14d / 0.30) need tuning | config-driven defaults + optional manual promotion gate |
| Golden trustworthiness (not circular) | generated once, reviewed in PR, frozen; `make regen-golden` + re-review |
| License drift via transitive deps / swaps | default 100% permissive; CI full-graph audit; copyleft only in optional profiles |
| **`sample_data` provenance** | de-identified real Move/Realtor.com data — NOT committed; replace with synthetic/public for a publishable demo |
| Single VM = no HA | acceptable at scale; Helm/K8s stub + backups + Valkey/object-store swaps documented |
