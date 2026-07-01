# tabular-automl-template

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Dependencies: permissive](https://img.shields.io/badge/deps-Apache%2FMIT%2FBSD-brightgreen.svg)](docs/LICENSES.md)
[![Status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)](docs/ARCHITECTURE.md)

> Copy two YAML files, drop in your tabular data, `make demo` — and you have a daily,
> drift-aware, **batch-scoring-to-online-store** ML pipeline running on a laptop with a
> 100%-permissive open-source stack.

A domain-agnostic, single-node, `docker compose`-portable **tabular AutoML pipeline template**.
It is the open-source rewrite of a proprietary AWS pipeline (Step Functions + Lambdas + Athena/EMR +
SageMaker Autopilot + Batch Transform + DynamoDB + Glue + CloudFormation + Jenkins), collapsed into a
single [Dagster](https://dagster.io) job of software-defined assets.

The original real-estate **buyer-stage** classifier ships only as the worked example under
[`examples/buyer_stage/`](examples/buyer_stage/) — all domain knowledge lives in data (YAML + seed
loader), never in framework code.

## What you get

- **One command to run everything** — `./start.sh` builds and starts the whole stack; `./stop.sh`
  tears it down. (Or `pip install` + `automl-run` for a zero-Docker, SQLite-only loop.)
- **A visual control panel** — a [Streamlit dashboard](dashboard/) to launch pipeline runs, tune
  the config, and watch the champion model, predictions, and service health **live in the browser**.
- **A complete ML pipeline** — reshape → validate (Pandera + a ±2-day point-in-time contract) →
  AutoML train (FLAML, macro-F1) → champion/challenger promotion (MLflow registry) → batch score →
  atomic swap into an online store → serve over FastAPI, with drift monitoring and notifications.
- **Real orchestration** — the exact same flow runs as one [Dagster](https://dagster.io) job (asset
  graph, schedules, sensors, asset checks) with a UI at `:3000`.
- **Bring your own domain** — `make new-example`, edit two YAML files, drop in your data. The
  framework carries **zero domain knowledge**, so no framework edits are needed.

## Stack

| Layer | Tool | License |
|---|---|---|
| Orchestration | Dagster | Apache-2.0 |
| ETL compute | DuckDB | MIT |
| Data shaping | Polars | MIT |
| AutoML (default) | FLAML | MIT |
| AutoML (optional) | AutoGluon | Apache-2.0 |
| Tracking + registry | MLflow | Apache-2.0 |
| Online store | SQLite (local) / PostgreSQL (compose) | MIT / PostgreSQL |
| Serving API | FastAPI | MIT |
| Dashboard (optional) | Streamlit | Apache-2.0 |
| Validation | Pandera | MIT |
| Drift | Evidently | Apache-2.0 |
| Notifications | Apprise | BSD-2 |
| Config | pydantic-settings | MIT |

The default `core` profile is **100% permissive** and trivially usable in commercial products
(Apache-2.0 / MIT core; BSD-3 only for foundational libs like numpy/pandas/scikit-learn). No
copyleft on the default path. See [docs/LICENSES.md](docs/LICENSES.md).

## Repository layout

```text
src/automl_template/   framework — ZERO domain knowledge
  pipeline.py            run_pipeline() — the single source of the end-to-end flow
  runtime.py             RunContext — resolved settings + config + run_date for one run
  constants.py errors.py config.py   shared names · typed errors · settings/config
  compute · etl · schemas · ml · store · monitoring · notify · dagster_defs
api/                   FastAPI online-serving app
dashboard/             optional Streamlit control panel (launch · configure · observe)
examples/buyer_stage/  the worked example — domain config + data loader + synthetic fallback ONLY
config/                generic pipeline config template
tests/                 unit + full end-to-end + Dagster-materialize (run on SQLite, no servers)
docs/                  ARCHITECTURE (deep plan) · ADAPTING (BYO domain) · LICENSES (audit)
infra/                 optional OpenTofu (VM) + Helm (K8s) deploy stubs
start.sh · stop.sh     one-command Docker up/down (with health checks + URLs)
```

## Quickstart

### Fastest — one command (Docker)

```bash
./start.sh --demo     # build + start the full stack, seed data, run the pipeline once
```

Then open the **dashboard at http://localhost:8501**. Stop with `./stop.sh` (add `--reset` to also
wipe the data volumes). `./start.sh` (no flag) just brings the stack up; `--build` forces a rebuild.

| Service | URL | What |
|---|---|---|
| **Dashboard** | http://localhost:8501 | launch runs, tune config, watch model + predictions + health |
| Dagster | http://localhost:3000 | orchestration UI (asset graph, schedules, runs) |
| MLflow | http://localhost:5000 | experiment tracking + model registry |
| FastAPI | http://localhost:8000/docs | serving API (`GET /predict/{entity_id}`) |

### Zero-Docker (SQLite only — fastest inner loop, what CI runs)

```bash
pip install -e ".[dev,dashboard]"
pytest                                     # full suite on SQLite + SQLite-backed MLflow
python -m examples.buyer_stage.seed_raw    # seed (synthetic fallback if CSVs absent)
automl-run                                 # train → score → load the local serving store
streamlit run dashboard/app.py             # optional: the dashboard on :8501
```

The pipeline runs end-to-end (train cold-start → register/promote champion → batch-score → atomic
swap into the serving store → serve), on SQLite with **zero external services**.

### Make targets

| Command | Does |
|---|---|
| `make up` / `make down` | start / stop the `core` stack (Dagster UI on :3000) |
| `make seed EXAMPLE=buyer_stage` | load example data into the offline store |
| `make run` | run the full pipeline once (CLI) |
| `make materialize` | run the same flow through the Dagster job (graph + checks) |
| `make serve` | (re)start the FastAPI service |
| `make dashboard` | start the optional Streamlit control panel on :8501 ([dashboard/](dashboard/)) |
| `make test` / `make lint` | pytest / ruff + mypy |
| `make new-example NAME=foo` | scaffold a new example from the template |

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the full deep plan + architecture blueprint.
- [docs/ADAPTING.md](docs/ADAPTING.md) — bring-your-own-domain guide.
- [docs/LICENSES.md](docs/LICENSES.md) — full license audit + scale-out swaps.

## Contributing & community

- [CONTRIBUTING.md](CONTRIBUTING.md) — dev setup, the quality gate, and the framework/example rule
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) · [SECURITY.md](SECURITY.md) · [CHANGELOG.md](CHANGELOG.md)
- Install the local hooks once: `pre-commit install` (mirrors CI: ruff, mypy, pytest, secret/large-file guards)

## Status

**Alpha — runs end-to-end with a green test suite.** `pytest` exercises unit tests, a full
end-to-end run (train → register/promote → score → serve), and an in-process Dagster
materialization, all on SQLite with zero external services. Opt-in / phase-gated pieces remain
(Evidently drift compute, the data-landing sensor, and the DuckDB reshape path for domains with
raw windowed inputs). See the **Implementation status** note in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

[Apache-2.0](LICENSE). See [NOTICE](NOTICE) and the dependency audit in [docs/LICENSES.md](docs/LICENSES.md).
