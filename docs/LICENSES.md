# License audit

**Goal:** the default `core` install and `docker compose --profile core` stack are 100%
permissive (Apache-2.0 / MIT / BSD / PostgreSQL). Anything copyleft is opt-in and flagged.

## Default `core` — all permissive

Grouped by the **actual** SPDX license (no MIT/BSD conflation):

| License | Components |
|---|---|
| **Apache-2.0** | Dagster, dagster-webserver, MLflow, Evidently (pinned 0.4.40), pyarrow, prometheus-client, AutoGluon (opt), pyspark (opt), Docker / docker compose |
| **MIT** | DuckDB, Polars, FLAML, Pandera, pydantic, pydantic-settings, PyYAML, SQLAlchemy, FastAPI, pytest, ruff, mypy |
| **BSD-3-Clause** | pandas, scikit-learn, numpy, scipy, Uvicorn, Starlette, pg8000, httpx (dev), Valkey (scale-swap image) |
| **BSD-2-Clause** | Apprise |
| **PostgreSQL License** (BSD-style permissive) | PostgreSQL (`postgres:16`) |

All of the above are permissive and fully commercial-safe. BSD-2/3 and the PostgreSQL License
impose only attribution / no-endorsement — no more burdensome than MIT. `numpy`, `pandas`,
`scikit-learn`, `scipy`, `Uvicorn`, `Starlette` are BSD-3 foundational libraries with no
MIT/Apache equivalent and are **deliberately not swapped**.

`pg8000` (BSD-3) is chosen over `psycopg2`/`psycopg` (LGPL) specifically to keep the driver
permissive.

## Quarantined — opt-in only, NOT in `core`

| Component | License | Status |
|---|---|---|
| Grafana | AGPL-3.0 | `monitoring` profile, **commented out**; do not enable in a redistributed product |
| MinIO **server** | AGPL-3.0 | never used; only the MinIO **client** (Apache-2.0) for `s3://` IO |
| Redis 7.4+ | SSPL / RSAL | **avoided** — use Valkey (BSD-3) for the high-QPS path |
| Terraform | BSL | **avoided** — use OpenTofu (MPL-2.0) |
| OpenTofu | MPL-2.0 | optional `infra/` only; file-level copyleft — note before vendoring |

## Enforcement

CI (`license-audit` job) installs the **full resolved graph including optional profiles**
(`.[dev,autogluon,spark]`) and fails if any installed distribution reports AGPL / SSPL /
Business Source / BSL. Re-run locally:

```bash
pip install pip-licenses && pip-licenses --format=csv | grep -Ei 'AGPL|SSPL|Business Source|BSL'
```

## Data licensing (separate from code)

The buyer-stage `sample_data/` CSVs are de-identified real Move/Realtor.com behavioral data and
are **not committed**. For a publishable release, replace them with a synthetic or public dataset
of the same shape — a data-rights question independent of this code license.
