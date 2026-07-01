# infra/helm — Kubernetes deploy (stub)

Optional scale-out path beyond the single VM (`infra/opentofu`). A Helm chart for the `core`
services (Dagster, MLflow, two Postgres or one with schemas, FastAPI) would live here.

**Status:** not yet implemented (phase-6). The single-node `docker compose` stack is the
supported deployment today; see [docs/ARCHITECTURE.md §7](../../docs/ARCHITECTURE.md) and
[docs/ADAPTING.md](../../docs/ADAPTING.md).

Contributions welcome — keep images pinned and permissively licensed (no Grafana/AGPL by
default; see [docs/LICENSES.md](../../docs/LICENSES.md)).
