# Contributing

Thanks for your interest in improving **tabular-automl-template**. This project aims to be a
clean, domain-agnostic reference for a tabular AutoML batch-scoring-to-online-store pipeline,
so contributions that keep the **framework / example** boundary crisp are especially welcome.

## Ground rules

- Framework code (`src/automl_template/`, `api/`, `db/`, `infra/`) must contain **zero domain
  knowledge**. Anything real-estate / buyer-stage specific belongs under `examples/buyer_stage/`.
- Keep the default `core` dependency set **permissively licensed** (Apache-2.0 / MIT / BSD /
  PostgreSQL). Anything copyleft (AGPL/SSPL/BSL/MPL) must be optional and documented in
  [docs/LICENSES.md](docs/LICENSES.md). The CI `license-audit` job enforces this.
- Never commit data, secrets, or model artifacts. `.gitignore` + the pre-commit
  `check-added-large-files` hook guard this, but stay alert.

## Dev setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

Or do everything in containers via the [Makefile](Makefile): `make up`, `make test`, `make lint`.

## Before you open a PR

Run the same gate CI runs:

```bash
ruff check .
ruff format --check .
mypy
pytest
```

- Add or update tests for any behavior change. The whole suite (unit + full end-to-end +
  Dagster materialize) runs on SQLite with **no Docker / no external services**.
- If you touch the buyer-stage reshape, regenerate and **review** the golden fixture:
  `make regen-golden EXAMPLE=buyer_stage` (the golden-parity test is meaningless if the golden
  is regenerated without review).
- Update [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) if you change the asset graph, contracts,
  or stack — keep the doc and code in sync (doc-vs-code drift is treated as a bug here).
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages
  (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).

## Adding a new example

See [docs/ADAPTING.md](docs/ADAPTING.md). In short: `make new-example NAME=foo TARGET=bar`,
edit the two YAMLs, wire `seed_raw.py` — **no framework edits**. If your example forces a
framework change, that's a signal the abstraction needs work; call it out in the PR.

## Reporting bugs / requesting features

Use the issue templates under [.github/ISSUE_TEMPLATE](.github/ISSUE_TEMPLATE). For security
issues, follow [SECURITY.md](SECURITY.md) instead of opening a public issue.

By contributing you agree your contributions are licensed under the project's
[Apache-2.0 License](LICENSE).
