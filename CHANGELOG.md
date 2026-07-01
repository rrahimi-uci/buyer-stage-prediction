# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial scaffold of `tabular-automl-template`: a domain-agnostic, single-node,
  docker-compose-portable tabular AutoML batch-scoring-to-online-store pipeline.
- Dagster job (assets + asset checks + schedule + failure sensor) replacing the original AWS
  Step Functions state machine.
- DuckDB + Polars feature reshape; FLAML AutoML (macro-F1); MLflow tracking + registry with a
  champion/challenger promotion gate; Postgres online store with a transactional swap; FastAPI
  serving; Pandera validation; Evidently drift; Apprise notifications.
- Worked `buyer_stage` example (config + data loader only; no framework domain knowledge).
- OSS community files: CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, issue/PR templates, pre-commit,
  EditorConfig, Dependabot, and a permissive-only license audit (`docs/LICENSES.md`).

### Notes
- Runs end-to-end with a green test suite (unit + full end-to-end + Dagster materialize) on
  SQLite with zero external services. Opt-in / phase-gated pieces remain (Evidently drift compute,
  the data-landing sensor, and the DuckDB reshape path for raw-windowed-input domains). See the
  implementation-status note in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

[Unreleased]: https://github.com/OWNER/tabular-automl-template/commits/main
