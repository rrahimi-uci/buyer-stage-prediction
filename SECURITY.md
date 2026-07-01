# Security Policy

## Reporting a Vulnerability

**Please do not open a public issue for security vulnerabilities.**

Instead, use GitHub's private vulnerability reporting (Security → *Report a vulnerability*) or
email the maintainers privately. Include:

- a description of the issue and its impact,
- steps to reproduce (a minimal proof of concept if possible),
- affected version / commit.

We aim to acknowledge reports within a few business days and to provide a remediation timeline
after triage.

## Scope notes for this project

- This is a **template/reference** project. Before deploying it, review the items in
  [docs/ARCHITECTURE.md §8](docs/ARCHITECTURE.md) (secrets handling, least-privilege DB roles).
- Default credentials in `docker-compose.yml` and `.env.example` are **for local development
  only**. Never use them in any reachable environment. Inject real secrets via your platform's
  secret manager (Docker/Swarm `*_FILE`, etc.); never bake them into an image.
- The `examples/buyer_stage/sample_data/` CSVs are **not** committed (data provenance). Do not
  add real or proprietary data to the repo.

## Supported Versions

This project is pre-1.0; only the latest `main` is supported. Pin a commit/tag for production use.
