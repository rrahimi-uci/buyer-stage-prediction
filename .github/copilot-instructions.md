# GitHub Copilot repository instructions

- You may work on assigned issues and development tasks, create a branch, and open a pull request. You must never merge a pull request, approve a pull request, push to `main`, or represent your review as human approval.
- Before opening or marking a pull request ready for human review, understand the issue acceptance criteria, implement the smallest complete change, and add or update tests for changed behavior.
- Run the repository CI-equivalent validation for affected areas: install `.[dev]`, then run `ruff check .`, `ruff format --check .`, `mypy`, `pytest --cov=automl_template`, and `test ! -f .env`. Run the golden-parity, license-audit, and Docker/demo checks when their areas are affected. Also run `git diff --check`.
- Keep validation deterministic and offline. Do not require credentials, network services, model calls, or uncommitted local data for ordinary tests.
- Do not submit a PR as ready if a required test, lint check, type check, build, or security check fails. If validation is blocked, keep the PR draft and state the exact blocker and command output in the PR body.
- The PR body must list changed behavior, tests added or updated, every validation command and result, limitations, risks, and deployment or documentation implications. If a change is documentation-only, explain which documentation/link/build checks replaced unit tests.
- During code review, identify correctness, security, test-coverage, and maintainability issues and leave comments. Never approve or merge the PR.
