# tabular-automl-template — developer entrypoints
# All targets run against the docker compose `core` profile by default.

COMPOSE ?= docker compose
EXAMPLE ?= buyer_stage

.PHONY: help up down logs seed run materialize serve dashboard test lint typecheck demo new-example regen-golden

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

up: ## Start the core stack (postgres x2, mlflow, dagster UI :3000, api :8000)
	$(COMPOSE) --profile core up -d --build

down: ## Stop the stack
	$(COMPOSE) down

logs: ## Tail logs
	$(COMPOSE) logs -f

seed: ## Load example sample data into the offline store (EXAMPLE=buyer_stage)
	$(COMPOSE) run --rm dagster python -m examples.$(EXAMPLE).seed_raw

run: ## Run the full pipeline once (train -> score -> load online store)
	$(COMPOSE) run --rm dagster automl-run

materialize: ## Materialize the same flow through the Dagster job (graph + asset checks)
	$(COMPOSE) run --rm dagster dagster job execute -j daily_pipeline -m automl_template.dagster_defs.definitions

serve: ## (Re)start the FastAPI service
	$(COMPOSE) --profile core up -d --build api

dashboard: ## Start the Streamlit control panel (compose `dashboard` profile, :8501)
	$(COMPOSE) --profile dashboard up -d --build dashboard

dashboard-local: ## Run the control panel locally (no Docker): pip install -e ".[dashboard]" first
	streamlit run dashboard/app.py

test: ## Run the test suite
	$(COMPOSE) run --rm dagster pytest

lint: ## ruff lint
	$(COMPOSE) run --rm dagster ruff check .

typecheck: ## mypy
	$(COMPOSE) run --rm dagster mypy

demo: up ## End-to-end demo: up + seed + run + smoke-test /predict
	$(MAKE) seed EXAMPLE=$(EXAMPLE)
	$(MAKE) run
	@echo ">> smoke-testing GET /predict ..."
	$(COMPOSE) run --rm dagster python scripts/smoke_predict.py

new-example: ## Scaffold a new example: make new-example NAME=churn TARGET=churned
	@test -n "$(NAME)" || (echo "NAME is required" && exit 1)
	python scripts/new_example.py --name "$(NAME)" --target "$(TARGET)"

regen-golden: ## Regenerate the checked-in golden parity fixture (requires PR review)
	$(COMPOSE) run --rm dagster python scripts/regen_golden.py --example $(EXAMPLE)
