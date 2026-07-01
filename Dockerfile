# Multi-stage image shared by the `dagster` and `api` services.
# Phase-0 skeleton: builds the framework package and installs the core deps.

FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

# System deps: curl for healthchecks; libgomp1 is the OpenMP runtime LightGBM/XGBoost
# (pulled in by flaml[automl]) load at import time. DuckDB/Polars/pg8000 ship wheels / are pure-python.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---- dependency layer (cached) ----
COPY pyproject.toml ./
COPY src/automl_template/__init__.py src/automl_template/__init__.py
RUN pip install --upgrade pip hatchling && pip install -e ".[dev]"

# ---- source ----
COPY . .

# ---- dagster target: webserver + daemon + code location ----
FROM base AS dagster
ENV DAGSTER_HOME=/app/.dagster
RUN mkdir -p $DAGSTER_HOME
EXPOSE 3000
CMD ["dagster", "dev", "-m", "automl_template.dagster_defs.definitions", "-h", "0.0.0.0", "-p", "3000"]

# ---- api target: FastAPI serving ----
FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
