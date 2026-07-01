"""FastAPI serving — the explicit, portable online-store consumer.

Reads ``predictions_online`` from the serving DB (SQLite locally, Postgres in compose) through
an ``OnlineStore`` built once in the lifespan and held on ``app.state``. Least privilege: this
process only needs the serving DB URL.

Endpoints:
  GET /predict/{entity_id}  -> the stored prediction
  GET /model/active         -> serving model_version(s)
  GET /healthz              -> DB connectivity
  GET /metrics              -> Prometheus
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

from automl_template.config import get_settings
from automl_template.store.online import OnlineStore

PREDICT_REQUESTS = Counter("predict_requests_total", "Prediction lookups", ["found"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the store once (engine + schema). get_settings() is resolved here, not at import,
    # so the process env (and test isolation) is honored.
    app.state.store = OnlineStore(get_settings())
    yield


app = FastAPI(title="tabular-automl-template serving", version="0.1.0", lifespan=lifespan)


def _store(request: Request) -> OnlineStore:
    return request.app.state.store


@app.get("/predict/{entity_id}")
def predict(entity_id: str, request: Request) -> dict:
    row = _store(request).get_prediction(entity_id)
    if not row:
        PREDICT_REQUESTS.labels(found="false").inc()
        raise HTTPException(status_code=404, detail=f"no prediction for entity_id={entity_id}")
    PREDICT_REQUESTS.labels(found="true").inc()
    return row


@app.get("/model/active")
def active_model(request: Request) -> dict:
    return {"model_versions": _store(request).active_model_versions()}


@app.get("/healthz")
def healthz(request: Request) -> dict:
    try:
        _store(request).active_model_versions()  # cheap query == connectivity probe
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
