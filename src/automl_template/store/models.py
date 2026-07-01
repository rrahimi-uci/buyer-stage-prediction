"""Portable SQLAlchemy schema for the online prediction store.

Defined as Core metadata (not raw SQL) so the SAME schema runs on SQLite (local/tests, zero
servers) and PostgreSQL (docker compose). ``JSON`` maps to JSONB on Postgres and TEXT-JSON on
SQLite; ``DateTime`` is portable. The app calls ``ensure_schema`` (idempotent ``create_all``),
so no hand-maintained ``init.sql`` is required.
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import JSON, Column, DateTime, MetaData, String, Table, create_engine
from sqlalchemy.engine import Engine

metadata = MetaData()

# Online predictions, point-read by entity_id from the FastAPI service.
# entity_id == the original DynamoDB hash key; ttl preserves DDB eviction semantics.
predictions_online = Table(
    "predictions_online",
    metadata,
    Column("entity_id", String, primary_key=True),
    Column("predicted_class", String, nullable=False),
    Column("class_probabilities", JSON, nullable=True),
    Column("model_version", String, nullable=False),
    Column("scored_at", DateTime(timezone=True), nullable=False),
    Column("ttl", DateTime(timezone=True), nullable=True),
)


@lru_cache
def make_engine(url: str) -> Engine:
    """Create (and cache) a SQLAlchemy engine for a serving DB URL."""
    # check_same_thread=False lets FastAPI's threadpool share a SQLite engine.
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, future=True, connect_args=connect_args)


def ensure_schema(engine: Engine) -> None:
    """Create tables if absent (idempotent). Replaces the mounted init.sql."""
    metadata.create_all(engine)


__all__ = ["metadata", "predictions_online", "make_engine", "ensure_schema"]
