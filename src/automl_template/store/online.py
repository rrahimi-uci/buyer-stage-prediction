"""Online prediction store. Replaces DynamoDB + the bs_push2ddb_glue.py per-row put_item.

A small cohesive object that owns the serving DB connection. Properties (ARCHITECTURE §4):
  * Full-table SWAP in a single transaction (DELETE-all + INSERT) — readers never observe a
    half-loaded mix of model versions; after a run the store holds exactly this run's rows under
    one uniform model_version (which ``check_consistency`` asserts).
  * Portable across SQLite (local/tests) and Postgres (compose) via SQLAlchemy Core.

The engine is cached per URL and the schema is ensured once per store, rather than on every call.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
from sqlalchemy import delete, func, insert, select
from sqlalchemy.engine import Engine

from automl_template.config import PipelineConfig, Settings
from automl_template.store.models import ensure_schema, make_engine, predictions_online


class OnlineStore:
    """Read/write access to ``predictions_online`` for one serving database."""

    def __init__(self, settings: Settings) -> None:
        self._engine: Engine = make_engine(settings.serving_db_url)
        ensure_schema(self._engine)

    def load_predictions(
        self, predictions: pl.DataFrame, model_version: str, cfg: PipelineConfig
    ) -> int:
        """Atomically swap the run's predictions into the live table. Returns rows loaded.

        ``predictions`` must carry the entity key + ``predicted_class`` (and optionally
        ``class_probabilities``). The daily batch re-scores the full entity set, matching the
        original pipeline; an empty batch is a no-op (never wipes live data).
        """
        if predictions.is_empty():
            return 0

        # One row per entity (the table is keyed by entity_id). An entity may appear multiple
        # times in a scored batch (e.g. several snapshots of the same member); keep the last,
        # mirroring the original DynamoDB put_item last-write-wins semantics.
        predictions = predictions.unique(subset=[cfg.entity_key], keep="last")

        now = datetime.now(UTC)
        ttl = now + timedelta(days=cfg.online_store.ttl_days)
        has_proba = "class_probabilities" in predictions.columns
        rows = [
            {
                "entity_id": str(r[cfg.entity_key]),
                "predicted_class": str(r["predicted_class"]),
                "class_probabilities": r["class_probabilities"] if has_proba else None,
                "model_version": model_version,
                "scored_at": now,
                "ttl": ttl,
            }
            for r in predictions.iter_rows(named=True)
        ]

        with self._engine.begin() as conn:  # single transaction == the atomic swap
            conn.execute(delete(predictions_online))
            conn.execute(insert(predictions_online), rows)
        return len(rows)

    def check_consistency(self) -> list[str]:
        """Verify the post-swap store: rows exist and all carry a single model_version."""
        with self._engine.connect() as conn:
            total = conn.execute(select(func.count()).select_from(predictions_online)).scalar_one()
            distinct = conn.execute(
                select(func.count(func.distinct(predictions_online.c.model_version)))
            ).scalar_one()
        problems: list[str] = []
        if total == 0:
            problems.append("predictions_online is empty after load")
        if distinct and distinct > 1:
            problems.append(f"predictions_online has {distinct} model_versions (expected 1)")
        return problems

    def get_prediction(self, entity_id: str) -> dict | None:
        """Point lookup used by the serving API."""
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    select(predictions_online).where(predictions_online.c.entity_id == entity_id)
                )
                .mappings()
                .first()
            )
        return dict(row) if row else None

    def active_model_versions(self) -> list[str]:
        with self._engine.connect() as conn:
            return list(
                conn.execute(select(predictions_online.c.model_version).distinct()).scalars().all()
            )

    def summary(self) -> dict:
        """Read-only serving snapshot for dashboards/ops: rowcount, versions, class mix.

        Cheap aggregate queries over ``predictions_online`` — a friendly companion to
        ``check_consistency`` that returns *what* is served rather than a pass/fail list.
        """
        with self._engine.connect() as conn:
            total = conn.execute(select(func.count()).select_from(predictions_online)).scalar_one()
            versions = list(
                conn.execute(select(predictions_online.c.model_version).distinct()).scalars().all()
            )
            class_rows = conn.execute(
                select(predictions_online.c.predicted_class, func.count()).group_by(
                    predictions_online.c.predicted_class
                )
            ).all()
            last_scored = conn.execute(
                select(func.max(predictions_online.c.scored_at))
            ).scalar_one()
        return {
            "total": int(total),
            "model_versions": versions,
            "class_counts": {str(cls): int(n) for cls, n in class_rows},
            "last_scored_at": last_scored,
        }


__all__ = ["OnlineStore"]
