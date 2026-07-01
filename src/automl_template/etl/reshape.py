"""Reshape: port of ``buyer_intent_training_set_..._outlier_filter.sql`` (the 401-line CTAS).

This is a CONCRETE, spec-driven reshape (per ARCHITECTURE §5 / MustFix #2). It does NOT
aggregate raw events — the original ``consp_member_id_summary_t0NN`` files are external-table
DDL and windows arrive pre-materialized. The only transformation is the per-window
join + one-hot + coalesce-default + dwell-clip + ``_00N`` suffixing.

``build_feature_matrix`` is DESIGNED as the single shared reshape entrypoint for both the
training and the batch-scoring assets; wiring both call sites to it (so training and serving
provably share one code path) is a phase-1 TODO. Until then, the concrete skew guard is
``shaping.apply_canonical_order`` (fills any absent column with a typed 0 in a fixed order).
"""

from __future__ import annotations

import polars as pl

from automl_template.compute.duckdb_backend import DuckDBBackend
from automl_template.config import FeatureSpec, PipelineConfig

# Identity/date columns produced once via COALESCE across windows (see original
# sql_query_training_set). Everything else is per-window and suffixed.
_KEY_COLS = ("member_id", "snapshot_date_mst_yyyymmdd")


def _per_window_select(table: str, window: str, spec: FeatureSpec, key_col: str) -> str:
    """Build the per-window projection for one input table.

    Mirrors ``sql_query_base`` in the original constants.py: one-hot CASE expansion,
    COALESCE-to-default for counts, and the >180 -> 180 dwell clip. Feature columns are
    suffixed with ``_<window>`` (e.g. ``total_searches_001``).

    The entity key is aliased to its bare name (NOT suffixed) so the cross-window join can
    use ``USING (<key>)`` and collapse to exactly one row per entity — including entities
    absent from the base window. The snapshot column stays suffixed and is coalesced later.
    """
    parts: list[str] = [
        f"{table}.{key_col} AS {key_col}",
        f"{table}.snapshot_date_mst_yyyymmdd AS snapshot_date_mst_yyyymmdd_{window}",
    ]

    # One-hot: for each categorical column, emit one 0/1 column per declared value.
    for col, values in spec.one_hot.items():
        for value in values:
            slug = value.replace(" ", "_")
            parts.append(
                f"(CASE WHEN trim({table}.{col}) = '{value}' THEN 1 ELSE 0 END) "
                f"AS {col}_{slug}_{window}"
            )

    # Clipped (dwell) columns: >max -> max, else coalesce-to-default.
    for col, max_val in spec.clip.items():
        default = spec.coalesce_default.get(col, 0)
        parts.append(
            f"(CASE WHEN COALESCE({table}.{col}, {default}) > {max_val} THEN {max_val} "
            f"ELSE COALESCE({table}.{col}, {default}) END) AS {col}_{window}"
        )

    # Plain coalesce-to-default numeric columns.
    for col, default in spec.coalesce_default.items():
        if col in spec.clip:
            continue  # already emitted above
        parts.append(f"COALESCE({table}.{col}, {default}) AS {col}_{window}")

    return "SELECT\n    " + ",\n    ".join(parts) + f"\nFROM {table}"


def build_feature_matrix(
    backend: DuckDBBackend,
    cfg: PipelineConfig,
    spec: FeatureSpec,
    window_tables: dict[str, str],
) -> pl.DataFrame:
    """Join the per-window tables on member_id and emit the canonical feature matrix.

    ``window_tables`` maps window suffix -> registered relation name, e.g.
    ``{"001": "t001", "007": "t007", ...}``.

    Returns a Polars DataFrame keyed by ``entity_key`` with the canonical column order.
    Aligns to ARCHITECTURE §5: entity_key is carried but later excluded from features.
    """
    key = cfg.entity_key
    # 1. Per-window projections as CTEs.
    ctes = []
    for window in cfg.windows:
        table = window_tables[window]
        ctes.append(f"w{window} AS (\n{_per_window_select(table, window, spec, key)}\n)")

    # 2. FULL OUTER JOIN across windows USING the key. USING coalesces the join key into a
    #    SINGLE column, so an entity present in ANY window yields exactly one row (the prior
    #    ``ON base.key = other.key`` form dropped entities missing from the base window —
    #    their NULL base key matched nothing and fragmented them into partial rows).
    base, *rest = cfg.windows
    join_sql = f"w{base}"
    for window in rest:
        join_sql += f"\nFULL OUTER JOIN w{window} USING ({key})"

    # The key already collapses to one column via USING; coalesce the per-window snapshots
    # and EXCLUDE the per-window snapshot helpers so only features + key + snapshot survive.
    snapshot_cols = ", ".join(f"snapshot_date_mst_yyyymmdd_{w}" for w in cfg.windows)
    snapshot_excludes = ", ".join(f"snapshot_date_mst_yyyymmdd_{w}" for w in cfg.windows)

    query = f"""
    WITH {", ".join(ctes)}
    SELECT
        COALESCE({snapshot_cols}) AS snapshot_date_mst_yyyymmdd,
        * EXCLUDE ({snapshot_excludes})
    FROM {join_sql}
    """

    df = backend.sql(query)
    # TODO(phase-1): hand off to shaping.apply_canonical_order(df, order) for the frozen
    # column order, then shaping.split_features_and_key before training/scoring.
    return df


__all__ = ["build_feature_matrix"]
