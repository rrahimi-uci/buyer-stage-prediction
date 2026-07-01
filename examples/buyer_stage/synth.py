"""Synthetic buyer-stage-shaped data.

Lets the demo and the test suite run with ZERO proprietary data: the real CSVs are not
committed (provenance), so when they're absent ``seed_raw`` falls back to this generator, and
the end-to-end test uses it directly. Features are non-negative numeric (no ``dwell_time`` names,
so they trivially satisfy the validation checks) and the label has learnable signal + balance.
"""

from __future__ import annotations

import numpy as np
import polars as pl

DEFAULT_LABELS = ["Active Searcher", "Ready to Transact", "Dreamer", "Casual Explorer"]


def make_synthetic(
    n_train: int = 600,
    n_test: int = 200,
    n_features: int = 24,
    labels: list[str] | None = None,
    entity_key: str = "member_id",
    target: str = "buyer_stage",
    seed: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (train_df, test_df), each [entity_key, feat_000..., target]."""
    labels = labels or DEFAULT_LABELS
    rng = np.random.default_rng(seed)
    feat_cols = [f"feat_{i:03d}" for i in range(n_features)]

    def frame(n: int, prefix: str) -> pl.DataFrame:
        X = rng.integers(0, 50, size=(n, n_features)).astype("float64")
        # Learnable, balanced label: bucket a linear score into len(labels) quantiles.
        score = X[:, 0] * 1.0 + X[:, 1] * 0.5 - X[:, 2] * 0.7 + rng.normal(0, 3, n)
        edges = np.quantile(score, np.linspace(0, 1, len(labels) + 1)[1:-1])
        idx = np.digitize(score, edges)
        y = [labels[i] for i in idx]
        data = {entity_key: [f"{prefix}_{i}" for i in range(n)]}
        data.update({c: X[:, j] for j, c in enumerate(feat_cols)})
        data[target] = y
        return pl.DataFrame(data)

    return frame(n_train, "train"), frame(n_test, "test")


__all__ = ["make_synthetic", "DEFAULT_LABELS"]
