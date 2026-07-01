"""Offline feature matrix IO. Parquet on a local volume is the source of truth.

Replaces the S3 CSV matrix. Partitioned by ``run_date`` (a path at this scale). A partition may
hold several named matrices (e.g. ``train`` / ``test`` / ``score``).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def partition_dir(run_date: str, root: Path) -> Path:
    return root / "feature_matrix" / f"run_date={run_date}"


def write_matrix(df: pl.DataFrame, run_date: str, root: Path, name: str = "part") -> Path:
    out = partition_dir(run_date, root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.parquet"
    df.write_parquet(path)
    return path


def read_matrix(run_date: str, root: Path, name: str = "part") -> pl.DataFrame:
    return pl.read_parquet(partition_dir(run_date, root) / f"{name}.parquet")


def has_matrix(run_date: str, root: Path, name: str = "part") -> bool:
    return (partition_dir(run_date, root) / f"{name}.parquet").exists()


def latest_run_date(root: Path) -> str | None:
    base = root / "feature_matrix"
    if not base.exists():
        return None
    parts = sorted(p.name.split("=", 1)[1] for p in base.glob("run_date=*"))
    return parts[-1] if parts else None


__all__ = ["partition_dir", "write_matrix", "read_matrix", "has_matrix", "latest_run_date"]
