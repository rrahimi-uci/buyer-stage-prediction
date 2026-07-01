"""ComputeBackend protocol — the seam that lets DuckDB (default) be swapped for Spark.

A backend's only job is: register the pre-windowed input tables, run a SQL string,
and return a Polars DataFrame. The reshape SQL itself is backend-agnostic ANSI-ish SQL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import polars as pl


class ComputeBackend(Protocol):
    def register_parquet(self, name: str, path: str | Path) -> None:
        """Register a (pre-windowed) input table under ``name``."""
        ...

    def register_dataframe(self, name: str, df: pl.DataFrame) -> None:
        """Register an in-memory frame as a queryable relation."""
        ...

    def sql(self, query: str) -> pl.DataFrame:
        """Execute ``query`` and return the result as a Polars DataFrame."""
        ...
