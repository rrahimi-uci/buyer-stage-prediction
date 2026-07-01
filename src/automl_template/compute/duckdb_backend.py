"""DuckDB compute backend (DEFAULT).

Replaces the AWS EMR/Spark + Athena estate. At ~5k rows the entire join+reshape runs
in-process in milliseconds. The 5 ``consp_member_id_summary_t0NN`` external tables of the
original become DuckDB relations registered from Parquet/CSV or in-memory frames.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl


class DuckDBBackend:
    def __init__(self) -> None:
        self.con = duckdb.connect(database=":memory:")

    def register_parquet(self, name: str, path: str | Path) -> None:
        self.con.execute(
            f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet(?)",
            [str(path)],
        )

    def register_csv(self, name: str, path: str | Path) -> None:
        self.con.execute(
            f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_csv_auto(?, header=true)",
            [str(path)],
        )

    def register_dataframe(self, name: str, df: pl.DataFrame) -> None:
        # DuckDB reads Polars via Arrow zero-copy; register it as a named relation.
        self.con.register(name, df.to_arrow())

    def sql(self, query: str) -> pl.DataFrame:
        return self.con.execute(query).pl()

    def close(self) -> None:
        self.con.close()
