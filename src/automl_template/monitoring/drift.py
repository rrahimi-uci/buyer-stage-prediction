"""Drift detection with Evidently.

Compares the current feature matrix against a reference partition (default: the partition
the current champion trained on). Emits an HTML report artifact and a single
``dataset_drift_share`` score that the ``train_decision`` policy consults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass
class DriftResult:
    dataset_drift_share: float
    report_html_path: str | None


def compute_drift(
    current: pl.DataFrame,
    reference: pl.DataFrame,
    out_dir: str | Path = "data/drift",
) -> DriftResult:
    """Run Evidently's data-drift preset and return the overall drift share."""
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference.to_pandas(), current_data=current.to_pandas())

    # DataDriftPreset expands to several metrics; do NOT assume index 0 carries the share
    # (silently defaulting to 0.0 would make the retrain gate never fire). Scan all metrics.
    result = report.as_dict()
    share: float | None = None
    for metric in result.get("metrics", []):
        res = metric.get("result", {})
        if "share_of_drifted_columns" in res:
            share = float(res["share_of_drifted_columns"])
            break
    if share is None:
        raise ValueError(
            "Evidently DataDriftPreset returned no 'share_of_drifted_columns' "
            "(API drift? pinned to evidently==0.4.x)."
        )
    html_path = out / "drift_report.html"
    report.save_html(str(html_path))
    return DriftResult(dataset_drift_share=share, report_html_path=str(html_path))


__all__ = ["DriftResult", "compute_drift"]
