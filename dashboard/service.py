"""Control plane for the dashboard — the logic behind every button, no UI framework here.

Everything the Streamlit app does routes through these functions so they can be unit-tested
without a browser and reused by any other front end. Design choices:

* **Runs execute out-of-process.** A click spawns a fresh ``python -c "... run()"`` that drives
  the exact same ``automl_template.cli.run`` entrypoint as ``automl-run``. Isolation is the point:
  the child reads env/YAML fresh (no ``get_settings`` lru_cache to bust), a long FLAML train never
  blocks the Streamlit event loop, logs stream to a file we can tail across reruns, and the exit
  code + printed summary tell us how it went. This works identically zero-server (SQLite) and in
  compose (Postgres) — no Dagster server required.
* **Reads are direct.** Champion info comes from the MLflow registry; the serving snapshot from
  ``OnlineStore.summary``; health from cheap probes. All are defensive: a missing champion, an
  empty store, or an unreachable service returns a typed "not ready" state, never an exception.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

from automl_template.config import PipelineConfig, Settings, get_settings
from automl_template.constants import MODEL_NAME

RUNS_SUBDIR = "dashboard/runs"
_SUMMARY_RE = re.compile(r"^\[automl-run\]\s+(.*)$", re.MULTILINE)


# --------------------------------------------------------------------------- #
# Config: read the active domain YAML, validate edits, write back             #
# --------------------------------------------------------------------------- #
def active_config_path(settings: Settings | None = None) -> Path:
    """Absolute path to the active example's ``pipeline.yaml`` (from ``PIPELINE_CONFIG``)."""
    settings = settings or get_settings()
    return Path(settings.pipeline_config).resolve()


def load_config_raw(settings: Settings | None = None) -> dict:
    """The active pipeline YAML as a plain dict (preserves keys the form doesn't expose)."""
    return yaml.safe_load(active_config_path(settings).read_text()) or {}


def validate_config(data: dict) -> tuple[bool, str]:
    """Typecheck an edited config dict through ``PipelineConfig`` without writing it."""
    try:
        PipelineConfig.model_validate(data)
        return True, "valid"
    except Exception as exc:  # noqa: BLE001 - surface pydantic's message verbatim to the UI
        return False, str(exc)


def save_config(data: dict, settings: Settings | None = None) -> None:
    """Validate then atomically write the config back to the active YAML.

    Raises ``ValueError`` on invalid input so a bad edit can never corrupt the domain file that
    the very next run will read.
    """
    ok, msg = validate_config(data)
    if not ok:
        raise ValueError(msg)
    path = active_config_path(settings)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(data, sort_keys=False))
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Runs: launch out-of-process, tail logs, parse the summary                    #
# --------------------------------------------------------------------------- #
@dataclass
class RunHandle:
    """A launched pipeline run. Serializable metadata + a live ``Popen`` handle."""

    run_id: str
    run_date: str
    log_path: Path
    started_at: str
    proc: subprocess.Popen | None = field(default=None, repr=False)
    returncode: int | None = None

    @property
    def is_running(self) -> bool:
        if self.proc is None:
            return False
        code = self.proc.poll()
        if code is not None:
            self.returncode = code
        return code is None

    def read_log(self, tail: int | None = None) -> str:
        if not self.log_path.exists():
            return ""
        lines = self.log_path.read_text(errors="replace").splitlines()
        if tail is not None:
            lines = lines[-tail:]
        return "\n".join(lines)


def _runs_dir(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    d = Path(settings.data_dir) / RUNS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def launch_run(
    run_date: str | None = None,
    settings: Settings | None = None,
    _now: datetime | None = None,
) -> RunHandle:
    """Spawn a detached pipeline run that streams to a per-run log file.

    ``run_date`` (and the active ``PIPELINE_CONFIG``) are passed through the child's environment,
    so a fresh process resolves settings and YAML with no cache to invalidate.
    """
    settings = settings or get_settings()
    now = _now or datetime.now(UTC)
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    log_path = _runs_dir(settings) / f"{run_id}.log"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # flush child logs to the file promptly for live tailing
    env["PIPELINE_CONFIG"] = str(active_config_path(settings))
    if run_date:
        env["RUN_DATE"] = run_date

    # -c so we don't depend on the `automl-run` console script being on PATH, and so we can turn
    # on INFO logging (the pipeline logs decisions but never configures a handler itself).
    code = (
        "import logging;"
        "logging.basicConfig(level=logging.INFO,"
        "format='%(asctime)s %(levelname)s %(name)s: %(message)s');"
        "from automl_template.cli import run; run()"
    )
    log_file = log_path.open("w")
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=Path.cwd(),
    )
    return RunHandle(
        run_id=run_id,
        run_date=run_date or settings.resolved_run_date(),
        log_path=log_path,
        started_at=now.isoformat(timespec="seconds"),
        proc=proc,
    )


def parse_summary(log_text: str) -> dict | None:
    """Pull the ``[automl-run] key=value ...`` summary line into a dict, or None if absent."""
    matches = _SUMMARY_RE.findall(log_text)
    if not matches:
        return None
    fields = dict(re.findall(r"(\w+)=(\[.*?\]|\S+)", matches[-1]))

    def _num(v: str) -> object:
        if v in ("None", ""):
            return None
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v

    return {
        "run_date": fields.get("run_date"),
        "retrained": fields.get("retrained") == "True",
        "model_version": None if fields.get("model") in (None, "None") else fields.get("model"),
        "rows_scored": _num(fields.get("rows_scored", "0")),
        "macro_f1": _num(fields.get("macro_f1", "None")),
        "problems": fields.get("problems", "[]"),
    }


def list_past_runs(settings: Settings | None = None, limit: int = 20) -> list[dict]:
    """Most-recent-first log files with their parsed summary (for a run-history table)."""
    logs = sorted(_runs_dir(settings).glob("*.log"), reverse=True)[:limit]
    out: list[dict] = []
    for log in logs:
        summary = parse_summary(log.read_text(errors="replace")) or {}
        out.append({"run_id": log.stem, "log_path": str(log), **summary})
    return out


# --------------------------------------------------------------------------- #
# Reads: champion, serving snapshot, service health                           #
# --------------------------------------------------------------------------- #
def champion_info(settings: Settings | None = None) -> dict:
    """Current champion (version / macro_f1 / age), or ``{'available': False}`` on cold-start."""
    from automl_template.config import configure_mlflow
    from automl_template.ml import registry

    try:
        configure_mlflow(settings)
        champ = registry.get_champion(MODEL_NAME)
    except Exception as exc:  # noqa: BLE001 - registry unreachable is a UI state, not a crash
        return {"available": False, "error": str(exc)}
    if champ is None:
        return {"available": False}
    return {
        "available": True,
        "version": champ.version,
        "macro_f1": champ.macro_f1,
        "age_days": champ.age_days,
    }


def predictions_summary(settings: Settings | None = None) -> dict:
    """Serving snapshot + consistency verdict, or ``{'available': False}`` if unreachable."""
    from automl_template.store.online import OnlineStore

    settings = settings or get_settings()
    try:
        store = OnlineStore(settings)
        snap = store.summary()
        snap["problems"] = store.check_consistency()
        snap["available"] = True
        return snap
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}


def lookup_prediction(entity_id: str, settings: Settings | None = None) -> dict | None:
    """Point lookup mirroring the FastAPI ``/predict/{entity_id}`` endpoint."""
    from automl_template.store.online import OnlineStore

    return OnlineStore(settings or get_settings()).get_prediction(entity_id)


def _http_ok(url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - operator-set URL
            return 200 <= resp.status < 500  # any response == the service is up
    except (urllib.error.URLError, OSError, ValueError):
        return False


def service_health(settings: Settings | None = None) -> list[dict]:
    """Status tiles for the moving parts. Everything degrades gracefully when absent."""
    settings = settings or get_settings()
    tiles: list[dict] = []

    # Serving DB: reachable iff we can list model versions.
    try:
        from automl_template.store.online import OnlineStore

        OnlineStore(settings).active_model_versions()
        tiles.append({"name": "serving-db", "ok": True, "detail": settings.serving_db_url})
    except Exception as exc:  # noqa: BLE001
        tiles.append({"name": "serving-db", "ok": False, "detail": str(exc)})

    # MLflow registry: reachable iff the champion probe doesn't raise.
    champ = champion_info(settings)
    tiles.append(
        {
            "name": "mlflow",
            "ok": "error" not in champ,
            "detail": settings.mlflow_tracking_uri,
        }
    )

    # Optional HTTP surfaces (compose): only probed when their URL env is set.
    for name, env_key, health_path in (
        ("api", "API_URL", "/healthz"),
        ("dagster", "DAGSTER_URL", ""),
        ("mlflow-ui", "MLFLOW_UI_URL", ""),
    ):
        base = os.environ.get(env_key)
        if base:
            tiles.append(
                {"name": name, "ok": _http_ok(base.rstrip("/") + health_path), "detail": base}
            )
    return tiles


def seed_example(example: str = "buyer_stage") -> RunHandle:
    """Run an example's seed loader out-of-process (same pattern as ``launch_run``)."""
    now = datetime.now(UTC)
    run_id = f"seed-{now.strftime('%Y%m%dT%H%M%SZ')}"
    log_path = _runs_dir() / f"{run_id}.log"
    log_file = log_path.open("w")
    proc = subprocess.Popen(
        [sys.executable, "-m", f"examples.{example}.seed_raw"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        cwd=Path.cwd(),
    )
    return RunHandle(
        run_id=run_id,
        run_date="-",
        log_path=log_path,
        started_at=now.isoformat(timespec="seconds"),
        proc=proc,
    )


__all__ = [
    "RunHandle",
    "active_config_path",
    "champion_info",
    "launch_run",
    "list_past_runs",
    "load_config_raw",
    "lookup_prediction",
    "parse_summary",
    "predictions_summary",
    "save_config",
    "seed_example",
    "service_health",
    "validate_config",
]
