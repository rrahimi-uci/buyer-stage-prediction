"""Demo smoke test: pick a seeded entity, call the serving API, assert a valid prediction.

Run by ``make demo`` after the pipeline loads the online store. Reads a real entity id from the
seeded matrix (works for synthetic or real data), then hits the API. Exits non-zero on failure.
"""

from __future__ import annotations

import os
import sys
import time

import httpx

from automl_template.config import get_settings, load_pipeline
from automl_template.store import offline

API = os.environ.get("API_URL", "http://api:8000")


def _an_entity_id(settings, cfg) -> str:
    run_date = settings.resolved_run_date()
    name = "test" if offline.has_matrix(run_date, settings.data_dir, "test") else "train"
    matrix = offline.read_matrix(run_date, settings.data_dir, name)
    return str(matrix.get_column(cfg.entity_key)[0])


def main() -> int:
    settings = get_settings()
    cfg = load_pipeline(settings)
    entity_id = _an_entity_id(settings, cfg)

    last = ""
    for _ in range(10):  # tolerate a cold-starting uvicorn
        try:
            resp = httpx.get(f"{API}/predict/{entity_id}", timeout=10)
            if resp.status_code == 200:
                body = resp.json()
                if body.get("predicted_class") in cfg.labels:
                    cls, ver = body["predicted_class"], body["model_version"]
                    print(f"OK: {entity_id} -> {cls} (model {ver})")
                    return 0
                last = f"predicted_class {body.get('predicted_class')!r} not in {cfg.labels}"
            else:
                last = f"status {resp.status_code}: {resp.text}"
        except Exception as exc:  # noqa: BLE001
            last = f"request failed: {exc}"
        time.sleep(2)
    print(f"FAIL: {last}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
