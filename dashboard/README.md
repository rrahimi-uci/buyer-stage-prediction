# Dashboard — Streamlit control panel (optional)

A permissively-licensed (Streamlit, Apache-2.0) browser UI to **launch, tune, and observe** the
pipeline. It is a thin face over the same `automl_template.pipeline.run_pipeline` the CLI and the
Dagster job drive — it holds **zero domain knowledge** (every label / window / knob is read from
the active example's YAML), and it is **not** part of the `core` compose profile.

## What it does

| Tab | Powered by | You can |
|---|---|---|
| **▶ Run** | out-of-process `automl-run` (streamed to a log file) | launch a run (pick a run date), seed example data, watch live logs, see the result summary + recent-run history |
| **🏆 Model** | MLflow registry (`ml/registry.get_champion`) | see the champion version / `macro_f1` / age |
| **📊 Predictions** | `store.online.OnlineStore.summary` | see rows served, model version(s), class distribution, consistency, and do a point lookup |
| **❤️ Health** | serving DB + MLflow probes + optional HTTP tiles | check the moving parts are up |
| **⚙️ Config** | `PipelineConfig` (validated) | edit retrain/promotion/AutoML/validation knobs and save them back to the YAML |

## Run it

**Local (no Docker)** — fastest; uses the SQLite + SQLite-MLflow zero-server defaults:

```bash
pip install -e ".[dashboard]"
streamlit run dashboard/app.py        # or: make dashboard-local
# → http://localhost:8501
```

Seed data first (Run tab → *Seed example data*, or `automl-seed`) so there's something to score.

**Docker (compose)** — alongside the `core` stack, on port 8501:

```bash
make up            # core stack (postgres x2, mlflow, dagster, api)
make dashboard     # adds the Streamlit service (dashboard profile)
# → http://localhost:8501
```

## Design notes

- **Runs execute out-of-process** (`python -c "... cli.run()"`), so a long FLAML train never
  blocks the UI, the child reads env/YAML fresh (no `get_settings` cache to bust), and logs tail
  cleanly across Streamlit reruns. Works the same on SQLite and Postgres — no Dagster server needed.
- **Reads degrade gracefully**: a missing champion, an empty store, or an unreachable service
  render as a friendly "not ready" state, never a traceback.
- **Config edits are validated** through `PipelineConfig` before an atomic write, so a bad value
  can't corrupt the YAML the next run will read.

The control-plane logic lives in [`service.py`](service.py) (no Streamlit import — unit-testable);
[`app.py`](app.py) is presentation only.
