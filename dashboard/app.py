"""Streamlit control panel: launch · configure · observe the pipeline from a browser.

Run it with ``streamlit run dashboard/app.py`` (or ``make dashboard`` / the compose ``dashboard``
profile). Every action delegates to :mod:`dashboard.service`; this module is presentation only.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from automl_template.config import get_settings

# `streamlit run dashboard/app.py` puts dashboard/ on sys.path but NOT the project root, so the
# `dashboard` package isn't importable by default. Add the repo root (this file's grandparent) so
# the import below resolves regardless of launch method or working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard import service  # noqa: E402  (must follow the sys.path bootstrap above)

REFRESH_SECONDS = 2.0  # cadence of the live-run log tail

st.set_page_config(page_title="AutoML Pipeline Console", page_icon="🎯", layout="wide")
settings = get_settings()


# --------------------------------------------------------------------------- #
# Cached reads — keep the live-run auto-refresh cheap; a button clears them.   #
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=10)
def _champion() -> dict:
    return service.champion_info(settings)


@st.cache_data(ttl=10)
def _predictions() -> dict:
    return service.predictions_summary(settings)


@st.cache_data(ttl=10)
def _health() -> list[dict]:
    return service.service_health(settings)


@st.cache_data(ttl=10)
def _past_runs() -> list[dict]:
    return service.list_past_runs(settings)


# --------------------------------------------------------------------------- #
# Header + sidebar                                                             #
# --------------------------------------------------------------------------- #
st.title("🎯 AutoML Pipeline Console")
st.caption(
    "Launch, tune, and watch the batch-scoring-to-online-store pipeline — "
    "the same `run_pipeline` the CLI and Dagster job drive."
)

with st.sidebar:
    st.subheader("Active example")
    st.code(str(service.active_config_path(settings)), language="text")
    st.text_input("Serving DB", value=settings.serving_db_url, disabled=True)
    st.text_input("MLflow", value=settings.mlflow_tracking_uri, disabled=True)
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

run_tab, model_tab, pred_tab, health_tab, config_tab = st.tabs(
    ["▶ Run", "🏆 Model", "📊 Predictions", "❤️ Health", "⚙️ Config"]
)


# --------------------------------------------------------------------------- #
# Run                                                                         #
# --------------------------------------------------------------------------- #
with run_tab:
    active: service.RunHandle | None = st.session_state.get("run")
    disabled = bool(active and active.is_running)

    left, right = st.columns([2, 1])
    with left:
        run_date = st.text_input(
            "Run date (blank = today, UTC)",
            value=settings.run_date or "",
            placeholder=settings.resolved_run_date(),
            disabled=disabled,
        )
    with right:
        st.write("")
        st.write("")
        launch = st.button(
            "▶ Run pipeline", type="primary", use_container_width=True, disabled=disabled
        )
        seed = st.button("🌱 Seed example data", use_container_width=True, disabled=disabled)

    if launch:
        st.session_state["run"] = service.launch_run(run_date.strip() or None, settings)
        st.rerun()
    if seed:
        st.session_state["run"] = service.seed_example()
        st.rerun()

    active = st.session_state.get("run")
    if active:
        running = active.is_running
        header = f"Run `{active.run_id}` · {active.run_date}"
        if running:
            with st.status(f"{header} — running…", state="running"):
                st.code(active.read_log(tail=200) or "(waiting for output…)", language="log")
        else:
            ok = active.returncode == 0
            summary = service.parse_summary(active.read_log())
            with st.status(
                f"{header} — {'finished ✅' if ok else 'failed ❌'}",
                state="complete" if ok else "error",
            ):
                if summary:
                    c = st.columns(4)
                    c[0].metric("Retrained", "yes" if summary["retrained"] else "no")
                    c[1].metric("Model version", summary["model_version"] or "—")
                    c[2].metric("Rows scored", summary["rows_scored"])
                    c[3].metric(
                        "macro_f1",
                        f"{summary['macro_f1']:.4f}"
                        if isinstance(summary["macro_f1"], float)
                        else "—",
                    )
                    if summary["problems"] not in ("[]", None):
                        st.error(f"Consistency problems: {summary['problems']}")
                with st.expander("Full log"):
                    st.code(active.read_log() or "(no output)", language="log")
            # Refresh serving/model numbers once, now that this run wrote them.
            if st.session_state.get("cleared_for") != active.run_id:
                st.cache_data.clear()
                st.session_state["cleared_for"] = active.run_id

    st.divider()
    st.subheader("Recent runs")
    past = _past_runs()
    if past:
        st.dataframe(pd.DataFrame(past), use_container_width=True, hide_index=True)
    else:
        st.info("No runs yet — launch one above.")


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
with model_tab:
    st.subheader("Champion model")
    champ = _champion()
    if champ.get("error"):
        st.warning(f"MLflow registry unreachable: {champ['error']}")
    elif not champ["available"]:
        st.info("No champion yet — run the pipeline once (cold-start trains unconditionally).")
    else:
        c = st.columns(3)
        c[0].metric("Version", champ["version"])
        c[1].metric("macro_f1", f"{champ['macro_f1']:.4f}")
        c[2].metric("Age (days)", f"{champ['age_days']:.1f}")
    st.caption(
        "Promotion is champion/challenger: a new model wins the `champion` alias only if it "
        "beats the incumbent by the configured margin (see the Config tab). Flip aliases "
        "manually in the MLflow UI."
    )


# --------------------------------------------------------------------------- #
# Predictions                                                                 #
# --------------------------------------------------------------------------- #
with pred_tab:
    st.subheader("Serving store")
    snap = _predictions()
    if not snap.get("available"):
        st.warning(f"Serving store unreachable: {snap.get('error', 'unknown error')}")
    elif snap["total"] == 0:
        st.info("Serving store is empty — no successful run has loaded predictions yet.")
    else:
        c = st.columns(3)
        c[0].metric("Rows served", f"{snap['total']:,}")
        c[1].metric("Model version(s)", ", ".join(snap["model_versions"]) or "—")
        c[2].metric("Last scored", str(snap["last_scored_at"] or "—"))
        if snap["problems"]:
            st.error(f"Consistency: {snap['problems']}")
        else:
            st.success("Consistency OK — single model version, non-empty.")

        counts = snap["class_counts"]
        if counts:
            df = pd.DataFrame({"class": list(counts), "count": list(counts.values())})
            st.bar_chart(df.set_index("class"), horizontal=True)

    st.divider()
    st.subheader("Point lookup")
    entity = st.text_input("entity_id", placeholder="e.g. a member_id")
    if entity:
        row = service.lookup_prediction(entity.strip(), settings)
        st.json(row) if row else st.warning(f"No prediction stored for entity_id={entity!r}")


# --------------------------------------------------------------------------- #
# Health                                                                      #
# --------------------------------------------------------------------------- #
with health_tab:
    st.subheader("Services")
    tiles = _health()
    cols = st.columns(len(tiles) or 1)
    for col, t in zip(cols, tiles, strict=False):
        with col:
            st.markdown(f"### {'🟢' if t['ok'] else '🔴'} {t['name']}")
            st.caption(t["detail"])
    st.caption(
        "HTTP tiles (api / dagster / mlflow-ui) appear only when their `*_URL` env is set "
        "(they're wired in the compose `dashboard` profile)."
    )


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
with config_tab:
    st.subheader("Domain config")
    st.caption("Edits are validated through `PipelineConfig`, then written to the active YAML.")
    cfg = service.load_config_raw(settings)
    automl = cfg.setdefault("automl", {})
    retrain = cfg.setdefault("retrain", {})
    promotion = cfg.setdefault("promotion", {})
    validation = cfg.setdefault("validation", {})
    store = cfg.setdefault("online_store", {})

    with st.form("config"):
        a, b = st.columns(2)
        with a:
            st.markdown("**AutoML**")
            automl["time_budget_seconds"] = st.number_input(
                "time_budget_seconds", 1, 3600, int(automl.get("time_budget_seconds", 120))
            )
            automl["max_candidates"] = st.number_input(
                "max_candidates", 1, 500, int(automl.get("max_candidates", 20))
            )
            automl["seed"] = st.number_input("seed", 0, 10_000, int(automl.get("seed", 42)))
            st.markdown("**Validation**")
            validation["min_rows_per_class"] = st.number_input(
                "min_rows_per_class", 1, 100_000, int(validation.get("min_rows_per_class", 50))
            )
            validation["pit_window_days"] = st.number_input(
                "pit_window_days", 0, 30, int(validation.get("pit_window_days", 2))
            )
        with b:
            st.markdown("**Retrain policy**")
            retrain["max_model_age_days"] = st.number_input(
                "max_model_age_days", 1, 365, int(retrain.get("max_model_age_days", 14))
            )
            retrain["drift_threshold"] = st.slider(
                "drift_threshold", 0.0, 1.0, float(retrain.get("drift_threshold", 0.30)), 0.01
            )
            st.markdown("**Promotion**")
            promotion["margin"] = st.number_input(
                "margin", 0.0, 1.0, float(promotion.get("margin", 0.0)), 0.01
            )
            promotion["require_manual_promotion"] = st.checkbox(
                "require_manual_promotion", bool(promotion.get("require_manual_promotion", False))
            )
            st.markdown("**Online store**")
            store["ttl_days"] = st.number_input("ttl_days", 1, 365, int(store.get("ttl_days", 7)))

        if st.form_submit_button("💾 Save config", type="primary"):
            try:
                service.save_config(cfg, settings)
                st.success("Saved — the next run will use these values.")
            except ValueError as exc:
                st.error(f"Invalid config, not saved:\n\n{exc}")

    with st.expander("Raw YAML"):
        st.code(service.active_config_path(settings).read_text(), language="yaml")


# --------------------------------------------------------------------------- #
# Live-run auto-refresh — placed last so every tab renders each cycle, letting #
# you watch the log on the Run tab or browse the others while a run is going.  #
# --------------------------------------------------------------------------- #
_active = st.session_state.get("run")
if _active and _active.is_running:
    time.sleep(REFRESH_SECONDS)
    st.rerun()
