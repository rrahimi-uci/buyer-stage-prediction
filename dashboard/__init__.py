"""Streamlit control panel — launch, configure, and observe the pipeline from a browser.

An OPTIONAL, permissively-licensed (Streamlit, Apache-2.0) face over the same
``pipeline.run_pipeline`` the CLI / Dagster job / tests drive. It adds ZERO domain knowledge:
every label, window, and knob it renders is read from the active example's YAML. Kept out of
the ``core`` compose profile (its own ``dashboard`` profile) so the default stack stays lean.

- ``service`` — the control plane: launch runs, read/validate/save config, champion + serving
  snapshots, service health. Pure functions, no Streamlit import, so it stays unit-testable.
- ``app`` — the Streamlit presentation layer built on ``service``.
"""
