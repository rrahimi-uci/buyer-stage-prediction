"""Framework-wide constants.

Centralized so identifiers (the registered-model name, the MLflow alias, artifact and matrix
names) are defined exactly once instead of being repeated as string literals across modules.
"""

from __future__ import annotations

# MLflow Model Registry
MODEL_NAME = "tabular_automl_champion"
CHAMPION_ALIAS = "champion"
FEATURE_COLUMNS_ARTIFACT = "feature_columns.json"
MODEL_ARTIFACT_PATH = "model"

# Offline feature-matrix partition names
TRAIN_MATRIX = "train"
SCORE_MATRIX = "test"

__all__ = [
    "MODEL_NAME",
    "CHAMPION_ALIAS",
    "FEATURE_COLUMNS_ARTIFACT",
    "MODEL_ARTIFACT_PATH",
    "TRAIN_MATRIX",
    "SCORE_MATRIX",
]
