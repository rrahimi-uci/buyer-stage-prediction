"""Typed exception hierarchy.

A single base (``AutoMLTemplateError``) lets callers catch everything this framework raises,
while the specific subclasses give precise, self-documenting failures instead of bare
``ValueError`` / ``RuntimeError``.
"""

from __future__ import annotations


class AutoMLTemplateError(Exception):
    """Base class for all errors raised by this framework."""


class DataValidationError(AutoMLTemplateError):
    """Input/feature/training data failed a validation contract."""

    def __init__(self, violations: list[str]):
        self.violations = violations
        super().__init__("; ".join(violations) or "data validation failed")


class ChampionUnavailableError(AutoMLTemplateError):
    """Scoring/serving was attempted but no champion model is registered."""


class ConfigurationError(AutoMLTemplateError):
    """Invalid or unsupported configuration (e.g. unknown AutoML engine)."""


__all__ = [
    "AutoMLTemplateError",
    "DataValidationError",
    "ChampionUnavailableError",
    "ConfigurationError",
]
