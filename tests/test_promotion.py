"""Champion/challenger promotion gate (pure-logic unit; MLflow mocked).

Asserts: cold-start promotes unconditionally; challenger promotes iff it beats the
incumbent by the margin; manual-promotion flag blocks auto-promotion.
"""

from __future__ import annotations

from unittest import mock

from automl_template.config import PipelineConfig
from automl_template.ml import registry
from automl_template.ml.registry import ChampionInfo


def _cfg(margin: float = 0.0, manual: bool = False) -> PipelineConfig:
    cfg = PipelineConfig(target="t", entity_key="e")
    cfg.promotion.margin = margin
    cfg.promotion.require_manual_promotion = manual
    return cfg


# No create=True: registry now imports mlflow at module level, so the attribute really
# exists and the patch takes effect (a future rename would now surface as an error).
@mock.patch("automl_template.ml.registry.mlflow")
def _promote(challenger_f1, champion, cfg, mlflow_mock):
    mlflow_mock.register_model.return_value = mock.Mock(version="7")
    mlflow_mock.MlflowClient.return_value = mock.Mock()
    with mock.patch.object(registry, "get_champion", return_value=champion):
        return registry.register_and_maybe_promote("m", "run-1", challenger_f1, cfg)


def test_cold_start_promotes() -> None:
    assert _promote(0.10, None, _cfg()) is True


def test_better_challenger_promotes() -> None:
    champ = ChampionInfo(version="6", macro_f1=0.70, age_days=1)
    assert _promote(0.75, champ, _cfg(margin=0.0)) is True


def test_worse_challenger_rejected() -> None:
    champ = ChampionInfo(version="6", macro_f1=0.70, age_days=1)
    assert _promote(0.69, champ, _cfg(margin=0.0)) is False


def test_manual_flag_blocks_auto_promotion() -> None:
    champ = ChampionInfo(version="6", macro_f1=0.70, age_days=1)
    assert _promote(0.99, champ, _cfg(manual=True)) is False
