"""Unit tests for the pure helpers in ``defs/jobs.py``.

These call ``_resolve_forecaster_config`` (against the real ``conf/model/xgboost.yaml``) and
``_fold_ids_for_run_mode`` (against a synthetic CV config) directly — no MLflow, Dagster, or
Settings — so they stay in the fast, unmarked unit tier. The job wiring itself is covered by the
integration test in ``test_register_experiment_job.py``.
"""

from datetime import date

from contracts.hydra_schemas import CvConfig, CvFoldConfig
from xgboost_forecaster import XGBoostConfig, XGBoostForecaster

from nged_substation_forecast.defs.jobs import (
    _fold_ids_for_run_mode,
    _resolve_forecaster_config,
)

_BASE_CONFIG = "conf/model/xgboost.yaml"


def test_resolve_applies_scalar_override() -> None:
    forecaster_cls, config = _resolve_forecaster_config(_BASE_CONFIG, {"n_estimators": 7}, "exp")

    assert forecaster_cls is XGBoostForecaster
    assert forecaster_cls.MODEL_NAME == "xgboost"
    assert isinstance(config, XGBoostConfig)
    assert config.n_estimators == 7
    assert config.experiment_name == "exp"


def test_resolve_replaces_list_override() -> None:
    """A list override swaps the whole list (merge replaces, not extends) — then coerces to a set."""
    _, config = _resolve_forecaster_config(
        _BASE_CONFIG, {"selected_features": ["power_lag_24h"]}, "exp"
    )

    assert config.selected_features == {"power_lag_24h"}


def test_resolve_no_overrides_uses_yaml_defaults() -> None:
    _, config = _resolve_forecaster_config(_BASE_CONFIG, {}, "exp")

    assert isinstance(config, XGBoostConfig)
    assert config.n_estimators == 500
    assert config.experiment_name == "exp"


def _mixed_fold_config() -> CvConfig:
    """A synthetic config mixing leaderboard folds with a non-leaderboard dev fold, so the
    flag-based run-mode selection is testable independently of conf/cv/default.yaml."""
    leaderboard_folds = [
        CvFoldConfig(
            fold_id=f"fold_{i}",
            train_start=date(2024, 4, 1),
            train_end=date(2025, 1, 1),
            val_start=date(2025, 1, 2),
            val_end=date(2025, 6, 1),
        )
        for i in range(2)
    ]
    dev_fold = CvFoldConfig(
        fold_id="dev",
        leaderboard=False,
        min_training_months=1,
        train_start=date(2025, 1, 1),
        train_end=date(2025, 1, 31),
        val_start=date(2025, 2, 1),
        val_end=date(2025, 2, 28),
    )
    return CvConfig(folds=[*leaderboard_folds, dev_fold])


def test_smoke_test_uses_the_non_leaderboard_folds() -> None:
    assert _fold_ids_for_run_mode("smoke_test", _mixed_fold_config()) == ["dev"]


def test_full_cv_uses_the_leaderboard_folds() -> None:
    assert _fold_ids_for_run_mode("full_cv", _mixed_fold_config()) == ["fold_0", "fold_1"]


def test_register_only_uses_the_leaderboard_folds() -> None:
    assert _fold_ids_for_run_mode("register_only", _mixed_fold_config()) == ["fold_0", "fold_1"]
