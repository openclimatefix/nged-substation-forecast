"""Unit tests for the pure config-resolution helper in ``defs/jobs.py``.

These call ``_resolve_forecaster_config`` directly against the real ``conf/model/xgboost.yaml`` —
no MLflow, Dagster, or Settings — so they stay in the fast, unmarked unit tier. The job wiring
itself is covered by the integration test in ``test_register_experiment_job.py``.
"""

from xgboost_forecaster import XGBoostConfig, XGBoostForecaster

from nged_substation_forecast.defs.jobs import _resolve_forecaster_config

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
