"""Unit tests for the pure helpers in ``defs/jobs.py``.

These call ``_resolve_forecaster_config`` (against the real ``conf/model/xgboost.yaml``),
``_fold_ids_for_run_mode`` (against a synthetic CV config) and the identity helpers directly — no
MLflow, Dagster, or Settings — so they stay in the fast, unmarked unit tier. The job wiring itself
is covered by the integration test in ``test_register_experiment_job.py``.
"""

import json
from datetime import date
from typing import Any

import pytest
from contracts.hydra_schemas import CvConfig, CvFoldConfig
from ml_core._cv_helpers import flatten_config
from xgboost_forecaster import XGBoostConfig, XGBoostForecaster

from nged_substation_forecast.defs.jobs import (
    ExperimentIdentityChangedError,
    IdentityTagsType,
    _fold_ids_for_run_mode,
    _identity_tags,
    _reject_changed_identity,
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
    """A list override swaps the whole list (merge replaces, not extends), then coerces to a set."""
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


def _identity_of(**overrides: Any) -> IdentityTagsType:
    """The identity tags a registration of ``conf/model/xgboost.yaml`` would ask for."""
    forecaster_cls, config = _resolve_forecaster_config(_BASE_CONFIG, overrides, "exp")
    return _identity_tags(forecaster_cls, config)


def _as_stored(tags: IdentityTagsType, **extra: str) -> dict[str, str]:
    """Widen an identity to the open tag mapping MLflow returns for an existing experiment."""
    return {**tags, **extra}


def test_identity_tags_serialise_the_feature_set_sorted() -> None:
    """The ``config`` tag must not depend on the feature set's iteration order.

    ``selected_features`` is a ``set``, and a set of strings iterates in a different order in every
    process (hash randomisation). Since Dagster launches each job run in its own process, an
    order-sensitive dump would make a re-registration of an *unchanged* config look changed —
    rejected by the identity check, and by MLflow's write-once params before that. Sorted output is
    the canonical form that makes the comparison mean what it says. Asserted against the base
    YAML's whole feature set (two dozen features), so the assertion cannot pass by an accident of
    hash ordering.
    """
    _, config = _resolve_forecaster_config(_BASE_CONFIG, {}, "exp")

    dumped = json.loads(_identity_tags(XGBoostForecaster, config)["config"])

    assert dumped["selected_features"] == sorted(config.selected_features)


def test_the_override_lists_order_does_not_change_the_identity() -> None:
    features = ["power_lag_24h", "hour_of_day", "temperature_2m"]

    assert _identity_of(selected_features=features) == _identity_of(
        selected_features=list(reversed(features))
    )


def test_unchanged_identity_is_accepted() -> None:
    """A re-registration of the same config passes — and the description is free to differ."""
    tags = _identity_of(n_estimators=7)

    _reject_changed_identity("exp", _as_stored(tags, description="unrelated"), tags)


def test_absent_identity_tags_are_accepted() -> None:
    """An untagged experiment (get_or_create_experiment's fallback) is completable, not a clash."""
    _reject_changed_identity("exp", {}, _identity_of())


def test_changed_config_is_rejected_and_names_the_changed_field() -> None:
    with pytest.raises(ExperimentIdentityChangedError) as excinfo:
        _reject_changed_identity(
            "exp", _as_stored(_identity_of(n_estimators=7)), _identity_of(n_estimators=300)
        )

    message = str(excinfo.value)
    assert "'exp'" in message
    assert "config.n_estimators: 7 -> 300" in message
    # Only the field that actually changed is reported.
    assert "max_depth" not in message


def test_changed_forecaster_class_is_rejected() -> None:
    stored = _as_stored(_identity_of(), forecaster_target="some.other.Forecaster")

    with pytest.raises(ExperimentIdentityChangedError, match="forecaster_target"):
        _reject_changed_identity("exp", stored, _identity_of())


def test_a_reserialised_config_is_not_a_change() -> None:
    """The comparison is on the parsed JSON, so key order and formatting are not a config change.

    Otherwise a stored tag written by an older serialisation of the same values — a field added
    with a default, or reordered in the model class — would falsely make the experiment
    un-re-registerable, and would do it while naming no differing field at all.
    """
    requested = _identity_of(n_estimators=7)
    reserialised = json.dumps(dict(reversed(list(json.loads(requested["config"]).items()))))

    _reject_changed_identity("exp", _as_stored({**requested, "config": reserialised}), requested)


def test_a_field_absent_on_one_side_is_reported_not_swallowed() -> None:
    """``None`` and "field missing" are different, and neither may produce a blank explanation."""
    requested = _identity_of()
    stored_config = json.loads(requested["config"])
    del stored_config["ml_flow_experiment_id"]  # its value on the requested side is null
    stored = _as_stored({**requested, "config": json.dumps(stored_config)})

    with pytest.raises(ExperimentIdentityChangedError) as excinfo:
        _reject_changed_identity("exp", stored, requested)

    assert "config.ml_flow_experiment_id: <absent> -> None" in str(excinfo.value)


def test_flattened_config_params_are_order_stable() -> None:
    """The write-once MLflow params must not vary with feature-set iteration order either."""
    _, config = _resolve_forecaster_config(_BASE_CONFIG, {}, "exp")

    assert flatten_config(config)["selected_features"] == str(sorted(config.selected_features))
