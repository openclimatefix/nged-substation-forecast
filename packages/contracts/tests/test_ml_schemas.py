from datetime import datetime, timezone
from typing import get_args

import patito as pt
import pytest
from contracts.ml_schemas import (
    AllFeatures,
    Metrics,
    SafeInputBaseColumn,
    TimeFeature,
)


def test_all_features_validation():
    # Valid
    df = (
        pt.DataFrame(
            {
                "valid_time": [datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)],
                "power_fcst_init_time": [datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)],
                "nwp_init_time": [datetime(2025, 12, 31, 18, 0, tzinfo=timezone.utc)],
                "time_series_id": [123],
                "time_series_type": ["BESS"],
                "power": [10.0],
                "nwp_lead_time_hours": [1.0],
                "local_day_of_week": ["Monday"],
            }
        )
        .set_model(AllFeatures)
        .cast()
    )
    df.validate()


def test_all_features_invalid_day_of_week():
    # Invalid day of week
    df = pt.DataFrame(
        {
            "valid_time": [datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)],
            "power_fcst_init_time": [datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)],
            "nwp_init_time": [datetime(2025, 12, 31, 18, 0, tzinfo=timezone.utc)],
            "time_series_id": [123],
            "time_series_type": ["BESS"],
            "power": [10.0],
            "nwp_lead_time_hours": [1.0],
            "local_day_of_week": ["InvalidDay"],
        }
    ).set_model(AllFeatures)

    # We expect validation to fail, either during cast or validate
    with pytest.raises(Exception):
        df.cast().validate()


def test_all_features_invalid_time_series_type():
    # Invalid time_series_type
    df = pt.DataFrame(
        {
            "valid_time": [datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)],
            "power_fcst_init_time": [datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)],
            "nwp_init_time": [datetime(2025, 12, 31, 18, 0, tzinfo=timezone.utc)],
            "time_series_id": [123],
            "time_series_type": ["InvalidType"],
            "power": [10.0],
            "nwp_lead_time_hours": [1.0],
            "local_day_of_week": ["Monday"],
        }
    ).set_model(AllFeatures)

    # We expect validation to fail, either during cast or validate
    with pytest.raises(Exception):
        df.cast().validate()


def test_time_feature_names_are_all_features_fields():
    """Every name in TimeFeature must be a field of AllFeatures.

    Guards against renaming an AllFeatures column without updating the Literal.
    """
    assert frozenset(get_args(TimeFeature)) <= frozenset(AllFeatures.model_fields)


def test_safe_input_base_column_names_are_all_features_fields():
    """Every name in SafeInputBaseColumn must be a field of AllFeatures.

    Guards against renaming an AllFeatures column without updating the Literal.
    """
    assert frozenset(get_args(SafeInputBaseColumn)) <= frozenset(AllFeatures.model_fields)


def test_metrics_validation():
    """The core metric columns validate on their own (the window/scope columns are optional)."""
    df = (
        pt.DataFrame(
            {
                "time_series_id": [123],
                "power_fcst_model_name": ["model_a"],
                "fold_id": ["2022"],
                "horizon_slice": ["all"],
                "metric_name": ["mae"],
                "metric_param": ["all"],
                "metric_value": [5.2],
            }
        )
        .set_model(Metrics)
        .cast()
    )
    df.validate()


def test_metrics_validation_with_scope_and_window_columns():
    """A full leaderboard row including the scope/window provenance columns validates."""
    df = (
        pt.DataFrame(
            {
                "time_series_id": [123],
                "power_fcst_model_name": ["model_a"],
                "fold_id": ["2022"],
                "horizon_slice": ["all"],
                "metric_name": ["mae"],
                "metric_param": ["all"],
                "metric_value": [5.2],
                "evaluation_scope": ["leaderboard"],
                "time_series_type": ["all"],
                "window_start": [datetime(2022, 1, 1, 0, 0, tzinfo=timezone.utc)],
                "window_end": [datetime(2022, 12, 31, 23, 59, 59, tzinfo=timezone.utc)],
                "window_label": ["2022"],
                "computed_at": [datetime(2026, 6, 24, 12, 0, tzinfo=timezone.utc)],
                "mlflow_run_id": ["abc123"],
            }
        )
        .set_model(Metrics)
        .cast()
    )
    df.validate()
