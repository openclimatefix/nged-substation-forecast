from datetime import datetime, timezone
from typing import get_args

import patito as pt
import pytest
from contracts.ml_schemas import AllFeatures, Metrics, SafeInputBaseColumn, TimeFeature


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
    # Valid
    df = (
        pt.DataFrame(
            {
                "time_series_id": [123],
                "power_fcst_model_name": ["model_a"],
                "nwp_lead_time_hours": [1.0],
                "mae": [0.5],
            }
        )
        .set_model(Metrics)
        .cast()
    )
    df.validate()
