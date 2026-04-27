from datetime import datetime, timezone

import patito as pt
import polars as pl
import pytest
from contracts.ml_schemas import AllFeatures, Metrics
from patito.exceptions import DataFrameValidationError


def test_all_features_validation():
    # Valid
    df = (
        pt.DataFrame(
            {
                "valid_time": [datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)],
                "time_series_id": [123],
                "time_series_type": ["BESS"],
                "power": [10.0],
                "lead_time_hours": [1.0],
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
            "time_series_id": [123],
            "time_series_type": ["BESS"],
            "power": [10.0],
            "lead_time_hours": [1.0],
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
            "time_series_id": [123],
            "time_series_type": ["InvalidType"],
            "power": [10.0],
            "lead_time_hours": [1.0],
            "local_day_of_week": ["Monday"],
        }
    ).set_model(AllFeatures)

    # We expect validation to fail, either during cast or validate
    with pytest.raises(Exception):
        df.cast().validate()


def test_metrics_validation():
    # Valid
    df = (
        pt.DataFrame(
            {
                "time_series_id": [123],
                "power_fcst_model_name": ["model_a"],
                "lead_time_hours": [1.0],
                "mae": [0.5],
            }
        )
        .set_model(Metrics)
        .cast()
    )
    df.validate()
