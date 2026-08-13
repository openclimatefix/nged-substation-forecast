from datetime import UTC, datetime, timedelta
from typing import Any

import patito as pt
import pytest
from contracts.power_schemas import PowerForecast


def test_power_forecast_validation():
    # Valid
    df = (
        pt.DataFrame(
            {
                "valid_time": [datetime(2026, 1, 1, 0, 30, tzinfo=UTC)],
                "time_series_id": [123],
                "ensemble_member": [1],
                "ml_flow_experiment_id": [1],
                "nwp_init_time": [datetime(2026, 1, 1, 0, 0, tzinfo=UTC)],
                "power_fcst_model_name": ["model_a"],
                "power_fcst_model_version": [1],
                "power_fcst_init_time": [datetime(2026, 1, 1, 0, 0, tzinfo=UTC)],
                "power_fcst": [10.0],
                "experiment_name": ["baseline"],
                "fold_id": ["live"],
            }
        )
        .set_model(PowerForecast)
        .cast()
    )

    # Should pass
    df.validate()


@pytest.mark.parametrize(
    ("data", "expected_error"),
    [
        # Invalid ensemble_member (too high for Int8)
        (
            {
                "valid_time": [datetime(2026, 1, 1, 0, 30, tzinfo=UTC)],
                "time_series_id": [123],
                "ensemble_member": [200],
                "ml_flow_experiment_id": [1],
                "nwp_init_time": [datetime(2026, 1, 1, 0, 0, tzinfo=UTC)],
                "power_fcst_model_name": ["model_a"],
                "power_fcst_model_version": [1],
                "power_fcst_init_time": [datetime(2026, 1, 1, 0, 0, tzinfo=UTC)],
                "power_fcst": [10.0],
                "experiment_name": ["baseline"],
                "fold_id": ["live"],
            },
            "ensemble_member",
        ),
    ],
)
def test_power_forecast_invalid_data(data: dict[str, list[Any]], expected_error: str):
    # We need to cast to ensure the types are checked
    df = pt.DataFrame(data).set_model(PowerForecast)

    # We expect validation to fail
    with pytest.raises(Exception, match=expected_error):
        df.cast().validate()


def _two_forecast_rows(
    ensemble_members: list[int], valid_time_offsets_minutes: list[int]
) -> pt.DataFrame:
    """Two forecast rows differing only in the primary-key columns named."""
    return (
        pt.DataFrame(
            {
                "valid_time": [
                    datetime(2026, 1, 1, 0, 0, tzinfo=UTC) + timedelta(minutes=offset)
                    for offset in valid_time_offsets_minutes
                ],
                "time_series_id": [123] * 2,
                "ensemble_member": ensemble_members,
                "ml_flow_experiment_id": [1] * 2,
                "nwp_init_time": [datetime(2026, 1, 1, 0, 0, tzinfo=UTC)] * 2,
                "power_fcst_model_name": ["model_a"] * 2,
                "power_fcst_model_version": [1] * 2,
                "power_fcst_init_time": [datetime(2026, 1, 1, 0, 0, tzinfo=UTC)] * 2,
                "power_fcst": [10.0, 20.0],
                "experiment_name": ["baseline"] * 2,
                "fold_id": ["live"] * 2,
            }
        )
        .set_model(PowerForecast)
        .cast()
    )


def test_power_forecast_rejects_duplicate_primary_key():
    """Two rows sharing the full primary key mean the same forecast reached us twice."""
    with pytest.raises(ValueError, match="Duplicate entries found for primary key"):
        _two_forecast_rows(ensemble_members=[1, 1], valid_time_offsets_minutes=[30, 30]).validate()


def test_power_forecast_accepts_rows_differing_only_in_ensemble_member():
    """`ensemble_member` is part of the key, so an ensemble is not a duplicate."""
    _two_forecast_rows(ensemble_members=[1, 2], valid_time_offsets_minutes=[30, 30]).validate()


def test_power_forecast_accepts_rows_differing_only_in_valid_time():
    """`valid_time` is part of the key, so two lead times of one run are not duplicates."""
    _two_forecast_rows(ensemble_members=[1, 1], valid_time_offsets_minutes=[30, 60]).validate()
