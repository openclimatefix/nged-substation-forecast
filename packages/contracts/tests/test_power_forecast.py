from datetime import UTC, datetime
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


def _two_forecast_rows(**overrides: list[Any]) -> pt.DataFrame:
    """Two otherwise-identical forecast rows, with the named columns overridden."""
    columns: dict[str, list[Any]] = {
        "valid_time": [datetime(2026, 1, 1, 0, 30, tzinfo=UTC)] * 2,
        "time_series_id": [123] * 2,
        "ensemble_member": [1] * 2,
        "ml_flow_experiment_id": [1] * 2,
        "nwp_init_time": [datetime(2026, 1, 1, 0, 0, tzinfo=UTC)] * 2,
        "power_fcst_model_name": ["model_a"] * 2,
        "power_fcst_model_version": [1] * 2,
        "power_fcst_init_time": [datetime(2026, 1, 1, 0, 0, tzinfo=UTC)] * 2,
        "power_fcst": [10.0, 20.0],
        "experiment_name": ["baseline"] * 2,
        "fold_id": ["live"] * 2,
    }
    columns |= overrides
    return pt.DataFrame(columns).set_model(PowerForecast).cast()


def test_power_forecast_rejects_duplicate_primary_key():
    """Two rows sharing the full primary key mean an upstream join fanned out."""
    with pytest.raises(ValueError, match="Duplicate entries found for primary key"):
        _two_forecast_rows().validate()


def test_power_forecast_accepts_rows_differing_only_in_ensemble_member():
    """`ensemble_member` is part of the key, so an ensemble is not a duplicate."""
    _two_forecast_rows(ensemble_member=[1, 2]).validate()
