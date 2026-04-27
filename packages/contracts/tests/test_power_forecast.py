from datetime import datetime, timezone

import patito as pt
import pytest
from contracts.power_schemas import PowerForecast


def test_power_forecast_validation():
    # Valid
    df = (
        pt.DataFrame(
            {
                "valid_time": [datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)],
                "time_series_id": [123],
                "ensemble_member": [1],
                "ml_flow_experiment_id": [1],
                "nwp_init_time": [datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)],
                "power_fcst_model_name": ["model_a"],
                "power_fcst_init_time": [datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)],
                "power_fcst": [10.0],
            }
        )
        .set_model(PowerForecast)
        .cast()
    )

    # Should pass
    df.validate()


@pytest.mark.parametrize(
    "data, expected_error",
    [
        # Invalid ensemble_member (too high for Int8)
        (
            {
                "valid_time": [datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)],
                "time_series_id": [123],
                "ensemble_member": [200],
                "ml_flow_experiment_id": [1],
                "nwp_init_time": [datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)],
                "power_fcst_model_name": ["model_a"],
                "power_fcst_init_time": [datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)],
                "power_fcst": [10.0],
            },
            "ensemble_member",
        ),
    ],
)
def test_power_forecast_invalid_data(data, expected_error):
    # We need to cast to ensure the types are checked
    df = pt.DataFrame(data).set_model(PowerForecast)

    # We expect validation to fail
    with pytest.raises(Exception, match=expected_error):
        df.cast().validate()
