"""Known vectors pin the model-facing wind convention independently of ingest."""

from datetime import datetime, timedelta
from typing import Literal

import patito as pt
import polars as pl
import pytest
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from ml_core.features import wind_feature_expressions
from ml_core.features._nwp import _upsample_nwp_to_half_hourly
from ml_core.features.tabular_feature_engineer import _derive_wind_features, _engineer_features


@pytest.mark.parametrize("height", ["10m", "100m"])
def test_wind_features_preserve_bearings_calm_and_nulls(height: Literal["10m", "100m"]) -> None:
    frame = pl.DataFrame(
        {
            f"wind_u_{height}": [0.0, -3.0, 0.0, 3.0, 3.0, 0.0, None],
            f"wind_v_{height}": [-3.0, 0.0, 3.0, 0.0, 4.0, 0.0, 4.0],
        }
    ).select(*wind_feature_expressions(height=height))
    assert frame[f"wind_speed_{height}"].to_list() == [3.0, 3.0, 3.0, 3.0, 5.0, 0.0, None]
    assert frame[f"wind_direction_{height}"].to_list() == pytest.approx(
        [0.0, 90.0, 180.0, 270.0, 216.8699, 180.0, None], abs=1e-4
    )
    assert frame.dtypes == [pl.Float32, pl.Float32]


@pytest.mark.parametrize("height", ["10m", "100m"])
def test_interpolation_precedes_polar_derivation(height: Literal["10m", "100m"]) -> None:
    start = datetime(2025, 1, 1)
    # Group 0 crosses North; group 1 reverses direction and cancels at its midpoint.
    frame = pl.DataFrame(
        {
            "nwp_init_time": [start] * 4,
            "ensemble_member": [0, 0, 1, 1],
            "valid_time": [start, start + timedelta(hours=3)] * 2,
            f"wind_u_{height}": [1.0, -1.0, 10.0, -10.0],
            f"wind_v_{height}": [-10.0, -10.0, 0.0, 0.0],
        }
    )
    result = _derive_wind_features(_upsample_nwp_to_half_hourly(frame.lazy())).collect()
    midpoint = result.filter(pl.col("valid_time") == start + timedelta(minutes=90)).sort(
        "ensemble_member"
    )
    assert midpoint[f"wind_speed_{height}"].to_list() == [10.0, 0.0]
    assert midpoint[f"wind_direction_{height}"].to_list() == [0.0, 180.0]
    assert result.height == 14


def test_wind_interpolation_preserves_missing_endpoints() -> None:
    start = datetime(2025, 1, 1)
    frame = pl.LazyFrame(
        {
            "nwp_init_time": [start] * 3,
            "ensemble_member": [0] * 3,
            "valid_time": [start + timedelta(hours=hour) for hour in (0, 3, 6)],
            "wind_u_10m": [None, 3.0, None],
            "wind_v_10m": [None, 4.0, None],
        }
    )
    result = _derive_wind_features(_upsample_nwp_to_half_hourly(frame)).collect().sort("valid_time")
    assert result["wind_speed_10m"].to_list() == [None] * 6 + [5.0] + [None] * 6


def test_windchill_only_uses_interpolated_vector_speed() -> None:
    start = datetime(2025, 1, 1)
    midpoint = start + timedelta(minutes=90)
    nwp = pl.LazyFrame(
        {
            "init_time": [start, start],
            "valid_time": [start, start + timedelta(hours=3)],
            "time_series_id": ["synthetic", "synthetic"],
            "ensemble_member": [0, 0],
            "temperature_2m": [0.0, 0.0],
            "wind_u_10m": [6.0, -6.0],
            "wind_v_10m": [8.0, 8.0],
        }
    )
    power = pt.LazyFrame(
        {"time_series_id": ["synthetic"], "time": [midpoint], "power": [1.0]}
    ).set_model(PowerTimeSeries)
    metadata = pt.DataFrame(
        {"time_series_id": ["synthetic"], "time_series_type": ["substation"]}
    ).set_model(TimeSeriesMetadata)
    result = _engineer_features(
        selected_features={"windchill"},
        power_time_series=power,
        time_series_metadata=metadata,
        nwp=nwp,
        power_fcst_init_time=start,
        nwp_init_time=start,
    ).collect()
    # At the midpoint speed is 8 m/s, rather than the endpoint magnitude of 10 m/s.
    expected = 13.12 - 11.37 * (8.0 * 3.6) ** 0.16
    assert result["windchill"].item() == pytest.approx(expected)
