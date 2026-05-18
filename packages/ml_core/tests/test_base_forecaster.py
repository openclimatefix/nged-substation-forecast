from datetime import datetime, timezone
from pathlib import Path

import patito as pt
import polars as pl
import pytest
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerForecast, PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import NwpOnDisk
from ml_core.base_forecaster import BaseForecaster


class DummyForecaster(BaseForecaster):
    def _train_algo(self, data: pt.LazyFrame[AllFeatures]) -> None:
        pass

    def _predict_algo(self, data: pt.LazyFrame[AllFeatures]) -> pt.DataFrame[PowerForecast]:
        return pt.DataFrame[PowerForecast]()

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass


@pytest.fixture
def dummy_power_data() -> pt.LazyFrame[PowerTimeSeries]:
    df = pl.DataFrame(
        {
            "valid_time": [
                datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc),
            ],
            "time_series_id": [1, 1, 1],
            "power": [10.0, 20.0, 30.0],
        }
    )
    # Rename valid_time to time to match PowerTimeSeries schema
    df = df.rename({"valid_time": "time"})
    return pt.LazyFrame.from_existing(df.lazy()).set_model(PowerTimeSeries)  # type: ignore[unresolved-attribute]


@pytest.fixture
def dummy_metadata() -> pt.DataFrame[TimeSeriesMetadata]:
    df = pl.DataFrame(
        {
            "time_series_id": [1],
            "time_series_name": ["Test Substation"],
            "time_series_type": ["Primary"],
            "units": ["MW"],
            "licence_area": ["EMids"],
            "substation_number": [123],
            "substation_type": ["Primary"],
            "latitude": [51.5],
            "longitude": [-0.1],
            "h3_res_5": [123456789],
        }
    )
    return pt.DataFrame[TimeSeriesMetadata](df)


def test_engineer_features_success(dummy_power_data, dummy_metadata):
    selected_features = {"local_time_of_day_sin", "power_lag_1h"}
    forecaster = DummyForecaster(selected_features=selected_features, model_params={})

    result_lf = forecaster._engineer_features(
        power_time_series=dummy_power_data,
        time_series_metadata=dummy_metadata,
        nwp=None,
    )

    result_df = result_lf.collect()

    assert "local_time_of_day_sin" in result_df.columns
    assert "power_lag_1h" in result_df.columns
    assert "valid_time" in result_df.columns
    assert "time_series_id" in result_df.columns
    assert "time_series_type" in result_df.columns
    assert "power" in result_df.columns
    # lead_time_hours is only present if NWP data is provided
    assert "lead_time_hours" not in result_df.columns


@pytest.fixture
def dummy_nwp_data() -> pt.LazyFrame[NwpOnDisk]:
    df = pl.DataFrame(
        {
            "valid_time": [
                datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc),
            ],
            "init_time": [
                datetime(2023, 12, 31, 12, 0, tzinfo=timezone.utc),
                datetime(2023, 12, 31, 12, 0, tzinfo=timezone.utc),
                datetime(2023, 12, 31, 12, 0, tzinfo=timezone.utc),
            ],
            "time_series_id": [1, 1, 1],
            "ensemble_member": [0, 0, 0],
            "temperature_2m": [10.0, 12.0, 14.0],
            "wind_speed_10m": [5.0, 6.0, 7.0],
            "downward_short_wave_radiation_flux_surface": [0.0, 100.0, 200.0],
        }
    )
    return pt.LazyFrame.from_existing(df.lazy()).set_model(NwpOnDisk)  # type: ignore[unresolved-attribute]

def test_engineer_features_with_nwp(dummy_power_data, dummy_metadata, dummy_nwp_data):
    selected_features = {
        "windchill",
        "temperature_2m_trend_6h",
        "latest_weekly_lagged_power",
        "latest_weekly_lagged_temperature_2m",
        "latest_weekly_lagged_wind_speed_10m",
        "latest_weekly_lagged_downward_short_wave_radiation_flux_surface",
    }
    forecaster = DummyForecaster(selected_features=selected_features, model_params={})

    result_lf = forecaster._engineer_features(
        power_time_series=dummy_power_data,
        time_series_metadata=dummy_metadata,
        nwp=dummy_nwp_data,
    )

    result_df = result_lf.collect()

    assert "windchill" in result_df.columns
    assert "temperature_2m_trend_6h" in result_df.columns
    assert "latest_weekly_lagged_power" in result_df.columns
    assert "latest_weekly_lagged_temperature_2m" in result_df.columns
    assert "latest_weekly_lagged_wind_speed_10m" in result_df.columns
    assert "latest_weekly_lagged_downward_short_wave_radiation_flux_surface" in result_df.columns
    assert "lead_time_hours" in result_df.columns
    
    # Check lead time calculation (valid_time - init_time)
    # 2024-01-01 00:00 - 2023-12-31 12:00 = 12 hours
    assert result_df["lead_time_hours"][0] == 12.0
    assert result_df["lead_time_hours"][1] == 13.0
    assert result_df["lead_time_hours"][2] == 14.0

def test_engineer_features_typo(dummy_power_data, dummy_metadata):
    selected_features = {"power_lagg_24h"}  # Typo
    forecaster = DummyForecaster(selected_features=selected_features, model_params={})

    with pytest.raises(ValueError, match="Feature engineering failed to create or find"):
        forecaster._engineer_features(
            power_time_series=dummy_power_data,
            time_series_metadata=dummy_metadata,
            nwp=None,
        )


def test_engineer_features_missing_base_col(dummy_power_data, dummy_metadata):
    # Request a feature that depends on a missing base column
    # e.g., temperature_rolling_mean_2h depends on temperature_2m which is not in dummy data
    selected_features = {"temperature_rolling_mean_2h"}
    forecaster = DummyForecaster(selected_features=selected_features, model_params={})

    # Since nwp is None, the feature is never created, so the dynamic schema assertion catches it
    with pytest.raises(ValueError, match="Feature engineering failed to create or find"):
        forecaster._engineer_features(
            power_time_series=dummy_power_data,
            time_series_metadata=dummy_metadata,
            nwp=None,
        )
