import pytest
import polars as pl
import patito as pt
from collections.abc import Mapping
from datetime import datetime, timezone, timedelta
from typing import Any, cast
from unittest.mock import MagicMock
from contracts.data_schemas import (
    InferenceParams,
    Nwp,
    PowerTimeSeries,
    TimeSeriesMetadata,
)
from contracts.hydra_schemas import (
    ModelConfig,
    ModelFeaturesConfig,
    NwpModel,
)
from xgboost_forecaster.config import XGBoostHyperparameters
from xgboost_forecaster.model import XGBoostForecaster


@pytest.fixture
def base_data():
    """Fixture that generates minimal, valid base DataFrames for Nwp, PowerTimeSeries, and TimeSeriesMetadata."""
    valid_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    nwp = pl.DataFrame(
        {
            "init_time": [valid_time - timedelta(hours=6)],
            "valid_time": [valid_time],
            "h3_index": [1],
            "ensemble_member": [0],
            "temperature_2m": [10.0],
            "dew_point_temperature_2m": [5.0],
            "wind_speed_10m": [2.0],
            "wind_direction_10m": [180.0],
            "wind_speed_100m": [3.0],
            "wind_direction_100m": [185.0],
            "pressure_surface": [100.0],
            "pressure_reduced_to_mean_sea_level": [101.0],
            "geopotential_height_500hpa": [50.0],
            "downward_short_wave_radiation_flux_surface": [100.0],
            "categorical_precipitation_type_surface": [0.0],
            "lead_time_hours": [6.0],
        }
    ).lazy()

    flows = (
        pl.DataFrame(
            {
                "period_end_time": [valid_time, valid_time - timedelta(days=7)],
                "time_series_id": [1, 1],
                "power": [50.0, 50.0],
                "MVA": [50.0, 50.0],
            }
        )
        .with_columns(
            [
                pl.col("time_series_id").cast(pl.Int32),
                pl.col("power").cast(pl.Float32),
            ]
        )
        .lazy()
    )

    metadata = pl.DataFrame(
        {"time_series_id": [1], "substation_number": [1], "h3_res_5": [1]}
    ).with_columns(pl.col("time_series_id").cast(pl.Int32))

    return {"nwp": nwp, "flows": flows, "metadata": metadata}


def test_nwp_validation_missing_accumulated_variables_at_step_1():
    """Test Nwp validation when accumulated variables are missing from the second step onwards."""
    init_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    df = pl.DataFrame(
        {
            "init_time": [init_time, init_time],
            "valid_time": [init_time, init_time + timedelta(hours=1)],
            "ensemble_member": [0, 0],
            "h3_index": [123, 123],
            "temperature_2m": [10, 11],
            "dew_point_temperature_2m": [5, 6],
            "wind_u_10m": [2.0, 3.0],
            "wind_v_10m": [0.0, 0.0],
            "wind_u_100m": [4.0, 5.0],
            "wind_v_100m": [0.0, 0.0],
            "pressure_surface": [100, 101],
            "pressure_reduced_to_mean_sea_level": [102, 103],
            "geopotential_height_500hpa": [50, 51],
            "categorical_precipitation_type_surface": [0, 0],
            "precipitation_surface": [None, None],  # Step 1 is null!
            "downward_short_wave_radiation_flux_surface": [None, 100],
            "downward_long_wave_radiation_flux_surface": [None, 200],
        }
    ).with_columns(
        [
            pl.col("ensemble_member").cast(pl.UInt8),
            pl.col("h3_index").cast(pl.UInt64),
            pl.col("temperature_2m").cast(pl.Float32),
            pl.col("dew_point_temperature_2m").cast(pl.Float32),
            pl.col("wind_u_10m").cast(pl.Float32),
            pl.col("wind_v_10m").cast(pl.Float32),
            pl.col("wind_u_100m").cast(pl.Float32),
            pl.col("wind_v_100m").cast(pl.Float32),
            pl.col("pressure_surface").cast(pl.Float32),
            pl.col("pressure_reduced_to_mean_sea_level").cast(pl.Float32),
            pl.col("geopotential_height_500hpa").cast(pl.Float32),
            pl.col("categorical_precipitation_type_surface").cast(pl.UInt8),
            pl.col("precipitation_surface").cast(pl.Float32),
            pl.col("downward_short_wave_radiation_flux_surface").cast(pl.Float32),
            pl.col("downward_long_wave_radiation_flux_surface").cast(pl.Float32),
        ]
    )

    with pytest.raises(ValueError, match="Column 'precipitation_surface' contains 1 null values"):
        Nwp.validate(df)


def test_xgboost_forecaster_train_with_nans(base_data):
    """Test that train raises ValueError if input features contain NaNs."""
    forecaster = XGBoostForecaster()
    config = ModelConfig(
        power_fcst_model_name="test",
        hyperparameters=XGBoostHyperparameters().model_dump(),
        features=ModelFeaturesConfig(
            nwps=[NwpModel.ECMWF_ENS_0_25DEG],
            feature_names=[
                "time_series_id",
                "lead_time_hours",
                "latest_available_weekly_power_lag",
                "temperature_2m",
                "downward_short_wave_radiation_flux_surface",
                "wind_speed_10m",
                "hour_sin",
                "hour_cos",
                "day_of_year_sin",
                "day_of_year_cos",
                "day_of_week",
            ],
        ),
    )

    # Inject NaN
    nwp = base_data["nwp"].with_columns(pl.lit(float("nan")).alias("temperature_2m"))

    with pytest.raises(ValueError, match="Input features X contain NaN or Inf values"):
        forecaster.fit(
            config=config,
            power_time_series=cast(pt.LazyFrame[PowerTimeSeries], base_data["flows"]),
            time_series_metadata=cast(
                pt.DataFrame[TimeSeriesMetadata],
                base_data["metadata"],
            ),
            nwps=cast(
                Mapping[NwpModel, pt.LazyFrame[Nwp]],
                {NwpModel.ECMWF_ENS_0_25DEG: cast(pt.LazyFrame[Nwp], nwp)},
            ),
        )


def test_xgboost_forecaster_train_with_infs(base_data):
    """Test that train raises ValueError if input features contain Infs."""
    forecaster = XGBoostForecaster()
    config = ModelConfig(
        power_fcst_model_name="test",
        hyperparameters=XGBoostHyperparameters().model_dump(),
        features=ModelFeaturesConfig(
            nwps=[NwpModel.ECMWF_ENS_0_25DEG],
            feature_names=[
                "time_series_id",
                "lead_time_hours",
                "latest_available_weekly_power_lag",
                "temperature_2m",
                "downward_short_wave_radiation_flux_surface",
                "wind_speed_10m",
                "hour_sin",
                "hour_cos",
                "day_of_year_sin",
                "day_of_year_cos",
                "day_of_week",
            ],
        ),
    )

    # Inject Inf
    nwp = base_data["nwp"].with_columns(pl.lit(float("inf")).alias("temperature_2m"))

    with pytest.raises(ValueError, match="Input features X contain NaN or Inf values"):
        forecaster.fit(
            config=config,
            power_time_series=cast(pt.LazyFrame[PowerTimeSeries], base_data["flows"]),
            time_series_metadata=cast(
                pt.DataFrame[TimeSeriesMetadata],
                base_data["metadata"],
            ),
            nwps=cast(
                Mapping[NwpModel, pt.LazyFrame[Nwp]],
                {NwpModel.ECMWF_ENS_0_25DEG: cast(pt.LazyFrame[Nwp], nwp)},
            ),
        )


def test_xgboost_forecaster_predict_with_nans(base_data):
    """Test that predict raises ValueError if input features contain NaNs."""
    forecaster = XGBoostForecaster()
    mock_model = MagicMock()
    mock_model.feature_names_in_ = [
        "temperature_2m",
        "latest_available_weekly_power_lag",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "day_of_week",
    ]
    forecaster.model = cast(Any, mock_model)
    forecaster.feature_names = mock_model.feature_names_in_

    # Inject NaN
    nwp = base_data["nwp"].with_columns(pl.lit(float("nan")).alias("temperature_2m"))

    inference_params = InferenceParams(
        forecast_time=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        power_fcst_model_name="test",
    )

    with pytest.raises(ValueError, match="Input features X contain NaN or Inf values"):
        forecaster.predict(
            time_series_metadata=cast(
                pt.DataFrame[TimeSeriesMetadata],
                base_data["metadata"],
            ),
            inference_params=inference_params,
            nwps=cast(
                Mapping[NwpModel, pt.LazyFrame[Nwp]],
                {NwpModel.ECMWF_ENS_0_25DEG: cast(pt.LazyFrame[Nwp], nwp)},
            ),
            power_time_series=cast(pt.LazyFrame[PowerTimeSeries], base_data["flows"]),
        )


def test_xgboost_forecaster_train_empty_data_after_drop_nulls(base_data):
    """Test that train raises ValueError if no training data remains after dropping nulls."""
    forecaster = XGBoostForecaster()
    config = ModelConfig(
        power_fcst_model_name="test",
        hyperparameters=XGBoostHyperparameters().model_dump(),
        features=ModelFeaturesConfig(
            nwps=[NwpModel.ECMWF_ENS_0_25DEG],
            feature_names=[
                "time_series_id",
                "lead_time_hours",
                "latest_available_weekly_power_lag",
                "temperature_2m",
                "downward_short_wave_radiation_flux_surface",
                "wind_speed_10m",
                "hour_sin",
                "hour_cos",
                "day_of_year_sin",
                "day_of_year_cos",
                "day_of_week",
            ],
        ),
    )

    # Inject Null
    nwp = base_data["nwp"].with_columns(pl.lit(None).cast(pl.Float32).alias("temperature_2m"))

    with pytest.raises(
        ValueError, match="No training data remaining after dropping nulls in critical columns."
    ):
        forecaster.fit(
            config=config,
            power_time_series=cast(pt.LazyFrame[PowerTimeSeries], base_data["flows"]),
            time_series_metadata=cast(
                pt.DataFrame[TimeSeriesMetadata],
                base_data["metadata"],
            ),
            nwps=cast(
                Mapping[NwpModel, pt.LazyFrame[Nwp]],
                {NwpModel.ECMWF_ENS_0_25DEG: cast(pt.LazyFrame[Nwp], nwp)},
            ),
        )

    # Inject Inf
    nwp = base_data["nwp"].with_columns(pl.lit(float("inf")).alias("temperature_2m"))

    with pytest.raises(ValueError, match="Input features X contain NaN or Inf values"):
        forecaster.fit(
            config=config,
            power_time_series=cast(pt.LazyFrame[PowerTimeSeries], base_data["flows"]),
            time_series_metadata=cast(
                pt.DataFrame[TimeSeriesMetadata],
                base_data["metadata"],
            ),
            nwps=cast(
                Mapping[NwpModel, pt.LazyFrame[Nwp]],
                {NwpModel.ECMWF_ENS_0_25DEG: cast(pt.LazyFrame[Nwp], nwp)},
            ),
        )
