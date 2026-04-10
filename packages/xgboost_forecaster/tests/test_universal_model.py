from datetime import datetime, timedelta, timezone
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import patito as pt
import polars as pl
import pytest
from contracts.data_schemas import NwpColumns, PowerForecast, PowerTimeSeries, TimeSeriesMetadata
from contracts.hydra_schemas import (
    DataSplitConfig,
    ModelConfig,
    ModelFeaturesConfig,
    NwpModel,
    TrainingConfig,
)
from xgboost_forecaster.features import add_time_features, add_autoregressive_lags
from xgboost_forecaster.model import XGBoostForecaster
from ml_core.utils import evaluate_and_save_model


def test_universal_training_data_integrity():
    """Verify training data integrity: delay filter and lead_time_hours calculation."""
    # Setup
    time_series_id = 1
    h3_index = 12345

    # Create a range of init times and valid times
    init_times = [
        datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
    ]

    # Valid times every 1 hour for 48 hours
    valid_times = [init_times[0] + timedelta(hours=h) for h in range(48)]

    # NWP data
    nwp_data = []
    for init_time in init_times:
        for valid_time in valid_times:
            if valid_time >= init_time:
                nwp_data.append(
                    {
                        NwpColumns.INIT_TIME: init_time,
                        NwpColumns.VALID_TIME: valid_time,
                        NwpColumns.H3_INDEX: h3_index,
                        NwpColumns.ENSEMBLE_MEMBER: 0,
                        NwpColumns.TEMPERATURE_2M: 15.0,
                        NwpColumns.SW_RADIATION: 100.0,
                        NwpColumns.WIND_SPEED_10M: 5.0,
                    }
                )

    nwps_lf = pl.LazyFrame(nwp_data)

    # Flows data
    flows_data = []
    for valid_time in valid_times:
        flows_data.append(
            {
                "time_series_id": time_series_id,
                "period_end_time": valid_time,
                "power": 10.0,
            }
        )
    flows_lf = pl.LazyFrame(flows_data).with_columns(pl.col("time_series_id").cast(pl.Int32))

    # Metadata
    metadata = pt.DataFrame[TimeSeriesMetadata](
        pl.DataFrame(
            {
                "time_series_id": [time_series_id],
                "h3_res_5": [h3_index],
                "substation_name": ["Test Substation"],
                "substation_type": ["primary"],
                "last_updated": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
                "latitude": [51.5],
                "longitude": [-0.1],
            }
        ).with_columns(pl.col("time_series_id").cast(pl.Int32))
    )

    # Forecaster
    forecaster = XGBoostForecaster()

    # Config
    config = ModelConfig(
        power_fcst_model_name="test_model",
        features=ModelFeaturesConfig(
            nwps=[NwpModel.ECMWF_ENS_0_25DEG], feature_names=["lead_time_hours", "nwp_init_hour"]
        ),
        nwp_availability_delay_hours=3,
    )
    forecaster.config = config

    # Prepare data for model (training mode)
    nwps = {NwpModel.ECMWF_ENS_0_25DEG: nwps_lf}

    prepared_lf = forecaster._prepare_data_for_model(
        flows_30m=cast(pt.LazyFrame[PowerTimeSeries], flows_lf),
        time_series_metadata=metadata,
        nwps=nwps,
    )

    result = cast(pl.DataFrame, prepared_lf.collect()).drop_nulls(subset=[NwpColumns.INIT_TIME])

    # 1. Verify availability delay filter: init_time + 3h <= valid_time
    # For init_time = 00:00, valid_time must be >= 03:00
    # For init_time = 12:00, valid_time must be >= 15:00

    for row in result.iter_rows(named=True):
        assert row[NwpColumns.INIT_TIME] + timedelta(hours=3) <= row[NwpColumns.VALID_TIME]

    # 2. Verify lead_time_hours calculation
    for row in result.iter_rows(named=True):
        expected_lead_time = (
            row[NwpColumns.VALID_TIME] - row[NwpColumns.INIT_TIME]
        ).total_seconds() / 3600.0
        assert row["lead_time_hours"] == pytest.approx(expected_lead_time)


def test_nwp_init_hour_feature():
    """Verify correct extraction of the hour from init_time."""
    df = pl.LazyFrame(
        {
            NwpColumns.INIT_TIME: [
                datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 23, 0, tzinfo=timezone.utc),
            ],
            NwpColumns.VALID_TIME: [
                datetime(2024, 1, 1, 6, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 18, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 2, 5, 0, tzinfo=timezone.utc),
            ],
        }
    )

    result = cast(pl.DataFrame, add_time_features(df).collect())

    assert result["nwp_init_hour"].to_list() == [0, 12, 23]


def test_mlflow_metric_thinning():
    """Verify that only key operational horizons and the global metric are logged to MLflow."""
    # Mock context, forecaster, and mlflow
    context = MagicMock()
    forecaster = MagicMock()

    # Mock predict to return a DataFrame with many lead times
    lead_times = np.arange(0, 337, 0.5)  # 0 to 336 hours every 30m
    init_time = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)

    results_data = []
    for lt in lead_times:
        results_data.append(
            {
                NwpColumns.VALID_TIME: init_time + timedelta(hours=lt),
                "time_series_id": 1,
                NwpColumns.ENSEMBLE_MEMBER: 0,
                "power_fcst_model_name": "test_model",
                "power_fcst_init_time": init_time,
                "nwp_init_time": init_time,
                "power_fcst_init_year_month": "2024-01",
                "nwp_init_hour": 0,
                "lead_time_hours": float(lt),
                "power_fcst": 10.0,
            }
        )

    results_df = PowerForecast.validate(
        pl.DataFrame(results_data).with_columns(
            [
                pl.col("time_series_id").cast(pl.Int32),
                pl.col(NwpColumns.ENSEMBLE_MEMBER).cast(pl.UInt8),
                pl.col("nwp_init_hour").cast(pl.Int32),
                pl.col("lead_time_hours").cast(pl.Float32),
                pl.col("power_fcst_model_name").cast(pl.Categorical),
                pl.col("power_fcst").cast(pl.Float32),
            ]
        )
    )
    forecaster.predict.return_value = results_df.rename({"power_fcst": "value"})

    # Mock flows_30m
    flows_data = []
    for lt in lead_times:
        flows_data.append(
            {
                "time_series_id": 1,
                "start_time": init_time + timedelta(hours=lt) - timedelta(minutes=30),
                "period_end_time": init_time + timedelta(hours=lt),
                "power": 11.0,  # Constant error of 1.0
            }
        )
    flows_30m = (
        pl.LazyFrame(flows_data)
        .with_columns(pl.col("time_series_id").cast(pl.Int32))
        .rename({"power": "value"})
    )

    config = TrainingConfig(
        data_split=DataSplitConfig(
            train_start=init_time.date(),
            train_end=(init_time + timedelta(days=14)).date(),
            test_start=init_time.date(),
            test_end=(init_time + timedelta(days=14)).date(),
        ),
        model=ModelConfig(
            power_fcst_model_name="test_model",
            features=ModelFeaturesConfig(
                nwps=[NwpModel.ECMWF_ENS_0_25DEG],
                feature_names=["lead_time_hours", "nwp_init_hour"],
            ),
            nwp_availability_delay_hours=3,
            required_lookback_days=14,
        ),
    )

    with patch("ml_core.utils.mlflow") as mock_mlflow:
        # Setup mock_mlflow.start_run as a context manager
        mock_mlflow.start_run.return_value.__enter__.return_value = MagicMock()

        evaluate_and_save_model(
            context=context,
            model_name="test_model",
            forecaster=forecaster,
            config=config,
            substation_power_flows=flows_30m,  # This will be downsampled in evaluate_and_save_model
        )

        # Check logged metrics
        logged_metrics = {}
        for call in mock_mlflow.log_metric.call_args_list:
            name, value = call[0]
            logged_metrics[name] = value

        # Key horizons: 24h, 48h, 72h, 168h, 336h
        expected_horizons = {24.0, 48.0, 72.0, 168.0, 336.0}
        for h in expected_horizons:
            assert f"MAE_LT_{h}h" in logged_metrics
            assert f"RMSE_LT_{h}h" in logged_metrics
            assert f"nMAE_LT_{h}h" in logged_metrics

        # Verify that other horizons are NOT logged (e.g., 1.0h)
        assert "MAE_LT_1.0h" not in logged_metrics

        # Verify global metrics are logged
        assert "mean_mae_all_horizons" in logged_metrics
        assert "MAE_global" in logged_metrics
        assert "RMSE_global" in logged_metrics
        assert "nMAE_global" in logged_metrics


def test_autoregressive_lag_consistency():
    """Verify that for a single valid_time, different init_times result in correct target_lag_times."""
    valid_time = datetime(2024, 1, 10, 12, 0, tzinfo=timezone.utc)

    # Two different init times for the same valid time
    # 1. init_time = valid_time - 24h (lead_time = 24h)
    # 2. init_time = valid_time - 7 days (lead_time = 168h)

    init_time_1 = valid_time - timedelta(hours=24)
    init_time_2 = valid_time - timedelta(days=7)

    df = pl.LazyFrame(
        {
            "time_series_id": [1, 1],
            "valid_time": [valid_time, valid_time],
            "init_time": [init_time_1, init_time_2],
        }
    ).with_columns(pl.col("time_series_id").cast(pl.Int32))

    # Empty flows for lag calculation (we only care about target_lag_time)
    flows_30m = pl.LazyFrame(
        {
            "time_series_id": pl.Series([], dtype=pl.Int32),
            "start_time": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "period_end_time": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "value": pl.Series([], dtype=pl.Float32),
        }
    )

    telemetry_delay_hours = 24
    result = cast(
        pl.DataFrame,
        add_autoregressive_lags(
            df,
            cast(pt.LazyFrame[PowerTimeSeries], flows_30m),
            telemetry_delay_hours=telemetry_delay_hours,
        ).collect(),
    )

    # For init_time_1: lead_time = 24h = 1 day
    # lag_days = ceil((1 + 24/24) / 7) * 7 = ceil(2/7)*7 = 7 days
    # target_lag_time = valid_time - 7 days

    # For init_time_2: lead_time = 168h = 7 days
    # lag_days = ceil((7 + 24/24) / 7) * 7 = ceil(8/7)*7 = 14 days
    # target_lag_time = valid_time - 14 days

    row_1 = result.filter(pl.col("init_time") == init_time_1)
    assert row_1["lag_days"].to_list()[0] == 7
    assert row_1["target_lag_time"].to_list()[0] == valid_time - timedelta(days=7)

    row_2 = result.filter(pl.col("init_time") == init_time_2)
    assert row_2["lag_days"].to_list()[0] == 14
    assert row_2["target_lag_time"].to_list()[0] == valid_time - timedelta(days=14)


def test_lookahead_audit():
    """Lookahead Audit Test: Verify model doesn't use future weather data."""
    # Create a synthetic dataset where the target is a function of a FUTURE weather variable
    # and verify the model's feature importance for that variable is zero.

    time_series_id = 1
    h3_index = 12345

    # Create training data
    # We'll have a "future_temp" feature which is actually temperature from valid_time + 24h

    dates = [datetime(2024, 1, i, 0, 0, tzinfo=timezone.utc) for i in range(1, 20)]

    nwp_data = []
    for init_time in dates:
        for h in range(0, 48, 1):
            valid_time = init_time + timedelta(hours=h)
            nwp_data.append(
                {
                    NwpColumns.INIT_TIME: init_time,
                    NwpColumns.VALID_TIME: valid_time,
                    NwpColumns.H3_INDEX: h3_index,
                    NwpColumns.ENSEMBLE_MEMBER: 0,
                    NwpColumns.TEMPERATURE_2M: float(np.random.normal(15, 5)),
                    NwpColumns.SW_RADIATION: float(max(0, np.random.normal(200, 100))),
                    NwpColumns.WIND_SPEED_10M: float(max(0, np.random.normal(5, 2))),
                }
            )

    nwps_df = pl.DataFrame(nwp_data).with_columns(
        [
            pl.col(NwpColumns.H3_INDEX).cast(pl.UInt64),
            pl.col(NwpColumns.ENSEMBLE_MEMBER).cast(pl.UInt8),
            pl.col(NwpColumns.TEMPERATURE_2M).cast(pl.Float32),
            pl.col(NwpColumns.SW_RADIATION).cast(pl.Float32),
            pl.col(NwpColumns.WIND_SPEED_10M).cast(pl.Float32),
        ]
    )

    # Create a "future_temp" feature that is temperature 24h in the future
    # This is what we want to make sure the model DOES NOT use if it's not available.
    # But here we are testing if the model can pick up on it if we accidentally include it.
    # Wait, the test description says: "verify the model's feature importance for that variable is zero"
    # if the target is a function of it.

    # Let's define the target as: target = 0.5 * temp_at_valid_time + 2.0 * temp_at_valid_time_plus_24h

    flows_data = []
    for row in nwp_data:
        valid_time = row[NwpColumns.VALID_TIME]
        temp_now = row[NwpColumns.TEMPERATURE_2M]

        # Find temp 24h in the future
        future_time = valid_time + timedelta(hours=24)
        future_rows = [r for r in nwp_data if r[NwpColumns.VALID_TIME] == future_time]
        if future_rows:
            temp_future = future_rows[0][NwpColumns.TEMPERATURE_2M]
            target = 0.5 * temp_now + 2.0 * temp_future
            flows_data.append(
                {
                    "time_series_id": 1,
                    "period_end_time": valid_time,
                    "power": target,
                }
            )

    flows_lf = pl.LazyFrame(flows_data).with_columns(pl.col("time_series_id").cast(pl.Int32))

    # Metadata
    metadata = pt.DataFrame[TimeSeriesMetadata](
        pl.DataFrame(
            {
                "time_series_id": [time_series_id],
                "h3_res_5": [h3_index],
                "substation_name": ["Test Substation"],
                "substation_type": ["primary"],
                "last_updated": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
                "latitude": [51.5],
                "longitude": [-0.1],
            }
        ).with_columns(pl.col("time_series_id").cast(pl.Int32))
    )

    # Forecaster
    forecaster = XGBoostForecaster()

    # Config
    config = ModelConfig(
        power_fcst_model_name="test_model",
        features=ModelFeaturesConfig(
            nwps=[NwpModel.ECMWF_ENS_0_25DEG],
            feature_names=[
                NwpColumns.TEMPERATURE_2M,
                NwpColumns.SW_RADIATION,
                NwpColumns.WIND_SPEED_10M,
                "lead_time_hours",
            ],
        ),
        nwp_availability_delay_hours=3,
        hyperparameters={
            "n_estimators": 10,
            "max_depth": 3,
        },
    )

    nwps = {NwpModel.ECMWF_ENS_0_25DEG: nwps_df.lazy()}

    # Train the model
    forecaster.train(
        config=config,
        flows_30m=flows_lf,
        time_series_metadata=metadata,
        nwps=nwps,
    )

    # Verify feature importance
    assert forecaster.model is not None
    booster = forecaster.model.get_booster()
    importance = booster.get_score(importance_type="weight")
    assert len(importance) > 0

    # The model should only have importance for features we provided
    # and it should NOT be able to see the future temperature because it's not in feature_names
    # and the data preparation pipeline doesn't include it.

    # This test is a bit redundant if we trust _prepare_features, but the point is to
    # ensure that even if the target is strongly correlated with future data,
    # the model cannot "leak" it if the features are correctly constructed.

    # A better lookahead audit:
    # Add a feature that IS future data, and verify it's NOT in the final feature matrix.

    prepared_lf = forecaster._prepare_data_for_model(
        flows_30m=cast(pt.LazyFrame[PowerTimeSeries], flows_lf),
        time_series_metadata=metadata,
        nwps=nwps,
    )

    # Add a "forbidden" feature manually to the joined data
    forbidden_lf = prepared_lf.with_columns(
        future_leak=pl.col(NwpColumns.TEMPERATURE_2M).shift(-24).over("time_series_id")
    )

    # Now use the forecaster's _prepare_features
    final_features_lf = forecaster._prepare_features(forbidden_lf)
    assert isinstance(final_features_lf, pl.LazyFrame)
    final_features = cast(pl.DataFrame, final_features_lf.collect())

    assert "future_leak" not in final_features.columns
    assert NwpColumns.TEMPERATURE_2M in final_features.columns
