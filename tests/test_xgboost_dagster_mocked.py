import dagster as dg
import polars as pl
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
from pathlib import Path

from contracts.data_schemas import UTC_DATETIME_DTYPE
from src.nged_substation_forecast.defs.xgb_assets import (
    train_xgboost,
    evaluate_xgboost,
    XGBoostConfig,
)
from xgboost_forecaster.model import XGBoostForecaster
from contracts.settings import Settings


def test_xgboost_dagster_assets_materialize_with_dummy_data(tmp_path: Path):
    """Test that XGBoost assets can be materialized with dummy data using Dagster's materialize."""

    # Setup dummy data
    # Create 4 weeks of data to satisfy the dynamic lag
    timestamps = pl.datetime_range(
        datetime(2025, 12, 15, tzinfo=timezone.utc),
        datetime(2026, 1, 15, tzinfo=timezone.utc),
        "30m",
        eager=True,
    )

    sub_flows = pl.DataFrame(
        {
            "time_series_id": [1] * len(timestamps),
            "start_time": timestamps,
            "period_end_time": timestamps + timedelta(minutes=30),
            "power": [10.0] * len(timestamps),
            "ingested_at": [datetime(2026, 1, 1, tzinfo=timezone.utc)] * len(timestamps),
        }
    ).with_columns(
        [
            pl.col("time_series_id").cast(pl.Int32),
            pl.col("start_time").cast(UTC_DATETIME_DTYPE),
            pl.col("period_end_time").cast(UTC_DATETIME_DTYPE),
            pl.col("power").cast(pl.Float32),
        ]
    )

    # Write flows to Delta as the asset now loads from Delta
    delta_dir = tmp_path / "delta"
    raw_flows_path = delta_dir / "raw_power_time_series"
    cleaned_actuals_path = delta_dir / "cleaned_actuals"
    raw_flows_path.mkdir(parents=True)
    cleaned_actuals_path.mkdir(parents=True)

    sub_flows.write_delta(str(raw_flows_path))
    sub_flows.write_delta(str(cleaned_actuals_path))

    settings = Settings(nged_data_path=tmp_path)

    # NWPs for the training period AND the required 14-day lag range
    nwp_timestamps = pl.datetime_range(
        datetime(2025, 12, 15, tzinfo=timezone.utc),
        datetime(2026, 1, 15, tzinfo=timezone.utc),
        "30m",
        eager=True,
    )

    nwps = (
        pl.DataFrame(
            {
                "valid_time": nwp_timestamps,
                "init_time": nwp_timestamps - timedelta(hours=3),
                "lead_time_hours": [3.0] * len(nwp_timestamps),
                "h3_index": [123] * len(nwp_timestamps),
                "ensemble_member": [0] * len(nwp_timestamps),
                "temperature_2m": [15.0] * len(nwp_timestamps),
                "dew_point_temperature_2m": [10.0] * len(nwp_timestamps),
                "wind_speed_10m": [5.0] * len(nwp_timestamps),
                "wind_direction_10m": [180.0] * len(nwp_timestamps),
                "wind_speed_100m": [7.0] * len(nwp_timestamps),
                "wind_direction_100m": [185.0] * len(nwp_timestamps),
                "pressure_surface": [100.0] * len(nwp_timestamps),
                "pressure_reduced_to_mean_sea_level": [101.0] * len(nwp_timestamps),
                "geopotential_height_500hpa": [50.0] * len(nwp_timestamps),
                "downward_short_wave_radiation_flux_surface": [100.0] * len(nwp_timestamps),
                "categorical_precipitation_type_surface": [0.0] * len(nwp_timestamps),
            }
        )
        .with_columns(
            [
                pl.col("h3_index").cast(pl.UInt64),
                pl.col("ensemble_member").cast(pl.UInt8),
                pl.col("categorical_precipitation_type_surface").cast(pl.UInt8),
            ]
        )
        .lazy()
    )

    config = XGBoostConfig(
        train_start="2026-01-01",
        train_end="2026-01-10",
        test_start="2026-01-11",
        test_end="2026-01-15",
    )

    # Mock MLflow to avoid actual logging
    with (
        patch("mlflow.start_run"),
        patch("mlflow.log_params"),
        patch("mlflow.log_metric"),
        patch("mlflow.log_artifact"),
        patch("mlflow.xgboost.log_model"),
        patch("mlflow.set_experiment"),
    ):
        # Materialize assets
        with dg.build_asset_context() as context:
            metadata = pl.DataFrame(
                {
                    "time_series_id": [1],
                    "substation_number": [1],
                    "h3_res_5": [123],
                }
            ).with_columns(
                [
                    pl.col("time_series_id").cast(pl.Int32),
                    pl.col("substation_number").cast(pl.Int32),
                ]
            )
            metadata_dir = tmp_path / "parquet"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            metadata.write_parquet(metadata_dir / "time_series_metadata.parquet")

            model = train_xgboost(
                context=context,
                config=config,
                settings=settings,
                nwp=nwps,
            )

            assert isinstance(model, XGBoostForecaster)
            assert model.model is not None

            forecasts = evaluate_xgboost(
                context=context,
                config=config,
                settings=settings,
                model=model,
                nwp=nwps,
            )

            assert isinstance(forecasts, pl.DataFrame)
            assert not forecasts.is_empty()
            assert "power_fcst" in forecasts.columns
