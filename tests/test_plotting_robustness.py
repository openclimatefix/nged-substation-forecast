from unittest.mock import patch
import polars as pl
from src.nged_substation_forecast.defs.plotting_assets import forecast_vs_actual_plot, PlotConfig
from contracts.settings import Settings
from contracts.data_schemas import TimeSeriesMetadata
from datetime import datetime, timezone
import dagster as dg


@patch("src.nged_substation_forecast.defs.plotting_assets.pl.read_parquet")
@patch("src.nged_substation_forecast.defs.plotting_assets.get_cleaned_actuals_lazy")
def test_forecast_vs_actual_plot_filters_actuals(mock_get_lazy, mock_read_parquet):
    """Test that the forecast vs actual plot asset doesn't crash with many actuals."""
    # Mock predictions for only 1 substation
    predictions = pl.DataFrame(
        {
            "valid_time": [datetime(2026, 3, 1, 12, tzinfo=timezone.utc)],
            "time_series_id": [110375],
            "ensemble_member": [0],
            "power_fcst": [10.0],
            "nwp_init_time": [datetime(2026, 3, 1, 0, tzinfo=timezone.utc)],
            "power_fcst_model_name": ["test"],
            "power_fcst_init_time": [datetime(2026, 3, 1, 3, tzinfo=timezone.utc)],
            "power_fcst_init_year_month": ["2026-03"],
        }
    )

    # Mock cleaned_actuals with many substations
    actuals = pl.DataFrame(
        {
            "period_end_time": [datetime(2026, 3, 1, 12, tzinfo=timezone.utc)] * 11,
            "time_series_id": list(range(10)) + [110375],
            "power": [1.0] * 11,
        }
    )
    mock_get_lazy.return_value = actuals.lazy()

    # Mock time_series_metadata using Patito
    time_series_metadata = TimeSeriesMetadata.validate(
        pl.DataFrame(
            {
                "time_series_id": [110375],
                "time_series_name": ["Test Substation"],
                "time_series_type": ["Disaggregated Demand"],
                "units": ["MW"],
                "licence_area": ["EMids"],
                "substation_number": [110375],
                "substation_type": ["Primary"],
                "latitude": [52.0],
                "longitude": [0.0],
            }
        ).cast(
            {
                "time_series_id": pl.Int32,
                "substation_number": pl.Int32,
                "substation_type": pl.Categorical,
                "time_series_type": pl.String,
                "latitude": pl.Float32,
                "longitude": pl.Float32,
            }
        ),
        allow_missing_columns=True,
        allow_superfluous_columns=True,
    )
    mock_read_parquet.return_value = time_series_metadata

    config = PlotConfig(output_path="tests/temp_test_plot.html")
    settings = Settings()
    with dg.build_asset_context() as context:
        # We want to verify that forecast_vs_actual_plot doesn't crash
        try:
            result = forecast_vs_actual_plot(
                context=context,
                predictions=predictions,
                config=config,
                settings=settings,
            )
            assert result is not None
        finally:
            import os

            if os.path.exists("tests/temp_test_plot.html"):
                os.remove("tests/temp_test_plot.html")


@patch("src.nged_substation_forecast.defs.plotting_assets.pl.read_parquet")
@patch("src.nged_substation_forecast.defs.plotting_assets.get_cleaned_actuals_lazy")
def test_forecast_vs_actual_plot_handles_no_overlap(mock_get_lazy, mock_read_parquet):
    predictions = pl.DataFrame(
        {
            "valid_time": [datetime(2026, 3, 1, 12, tzinfo=timezone.utc)],
            "time_series_id": [110375],
            "ensemble_member": [0],
            "power_fcst": [10.0],
            "nwp_init_time": [datetime(2026, 3, 1, 0, tzinfo=timezone.utc)],
        }
    )

    # Actuals at a different time
    actuals = pl.DataFrame(
        {
            "period_end_time": [datetime(2026, 1, 1, 12, tzinfo=timezone.utc)],
            "time_series_id": [110375],
            "power": [1.0],
        }
    )
    mock_get_lazy.return_value = actuals.lazy()

    # Mock time_series_metadata using Patito
    time_series_metadata = TimeSeriesMetadata.validate(
        pl.DataFrame(
            {
                "time_series_id": [110375],
                "time_series_name": ["Test Substation"],
                "time_series_type": ["Disaggregated Demand"],
                "units": ["MW"],
                "licence_area": ["EMids"],
                "substation_number": [110375],
                "substation_type": ["Primary"],
                "latitude": [52.0],
                "longitude": [0.0],
            }
        ).cast(
            {
                "time_series_id": pl.Int32,
                "substation_number": pl.Int32,
                "substation_type": pl.Categorical,
                "time_series_type": pl.String,
                "latitude": pl.Float32,
                "longitude": pl.Float32,
            }
        ),
        allow_missing_columns=True,
        allow_superfluous_columns=True,
    )
    mock_read_parquet.return_value = time_series_metadata

    config = PlotConfig(output_path="tests/test_plot_no_overlap.html")

    # Use dagster's build_asset_context
    with dg.build_asset_context() as context:
        result = forecast_vs_actual_plot(
            context=context,
            predictions=predictions,
            config=config,
            settings=Settings(),
        )

        assert result is None
