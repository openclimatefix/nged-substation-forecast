from unittest.mock import patch
import polars as pl
from src.nged_substation_forecast.defs.plotting_assets import forecast_vs_actual_plot, PlotConfig
from contracts.settings import Settings
from contracts.data_schemas import SubstationMetadata
from datetime import datetime, timezone
import dagster as dg


@patch("src.nged_substation_forecast.defs.plotting_assets.get_cleaned_actuals_lazy")
def test_forecast_vs_actual_plot_filters_actuals(mock_get_lazy):
    """Test that the forecast vs actual plot asset doesn't crash with many actuals."""
    # Mock predictions for only 1 substation
    predictions = pl.DataFrame(
        {
            "valid_time": [datetime(2026, 3, 1, 12, tzinfo=timezone.utc)],
            "substation_number": [110375],
            "ensemble_member": [0],
            "MW_or_MVA": [10.0],
            "nwp_init_time": [datetime(2026, 3, 1, 0, tzinfo=timezone.utc)],
            "power_fcst_model_name": ["test"],
            "power_fcst_init_time": [datetime(2026, 3, 1, 3, tzinfo=timezone.utc)],
            "power_fcst_init_year_month": ["2026-03"],
        }
    )

    # Mock cleaned_actuals with many substations
    actuals = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 3, 1, 12, tzinfo=timezone.utc)] * 11,
            "substation_number": list(range(10)) + [110375],
            "MW": [1.0] * 11,
            "MVA": [1.0] * 11,
            "MW_or_MVA": [1.0] * 11,
        }
    )
    mock_get_lazy.return_value = actuals.lazy()

    # Mock substation_metadata using Patito
    substation_metadata = SubstationMetadata.validate(
        pl.DataFrame(
            {
                "substation_number": [110375],
                "substation_name_in_location_table": ["Test Substation"],
                "substation_type": ["Primary"],
                "last_updated": [datetime(2026, 3, 1, tzinfo=timezone.utc)],
            }
        ).cast({"substation_number": pl.Int32, "substation_type": pl.Categorical}),
        allow_missing_columns=True,
    )

    config = PlotConfig(output_path="tests/temp_test_plot.html")
    settings = Settings()
    context = dg.build_asset_context()

    # We want to verify that forecast_vs_actual_plot doesn't crash
    try:
        result = forecast_vs_actual_plot(
            context=context,
            predictions=predictions,
            substation_metadata=substation_metadata,
            config=config,
            settings=settings,
        )
        assert result is not None
    finally:
        import os

        if os.path.exists("tests/temp_test_plot.html"):
            os.remove("tests/temp_test_plot.html")


@patch("src.nged_substation_forecast.defs.plotting_assets.get_cleaned_actuals_lazy")
def test_forecast_vs_actual_plot_handles_no_overlap(mock_get_lazy):
    predictions = pl.DataFrame(
        {
            "valid_time": [datetime(2026, 3, 1, 12, tzinfo=timezone.utc)],
            "substation_number": [110375],
            "ensemble_member": [0],
            "MW_or_MVA": [10.0],
            "nwp_init_time": [datetime(2026, 3, 1, 0, tzinfo=timezone.utc)],
        }
    )

    # Actuals at a different time
    actuals = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1, 12, tzinfo=timezone.utc)],
            "substation_number": [110375],
            "MW": [1.0],
            "MVA": [1.0],
            "MW_or_MVA": [1.0],
        }
    )
    mock_get_lazy.return_value = actuals.lazy()

    # Mock substation_metadata using Patito
    substation_metadata = SubstationMetadata.validate(
        pl.DataFrame(
            {
                "substation_number": [110375],
                "substation_name_in_location_table": ["Test Substation"],
                "substation_type": ["Primary"],
                "last_updated": [datetime(2026, 3, 1, tzinfo=timezone.utc)],
            }
        ).cast({"substation_number": pl.Int32, "substation_type": pl.Categorical}),
        allow_missing_columns=True,
    )

    config = PlotConfig(output_path="tests/test_plot_no_overlap.html")

    # Use dagster's build_asset_context
    context = dg.build_asset_context()

    result = forecast_vs_actual_plot(
        context=context,
        predictions=predictions,
        substation_metadata=substation_metadata,
        config=config,
        settings=Settings(),
    )

    assert result is None
