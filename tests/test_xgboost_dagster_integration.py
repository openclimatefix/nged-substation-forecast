from datetime import date, datetime, timedelta
from pathlib import Path
from typing import cast

import dagster as dg
import polars as pl
import pytest
from contracts.settings import Settings
from nged_substation_forecast.definitions import defs


@pytest.mark.integration
def test_xgboost_dagster_integration():
    """True integration test for XGBoost pipeline inside Dagster."""
    # 1. Get the job from definitions
    job = defs.get_job_def("xgboost_integration_job")

    # 2. Dynamically calculate dates from local data if available, otherwise use defaults
    settings = Settings()
    actuals_path = settings.nged_data_path / "delta" / "live_primary_flows"

    if actuals_path.exists():
        df = pl.scan_delta(str(actuals_path))
        max_dt_df = cast(pl.DataFrame, df.select(pl.col("timestamp").max()).collect())
        max_dt = max_dt_df.get_column("timestamp").max()
        if max_dt is None:
            train_start, train_end = date(2026, 1, 1), date(2026, 2, 28)
            test_start, test_end = date(2026, 3, 1), date(2026, 3, 14)
        else:
            test_end = cast(datetime, max_dt).date()
            test_start = test_end - timedelta(days=14)
            train_end = test_start - timedelta(days=1)
            min_dt_df = cast(pl.DataFrame, df.select(pl.col("timestamp").min()).collect())
            min_dt = min_dt_df.get_column("timestamp").min()
            if min_dt is not None:
                train_start = cast(datetime, min_dt).date()
            else:
                train_start = train_end - timedelta(days=60)

            # Safety check: ensure train_start <= train_end
            # If the local data range is too small, fall back to a minimum training window
            # of 30 days (even if it extends before the available data) to prevent empty sets.
            if train_start > train_end:
                train_start = train_end - timedelta(days=30)
    else:
        # Fallback for CI or if data is missing (Dagster will attempt to download)
        train_start = date(2026, 1, 1)
        train_end = date(2026, 2, 28)
        test_start = date(2026, 3, 1)
        test_end = date(2026, 3, 14)

    # 3. Define the 5 substations for the test
    substations = [110375, 110644, 110772, 110803, 110804]

    # 4. Provide run configuration
    run_config = {
        "ops": {
            "live_primary_flows": {
                "config": {
                    "substation_numbers": substations,
                    "limit": 5,
                }
            },
            "processed_nwp_data": {
                "config": {
                    "substation_ids": substations,
                }
            },
            "train_xgboost": {
                "config": {
                    "train_start": str(train_start),
                    "train_end": str(train_end),
                    "substation_ids": substations,
                }
            },
            "evaluate_xgboost": {
                "config": {
                    "test_start": str(test_start),
                    "test_end": str(test_end),
                    "substation_ids": substations,
                }
            },
            "forecast_vs_actual_plot": {
                "config": {
                    "output_path": "tests/xgboost_dagster_integration_plot.html",
                }
            },
        }
    }

    # 5. Execute the job in-process
    # We use the resources from the definitions (which includes the real settings)
    # We override the io_manager to use mem_io_manager to avoid pickling issues with Patito DataFrames
    resources = defs.resources or {}
    result = job.execute_in_process(
        run_config=run_config,
        resources={
            **resources,
            "io_manager": dg.mem_io_manager,
        },
        partition_key=str(test_end),
    )

    # 6. Assertions
    assert result.success, "Dagster job failed"

    # Verify plot generation
    plot_path = Path("tests/xgboost_dagster_integration_plot.html")
    assert plot_path.exists(), "Integration plot was not generated"
    assert plot_path.stat().st_size > 0, "Integration plot is empty"

    # Read HTML content for further verification
    html_content = plot_path.read_text()

    # Verify substation name is present (110375 is Woodland Way)
    assert "Woodland Way" in html_content, "Substation name 'Woodland Way' not found in plot HTML"

    # Verify independent y-axes configuration
    # We remove spaces to make the check more robust against formatting changes
    assert '"resolve":{"scale":{"y":"independent"}}' in html_content.replace(" ", "").replace(
        "\n", ""
    ), "Independent y-axis configuration not found in plot HTML"

    # Verify NWP init time is in the title/subtitle
    # Get the materialization event for the plot asset to find the chosen_init_time
    plot_materialization = next(
        event
        for event in result.get_asset_materialization_events()
        if event.asset_key and event.asset_key.path[-1] == "forecast_vs_actual_plot"
    )
    chosen_init_time_str = plot_materialization.materialization.metadata["chosen_init_time"].value
    assert f"NWP Init Time: {chosen_init_time_str}" in html_content, (
        f"NWP Init Time '{chosen_init_time_str}' not found in plot HTML"
    )

    # Clean up plot after successful test
    # os.remove(plot_path)
