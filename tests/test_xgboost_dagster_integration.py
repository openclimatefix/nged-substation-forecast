from datetime import date, datetime, timedelta
from pathlib import Path
from typing import cast

import dagster as dg
import polars as pl
import pytest
from contracts.settings import Settings
from nged_substation_forecast.definitions import defs


@pytest.mark.integration
@pytest.mark.manual
def test_xgboost_dagster_integration():
    """True integration test for XGBoost pipeline inside Dagster.

    Note: This test requires actual data in the Delta tables. In CI/CD environments
    without real NGED data, this test is skipped. Run manually with:
        uv run pytest tests/test_xgboost_dagster_integration.py -v -m manual
    """
    # 1. Get the job from definitions
    job = defs.get_job_def("xgboost_integration_job")

    # 2. Dynamically calculate dates from local data if available, otherwise use defaults
    settings = Settings()
    actuals_path = settings.nged_data_path / "delta" / "live_primary_flows"

    if actuals_path.exists():
        df = pl.scan_delta(str(actuals_path))
        max_dt_df = cast(pl.DataFrame, df.select(pl.col("timestamp").max()).collect())
        max_dt = cast(datetime | None, max_dt_df.get_column("timestamp").max())
        if max_dt is None:
            train_start, train_end = date(2026, 1, 1), date(2026, 2, 28)
            test_start, test_end = date(2026, 3, 1), date(2026, 3, 14)
        else:
            min_dt_df = cast(pl.DataFrame, df.select(pl.col("timestamp").min()).collect())
            min_dt = cast(datetime | None, min_dt_df.get_column("timestamp").min())

            if min_dt is None:
                pytest.skip("No minimum timestamp found in data.")

            # Check for sufficient data duration
            total_duration = max_dt - min_dt
            if total_duration < timedelta(days=30):
                pytest.skip(
                    f"Insufficient data for integration test. Found {total_duration.days} days, require at least 30."
                )

            test_end = max_dt.date()
            test_start = test_end - timedelta(days=14)
            train_end = test_start - timedelta(days=1)
            train_start = min_dt.date()

            # Safety check: ensure train_start <= train_end
            if train_start > train_end:
                pytest.skip("Calculated train_start is after train_end. Insufficient data.")
    else:
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
    resources = defs.resources or {}
    result = job.execute_in_process(
        run_config=run_config,
        resources={
            **resources,
            "io_manager": dg.mem_io_manager,
        },
    )

    # 6. Assertions
    assert result.success, "Dagster job failed"

    # Verify ensemble forecasts
    predictions = result.output_for_node("evaluate_xgboost")
    assert predictions["ensemble_member"].n_unique() > 1, (
        "Model did not output multiple ensemble members"
    )

    # Verify plot generation
    plot_path = Path("tests/xgboost_dagster_integration_plot.html")

    plot_materialization = next(
        event
        for event in result.get_asset_materialization_events()
        if event.asset_key and event.asset_key.path[-1] == "forecast_vs_actual_plot"
    )

    if "chosen_init_time" not in plot_materialization.materialization.metadata:
        pytest.skip("Plot was skipped due to missing data or no healthy substations.")

    assert plot_path.exists(), "Integration plot was not generated"
    assert plot_path.stat().st_size > 0, "Integration plot is empty"

    html_content = plot_path.read_text()

    assert "Woodland Way" in html_content, "Substation name 'Woodland Way' not found in plot HTML"

    assert '"resolve":{"scale":{"y":"independent"}}' in html_content.replace(" ", "").replace(
        "\n", ""
    ), "Independent y-axis configuration not found in plot HTML"

    chosen_init_time_str = plot_materialization.materialization.metadata["chosen_init_time"].value
    assert f"NWP Init Time: {chosen_init_time_str}" in html_content, (
        f"NWP Init Time '{chosen_init_time_str}' not found in plot HTML"
    )
