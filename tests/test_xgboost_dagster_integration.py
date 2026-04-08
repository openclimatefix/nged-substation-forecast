import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import dagster as dg
import polars as pl
from contracts.settings import Settings
from nged_substation_forecast.definitions import defs


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.manual
def test_xgboost_dagster_integration() -> None:
    """True integration test for XGBoost pipeline inside Dagster.

    Note: This test requires actual data in the Delta tables and NWP parquet files.
    If data is missing, it will skip the test.
    Run manually with:
        uv run pytest tests/test_xgboost_dagster_integration.py -v -m manual

    The generated plot is saved to `tests/xgboost_dagster_integration_plot.html`
    for manual inspection.
    """
    # We use resolve_job_def instead of get_job_def to avoid the UnresolvedAssetJobDefinition
    # deprecation warning in newer Dagster versions. This ensures we correctly retrieve
    # the job definition.
    job = defs.resolve_job_def("xgboost_integration_job")

    # We use an ephemeral Dagster instance to ensure that all resources,
    # including SQLite databases and SQLAlchemy connection pools, are
    # properly cleaned up after the job execution. This prevents
    # "Cannot operate on a closed database" errors in tests.
    with dg.DagsterInstance.ephemeral() as instance:
        settings = Settings()
        actuals_path = settings.nged_data_path / "delta" / "live_primary_flows"
        cleaned_path = settings.nged_data_path / "delta" / "cleaned_actuals"

        if not actuals_path.exists() or not cleaned_path.exists():
            pytest.skip("Required Delta tables missing. Please download data.")

        # Define the 5 substations for the test
        substations = [110375, 110644, 110772, 110803, 110804]

        # 3. Dynamically calculate dates from local data
        df = pl.scan_delta(str(actuals_path))
        max_dt_df = cast(pl.DataFrame, df.select(pl.col("timestamp").max()).collect())
        max_dt = max_dt_df.get_column("timestamp").max()

        if max_dt is None:
            pytest.skip("No maximum timestamp found in data.")

        min_dt_df = cast(pl.DataFrame, df.select(pl.col("timestamp").min()).collect())
        min_dt = min_dt_df.get_column("timestamp").min()

        if min_dt is None:
            pytest.skip("No minimum timestamp found in data.")

        # Check for sufficient data duration
        total_duration = cast(datetime, max_dt) - cast(datetime, min_dt)
        if total_duration < timedelta(days=6):
            pytest.skip(
                f"Insufficient data for integration test. Found {total_duration.days} days, require at least 6."
            )

        # 1. Select an NWP initialization time from exactly two weeks ago.
        today = datetime.now(timezone.utc).date()
        nwp_init_time = today - timedelta(weeks=2)

        # 2. Configure the `evaluate_xgboost` op to run inference for that specific NWP run,
        # covering a 2-week prediction horizon.
        test_start = nwp_init_time
        test_end = nwp_init_time + timedelta(weeks=2)

        # Check for required NWP data
        for i in range((test_end - nwp_init_time).days + 1):
            date = nwp_init_time + timedelta(days=i)
            date_str = date.isoformat()
            nwp_filename = f"{date_str}T00Z.parquet"
            nwp_path = settings.nwp_data_path / "ECMWF" / "ENS" / nwp_filename
            if not nwp_path.exists():
                pytest.skip(f"Required NWP file missing: {nwp_path}. Please download data.")

        # 3. Ensure the `train_xgboost` op is configured to train only on data *before* that 2-week period,
        # ensuring no data leakage.
        train_end = test_start - timedelta(days=1)
        min_dt_date = cast(datetime, min_dt).date()
        train_start = max(min_dt_date, train_end - timedelta(weeks=4))

        # Safety check: ensure train_start <= train_end
        if train_start > train_end:
            pytest.skip("Calculated train_start is after train_end. Insufficient data.")

        print(f"train_start: {train_start}, train_end: {train_end}")
        print(f"nwp_init_time: {nwp_init_time}")
        print(f"test_start: {test_start}, test_end: {test_end}")

        # 4. Define plot output path
        plot_path = Path("tests/xgboost_dagster_integration_plot.html")

        # 5. Provide run configuration
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
                        "start_date": str(nwp_init_time),
                        "end_date": str(test_end),
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
                        "output_path": str(plot_path),
                    }
                },
            }
        }

        # 6. Execute the job in-process
        # Note: This test takes approximately 3.5 minutes (215 seconds) to run on a
        # standard development machine as it executes the full XGBoost pipeline
        # (data loading, cleaning, training, evaluation, and plotting).
        resources = defs.resources or {}

        result = job.execute_in_process(
            run_config=run_config,
            partition_key=test_end.isoformat(),
            resources={
                **resources,
                "io_manager": dg.mem_io_manager,
            },
            instance=instance,
        )

    # 7. Assertions
    assert result.success, "Dagster job failed"

    # Verify ensemble forecasts
    predictions = result.output_for_node("evaluate_xgboost")
    assert predictions["ensemble_member"].n_unique() > 1, (
        "Model did not output multiple ensemble members"
    )

    # Check if predictions are empty
    assert len(predictions) > 0, "No predictions were generated by the model."

    # Verify plot generation
    materializations = [
        event
        for event in result.get_asset_materialization_events()
        if event.asset_key and event.asset_key.path[-1] == "forecast_vs_actual_plot"
    ]
    assert materializations, "Asset 'forecast_vs_actual_plot' was not materialized."
    plot_materialization = materializations[0]

    assert "chosen_init_time" in plot_materialization.materialization.metadata, (
        "Plot metadata missing 'chosen_init_time'"
    )

    assert plot_path.exists(), "Integration plot was not generated"
    assert plot_path.stat().st_size > 0, "Integration plot is empty"

    html_content = plot_path.read_text()

    # Dynamically check that at least one requested substation ID is in the HTML
    found_substation = any(str(sub_id) in html_content for sub_id in substations)
    assert found_substation, (
        f"None of the requested substation IDs {substations} found in plot HTML"
    )

    assert '"resolve":{"scale":{"y":"independent"}}' in html_content.replace(" ", "").replace(
        "\n", ""
    ), "Independent y-axis configuration not found in plot HTML"

    chosen_init_time_str = plot_materialization.materialization.metadata["chosen_init_time"].value
    assert f"NWP Init Time: {chosen_init_time_str}" in html_content, (
        f"NWP Init Time '{chosen_init_time_str}' not found in plot HTML"
    )
