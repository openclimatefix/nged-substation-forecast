from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import dagster as dg
import polars as pl
import pytest
from contracts.settings import Settings
from nged_substation_forecast.definitions import defs
from nged_substation_forecast.defs.nged_assets import (
    LivePrimaryFlowsConfig,
    live_primary_flows,
    substation_metadata,
)
from nged_substation_forecast.defs.reference_data import gb_h3_grid_weights, uk_boundary
from nged_substation_forecast.defs.weather_assets import ecmwf_ens_forecast
from nged_substation_forecast.defs.data_cleaning_assets import cleaned_actuals
from geo.assets import H3GridConfig


@pytest.mark.integration
@pytest.mark.manual
def test_xgboost_dagster_integration() -> None:
    """True integration test for XGBoost pipeline inside Dagster.

    Note: This test requires actual data in the Delta tables. If data is missing,
    it will attempt to materialize it using real Dagster assets.
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
        # 2. Ensure data is available by materializing if necessary
        settings = Settings()
        actuals_path = settings.nged_data_path / "delta" / "live_primary_flows"
        cleaned_path = settings.nged_data_path / "delta" / "cleaned_actuals"

        # Define the 5 substations for the test
        substations = [110375, 110644, 110772, 110803, 110804]

        # Check if we have enough data (at least 6 days)
        has_enough_data = False
        if actuals_path.exists() and cleaned_path.exists():
            try:
                df = pl.scan_delta(str(actuals_path))
                stats = cast(
                    pl.DataFrame,
                    df.select(
                        pl.col("timestamp").min().alias("min"),
                        pl.col("timestamp").max().alias("max"),
                    ).collect(),
                )
                if not stats.is_empty():
                    min_dt = stats.get_column("min")[0]
                    max_dt = stats.get_column("max")[0]
                    if min_dt and max_dt and (max_dt - min_dt) >= timedelta(days=6):
                        has_enough_data = True
            except Exception:
                pass

        if not has_enough_data:
            test_end_date = datetime.now(timezone.utc).date() - timedelta(days=1)

            # Materialize reference data once at the beginning
            # We call the asset functions directly to avoid Dagster overhead and pickling issues
            # in the integration test setup.
            context = dg.build_asset_context(resources={"settings": settings}, instance=instance)
            sub_meta_df = substation_metadata(context)
            uk_bound = uk_boundary(context)
            grid_weights = gb_h3_grid_weights(context, config=H3GridConfig(), uk_boundary=uk_bound)

            # Materialize live_primary_flows for the latest partition to get the full history
            # (The asset downloads the full history for the requested substations)
            lp_ctx = dg.build_asset_context(
                partition_key=test_end_date.isoformat(),
                resources={"settings": settings},
                instance=instance,
            )
            for _ in live_primary_flows(  # type: ignore
                lp_ctx, config=LivePrimaryFlowsConfig(substation_numbers=substations)
            ):
                pass

            # Materialize historical data in a loop (45 days)
            # We use a loop to ensure each daily partition is created.
            # In the loop, only materialize 'ecmwf_ens_forecast' and 'cleaned_actuals'.
            for i in range(45):
                date_str = (test_end_date - timedelta(days=i)).isoformat()
                ctx = dg.build_asset_context(
                    partition_key=date_str, resources={"settings": settings}, instance=instance
                )

                try:
                    # Skip NWP download if file already exists to speed up retries
                    nwp_filename = f"{date_str}T00Z.parquet"
                    nwp_path = settings.nwp_data_path / "ECMWF" / "ENS" / nwp_filename
                    if not nwp_path.exists():
                        ecmwf_ens_forecast(ctx, gb_h3_grid_weights=grid_weights)

                    cleaned_actuals(ctx, substation_metadata=sub_meta_df)
                except Exception as e:
                    # We catch exceptions here to make the setup robust to occasional data quality
                    # issues in the real-world datasets. If enough days are materialized, the
                    # test can still proceed.
                    print(f"Warning: Failed to materialize data for {date_str}: {e}")
                    continue

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

        test_end = cast(datetime, max_dt).date()
        test_start = test_end - timedelta(days=2)
        train_end = test_start - timedelta(days=1)
        train_start = cast(datetime, min_dt).date()

        # Safety check: ensure train_start <= train_end
        if train_start > train_end:
            pytest.skip("Calculated train_start is after train_end. Insufficient data.")

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
                        "start_date": str(train_start),
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
