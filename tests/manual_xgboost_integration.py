import dagster as dg
import polars as pl
import altair as alt
import pandas as pd
from datetime import timedelta, datetime, timezone

from contracts.settings import Settings
from contracts.hydra_schemas import NwpModel
from contracts.data_schemas import InferenceParams
from nged_substation_forecast.definitions import defs
from nged_substation_forecast.defs.xgb_assets import load_hydra_config
from ml_core.utils import train_and_log_model, _slice_temporal_data
from xgboost_forecaster.model import XGBoostForecaster
from ml_core.data import downsample_power_flows

from typing import Any, cast

# 1. Define the 5 substations
SUBSTATIONS = [110375, 110644, 110772, 110803, 110804]


def get_healthy_substations(settings: Settings) -> list[int]:
    """Filter out substations with bad telemetry."""
    actuals_path = settings.nged_data_path / "delta" / "live_primary_flows"
    df = pl.read_delta(str(actuals_path)).lazy()
    df = df.filter(pl.col("substation_number").is_in(SUBSTATIONS))

    health_df = cast(
        pl.DataFrame,
        df.with_columns(MW_or_MVA=pl.coalesce(["MW", "MVA"]), date=pl.col("timestamp").dt.date())
        .group_by(["substation_number", "date"])
        .agg(
            max_abs=pl.col("MW_or_MVA").abs().max(),
            std=pl.col("MW_or_MVA").std(),
            count=pl.col("MW_or_MVA").count(),
        )
        .filter(
            (pl.col("count") > 50)
            & ((pl.col("max_abs") < 0.5) | (pl.col("std").fill_null(0.0) < 0.01))
        )
        .select("substation_number")
        .unique()
        .collect(),
    )

    bad_substations = health_df.get_column("substation_number").to_list()
    print(f"Filtering out bad substations: {bad_substations}")

    return [s for s in SUBSTATIONS if s not in bad_substations]


# 2. Define filtered assets
@dg.asset(name="substation_metadata", compute_kind="python", group_name="nged")
def filtered_substation_metadata(settings: dg.ResourceParam[Settings]) -> pl.DataFrame:
    """Mock substation_metadata to filter to healthy substations."""
    healthy_subs = get_healthy_substations(settings)
    metadata_path = settings.nged_data_path / "parquet" / "substation_metadata.parquet"
    df = pl.read_parquet(metadata_path)
    return df.filter(pl.col("substation_number").is_in(healthy_subs))


@dg.asset(name="combined_actuals", compute_kind="python", group_name="nged")
def filtered_combined_actuals(settings: dg.ResourceParam[Settings]) -> pl.LazyFrame:
    """Mock combined_actuals to filter to healthy substations."""
    healthy_subs = get_healthy_substations(settings)
    actuals_path = settings.nged_data_path / "delta" / "live_primary_flows"
    df = pl.read_delta(str(actuals_path)).lazy()
    return df.filter(pl.col("substation_number").is_in(healthy_subs))


@dg.asset(name="all_nwp_data", compute_kind="python", group_name="weather")
def filtered_all_nwp_data(settings: dg.ResourceParam[Settings]) -> pl.LazyFrame:
    """Mock all_nwp_data to avoid running the download step."""
    return pl.scan_parquet(settings.nwp_data_path / "ECMWF" / "ENS" / "*.parquet")


@dg.asset(name="train_xgboost", compute_kind="python", group_name="models")
def mock_train_xgboost(
    context: dg.AssetExecutionContext,
    processed_nwp_data: pl.LazyFrame,
    combined_actuals: pl.LazyFrame,
    substation_metadata: pl.DataFrame,
):
    config = load_hydra_config("xgboost")

    # Dynamically calculate train/test split
    max_date_df = cast(pl.DataFrame, combined_actuals.select(pl.col("timestamp").max()).collect())
    max_date = max_date_df.item()

    test_end = max_date.date()
    test_start = test_end - timedelta(days=14)
    train_end = test_start - timedelta(days=1)

    # Use all available data before test_start for training
    min_date_df = cast(pl.DataFrame, combined_actuals.select(pl.col("timestamp").min()).collect())
    train_start = min_date_df.item().date()

    context.log.info(f"Training period: {train_start} to {train_end}")

    config.data_split.train_start = train_start
    config.data_split.train_end = train_end

    nwp_train = processed_nwp_data.filter(pl.col("ensemble_member") == 0)
    return train_and_log_model(
        context=context,
        model_name="xgboost",
        trainer=XGBoostForecaster(),
        config=config,
        nwps={NwpModel.ECMWF_ENS_0_25DEG: nwp_train},
        substation_power_flows=combined_actuals,
        substation_metadata=substation_metadata,
    )


@dg.asset(name="evaluate_xgboost", compute_kind="python", group_name="models")
def mock_evaluate_xgboost(
    context: dg.AssetExecutionContext,
    train_xgboost: Any,
    processed_nwp_data: pl.LazyFrame,
    combined_actuals: pl.LazyFrame,
    substation_metadata: pl.DataFrame,
):
    config = load_hydra_config("xgboost")

    # Dynamically calculate train/test split
    max_date_df = cast(pl.DataFrame, combined_actuals.select(pl.col("timestamp").max()).collect())
    max_date = max_date_df.item()

    test_end = max_date.date()
    test_start = test_end - timedelta(days=14)

    context.log.info(f"Testing period: {test_start} to {test_end}")

    config.data_split.test_start = test_start
    config.data_split.test_end = test_end
    forecaster = train_xgboost
    forecaster.config = config.model

    # Do inference manually to avoid the Patito bug in evaluate_and_save_model
    lookback = getattr(config.model, "required_lookback_days", 14)
    slice_start = test_start - timedelta(days=lookback)

    sliced_nwps = _slice_temporal_data(processed_nwp_data, slice_start, test_end, "valid_time")
    sliced_flows = _slice_temporal_data(combined_actuals, slice_start, test_end, "timestamp")

    forecast_time = datetime.now(timezone.utc)
    df = cast(pl.DataFrame, sliced_nwps.select(pl.col("init_time").max()).collect())
    if not df.is_empty():
        max_init = df.item()
        forecast_time = max_init + timedelta(hours=3)
        # Only show predictions for a single NWP run (the latest one available in the test set)
        sliced_nwps = sliced_nwps.filter(pl.col("init_time") == max_init)

    inference_params = InferenceParams(
        forecast_time=forecast_time,
        power_fcst_model_name="xgboost",
    )

    results_df = forecaster.predict(
        inference_params=inference_params,
        collapse_lead_times=False,  # We already filtered to a single init_time
        substation_metadata=substation_metadata,
        substation_power_flows=sliced_flows,
        nwps={NwpModel.ECMWF_ENS_0_25DEG: sliced_nwps},
    )
    return results_df


def main():
    # 3. Filter out the original assets and add the new ones
    assets = defs.assets
    if assets is None:
        return

    new_assets: list[Any] = []
    for a in assets:
        if isinstance(a, dg.AssetsDefinition):
            if not any(
                k.to_user_string()
                in [
                    "substation_metadata",
                    "combined_actuals",
                    "all_nwp_data",
                    "train_xgboost",
                    "evaluate_xgboost",
                ]
                for k in a.keys
            ):
                new_assets.append(a)
        elif isinstance(a, (dg.AssetSpec, dg.SourceAsset)):
            if a.key.to_user_string() not in [
                "substation_metadata",
                "combined_actuals",
                "all_nwp_data",
                "train_xgboost",
                "evaluate_xgboost",
            ]:
                new_assets.append(a)
        else:
            new_assets.append(a)
    new_assets.extend(
        [
            filtered_substation_metadata,
            filtered_combined_actuals,
            filtered_all_nwp_data,
            mock_train_xgboost,
            mock_evaluate_xgboost,
        ]
    )

    # 4. Run materialize
    print("Materializing assets...")
    resources = defs.resources or {}
    result = dg.materialize(
        new_assets,
        selection=dg.AssetSelection.assets(
            "substation_metadata",
            "combined_actuals",
            "all_nwp_data",
            "processed_nwp_data",
            "train_xgboost",
            "evaluate_xgboost",
        ),
        resources={**resources, "io_manager": dg.mem_io_manager},
    )

    # 5. Get the trained model and predictions
    model = result.asset_value("train_xgboost")
    predictions = result.asset_value("evaluate_xgboost")
    actuals = result.asset_value("combined_actuals")

    # 6. Print feature importance
    print("\n--- XGBoost Feature Importance ---")
    if hasattr(model, "model") and hasattr(model.model, "feature_importances_"):
        importances = model.model.feature_importances_
        features = model.feature_names
        importance_df = pd.DataFrame({"feature": features, "importance": importances}).sort_values(
            "importance", ascending=False
        )
        print(importance_df.head(10))
    else:
        print("Feature importance not available.")

    # 7. Print a sample of predictions
    print("\n--- Sample Predictions ---")
    print(predictions.head(5))

    # 8. Join predictions with actuals for plotting and metrics
    # Downsample actuals to 30m to match predictions
    actuals_30m = cast(pl.DataFrame, downsample_power_flows(actuals).collect())

    # Cast predictions to normal DataFrame to avoid Patito join issues
    predictions = pl.DataFrame(predictions)

    eval_df = predictions.join(
        actuals_30m.rename({"timestamp": "valid_time", "MW_or_MVA": "actual"}),
        on=["valid_time", "substation_number"],
        how="inner",
    )

    # Join peak capacity for nMAE
    target_map = model.target_map
    if isinstance(target_map, pl.LazyFrame):
        target_map = target_map.collect()
    target_map = pl.DataFrame(target_map)

    eval_df = eval_df.join(
        target_map.select(["substation_number", "peak_capacity"]),
        on="substation_number",
        how="left",
    )

    # 9. Print summary metrics
    print("\n--- Summary Metrics ---")
    metrics = eval_df.select(
        [
            (pl.col("MW_or_MVA") - pl.col("actual")).abs().mean().alias("MAE"),
            ((pl.col("MW_or_MVA") - pl.col("actual")) ** 2).mean().sqrt().alias("RMSE"),
            ((pl.col("MW_or_MVA") - pl.col("actual")).abs() / pl.col("peak_capacity"))
            .mean()
            .alias("nMAE"),
        ]
    )
    print(metrics)

    # 10. Generate plot
    print("\nGenerating plot...")

    # Filter plot to only show the last 2 weeks
    max_date_df = actuals.select(pl.col("timestamp").max()).collect()
    max_date = max_date_df.item()
    test_start = max_date - timedelta(days=14)

    plot_df = (
        eval_df.filter(pl.col("valid_time") >= test_start)
        .select(["valid_time", "substation_number", "ensemble_member", "MW_or_MVA", "actual"])
        .to_pandas()
    )

    # Melt the dataframe for Altair
    melted_df = plot_df.melt(
        id_vars=["valid_time", "substation_number", "ensemble_member"],
        value_vars=["MW_or_MVA", "actual"],
        var_name="type",
        value_name="power",
    )

    chart = (
        alt.Chart(melted_df)
        .mark_line()
        .encode(
            x="valid_time:T",
            y="power:Q",
            color="type:N",
            detail="ensemble_member:N",
            strokeWidth=alt.condition(alt.datum.type == "actual", alt.value(2.0), alt.value(0.5)),
            opacity=alt.condition(alt.datum.type == "actual", alt.value(1.0), alt.value(0.3)),
        )
        .properties(width=400, height=200)
        .facet("substation_number:N", columns=2)
        .properties(title="Actuals vs Predictions (51 members) for Last 2 Weeks")
    )

    chart.save("tests/xgboost_integration_plot.html")
    print("Plot saved to tests/xgboost_integration_plot.html")


if __name__ == "__main__":
    main()
