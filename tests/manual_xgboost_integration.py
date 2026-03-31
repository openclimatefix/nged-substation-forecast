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


# 2. Define filtered assets
@dg.asset(name="substation_metadata", compute_kind="python", group_name="nged")
def filtered_substation_metadata(settings: dg.ResourceParam[Settings]) -> pl.DataFrame:
    """Mock substation_metadata to filter to 5 substations."""
    metadata_path = settings.nged_data_path / "parquet" / "substation_metadata.parquet"
    df = pl.read_parquet(metadata_path)
    return df.filter(pl.col("substation_number").is_in(SUBSTATIONS))


@dg.asset(name="combined_actuals", compute_kind="python", group_name="nged")
def filtered_combined_actuals(settings: dg.ResourceParam[Settings]) -> pl.LazyFrame:
    """Mock combined_actuals to filter to 5 substations."""
    actuals_path = settings.nged_data_path / "delta" / "live_primary_flows"
    df = pl.read_delta(str(actuals_path)).lazy()
    return df.filter(pl.col("substation_number").is_in(SUBSTATIONS))


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
    config.data_split.train_start = pd.to_datetime("2026-01-01").date()
    config.data_split.train_end = pd.to_datetime("2026-02-28").date()
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
    config.data_split.test_start = pd.to_datetime("2026-03-01").date()
    config.data_split.test_end = pd.to_datetime("2026-03-31").date()
    forecaster = train_xgboost
    forecaster.config = config.model

    # Do inference manually to avoid the Patito bug in evaluate_and_save_model
    test_start = config.data_split.test_start
    test_end = config.data_split.test_end
    lookback = getattr(config.model, "required_lookback_days", 14)
    slice_start = test_start - timedelta(days=lookback)

    sliced_nwps = _slice_temporal_data(processed_nwp_data, slice_start, test_end, "valid_time")
    sliced_flows = _slice_temporal_data(combined_actuals, slice_start, test_end, "timestamp")

    forecast_time = datetime.now(timezone.utc)
    df = cast(pl.DataFrame, sliced_nwps.select(pl.col("init_time").max()).collect())
    if not df.is_empty():
        forecast_time = df.item() + timedelta(hours=3)

    inference_params = InferenceParams(
        forecast_time=forecast_time,
        power_fcst_model_name="xgboost",
    )

    results_df = forecaster.predict(
        inference_params=inference_params,
        collapse_lead_times=False,
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
    # Convert to pandas for Altair
    plot_df = eval_df.select(["valid_time", "substation_number", "MW_or_MVA", "actual"]).to_pandas()
    plot_df = plot_df.melt(
        id_vars=["valid_time", "substation_number"],
        value_vars=["MW_or_MVA", "actual"],
        var_name="type",
        value_name="power",
    )

    chart = (
        alt.Chart(plot_df)
        .mark_line()
        .encode(
            x="valid_time:T",
            y="power:Q",
            color="type:N",
            facet=alt.Facet("substation_number:N", columns=2),
        )
        .properties(width=400, height=200, title="Actuals vs Predictions for 5 Substations")
    )

    chart.save("tests/xgboost_integration_plot.html")
    print("Plot saved to tests/xgboost_integration_plot.html")


if __name__ == "__main__":
    main()
