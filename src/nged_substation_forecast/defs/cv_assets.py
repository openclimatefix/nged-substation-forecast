"""Cross-validation Dagster assets.

These assets implement the experiment-independent, fold-partitioned CV layer. Each asset is a
thin orchestration shell delegating its logic to the pure helpers in ``ml_core._cv_helpers`` so
the logic stays fast to unit-test and the assets stay readable.
"""

from datetime import datetime
from typing import Final

import mlflow
import patito as pt
import polars as pl
from contracts.hydra_schemas import load_cv_config
from contracts.ml_schemas import EligibleTimeSeries
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.settings import Settings
from contracts.weather_schemas import NwpInMemory, NwpOnDisk
from dagster import (
    AssetExecutionContext,
    DynamicPartitionsDefinition,
    StaticPartitionsDefinition,
    asset,
)
from deltalake import write_deltalake
from ml_core._cv_helpers import (
    date_to_utc_datetime,
    eligible_time_series_ids,
    parse_cv_partition_key,
)
from ml_core._mlflow_runs import (
    get_or_create_experiment,
    get_or_create_fold_run,
    get_or_create_parent_run,
    load_experiment_forecaster,
)

# The CV folds are the shared leaderboard evaluation protocol, read from conf/cv/default.yaml
# (never hard-coded) so every experiment and asset agrees on the same folds. Loaded at import so
# the partition keys are available when Dagster builds the asset graph. PROJECT_ROOT is used here
# (rather than Settings, which needs the .env secrets) so the partition set can be built without
# any credentials.
_cv_config = load_cv_config(Settings.model_fields["cv_config_path"].default)

cv_fold_partitions = StaticPartitionsDefinition(_cv_config.fold_ids)
"""One partition per canonical CV fold (by fold year, e.g. "2022"). Experiment-independent."""

CV_EXPERIMENT_FOLDS_NAME: Final[str] = "cv_experiment_folds"
"""Name of the dynamic partition set keyed by ``"{experiment_name}__{fold_id}"``.

A named constant so callers (``register_experiment_job``) pass a definite ``str`` to
``instance.add_dynamic_partitions`` / ``get_dynamic_partitions``.
"""

cv_experiment_folds = DynamicPartitionsDefinition(name=CV_EXPERIMENT_FOLDS_NAME)
"""One partition per (experiment, fold); key format ``"{experiment_name}__{fold_id}"``.

Keys are added by ``register_experiment_job`` and consumed by the per-fold CV assets
(``trained_cv_model`` / ``cv_power_forecasts``). Dynamic (not static) because experiments are
registered at runtime and there can be thousands of them.
"""


@asset(
    partitions_def=cv_fold_partitions,
    deps=["power_time_series_and_metadata"],
)
def eligible_time_series(context: AssetExecutionContext) -> None:
    """Compute and persist the canonical eligible ``time_series_id``s for one CV fold.

    A time series is eligible for a fold when its observed-power coverage has at least
    ``min_training_months`` of history before the fold's ``val_start`` *and* reaches the fold's
    ``val_end``. Eligibility is derived from data coverage alone (not from any model/config), so
    every experiment evaluates the fold on the identical population — this is what keeps the
    leaderboard apples-to-apples.

    The result is written to the ``eligible_time_series`` Delta table as one partition per
    ``fold_id`` via an idempotent partition overwrite, so re-materialising a fold replaces its
    rows rather than duplicating them.
    """
    settings = Settings()
    fold_id = context.partition_key
    fold = _cv_config.get_fold(fold_id)

    power_path = settings.nged_data_path / "power_time_series.delta"
    coverage = (
        pl.scan_delta(str(power_path))
        .group_by("time_series_id")
        .agg(
            first_time=pl.col("time").min(),
            last_time=pl.col("time").max(),
        )
        .collect()
    )

    eligible_ids = eligible_time_series_ids(
        coverage, fold, min_training_months=_cv_config.min_training_months
    )

    eligible_df = EligibleTimeSeries.validate(
        pl.DataFrame(
            {
                "fold_id": pl.Series([fold_id] * len(eligible_ids), dtype=pl.String),
                "time_series_id": pl.Series(eligible_ids, dtype=pl.Int32),
            }
        )
    )

    settings.eligible_time_series_data_path.parent.mkdir(parents=True, exist_ok=True)
    write_deltalake(
        table_or_uri=settings.eligible_time_series_data_path,
        data=eligible_df.to_arrow(),
        mode="overwrite",
        predicate=f"fold_id = '{fold_id}'",
        partition_by=["fold_id"],
    )

    context.add_output_metadata(
        {
            "fold_id": fold_id,
            "n_eligible_time_series": len(eligible_ids),
            "n_time_series_in_coverage": len(coverage),
            "eligible_time_series_ids": str(eligible_ids),
            "val_start": str(fold.val_start),
            "val_end": str(fold.val_end),
            "min_training_months": _cv_config.min_training_months,
        }
    )


def _load_engineering_inputs(
    settings: Settings,
    time_series_ids: list[int],
    window_start: datetime,
    window_end: datetime,
) -> tuple[
    pt.LazyFrame[PowerTimeSeries], pt.DataFrame[TimeSeriesMetadata], pt.LazyFrame[NwpInMemory]
]:
    """Load observed power, metadata, and in-memory NWP for a window and time-series population.

    Shared by ``trained_cv_model`` (training window + eligible population) and
    ``cv_power_forecasts`` (validation window + trained population). Power and NWP are filtered to
    the inclusive ``[window_start, window_end]`` window; all three inputs are filtered to
    ``time_series_ids``. NWP is **not** filtered to the control member, so every ensemble member is
    carried through to feature engineering.
    """
    power_lf = pl.scan_delta(str(settings.nged_data_path / "power_time_series.delta")).filter(
        pl.col("time_series_id").is_in(time_series_ids),
        pl.col("time") >= window_start,
        pl.col("time") <= window_end,
    )
    power_ts = pt.LazyFrame.from_existing(power_lf).set_model(PowerTimeSeries)

    nwp_in_window = NwpOnDisk.to_nwp_in_memory(NwpOnDisk.scan_delta(settings.nwp_data_path)).filter(
        pl.col("valid_time") >= window_start, pl.col("valid_time") <= window_end
    )
    nwp_lf = pt.LazyFrame.from_existing(nwp_in_window).set_model(NwpInMemory)

    metadata_df = pt.DataFrame(
        pl.read_parquet(settings.nged_data_path / "metadata.parquet").filter(
            pl.col("time_series_id").is_in(time_series_ids)
        )
    ).set_model(TimeSeriesMetadata)

    return power_ts, metadata_df, nwp_lf


@asset(
    partitions_def=cv_experiment_folds,
    deps=["power_time_series_and_metadata", "ecmwf_ens", "eligible_time_series"],
)
def trained_cv_model(context: AssetExecutionContext) -> None:
    """Train one forecaster for a single ``(experiment, fold)`` partition and save it to MLflow.

    Reads the experiment's resolved config from MLflow (the immutable record registered by
    ``register_experiment_job``), the fold's canonical eligible ``time_series_id`` population from
    the ``eligible_time_series`` asset, and the observed power + gridded NWP over the fold's
    **inclusive** training window. Features are engineered through the forecaster's own
    ``FeatureEngineer`` (so the spatial NWP mapping and feature pipeline are a model concern), the
    model is trained, and its artifacts are uploaded to the fold's MLflow run alongside the
    training params.

    The fold run is resolved **by tag**, never by a handle passed between assets, so this is safe
    across processes and idempotent under Dagster retries.
    """
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    experiment_name, fold_id = parse_cv_partition_key(context.partition_key)
    forecaster_cls, config = load_experiment_forecaster(experiment_name)

    fold = _cv_config.get_fold(fold_id)
    train_start = date_to_utc_datetime(fold.train_start)
    train_end = date_to_utc_datetime(fold.train_end, end_of_day=True)

    eligible_ids = (
        pl.scan_delta(str(settings.eligible_time_series_data_path))
        .filter(pl.col("fold_id") == fold_id)
        .select("time_series_id")
        .collect()["time_series_id"]
        .to_list()
    )

    power_ts, metadata_df, nwp_lf = _load_engineering_inputs(
        settings, eligible_ids, train_start, train_end
    )

    forecaster = forecaster_cls(model_params=config)
    features = forecaster.feature_engineer.engineer(
        selected_features=config.selected_features,
        power_time_series=power_ts,
        time_series_metadata=metadata_df,
        nwp=nwp_lf,
    )
    forecaster.train(features)

    experiment_id = get_or_create_experiment(experiment_name)
    parent_run_id = get_or_create_parent_run(experiment_id)
    fold_run_id = get_or_create_fold_run(experiment_id, parent_run_id, fold_id)

    forecaster.save_to_mlflow(fold_run_id)
    training_params = {
        "fold_id": fold_id,
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "n_eligible_time_series": len(eligible_ids),
    }
    with mlflow.start_run(run_id=fold_run_id):
        mlflow.log_params(training_params)

    context.add_output_metadata(
        {
            "experiment_name": experiment_name,
            "fold_id": fold_id,
            "n_eligible_time_series": len(eligible_ids),
            "train_start": str(train_start),
            "train_end": str(train_end),
            "fold_run_id": fold_run_id,
        }
    )


@asset(
    partitions_def=cv_experiment_folds,
    deps=["trained_cv_model"],
)
def cv_power_forecasts(context: AssetExecutionContext) -> None:
    """Predict the validation window for one ``(experiment, fold)`` partition and persist forecasts.

    Loads the model ``trained_cv_model`` saved for this fold back from MLflow (via the local-disk
    cache), then forecasts the fold's **inclusive** validation window across **all** NWP ensemble
    members — the probabilistic leaderboard metrics are meaningless on a single member. The scored
    population is the model's own ``trained_time_series_ids`` (the train==predict invariant), so a
    fold is always scored on exactly the population it was trained on even if power coverage has
    drifted since training.

    Forecasts are written to the ``power_forecasts`` Delta table via an **idempotent partition
    overwrite** keyed by ``(experiment_name, fold_id)``: re-materialising a fold replaces its rows
    rather than appending, so a retry can never duplicate forecasts (which would double-count in
    metrics). The fold's MLflow run is resolved **by tag** — never by a handle from
    ``trained_cv_model`` — so this is safe across processes.
    """
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    experiment_name, fold_id = parse_cv_partition_key(context.partition_key)
    forecaster_cls, config = load_experiment_forecaster(experiment_name)

    fold = _cv_config.get_fold(fold_id)
    val_start = date_to_utc_datetime(fold.val_start)
    val_end = date_to_utc_datetime(fold.val_end, end_of_day=True)

    experiment_id = get_or_create_experiment(experiment_name)
    parent_run_id = get_or_create_parent_run(experiment_id)
    fold_run_id = get_or_create_fold_run(experiment_id, parent_run_id, fold_id)

    forecaster = forecaster_cls.load_from_mlflow(fold_run_id, settings.model_cache_base_path)
    trained_ids = forecaster.trained_time_series_ids

    power_ts, metadata_df, nwp_lf = _load_engineering_inputs(
        settings, trained_ids, val_start, val_end
    )
    features = forecaster.feature_engineer.engineer(
        selected_features=config.selected_features,
        power_time_series=power_ts,
        time_series_metadata=metadata_df,
        nwp=nwp_lf,
    )
    forecasts = forecaster.predict(features, fold_id=fold_id)

    # Cast the partition columns to plain strings so Delta stores them as Hive-style string
    # directories (delta-rs cannot partition on a dictionary-encoded Categorical column).
    delta_data = forecasts.with_columns(
        experiment_name=pl.col("experiment_name").cast(pl.String),
        fold_id=pl.col("fold_id").cast(pl.String),
    ).to_arrow()
    settings.power_forecasts_data_path.parent.mkdir(parents=True, exist_ok=True)
    write_deltalake(
        table_or_uri=settings.power_forecasts_data_path,
        data=delta_data,
        mode="overwrite",
        predicate=f"experiment_name = '{experiment_name}' AND fold_id = '{fold_id}'",
        partition_by=["experiment_name", "fold_id"],
    )

    n_rows = forecasts.height
    n_time_series = forecasts["time_series_id"].n_unique()
    n_ensemble_members = forecasts["ensemble_member"].n_unique()
    with mlflow.start_run(run_id=fold_run_id):
        mlflow.log_params({"val_start": val_start.isoformat(), "val_end": val_end.isoformat()})
        mlflow.log_metrics(
            {
                "n_forecast_rows": float(n_rows),
                "n_forecast_time_series": float(n_time_series),
                "n_ensemble_members": float(n_ensemble_members),
            }
        )

    context.add_output_metadata(
        {
            "experiment_name": experiment_name,
            "fold_id": fold_id,
            "n_rows": n_rows,
            "n_time_series": n_time_series,
            "n_ensemble_members": n_ensemble_members,
            "val_start": str(val_start),
            "val_end": str(val_end),
            "fold_run_id": fold_run_id,
        }
    )
