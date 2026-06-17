import ast
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generic, Self, TypeVar, cast

import hydra.utils
import mlflow
import patito as pt
import polars as pl
from contracts.geo_schemas import H3GridWeights
from contracts.hydra_schemas import CvConfig, CvFoldConfig
from contracts.power_schemas import PowerForecast, PowerTimeSeries, TimeSeriesMetadata
from contracts.settings import PROJECT_ROOT, Settings
from contracts.weather_schemas import NwpModelId, NwpOnDisk, NwpScalingParams
from dagster import (
    AssetExecutionContext,
    DailyPartitionsDefinition,
    MetadataValue,
    TableMetadataValue,
    TableRecord,
    asset,
)
from deltalake import WriterProperties, write_deltalake
from dynamical_data.ecmwf_ens.convert_to_polars import (
    convert_nwp_xarray_dataset_to_polars_dataframe,
)
from dynamical_data.ecmwf_ens.download import download_ecmwf_ens_run
from geo.great_britain.load import load_gb_boundary
from geo.h3 import compute_h3_grid_weights_for_boundary
from ml_core.base_forecaster import BaseForecaster
from ml_core.cross_validate import cross_validate
from ml_core.metrics import compute_metrics
from nged_data.read_nged_json import _H3_RESOLUTION
from nged_data.storage import (
    NoNewData,
    UpsertMetadataStats,
    _ProcessedFileListing,
    download_and_parse_files,
    list_timeseries_json_files,
    remove_small_files_from_listing,
    select_new_rows,
    upsert_metadata,
)
from omegaconf import OmegaConf
from pydantic import BaseModel, computed_field, field_validator


@asset
def power_time_series_and_metadata(context: AssetExecutionContext) -> None:
    """
    Ingests raw telemetry and metadata from NGED S3 into our local storage.

    This asset acts as the entry point for NGED data into our system. It fetches
    the latest available data from the external S3 bucket and appends it to our
    local Delta table for time series data, while upserting the latest metadata.
    This raw data will later be consumed by downstream cleaning assets to prepare
    it for forecasting models.

    WHY UNPARTITIONED? Because NGED's JSON files are published roughly every 5 hours, and so
    the start time changes every day. And because we don't want people to have to spin up
    thousands of Dagster runs (one per partition) when first backfilling. It's much more efficient
    to just check what's available on NGED's S3 bucket and append to our local Delta table.
    """
    settings = Settings()
    delta_path = settings.nged_data_path / "power_time_series.delta"
    metadata_path = settings.nged_data_path / "metadata.parquet"

    # Fetch new data from S3, using the existing delta table to determine what's new.
    # We are deliberately keeping the code simple for now, but may move the S3 store
    # to a Dagster ConfigurableResource in the future.
    store = settings.get_nged_s3_store()
    list_of_all_json_files = list_timeseries_json_files(store)
    list_of_large_json_files = remove_small_files_from_listing(list_of_all_json_files)
    list_of_new_json_files = select_new_rows(list_of_large_json_files, delta_path)

    # Log statistics to be shown in Dagster's UI.
    context.add_output_metadata(
        _FileListingSummary.make_table(
            "nged_s3_paths",
            {
                "All JSON files on S3": list_of_all_json_files,
                "Files larger than 1kB": list_of_large_json_files,
                "Files with new data": list_of_new_json_files,
            },
        )
    )

    try:
        new_metadata, new_power_ts = download_and_parse_files(store, list_of_new_json_files)
    except NoNewData:
        context.add_output_metadata(
            UpsertMetadataStats(metadata_n_new_TimeSeriesIDs=0, metadata_n_updated_TimeSeriesIDs=0)
        )
        return

    # Save TimeSeriesMetadata:
    upsert_metadata_stats = upsert_metadata(new_metadata, metadata_path)
    context.add_output_metadata(upsert_metadata_stats)

    # Save PowerTimeSeries:
    new_power_ts_deduped = select_new_rows(new_power_ts, delta_path)
    if not new_power_ts_deduped.is_empty():
        # FIXME: mkdir won't work when delta_path is on S3!
        delta_path.parent.mkdir(parents=True, exist_ok=True)
        new_power_ts_deduped.write_delta(
            delta_path, mode="append", delta_write_options={"partition_by": "time_series_id"}
        )

    # Log statistics to be shown in Dagster's UI.
    context.add_output_metadata(
        _PowerTimeSeriesSummary.make_table(
            "PowerTimeSeries",
            {
                "Downloaded timeseries": new_power_ts,
                "De-duped rows appended to disk": new_power_ts_deduped,
            },
        )
    )


@asset
def h3_grid_weights(context: AssetExecutionContext) -> None:
    """
    Computes H3 grid weights for the Great Britain boundary.

    This asset calculates the fractional overlap of H3 cells with the GB boundary
    at various resolutions, which is used for spatial aggregation of weather data.
    """
    settings = Settings()
    boundary = load_gb_boundary()
    weights = compute_h3_grid_weights_for_boundary(
        boundary, nwp_grid_size_degrees=0.25, h3_res=_H3_RESOLUTION
    )

    # Save to parquet
    # FIXME: mkdir won't work when we're saving to S3!
    settings.h3_grid_weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights.write_parquet(settings.h3_grid_weights_path)

    # Add metadata to Dagster context
    context.add_output_metadata(
        {
            "n_rows": len(weights),
            "path": str(settings.h3_grid_weights_path),
        }
    )


@asset(
    partitions_def=DailyPartitionsDefinition(start_date="2024-04-01", timezone="UTC", end_offset=1),
    deps=["h3_grid_weights"],
    # The `pool="ECMWF"` works in conjunction with the Dagster instance configuration
    # (e.g., in `dagster.yaml`) to limit the number of times this asset can be run
    # concurrently. This is crucial because downloading ECMWF data is memory-intensive.
    # See: https://docs.dagster.io/guides/operate/managing-concurrency/concurrency-pools
    pool="ECMWF",
)
def ecmwf_ens(context: AssetExecutionContext) -> None:
    """
    Downloads and processes ECMWF ensemble NWP data for a specific day.

    This asset fetches the 00Z NWP run for the partition date, converts it to a
    Polars DataFrame, scales it to integer representation, and appends it to the
    Delta table.
    """
    settings = Settings()
    partition_date_str = context.partition_key
    nwp_init_time = datetime.strptime(partition_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Load dependencies
    h3_grid = pt.DataFrame(pl.read_parquet(settings.h3_grid_weights_path)).set_model(H3GridWeights)
    scaling_params = NwpScalingParams.load()

    # Download and convert
    ds = download_ecmwf_ens_run(nwp_init_time=nwp_init_time, h3_grid=h3_grid)
    nwp_in_memory = convert_nwp_xarray_dataset_to_polars_dataframe(ds=ds, h3_grid=h3_grid)

    # Validate and scale
    nwp_on_disk = NwpOnDisk.from_nwp_in_memory(nwp_in_memory, scaling_params)

    context.log.info(f"Columns: {nwp_on_disk.columns}")

    settings.nwp_data_path.parent.mkdir(parents=True, exist_ok=True)

    # Delta Lake doesn't support Enums, so we must cast to String:
    df_to_write = nwp_on_disk.as_polars().cast({"nwp_model_id": pl.String})

    write_deltalake(
        table_or_uri=settings.nwp_data_path,
        data=df_to_write,
        mode="append",
        partition_by=["nwp_model_id", "init_time"],
        writer_properties=WriterProperties(compression="ZSTD", compression_level=14),
    )

    context.add_output_metadata(
        {
            "n_rows": len(nwp_on_disk),
            "path": str(settings.nwp_data_path),
            "init_time": str(nwp_init_time),
        }
    )


def _load_cv_folds(cv_config_path: Path) -> tuple[list[CvFoldConfig], int]:
    """Load CV fold definitions from a Hydra YAML and return (folds, min_training_months)."""
    raw = OmegaConf.to_container(OmegaConf.load(cv_config_path), resolve=True)
    cv_cfg = CvConfig.model_validate(raw)
    return cv_cfg.folds, cv_cfg.min_training_months


def _load_model_config(model_config_path: Path) -> tuple[type[BaseForecaster], Any]:
    """Load and validate a model config YAML, returning (forecaster_class, forecaster_config).

    The YAML must have a top-level ``_target_`` key identifying the ``BaseForecaster``
    subclass and a ``model_params`` block whose ``_target_`` identifies the matching
    ``BaseForecasterConfig`` subclass.  ``hydra.utils.instantiate`` constructs and
    validates the config object via Pydantic — this is the enforcement point for
    ``conf/model/*.yaml`` files.
    """
    raw = OmegaConf.load(model_config_path)
    forecaster_class = cast(type[BaseForecaster], hydra.utils.get_class(str(raw._target_)))
    forecaster_config = hydra.utils.instantiate(raw.model_params)
    return forecaster_class, forecaster_config


@asset(deps=["power_time_series_and_metadata", "ecmwf_ens"])
def cv_power_forecasts(context: AssetExecutionContext) -> None:
    """Run expanding-window cross-validation and persist half-hourly predictions.

    Loads the model config from ``conf/model/xgboost.yaml`` and the CV fold
    definitions from ``conf/cv/default.yaml``, then calls ``cross_validate()``
    to produce a ``PowerForecast`` DataFrame with ``fold_id`` set to the
    validation year (e.g. ``"2022"``).  Results are appended to the shared
    ``power_forecasts`` Delta table alongside any live production forecasts.

    Only the control NWP ensemble member (member 0) is used.  Full ensemble
    support will be added once the XGBoost iterator-API is implemented.

    The downstream ``cv_metrics`` asset can be re-run without re-training by
    recomputing metrics directly from the power_forecasts table.
    """
    settings = Settings()

    # --- Load configs ---
    forecaster_class, forecaster_config = _load_model_config(
        PROJECT_ROOT / "conf" / "model" / "xgboost.yaml"
    )
    folds, min_training_months = _load_cv_folds(PROJECT_ROOT / "conf" / "cv" / "default.yaml")

    # --- Load data (lazy) ---
    power_lf = pt.LazyFrame.from_existing(
        pl.scan_delta(str(settings.nged_data_path / "power_time_series.delta"))
    ).set_model(PowerTimeSeries)

    metadata_df = pt.DataFrame(
        pl.read_parquet(settings.nged_data_path / "metadata.parquet")
    ).set_model(TimeSeriesMetadata)

    # Filter to control member and cast nwp_model_id back from String → Enum
    # (Delta Lake doesn't store Enum; the ecmwf_ens asset casts to String before writing).
    nwp_lf = pt.LazyFrame.from_existing(
        pl.scan_delta(str(settings.nwp_data_path))
        .filter(pl.col("ensemble_member") == 0)
        .cast({"nwp_model_id": pl.Enum([m.name for m in NwpModelId])})
    ).set_model(NwpOnDisk)

    # --- Run cross-validation ---
    cv_results = cross_validate(
        forecaster_class=forecaster_class,
        forecaster_config=forecaster_config,
        power_lf=power_lf,
        nwp_lf=nwp_lf,
        metadata_df=metadata_df,
        folds=folds,
        min_training_months=min_training_months,
    )

    # --- Persist to the shared power_forecasts Delta table ---
    settings.power_forecasts_data_path.parent.mkdir(parents=True, exist_ok=True)
    cv_results.write_delta(
        str(settings.power_forecasts_data_path),
        mode="append",
        delta_write_options={
            "partition_by": ["power_fcst_model_name", "fold_id"],
            "writer_properties": WriterProperties(compression="ZSTD", compression_level=14),
        },
    )

    context.add_output_metadata(
        {
            "n_rows": cv_results.height,
            "n_folds": cv_results["fold_id"].n_unique(),
            "n_time_series": cv_results["time_series_id"].n_unique(),
            "model_name": forecaster_config.power_fcst_model_name,
            "path": str(settings.power_forecasts_data_path),
        }
    )


@asset(deps=["cv_power_forecasts"])
def cv_metrics(context: AssetExecutionContext) -> None:
    """Compute and persist CV metrics, and log aggregate metrics to MLflow.

    Loads CV rows (``fold_id != 'live'``) from the shared ``power_forecasts``
    Delta table and the raw ``power_time_series`` Delta table, calls
    ``compute_metrics()``, then:

    1. Writes the tall ``Metrics`` table to ``cv_metrics`` Delta.
    2. Logs per-metric mean-across-folds values to an MLflow run, tagged with
       the model family, weather source, and other experiment dimensions so the
       leaderboard can group and filter entries.
    """
    settings = Settings()
    _, forecaster_config = _load_model_config(PROJECT_ROOT / "conf" / "model" / "xgboost.yaml")

    # --- Load data (CV rows only — exclude live production forecasts) ---
    cv_forecasts_df = PowerForecast.validate(
        pl.scan_delta(str(settings.power_forecasts_data_path))
        .filter(pl.col("fold_id") != "live")
        .cast({"power_fcst_model_name": pl.Categorical, "fold_id": pl.Categorical})
        .collect(),
        allow_superfluous_columns=True,
    )

    actuals_lf = pt.LazyFrame.from_existing(
        pl.scan_delta(str(settings.nged_data_path / "power_time_series.delta"))
    ).set_model(PowerTimeSeries)

    # --- Compute metrics ---
    metrics_df = compute_metrics(cv_forecasts_df, actuals_lf)

    # --- Persist to Delta ---
    settings.cv_metrics_data_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.write_delta(
        str(settings.cv_metrics_data_path),
        mode="append",
        delta_write_options={
            "partition_by": ["power_fcst_model_name", "fold_id"],
            "writer_properties": WriterProperties(compression="ZSTD", compression_level=14),
        },
    )

    # --- Log aggregate metrics to MLflow ---
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment_name = forecaster_config.power_fcst_model_name
    mlflow.set_experiment(experiment_name)

    # Compute mean-across-folds for each (metric_name, metric_param, horizon_slice).
    agg_metrics = metrics_df.group_by(["metric_name", "metric_param", "horizon_slice"]).agg(
        pl.col("metric_value").mean().alias("mean_across_folds")
    )

    with mlflow.start_run():
        # Tag with experiment dimensions for leaderboard grouping.
        mlflow.set_tags(
            {
                "model_family": forecaster_config.model_family,
                "weather_source": forecaster_config.weather_source,
                "training_strategy": forecaster_config.training_strategy,
                "power_fcst_model_name": experiment_name,
            }
        )
        for row in agg_metrics.iter_rows(named=True):
            key = f"{row['metric_name']}/{row['horizon_slice']}/{row['metric_param']}"
            mlflow.log_metric(key, float(row["mean_across_folds"]))

    context.add_output_metadata(
        {
            "n_metric_rows": metrics_df.height,
            "n_time_series": metrics_df["time_series_id"].n_unique(),
            "mlflow_experiment": experiment_name,
            "path": str(settings.cv_metrics_data_path),
        }
    )


##############################################################################
# All the code below this line is just for outputting summary stats to Dagster
# TODO: Move the code below this line to a separate file.


T = TypeVar("T", bound=pt.Model)


class _BaseSummary(ABC, BaseModel, Generic[T]):
    """Create a Dagster table of summary statistics.

    The Generic[T] makes this superclass generic over pt.Models."""

    stage: str
    start_time: str = "N/A"
    end_time: str = "N/A"
    time_series_ids: str = "N/A"  # str representation of a list of ints

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def datetime_to_string(cls, v: Any) -> Any:
        return v.strftime("%Y-%m-%d %H:%M") if isinstance(v, datetime) else v

    @field_validator("time_series_ids", mode="before")
    @classmethod
    def unique_time_series_ids(cls, v: Any) -> Any:
        return str(v.unique().sort().to_list()) if isinstance(v, pl.Series) else v

    @computed_field
    @property
    def n_time_series_ids(self) -> int:
        return 0 if self.time_series_ids == "N/A" else len(ast.literal_eval(self.time_series_ids))

    @classmethod
    def make_table(
        cls, key: str, dataframes: dict[str, pt.DataFrame[T]]
    ) -> dict[str, TableMetadataValue]:
        table: list[TableRecord] = []
        for stage_name, df in dataframes.items():
            summary = cls.from_data_frame(stage_name, df)
            table_record = TableRecord(summary.model_dump())
            table.append(table_record)
        return {key: MetadataValue.table(table)}

    @classmethod
    @abstractmethod
    def from_data_frame(cls, stage_name: str, df: pt.DataFrame[T]) -> Self:
        pass


class _FileListingSummary(_BaseSummary[_ProcessedFileListing]):
    n_files: int
    min_file_size_bytes: int = 0
    max_file_size_bytes: int = 0

    @classmethod
    def from_data_frame(cls, stage_name: str, df: pt.DataFrame[_ProcessedFileListing]) -> Self:
        # The `ty: ignore` comments are because `ty` only looks at the types specified in the BaseModel.
        # `ty` doesn't know that we're casting the types in the `field_validator` methods.
        if len(df) > 0:
            return cls(
                stage=stage_name,
                n_files=len(df),
                start_time=df["start_time"].min(),  # ty: ignore[invalid-argument-type]
                end_time=df["end_time"].max(),  # ty: ignore[invalid-argument-type]
                # TODO: We can't list *all* time_series_ids when we're handling 1,000s of IDs!
                time_series_ids=df["time_series_id"],  # ty: ignore[invalid-argument-type]
                min_file_size_bytes=df["filesize_bytes"].min(),  # ty: ignore[invalid-argument-type]
                max_file_size_bytes=df["filesize_bytes"].max(),  # ty: ignore[invalid-argument-type]
            )
        else:
            return cls(stage=stage_name, n_files=0)


class _PowerTimeSeriesSummary(_BaseSummary[PowerTimeSeries]):
    n_rows: int

    @classmethod
    def from_data_frame(cls, stage_name: str, df: pt.DataFrame[PowerTimeSeries]) -> Self:
        # The `ty: ignore` comments are because `ty` only looks at the types specified in the BaseModel.
        # `ty` doesn't know that we're casting the types in the `field_validator` methods.
        if len(df) > 0:
            return cls(
                stage=stage_name,
                n_rows=len(df),
                start_time=df["time"].min(),  # ty: ignore[invalid-argument-type]
                end_time=df["time"].max(),  # ty: ignore[invalid-argument-type]
                # TODO: We can't list *all* time_series_ids when we're handling 1,000s of IDs!
                time_series_ids=df["time_series_id"],  # ty: ignore[invalid-argument-type]
            )
        else:
            return cls(stage=stage_name, n_rows=0)
