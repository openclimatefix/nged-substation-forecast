import ast
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Final, Generic, Self, TypeVar

import patito as pt
import polars as pl
from contracts.geo_schemas import H3GridWeights
from contracts.power_schemas import PowerTimeSeries
from contracts.settings import Settings
from dagster import (
    AssetExecutionContext,
    DailyPartitionsDefinition,
    MetadataValue,
    RetryRequested,
    TableMetadataValue,
    TableRecord,
    asset,
)
from delta_store.nwp import write_nwp
from dynamical_data.ecmwf_ens.convert_to_polars import (
    convert_nwp_xarray_dataset_to_polars_dataframe,
)
from dynamical_data.ecmwf_ens.download import NwpRunNotYetAvailable, download_ecmwf_ens_run
from geo.great_britain.load import load_gb_boundary
from geo.h3 import compute_h3_grid_weights_for_boundary
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


ecmwf_ens_partitions = DailyPartitionsDefinition(
    start_date="2024-04-01", timezone="UTC", end_offset=1
)
"""One partition per day of ECMWF ENS 00Z runs. ``end_offset=1`` makes today's key exist before
its 00Z run has actually landed, matching Dynamical's publication lag; shared with
``ecmwf_ens_job``/``ecmwf_ens_schedule`` in ``defs/schedules.py``."""

_ECMWF_ENS_MAX_RETRIES: Final[int] = 8
"""Retries × ``_ECMWF_ENS_RETRY_DELAY_SECONDS`` ≈ 4h of coverage past the 08:30 UTC schedule
(``ecmwf_ens_schedule``), comfortably past Dynamical's typical publication time. Only applies to
``NwpRunNotYetAvailable``; a genuine bug fails immediately instead of retrying for hours."""

_ECMWF_ENS_RETRY_DELAY_SECONDS: Final[int] = 1800
"""How long to wait between retries of a not-yet-published ECMWF run."""


@asset(
    partitions_def=ecmwf_ens_partitions,
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
    Polars DataFrame, and appends it to the Delta table through
    ``delta_store.nwp.write_nwp`` (Float32, significand-rounded).
    """
    settings = Settings()
    partition_date_str = context.partition_key
    nwp_init_time = datetime.strptime(partition_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Load dependencies
    h3_grid = pt.DataFrame(pl.read_parquet(settings.h3_grid_weights_path)).set_model(H3GridWeights)

    # Download and convert
    try:
        ds = download_ecmwf_ens_run(nwp_init_time=nwp_init_time, h3_grid=h3_grid)
    except NwpRunNotYetAvailable as exc:
        raise RetryRequested(
            max_retries=_ECMWF_ENS_MAX_RETRIES, seconds_to_wait=_ECMWF_ENS_RETRY_DELAY_SECONDS
        ) from exc
    nwp = convert_nwp_xarray_dataset_to_polars_dataframe(ds=ds, h3_grid=h3_grid)

    context.log.info(f"Columns: {nwp.columns}")

    settings.nwp_data_path.parent.mkdir(parents=True, exist_ok=True)
    write_nwp(nwp, settings.nwp_data_path)

    context.add_output_metadata(
        {
            "n_rows": len(nwp),
            "path": str(settings.nwp_data_path),
            "init_time": str(nwp_init_time),
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
