"""The Dagster asset that cleans NGED's power telemetry into ``cleaned_power_time_series``.

The cleaned table is the one read path for observed power: every consumer except the ingest and its
freshness check reads it through ``nged_data.storage.scan_cleaned_power``.
"""

import hashlib
import os
from typing import Final

import patito as pt
import polars as pl
from contracts.power_schemas import CleanedPowerTimeSeries, PowerTimeSeries, TimeSeriesMetadata
from contracts.settings import Settings
from contracts.typing_utils import typeddict_to_dict
from contracts.uri import delta_table_exists, if_local_path_then_make_parent_dir
from dagster import AssetExecutionContext, Config, asset
from delta_store.cleaned_power_time_series import (
    CleaningProvenance,
    VacuumError,
    read_cleaning_provenance,
    write_cleaned_power_time_series,
)
from deltalake import DeltaTable
from ml_core.repro import UNKNOWN, get_git_info
from nged_data.cleaning import CLEANING_CODE_HASH, flag_nged_power

from nged_substation_forecast._sentry import report_asset_degradation
from nged_substation_forecast.defs._tags import PRODUCTION_LAYER_TAGS

GIT_SHA_ENV_VAR: Final[str] = "GIT_SHA"
"""Environment variable holding the git SHA of the code, set when the container image is built.
`current_git_sha` falls back to it, because a container has no git repository to ask."""


def current_git_sha() -> str:
    """Return the git SHA of the running code, for `CleaningProvenance.git_sha`.

    Uses `ml_core.repro.get_git_info`, and when that returns `UNKNOWN` (as it does in a container)
    falls back to a non-empty `GIT_SHA` environment variable, else `UNKNOWN`. The fallback lives
    here rather than in `get_git_info` so the MLflow provenance tags of the other stages do not
    change.

    Returns:
        A git SHA, or `UNKNOWN`.
    """
    sha = get_git_info()["git_sha"]
    if sha != UNKNOWN:
        return sha
    return os.environ.get(GIT_SHA_ENV_VAR) or UNKNOWN


def metadata_fingerprint(metadata: pl.DataFrame) -> str:
    """Return a hex digest that changes when any value in the `TimeSeriesMetadata` table changes.

    The digest is the SHA-256 of Polars' per-row hashes (`hash_rows`, default seeds) of the table
    sorted by `time_series_id`, so it does not depend on row order. The digest is stable between
    runs on one Polars version. A Polars upgrade can change it, which causes one extra rebuild.

    Args:
        metadata: The validated `TimeSeriesMetadata` table.

    Returns:
        A SHA-256 hex digest.
    """
    row_hashes = metadata.sort("time_series_id").hash_rows().to_list()
    return hashlib.sha256(repr(row_hashes).encode()).hexdigest()


class CleanNgedPowerDataConfig(Config):
    """Run config for the ``clean_nged_power_data`` asset."""

    force: bool = False
    """Rebuild the cleaned table even when its raw table version, cleaning code and
    `TimeSeriesMetadata` table are unchanged."""


def _drop_reason_metadata(cleaned: pl.DataFrame) -> dict[str, int | float | str]:
    """Per drop reason: rows and series flagged, flagged power range, and flagged time range."""
    per_reason = (
        cleaned.filter(pl.col("drop_reason").is_not_null())
        .group_by("drop_reason")
        .agg(
            n_rows=pl.len(),
            n_time_series=pl.col("time_series_id").n_unique(),
            min_power=pl.col("power").min(),
            max_power=pl.col("power").max(),
            first_time=pl.col("time").min(),
            last_time=pl.col("time").max(),
        )
    )
    metadata: dict[str, int | float | str] = {}
    for row in per_reason.iter_rows(named=True):
        prefix = f"drop_reason/{row['drop_reason']}"
        metadata[f"{prefix}/n_rows"] = int(row["n_rows"])
        metadata[f"{prefix}/n_time_series"] = int(row["n_time_series"])
        metadata[f"{prefix}/min_power"] = float(row["min_power"])
        metadata[f"{prefix}/max_power"] = float(row["max_power"])
        # ISO-8601 strings, because Dagster rejects `datetime` metadata values.
        metadata[f"{prefix}/first_time"] = row["first_time"].isoformat()
        metadata[f"{prefix}/last_time"] = row["last_time"].isoformat()
    return metadata


@asset(tags=PRODUCTION_LAYER_TAGS, deps=["power_time_series_and_metadata"])
def clean_nged_power_data(context: AssetExecutionContext, config: CleanNgedPowerDataConfig) -> None:
    """Flags NGED power readings that should not be used, writing ``cleaned_power_time_series``.

    Reads the whole raw ``power_time_series`` Delta table and the ``TimeSeriesMetadata`` table,
    passes both to ``nged_data.cleaning.flag_nged_power``, and overwrites the
    ``cleaned_power_time_series`` Delta table with the result: every raw row, plus a nullable
    ``drop_reason`` column that is null for a row that passed and otherwise names the rule that
    flagged it. The raw table is never modified. Training, CV prediction, ``live_forecasts``,
    eligibility, effective capacity, ``metrics``, and the dashboards read only the unflagged rows
    of the cleaned table. Only the ingest and its ``power_data_is_fresh`` check read the raw table.

    Runs hourly in ``power_time_series_and_metadata_job``, straight after the ingest. NGED
    delivers about every 6 hours, so most runs skip. The asset compares four values with what the
    cleaned table's newest write commit recorded — the raw table's id, the raw table's Delta
    version, the hash of the cleaning code, and a fingerprint of the ``TimeSeriesMetadata`` table
    — and returns at once with ``skipped: True`` when all four match. A failed rebuild leaves the
    old version recorded, so the next hourly run rebuilds again. The
    ``cleaned_power_keeps_up_with_raw`` check warns if rebuilds keep failing. Set the run config
    ``force`` to rebuild regardless.

    A degraded run is a missing raw table (the asset logs and reports ``n_rows: 0``) or a vacuum
    failure after the new table is written (``vacuum_failed: True``, reported to Sentry). A failing
    cleaning rule or a contract violation is our own bug, so the asset raises, and every reader
    carries on with the last good cleaned table.

    Output metadata: ``n_rows``, ``n_rows_kept``, ``n_time_series``, and for each drop reason
    ``drop_reason/<reason>/n_rows``, ``n_time_series``, ``min_power``, ``max_power``,
    ``first_time``, and ``last_time`` of the flagged rows.

    Further reading: the cleaning roadmap at
    <https://openclimatefix.github.io/nged-substation-forecast/roadmap/data-cleaning/>.
    """
    settings = Settings()
    storage_options = settings.storage_options
    options = typeddict_to_dict(storage_options)
    raw_path = settings.power_time_series_data_path
    cleaned_path = settings.cleaned_power_time_series_data_path

    if not delta_table_exists(raw_path, storage_options):
        context.log.warning(f"{raw_path} does not exist yet, so there is nothing to clean.")
        context.add_output_metadata({"n_rows": 0})
        return

    raw_table = DeltaTable(raw_path, storage_options=options)
    raw_version = raw_table.version()
    raw_table_id = str(raw_table.metadata().id)

    # allow_superfluous_columns because the parquet also carries h3_res_5 and other geo columns.
    metadata = TimeSeriesMetadata.validate(
        pl.read_parquet(settings.metadata_path, storage_options=options),
        allow_superfluous_columns=True,
    )
    metadata_hash = metadata_fingerprint(pl.DataFrame._from_pydf(metadata._df))

    existing = read_cleaning_provenance(cleaned_path, storage_options)
    if (
        not config.force
        and existing is not None
        and existing.raw_table_id == raw_table_id
        and existing.raw_version == raw_version
        and existing.code_hash == CLEANING_CODE_HASH
        and existing.metadata_hash == metadata_hash
    ):
        context.log.info(f"{cleaned_path} is up to date with raw version {raw_version}.")
        context.add_output_metadata({"skipped": True, "raw_version": raw_version})
        return

    # Pinned to the version just read, so an ingest that commits mid-run cannot make the recorded
    # version disagree with the rows cleaned.
    power = pt.LazyFrame.from_existing(
        pl.scan_delta(raw_path, version=raw_version, storage_options=options)
    ).set_model(PowerTimeSeries)
    flagged = flag_nged_power(power=power, metadata=metadata)
    # Delta file order is not sorted, and `PowerTimeSeries.validate` rejects unsorted rows.
    cleaned = CleanedPowerTimeSeries.validate(
        flagged.collect(engine="streaming").sort("time_series_id", "time")
    )

    if_local_path_then_make_parent_dir(cleaned_path)
    provenance = CleaningProvenance(
        raw_table_id=raw_table_id,
        raw_version=raw_version,
        code_hash=CLEANING_CODE_HASH,
        metadata_hash=metadata_hash,
        git_sha=current_git_sha(),
    )
    vacuum_failed = False
    try:
        write_cleaned_power_time_series(
            cleaned=cleaned,
            table_uri=cleaned_path,
            provenance=provenance,
            storage_options=storage_options,
        )
    except VacuumError as exc:
        # The new table is already written; only deleting the superseded files failed.
        context.log.exception(f"Could not vacuum {cleaned_path}")
        report_asset_degradation(asset_name="clean_nged_power_data", exc=exc)
        vacuum_failed = True

    context.add_output_metadata(
        {
            "n_rows": cleaned.height,
            "n_rows_kept": cleaned.filter(pl.col("drop_reason").is_null()).height,
            "n_time_series": cleaned["time_series_id"].n_unique(),
            "raw_version": raw_version,
            "vacuum_failed": vacuum_failed,
            **_drop_reason_metadata(cleaned),
        }
    )
