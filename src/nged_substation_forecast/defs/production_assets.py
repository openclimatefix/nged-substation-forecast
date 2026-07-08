"""Production Dagster assets: model promotion and 6-hourly live inference.

New file (``defs/cv_assets.py`` is already ~900 lines): ``promoted_model`` (manually-triggered
promotion of a champion model to local disk) and ``live_forecasts`` (the 6-hourly inference asset
that reads it). Both stay thin shells over the pure/IO-light helpers in
``ml_core._production_helpers``.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import mlflow
import patito as pt
import polars as pl
from contracts._uri import ensure_local_parent
from contracts.ml_schemas import AllFeatures
from contracts.settings import Settings
from dagster import (
    AssetDep,
    AssetExecutionContext,
    Config,
    MetadataValue,
    TableRecord,
    TimeWindowPartitionMapping,
    TimeWindowPartitionsDefinition,
    asset,
)
from delta_store.power_forecasts import write_power_forecasts
from deltalake import DeltaTable
from ml_core._mlflow_runs import list_promotable_runs
from ml_core._production_helpers import (
    AvailabilityModeType,
    build_live_power_frame,
    fetch_model_artifacts,
    load_forecaster_from_dir,
    select_nwp_init_time,
)

from nged_substation_forecast.defs.cv_assets import _load_engineering_inputs

LIVE_FORECAST_HORIZON: Final[timedelta] = timedelta(days=14)
"""How far past ``power_fcst_init_time`` ``live_forecasts`` forecasts — inside ECMWF ENS's
~15-day horizon."""

LIVE_POWER_HISTORY: Final[timedelta] = timedelta(days=15)
"""How far before ``power_fcst_init_time`` the live power spine (``build_live_power_frame``)
reaches.

Must cover the longest power lag feature any production model uses (currently up to 336 h /
14 days) plus a margin.
"""

live_forecast_partitions = TimeWindowPartitionsDefinition(
    cron_schedule="0 0,6,12,18 * * *",
    start="2026-06-28-00:00",
    fmt="%Y-%m-%d-%H:%M",
    timezone="UTC",
)
"""One partition per 6-hourly tick (00/06/12/18 UTC). ``start`` is a few days before this asset
shipped — no need for a deep empty backlog on a brand-new live asset.

**Partition semantics**: a partition key names the *start* of its 6-hour window; the window runs
until the *next* partition's key, 6 hours later. A partition's ``power_fcst_init_time`` (when its
forecast is initialised) is that window's *end* — i.e. the next tick's timestamp, not the key's
own timestamp (see ``live_forecasts``'s docstring for why). E.g. partition key
``"2026-07-04-00:00"`` covers the window from 2026-07-04 00:00 UTC (the key itself) to
2026-07-04 06:00 UTC (the next tick), so its ``power_fcst_init_time`` is 2026-07-04 06:00 UTC —
six hours after the timestamp named in the key, not at the midnight the key names.
"""


@asset
def promotable_model_runs(context: AssetExecutionContext) -> None:
    """List MLflow fold runs eligible for promotion via ``promoted_model``.

    Purely informational: materialise this on demand (it has no dependents and writes nothing to
    disk) to refresh the candidate list as a metadata table in the Dagster UI, then copy the
    champion's ``run_id`` into ``promoted_model``'s launchpad. The champion is still picked by
    eye off the MLflow leaderboard (metrics vary per experiment, so there is no single sort key to
    automate the pick) — this just saves retyping/misremembering the run id.
    """
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    runs = list_promotable_runs()

    table = [
        TableRecord(
            {
                "run_id": run.run_id,
                "experiment_name": run.experiment_name,
                "fold_id": run.fold_id,
                "started_at": run.start_time.strftime("%Y-%m-%d %H:%M UTC"),
            }
        )
        for run in runs
    ]
    context.add_output_metadata(
        {"n_candidates": len(runs), "candidates": MetadataValue.table(table)}
    )


class PromotedModelConfig(Config):
    """Run config for the manually-triggered ``promoted_model`` asset."""

    mlflow_run_id: str
    """The champion fold run id, picked from the MLflow leaderboard (or from
    ``promotable_model_runs``'s candidate table)."""


@asset
def promoted_model(context: AssetExecutionContext, config: PromotedModelConfig) -> None:
    """Promote a champion model from MLflow to local disk for zero-MLflow-at-runtime inference.

    Manually triggered from the Dagster UI launchpad with ``mlflow_run_id`` set to the champion
    fold's run id — materialise ``promotable_model_runs`` first and copy the id from its output
    metadata table if you don't have it to hand. Downloads that run's saved model artifacts to
    ``Settings.production_model_path`` (via ``ml_core._production_helpers.fetch_model_artifacts``,
    which replaces the directory atomically), then reads back ``meta.json`` to report provenance.
    ``live_forecasts`` reads this directory with a plain disk load — never MLflow.

    Promotion as a Dagster materialisation gives an audit trail and lineage for free, rather than
    a bare script (a script wrapper for the eventual Docker build (#222) stays trivial by calling
    the same ``fetch_model_artifacts`` helper).
    """
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    production_model_path = Path(settings.production_model_path)
    fetch_model_artifacts(config.mlflow_run_id, production_model_path)

    meta = json.loads((production_model_path / "meta.json").read_text())
    model_params = meta.get("model_params", {})
    context.add_output_metadata(
        {
            "mlflow_run_id": config.mlflow_run_id,
            "model_class": meta.get("model_class"),
            "experiment_name": model_params.get("experiment_name"),
            "n_trained_time_series": len(meta.get("trained_time_series_ids", [])),
            "path": str(production_model_path),
        }
    )


def _available_nwp_init_times(settings: Settings) -> list[datetime]:
    """Return the distinct ``init_time``s present in the ``nwp`` Delta table.

    Reads only Delta partition metadata (``DeltaTable.partitions()``, no data scan) and parses
    the ``init_time`` partition values — naive ``"YYYY-MM-DD HH:MM:SS.ffffff"`` strings on disk —
    into tz-aware UTC datetimes.
    """
    delta_table = DeltaTable(settings.nwp_data_path, storage_options=settings.storage_options)
    raw_values = {partition["init_time"] for partition in delta_table.partitions()}
    return [
        datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
        for value in raw_values
    ]


class LiveForecastsConfig(Config):
    """Run config for the ``live_forecasts`` asset."""

    availability_mode: AvailabilityModeType = "live"
    """``"live"`` (the scheduled default) uses the freshest NWP run present as of
    ``power_fcst_init_time``, no modelled delay. ``"replay"`` (manual backfills only)
    reconstructs what was available ``nwp_publication_delay_hours`` before
    ``power_fcst_init_time``. See ``select_nwp_init_time``."""


@asset(
    partitions_def=live_forecast_partitions,
    deps=[
        AssetDep(
            "ecmwf_ens",
            partition_mapping=TimeWindowPartitionMapping(start_offset=-16, end_offset=0),
        ),
        "power_time_series_and_metadata",
        "promoted_model",
    ],
)
def live_forecasts(context: AssetExecutionContext, config: LiveForecastsConfig) -> None:
    """Production inference: forecast from the latest NWP for one 6-hourly slot.

    **Partition semantics — read this before backfilling**: a partition key names the *start* of
    its 6-hour window (``context.partition_time_window.start``); ``power_fcst_init_time`` — when
    this partition's forecast is initialised — is that window's *end*
    (``context.partition_time_window.end``). Dagster always defines a
    ``TimeWindowPartitionsDefinition`` this way: each key is a window's start, and the window
    extends until the *next* partition's key. ``live_forecast_partitions`` ticks every 6 hours
    (00/06/12/18 UTC), so every window — and the gap between a key's timestamp and its
    ``power_fcst_init_time`` — is exactly 6 hours.

    For example, partition key ``"2026-07-04-00:00"`` covers the window from 2026-07-04 00:00 UTC
    (the key itself) up to 2026-07-04 06:00 UTC (the next tick). So that partition's
    ``power_fcst_init_time`` is 2026-07-04 06:00 UTC: six hours after the timestamp named in the
    key, not the midnight the key names.

    Loads the production model from a plain disk directory (``load_forecaster_from_dir`` against
    ``Settings.production_model_path``, populated out-of-band by the ``promoted_model``
    asset) — **no MLflow import or call anywhere in this asset**; live performance is tracked by
    production monitoring, never logged here. Forecasts exactly
    ``forecaster.trained_time_series_ids`` (never today's eligibility set — the train==predict
    population invariant) across every NWP ensemble member, using single-run feature engineering
    stamped with this partition's ``power_fcst_init_time``.

    NWP availability is resolved via ``config.availability_mode``: the scheduled tick always
    uses ``"live"`` (freshest run actually present, no modelled delay); manual backfills of past
    partitions pass ``"replay"`` (reconstructs what was available ``nwp_publication_delay_hours``
    before ``power_fcst_init_time``). See ``select_nwp_init_time``.

    Writes idempotently: overwrites exactly this partition's rows in ``power_forecasts``
    (``experiment_name``, ``fold_id="live"``, and this ``power_fcst_init_time``) via
    ``write_power_forecasts``'s ``replace_predicate_extra``, so re-running a 6-hourly slot (or
    replaying one) never duplicates rows or wipes the rest of the ``"live"`` fold.

    Note: weather-lag features would go null at live time (only one NWP run is loaded here) —
    none are in the current champion config, but a future feature change touching weather lags
    should trip over this consciously.
    """
    settings = Settings()
    power_fcst_init_time = context.partition_time_window.end

    forecaster = load_forecaster_from_dir(Path(settings.production_model_path))
    trained_ids = forecaster.trained_time_series_ids
    if not trained_ids:
        raise ValueError(
            "The production model has no trained time series, so there is nothing to "
            "forecast. Re-promote `promoted_model` with a model that has trained boosters."
        )

    available = _available_nwp_init_times(settings)
    nwp_init = select_nwp_init_time(
        available,
        power_fcst_init_time=power_fcst_init_time,
        availability_mode=config.availability_mode,
    )

    power_ts, metadata_df, nwp_lf = _load_engineering_inputs(
        settings,
        trained_ids,
        window_start=power_fcst_init_time - LIVE_POWER_HISTORY,
        window_end=power_fcst_init_time + LIVE_FORECAST_HORIZON,
        init_time_start=nwp_init,
        init_time_end=nwp_init,
    )
    power_full = build_live_power_frame(
        power_ts,
        trained_ids,
        power_fcst_init_time=power_fcst_init_time,
        history=LIVE_POWER_HISTORY,
        horizon=LIVE_FORECAST_HORIZON,
    )

    features = forecaster.feature_engineer.engineer(
        selected_features=forecaster.model_params.selected_features,
        power_time_series=power_full,
        time_series_metadata=metadata_df,
        nwp=nwp_lf,
        power_fcst_init_time=power_fcst_init_time,
        nwp_init_time=nwp_init,
    )
    # History rows (valid_time <= power_fcst_init_time) and rows outside this NWP run's coverage
    # (ensemble_member null from the join miss) are join artefacts, not genuine forecasts.
    genuine_forecasts: pl.LazyFrame = features
    genuine_forecasts = genuine_forecasts.filter(
        pl.col("valid_time") > power_fcst_init_time, pl.col("ensemble_member").is_not_null()
    )
    features = pt.LazyFrame.from_existing(genuine_forecasts).set_model(AllFeatures)

    forecasts = forecaster.predict(features)  # fold_id="live" is the default.
    if forecasts.height == 0:
        raise ValueError(
            f"live_forecasts produced 0 rows for power_fcst_init_time="
            f"{power_fcst_init_time.isoformat()} (nwp_init_time={nwp_init.isoformat()}). Check "
            "NWP coverage and the model's trained population."
        )

    ensure_local_parent(settings.power_forecasts_data_path)
    write_power_forecasts(
        forecasts,
        settings.power_forecasts_data_path,
        replace_partition=(forecaster.model_params.experiment_name, "live"),
        replace_predicate_extra=f"power_fcst_init_time = '{power_fcst_init_time.isoformat()}'",
        storage_options=settings.storage_options,
    )

    context.add_output_metadata(
        {
            "power_fcst_init_time": str(power_fcst_init_time),
            "availability_mode": config.availability_mode,
            "nwp_init_time": str(nwp_init),
            "n_rows": forecasts.height,
            "n_time_series": len(set(forecasts["time_series_id"].to_list())),
            "n_ensemble_members": len(set(forecasts["ensemble_member"].to_list())),
            "experiment_name": forecaster.model_params.experiment_name,
        }
    )
