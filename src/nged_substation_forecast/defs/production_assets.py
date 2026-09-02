"""Production Dagster assets: model promotion and 6-hourly live inference.

``promoted_model`` promotes a champion model to local disk; ``live_forecasts`` is the 6-hourly
inference asset that reads it. Both are thin shells over ``ml_core.production_helpers``. Design
rationale is on [Production Deployment —
Design](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/).
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import mlflow
import patito as pt
import polars as pl
from contracts.ml_schemas import AllFeatures
from contracts.settings import Settings
from contracts.typing_utils import typeddict_to_dict
from contracts.uri import if_local_path_then_make_parent_dir
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
from ml_core.base_forecaster import load_trained_metadata
from ml_core.mlflow_runs import list_promotable_runs
from ml_core.production_helpers import (
    AvailabilityModeType,
    build_live_power_frame,
    fetch_model_artifacts,
    load_forecaster_from_dir,
    select_nwp_init_time,
)

from nged_substation_forecast._sentry import send_forecast_checkin
from nged_substation_forecast.defs._engineering_inputs import load_engineering_inputs
from nged_substation_forecast.defs._tags import PRODUCTION_LAYER_TAGS, RESEARCH_LAYER_TAGS

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
    # DUPLICATED SCHEDULE: this crontab is the canonical live cadence, but it is also copied into
    # LIVE_FORECAST_MONITOR_CONFIG in _sentry.py so the Sentry missed-check-in monitor expects a
    # heartbeat on the same 6-hourly cadence. That module can't import this one back (it would be a
    # circular import — this module imports send_forecast_checkin from it). If you change this
    # crontab, change the copy in _sentry.py too.
    cron_schedule="0 0,6,12,18 * * *",
    start="2026-06-28-00:00",
    fmt="%Y-%m-%d-%H:%M",
    timezone="UTC",
)
"""One partition per 6-hourly tick (00/06/12/18 UTC).

**Partition semantics**: a partition key names the *start* of its 6-hour window, and
``power_fcst_init_time`` is that window's *end*, six hours later — see ``live_forecasts``'s
docstring for the full explanation and a worked example.
"""


@asset(tags=RESEARCH_LAYER_TAGS)
def promotable_model_runs(context: AssetExecutionContext) -> None:
    """List MLflow fold runs eligible for promotion via ``promoted_model``.

    Purely informational: writes nothing to disk and has no dependents. See [Operating the live
    service → Step 1 — Pick a champion
    model](https://openclimatefix.github.io/nged-substation-forecast/live_service/operations/#step-1-pick-a-champion-model).
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


@asset(tags=RESEARCH_LAYER_TAGS)
def promoted_model(context: AssetExecutionContext, config: PromotedModelConfig) -> None:
    """Promote a champion model from MLflow to local disk for zero-MLflow-at-runtime inference.

    Manually triggered from the Dagster UI launchpad with ``mlflow_run_id`` set to the champion
    fold's run id. Downloads that run's saved model artifacts to
    ``Settings.production_model_path`` (via ``ml_core.production_helpers.fetch_model_artifacts``,
    which replaces the directory atomically), then reads back ``meta.json`` to report provenance.

    Refuses a model whose saved config this code cannot rebuild — a feature name it cannot parse,
    or a ``model_params`` key it no longer declares — before the directory is replaced, so the
    previous champion stays in place and keeps serving. This asset catches nothing, unlike the
    rest of ``defs/``: every such refusal reaches the operator as a failed materialisation. See
    [The rules](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules),
    [Promote the champion via a Dagster asset, not a
    script](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#promote-the-champion-via-a-dagster-asset-not-a-script),
    and [Operating the live service → Step 2 — Materialise
    promoted_model](https://openclimatefix.github.io/nged-substation-forecast/live_service/operations/#step-2-materialise-promoted_model).
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


def _parse_utc_init_time(value: str) -> datetime:
    """Parse one partition-value ``init_time`` string into a tz-aware UTC datetime.

    delta-rs renders our own writes as naive ``"YYYY-MM-DD HH:MM:SS[.ffffff]"`` strings, so the
    common case just attaches UTC. An offset-carrying string is handled separately —
    ``.replace(tzinfo=UTC)`` would silently *relabel* it to the wrong instant rather than convert
    it, and nothing upstream guarantees delta-rs never renders one.
    """
    parsed = datetime.fromisoformat(value)
    return parsed.astimezone(UTC) if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _available_nwp_init_times(settings: Settings) -> list[datetime]:
    """Return the distinct ``init_time``s present in the ``nwp`` Delta table.

    Reads only Delta partition metadata (``DeltaTable.partitions()``, no data scan) and parses the
    ``init_time`` values via ``_parse_utc_init_time``. Uses ``datetime.fromisoformat`` rather than
    a fixed ``strptime`` pattern, because a whole-second ``init_time`` has no fractional part for
    ``strptime``'s ``%f`` to match, and it would raise.

    Deliberately uncaught: an unparseable value means our own write path
    (``delta_store.nwp.write_nwp``) corrupted its own metadata, and [rule
    1](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules)
    reserves raising for exactly that case. ``live_forecasts`` calls this directly and is meant to
    fail loudly; ``live_forecasts_are_healthy``, the one caller that must stay fail-open, already
    wraps the whole evaluation in ``except BaseException``.
    """
    delta_table = DeltaTable(
        settings.nwp_data_path, storage_options=typeddict_to_dict(settings.storage_options)
    )
    raw_values = {partition["init_time"] for partition in delta_table.partitions()}
    return [_parse_utc_init_time(value) for value in raw_values]


class LiveForecastsConfig(Config):
    """Run config for the ``live_forecasts`` asset."""

    availability_mode: AvailabilityModeType = "live"
    """``"live"`` (the scheduled default) uses the freshest NWP run present, no modelled delay.
    ``"replay"`` (manual backfills only) reconstructs what was available
    ``nwp_publication_delay_hours`` before ``power_fcst_init_time``. See ``select_nwp_init_time``
    and [Resolve NWP availability
    asymmetrically](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#resolve-nwp-availability-asymmetrically-live-vs-replay).
    """


@asset(
    tags=PRODUCTION_LAYER_TAGS,
    partitions_def=live_forecast_partitions,
    deps=[
        AssetDep(
            "ecmwf_ens",
            # Dagster expresses a TimeWindowPartitionMapping offset in units of the *downstream*
            # partitions_def when upstream and downstream differ — here, live_forecast_partitions'
            # 6-hourly ticks, not ecmwf_ens_partitions' daily ones. So start_offset=-16 reaches
            # back 16 * 6h = 4 days, not 16 days. This is a lineage-only safety margin (it decides
            # what the Dagster UI graph and a `--with upstream` materialisation consider a parent,
            # not what live_forecasts actually reads): comfortably more than a healthy NWP run is
            # ever stale, so it covers several missed daily runs before live_forecasts_are_healthy
            # would already have alarmed. See the missed-NWP-run deadline at
            # https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#read-the-live-forecast-back-off-disk-with-a-second-asset-check
            partition_mapping=TimeWindowPartitionMapping(start_offset=-16, end_offset=0),
        ),
        "power_time_series_and_metadata",
        # `promoted_model` is deliberately NOT a dep: the model reaches
        # `Settings.production_model_path` out-of-band (an MLflow fetch on a laptop, or the Docker
        # image on the production box) — a filesystem input, not a Dagster data-flow edge.
        # Declaring the edge would leave a permanently un-materialised `promoted_model` parent on
        # the box. See
        # https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#bake-the-model-into-the-image-at-build-time
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
    ``Settings.production_model_path``, populated out-of-band by the ``promoted_model`` asset) —
    no MLflow import or call anywhere in this asset. A model whose saved config this code cannot
    rebuild fails at that load rather than partway through feature engineering; promotion applies
    the same check, so this only bites when the code changed after the champion was promoted.

    Forecasts exactly ``forecaster.trained_time_series_ids`` — never today's eligibility set, the
    train==predict population invariant — across every NWP ensemble member, using single-run
    feature engineering stamped with this partition's ``power_fcst_init_time``. See [Serve only the
    trained
    population](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#serve-only-the-trained-population)
    and [Run live inference in single-run
    mode](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#run-live-inference-in-single-run-mode-not-bulk).

    Each series' location comes from the model's own frozen metadata copy
    (``load_trained_metadata``), never from the ``TimeSeriesMetadata`` roster, so a roster that is
    unreadable or has lost rows can neither fail a slot nor silently drop a series from it. The H3
    cells the NWP scan is pruned to are therefore the cells the model trained against, not whatever
    the roster says today.

    NWP availability follows ``config.availability_mode`` — see ``LiveForecastsConfig``.

    Writes idempotently: overwrites exactly this partition's rows in ``power_forecasts``
    (``experiment_name``, ``fold_id="live"``, and this ``power_fcst_init_time``) via
    ``write_power_forecasts``'s ``replace_predicate_extra``, so re-running a 6-hourly slot (or
    replaying one) never duplicates rows or wipes the rest of the ``"live"`` fold.

    Note: only one NWP run is loaded here, and "live" availability applies no publication delay,
    so a weather-lag feature goes null only when that run is closer than
    ``NWP_PUBLICATION_DELAY_HOURS`` to ``power_fcst_init_time`` — e.g. the 06:00 slot, when only
    that morning's run has landed — and is populated at every other slot; see
    ``test_live_weather_lag_nulls_only_when_the_selected_run_is_too_fresh``. None are in the
    current champion config, but a future feature change touching weather lags should trip over
    this consciously.
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

    metadata_df = load_trained_metadata(Path(settings.production_model_path))
    power_ts, nwp_lf = load_engineering_inputs(
        settings,
        time_series_ids=trained_ids,
        metadata=metadata_df,
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

    if_local_path_then_make_parent_dir(settings.power_forecasts_data_path)
    write_power_forecasts(
        forecasts,
        settings.power_forecasts_data_path,
        replace_partition=(forecaster.model_params.experiment_name, "live"),
        replace_predicate_extra=f"power_fcst_init_time = '{power_fcst_init_time.isoformat()}'",
        storage_options=settings.storage_options,
    )

    # Heartbeat to Sentry's missed-check-in alarm, after a successful live write only — never on a
    # replay backfill. See nged_substation_forecast._sentry and
    # https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#send-telemetry-to-sentry-and-alarm-on-absence
    if config.availability_mode == "live":
        send_forecast_checkin(settings)

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
