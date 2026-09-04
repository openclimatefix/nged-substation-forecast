"""Cross-validation Dagster assets.

These assets implement the experiment-independent, fold-partitioned CV layer. Each asset is a
thin orchestration shell delegating its logic to the pure helpers in ``ml_core.cv_helpers`` so
the logic stays fast to unit-test and the assets stay readable.
"""

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import mlflow
import patito as pt
import polars as pl
from contracts.config_schemas import load_cv_config
from contracts.ml_schemas import EligibleTimeSeries, EvalScopeType, Metrics
from contracts.power_schemas import (
    EffectiveCapacity,
    PowerForecast,
    PowerTimeSeries,
    TimeSeriesMetadata,
)
from contracts.settings import Settings
from contracts.typing_utils import typeddict_to_dict
from contracts.uri import (
    ObjectStoreOptions,
    delta_table_exists,
    if_local_path_then_make_parent_dir,
)
from dagster import (
    AssetExecutionContext,
    Config,
    DynamicPartitionsDefinition,
    StaticPartitionsDefinition,
    asset,
)
from delta_store.power_forecasts import write_power_forecasts
from deltalake import write_deltalake
from ml_core.cv_helpers import (
    date_to_utc_datetime,
    eligible_time_series_ids,
    parse_cv_partition_key,
)
from ml_core.metrics import (
    NoOverlappingActualsError,
    build_mlflow_aggregate_metrics,
    compute_effective_capacity,
    compute_metrics,
    enrich_metrics_rows,
)
from ml_core.mlflow_runs import (
    get_or_create_experiment,
    get_or_create_fold_run,
    get_or_create_parent_run,
    load_experiment_forecaster,
)
from ml_core.repro import MlflowTags, provenance_tags
from nged_data.storage import time_series_coverage

from nged_substation_forecast.defs._engineering_inputs import (
    MAX_NWP_LEAD,
    load_engineering_inputs,
)
from nged_substation_forecast.defs._tags import RESEARCH_LAYER_TAGS

# The CV folds are the shared leaderboard evaluation protocol, read from conf/cv/default.yaml
# (never hard-coded) so every experiment and asset agrees on the same folds. Loaded at import so
# the partition keys are available when Dagster builds the asset graph. The raw CV_CONFIG_PATH
# env var is read directly here (rather than instantiating Settings, which needs the .env
# secrets) so the partition set can be built without any credentials — while still respecting
# the same env var Settings' cv_config_path field would use, unlike reading
# Settings.model_fields["cv_config_path"].default directly, which silently ignores it.
_cv_config = load_cv_config(
    Path(os.environ.get("CV_CONFIG_PATH", Settings.model_fields["cv_config_path"].default))
)

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

_PREDICT_INIT_CHUNK: Final[timedelta] = timedelta(days=14)
"""``init_time`` window processed per ``cv_power_forecasts`` iteration.

Prediction fans every NWP run out across all ~51 ensemble members, so the full validation window at
once is tens of GB. ``init_time`` is one of the table's two partition columns and the axis that
inflates the output,
so chunking by it bounds the per-iteration forecast frame (~2-3 GB at 14 days) while each partition
is still read exactly once. See ``cv_power_forecasts``.
"""


@asset(
    tags=RESEARCH_LAYER_TAGS,
    partitions_def=cv_fold_partitions,
    deps=["power_time_series_and_metadata"],
)
def eligible_time_series(context: AssetExecutionContext) -> None:
    """Compute and persist the canonical eligible ``time_series_id``s for one CV fold.

    Reads observed-power coverage from the ``power_time_series_and_metadata`` Delta table. A time
    series is eligible for a fold when its coverage has at least ``min_training_months`` of history
    before the fold's ``val_start`` *and* reaches the fold's ``val_end``. Eligibility is derived
    from data coverage alone (not from any model/config), so every experiment evaluates the fold on
    the identical population — this is what keeps the leaderboard apples-to-apples. See
    "Eligibility" in
    <https://openclimatefix.github.io/nged-substation-forecast/ml_experimentation/cross-validation-folds/#eligibility>
    for the fold definitions this population is computed against.

    The result is written to the ``eligible_time_series`` Delta table as one partition per
    ``fold_id`` via an idempotent partition overwrite, so re-materialising a fold replaces its rows
    rather than duplicating them. ``trained_cv_model`` reads this fold's partition to select which
    series to train on; ``cv_power_forecasts`` does not read it directly, but inherits the same
    population through the trained model's ``trained_time_series_ids``. A stale or missing
    partition here therefore shows up downstream as a fold trained on the wrong population, not as
    a failure at either of those assets.
    """
    settings = Settings()
    storage_options = settings.storage_options
    fold_id = context.partition_key
    fold = _cv_config.get_fold(fold_id)

    coverage = time_series_coverage(settings.power_time_series_data_path, storage_options)

    min_training_months = fold.min_training_months or _cv_config.min_training_months
    eligible_ids = eligible_time_series_ids(coverage, fold, min_training_months=min_training_months)

    eligible_df = EligibleTimeSeries.validate(
        pl.DataFrame(
            {
                "fold_id": pl.Series([fold_id] * len(eligible_ids), dtype=pl.String),
                "time_series_id": pl.Series(eligible_ids, dtype=pl.Int32),
            }
        )
    )

    if_local_path_then_make_parent_dir(settings.eligible_time_series_data_path)
    write_deltalake(
        table_or_uri=settings.eligible_time_series_data_path,
        data=eligible_df.to_arrow(),
        mode="overwrite",
        predicate=f"fold_id = '{fold_id}'",
        partition_by=["fold_id"],
        storage_options=typeddict_to_dict(storage_options),
    )

    context.add_output_metadata(
        {
            "fold_id": fold_id,
            "n_eligible_time_series": len(eligible_ids),
            "n_time_series_in_coverage": len(coverage),
            "eligible_time_series_ids": str(eligible_ids),
            "val_start": str(fold.val_start),
            "val_end": str(fold.val_end),
            "min_training_months": min_training_months,
        }
    )


@asset(tags=RESEARCH_LAYER_TAGS, deps=["power_time_series_and_metadata"])
def effective_capacity(context: AssetExecutionContext) -> None:
    """Compute and persist each series' v0.1 effective capacity (full-history P99 of ``|power|``).

    Reads the full ``power_time_series`` Delta and writes one row per ``time_series_id`` to the
    ``effective_capacity`` Delta table (``Settings.effective_capacity_data_path``): the 99th
    percentile of ``abs(power)`` over the series' entire observed history, with ``time`` set to the
    latest observed timestep. This full-history capacity is the NMAE denominator used by the
    ``metrics`` asset, replacing the validation-window P99 that would otherwise vary fold to fold.

    The whole (small — one row per series) table is overwritten on each materialisation. v0.1 is
    deliberately one scalar row per series, **not** the value repeated at every half-hour —
    densifying a constant buys nothing.

    A future upgrade (v0.7) swaps the P99 for a time-varying capacity estimator,
    emitting one row per ``(time_series_id, time)``; the ``EffectiveCapacity`` schema is unchanged,
    but ``compute_metrics`` then joins capacity as a temporal as-of join rather than on
    ``time_series_id`` alone (same doc section).
    """
    settings = Settings()
    storage_options = settings.storage_options
    power_lf = pt.LazyFrame.from_existing(
        pl.scan_delta(
            settings.power_time_series_data_path, storage_options=typeddict_to_dict(storage_options)
        )
    ).set_model(PowerTimeSeries)
    capacity_df = compute_effective_capacity(power_lf)

    if_local_path_then_make_parent_dir(settings.effective_capacity_data_path)
    write_deltalake(
        table_or_uri=settings.effective_capacity_data_path,
        data=capacity_df.to_arrow(),
        mode="overwrite",
        storage_options=typeddict_to_dict(storage_options),
    )

    context.add_output_metadata(
        {
            "n_time_series": capacity_df.height,
            "effective_capacity_data_path": str(settings.effective_capacity_data_path),
        }
    )


def _time_series_ids_missing_metadata(
    metadata: pt.DataFrame[TimeSeriesMetadata], time_series_ids: list[int]
) -> list[int]:
    """The requested series that have no row in the metadata parquet.

    Such a series cannot be forecast: ``TabularFeatureEngineer`` maps NWP to series through the
    metadata's ``h3_res_5``, so a series with no metadata row has no weather and produces no
    output rows. Worth naming rather than inferring from an empty result, and cheap to compute —
    the metadata frame is already eager and already filtered to ``time_series_ids``.

    The CV assets raise on a non-empty answer; ``live_forecasts`` does not, and reports the
    missing series through ``live_forecasts_are_healthy`` instead. CV is R&D and fails fast, so a
    silently-shrunk population can never poison a leaderboard comparison; ``live_forecasts`` is
    production and must never raise on an absent input, so it degrades and reports the gap. The
    split in full:
    <https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/>.
    """
    return sorted(set(time_series_ids) - set(metadata["time_series_id"].to_list()))


def _require_metadata_coverage(
    metadata: pt.DataFrame[TimeSeriesMetadata], time_series_ids: list[int], population: str
) -> None:
    """Raise if any requested series has no metadata row.

    Names the metadata cause specifically, because a series dropped for want of an H3 cell
    otherwise shows up only as a fold that trained on fewer series than its population — which is
    hard to diagnose and easy to miss. It is not a complete guard on that larger property: a
    series can also drop out for want of NWP in the window, or of any non-null power.

    Args:
        metadata: The loaded metadata, already filtered to ``time_series_ids``.
        time_series_ids: The population the caller asked for.
        population: What that population is, for the error message (e.g. ``"eligible"``).
    """
    missing = _time_series_ids_missing_metadata(metadata, time_series_ids)
    if missing:
        # Capped for the same reason `checks._MAX_MISSING_SERIES_LISTED` caps its list: at V2
        # scale an empty metadata parquet would otherwise spell out ~2,500 ids.
        listed = ", ".join(str(ts_id) for ts_id in missing[:20])
        suffix = "" if len(missing) <= 20 else ", …"
        raise ValueError(
            f"{len(missing)} of {len(time_series_ids)} {population} time series have no row in "
            f"the metadata parquet, so they have no H3 cell and cannot be joined to NWP: "
            f"[{listed}{suffix}]. Re-materialise `power_time_series_and_metadata`."
        )


def _load_roster(
    settings: Settings, time_series_ids: list[int]
) -> pt.DataFrame[TimeSeriesMetadata]:
    """Read the ``TimeSeriesMetadata`` roster, filtered to ``time_series_ids``.

    **Research callers only.** The roster is the live registry of what NGED operates, so a fault in
    it — an off-contract file, or one rebuilt from a snapshot that dropped rows — must stop a
    training or scoring run rather than silently shrink its population. ``live_forecasts`` reads
    ``ml_core.base_forecaster.load_trained_metadata`` instead.

    Args:
        settings: Application settings (data paths, credentials).
        time_series_ids: The population to keep.

    Returns:
        One row per series in ``time_series_ids`` that the roster covers — check the coverage with
        ``_require_metadata_coverage``.
    """
    return pt.DataFrame(
        pl.read_parquet(
            settings.metadata_path,
            storage_options=typeddict_to_dict(settings.storage_options),
        ).filter(pl.col("time_series_id").is_in(time_series_ids))
    ).set_model(TimeSeriesMetadata)


@asset(
    tags=RESEARCH_LAYER_TAGS,
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
    model is trained, and its artifacts are uploaded to the fold's MLflow run alongside a record of
    the training window and population.

    The fold run is resolved **by tag**, never by a handle passed between assets, so this is safe
    across processes and idempotent under Dagster retries — see "Cross-process run resolution:
    discover by tag, never pass handles" in
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/ml-orchestration/#cross-process-run-resolution-discover-by-tag-never-pass-handles>.
    Because that run is *reused* on every re-materialisation, the training window and population go
    in tags rather than MLflow params (which are write-once and would reject a changed value).
    """
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    experiment_name, fold_id = parse_cv_partition_key(context.partition_key)
    forecaster_cls, config = load_experiment_forecaster(experiment_name)

    fold = _cv_config.get_fold(fold_id)
    train_start = date_to_utc_datetime(fold.train_start)
    train_end = date_to_utc_datetime(fold.train_end, end_of_day=True)

    eligible_ids = (
        pl.scan_delta(
            settings.eligible_time_series_data_path,
            storage_options=typeddict_to_dict(settings.storage_options),
        )
        .filter(pl.col("fold_id") == fold_id)
        .select("time_series_id")
        .collect()["time_series_id"]
        .to_list()
    )
    if not eligible_ids:
        raise ValueError(
            f"No eligible time series for fold {fold_id!r}, so there is nothing to train. "
            "Either the `eligible_time_series` asset has not been materialised for this fold, "
            "or no time series meets the eligibility window — a series must have "
            f"`min_training_months` of history before val_start and observations through the "
            f"fold's val_end ({date_to_utc_datetime(fold.val_end, end_of_day=True)}). Materialise "
            "`eligible_time_series` for this fold and confirm power coverage reaches val_end."
        )

    metadata_df = _load_roster(settings, eligible_ids)
    _require_metadata_coverage(metadata_df, eligible_ids, population="eligible")
    power_ts, nwp_lf = load_engineering_inputs(
        settings,
        time_series_ids=eligible_ids,
        metadata=metadata_df,
        window_start=train_start,
        window_end=train_end,
        ensemble_members=[0],
    )

    forecaster = forecaster_cls(model_params=config)
    features = forecaster.feature_engineer.engineer(
        selected_features=config.selected_features,
        power_time_series=power_ts,
        time_series_metadata=metadata_df,
        nwp=nwp_lf,
    )
    forecaster.train(features, eligible_ids)

    n_trained = len(forecaster.trained_time_series_ids)
    if n_trained == 0:
        raise ValueError(
            f"Trained 0 of {len(eligible_ids)} eligible time series for fold {fold_id!r}: every "
            "eligible series had no usable (non-null power) rows in the training window "
            f"[{train_start}, {train_end}]. Check that power and NWP data exist for these series "
            "across the training window."
        )

    experiment_id = get_or_create_experiment(experiment_name)
    parent_run_id = get_or_create_parent_run(experiment_id)
    fold_run_id = get_or_create_fold_run(experiment_id, parent_run_id, fold_id)

    forecaster.save_to_mlflow(fold_run_id, time_series_metadata=metadata_df)
    with mlflow.start_run(run_id=fold_run_id):
        # MLflow params are immutable and the fold run is reused on every re-materialisation, so
        # nothing here that can legitimately change between materialisations may be a param.
        # `fold_id` is already set as a tag at run creation (get_or_create_fold_run, which is also
        # what resolves the run) — no need to duplicate it here as a param too.
        #
        # The training window comes from the CV config, which is edited between materialisations
        # as the archive grows (a fold's train_end is extended). The eligible/trained counters are
        # outputs of *this* materialisation, not identifying inputs — the eligible population
        # grows as power coverage extends, and can also shrink. All four are tags rather than
        # metrics: MLflow resolves a metric's "latest" value as the max over
        # (step, timestamp, value), not the newest write, so a metric would under-report a
        # genuinely *shrunk* count if two materialisations ever landed the same timestamp/step.
        # Tags are last-write-wins, which is the semantic actually wanted here.
        mlflow.set_tags(
            {
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "n_eligible_time_series": str(len(eligible_ids)),
                "n_trained_time_series": str(n_trained),
            }
        )
        # Provenance: the code + data versions that produced this fold's model — the load-bearing
        # stamp, since a fold can be trained days after registration on a different SHA. Tags (not
        # params) because provenance overwrites cleanly on re-materialise; these are the three
        # Delta tables the training path reads above.
        mlflow.set_tags(
            provenance_tags(
                "train",
                {
                    "power_time_series": settings.power_time_series_data_path,
                    "nwp_data": settings.nwp_data_path,
                    "eligible_time_series": settings.eligible_time_series_data_path,
                },
                storage_options=settings.storage_options,
            )
        )

    context.add_output_metadata(
        {
            "experiment_name": experiment_name,
            "fold_id": fold_id,
            "n_eligible_time_series": len(eligible_ids),
            "n_trained_time_series": n_trained,
            "train_start": str(train_start),
            "train_end": str(train_end),
            "fold_run_id": fold_run_id,
        }
    )


@asset(
    tags=RESEARCH_LAYER_TAGS,
    partitions_def=cv_experiment_folds,
    deps=["trained_cv_model"],
)
def cv_power_forecasts(context: AssetExecutionContext) -> None:
    """Predict the validation window for one ``(experiment, fold)`` partition and persist forecasts.

    Loads the model ``trained_cv_model`` saved for this fold back from MLflow, then forecasts the
    fold's **inclusive** validation window across **all** NWP ensemble members — the probabilistic
    leaderboard metrics are meaningless on a single member. The scored
    population is the model's own ``trained_time_series_ids`` (the train==predict invariant), so a
    fold is always scored on exactly the population it was trained on even if power coverage has
    drifted since training.

    To keep RAM bounded, prediction runs **one ``init_time`` window at a time**
    (``_PREDICT_INIT_CHUNK``). The full validation window fans every NWP run out across all ~51
    ensemble members and all trained series — tens of GB. ``init_time`` is one of the NWP table's
    two partition columns *and* the axis that inflates the output, so chunking by it bounds the
    per-iteration forecast frame (~2-3 GB) while each partition is still read exactly once. See
    "Bounding feature-engineering memory: prune the inputs, not the output" in
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/performance/#bounding-feature-engineering-memory-prune-the-inputs-not-the-output>.

    Forecasts are written to the ``power_forecasts`` Delta table keyed by
    ``(experiment_name, fold_id)``: the **first** chunk overwrites the partition (clearing any prior
    run) and the rest **append** to it, so a full re-materialisation replaces the fold's rows
    without ever holding all forecasts in memory. Chunks are written through
    ``delta_store.power_forecasts.write_power_forecasts``, which owns the table's compressed
    storage format (sort order, ``power_fcst`` precision rounding, parquet encodings). The
    fold's MLflow run is resolved **by tag** — never by a handle from ``trained_cv_model`` — so
    this is safe across processes.
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

    forecaster = forecaster_cls.load_from_mlflow(fold_run_id)
    trained_ids = forecaster.trained_time_series_ids
    if not trained_ids:
        raise ValueError(
            f"The model loaded for fold {fold_id!r} has no trained time series, so there is "
            "nothing to forecast. Re-materialise `trained_cv_model` for this fold."
        )

    if_local_path_then_make_parent_dir(settings.power_forecasts_data_path)
    n_rows = 0
    time_series_seen: set[int] = set()
    ensemble_members_seen: set[int] = set()

    # Before the loop, not inside it: the metadata does not vary by init_time window, and raising
    # on a later chunk would leave the partition holding a partial fold.
    metadata_df = _load_roster(settings, trained_ids)
    _require_metadata_coverage(metadata_df, trained_ids, population="trained")

    # Walk disjoint init_time chunks covering every run that can forecast into the window:
    # init_time in [val_start - MAX_NWP_LEAD, val_end].
    chunk_start = val_start - MAX_NWP_LEAD
    is_first = True
    while chunk_start <= val_end:
        chunk_end = min(chunk_start + _PREDICT_INIT_CHUNK, val_end)
        power_ts, nwp_lf = load_engineering_inputs(
            settings,
            time_series_ids=trained_ids,
            metadata=metadata_df,
            window_start=val_start,
            window_end=val_end,
            init_time_start=chunk_start,
            init_time_end=chunk_end,
        )
        features = forecaster.feature_engineer.engineer(
            selected_features=config.selected_features,
            power_time_series=power_ts,
            time_series_metadata=metadata_df,
            nwp=nwp_lf,
        )
        forecasts = forecaster.predict(features, fold_id=fold_id)
        # init_times fall on day boundaries, so the 1µs step never drops a run from the next chunk.
        chunk_start = chunk_end + timedelta(microseconds=1)

        # Skip empty chunks except the first, which must run to (over)write the partition so a
        # re-materialisation always replaces the fold's prior rows.
        if forecasts.height == 0 and not is_first:
            continue

        # The first chunk overwrites the (experiment, fold) partition, clearing any prior run;
        # later chunks append into the partition the first chunk established.
        write_power_forecasts(
            forecasts,
            settings.power_forecasts_data_path,
            replace_partition=(experiment_name, fold_id) if is_first else None,
            storage_options=settings.storage_options,
        )
        is_first = False
        n_rows += forecasts.height
        # `.unique()` before `.to_list()`: only the distinct values reach the set, so this builds
        # a ~30-element Python list per chunk rather than a ~14M-element one. Measured on a
        # 14M-row Int32 column: 2.78 s and 560 MB peak, against 0.05 s and no measurable
        # allocation. The 560 MB landed inside the loop whose whole job is holding the frame at
        # 2-3 GB.
        time_series_seen.update(forecasts["time_series_id"].unique().to_list())
        ensemble_members_seen.update(forecasts["ensemble_member"].unique().to_list())

    n_time_series = len(time_series_seen)
    n_ensemble_members = len(ensemble_members_seen)
    with mlflow.start_run(run_id=fold_run_id):
        # Use set_tag (not log_params) so re-materialising with an extended val_end doesn't
        # raise "Changing param values is not allowed" — the covered window is mutable metadata.
        mlflow.set_tag("val_start", val_start.isoformat())
        mlflow.set_tag("val_end", val_end.isoformat())
        # Provenance: prediction may run on yet another SHA / data state than training. Stamps the
        # two Delta tables the forecasting path reads (via load_engineering_inputs) — not
        # eligible_time_series, which prediction does not read (the trained series come from the
        # loaded model).
        mlflow.set_tags(
            provenance_tags(
                "predict",
                {
                    "power_time_series": settings.power_time_series_data_path,
                    "nwp_data": settings.nwp_data_path,
                },
                storage_options=settings.storage_options,
            )
        )
        # Tags, not metrics, for the same reason the training counters are tags (see
        # `trained_cv_model`): all three shrink when the trained population shrinks or the
        # validation window is narrowed, and MLflow resolves a metric's "latest" value as the max
        # over (step, timestamp, value) rather than the newest write — so a shrunk count would be
        # under-reported. Tags are last-write-wins, which is the semantic wanted here.
        mlflow.set_tags(
            {
                "n_forecast_rows": str(n_rows),
                "n_forecast_time_series": str(n_time_series),
                "n_ensemble_members": str(n_ensemble_members),
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


# ---------------------------------------------------------------------------
# metrics asset
# ---------------------------------------------------------------------------


class PopulationFilter(Config):
    """Typed filter over ``power_forecasts`` rows to score.

    Every field is tabulated with a worked example under "Step 9 — Materialise ``metrics``":
    <https://openclimatefix.github.io/nged-substation-forecast/ml_experimentation/dagster-workflow/>

    All fields default to ``None`` (= no filter on that dimension). A mistyped field name is a
    Dagster config validation error, not a silent wrong population — that is the whole point of
    using a typed config rather than a free ``dict``.
    """

    experiment_name: str | None = None
    fold_id: str | None = None
    valid_time_min: str | None = None
    """ISO-8601 UTC timestamp; rows with ``valid_time`` before this are excluded."""
    valid_time_max: str | None = None
    """ISO-8601 UTC timestamp; rows with ``valid_time`` after this are excluded."""

    def apply(self, scan: pt.LazyFrame[PowerForecast]) -> pt.LazyFrame[PowerForecast]:
        """Return ``scan`` with all non-``None`` filter fields applied as Polars predicates.

        ``experiment_name`` / ``fold_id`` are ``String`` in ``PowerForecast`` (matching how
        delta-rs stores the on-disk partition columns), so the ``pl.scan_delta`` result can be
        typed with ``set_model(PowerForecast)`` directly and these ``.filter()`` predicates push
        straight down into the Delta scan — partition pruning on ``experiment_name`` / ``fold_id``
        plus row-group skipping — with no dtype cast in the way.

        Args:
            scan: Typed lazy scan of the ``power_forecasts`` Delta table.

        Returns:
            The same scan with the filter predicates applied, re-wrapped as a
            ``pt.LazyFrame[PowerForecast]`` (``.filter()`` is typed as a plain ``pl.LazyFrame``).
        """
        # .filter() is typed as plain pl.LazyFrame; accumulate on one, re-wrap on return.
        lf: pl.LazyFrame = scan
        if self.experiment_name is not None:
            lf = lf.filter(pl.col("experiment_name") == self.experiment_name)
        if self.fold_id is not None:
            lf = lf.filter(pl.col("fold_id") == self.fold_id)
        if self.valid_time_min is not None:
            min_dt = datetime.fromisoformat(self.valid_time_min)
            if min_dt.tzinfo is None:
                min_dt = min_dt.replace(tzinfo=UTC)
            lf = lf.filter(pl.col("valid_time") >= min_dt)
        if self.valid_time_max is not None:
            max_dt = datetime.fromisoformat(self.valid_time_max)
            if max_dt.tzinfo is None:
                max_dt = max_dt.replace(tzinfo=UTC)
            lf = lf.filter(pl.col("valid_time") <= max_dt)
        return pt.LazyFrame.from_existing(lf).set_model(PowerForecast)


class MetricsConfig(Config):
    """Run config for the ``metrics`` asset.

    Filled in from the Dagster run-config dialog; see "Step 9 — Materialise ``metrics``":
    <https://openclimatefix.github.io/nged-substation-forecast/ml_experimentation/dagster-workflow/>
    """

    population_filter: PopulationFilter = PopulationFilter()
    evaluation_scope: EvalScopeType = "leaderboard"
    """Which evaluation scope this run produces.

    - ``"leaderboard"``: logs per-fold + aggregate metrics to the golden leaderboard MLflow
      experiments (canonical complete-window folds only).
    - ``"ad_hoc"``: writes to ``forecast_metrics`` Delta only; no MLflow logging.
    """


def _resolve_eval_window(
    evaluation_scope: EvalScopeType,
    fold_id: str,
    group_scan: pl.LazyFrame,
) -> tuple[datetime, datetime, str]:
    """Return ``(window_start, window_end, window_label)`` for a metrics group.

    For ``"leaderboard"`` scope the bounds come from the fold config; for ``"ad_hoc"``
    they are the observed ``valid_time`` extent of the forecast group.

    Args:
        evaluation_scope: ``"leaderboard"`` uses fold-config dates; ``"ad_hoc"`` uses the
            observed ``valid_time`` range of the forecast group.
        fold_id: Fold identifier; only used when ``evaluation_scope == "leaderboard"``.
        group_scan: Lazy scan of this ``(experiment_name, fold_id)`` group's forecast rows.
            Only a streaming min/max over ``valid_time`` is computed, and only for the
            ``"ad_hoc"`` branch — the group is never materialised here.

    Returns:
        ``(window_start, window_end, window_label)`` — the inclusive evaluation window bounds
        and a human-readable label (``fold_id`` for leaderboard; ``"ad_hoc"`` otherwise).
    """
    if evaluation_scope == "leaderboard":
        fold = _cv_config.get_fold(fold_id)
        return (
            date_to_utc_datetime(fold.val_start),
            date_to_utc_datetime(fold.val_end, end_of_day=True),
            fold_id,
        )
    bounds = group_scan.select(
        window_start=pl.col("valid_time").min(), window_end=pl.col("valid_time").max()
    ).collect(engine="streaming")
    window_start = bounds["window_start"][0]
    window_end = bounds["window_end"][0]
    # groups is derived from a non-empty forecast scan, so min/max are always non-null datetimes.
    assert isinstance(window_start, datetime)
    assert isinstance(window_end, datetime)
    return window_start, window_end, "ad_hoc"


def _write_metrics_to_delta(
    path: str,
    enriched: pt.DataFrame[Metrics],
    exp_name: str,
    fold_id: str,
    storage_options: ObjectStoreOptions | None = None,
) -> None:
    """Write enriched Metrics rows to the ``forecast_metrics`` Delta table.

    Casts the ``Enum`` columns (e.g. ``metric_name``, ``horizon_slice``) to ``String`` before
    writing — delta-rs stores Arrow dictionary arrays as plain String in Parquet, so the on-disk
    schema is always String. Re-sending Enum data on an overwrite would cause a schema-mismatch
    error. Performs an idempotent overwrite of the ``(experiment_name, fold_id)`` partition
    so re-materialising the asset replaces rows rather than duplicating them.

    Args:
        path: Local path or remote URI of the ``forecast_metrics`` Delta table.
        enriched: Fully populated ``Metrics`` rows, with all provenance columns set by
            ``enrich_metrics_rows()``.
        exp_name: Experiment name; used in the Delta overwrite predicate to scope the
            replacement to this ``(experiment_name, fold_id)`` partition.
        fold_id: Fold identifier; used alongside ``exp_name`` in the predicate.
        storage_options: Object-store options for a remote ``path``; ``None``/empty for local.
    """
    enum_cols = [c for c, dtype in enriched.schema.items() if isinstance(dtype, pl.Enum)]
    delta_data = enriched.with_columns(pl.col(c).cast(pl.String) for c in enum_cols).to_arrow()
    write_deltalake(
        table_or_uri=path,
        data=delta_data,
        mode="overwrite",
        predicate=f"experiment_name = '{exp_name}' AND fold_id = '{fold_id}'",
        partition_by=["experiment_name", "fold_id"],
        storage_options=typeddict_to_dict(storage_options),
    )


_METRICS_SERIES_BATCH_SIZE: Final[int] = 4
"""How many ``time_series_id`` values to materialise per scoring batch in the ``metrics`` asset.

A single leaderboard fold is far too big to collect whole: at 364M rows — the
``mid_2025_to_mid_2026`` fold as it stood at 28 series, and it has gained series since — the full
``PowerForecast`` schema OOM-kills a 29 GB machine. But ``compute_metrics`` is independent per
``time_series_id`` — every group key includes it — so scoring per-series batches and
concatenating the tall ``Metrics`` results is exactly equivalent to one big call. For the
batch-size measurements behind the value 4, see "Scoring the metrics: batch the series, and
stream every scan" in
<https://openclimatefix.github.io/nged-substation-forecast/architecture/performance/#scoring-the-metrics-batch-the-series-and-stream-every-scan>.
"""


def _group_scan(
    pruned_scan: pt.LazyFrame[PowerForecast], exp_name: str, fold_id: str
) -> pl.LazyFrame:
    """Narrow the already-pruned scan to a single ``(experiment_name, fold_id)`` partition.

    The ``.filter()`` on the String partition columns pushes down into the Delta scan, so
    downstream collects read only this partition's Parquet files.

    Args:
        pruned_scan: The ``pt.LazyFrame[PowerForecast]`` returned by ``PopulationFilter.apply``
            (partition predicates already applied).
        exp_name: Experiment name of the group.
        fold_id: Fold identifier of the group.

    Returns:
        A lazy scan of this one group's rows. Left as a plain ``pl.LazyFrame`` rather than
        re-wrapped as ``pt.LazyFrame[PowerForecast]``, because every consumer takes it plain and
        ``_load_series_batch`` validates each batch on collect — a re-wrap here would be discarded
        unused.
    """
    return pruned_scan.filter(
        (pl.col("experiment_name") == exp_name) & (pl.col("fold_id") == fold_id)
    )


def _series_ids_in_group(group_scan: pl.LazyFrame) -> list[int]:
    """Return the sorted ``time_series_id`` values present in one forecast group.

    A streaming single-column aggregation — cheap relative to the row data.
    """
    return sorted(
        group_scan.select("time_series_id").unique().collect(engine="streaming")["time_series_id"]
    )


def _load_series_batch(
    group_scan: pl.LazyFrame, series_ids: list[int]
) -> pt.DataFrame[PowerForecast]:
    """Collect and validate one batch of series from a single-group forecast scan.

    Args:
        group_scan: Lazy scan of one ``(experiment_name, fold_id)`` group's rows.
        series_ids: The ``time_series_id`` values to materialise.

    Returns:
        The validated ``PowerForecast`` rows for these series.
    """
    collected = group_scan.filter(pl.col("time_series_id").is_in(series_ids)).collect(
        engine="streaming"
    )
    return PowerForecast.validate(collected, allow_superfluous_columns=True)


def _score_forecast_group(
    exp_name: str,
    fold_id: str,
    group_scan: pl.LazyFrame,
    actuals_lf: pt.LazyFrame[PowerTimeSeries],
    metadata_df: pt.DataFrame[TimeSeriesMetadata],
    capacity_df: pt.DataFrame[EffectiveCapacity],
    evaluation_scope: EvalScopeType,
    metrics_path: str,
    now: datetime,
    storage_options: ObjectStoreOptions | None = None,
    provenance: MlflowTags | None = None,
) -> tuple[int, dict[str, float] | None]:
    """Score one ``(experiment_name, fold_id)`` group.

    Writes ``Metrics`` to Delta and optionally logs to MLflow.

    The group is scored in per-series batches of ``_METRICS_SERIES_BATCH_SIZE`` so that peak
    memory is one batch, never the whole fold (a single leaderboard fold is already too big to
    materialise). ``compute_metrics`` is independent per ``time_series_id``, so concatenating
    the per-batch ``Metrics`` frames produces exactly the metric values a whole-group call
    would. A batch whose series have no overlapping actuals is skipped — mirroring how such
    series silently vanish from the inner join in a whole-group call — but a group where *no*
    batch overlaps raises, exactly as the whole-group call would.

    One deliberate divergence from whole-group scoring: error messages from
    ``compute_metrics`` (missing capacity, negative lead times) describe only the first
    offending *batch* — at most ``_METRICS_SERIES_BATCH_SIZE`` series — rather than the
    whole group, since the raise aborts the loop before later batches load.

    Args:
        exp_name: Experiment name for this group.
        fold_id: Fold identifier for this group.
        group_scan: Lazy scan of this single ``(exp_name, fold_id)`` group's forecast rows
            (from ``_group_scan``); materialised one series batch at a time.
        actuals_lf: Lazy observed power scan (only the joined subset is collected inside
            ``compute_metrics()``).
        metadata_df: Substation metadata used to join ``time_series_type`` onto each metric row.
        capacity_df: Per-series effective capacity used as the NMAE denominator inside
            ``compute_metrics()``; must cover every scored series.
        evaluation_scope: ``"leaderboard"`` logs per-fold metrics to MLflow; ``"ad_hoc"``
            skips MLflow entirely.
        metrics_path: Local path or remote URI of the ``forecast_metrics`` Delta table.
        now: UTC timestamp stamped on every row as ``computed_at`` (injected so all rows in
            one asset materialisation share the same timestamp).
        storage_options: Object-store options for a remote ``metrics_path``; ``None``/empty
            for local.
        provenance: Stage-prefixed provenance tags (git SHA + scored Delta-table versions) to
            stamp on the fold's MLflow run in ``"leaderboard"`` scope; built once by the caller
            so every group shares one snapshot. ``None`` skips the stamp.

    Returns:
        A ``(n_rows_written, fold_metric_dict)`` tuple where:

        - ``n_rows_written`` — number of ``Metrics`` rows written to Delta for this group.
        - ``fold_metric_dict`` — flat ``{mlflow_metric_key: mean_value}`` dict logged to the
          fold's MLflow child run (e.g. ``{"rmse__all": 0.42, "rmse__pv": 0.31}``). ``None``
          for ``"ad_hoc"`` scope, where no MLflow run exists.

    Raises:
        NoOverlappingActualsError: If no series in the group has any overlapping actuals.
    """
    series_ids = _series_ids_in_group(group_scan)
    batch_metrics: list[pt.DataFrame[Metrics]] = []
    for start in range(0, len(series_ids), _METRICS_SERIES_BATCH_SIZE):
        batch_ids = series_ids[start : start + _METRICS_SERIES_BATCH_SIZE]
        batch_forecasts = _load_series_batch(group_scan, batch_ids)
        try:
            batch_metrics.append(
                compute_metrics(batch_forecasts, actuals_lf, metadata_df, capacity_df)
            )
        except NoOverlappingActualsError:
            continue
    if not batch_metrics:
        raise NoOverlappingActualsError(
            f"No rows in the joined forecast/actuals data for group ({exp_name}, {fold_id}). "
            "Check that cv_forecasts and actuals overlap in time."
        )
    per_series_metrics = Metrics.validate(pl.concat(batch_metrics), allow_superfluous_columns=True)

    window_start, window_end, window_label = _resolve_eval_window(
        evaluation_scope, fold_id, group_scan
    )
    mlflow_run_id: str | None = None
    if evaluation_scope == "leaderboard":
        experiment_id = get_or_create_experiment(exp_name)
        parent_run_id = get_or_create_parent_run(experiment_id)
        mlflow_run_id = get_or_create_fold_run(experiment_id, parent_run_id, fold_id)

    enriched = enrich_metrics_rows(
        per_series_metrics,
        exp_name,
        evaluation_scope,
        window_start,
        window_end,
        window_label,
        now,
        mlflow_run_id,
    )
    _write_metrics_to_delta(metrics_path, enriched, exp_name, fold_id, storage_options)

    if evaluation_scope == "leaderboard":
        fold_metric_dict = build_mlflow_aggregate_metrics(per_series_metrics)
        with mlflow.start_run(run_id=mlflow_run_id):
            if provenance is not None:
                mlflow.set_tags(provenance)
            mlflow.log_metrics(fold_metric_dict)
        return enriched.height, fold_metric_dict

    return enriched.height, None


@asset(tags=RESEARCH_LAYER_TAGS, deps=["cv_power_forecasts", "effective_capacity"])
def metrics(context: AssetExecutionContext, config: MetricsConfig) -> None:
    """Compute evaluation metrics and write to ``forecast_metrics``.

    Reads the filtered ``power_forecasts`` Delta, joins observed power, computes the
    deterministic and probabilistic metrics per series and horizon slice (via
    ``compute_metrics()`` — see
    <https://openclimatefix.github.io/nged-substation-forecast/techniques/evaluation-metrics/>),
    enriches each row with scope and evaluation-window provenance, and writes to the
    ``forecast_metrics`` Delta table partitioned by ``(experiment_name, fold_id)``.

    For ``evaluation_scope="leaderboard"``, also logs per-type + overall aggregate metrics to each
    fold's MLflow child run and the mean-across-folds aggregates to the experiment's parent run.
    Lookup is by tag — never by a handle passed between assets — so this is idempotent under
    Dagster retries and safe across processes; see "Cross-process run resolution" in
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/ml-orchestration/>.

    ``power_forecasts`` also holds the live service's output under ``fold_id="live"`` and any
    non-leaderboard dev fold such as ``smoke_test``. Leaderboard scope skips any ``fold_id`` that
    is not a leaderboard fold in the CV config, naming them in a warning and in the
    ``skipped_fold_ids`` output metadata, because it dates its evaluation window from that config
    and has none for them. Use ``evaluation_scope="ad_hoc"`` to score live or dev-fold rows: it
    takes the window from the rows themselves.

    Args:
        context: Dagster execution context; used for logging and ``add_output_metadata``.
        config: Population filter and evaluation scope for this materialisation. Defaults to
            no filter and ``"leaderboard"`` scope.
    """
    settings = Settings()
    storage_options = settings.storage_options
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    # Apply the population filter to the scan so its predicates push into the Delta scan
    # (experiment_name / fold_id are the on-disk partition columns → partition pruning). These
    # columns are String in PowerForecast, matching how delta-rs stores them, so no dtype cast
    # sits between scan_delta and the filter to defeat pushdown. See PopulationFilter.apply.
    scan = pt.LazyFrame.from_existing(
        pl.scan_delta(
            settings.power_forecasts_data_path, storage_options=typeddict_to_dict(storage_options)
        )
    ).set_model(PowerForecast)
    pruned_scan = config.population_filter.apply(scan)

    # Discover the matching groups from the pruned scan. The streaming engine is essential here
    # even though only the two partition columns are projected — the eager equivalent still
    # materialises every row before the unique. See "Scoring the metrics: batch the series, and
    # stream every scan":
    # <https://openclimatefix.github.io/nged-substation-forecast/architecture/performance/#scoring-the-metrics-batch-the-series-and-stream-every-scan>.
    # Each group is then scored in per-series batches, so peak memory is one batch, never a whole
    # fold.
    groups = (
        pruned_scan.select(["experiment_name", "fold_id"])
        .unique()
        .sort(["experiment_name", "fold_id"])
        .collect(engine="streaming")
        .rows()
    )
    # `live_forecasts` writes its output to this same table under `fold_id="live"`, so an
    # unfiltered leaderboard run — which the operator guide tells you to launch, leaving
    # `fold_id` null to score every fold — discovers a group the CV config has never heard of.
    # The same table also holds any non-leaderboard dev fold (e.g. `smoke_test`), which the CV
    # config does define but which is not part of the leaderboard's evaluation protocol. Either
    # way, leaderboard scope dates its window from the config's leaderboard folds, so those rows
    # have no window to be scored against; skip them rather than fail the whole run on the first
    # one. Ad-hoc scope takes its window from the rows themselves and is the supported way to
    # score live output or dev folds.
    skipped_fold_ids: list[str] = []
    if config.evaluation_scope == "leaderboard":
        configured_fold_ids = set(_cv_config.leaderboard_fold_ids)
        skipped_fold_ids = sorted({fold for _, fold in groups if fold not in configured_fold_ids})
        if skipped_fold_ids:
            context.log.warning(
                f"Skipping {skipped_fold_ids} — not leaderboard folds in the CV config "
                f"({sorted(configured_fold_ids)}), so leaderboard scope has no evaluation window "
                'for them. Score these with evaluation_scope="ad_hoc", which dates its window '
                "from the forecast rows themselves."
            )
            groups = [group for group in groups if group[1] in configured_fold_ids]

    if not groups:
        context.log.warning("No forecasts matched the population filter — nothing to score.")
        context.add_output_metadata(
            {"n_rows_written": 0, "n_groups": 0, "skipped_fold_ids": str(skipped_fold_ids)}
        )
        return

    actuals_lf = pt.LazyFrame.from_existing(
        pl.scan_delta(
            settings.power_time_series_data_path, storage_options=typeddict_to_dict(storage_options)
        )
    ).set_model(PowerTimeSeries)
    # allow_superfluous_columns because the parquet also carries h3_res_5 and other geo columns.
    metadata_df = TimeSeriesMetadata.validate(
        pl.read_parquet(settings.metadata_path, storage_options=typeddict_to_dict(storage_options)),
        allow_superfluous_columns=True,
    )

    # The full-history effective capacity is the NMAE denominator (a declared dep of this asset).
    if not delta_table_exists(settings.effective_capacity_data_path, storage_options):
        raise FileNotFoundError(
            f"effective_capacity Delta not found at {settings.effective_capacity_data_path}; "
            "materialise the effective_capacity asset before running metrics."
        )
    capacity_df = EffectiveCapacity.validate(
        pl.read_delta(
            settings.effective_capacity_data_path,
            storage_options=typeddict_to_dict(storage_options),
        )
    )

    if_local_path_then_make_parent_dir(settings.forecast_metrics_data_path)
    now = datetime.now(UTC)
    # Provenance for every fold + parent run this materialisation touches: the code SHA and the
    # versions of the three Delta tables scoring reads (forecasts, actuals, capacity). Built once
    # — all groups are scored in this one process, so they share a single code + data snapshot.
    metrics_provenance = provenance_tags(
        "metrics",
        {
            "power_forecasts": settings.power_forecasts_data_path,
            "power_time_series": settings.power_time_series_data_path,
            "effective_capacity": settings.effective_capacity_data_path,
        },
        storage_options=storage_options,
    )
    total_rows = 0
    # Accumulates per-fold metric values for parent-run aggregation (leaderboard scope only).
    # Structure: {experiment_name: {mlflow_metric_key: [value_per_fold, ...]}}
    # e.g. {"xgboost_baseline": {"rmse__all": [0.42, 0.39], "rmse__pv": [0.31, 0.28]}}
    # After the loop, each list is averaged and logged to the experiment's MLflow parent run.
    experiment_fold_metrics: dict[str, dict[str, list[float]]] = {}

    for exp_name, fold_id in groups:
        n_rows, fold_metric_dict = _score_forecast_group(
            exp_name,
            fold_id,
            _group_scan(pruned_scan, exp_name, fold_id),
            actuals_lf,
            metadata_df,
            capacity_df,
            config.evaluation_scope,
            settings.forecast_metrics_data_path,
            now,
            storage_options,
            provenance=metrics_provenance,
        )
        total_rows += n_rows
        if fold_metric_dict is not None:
            exp_metrics = experiment_fold_metrics.setdefault(exp_name, {})
            for key, value in fold_metric_dict.items():
                exp_metrics.setdefault(key, []).append(value)

    if config.evaluation_scope == "leaderboard":
        for exp_name, fold_metrics in experiment_fold_metrics.items():
            experiment_id = get_or_create_experiment(exp_name)
            parent_run_id = get_or_create_parent_run(experiment_id)
            parent_metric_dict = {k: sum(v) / len(v) for k, v in fold_metrics.items()}
            with mlflow.start_run(run_id=parent_run_id):
                mlflow.set_tags(metrics_provenance)
                mlflow.log_metrics(parent_metric_dict)

    context.add_output_metadata(
        {
            "n_rows_written": total_rows,
            "n_groups": len(groups),
            "evaluation_scope": config.evaluation_scope,
            "groups": str(groups),
            "skipped_fold_ids": str(skipped_fold_ids),
        }
    )
