"""Cross-validation Dagster assets.

These assets implement the experiment-independent, fold-partitioned CV layer. Each asset is a
thin orchestration shell delegating its logic to the pure helpers in ``ml_core.cv_helpers`` so
the logic stays fast to unit-test and the assets stay readable.
"""

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, NamedTuple, Self

import mlflow
import patito as pt
import polars as pl
from contracts.config_schemas import STUDY_EXPERIMENT_PREFIX, load_cv_config
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
from delta_store.cleaned_power_time_series import read_cleaning_provenance
from delta_store.effective_capacity import write_effective_capacity
from delta_store.eligible_time_series import write_eligible_time_series
from delta_store.forecast_metrics import write_forecast_metrics
from delta_store.power_forecasts import write_power_forecasts
from ml_core.cv_helpers import (
    date_to_utc_datetime,
    eligible_time_series_ids,
    parse_cv_partition_key,
)
from ml_core.features._parsed_features import ParsedFeatures
from ml_core.metrics import (
    NoOverlappingActualsError,
    build_mlflow_aggregate_metrics,
    compute_effective_capacity,
    compute_metrics,
    enrich_metrics_rows,
    require_same_row_keys,
    require_single_model_name,
    require_valid_times_within_window,
    require_window_within_guard,
    row_key_fingerprint,
)
from ml_core.mlflow_runs import (
    get_or_create_experiment,
    get_or_create_fold_run,
    get_or_create_parent_run,
    load_experiment_forecaster,
)
from ml_core.repro import ABSENT, MlflowTags, StageType, TableNameType, provenance_tags
from mlflow.tracking import MlflowClient
from nged_data.storage import coverage_from_power, scan_cleaned_power, time_series_coverage
from pydantic import model_validator

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
    deps=["clean_nged_power_data"],
)
def eligible_time_series(context: AssetExecutionContext) -> None:
    """Compute and persist the canonical eligible ``time_series_id``s for one CV fold.

    Reads observed-power coverage from the unflagged rows of the ``cleaned_power_time_series`` Delta
    table, which ``clean_nged_power_data`` writes. An absent table gives an empty population. A time
    series is eligible for a fold when its coverage has at least ``min_training_months`` of history
    before the fold's ``val_start`` *and* reaches the fold's ``val_end``. Eligibility is derived
    from data coverage alone (not from any model/config), so every experiment evaluates the fold on
    the identical population — this is what keeps the leaderboard apples-to-apples.
    See "Eligibility" in
    <https://openclimatefix.github.io/nged-substation-forecast/ml_experimentation/cross-validation-folds/#eligibility>
    for the fold definitions this population is computed against.

    The result is written to the ``eligible_time_series`` Delta table as one partition per
    ``fold_id`` via an idempotent partition overwrite, so re-materialising a fold replaces its
    rows rather than duplicating them. ``trained_cv_model`` reads this fold's partition to select
    which series to train on; ``cv_power_forecasts`` does not read it directly, but inherits the
    same population through the trained model's ``trained_time_series_ids``. A stale or missing
    partition here therefore shows up downstream as a fold trained on the wrong population, not
    as a failure at either of those assets.
    """
    settings = Settings()
    storage_options = settings.storage_options
    fold_id = context.partition_key
    fold = _cv_config.get_fold(fold_id)

    cleaned_path = settings.cleaned_power_time_series_data_path
    if delta_table_exists(cleaned_path, storage_options):
        coverage = coverage_from_power(scan_cleaned_power(cleaned_path, storage_options))
    else:
        coverage = time_series_coverage(cleaned_path, storage_options)  # an empty frame

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
    write_eligible_time_series(
        eligible=eligible_df,
        table_uri=settings.eligible_time_series_data_path,
        fold_id=fold_id,
        storage_options=storage_options,
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


@asset(tags=RESEARCH_LAYER_TAGS, deps=["clean_nged_power_data"])
def effective_capacity(context: AssetExecutionContext) -> None:
    """Compute and persist each series' v0.1 effective capacity (full-history P99 of ``|power|``).

    Reads the unflagged rows of the full ``cleaned_power_time_series`` Delta and writes one row per
    ``time_series_id`` to the ``effective_capacity`` Delta table
    (``Settings.effective_capacity_data_path``): the 99th percentile of ``abs(power)`` over the
    series' entire observed history, with ``time`` set to the latest observed timestep. This
    full-history capacity is the NMAE denominator used by the ``metrics`` asset, replacing the
    validation-window P99 that would otherwise vary fold to fold.

    The whole (small — one row per series) table is overwritten on each materialisation. v0.1 is
    deliberately one scalar row per series, **not** the value repeated at every half-hour —
    densifying a constant buys nothing.

    A future upgrade (v0.7) swaps the P99 for a time-varying capacity estimator, emitting one row
    per ``(time_series_id, time)``; the ``EffectiveCapacity`` schema is unchanged, but
    ``compute_metrics`` then joins capacity as a temporal as-of join rather than on
    ``time_series_id`` alone (same doc section).
    """
    settings = Settings()
    storage_options = settings.storage_options
    power_lf = scan_cleaned_power(settings.cleaned_power_time_series_data_path, storage_options)
    capacity_df = compute_effective_capacity(power_lf)

    if_local_path_then_make_parent_dir(settings.effective_capacity_data_path)
    write_effective_capacity(
        capacity=capacity_df,
        table_uri=settings.effective_capacity_data_path,
        storage_options=storage_options,
    )

    context.add_output_metadata(
        {
            "n_time_series": capacity_df.height,
            "effective_capacity_data_path": str(settings.effective_capacity_data_path),
        }
    )


def _provenance_tags_with_cleaned_power(
    stage: StageType, settings: Settings, delta_paths: dict[TableNameType, str]
) -> MlflowTags:
    """``provenance_tags`` for ``stage``, plus the cleaned power table's provenance.

    The extra tag, ``{stage}_cleaned_power_time_series_source``, names the raw power table and
    version the cleaning read and the git SHA of the cleaning code, or ``ABSENT``. The
    ``ml_core.repro`` module docstring explains why the cleaned table's own Delta version is not
    stamped.

    Args:
        stage: The stage prefix passed to ``provenance_tags``.
        settings: Supplies the table paths and the object-store options.
        delta_paths: The other Delta tables this stage reads.

    Returns:
        The stage-prefixed provenance tags.
    """
    tags = provenance_tags(stage, delta_paths, storage_options=settings.storage_options)
    provenance = read_cleaning_provenance(
        settings.cleaned_power_time_series_data_path, settings.storage_options
    )
    source_tag = f"{stage}_cleaned_power_time_series_source"
    if provenance is None:
        tags[source_tag] = ABSENT
    else:
        tags[source_tag] = (
            f"raw_table_id={provenance.raw_table_id};raw_version={provenance.raw_version};"
            f"git_sha={provenance.git_sha}"
        )
    return tags


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


def _load_time_series_metadata(
    settings: Settings, time_series_ids: list[int]
) -> pt.DataFrame[TimeSeriesMetadata]:
    """Read the ``TimeSeriesMetadata`` table, filtered to ``time_series_ids``.

    **Research callers only.** The metadata table is the live registry of what NGED operates, so a
    fault in it must stop a training or scoring run rather than silently shrink its population. A
    fault here means an off-contract file, or a metadata table rebuilt from a snapshot that dropped
    rows. ``live_forecasts`` reads ``ml_core.base_forecaster.load_trained_metadata`` instead.

    Args:
        settings: Application settings (data paths, credentials).
        time_series_ids: The population to keep.

    Returns:
        One row per series in ``time_series_ids`` that the metadata table covers — check the
        coverage with ``_require_metadata_coverage``.
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
    deps=["clean_nged_power_data", "ecmwf_ens", "eligible_time_series"],
)
def trained_cv_model(context: AssetExecutionContext) -> None:
    """Train one forecaster for a single ``(experiment, fold)`` partition and save it to MLflow.

    Reads the experiment's resolved config from MLflow (the immutable record registered by
    ``register_experiment_job``), the fold's canonical eligible ``time_series_id`` population
    from the ``eligible_time_series`` asset, and the observed power + gridded NWP over the fold's
    **inclusive** training window. Features are engineered through the forecaster's own
    ``FeatureEngineer`` (so the spatial NWP mapping and feature pipeline are a model concern),
    the model is trained, and its artifacts are uploaded to the fold's MLflow run alongside a
    record of the training window and population.

    The fold run is resolved **by tag**, never by a handle passed between assets, so this is safe
    across processes and idempotent under Dagster retries — see "Cross-process run resolution:
    discover by tag, never pass handles" in
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/ml-orchestration/#cross-process-run-resolution-discover-by-tag-never-pass-handles>.
    Because that run is *reused* on every re-materialisation, the training window and population
    go in tags rather than MLflow params (which are write-once and would reject a changed value).
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

    metadata_df = _load_time_series_metadata(settings, eligible_ids)
    _require_metadata_coverage(metadata_df, eligible_ids, population="eligible")
    power_lookback = ParsedFeatures.from_strings(config.selected_features).max_power_lag()
    power_ts, nwp_lf = load_engineering_inputs(
        settings,
        time_series_ids=eligible_ids,
        metadata=metadata_df,
        window_start=train_start,
        window_end=train_end,
        ensemble_members=[0],
        power_lookback=power_lookback,
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
        # params) because provenance overwrites cleanly on re-materialise; these are the Delta
        # tables the training path reads above.
        mlflow.set_tags(
            _provenance_tags_with_cleaned_power(
                "train",
                settings,
                {
                    "nwp_data": settings.nwp_data_path,
                    "eligible_time_series": settings.eligible_time_series_data_path,
                },
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
    fold's **inclusive** validation window across **all** NWP ensemble members — the
    probabilistic leaderboard metrics are meaningless on a single member. The scored population
    is the model's own ``trained_time_series_ids`` (the train==predict invariant), so a fold is
    always scored on exactly the population it was trained on even if power coverage has drifted
    since training.

    To keep RAM bounded, prediction runs **one ``init_time`` window at a time**
    (``_PREDICT_INIT_CHUNK``). The full validation window fans every NWP run out across all ~51
    ensemble members and all trained series — tens of GB. ``init_time`` is one of the NWP table's
    two partition columns *and* the axis that inflates the output, so chunking by it bounds the
    per-iteration forecast frame (~2-3 GB) while each partition is still read exactly once. See
    "Bounding feature-engineering memory: prune the inputs, not the output" in
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/performance/#bounding-feature-engineering-memory-prune-the-inputs-not-the-output>.

    Forecasts are written to the ``power_forecasts`` Delta table keyed by ``(experiment_name,
    fold_id)``: the **first** chunk overwrites the partition (clearing any prior run) and the
    rest **append** to it, so a full re-materialisation replaces the fold's rows without ever
    holding all forecasts in memory. Chunks are written through
    ``delta_store.power_forecasts.write_power_forecasts``, which owns the table's compressed
    storage format (sort order, ``power_fcst`` precision rounding, parquet encodings). The fold's
    MLflow run is resolved **by tag** — never by a handle from ``trained_cv_model`` — so this is
    safe across processes.
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
    metadata_df = _load_time_series_metadata(settings, trained_ids)
    _require_metadata_coverage(metadata_df, trained_ids, population="trained")

    power_lookback = ParsedFeatures.from_strings(config.selected_features).max_power_lag()

    # Walk disjoint init_time chunks covering every run that can forecast into the window:
    # init_time in [val_start - MAX_NWP_LEAD, val_end].
    chunk_start = val_start - MAX_NWP_LEAD
    is_first = True
    while chunk_start <= val_end:
        chunk_end = min(chunk_start + _PREDICT_INIT_CHUNK, val_end)
        # The power scan is lazy and re-issued per init_time chunk; it is small next to the NWP
        # scan, whose partition-pruned chunking is what this loop exists for.
        power_ts, nwp_lf = load_engineering_inputs(
            settings,
            time_series_ids=trained_ids,
            metadata=metadata_df,
            window_start=val_start,
            window_end=val_end,
            init_time_start=chunk_start,
            init_time_end=chunk_end,
            power_lookback=power_lookback,
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
            _provenance_tags_with_cleaned_power(
                "predict", settings, {"nwp_data": settings.nwp_data_path}
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
      experiments (canonical complete-window folds only). The population filter may not
      set ``valid_time_min`` or ``valid_time_max`` in this scope, because a trimmed window would be
      scored under the full fold's label.
    - ``"ad_hoc"``: writes to ``forecast_metrics`` Delta only; no MLflow logging.
    """

    @model_validator(mode="after")
    def _refuse_trimmed_leaderboard_window(self) -> Self:
        """Reject a ``valid_time`` bound on the population filter in leaderboard scope."""
        trimmed = (
            self.population_filter.valid_time_min is not None
            or self.population_filter.valid_time_max is not None
        )
        if self.evaluation_scope == "leaderboard" and trimmed:
            raise ValueError(
                "population_filter.valid_time_min/max trims the window, so it is only allowed "
                'with evaluation_scope="ad_hoc".'
            )
        return self


FINAL_TEST_ENV_VAR: Final[str] = "NGED_FINAL_TEST"
"""The environment variable that lets ``metrics`` score a window reaching ``final_test_start``.

Only the maintainer's own shell sets it to ``"1"``; an experiment or a study never does.
"""


class _EvalWindow(NamedTuple):
    """The inclusive evaluation window of one metrics group, and the label stamped on its rows."""

    start: datetime
    end: datetime
    label: str


def _final_test_enabled() -> bool:
    """Return whether the maintainer's environment allows scoring past ``final_test_start``."""
    return os.environ.get(FINAL_TEST_ENV_VAR) == "1"


def _valid_time_extent(group_scan: pl.LazyFrame) -> tuple[datetime, datetime]:
    """Return the earliest and latest ``valid_time`` of a forecast group, by a streaming scan."""
    bounds = group_scan.select(
        valid_time_min=pl.col("valid_time").min(), valid_time_max=pl.col("valid_time").max()
    ).collect(engine="streaming")
    valid_time_min = bounds["valid_time_min"][0]
    valid_time_max = bounds["valid_time_max"][0]
    # groups is derived from a non-empty forecast scan, so min/max are always non-null datetimes.
    assert isinstance(valid_time_min, datetime)
    assert isinstance(valid_time_max, datetime)
    return valid_time_min, valid_time_max


def _resolve_eval_window(
    evaluation_scope: EvalScopeType,
    fold_id: str,
    group_scan: pl.LazyFrame,
) -> _EvalWindow:
    """Return the evaluation window of a metrics group.

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
        The inclusive window bounds and a human-readable label (``fold_id`` for leaderboard;
        ``"ad_hoc"`` otherwise).
    """
    if evaluation_scope == "leaderboard":
        fold = _cv_config.get_fold(fold_id)
        return _EvalWindow(
            start=date_to_utc_datetime(fold.val_start),
            end=date_to_utc_datetime(fold.val_end, end_of_day=True),
            label=fold_id,
        )
    valid_time_min, valid_time_max = _valid_time_extent(group_scan)
    return _EvalWindow(start=valid_time_min, end=valid_time_max, label="ad_hoc")


def _validate_group(
    *,
    exp_name: str,
    fold_id: str,
    group_scan: pl.LazyFrame,
    scan: pt.LazyFrame[PowerForecast],
    evaluation_scope: EvalScopeType,
) -> _EvalWindow:
    """Refuse a group the scorer must not score, and return its evaluation window.

    The ``metrics`` asset calls this function for every group before scoring any group, so a
    refusal on a later group never leaves
    an earlier group's ``forecast_metrics`` rows or MLflow runs behind.

    Four checks, each raising. The final-test date guard applies to every scope. In leaderboard
    scope every row's ``valid_time`` must also lie inside the fold's window. A ``study/``
    experiment in leaderboard scope must additionally carry exactly the reference experiment's row
    keys for the fold, so it cannot abstain on hard rows, and a single ``power_fcst_model_name``,
    so it cannot down-weight them by spreading rows across model names.

    Args:
        exp_name: Experiment name of the group.
        fold_id: Fold identifier of the group.
        group_scan: Lazy scan of this group's forecast rows.
        scan: Typed lazy scan of the whole ``power_forecasts`` table, from which the reference
            experiment's rows are read.
        evaluation_scope: ``"leaderboard"`` or ``"ad_hoc"``.

    Returns:
        The group's evaluation window.
    """
    window = _resolve_eval_window(evaluation_scope, fold_id, group_scan)
    require_window_within_guard(
        window_end=window.end,
        final_test_start=_cv_config.final_test_start,
        fold_id=fold_id,
        final_test_enabled=_final_test_enabled(),
    )
    if evaluation_scope != "leaderboard":
        return window
    valid_time_min, valid_time_max = _valid_time_extent(group_scan)
    group_label = f"{exp_name}, {fold_id}"
    require_valid_times_within_window(
        valid_time_min=valid_time_min,
        valid_time_max=valid_time_max,
        window_start=window.start,
        window_end=window.end,
        group_label=group_label,
    )
    if exp_name.startswith(STUDY_EXPERIMENT_PREFIX):
        require_single_model_name(study=group_scan, group_label=group_label)
        reference = PopulationFilter(
            experiment_name=_cv_config.reference_experiment_name, fold_id=fold_id
        ).apply(scan)
        require_same_row_keys(
            study=group_scan,
            reference=reference,
            group_label=group_label,
            reference_label=_cv_config.reference_experiment_name,
            series_batch_size=_METRICS_SERIES_BATCH_SIZE,
        )
    return window


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


FINGERPRINT_TAG: Final[str] = "row_key_fingerprint"
"""The MLflow fold-run tag holding ``row_key_fingerprint`` of the group's forecast rows."""

REFERENCE_FINGERPRINT_TAG: Final[str] = "reference_row_key_fingerprint"
"""The MLflow fold-run tag holding the reference experiment's fingerprint for the same fold."""

MATCHES_REFERENCE_TAG: Final[str] = "row_keys_match_reference"
"""``"true"`` when the group and the reference experiment forecast the same series, initialisation
times, and valid times, so their scores rest on the same forecast problem; otherwise ``"false"``.
A reviewed experiment that differs from the reference is scored and tagged, never refused."""

STALE_TAG: Final[str] = "stale_against_reference"
"""``"true"`` on a study's fold run once the reference experiment's series, initialisation times, or
valid times have changed since the study was scored. A change in the reference's ensemble members
alone is not detected, because ``row_key_fingerprint`` ignores ``ensemble_member``. An unfiltered
``metrics`` run sets the tag on every study it skips."""

PAIRED_METRIC_PREFIX: Final[str] = "vs_reference__"
"""Prefix of the metric keys that hold a study's score minus the reference's, on a study fold run.
The reference is scored in the same ``metrics`` run, so both scores share one snapshot of the
observed power and of ``effective_capacity``."""


def _reference_fingerprint(scan: pt.LazyFrame[PowerForecast], fold_id: str) -> str:
    """Return the row-key fingerprint of the reference experiment's rows for ``fold_id``."""
    reference = PopulationFilter(
        experiment_name=_cv_config.reference_experiment_name, fold_id=fold_id
    ).apply(scan)
    return row_key_fingerprint(forecasts=reference, series_batch_size=_METRICS_SERIES_BATCH_SIZE)


def _fingerprint_tags(*, group_scan: pl.LazyFrame, reference_fingerprint: str) -> MlflowTags:
    """Return the fold-run tags that say which forecast problem a group scored."""
    fingerprint = row_key_fingerprint(
        forecasts=group_scan, series_batch_size=_METRICS_SERIES_BATCH_SIZE
    )
    return {
        FINGERPRINT_TAG: fingerprint,
        REFERENCE_FINGERPRINT_TAG: reference_fingerprint,
        MATCHES_REFERENCE_TAG: str(fingerprint == reference_fingerprint).lower(),
    }


def _log_paired_difference(
    *,
    exp_name: str,
    fold_id: str,
    study_metrics: dict[str, float],
    reference_metrics: dict[str, float],
) -> None:
    """Log a study's score minus the reference's, key by key, on the study's fold run."""
    experiment_id = get_or_create_experiment(exp_name)
    run_id = get_or_create_fold_run(experiment_id, get_or_create_parent_run(experiment_id), fold_id)
    differences = {
        f"{PAIRED_METRIC_PREFIX}{key}": value - reference_metrics[key]
        for key, value in study_metrics.items()
        if key in reference_metrics
    }
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(differences)


def _tag_stale_studies(
    *, skipped_groups: list[tuple[str, str]], scan: pt.LazyFrame[PowerForecast]
) -> None:
    """Tag each skipped study's fold run stale or current against the reference's row keys.

    A study's fold run holds the study's own fingerprint, which equalled the reference's when the
    study was scored, because a study is scored only when its row keys match the reference's. The
    study is stale when the reference's fingerprint for the fold has since changed. A group with no
    fold run was never scored and is left alone.
    """
    client = MlflowClient()
    reference_fingerprints: dict[str, str] = {}
    for exp_name, fold_id in skipped_groups:
        experiment = mlflow.get_experiment_by_name(exp_name)
        if experiment is None:
            continue
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.cv_role = 'fold' and tags.fold_id = '{fold_id}'",
            max_results=1,
        )
        if not runs:
            continue
        if fold_id not in reference_fingerprints:
            reference_fingerprints[fold_id] = _reference_fingerprint(scan, fold_id)
        scored_against = runs[0].data.tags.get(FINGERPRINT_TAG)
        stale = scored_against != reference_fingerprints[fold_id]
        client.set_tag(runs[0].info.run_id, STALE_TAG, str(stale).lower())


def _group_has_rows(scan: pt.LazyFrame[PowerForecast], exp_name: str, fold_id: str) -> bool:
    """Return whether ``power_forecasts`` holds any row for the group."""
    return _group_scan(scan, exp_name, fold_id).head(1).collect(engine="streaming").height > 0


def _split_off_studies(
    groups: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split ``groups`` into the reviewed groups and the ``study/`` groups, in that order."""
    studies = [group for group in groups if group[0].startswith(STUDY_EXPERIMENT_PREFIX)]
    return [group for group in groups if group not in studies], studies


def _with_reference_groups(
    *,
    groups: list[tuple[str, str]],
    scan: pt.LazyFrame[PowerForecast],
    evaluation_scope: EvalScopeType,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Add the reference experiment's group for every study fold, and put studies last.

    A study is scored beside the reference experiment in the same run, so the two scores share one
    snapshot of the observed power and of ``effective_capacity``. The population filter names only
    the study, so the added group is read from the whole table. A reference with no rows for the
    fold is not added, because the study's own check then refuses it by name.

    Args:
        groups: The ``(experiment_name, fold_id)`` groups the population filter matched.
        scan: Typed lazy scan of the whole ``power_forecasts`` table.
        evaluation_scope: Reference groups are added in ``"leaderboard"`` scope only.

    Returns:
        The groups to score, reviewed groups first and studies last, and the reference groups that
        were added.
    """
    if evaluation_scope != "leaderboard":
        return groups, []
    wanted = {
        (_cv_config.reference_experiment_name, fold_id)
        for exp_name, fold_id in groups
        if exp_name.startswith(STUDY_EXPERIMENT_PREFIX)
    } - set(groups)
    added = sorted(group for group in wanted if _group_has_rows(scan, *group))
    ordered = sorted(
        [*groups, *added], key=lambda group: (group[0].startswith(STUDY_EXPERIMENT_PREFIX), group)
    )
    return ordered, added


def _scoring_provenance(
    *,
    metrics_provenance: MlflowTags,
    scan: pt.LazyFrame[PowerForecast],
    group_scan: pl.LazyFrame,
    exp_name: str,
    fold_id: str,
    evaluation_scope: EvalScopeType,
    reference_fingerprints: dict[str, str],
) -> MlflowTags:
    """Return the tags for one group's fold run: the run's provenance and its row-key fingerprints.

    ``reference_fingerprints`` caches the reference's fingerprint per fold, so the reference is
    hashed once per fold in a run.
    """
    if evaluation_scope != "leaderboard":
        return metrics_provenance
    if fold_id not in reference_fingerprints:
        reference_fingerprints[fold_id] = _reference_fingerprint(scan, fold_id)
    tags = {
        **metrics_provenance,
        **_fingerprint_tags(
            group_scan=group_scan, reference_fingerprint=reference_fingerprints[fold_id]
        ),
    }
    if exp_name.startswith(STUDY_EXPERIMENT_PREFIX):
        tags[STALE_TAG] = "false"
    return tags


def _compare_with_reference(
    *,
    exp_name: str,
    fold_id: str,
    fold_metrics_by_group: dict[tuple[str, str], dict[str, float]],
) -> None:
    """Log a just-scored study's score minus the reference's, when the reference was also scored."""
    reference_metrics = fold_metrics_by_group.get((_cv_config.reference_experiment_name, fold_id))
    if exp_name.startswith(STUDY_EXPERIMENT_PREFIX) and reference_metrics is not None:
        _log_paired_difference(
            exp_name=exp_name,
            fold_id=fold_id,
            study_metrics=fold_metrics_by_group[(exp_name, fold_id)],
            reference_metrics=reference_metrics,
        )


def _read_scoring_inputs(
    settings: Settings,
) -> tuple[
    pt.LazyFrame[PowerTimeSeries], pt.DataFrame[TimeSeriesMetadata], pt.DataFrame[EffectiveCapacity]
]:
    """Read the observed power, the metadata table, and the effective capacity that scoring joins.

    Raises:
        FileNotFoundError: If the ``effective_capacity`` Delta table does not exist.
    """
    storage_options = settings.storage_options
    actuals_lf = scan_cleaned_power(settings.cleaned_power_time_series_data_path, storage_options)
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
    return actuals_lf, metadata_df, capacity_df


def _log_parent_aggregates(
    experiment_fold_metrics: dict[str, dict[str, list[float]]], metrics_provenance: MlflowTags
) -> None:
    """Log each experiment's mean-across-folds metrics to its MLflow parent run."""
    for exp_name, fold_metrics in experiment_fold_metrics.items():
        experiment_id = get_or_create_experiment(exp_name)
        parent_run_id = get_or_create_parent_run(experiment_id)
        parent_metric_dict = {k: sum(v) / len(v) for k, v in fold_metrics.items()}
        with mlflow.start_run(run_id=parent_run_id):
            mlflow.set_tags(metrics_provenance)
            mlflow.log_metrics(parent_metric_dict)


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
    window: _EvalWindow,
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
        window: The group's evaluation window, from ``_validate_group``; stamped on every row.
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

    mlflow_run_id: str | None = None
    if evaluation_scope == "leaderboard":
        experiment_id = get_or_create_experiment(exp_name)
        parent_run_id = get_or_create_parent_run(experiment_id)
        mlflow_run_id = get_or_create_fold_run(experiment_id, parent_run_id, fold_id)

    enriched = enrich_metrics_rows(
        per_series_metrics,
        exp_name,
        evaluation_scope,
        window.start,
        window.end,
        window.label,
        now,
        mlflow_run_id,
    )
    write_forecast_metrics(
        metrics=enriched,
        table_uri=metrics_path,
        experiment_name=exp_name,
        fold_id=fold_id,
        storage_options=storage_options,
    )

    if evaluation_scope == "leaderboard":
        fold_metric_dict = build_mlflow_aggregate_metrics(per_series_metrics)
        with mlflow.start_run(run_id=mlflow_run_id):
            if provenance is not None:
                mlflow.set_tags(provenance)
            mlflow.log_metrics(fold_metric_dict)
        return enriched.height, fold_metric_dict

    return enriched.height, None


@asset(
    tags=RESEARCH_LAYER_TAGS,
    deps=["cv_power_forecasts", "effective_capacity", "clean_nged_power_data"],
)
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

    Every group is checked before any group is scored, and each check raises. A window that reaches
    ``final_test_start`` in the CV config is refused unless ``NGED_FINAL_TEST=1`` is set in the
    environment of the maintainer's shell; live rows are exempt. In leaderboard scope, every row's
    ``valid_time`` must lie inside the fold's window.

    Experiments whose name starts with ``study/`` hold forecasts submitted through
    ``scripts/forecasting/score_study.py``. In leaderboard scope each study experiment must carry
    exactly the row keys ``(time_series_id, power_fcst_init_time, valid_time, ensemble_member)`` of
    the CV config's ``reference_experiment_name`` for the same fold, so a study cannot abstain on
    hard rows or choose its own ensemble members, and a
    single ``power_fcst_model_name``, so a study cannot down-weight hard rows by spreading rows
    across model names. A run whose population filter names no ``experiment_name`` skips study
    experiments, naming them in a warning and in the ``skipped_study_experiments`` output metadata;
    a run whose population filter names a study's ``experiment_name`` scores that study.

    In leaderboard scope, a run that scores a study also scores the reference experiment's group for
    the same fold, so the study and the reference share one snapshot of the observed power and of
    the effective capacity. The study's fold run then holds each score minus the reference's, under
    the ``vs_reference__`` metric prefix. Every fold run is tagged with ``row_key_fingerprint``, the
    reference's fingerprint for the fold, and ``row_keys_match_reference``. A reviewed experiment
    whose row keys differ from the reference's is tagged, not refused. A run that skips studies tags
    each skipped study's fold run ``stale_against_reference``, ``"true"`` when the reference's
    fingerprint has changed since the study was scored.

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
    # `live_forecasts` writes its output to this same table under `fold_id="live"`, so an unfiltered
    # leaderboard run discovers a group the CV config has never heard of. The operator guide tells
    # you to launch an unfiltered leaderboard run, leaving `fold_id` null to score every fold. The
    # same table also holds any non-leaderboard dev fold (e.g. `smoke_test`), which the CV config
    # does define but which is not part of the leaderboard's evaluation protocol. Either way,
    # leaderboard scope dates its window from the config's leaderboard folds, so those rows have no
    # window to be scored against; skip them rather than fail the whole run on the first one. Ad-hoc
    # scope takes its window from the rows themselves and is the supported way to score live output
    # or dev folds.
    # `study/` experiments hold forecasts submitted through scripts/forecasting/score_study.py, each
    # checked against the reference experiment when it is scored. An unfiltered run skips them: a
    # study goes stale when the reference is re-materialised, and one stale study must not stop the
    # reviewed experiments being scored. A run that names a study's `experiment_name` scores it.
    skipped_study_experiments: list[str] = []
    skipped_study_groups: list[tuple[str, str]] = []
    if config.population_filter.experiment_name is None:
        groups, skipped_study_groups = _split_off_studies(groups)
        skipped_study_experiments = sorted({exp for exp, _ in skipped_study_groups})
        if skipped_study_experiments:
            context.log.warning(
                f"Skipping {skipped_study_experiments} — study experiments are scored only when "
                "the population filter names their experiment_name."
            )
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
            skipped_study_groups = [g for g in skipped_study_groups if g[1] in configured_fold_ids]

    groups, reference_groups_added = _with_reference_groups(
        groups=groups, scan=scan, evaluation_scope=config.evaluation_scope
    )

    if not groups:
        context.log.warning("No forecasts matched the population filter — nothing to score.")
        context.add_output_metadata(
            {
                "n_rows_written": 0,
                "n_groups": 0,
                "skipped_fold_ids": str(skipped_fold_ids),
                "skipped_study_experiments": str(skipped_study_experiments),
            }
        )
        return

    # Refuse every group the scorer must not score before scoring any, so a refusal never leaves an
    # earlier group's rows or MLflow runs behind.
    windows = {
        (exp_name, fold_id): _validate_group(
            exp_name=exp_name,
            fold_id=fold_id,
            group_scan=_group_scan(
                scan if (exp_name, fold_id) in reference_groups_added else pruned_scan,
                exp_name,
                fold_id,
            ),
            scan=scan,
            evaluation_scope=config.evaluation_scope,
        )
        for exp_name, fold_id in groups
    }

    actuals_lf, metadata_df, capacity_df = _read_scoring_inputs(settings)

    if_local_path_then_make_parent_dir(settings.forecast_metrics_data_path)
    now = datetime.now(UTC)
    # Provenance for every fold + parent run this materialisation touches: the code SHA and the
    # versions of the three Delta tables scoring reads (forecasts, actuals, capacity). Built once
    # — all groups are scored in this one process, so they share a single code + data snapshot.
    metrics_provenance = _provenance_tags_with_cleaned_power(
        "metrics",
        settings,
        {
            "power_forecasts": settings.power_forecasts_data_path,
            "effective_capacity": settings.effective_capacity_data_path,
        },
    )
    total_rows = 0
    # Accumulates per-fold metric values for parent-run aggregation (leaderboard scope only).
    # Structure: {experiment_name: {mlflow_metric_key: [value_per_fold, ...]}}
    # e.g. {"xgboost_baseline": {"rmse__all": [0.42, 0.39], "rmse__pv": [0.31, 0.28]}}
    # After the loop, each list is averaged and logged to the experiment's MLflow parent run.
    experiment_fold_metrics: dict[str, dict[str, list[float]]] = {}
    fold_metrics_by_group: dict[tuple[str, str], dict[str, float]] = {}
    reference_fingerprints: dict[str, str] = {}

    for exp_name, fold_id in groups:
        group_scan = _group_scan(
            scan if (exp_name, fold_id) in reference_groups_added else pruned_scan,
            exp_name,
            fold_id,
        )
        group_provenance = _scoring_provenance(
            metrics_provenance=metrics_provenance,
            scan=scan,
            group_scan=group_scan,
            exp_name=exp_name,
            fold_id=fold_id,
            evaluation_scope=config.evaluation_scope,
            reference_fingerprints=reference_fingerprints,
        )
        n_rows, fold_metric_dict = _score_forecast_group(
            exp_name,
            fold_id,
            group_scan,
            actuals_lf,
            metadata_df,
            capacity_df,
            windows[(exp_name, fold_id)],
            config.evaluation_scope,
            settings.forecast_metrics_data_path,
            now,
            storage_options,
            provenance=group_provenance,
        )
        total_rows += n_rows
        if fold_metric_dict is not None:
            fold_metrics_by_group[(exp_name, fold_id)] = fold_metric_dict
            _compare_with_reference(
                exp_name=exp_name, fold_id=fold_id, fold_metrics_by_group=fold_metrics_by_group
            )
            exp_metrics = experiment_fold_metrics.setdefault(exp_name, {})
            for key, value in fold_metric_dict.items():
                exp_metrics.setdefault(key, []).append(value)

    if config.evaluation_scope == "leaderboard":
        _log_parent_aggregates(experiment_fold_metrics, metrics_provenance)

    if config.evaluation_scope == "leaderboard":
        _tag_stale_studies(skipped_groups=skipped_study_groups, scan=scan)

    context.add_output_metadata(
        {
            "n_rows_written": total_rows,
            "n_groups": len(groups),
            "evaluation_scope": config.evaluation_scope,
            "groups": str(groups),
            "skipped_fold_ids": str(skipped_fold_ids),
            "skipped_study_experiments": str(skipped_study_experiments),
            "reference_groups_added": str(reference_groups_added),
        }
    )
