"""Expanding-window cross-validation for power forecasting models."""

from datetime import date, datetime, timedelta, timezone

import patito as pt
import polars as pl
from contracts.hydra_schemas import CvFoldConfig
from contracts.power_schemas import PowerForecast, PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import NwpInMemory, NwpOnDisk

from ml_core.base_forecaster import BaseForecaster, BaseForecasterConfig
from ml_core.features import engineer_features


def _date_to_utc_datetime(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def cross_validate(
    forecaster_class: type[BaseForecaster],
    forecaster_config: BaseForecasterConfig,
    power_lf: pt.LazyFrame[PowerTimeSeries],
    nwp_lf: pt.LazyFrame[NwpOnDisk] | None,
    metadata_df: pt.DataFrame[TimeSeriesMetadata],
    folds: list[CvFoldConfig],
    min_training_months: int = 6,
) -> pt.DataFrame[PowerForecast]:
    """Run expanding-window cross-validation and return all fold predictions.

    For each fold a fresh forecaster is trained on the training window and used to
    predict over the validation window.  Only time series with sufficient data are
    included in each fold (see ``min_training_months``).

    The returned ``PowerForecast`` DataFrame has ``fold_id`` set to the validation
    year string (e.g. ``"2022"``).  The full half-hourly predictions are the primary
    artefact — metrics can always be recomputed from them without re-running training.

    Args:
        forecaster_class: The uninitialised ``BaseForecaster`` subclass to use.
            A fresh instance is created per fold via
            ``forecaster_class(forecaster_config)`` to prevent state leakage
            between folds.
        forecaster_config: Configuration object passed to each fresh instance.
            Must be a concrete subclass of ``BaseForecasterConfig`` compatible
            with ``forecaster_class``.
        power_lf: Half-hourly power observations for all time series, spanning
            at least the full training and validation date range across all folds.
            Lazy — not collected until model boundary.
        nwp_lf: NWP data on disk (integer-scaled).  If ``None``, only
            power-based and time-based features can be requested (e.g. for
            persistence baselines).  The caller is responsible for loading the
            full relevant date range; this function filters by ``init_time``
            inside the fold loop.
        metadata_df: Time series metadata used both for feature engineering and
            for mapping NWP h3_index → time_series_id.
        folds: Ordered list of fold definitions.  Usually loaded from
            ``conf/cv/default.yaml`` via ``CvConfig``.
        min_training_months: A time series is eligible for a fold only if its
            first observation is at least this many months (approximated as
            30 days each) before ``fold.val_start``.  The default of 6 matches
            the project's evaluation protocol.

    Returns:
        A validated ``PowerForecast`` DataFrame with all fold predictions
        concatenated.  ``fold_id`` is the validation year string (e.g. ``"2022"``)
        for use in downstream metric computation.

    Raises:
        ValueError: If no eligible time series are found for any fold.
    """
    # Pre-compute date coverage per time series once (avoids re-scanning per fold).
    ts_coverage = (
        power_lf.group_by("time_series_id")
        .agg(
            pl.col("time").min().alias("first_obs"),
            pl.col("time").max().alias("last_obs"),
        )
        .collect()
    )

    # Broadcast NWP from h3_index space to time_series_id space (lazy, once).
    # engineer_features() expects NWP data indexed by time_series_id, not h3_index.
    nwp_in_memory_lf: pt.LazyFrame[NwpInMemory] | None
    if nwp_lf is not None:
        nwp_with_ts_lf = nwp_lf.join(
            metadata_df.lazy().select(["time_series_id", "h3_res_5"]),
            left_on="h3_index",
            right_on="h3_res_5",
        )
        # to_nwp_in_memory is lazy: converts integer-scaled columns to Float32.
        # The extra time_series_id column is preserved as a superfluous column.
        nwp_in_memory_lf = NwpOnDisk.to_nwp_in_memory(
            pt.LazyFrame.from_existing(nwp_with_ts_lf).set_model(NwpOnDisk)
        )
    else:
        nwp_in_memory_lf = None

    min_training_days = min_training_months * 30

    all_fold_forecasts: list[pl.DataFrame] = []

    for fold in folds:
        val_start_dt = _date_to_utc_datetime(fold.val_start)
        val_end_dt = _date_to_utc_datetime(fold.val_end).replace(hour=23, minute=59, second=59)
        train_start_dt = _date_to_utc_datetime(fold.train_start)
        train_end_dt = val_start_dt  # exclusive upper bound for training

        # A time series is eligible if:
        # (a) its first observation is at least min_training_months before val_start
        # (b) its last observation date is on or after val_end
        min_first_obs = val_start_dt - timedelta(days=min_training_days)
        eligible_ts_ids = ts_coverage.filter(
            (pl.col("first_obs") <= min_first_obs)
            & (pl.col("last_obs").dt.date() >= pl.lit(fold.val_end))
        )["time_series_id"].to_list()

        if not eligible_ts_ids:
            continue

        eligible_metadata = pt.DataFrame(
            metadata_df.filter(pl.col("time_series_id").is_in(eligible_ts_ids))
        ).set_model(TimeSeriesMetadata)

        # --- Training ---
        train_power_lf = pt.LazyFrame.from_existing(
            power_lf.filter(
                pl.col("time_series_id").is_in(eligible_ts_ids)
                & (pl.col("time") >= train_start_dt)
                & (pl.col("time") < train_end_dt)
            )
        ).set_model(PowerTimeSeries)

        train_nwp_lf: pt.LazyFrame[NwpInMemory] | None = (
            pt.LazyFrame.from_existing(
                nwp_in_memory_lf.filter(
                    pl.col("time_series_id").is_in(eligible_ts_ids)
                    & (pl.col("init_time") >= train_start_dt)
                    & (pl.col("init_time") < train_end_dt)
                )
            ).set_model(NwpInMemory)
            if nwp_in_memory_lf is not None
            else None
        )

        train_features_lf = engineer_features(
            selected_features=forecaster_config.selected_features,
            power_time_series=train_power_lf,
            time_series_metadata=eligible_metadata,
            nwp=train_nwp_lf,
        )

        forecaster = forecaster_class(forecaster_config)
        forecaster.train(train_features_lf)

        # --- Validation ---
        val_power_lf = pt.LazyFrame.from_existing(
            power_lf.filter(
                pl.col("time_series_id").is_in(eligible_ts_ids)
                & (pl.col("time") >= val_start_dt)
                & (pl.col("time") <= val_end_dt)
            )
        ).set_model(PowerTimeSeries)

        val_nwp_lf: pt.LazyFrame[NwpInMemory] | None = (
            pt.LazyFrame.from_existing(
                nwp_in_memory_lf.filter(
                    pl.col("time_series_id").is_in(eligible_ts_ids)
                    & (pl.col("init_time") >= val_start_dt)
                    & (pl.col("init_time") <= val_end_dt)
                )
            ).set_model(NwpInMemory)
            if nwp_in_memory_lf is not None
            else None
        )

        val_features_lf = engineer_features(
            selected_features=forecaster_config.selected_features,
            power_time_series=val_power_lf,
            time_series_metadata=eligible_metadata,
            nwp=val_nwp_lf,
        )

        fold_forecast = forecaster.predict(val_features_lf)
        # Overwrite the "live" default with the validation year for this fold.
        fold_forecast = fold_forecast.with_columns(
            fold_id=pl.lit(str(fold.val_start.year)).cast(pl.Categorical)
        )
        all_fold_forecasts.append(fold_forecast)

    if not all_fold_forecasts:
        raise ValueError(
            "No eligible time series found for any fold. "
            "Check that power data spans the required date ranges."
        )

    result = pl.concat(all_fold_forecasts)
    return PowerForecast.validate(result, allow_superfluous_columns=True)
