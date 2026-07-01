from datetime import datetime
from typing import Final, Literal, Sequence

import patito as pt
import polars as pl
from typing_extensions import Self

from contracts.power_schemas import LIST_OF_TIME_SERIES_TYPES

from .common import UTC_DATETIME_DTYPE, _get_time_series_id_dtype

_FEATURE_DTYPE = pt.Field(dtype=pl.Float32, allow_missing=True)


TimeFeature = Literal[
    "local_time_of_day_sin",
    "local_time_of_day_cos",
    "local_time_of_year_sin",
    "local_time_of_year_cos",
    "local_day_of_week_sin",
    "local_day_of_week_cos",
    "local_day_of_week",
    "local_utc_offset",
]

SafeInputBaseColumn = Literal[
    "time_series_id",
    "time_series_type",
    "nwp_lead_time_hours",
    "ensemble_member",
    "power_fcst_init_time",
    "nwp_init_time",
]


class AllFeatures(pt.Model):
    """Final joined dataset ready for the ML model.

    Weather features are kept in their physical units (e.g., degrees Celsius, m/s)
    to ensure precision during interpolation and feature engineering.

    DYNAMIC FEATURES:
    In addition to the explicitly defined columns below, the pipeline supports
    dynamically generated features. You can request these in your model config:

    * `power_lag_{hours}h`: The power value shifted by X hours (e.g., `power_lag_24h`).
    * `temperature_2m_rolling_mean_{hours}h`: Rolling average of temperature over X hours (e.g.,
      `temperature_2m_rolling_mean_6h`).

    Note: Dynamic features are not explicitly typed as Patito fields below.
    This is intentional to allow infinite parameterization (e.g., any lag hour)
    without the overhead of metaprogramming or defining hundreds of static fields.
    The pipeline dynamically asserts their presence during feature engineering.
    """

    valid_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)
    time_series_id: int = _get_time_series_id_dtype()
    time_series_type: str = pt.Field(dtype=pl.Enum(LIST_OF_TIME_SERIES_TYPES))
    ensemble_member: int | None = pt.Field(dtype=pl.UInt8, allow_missing=True)

    power_fcst_init_time: datetime = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        description="When OCF's power forecast model was initialised. This might be called `t0` in other OCF projects.",
    )

    nwp_init_time: datetime | None = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        allow_missing=True,
        description="When the NWP run was initialised.",
    )

    power: float = pt.Field(dtype=pl.Float32)
    nwp_lead_time_hours: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)

    # Weather features
    temperature_2m: float | None = _FEATURE_DTYPE
    dew_point_temperature_2m: float | None = _FEATURE_DTYPE
    wind_speed_10m: float | None = _FEATURE_DTYPE
    wind_direction_10m: float | None = _FEATURE_DTYPE
    wind_speed_100m: float | None = _FEATURE_DTYPE
    wind_direction_100m: float | None = _FEATURE_DTYPE
    pressure_surface: float | None = _FEATURE_DTYPE
    pressure_reduced_to_mean_sea_level: float | None = _FEATURE_DTYPE
    geopotential_height_500hpa: float | None = _FEATURE_DTYPE
    downward_long_wave_radiation_flux_surface: float | None = _FEATURE_DTYPE
    downward_short_wave_radiation_flux_surface: float | None = _FEATURE_DTYPE
    precipitation_surface: float | None = _FEATURE_DTYPE
    categorical_precipitation_type_surface: int | None = _FEATURE_DTYPE

    # Derived weather features
    windchill: float | None = _FEATURE_DTYPE

    # Temporal features. `local` means "in the local timezone", e.g. "Europe/London". We use `local`
    # as the main input feature, because it's the local time that mostly drives demand.
    local_utc_offset: int | None = pt.Field(dtype=pl.Int8, allow_missing=True)
    local_time_of_day_sin: float | None = _FEATURE_DTYPE
    local_time_of_day_cos: float | None = _FEATURE_DTYPE
    local_time_of_year_sin: float | None = _FEATURE_DTYPE
    local_time_of_year_cos: float | None = _FEATURE_DTYPE
    local_day_of_week_sin: float | None = _FEATURE_DTYPE
    local_day_of_week_cos: float | None = _FEATURE_DTYPE
    local_day_of_week: str | None = pt.Field(
        dtype=pl.Enum(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
    )

    @classmethod
    def validate(  # ty: ignore[invalid-method-override]
        cls,
        dataframe: pl.DataFrame,
        columns: Sequence[str] | None = None,
        allow_missing_columns: bool = False,
        allow_superfluous_columns: bool = False,
        drop_superfluous_columns: bool = False,
    ) -> pt.DataFrame[Self]:
        """Validate the given dataframe, ensuring uniqueness of the primary key.

        This custom validation step is crucial for preventing weather forecast leakage
        and ensuring data integrity. We enforce uniqueness on the combination of:

        - `time_series_id`: Identifies the specific substation/time series.
        - `power_fcst_init_time`: The initialization time of the power forecast.
        - `valid_time`: The target time for which the forecast is made.
        - `ensemble_member`: The weather forecast ensemble member (if present).

        If `ensemble_member` is present in the dataframe columns, we include it in the
        primary key check to support ensemble forecasts. If it is not present (e.g.,
        for deterministic forecasts or when ensemble members have been aggregated),
        we only check uniqueness across the remaining three columns.

        Args:
            dataframe: The Polars DataFrame to validate.
            columns: Optional list of columns to validate against.
            allow_missing_columns: Whether to allow missing columns.
            allow_superfluous_columns: Whether to allow extra columns.
            drop_superfluous_columns: Whether to drop extra columns.

        Returns:
            A validated Patito DataFrame.

        Raises:
            ValueError: If duplicate entries are found for the primary key.
        """
        validated_df = super().validate(
            dataframe=dataframe,
            columns=columns,
            allow_missing_columns=allow_missing_columns,
            allow_superfluous_columns=allow_superfluous_columns,
            drop_superfluous_columns=drop_superfluous_columns,
        )

        # Define the base primary key columns that must always be unique.
        # This ensures that for any given substation (time_series_id) and target time (valid_time),
        # there is at most one forecast initialized at a specific power_fcst_init_time.
        pk_cols = ["time_series_id", "power_fcst_init_time", "valid_time"]

        if "ensemble_member" in validated_df.columns:
            pk_cols.append("ensemble_member")

        # Check for duplicates. We fail loudly here to prevent downstream training or
        # evaluation on corrupted/duplicated datasets, which could lead to silent bugs.
        if validated_df.select(pk_cols).is_duplicated().any():
            raise ValueError(
                f"Duplicate entries found for primary key columns: {pk_cols}. "
                "This indicates a data integrity issue or potential weather forecast leakage."
            )

        return validated_df


HORIZON_SLICES: Final[tuple[str, ...]] = (
    "all",
    "intraday",
    "day_ahead",
    "short_medium_range",
    "extended_range",
)
"""Horizon slice labels matching the four forecast ranges from the project report.

- `"all"`: aggregates over all horizons and is always computed
- `"intraday"`: 0 – 6 h
- `"day_ahead"`: 6 – 36 h
- `"short_medium_range"`: Day 2 – Day 7
- `"extended_range"`: Day 8 – Day 14
"""

METRIC_NAMES: Final[tuple[str, ...]] = ("mae", "nmae", "rmse", "mbe")
"""Metric names currently implemented.

- `"mae"`: mean absolute error (MW)
- `"nmae"`: normalised MAE (dimensionless; normalised by mean |power|)
- `"rmse"`: root mean squared error (MW)
- `"mbe"`: mean bias error (MW; positive = over-prediction)

Extend as new metrics are added:

- ensemble: `"crps"`, `"spread_skill_ratio"`;
- quantile: `"pinball_loss"`, `"mean_pinball_loss"`;
- calibration: `"picp"`.
"""

METRIC_PARAMS: Final[tuple[str, ...]] = ("all",)
"""Parameter values for parametric metrics (e.g. Pinball Loss at a specific quantile).

- `"all"` is used for all scalar metrics with no extra parameter dimension.
- When Pinball Loss is added, extend with `"p10"`, `"p20"`, …, `"p90"`.
- When PICP is added, extend with `"p10_p90"`, `"p20_p80"`, etc.
"""

EVALUATION_SCOPES: Final[tuple[str, ...]] = ("leaderboard", "production_monitoring", "ad_hoc")
"""Evaluation scopes that coexist in the `forecast_metrics` table.

- `"leaderboard"`: CV-fold leaderboard metrics
- `"production_monitoring"`: live trailing-window monitoring
- `"ad_hoc"`: one-off analyses (no MLflow run)
"""

EvalScopeType = Literal["leaderboard", "ad_hoc"]
"""Type annotation for the evaluation scopes currently accepted by the ``metrics`` asset config.

Distinct from ``EVALUATION_SCOPES`` (the runtime tuple driving the Polars Enum): that tuple
includes ``"production_monitoring"`` for future use; this ``Literal`` reflects only the scopes
the asset actually handles today. Expand to match ``EVALUATION_SCOPES`` when Phase 8 lands.
"""

TIME_SERIES_TYPE_SLICES: Final[tuple[str, ...]] = ("all", *LIST_OF_TIME_SERIES_TYPES)
"""Values for the `time_series_type` metric slice.

Every time_series_type plus the sentinel `"all"` for the across-everything aggregate.
"""


class Metrics(pt.Model):
    """Evaluation metrics for power forecasts — tall format.

    One row per `(time_series_id, power_fcst_model_name, fold_id, horizon_slice, metric_name,
    metric_param)`. `metric_param` encodes the extra parameter dimension for metrics that have
    one, or `"all"` for scalar metrics with no extra dimension. Examples:

    | time_series_id | fold_id | horizon_slice | metric_name       | metric_param | metric_value |
    |----------------|---------|---------------|-------------------|--------------|--------------|
    | 1              | 1       | all           | mae               | all          | 5.2          |
    | 1              | 1       | day_ahead     | rmse              | all          | 7.1          |
    | 1              | 1       | day_ahead     | pinball_loss      | p10          | 2.1          |
    | 1              | 1       | day_ahead     | pinball_loss      | p50          | 3.4          |
    | 1              | 1       | day_ahead     | mean_pinball_loss | all          | 2.4          |
    | 1              | 1       | day_ahead     | picp              | p10_p90      | 0.78         |

    Primary key: `(time_series_id, power_fcst_model_name, fold_id, horizon_slice, metric_name,
    metric_param)`.
    """

    time_series_id: int = _get_time_series_id_dtype()

    # String (not Categorical): fold_id/experiment_name are Delta partition columns and delta-rs
    # stores dictionary-encoded columns as String anyway; String keeps them cast-free and lets
    # predicate pushdown work. See the "declare Delta filter/partition columns as String" gotcha:
    # ../../../../CLAUDE.md#delta-lake-dictionary-encoded-columns-declare-delta-filterpartition-columns-as-string
    power_fcst_model_name: str = pt.Field(
        dtype=pl.String,
        description="Identifier for the ML-based power forecasting model family.",
    )

    fold_id: str = pt.Field(
        dtype=pl.String,
        description="CV fold year (e.g. '2022'), or 'live' for production forecasts. Matches PowerForecast.fold_id.",
    )

    horizon_slice: str = pt.Field(
        dtype=pl.Enum(HORIZON_SLICES),
        description=(
            "'all' aggregates over all forecast horizons.  Other values select the HORIZON_SLICES bands."
        ),
    )

    metric_name: str = pt.Field(
        dtype=pl.Enum(METRIC_NAMES),
        description="The name of the metric (e.g. 'mae', 'rmse').",
    )

    metric_param: str = pt.Field(
        dtype=pl.Enum(METRIC_PARAMS),
        description=(
            "Extra parameter dimension for parametric metrics.  "
            "'all' for scalar metrics (MAE, RMSE, MBE, NMAE).  "
            "For Pinball Loss: 'p10', 'p50', etc.  "
            "For PICP: 'p10_p90', 'p20_p80', etc."
        ),
    )

    metric_value: float = pt.Field(dtype=pl.Float32, description="The computed metric value.")

    evaluation_scope: str = pt.Field(
        dtype=pl.Enum(EVALUATION_SCOPES),
        allow_missing=True,
        description=(
            "Which evaluation produced this row, so leaderboard, live-monitoring, and "
            "one-off metrics coexist in one table and stay separable."
        ),
    )
    """The columns from `evaluation_scope` and below are populated by the ``metrics`` Dagster asset.
    They are ``allow_missing`` so that the pure ``compute_metrics()`` helper can emit the core
    metric rows and have the asset enrich them with scope/window provenance before the frame is
    written to the ``forecast_metrics`` Delta table."""

    time_series_type: str | None = pt.Field(
        dtype=pl.Enum(TIME_SERIES_TYPE_SLICES),
        allow_missing=True,
        description=(
            "The time-series category for this row. Null when metadata is unavailable for a series."
        ),
    )

    window_start: datetime = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        allow_missing=True,
        description=(
            "Inclusive start of the valid_time window this row covers. For a CV fold this "
            "is the fold's val_start; for monitoring it is the trailing-window start."
        ),
    )

    window_end: datetime = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        allow_missing=True,
        description="End of the valid_time window this row covers (fold val_end or window end).",
    )

    window_label: str = pt.Field(
        dtype=pl.String,
        allow_missing=True,
        description="Human label for the window, e.g. '2025', '24h', '7d', 'full_fold'.",
    )

    computed_at: datetime = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        allow_missing=True,
        description=(
            "When this metric row was written (provenance; orders the append-only "
            "monitoring series and distinguishes recomputations)."
        ),
    )

    mlflow_run_id: str | None = pt.Field(
        dtype=pl.String,
        allow_missing=True,
        description=(
            "Convenience cross-link to the MLflow run this metric row belongs to. "
            "Null for 'ad_hoc' rows, which have no MLflow run."
        ),
    )

    experiment_name: str = pt.Field(
        dtype=pl.String,  # String (not Categorical) — Delta partition column, see fold_id above.
        allow_missing=True,
        description=(
            "Experiment that produced these forecasts. Delta partition key alongside fold_id. "
            "Populated by the metrics Dagster asset after compute_metrics() returns; "
            "allow_missing so compute_metrics() itself need not produce it."
        ),
    )


class EligibleTimeSeries(pt.Model):
    """The canonical per-fold population of eligible ``time_series_id``s.

    Written by the ``eligible_time_series`` Dagster asset (one Delta partition per ``fold_id``)
    and read by ``trained_cv_model`` and ``cv_power_forecasts``. Eligibility is a function of
    data coverage and the fold dates **only** — never the model or experiment config — so every
    experiment trains and scores a fold on the identical population, which is what makes
    leaderboard comparisons apples-to-apples.

    One row per eligible ``(fold_id, time_series_id)``.
    """

    fold_id: str = pt.Field(
        dtype=pl.String,
        description="The CV fold this eligibility row belongs to (e.g. 'mid_2025_to_mid_2026'); the Delta partition key.",
    )
    time_series_id: int = _get_time_series_id_dtype()
