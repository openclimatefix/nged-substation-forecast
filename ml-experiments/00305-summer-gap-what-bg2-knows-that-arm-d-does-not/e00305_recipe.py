"""The N1 fitting helpers of experiment 00087, copied so experiment 305 imports only its own folder.

Copied from `e00087_fit_demand.py` (the `n1h` configuration, control-member rows, and the 3.5-hour
refit cutoff of `e00087_regimes.py`), with the weekly-analogue and solar-offset options removed.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

import e00305_common as common
import numpy as np
import polars as pl
import xgboost as xgb

WEATHER: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
)


WEEK_STEPS: Final[int] = 7 * common.STEPS_PER_DAY


WEEKLY_ANALOGUE_WEEKS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6)


A7_COLUMNS: Final[tuple[str, ...]] = ("weekly_mean", "weekly_median")


@dataclass(frozen=True)
class Config:
    """A training configuration.

    Attributes:
        name: The folder name under `anchor_preds`.
        half_life_days: Recency half-life of the row weights, anchored at the refit time.
        rounds: Boosting rounds.
        eta: Learning rate.
        regime_names: The regimes (keys of `regimes.REGIMES`) whose refits the configuration fits.
        groups: The substation groups the configuration is fitted for.
        frozen: Train once on the pilot's control-member rows to 2025-06-30 only (experiment
            00033's recipe), with no rows from later runs.
        a7: Add experiment 00030's weekly-analogue mean and median as input columns.
        pooled: Fit one model per metered-units pool, with the series identifier as a category.
    """

    name: str
    half_life_days: float
    rounds: int
    eta: float
    regime_names: tuple[str, ...]
    groups: tuple[str, ...]
    frozen: bool = False
    a7: bool = False
    pooled: bool = False


CONFIGS: Final[dict[str, Config]] = {
    "n1h": Config(
        name="n1h",
        half_life_days=common.SHORT_HALF_LIFE_DAYS,
        rounds=common.N_ROUNDS,
        eta=0.05,
        regime_names=("weekly", "monthly"),
        groups=common.GROUPS,
    ),
}
REFIT_CUTOFF_AGE: Final[timedelta] = timedelta(hours=3.5)
"""A refit sees only rows whose valid time is at least this long before the refit."""


def input_columns(*, a7: bool) -> list[str]:
    """Return the input columns in matrix order."""
    return [
        *common.feature_columns(),
        *common.N1_EXTRA_COLUMNS,
        *common.LAG_COLUMNS,
        *(A7_COLUMNS if a7 else ()),
    ]


def add_lags(*, frame: pl.DataFrame, power: np.ndarray, a7: bool) -> pl.DataFrame:
    """Add the three origin lags (and the weekly-analogue columns if `a7`) to a frame of runs."""
    target = np.asarray(common.grid_index(times=frame["valid_time"]))
    origin = np.asarray(common.grid_index(times=frame["power_fcst_init_time"]))
    return frame.with_columns(
        pl.DataFrame(
            common.origin_lags(power=power, target=target, origin=origin),
            schema=list(common.LAG_COLUMNS),
            orient="row",
        ).cast(pl.Float32)
    )


def training_rows(
    *, row: dict, arrays: common.SeriesArrays, frozen: bool, a7: bool
) -> pl.DataFrame:
    """Return one series' control-member rows with derived columns and lags.

    The rows run to 2025-06-30 for the frozen configuration and to 2026-06-29 otherwise. The power
    of the rows after 2025-06-30 is filled from the dense grid. Every row's eligibility for a refit
    (valid time and run start) is applied per refit, not here.

    Args:
        row: The series' row of `common.series_table()`.
        arrays: The series' arrays on the dense grid.
        frozen: Whether to keep only the pilot's training rows, which end at 2025-06-30.
        a7: Whether to add the weekly-analogue columns.

    Returns:
        The rows keyed by series, run, and valid time, with `ensemble_member`, `power` (in MW),
        the weather columns, the derived columns, `lead_hours`, and the lags.
    """
    series_id = row["time_series_id"]
    before = (
        pl.scan_parquet(common.ENS_TRAIN_PATH)
        .filter(pl.col("time_series_id") == series_id)
        .select(*common.KEYS, "ensemble_member", "power", *WEATHER)
        .collect()
    )
    if frozen:
        rows = before
    else:
        after = pl.concat(
            [
                pl.read_parquet(path).filter(pl.col("time_series_id") == series_id)
                for path in sorted(common.AFTER_DIR.glob("init=*.parquet"))
            ]
        )
        padded = after.with_columns(power=pl.lit(None, dtype=pl.Float32)).select(before.columns)
        rows = pl.concat([before, padded.cast(before.schema)])
    index = np.asarray(common.grid_index(times=rows["valid_time"]))
    filled = common.take(values=arrays.power, index=index) * row["effective_capacity_mw"]
    rows = rows.with_columns(
        power=pl.when(pl.col("power").is_null()).then(pl.Series(filled)).otherwise(pl.col("power"))
    )
    rows = rows.filter(
        pl.col("power").is_finite() & (pl.col("valid_time") > pl.col("power_fcst_init_time"))
    )
    rows = common.with_derived(frame=rows, latitude=row["latitude"], longitude=row["longitude"])
    rows = rows.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
    )
    return add_lags(frame=rows, power=arrays.power, a7=a7)


def known_rows(*, rows: pl.DataFrame, refit: datetime, config: Config) -> pl.DataFrame:
    """Return the rows a refit at `refit` may see, and check that no row comes from its future.

    A refit sees a row only if its valid time is at least 3.5 hours before the refit and its run
    began at or before the refit. The frozen model, anchored at 2025-06-30 23:30, sees the pilot's
    training rows up to that valid time (experiment 00033's recipe has no 3.5-hour gap).

    Args:
        rows: One series' training rows with `valid_time` and `power_fcst_init_time`.
        refit: The refit time.
        config: The configuration, which says whether the model is frozen.

    Returns:
        The rows the refit may see.

    Raises:
        ValueError: If no row is usable, or a kept row is newer than the refit allows.
    """
    latest_valid = common.LAST_TRAINING_VALID if config.frozen else refit - REFIT_CUTOFF_AGE
    known = rows.filter(
        (pl.col("valid_time") <= latest_valid) & (pl.col("power_fcst_init_time") <= refit)
    )
    too_new = known.filter(
        (pl.col("valid_time") > latest_valid) | (pl.col("power_fcst_init_time") > refit)
    ).height
    if known.is_empty() or too_new:
        msg = f"refit {refit}: no usable rows or a row from after the refit"
        raise ValueError(msg)
    return known


def parameters(*, config: Config, seed: int, nthread: int) -> dict[str, object]:
    """Return the XGBoost parameters of a configuration."""
    return {**common.QUANTILE_PARAMETERS, "seed": seed, "eta": config.eta, "nthread": nthread}


_CACHE: dict[str, object] = {}


def _series_and_arrays() -> tuple[pl.DataFrame, dict[str, common.SeriesArrays]]:
    """Return the series table and arrays, loaded once per worker process."""
    if "series" not in _CACHE:
        series = common.series_table()
        _CACHE["series"] = series
        _CACHE["arrays"] = common.load_series_arrays(series=series)
        _CACHE["rows"] = {}
        _CACHE["anomaly"] = {}
    return _CACHE["series"], _CACHE["arrays"]  # ty: ignore[invalid-return-type]


def _training_rows_cached(
    *, row: dict, arrays: common.SeriesArrays, config: Config
) -> pl.DataFrame:
    """Return a series' training rows, kept for the worker's later jobs."""
    cache: dict[tuple[str, bool, bool], pl.DataFrame] = _CACHE["rows"]  # ty: ignore[invalid-assignment]
    key = (row["site"], config.frozen, config.a7)
    if key not in cache:
        cache[key] = training_rows(row=row, arrays=arrays, frozen=config.frozen, a7=config.a7)
    return cache[key]


def train_inputs(*, known: pl.DataFrame, row: dict, refit: datetime, config: Config) -> xgb.DMatrix:
    """Return the weighted training matrix of one series' rows, weights anchored at `refit`."""
    age_days = (refit - known["valid_time"]).dt.total_seconds() / 86_400
    label = (known["power"] / row["effective_capacity_mw"]).to_numpy()
    return xgb.DMatrix(
        known.select(input_columns(a7=config.a7)).to_numpy().astype(np.float32),
        label=label,
        weight=0.5 ** (age_days.to_numpy() / config.half_life_days),
    )


def prepared_run(
    *, run_rows: pl.DataFrame, row: dict, arrays: common.SeriesArrays, a7: bool
) -> pl.DataFrame:
    """Return one run's member rows for one series with derived columns and lags, sorted.

    Args:
        run_rows: The run's frame rows for the series, every valid time with every member.
        row: The series' row of `common.series_table()`.
        arrays: The series' arrays on the dense grid.
        a7: Whether to add experiment 00030's weekly-analogue summaries.

    Returns:
        The rows sorted by valid time and member, with the derived columns and lags.

    Raises:
        ValueError: If a valid time lacks a member.
    """
    prepared = common.with_derived(
        frame=run_rows, latitude=row["latitude"], longitude=row["longitude"]
    )
    prepared = add_lags(frame=prepared, power=arrays.power, a7=a7).sort(
        "valid_time", "ensemble_member"
    )
    members = prepared["ensemble_member"].n_unique()
    if prepared.height != prepared.select(*common.KEYS).n_unique() * members:
        msg = "every valid time must hold every member"
        raise ValueError(msg)
    return prepared
