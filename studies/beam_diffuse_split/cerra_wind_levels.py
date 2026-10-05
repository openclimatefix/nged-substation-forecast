"""Does blending CERRA's wind levels beat CERRA's 100 m wind alone at predicting wind power?

One-off throwaway script for <https://github.com/openclimatefix/nged-substation-forecast/issues/957>.
The plan is `plans/study-957-cerra-wind-blend.md` and was committed before the first fit.

**Data.** `data/studies/weather/CERRA/`: CERRA's wind speed at 10, 50, 75, 100 and 150 m, at the
190 grid cells around the trial area, every 3 hours (00, 03, ..., 21 UTC) from 2019-09-01 to
2026-06-30. The files hold speed only, with no direction. `studies.reanalysis_wind` reads each wind
farm's nearest cell and builds the power hour, centred on the label. The 10 m speed comes from
CERRA's single-levels product and the other four from its height-levels product.

**Row set.** Every 3-hourly wind-farm hour that has a centred power hour, minus every hour holding
an exactly-zero half-hour. The rule reads the target only, so every arm scores exactly the same
rows. The row set has one hour in three, and `hour_of_day` takes 8 values.

**Gates that run before any fit.** `era_step_table` looks for a step in the ratio of each height to
100 m at any month, which would mean CERRA's record joins two production streams; the run stops if
`ERA_GATE_Z` is exceeded. `power_hour_scan` fits the `speed_100m` arm at five power-hour offsets and
stops unless the centred hour scores best. `check_settings` stops unless every fit is on the CPU
with no column subsampling.

**Arms.** Every arm carries `hour_of_day`, `day_of_year`, and five wind columns, and every fit uses
`colsample_bytree=1` (the setting is absent from `studies.cross_validation.booster_parameters`).
An arm with fewer than five real wind columns is padded with monotone transforms of one of them,
which a tree cannot use. `check_column_counts` raises if two arms differ in width, and the report
prints every arm's column list.

- `speed_10m`, `speed_100m`: one height.
- `speed_10m_100m`: those two heights.
- `mean_near_100m`: the mean of the 75, 100 and 150 m speeds.
- `levels_50_to_150`: the 50, 75, 100 and 150 m speeds.
- `levels_all`: all five heights.
- `speed_100m_noise`: the negative control, 100 m plus four columns each holding another month's
  values of the 10, 50, 75 and 150 m speeds (`with_shuffled_levels`). Every arm has 7 columns (2
  shared features plus 5 wind columns), so the control measures four shuffled columns replacing
  inert padding.

**Folds and intervals.** `studies.cross_validation.assign_folds` cuts each farm's span into 5
contiguous blocks of whole months, and `raise_on_uncovered_months` checks that no calendar month is
without training rows. Intervals resample whole calendar months and one of three seeds
(`studies.bootstrap.bootstrap_difference`, 2,000 resamples). The three farms share their weather, so
the intervals describe these farms only.

**Planned contrasts** (`PLANNED_CONTRASTS`, written into the plan before any fit, each also run at
the second hyperparameter setting):

1. `speed_100m` minus `speed_10m`.
2. `mean_near_100m` minus `speed_100m`.
3. `levels_50_to_150` minus `speed_100m`.
4. `levels_all` minus `levels_50_to_150`.

Every other contrast is exploratory.

**Controls.** The negative control is `speed_100m_noise` minus `speed_100m`. The positive control is
a synthetic target, a fixed power curve applied to CERRA's speed interpolated to 120 m linearly in
log-height between the 100 and 150 m levels, plus noise, on which `levels_50_to_150` must beat
`speed_100m` by a margin that the interval excludes. The run writes every output first and then
raises if it does not.

Run it with `uv run python studies/beam_diffuse_split/cerra_wind_levels.py`.
`--report-only` rebuilds `report.md` and `intervals.parquet` from the saved losses, still checking
the saved fingerprint. A re-run stops while any output exists, until it is moved to a
`superseded/` subfolder. Only one agent
may run it at a time, because every worktree shares one data folder.
"""

import argparse
import inspect
import logging
import sys
from pathlib import Path
from typing import Any, Final, NamedTuple

import numpy as np
import polars as pl
from ens_past_solar import _arm_columns_lines, _fingerprint
from studies.arm_runner import Job, add_time_features, run_all
from studies.bootstrap import (
    bootstrap_absolute,
    bootstrap_difference,
    bootstrap_difference_at_level,
    per_fold_differences,
)
from studies.cross_validation import (
    PRIMARY_HYPER_PARAMETERS,
    SENSITIVITY_HYPER_PARAMETERS,
    THREADS_PER_FIT,
    assign_folds,
    booster_parameters,
    calendar_month_coverage,
    out_of_fold_losses,
    raise_on_uncovered_months,
)
from studies.guards import check_no_missing, refuse_to_overwrite
from studies.power import hourly_from_half_hourly
from studies.pv_dataset import POWER_DELTA_URI, wind_sites
from studies.reanalysis_wind import (
    derive_nearest_cells,
    join_centred_power,
    read_cerra_wind,
)
from studies.sources import STUDIES_DATA_DIR, WEATHER_DATA_DIR
from studies.wind_direction import shuffled_by_month
from weather_products import METRIC, PERCENTAGE_POINTS, _mae

_LOG: Final[logging.Logger] = logging.getLogger("cerra_wind_levels")

CERRA_DIR: Final[Path] = WEATHER_DATA_DIR / "CERRA"
GRID_PATH: Final[Path] = CERRA_DIR / "cerra_grid.parquet"
"""The grid's `y_index`, `x_index`, `latitude` and `longitude`. The file is private: nothing reads a
coordinate out of it into a log, a report or a chart."""

OUTPUT_DIR: Final[Path] = STUDIES_DATA_DIR / "cerra_wind_levels"
"""Where the script writes its outputs, and a `superseded/` folder for re-runs."""

MAX_WORKERS: Final[int] = 2
"""How many (arm, farm) fits run at once. Each fit uses `THREADS_PER_FIT` cores, so the run uses at
most `MAX_WORKERS * THREADS_PER_FIT` cores."""

MAX_CORES: Final[int] = 8
"""The most cores the run may use, from the study skill's limit."""

SPEED_10M: Final[str] = "wind_speed_10m"
SPEED_50M: Final[str] = "wind_speed_50m"
SPEED_75M: Final[str] = "wind_speed_75m"
SPEED_100M: Final[str] = "wind_speed_100m"
SPEED_150M: Final[str] = "wind_speed_150m"
MEAN_NEAR_100M: Final[str] = "wind_speed_mean_near_100m"
"""The mean of the 75, 100 and 150 m speeds, added by `with_arm_columns`."""

SPEED_COLUMNS: Final[tuple[str, ...]] = (SPEED_10M, SPEED_50M, SPEED_75M, SPEED_100M, SPEED_150M)
"""The five columns `studies.reanalysis_wind.read_cerra_wind` returns."""

SHARED_FEATURES: Final[tuple[str, ...]] = ("hour_of_day", "day_of_year")
"""Features every arm gets, on top of its five wind columns. No arm has an era column."""

WIND_COLUMN_COUNT: Final[int] = 5
"""The number of wind columns every arm carries: the widest arm's real columns."""

TRANSFORMS: Final[tuple[str, ...]] = ("squared", "sqrt", "log1p", "cubed")
"""The monotone transforms `with_arm_columns` adds beside each padded column, in padding order."""

SHUFFLED_HEIGHTS_M: Final[tuple[int, ...]] = (10, 50, 75, 150)
"""The heights whose speeds `speed_100m_noise` shows, each shuffled by month."""

NOISE_SEED: Final[int] = 957
"""Seeds the month shuffles and the synthetic target's noise."""

# The synthetic target's power curve, as a fraction of capacity: zero below cut-in, a cubic in speed
# up to rated speed, and rated power above it.
CUT_IN_M_S: Final[float] = 3.0
RATED_M_S: Final[float] = 12.0
SYNTHETIC_NOISE_FRACTION: Final[float] = 0.01
"""The standard deviation of the synthetic target's noise, as a fraction of capacity."""

INTERPOLATION_HEIGHT_M: Final[float] = 120.0
"""The height the synthetic target's wind is interpolated to, between the 100 and 150 m levels."""

SYNTHETIC_TARGET: Final[str] = "synthetic_power_mw"

CENTRED_SHIFT: Final[int] = 1
"""The power hour's offset in half-hours, scanned by `power_hour_scan`. With 0 the hour ends at its
label, as in the solar studies. With 1 the hour holds the half-hours ending at the label and at the
label plus 30 minutes, centring it on the label, which suits an instantaneous wind value."""

POWER_HOUR_SHIFTS: Final[tuple[int, ...]] = (-1, 0, 1, 2, 3)
"""The offsets `power_hour_scan` fits, in half-hours."""

ERA_GATE_Z: Final[float] = 5.0
"""The largest z-score of a step in a height's ratio to 100 m that `era_step_table` tolerates."""

BONFERRONI_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 4)
"""The coverage in percent of the interval adjusted for the four planned contrasts: 98.75."""

ERA_WINDOW_MONTHS: Final[int] = 12
"""The months either side of a candidate join that `era_step_table` averages."""

PRIMARY_SETTING: Final[str] = "primary"
SENSITIVITY_SETTING: Final[str] = "sensitivity"
POSITIVE_CONTROL_SETTING: Final[str] = "positive_control"
"""The names `run_all` labels each family of jobs with."""


class Contrast(NamedTuple):
    """One (treatment, reference) pairing, its plain-words meaning, and whether it is planned."""

    treatment: str
    reference: str
    meaning: str
    planned: bool


PLANNED_CONTRASTS: Final[tuple[Contrast, ...]] = (
    Contrast("speed_100m", "speed_10m", "what 100 m wind adds over 10 m wind", True),
    Contrast(
        "mean_near_100m", "speed_100m", "a mean of the near-100 m levels against 100 m alone", True
    ),
    Contrast(
        "levels_50_to_150", "speed_100m", "four heights as separate columns against 100 m", True
    ),
    Contrast("levels_all", "levels_50_to_150", "adding 10 m to the four higher heights", True),
)
"""The four contrasts the plan names before any fit. Every other contrast is exploratory."""

EXPLORATORY_CONTRASTS: Final[tuple[Contrast, ...]] = (
    Contrast("speed_10m_100m", "speed_100m", "adding 10 m to 100 m", False),
    Contrast("levels_all", "speed_100m", "all five heights against 100 m", False),
    Contrast("levels_all", "mean_near_100m", "all five heights against the mean", False),
    Contrast("levels_50_to_150", "mean_near_100m", "four heights against the mean", False),
)
"""Contrasts reported for context, not relied on."""

NEGATIVE_CONTROL: Final[Contrast] = Contrast(
    "speed_100m_noise",
    "speed_100m",
    "four shuffled columns against four inert padding columns, both arms with 7 columns",
    False,
)
POSITIVE_CONTROL: Final[Contrast] = Contrast(
    "levels_50_to_150",
    "speed_100m",
    "the blend against 100 m where the target needs a blend",
    False,
)

REAL_ARMS: Final[tuple[str, ...]] = (
    "speed_10m",
    "speed_100m",
    "speed_10m_100m",
    "mean_near_100m",
    "levels_50_to_150",
    "levels_all",
)
"""The arms fitted on the real target."""

NEGATIVE_CONTROL_ARM: Final[str] = "speed_100m_noise"

POSITIVE_CONTROL_ARMS: Final[tuple[str, ...]] = ("speed_100m", "levels_50_to_150")
"""The two arms the positive control compares, so the only ones fitted on the synthetic target."""


def padding_columns(*, base: str, count: int) -> tuple[str, ...]:
    """Name the monotone transforms of one column that pad an arm to full width.

    Args:
        base: The column to transform.
        count: How many transforms to name, at most `len(TRANSFORMS)`.

    Returns:
        The transform columns `with_arm_columns` adds, in `TRANSFORMS` order.

    Raises:
        ValueError: If `count` is negative or exceeds `len(TRANSFORMS)`.
    """
    if not 0 <= count <= len(TRANSFORMS):
        msg = f"count must be between 0 and {len(TRANSFORMS)}, got {count}"
        raise ValueError(msg)
    return tuple(f"{base}_{name}" for name in TRANSFORMS[:count])


def _wind_columns(*, real: tuple[str, ...], pad_from: str) -> tuple[str, ...]:
    """Return one arm's wind columns: its real columns, padded to `WIND_COLUMN_COUNT`.

    Args:
        real: The arm's real wind columns.
        pad_from: The column the padding transforms.

    Returns:
        A tuple of exactly `WIND_COLUMN_COUNT` names.
    """
    return (*real, *padding_columns(base=pad_from, count=WIND_COLUMN_COUNT - len(real)))


def shuffled_column(*, height_m: int) -> str:
    """Name the column holding another month's speed at one height.

    Args:
        height_m: The height in metres.

    Returns:
        The column `with_shuffled_levels` adds.
    """
    return f"wind_speed_{height_m}m_other_month"


def arm_columns() -> dict[str, tuple[str, ...]]:
    """Return every arm's feature columns, from one function so no arm can silently lose one.

    Returns:
        Each arm's name to its shared features then its five wind columns.
    """
    wind: dict[str, tuple[str, ...]] = {
        "speed_10m": _wind_columns(real=(SPEED_10M,), pad_from=SPEED_10M),
        "speed_100m": _wind_columns(real=(SPEED_100M,), pad_from=SPEED_100M),
        "speed_10m_100m": _wind_columns(real=(SPEED_10M, SPEED_100M), pad_from=SPEED_100M),
        "mean_near_100m": _wind_columns(real=(MEAN_NEAR_100M,), pad_from=MEAN_NEAR_100M),
        "levels_50_to_150": _wind_columns(
            real=(SPEED_50M, SPEED_75M, SPEED_100M, SPEED_150M), pad_from=SPEED_100M
        ),
        "levels_all": _wind_columns(real=SPEED_COLUMNS, pad_from=SPEED_100M),
        NEGATIVE_CONTROL_ARM: (
            SPEED_100M,
            *(shuffled_column(height_m=height) for height in SHUFFLED_HEIGHTS_M),
        ),
    }
    return {arm: (*SHARED_FEATURES, *columns) for arm, columns in wind.items()}


def check_column_counts(*, arms: dict[str, tuple[str, ...]]) -> None:
    """Stop unless every arm has the same number of distinct columns.

    Args:
        arms: Each arm's name to its feature columns.

    Raises:
        ValueError: Naming the arms whose width differs from the widest, or that repeat a column.
    """
    widths = {arm: len(columns) for arm, columns in arms.items()}
    repeats = [arm for arm, columns in arms.items() if len(set(columns)) != len(columns)]
    widest = max(widths.values())
    narrow = {arm: width for arm, width in widths.items() if width != widest}
    if narrow or repeats:
        msg = f"arm widths differ from {widest}: {narrow}; arms repeating a column: {repeats}"
        raise ValueError(msg)


def check_settings(*, job_list: list[Job], max_workers: int = MAX_WORKERS) -> None:
    """Stop unless the fits will run on the CPU, with no column subsampling, on at most `MAX_CORES`.

    `run_all` passes no device to `out_of_fold_losses`, so the fits use that function's default
    device, which this reads. Each job's XGBoost parameters are built by `booster_parameters`, the
    function `out_of_fold_losses` calls, from the job's own hyperparameters.

    Args:
        job_list: Every fit the run will make.
        max_workers: How many fits run at once.

    Raises:
        ValueError: Naming the setting that breaks the study's rules.
    """
    device = inspect.signature(out_of_fold_losses).parameters["device"].default
    if device != "cpu":
        msg = f"every fit must use device='cpu', but out_of_fold_losses defaults to {device!r}"
        raise ValueError(msg)
    for arm, setting, _target, _features, hyper_parameters, _quantiles in job_list:
        parameters = booster_parameters(hyper_parameters=hyper_parameters, seed=0, device=device)
        subsampling = [name for name in parameters if name.startswith("colsample")]
        if subsampling:
            msg = (
                f"{arm} at {setting}: column subsampling must stay off, "
                f"but the fit sets {subsampling}"
            )
            raise ValueError(msg)
    if max_workers * THREADS_PER_FIT > MAX_CORES:
        msg = f"{max_workers} fits of {THREADS_PER_FIT} threads exceed {MAX_CORES} cores"
        raise ValueError(msg)


def with_shuffled_levels(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add `speed_100m_noise`'s four information-free columns.

    Args:
        frame: Rows sorted by site then time, carrying `month` and the five speed columns.

    Returns:
        The frame with `shuffled_column` for each of `SHUFFLED_HEIGHTS_M`, each shuffled separately
        within each farm.
    """
    columns: dict[str, np.ndarray] = {}
    for height in SHUFFLED_HEIGHTS_M:
        values = frame[f"wind_speed_{height}m"].to_numpy()
        months = frame["month"].to_numpy()
        sites = frame["site"].to_numpy()
        shuffled = np.empty_like(values)
        for site_index, site in enumerate(sorted(set(sites))):
            rows = sites == site
            generator = np.random.default_rng((NOISE_SEED, height, site_index))
            shuffled[rows] = shuffled_by_month(
                values=values[rows], months=months[rows], generator=generator
            )
        columns[shuffled_column(height_m=height)] = shuffled
    return frame.with_columns(**{name: pl.Series(name, values) for name, values in columns.items()})


def with_arm_columns(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the mean of the near-100 m levels and every arm's monotone transforms.

    Args:
        frame: Rows carrying the five speed columns.

    Returns:
        The frame with `MEAN_NEAR_100M`, and `TRANSFORMS` of `SPEED_10M`, `SPEED_100M` and
        `MEAN_NEAR_100M`.
    """
    frame = frame.with_columns(
        pl.mean_horizontal(SPEED_75M, SPEED_100M, SPEED_150M).alias(MEAN_NEAR_100M)
    )
    transforms = {
        "squared": lambda column: pl.col(column).pow(2),
        "sqrt": lambda column: pl.col(column).sqrt(),
        "log1p": lambda column: pl.col(column).log1p(),
        "cubed": lambda column: pl.col(column).pow(3),
    }
    return frame.with_columns(
        transforms[name](base).alias(f"{base}_{name}")
        for base in (SPEED_10M, SPEED_100M, MEAN_NEAR_100M)
        for name in TRANSFORMS
    )


def with_synthetic_target(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the positive control's target.

    The target is a fixed power curve of CERRA's speed interpolated to `INTERPOLATION_HEIGHT_M`,
    linear in log-height between the 100 and 150 m levels, plus Gaussian noise, in megawatts. An arm
    has to combine the 100 and 150 m speeds to reproduce it.

    Args:
        frame: Rows carrying the 100 and 150 m speeds and `effective_capacity_mw`.

    Returns:
        The frame with `SYNTHETIC_TARGET`.
    """
    weight = float(np.log(INTERPOLATION_HEIGHT_M / 100.0) / np.log(150.0 / 100.0))
    speed = pl.col(SPEED_100M) + weight * (pl.col(SPEED_150M) - pl.col(SPEED_100M))
    fraction = ((speed - CUT_IN_M_S) / (RATED_M_S - CUT_IN_M_S)).clip(0.0, 1.0).pow(3)
    noise = np.random.default_rng(NOISE_SEED).normal(0.0, SYNTHETIC_NOISE_FRACTION, frame.height)
    return frame.with_columns(
        ((fraction + pl.Series(noise)).clip(0.0, 1.0) * pl.col("effective_capacity_mw")).alias(
            SYNTHETIC_TARGET
        )
    )


def read_half_hourly_power(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Read each wind farm's half-hourly power.

    Args:
        sites: The wind roster, with `time_series_id` and `site`.

    Returns:
        One row per (site, time) with `power_mw`, at the timestamps the power table holds.
    """
    return (
        pl.scan_delta(POWER_DELTA_URI)
        .filter(pl.col("time_series_id").is_in(sites["time_series_id"].to_list()))
        .collect()
        .join(sites.select("time_series_id", "site"), on="time_series_id")
        .select("site", "time", power_mw=pl.col("power"))
    )


def read_wind(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Read CERRA's five wind speeds at each wind farm's nearest cell.

    Args:
        sites: The wind roster, with `site`, `latitude` and `longitude`.

    Returns:
        The frame `studies.reanalysis_wind.read_cerra_wind` returns. The loader stops the run if a
        farm's cell is not strictly inside a file's crop.
    """
    cells = derive_nearest_cells(grid=pl.read_parquet(GRID_PATH), sites=sites)
    _LOG.info(
        "nearest cells: %d farms in %d cells, distance %.1f to %.1f km",
        cells.height,
        cells.select("y_index", "x_index").unique().height,
        cells["distance_km"].min(),
        cells["distance_km"].max(),
    )
    return read_cerra_wind(directory=CERRA_DIR, cells=cells)


def monthly_steps(*, values: np.ndarray, months: list[str]) -> tuple[dict[str, float], float]:
    """Compute the step at every month of a deseasonalised monthly series.

    Each value has its calendar month's mean removed. A month's step is the mean of the next
    `ERA_WINDOW_MONTHS` residuals minus the mean of the previous `ERA_WINDOW_MONTHS`.

    Args:
        values: One value per month, in month order.
        months: The `YYYY-MM` label of each value.

    Returns:
        Each month's step, for the months with a full window either side, and the standard error
        that independent months would give a step.
    """
    calendar = np.array([int(month[5:]) for month in months])
    residual = values.copy()
    for number in np.unique(calendar):
        residual[calendar == number] -= values[calendar == number].mean()
    standard_error = float(residual.std()) * np.sqrt(2.0 / ERA_WINDOW_MONTHS)
    steps = {
        months[index]: float(
            residual[index : index + ERA_WINDOW_MONTHS].mean()
            - residual[index - ERA_WINDOW_MONTHS : index].mean()
        )
        for index in range(ERA_WINDOW_MONTHS, len(residual) - ERA_WINDOW_MONTHS + 1)
    }
    return steps, standard_error


def _largest_step(*, values: np.ndarray, months: list[str]) -> tuple[float, float, str]:
    """Find the month with the largest step in a deseasonalised monthly series.

    Args:
        values: One value per month, in month order.
        months: The `YYYY-MM` label of each value.

    Returns:
        The largest step, its z-score, and its month.
    """
    steps, standard_error = monthly_steps(values=values, months=months)
    month = max(steps, key=lambda key: abs(steps[key]))
    return steps[month], steps[month] / standard_error, month


def era_step_table(*, wind: pl.DataFrame) -> pl.DataFrame:
    """Find the largest step, at any month, in each height's ratio to 100 m and in its own level.

    A join of two production streams in CERRA's record would move a height's mean speed, or its
    ratio to the 100 m speed, at one month. The ratio test removes weather common to every height,
    so it is the sharper test. The level test uses each height's own monthly mean speed, and a
    weather anomaly shows up as a step of the same sign at every height. Reading CERRA's
    documentation for the date of a production-stream join is a manual step this function does not
    do.

    Args:
        wind: `read_cerra_wind`'s frame.

    Returns:
        One row per height and test, with `test` (`ratio` or `level`), `height`, `step_percent`
        (of the series' mean), `z`, `month` (of the largest |z|), and `n_months`. The ratio test
        has no row for 100 m.
    """
    monthly = (
        wind.with_columns(month=pl.col("time").dt.strftime("%Y-%m"))
        .group_by("month")
        .agg(pl.col(SPEED_COLUMNS).mean())
        .sort("month")
    )
    months = monthly["month"].to_list()
    rows: list[dict[str, object]] = []
    for test in ("ratio", "level"):
        for column in SPEED_COLUMNS:
            if test == "ratio" and column == SPEED_100M:
                continue
            series = monthly[column].to_numpy()
            if test == "ratio":
                series = series / monthly[SPEED_100M].to_numpy()
            step, z, month = _largest_step(values=series, months=months)
            rows.append(
                {
                    "test": test,
                    "height": column.removeprefix("wind_speed_"),
                    "step_percent": 100.0 * step / float(series.mean()),
                    "z": z,
                    "month": month,
                    "n_months": len(months),
                }
            )
    return pl.DataFrame(rows)


def check_no_era(*, table: pl.DataFrame) -> None:
    """Stop if any height's ratio to 100 m or own level steps by more than `ERA_GATE_Z` errors.

    Args:
        table: `era_step_table`'s result.

    Raises:
        ValueError: Naming the tests, heights and months where a step was found. The folds then
            have to be cut inside each era, and every arm given an era column, before any fit.
    """
    found = table.filter(pl.col("z").abs() > ERA_GATE_Z)
    if found.height:
        msg = (
            f"a step in a height's ratio to 100 m or level exceeds {ERA_GATE_Z} standard errors: "
            f"{found.select('test', 'height', 'month', 'z').to_dicts()}; cut the folds by era and "
            "add an era column to every arm before fitting"
        )
        raise ValueError(msg)


def _joined_rows(
    *, wind: pl.DataFrame, half_hourly: pl.DataFrame, sites: pl.DataFrame, shift: int
) -> pl.DataFrame:
    """Join wind to power at one power-hour offset and drop hours holding a zero half-hour.

    Args:
        wind: `read_wind`'s frame.
        half_hourly: `read_half_hourly_power`'s frame.
        sites: The wind roster, with `site` and `effective_capacity_mw`.
        shift: The power hour's offset in half-hours.

    Returns:
        One row per farm-hour with a wind value and a power hour, sorted by site then time, with the
        calendar features and the `constrained` and `cap_mw` columns the fit loop reads.
    """
    if shift == CENTRED_SHIFT:
        joined = join_centred_power(wind=wind, half_hourly=half_hourly)
    else:
        hourly = hourly_from_half_hourly(
            half_hourly=half_hourly.with_columns(pl.col("time").dt.offset_by(f"{-30 * shift}m"))
        )
        joined = wind.join(hourly, on=["site", "time"], how="inner").sort("site", "time")
    return (
        joined.join(sites.select("site", "effective_capacity_mw"), on="site")
        .filter(~pl.col("has_zero_half_hour"))
        .with_columns(constrained=pl.lit(value=False), cap_mw=pl.lit(None, dtype=pl.Float64))
        .pipe(lambda rows: add_time_features(dataset=rows))
        .sort("site", "time")
    )


def build_rows(
    *,
    wind: pl.DataFrame,
    half_hourly: pl.DataFrame,
    sites: pl.DataFrame,
    shift: int = CENTRED_SHIFT,
    shared_keys: pl.DataFrame | None = None,
    with_controls: bool = True,
) -> pl.DataFrame:
    """Build the frame the fit loop reads, for one power-hour offset.

    Args:
        wind: `read_wind`'s frame.
        half_hourly: `read_half_hourly_power`'s frame.
        sites: The wind roster, with `site` and `effective_capacity_mw`.
        shift: The power hour's offset in half-hours; `CENTRED_SHIFT` for the main row set.
        shared_keys: If given, only the (site, time) rows it holds are kept, before folds are cut.
        with_controls: Whether to add the two controls' columns, the shuffled levels and the
            synthetic target. The power-hour scan needs neither.

    Returns:
        One row per farm-hour with a wind value and a power hour, minus every hour holding an
        exactly-zero half-hour, sorted by site then time. The frame carries the wind columns, every
        arm's derived columns, `power_mw`, `effective_capacity_mw`, the calendar features, `month`,
        `fold`, the controls' columns if asked for, and the `constrained` and `cap_mw` columns the
        fit loop reads. NGED has confirmed that no wind farm in the trial area is under active
        network management.

    Raises:
        ValueError: If a column an arm reads holds a missing value, or a calendar month has no
            training row in a fold.
    """
    frame = _joined_rows(wind=wind, half_hourly=half_hourly, sites=sites, shift=shift)
    if shared_keys is not None:
        frame = frame.join(shared_keys, on=["site", "time"], how="semi")
    frame = with_arm_columns(frame=frame)
    if with_controls:
        frame = with_synthetic_target(frame=with_shuffled_levels(frame=frame))
    frame = assign_folds(dataset=frame)
    arms = arm_columns()
    if not with_controls:
        arms.pop(NEGATIVE_CONTROL_ARM)
    check_no_missing(
        frame=frame,
        columns=(
            "power_mw",
            "effective_capacity_mw",
            *([SYNTHETIC_TARGET] if with_controls else []),
            *{name for columns in arms.values() for name in columns},
        ),
    )
    raise_on_uncovered_months(coverage=calendar_month_coverage(frame=frame))
    return frame


def jobs() -> list[Job]:
    """Return every fit: the arms at both settings, the negative control, and the positive control.

    Returns:
        The real-target arms and `NEGATIVE_CONTROL_ARM` at `PRIMARY_SETTING` and at
        `SENSITIVITY_SETTING`, and `POSITIVE_CONTROL_ARMS` on `SYNTHETIC_TARGET` at
        `POSITIVE_CONTROL_SETTING`.
    """
    columns = arm_columns()
    arms = (*REAL_ARMS, NEGATIVE_CONTROL_ARM)
    fits: list[Job] = []
    for setting, hyper_parameters in (
        (PRIMARY_SETTING, PRIMARY_HYPER_PARAMETERS),
        (SENSITIVITY_SETTING, SENSITIVITY_HYPER_PARAMETERS),
    ):
        fits += [(arm, setting, "power_mw", columns[arm], hyper_parameters, False) for arm in arms]
    fits += [
        (
            arm,
            POSITIVE_CONTROL_SETTING,
            SYNTHETIC_TARGET,
            columns[arm],
            PRIMARY_HYPER_PARAMETERS,
            False,
        )
        for arm in POSITIVE_CONTROL_ARMS
    ]
    return fits


def power_hour_scan(
    *, wind: pl.DataFrame, half_hourly: pl.DataFrame, sites: pl.DataFrame
) -> pl.DataFrame:
    """Fit the `speed_100m` arm at each power-hour offset, scored on the rows all offsets share.

    Args:
        wind: `read_wind`'s frame.
        half_hourly: `read_half_hourly_power`'s frame.
        sites: The wind roster.

    Returns:
        One row per offset with `shift`, `n_rows`, `mae_pp` (the mean absolute error as a
        percentage of each farm's capacity) and `seed_spread_pp` (the largest minus the smallest of
        the three seeds' errors). Every offset is fitted and scored on the same (site, time) rows,
        with folds cut after the rows are chosen.
    """
    keys = [
        _joined_rows(wind=wind, half_hourly=half_hourly, sites=sites, shift=shift).select(
            "site", "time"
        )
        for shift in POWER_HOUR_SHIFTS
    ]
    shared = keys[0]
    for other in keys[1:]:
        shared = shared.join(other, on=["site", "time"], how="inner")
    columns = arm_columns()["speed_100m"]
    rows: list[dict[str, float | int]] = []
    for shift in POWER_HOUR_SHIFTS:
        frame = build_rows(
            wind=wind,
            half_hourly=half_hourly,
            sites=sites,
            shift=shift,
            shared_keys=shared,
            with_controls=False,
        )
        losses = run_all(
            dataset=frame,
            jobs=[("speed_100m", "scan", "power_mw", columns, PRIMARY_HYPER_PARAMETERS, False)],
            max_workers=MAX_WORKERS,
        )
        per_seed = losses.group_by("seed").agg(pl.col(METRIC).mean())[METRIC]
        rows.append(
            {
                "shift": shift,
                "n_rows": frame.height,
                "mae_pp": _mae(losses=losses, arm="speed_100m"),
                "seed_spread_pp": float(per_seed.to_numpy().max() - per_seed.to_numpy().min())
                * PERCENTAGE_POINTS,
            }
        )
        _LOG.info("power-hour offset %d half-hours: %s", shift, rows[-1])
    return pl.DataFrame(rows)


def check_centred_is_best(*, scan: pl.DataFrame) -> None:
    """Stop if another power-hour offset beats the centred one by more than its seed spread.

    Args:
        scan: `power_hour_scan`'s result.

    Raises:
        ValueError: Naming the offsets that beat the centred offset by more than the centred
            offset's `seed_spread_pp`.
    """
    centred = scan.filter(pl.col("shift") == CENTRED_SHIFT).row(0, named=True)
    better = scan.filter(pl.col("mae_pp") < centred["mae_pp"] - centred["seed_spread_pp"])
    if better.height:
        msg = (
            f"the power-hour scan scores {better.select('shift', 'mae_pp').to_dicts()} better than "
            f"the centred offset {CENTRED_SHIFT} ({centred['mae_pp']:.4f} pp) by more than "
            f"its seed spread of {centred['seed_spread_pp']:.4f} pp"
        )
        raise ValueError(msg)


def check_same_rows(*, losses: pl.DataFrame) -> None:
    """Stop unless every arm of every setting was scored on the same (site, time, seed) rows.

    Args:
        losses: Every arm's losses.

    Raises:
        ValueError: If two arms of one target's fits differ in rows.
    """
    for target in losses["target"].unique().to_list():
        keys = {
            (arm, setting): frozenset(group.select("site", "time", "seed").iter_rows())
            for (arm, setting), group in losses.filter(pl.col("target") == target)
            .partition_by("arm", "setting", as_dict=True)
            .items()
        }
        if len(set(keys.values())) != 1:
            msg = f"arms of target {target!r} were scored on different rows"
            raise ValueError(msg)


def interval_record(
    *, losses: pl.DataFrame, contrast: Contrast, setting: str, scope: str
) -> dict[str, object]:
    """Compute one contrast's interval and both arms' errors on one scope of rows.

    Args:
        losses: One setting's per-row losses for both arms, restricted to the scope.
        contrast: The pairing.
        setting: The setting the losses came from.
        scope: A label for the rows: `all`, a farm label, or a half of the year.

    Returns:
        A record with the differences in percentage points of capacity, the interval, the fold
        signs, and both arms' absolute errors.
    """
    interval = bootstrap_difference(
        losses=losses, treatment=contrast.treatment, reference=contrast.reference, metric=METRIC
    )
    folds = per_fold_differences(
        losses=losses, treatment=contrast.treatment, reference=contrast.reference, metric=METRIC
    )
    same_sign = sum(np.sign(value) == np.sign(interval["difference"]) for value in folds)
    wide: tuple[float, float] | tuple[None, None] = (None, None)
    if contrast.planned:
        wide = bootstrap_difference_at_level(
            losses=losses,
            treatment=contrast.treatment,
            reference=contrast.reference,
            metric=METRIC,
            level=BONFERRONI_LEVEL,
        )
    return {
        "setting": setting,
        "scope": scope,
        "treatment": contrast.treatment,
        "reference": contrast.reference,
        "planned": contrast.planned,
        "treatment_mae_pp": _mae(losses=losses, arm=contrast.treatment),
        "reference_mae_pp": _mae(losses=losses, arm=contrast.reference),
        "difference_pp": interval["difference"] * PERCENTAGE_POINTS,
        "lower_95_pp": interval["lower_95"] * PERCENTAGE_POINTS,
        "upper_95_pp": interval["upper_95"] * PERCENTAGE_POINTS,
        "significant": interval["lower_95"] > 0.0 or interval["upper_95"] < 0.0,
        "lower_bonferroni_pp": None if wide[0] is None else wide[0] * PERCENTAGE_POINTS,
        "upper_bonferroni_pp": None if wide[1] is None else wide[1] * PERCENTAGE_POINTS,
        "folds_agreeing": int(same_sign),
        "n_folds": len(folds),
        "n_rows": interval["n_rows"],
        "n_months": interval["n_months"],
        "seed_spread_pp": interval["seed_spread"] * PERCENTAGE_POINTS,
    }


def _half_of_year(*, losses: pl.DataFrame, half: str) -> pl.DataFrame:
    """Restrict losses to October to March (`winter`) or April to September (`summer`).

    Args:
        losses: Per-row losses carrying `time`.
        half: `winter` or `summer`.

    Returns:
        The rows in that half.
    """
    month = pl.col("time").dt.month()
    return losses.filter(
        (month >= 10) | (month <= 3) if half == "winter" else month.is_between(4, 9)
    )


def contrast_records(*, losses: pl.DataFrame, sites: list[str]) -> pl.DataFrame:
    """Compute every interval the report and the charts quote.

    Args:
        losses: Every arm's losses, every setting.
        sites: The farm labels.

    Returns:
        One row per (setting, scope, contrast). The planned contrasts run at both real-target
        settings and, at the primary setting, per farm and per half of the year (exploratory
        splits). The exploratory contrasts and the negative control run at both real-target
        settings on all rows. The positive control runs on the synthetic target.
    """
    by_setting = {
        setting: losses.filter(pl.col("setting") == setting)
        for setting in (PRIMARY_SETTING, SENSITIVITY_SETTING, POSITIVE_CONTROL_SETTING)
    }
    records = []
    for setting in (PRIMARY_SETTING, SENSITIVITY_SETTING):
        records += [
            interval_record(
                losses=by_setting[setting], contrast=contrast, setting=setting, scope="all"
            )
            for contrast in (*PLANNED_CONTRASTS, *EXPLORATORY_CONTRASTS, NEGATIVE_CONTROL)
        ]
    primary = by_setting[PRIMARY_SETTING]
    for contrast in PLANNED_CONTRASTS:
        splits = [(site, primary.filter(pl.col("site") == site)) for site in sites]
        splits += [
            (half, _half_of_year(losses=primary, half=half)) for half in ("winter", "summer")
        ]
        records += [
            interval_record(
                losses=scoped,
                contrast=contrast._replace(planned=False),
                setting=PRIMARY_SETTING,
                scope=scope,
            )
            for scope, scoped in splits
        ]
    records.append(
        interval_record(
            losses=by_setting[POSITIVE_CONTROL_SETTING],
            contrast=POSITIVE_CONTROL,
            setting=POSITIVE_CONTROL_SETTING,
            scope="all",
        )
    )
    return pl.DataFrame(records, infer_schema_length=None)


def absolute_records(*, losses: pl.DataFrame) -> pl.DataFrame:
    """Compute every arm's absolute error and interval, at each setting and on each farm.

    Args:
        losses: Every arm's losses, every setting.

    Returns:
        One row per (setting, scope, arm) with the error and its 95% interval in percentage points
        of capacity.
    """
    records = []
    sites = sorted(losses["site"].unique().to_list())
    for setting in (PRIMARY_SETTING, SENSITIVITY_SETTING, POSITIVE_CONTROL_SETTING):
        in_setting = losses.filter(pl.col("setting") == setting)
        for scope, scoped in (
            ("all", in_setting),
            *((site, in_setting.filter(pl.col("site") == site)) for site in sites),
        ):
            for arm in sorted(scoped["arm"].unique().to_list()):
                interval = bootstrap_absolute(losses=scoped, arm=arm, metric=METRIC)
                records.append(
                    {
                        "setting": setting,
                        "scope": scope,
                        "arm": arm,
                        "mae_pp": interval["value"] * PERCENTAGE_POINTS,
                        "lower_95_pp": interval["lower_95"] * PERCENTAGE_POINTS,
                        "upper_95_pp": interval["upper_95"] * PERCENTAGE_POINTS,
                        "n_rows": interval["n_rows"],
                        "n_months": interval["n_months"],
                    }
                )
    return pl.DataFrame(records)


CONTRAST_TABLE_HEADER: Final[tuple[str, str]] = (
    (
        "| Setting, scope | Contrast | Treatment error | Reference error "
        "| Difference (pp of capacity) "
        "| Unadjusted 95% interval | Statistically significant at the 5% level, unadjusted? "
        f"| {BONFERRONI_LEVEL}% interval (Bonferroni, four planned contrasts) "
        "| Statistically significant after Bonferroni? | Folds agreeing | Rows |"
    ),
    "|---|---|---|---|---|---|---|---|---|---|---|",
)


def _bonferroni_text(*, row: dict[str, Any]) -> str:
    """Render a contrast row's Bonferroni interval, or a dash for a row that has none.

    Args:
        row: A row of `contrast_records`.

    Returns:
        The interval as text.
    """
    lower, upper = row["lower_bonferroni_pp"], row["upper_bonferroni_pp"]
    if lower is None or upper is None:
        return "n/a"
    return f"[{lower:+.3f}, {upper:+.3f}]"


def _bonferroni_verdict(*, row: dict[str, Any]) -> str:
    """Say whether a row's Bonferroni interval is statistically significant at the 5% level.

    Args:
        row: A row of `contrast_records`.

    Returns:
        `**yes**`, `no`, or `n/a` for a row with no Bonferroni interval.
    """
    lower, upper = row["lower_bonferroni_pp"], row["upper_bonferroni_pp"]
    if lower is None or upper is None:
        return "n/a"
    return "**yes**" if lower > 0.0 or upper < 0.0 else "no"


def _contrast_lines(*, records: pl.DataFrame) -> list[str]:
    """Render contrast records as a markdown table.

    Args:
        records: Rows of `contrast_records`.

    Returns:
        Markdown lines, header included.
    """
    lines = [*CONTRAST_TABLE_HEADER]
    lines += [
        (
            f"| {row['setting']}, {row['scope']} | {row['treatment']} − {row['reference']} "
            f"| {row['treatment_mae_pp']:.3f} | {row['reference_mae_pp']:.3f} "
            f"| {row['difference_pp']:+.3f} "
            f"| [{row['lower_95_pp']:+.3f}, {row['upper_95_pp']:+.3f}] "
            f"| {'**yes**' if row['significant'] else 'no'} "
            f"| {_bonferroni_text(row=row)} "
            f"| {_bonferroni_verdict(row=row)} "
            f"| {row['folds_agreeing']} of {row['n_folds']} | {row['n_rows']:,} |"
        )
        for row in records.iter_rows(named=True)
    ]
    return lines


def _row_count_lines(*, frame: pl.DataFrame) -> list[str]:
    """Render the number of rows per farm and year.

    Args:
        frame: The main row set.

    Returns:
        Markdown lines.
    """
    counts = (
        frame.group_by("site", year=pl.col("time").dt.year())
        .agg(rows=pl.len())
        .pivot(on="year", index="site", values="rows")
        .sort("site")
    )
    years = sorted(name for name in counts.columns if name != "site")
    lines = ["| Farm | " + " | ".join(years) + " |", "|---" * (len(years) + 1) + "|"]
    lines += [
        f"| {row['site']} | " + " | ".join(str(row[year] or 0) for year in years) + " |"
        for row in counts.iter_rows(named=True)
    ]
    return lines


ALL_ARMS: Final[tuple[str, ...]] = (*REAL_ARMS, NEGATIVE_CONTROL_ARM)
"""Every arm fitted on the real target."""


def _absolute_value(*, absolute: pl.DataFrame, setting: str, scope: str, arm: str) -> float:
    """Read one arm's absolute error from `absolute_records`.

    Args:
        absolute: `absolute_records`' frame.
        setting: The setting.
        scope: `all` or a farm label.
        arm: The arm.

    Returns:
        The error in percentage points of capacity.
    """
    return float(
        absolute.filter(
            (pl.col("setting") == setting) & (pl.col("scope") == scope) & (pl.col("arm") == arm)
        )["mae_pp"].item()
    )


def _absolute_lines(*, absolute: pl.DataFrame, setting: str, arms: tuple[str, ...]) -> list[str]:
    """Render each arm's pooled absolute error and 95% interval as a markdown table.

    Args:
        absolute: `absolute_records`' frame.
        setting: The setting.
        arms: The arms to list, in order.

    Returns:
        Markdown lines, header included.
    """
    pooled = absolute.filter((pl.col("setting") == setting) & (pl.col("scope") == "all"))
    lines = ["| Arm | Error (pp of capacity) | 95% interval | Rows |", "|---|---|---|---|"]
    for arm in arms:
        row = pooled.filter(pl.col("arm") == arm).row(0, named=True)
        lines.append(
            f"| {arm} | {row['mae_pp']:.3f} | [{row['lower_95_pp']:.3f}, {row['upper_95_pp']:.3f}] "
            f"| {row['n_rows']:,} |"
        )
    return lines


def report_lines(
    *,
    frame: pl.DataFrame,
    intervals: pl.DataFrame,
    absolute: pl.DataFrame,
    era: pl.DataFrame,
    scan: pl.DataFrame,
    job_list: list[Job],
) -> list[str]:
    """Assemble the report the page quotes.

    Args:
        frame: The main row set.
        intervals: `contrast_records`' frame.
        absolute: `absolute_records`' frame.
        era: `era_step_table`'s frame.
        scan: `power_hour_scan`'s frame.
        job_list: Every fit.

    Returns:
        Markdown lines.
    """
    sites = sorted(frame["site"].unique().to_list())
    parameters = booster_parameters(hyper_parameters=PRIMARY_HYPER_PARAMETERS, seed=0)
    lines = [
        (
            f"### CERRA wind levels on {frame.height:,} farm-hours of wind "
            f"({frame['time'].min():%Y-%m-%d} to {frame['time'].max():%Y-%m-%d})"
        ),
        "",
        (
            f"Every fit ran on device `{parameters['device']}` with no column subsampling, on "
            f"{MAX_WORKERS} fits of {THREADS_PER_FIT} threads at a time. Errors are means of each "
            "row's absolute error over its farm's capacity, in percentage points, on 3-hourly rows "
            f"(`hour_of_day` takes {frame['hour_of_day'].n_unique()} values). "
            f"The row set is the {frame['month'].n_unique()} calendar months from the first to the "
            "last row, with every hour holding an exactly-zero half-hour dropped."
        ),
        "",
        "#### Rows per farm and year",
        "",
        *_row_count_lines(frame=frame),
        "",
        "#### Era check (gate before any fit)",
        "",
        (
            f"The largest step, at any month, in each height's ratio to 100 m (`ratio`) and in its "
            f"own monthly mean speed (`level`), as the mean of the next {ERA_WINDOW_MONTHS} months "
            f"minus the mean of the previous {ERA_WINDOW_MONTHS} after removing each calendar "
            f"month's mean. The gate is |z| above {ERA_GATE_Z}. A weather anomaly moves every "
            "height's level in the same direction, and a production-stream join would not. "
            "Reading CERRA's documentation for the date of a production-stream join is a manual "
            "step that this script does not do."
        ),
        "",
        "| Test | Height | Largest step (% of mean) | z | Month |",
        "|---|---|---|---|---|",
        *(
            f"| {row['test']} | {row['height']} | {row['step_percent']:+.2f} | {row['z']:+.2f} "
            f"| {row['month']} |"
            for row in era.iter_rows(named=True)
        ),
        "",
        "#### Power-hour offset scan (gate before the main fits)",
        "",
        (
            f"The `speed_100m` arm, fitted with the power hour built from the half-hours ending at "
            f"the label plus `shift` half-hours less 30 minutes and at the label plus `shift` "
            f"half-hours, on the rows every offset shares. `shift` = {CENTRED_SHIFT} is the "
            "centred hour used everywhere else. The run stops if another offset beats the centred "
            "one by more than the centred offset's seed spread."
        ),
        "",
        "| Shift (half-hours) | Rows | Error (pp) | Seed spread (pp) |",
        "|---|---|---|---|",
        *(
            f"| {row['shift']} | {row['n_rows']:,} | {row['mae_pp']:.4f} "
            f"| {row['seed_spread_pp']:.4f} |"
            for row in scan.sort("shift").iter_rows(named=True)
        ),
        "",
        *_arm_columns_lines(job_list=job_list),
        "",
        "#### Every arm's absolute error, primary setting",
        "",
        *_absolute_lines(absolute=absolute, setting=PRIMARY_SETTING, arms=ALL_ARMS),
        "",
        "| Arm | " + " | ".join(sites) + " | Second setting |",
        "|---" * (len(sites) + 2) + "|",
    ]
    for arm in ALL_ARMS:

        def _farm_value(*, site: str, arm: str = arm) -> float:
            return _absolute_value(absolute=absolute, setting=PRIMARY_SETTING, scope=site, arm=arm)

        per_farm = " | ".join(f"{_farm_value(site=site):.3f}" for site in sites)
        second = _absolute_value(
            absolute=absolute, setting=SENSITIVITY_SETTING, scope="all", arm=arm
        )
        lines.append(f"| {arm} | {per_farm} | {second:.3f} |")
    lines += ["", "#### Planned contrasts, primary setting", ""]
    lines += _contrast_lines(
        records=intervals.filter(
            (pl.col("setting") == PRIMARY_SETTING) & pl.col("planned") & (pl.col("scope") == "all")
        )
    )
    lines += ["", "#### Planned contrasts, second setting", ""]
    lines += _contrast_lines(
        records=intervals.filter(
            (pl.col("setting") == SENSITIVITY_SETTING)
            & pl.col("planned")
            & (pl.col("scope") == "all")
        )
    )
    lines += ["", "#### Exploratory contrasts, both settings", ""]
    lines += _contrast_lines(
        records=intervals.filter(
            ~pl.col("planned")
            & (pl.col("scope") == "all")
            & (pl.col("setting") != POSITIVE_CONTROL_SETTING)
            & (pl.col("treatment") != NEGATIVE_CONTROL_ARM)
        ).sort("setting", maintain_order=True)
    )
    lines += ["", "#### Planned contrasts by farm and half of the year (exploratory splits)", ""]
    lines += _contrast_lines(
        records=intervals.filter(pl.col("scope") != "all").filter(
            pl.col("setting") == PRIMARY_SETTING
        )
    )
    lines += ["", "#### Negative control, both settings", ""]
    lines += _contrast_lines(records=intervals.filter(pl.col("treatment") == NEGATIVE_CONTROL_ARM))
    lines += ["", "#### Positive control (synthetic target)", ""]
    lines += _absolute_lines(
        absolute=absolute, setting=POSITIVE_CONTROL_SETTING, arms=POSITIVE_CONTROL_ARMS
    )
    lines += [""]
    lines += _contrast_lines(
        records=intervals.filter(pl.col("setting") == POSITIVE_CONTROL_SETTING)
    )
    lines += [""]
    return lines


def check_positive_control(*, intervals: pl.DataFrame) -> None:
    """Stop unless the blend beats 100 m on the synthetic target by a margin the interval excludes.

    Args:
        intervals: `contrast_records`' frame.

    Raises:
        ValueError: If the positive control's upper 95% bound is not below zero.
    """
    row = intervals.filter(pl.col("setting") == POSITIVE_CONTROL_SETTING).row(0, named=True)
    if not row["upper_95_pp"] < 0.0:
        msg = (
            "the positive control failed: levels_50_to_150 minus speed_100m on the synthetic "
            "target "
            f"is {row['difference_pp']:+.3f} pp, 95% interval [{row['lower_95_pp']:+.3f}, "
            f"{row['upper_95_pp']:+.3f}]; a null result on the real target "
            "cannot be read as no effect"
        )
        raise ValueError(msg)


def main() -> int:
    """Run the gates, fit every arm, bootstrap every contrast, and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Fit nothing; rebuild the report and intervals from the saved losses.",
    )
    arguments = parser.parse_args()

    job_list = jobs()
    check_settings(job_list=job_list)
    check_column_counts(arms=arm_columns())
    sites = wind_sites()
    wind = read_wind(sites=sites)
    half_hourly = read_half_hourly_power(sites=sites)

    paths = {
        name: OUTPUT_DIR / name
        for name in (
            "losses.parquet",
            "losses.fingerprint",
            "rows.parquet",
            "era_check.parquet",
            "power_hour_scan.parquet",
            "intervals.parquet",
            "absolute.parquet",
            "report.md",
        )
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    era = era_step_table(wind=wind)
    check_no_era(table=era)

    frame = build_rows(wind=wind, half_hourly=half_hourly, sites=sites)
    _LOG.info(
        "%d rows, %s to %s\n%s",
        frame.height,
        frame["time"].min(),
        frame["time"].max(),
        frame.group_by("site").agg(pl.len(), pl.col("month").n_unique()).sort("site"),
    )
    fingerprint = _fingerprint(frame=frame, job_list=job_list)

    if arguments.report_only:
        saved = paths["losses.fingerprint"].read_text().strip()
        if saved != fingerprint:
            msg = (
                "--report-only: the saved losses were fitted on a different row set, column set, "
                "seed set, or hyperparameter setting than this code now produces"
            )
            raise ValueError(msg)
        losses = pl.read_parquet(paths["losses.parquet"])
        scan = pl.read_parquet(paths["power_hour_scan.parquet"])
        refuse_to_overwrite(
            paths=[paths["report.md"], paths["intervals.parquet"], paths["absolute.parquet"]]
        )
    else:
        refuse_to_overwrite(paths=list(paths.values()))
        scan = power_hour_scan(wind=wind, half_hourly=half_hourly, sites=sites)
        check_centred_is_best(scan=scan)
        scan.write_parquet(paths["power_hour_scan.parquet"])
        losses = run_all(dataset=frame, jobs=job_list, max_workers=MAX_WORKERS)
        check_same_rows(losses=losses)
        losses.write_parquet(paths["losses.parquet"])
        paths["losses.fingerprint"].write_text(fingerprint)
        frame.write_parquet(paths["rows.parquet"])
        era.write_parquet(paths["era_check.parquet"])

    farms = sorted(frame["site"].unique().to_list())
    intervals = contrast_records(losses=losses, sites=farms)
    absolute = absolute_records(losses=losses)
    intervals.write_parquet(paths["intervals.parquet"])
    absolute.write_parquet(paths["absolute.parquet"])
    report = "\n".join(
        report_lines(
            frame=frame,
            intervals=intervals,
            absolute=absolute,
            era=era,
            scan=scan,
            job_list=job_list,
        )
    )
    paths["report.md"].write_text(report)
    sys.stdout.write(report)
    check_positive_control(intervals=intervals)
    return 0


if __name__ == "__main__":
    sys.exit(main())
