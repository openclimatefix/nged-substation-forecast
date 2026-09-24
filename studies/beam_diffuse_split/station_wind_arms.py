"""Score the nearest weather station's 10 m wind against gridded products for past wind.

One-off throwaway script, phase 2 of the past-wind study
<https://openclimatefix.github.io/nged-substation-forecast/studies/weather-products-for-past-wind/>.
The Met Office's MIDAS Open archive holds hourly 10 m wind from 18 stations in and around the trial
area. This script gives an XGBoost model per wind farm the nearest wind-reporting station's wind,
and compares it with ERA5's 10 m wind (S1), and a blend of UKV and the station with UKV padded to
the same number of columns (S2).

**Provenance.** The plan `plans/station-wind-arms.md`, which fixes the two planned contrasts S1 and
S2, the row rule and the controls, was committed (24870b53) before the first fit. The section
"Decisions made during implementation" in that plan was committed, with this script, before the
first fit as well. The exploratory items are the k=3 arm, the shear control, the August-to-December
restriction, and `ukv_station_wind - ukv_wind`.

**Privacy.** Nothing here writes which station serves which wind farm, a station's coordinates,
name or identifier, or a per-farm distance, row count, drop count or station coverage. The private
station metadata is read in memory, and every figure printed about stations, hours or distances is
pooled over farms. Per-farm tables carry errors only, labelled W1 to W3.

**Row set.** The page's own rows (`common_rows(joined(sites=sites))`) restricted to 2024-08-12 to
2025-12-31 (17 calendar months, the MIDAS Open download ending in December 2025), then to hours
where the farm's nearest eligible station has both a speed and a direction. Every arm is fitted on
exactly these rows, including every gridded product's wind arm, which is refitted here and never
read from a published loss. The window holds one UKV era, so `era_code` is constant and is kept so
that every arm carries the page's columns.

**Station rule.** `studies.midas.select_nearest_stations` with `k=1` and `min_coverage=0.9` of the
farm's required hours (the page's rows in the window). The rule reads no score and no target.

**Arms.** Every arm carries `SHARED_FEATURES` plus its wind columns:

- `station_wind`: the station's speed, sine and cosine of its direction (3 columns).
- `era5_10m_wind`: ERA5's 10 m speed and the sine and cosine of ERA5's 100 m direction (3 columns).
- `ukv_station_wind`: UKV's page columns plus the station's three (7 columns).
- `ukv_padded_wind`: UKV's page columns plus UKV's own 80 m speed, and sine and cosine of its 80 m
  direction (7 columns).
- `era5_wind`, `ukv_wind`, `icon_d2_wind`, `icon_eu_wind`, `icon_global_wind`: the page's arms.
- `station_k3_wind` (exploratory): the mean speed and the mean-wind-vector direction of the three
  nearest eligible stations.

Run it with `uv run python studies/beam_diffuse_split/station_wind_arms.py`. `--checks-only` runs
the pre-fit checks and stops. `--report-only` rebuilds `report.md` from the saved `losses.parquet`,
and raises if the saved fingerprint no longer matches. A fresh run raises on an uncommitted change
to this script and refuses to overwrite an output that exists.
"""

import argparse
import hashlib
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
import polars as pl
from build_dataset import _wind_sites
from run_experiment import Job, _add_time_features, run_all
from sources import STUDY_DATA_DIR, WEATHER_DATA_DIR
from studies.bootstrap import (
    BOOTSTRAP_SEED,
    N_BOOTSTRAP_RESAMPLES,
    bootstrap_absolute,
    bootstrap_difference,
    paired_differences,
    per_fold_differences,
)
from studies.cross_validation import (
    N_FOLDS,
    PRIMARY_HYPER_PARAMETERS,
    SEEDS,
    SENSITIVITY_HYPER_PARAMETERS,
    fit_one_fold,
)
from studies.guards import check_no_missing, refuse_to_overwrite
from studies.midas import read_hourly_weather, read_station_metadata, select_nearest_stations
from weather_products import METRIC, PERCENTAGE_POINTS, with_eras
from wind_products import (
    SHARED_FEATURES,
    UKV_80M_COLUMNS,
    _wind_columns,
    common_rows,
    joined,
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

OUTPUT_DIR_NAME: Final[str] = "station_wind_arms"
"""The results directory under `sources.STUDY_DATA_DIR / 'past_weather_v2'`."""

OUTPUT_DIR: Final[Path] = STUDY_DATA_DIR / "past_weather_v2" / OUTPUT_DIR_NAME
"""Where this study writes its outputs, and a `superseded/` folder for re-runs."""

MIDAS_DIR: Final[Path] = WEATHER_DATA_DIR / "MIDAS-OPEN"
"""The MIDAS Open download."""

HOURLY_WEATHER_PATH: Final[Path] = MIDAS_DIR / "uk_hourly_weather_obs.parquet"
"""The tidy hourly-weather parquet."""

STATION_METADATA_PATH: Final[Path] = (
    MIDAS_DIR
    / "_station_metadata"
    / "midas-open_uk-hourly-weather-obs_dv-202607_station-metadata.csv"
)
"""The private hourly-weather station metadata, read in memory only."""

CANDIDATE_STATIONS: Final[tuple[str, ...]] = (
    "00370",
    "00373",
    "00384",
    "00386",
    "00393",
    "00395",
    "00405",
    "00409",
    "00421",
    "00456",
    "00461",
    "00554",
    "00556",
    "00583",
    "00595",
    "16725",
    "61986",
    "62265",
)
"""The 18 hourly-weather stations that report wind. The 8 stations with one 09:00 return a day are
excluded, and the 12 stations that carry no wind are never candidates."""

WINDOW_START: Final[pl.Expr] = pl.datetime(2024, 8, 12, time_zone="UTC")
"""The first hour of the row set."""

WINDOW_END_EXCLUSIVE: Final[pl.Expr] = pl.datetime(2026, 1, 1, time_zone="UTC")
"""The row set ends on 2025-12-31, because the MIDAS Open download holds calendar years to 2025."""

MIN_COVERAGE: Final[float] = 0.9
"""The share of a farm's required hours a station must cover to be eligible."""

EXPECTED_UNIT_CODE: Final[int] = 4
"""The only wind-speed unit code (anemometer, knots) a row may carry."""

SHEAR_EXPONENT: Final[float] = 1.0 / 7.0
"""A common open-country power-law exponent, chosen a priori, used only by the shear control."""

SHEAR_HEIGHT_RATIO: Final[float] = 100.0 / 10.0
"""The ratio of 100 m to the station's 10 m height."""

STATION_COLUMNS: Final[tuple[str, str, str]] = ("station_speed", "station_sin", "station_cos")
"""The nearest station's wind, as a speed and the sine and cosine of its direction."""

K3_COLUMNS: Final[tuple[str, str, str]] = ("k3_speed", "k3_sin", "k3_cos")
"""The three nearest stations' wind: mean speed, and the direction of the mean wind vector."""

ERA5_10M_COLUMNS: Final[tuple[str, str, str]] = ("speed_10m_era5", "sin_100m_era5", "cos_100m_era5")
"""ERA5's 10 m speed and the sine and cosine of its 100 m direction (no 10 m direction is held)."""

UKV_PADDING_COLUMNS: Final[tuple[str, ...]] = UKV_80M_COLUMNS[:3]
"""UKV's served 80 m speed and direction only; `UKV_80M_COLUMNS` also holds the 10 m speed."""

PRODUCTS: Final[tuple[str, ...]] = ("era5", "ukv", "icon_d2", "icon_eu", "icon_global")
"""The gridded products whose page wind arm is refitted on the station rows."""

SENSITIVITY_ARMS: Final[tuple[str, ...]] = (
    "station_wind",
    "era5_10m_wind",
    "ukv_station_wind",
    "ukv_padded_wind",
    "ukv_wind",
)
"""The arms refitted at `SENSITIVITY_HYPER_PARAMETERS`."""

PLANNED_CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("S1", "station_wind", "era5_10m_wind"),
    ("S2", "ukv_station_wind", "ukv_padded_wind"),
)
"""The two planned contrasts: (name, first, second), the difference being first minus second."""

EXPLORATORY_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("ukv_station_wind", "ukv_wind"),
    ("station_k3_wind", "station_wind"),
)
"""Exploratory contrasts. The first compares 7 wind columns with 4, so it is not equal-count."""

BONFERRONI_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / (2 * len(PLANNED_CONTRASTS)))
"""The confidence level, in percent, adjusted for 4 planned intervals (2 contrasts, 2 settings)."""

AUGUST_TO_DECEMBER: Final[tuple[int, ...]] = (8, 9, 10, 11, 12)
"""The calendar months that occur in both years of the window."""

MS_PER_KNOT: Final[float] = 0.514444
"""Metres per second in one knot."""

WHOLE_KNOT_TOLERANCE_M_S: Final[float] = 0.01
"""How far a speed may lie from a whole number of knots and still count as one."""

MAX_DIRECTION_DEG: Final[float] = 360.0
"""The largest direction a row may carry; north is written as 360 and calm as 0."""

LAG_SCAN_HOURS: Final[tuple[int, ...]] = (-1, 0, 1)
"""Hour offsets scanned for the station-against-UKV speed correlation."""

CONTRAST_HEADER: Final[tuple[str, str]] = (
    (
        "| Scope | Contrast | ΔMAE (pp of capacity) | 95% interval | Excludes zero? "
        "| Folds agreeing | Rows | Months |"
    ),
    "|---|---|---|---|---|---|---|---|",
)
"""The header of every pooled contrast table."""

FARM_CONTRAST_HEADER: Final[tuple[str, str]] = (
    "| Farm | Contrast | ΔMAE (pp of capacity) | 95% interval | Excludes zero? | Folds agreeing |",
    "|---|---|---|---|---|---|",
)
"""The header of a per-farm contrast table, which carries no row count."""

POOLED_CAVEAT: Final[str] = (
    "Three wind farms are few independent sites; {rows:,} rows, {months} months."
)
"""The line printed under every pooled interval."""


# ---------------------------------------------------------------------------------------------
# Helpers that mirror PR #885's `ens_hres_past_wind.py`; replaced by imports after the rebase.
# ---------------------------------------------------------------------------------------------


class IntervalRecord(TypedDict):
    """One interval the report prints, saved to `intervals.parquet`."""

    section: str
    setting: str
    scope: str
    treatment: str
    reference: str | None
    value: float
    lower: float
    upper: float
    level: float
    n_rows: int | None
    n_months: int | None
    folds_agreeing: int | None
    n_folds: int | None


@dataclass
class IntervalLog:
    """Every interval the report prints, collected as the report is assembled.

    Mirrors PR #885's `IntervalLog`.
    """

    records: list[IntervalRecord] = field(default_factory=list)

    def frame(self) -> pl.DataFrame:
        """Return the collected intervals as one table.

        Returns:
            One row per interval printed, in the order printed.
        """
        return pl.DataFrame(
            self.records,
            schema={
                "section": pl.String,
                "setting": pl.String,
                "scope": pl.String,
                "treatment": pl.String,
                "reference": pl.String,
                "value": pl.Float64,
                "lower": pl.Float64,
                "upper": pl.Float64,
                "level": pl.Float64,
                "n_rows": pl.Int64,
                "n_months": pl.Int64,
                "folds_agreeing": pl.Int64,
                "n_folds": pl.Int64,
            },
        )


def _fingerprint(*, frame: pl.DataFrame, job_list: list[Job]) -> str:
    """Return a hash covering every row's values, every job's columns, and the seeds.

    Mirrors PR #885's `_fingerprint`. Every float column is cast to `Float32` before hashing, so
    last-bit noise cannot flip the fingerprint; the saved `losses.parquet` keeps full precision.

    Args:
        frame: The row set every job is fitted on, including `power_mw` and `fold`.
        job_list: Every job this run means to fit.

    Returns:
        A hex digest.
    """
    ordered = frame.select(sorted(frame.columns)).sort("site", "time")
    float_columns = [name for name, dtype in ordered.schema.items() if dtype.is_float()]
    stable = ordered.cast(dict.fromkeys(float_columns, pl.Float32))
    payload = repr(
        (
            stable.hash_rows(seed=0).to_list(),
            [
                (arm, setting, target, tuple(features), tuple(sorted(hyper_parameters.items())))
                for arm, setting, target, features, hyper_parameters, _ in job_list
            ],
            SEEDS,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _script_commit() -> str:
    """Return the commit the script is at, raising if the script has uncommitted changes.

    Mirrors PR #885's `_script_commit`.

    Returns:
        The short hash of the last commit that changed this file.

    Raises:
        ValueError: If the file differs from its last commit, or no commit has changed it.
    """
    script = str(Path(__file__).resolve())
    directory = Path(__file__).parent

    def git(*arguments: str) -> str:
        """Run one git command in the script's directory and return its standard output."""
        return subprocess.run(
            ["git", *arguments], capture_output=True, text=True, check=True, cwd=directory
        ).stdout.strip()

    if git("status", "--porcelain", "--", script):
        msg = "the script has uncommitted changes; commit it before the first fit"
        raise ValueError(msg)
    commit = git("log", "-1", "--format=%h", "--", script)
    if not commit:
        msg = "the script has no commit; commit it before the first fit"
        raise ValueError(msg)
    return commit


def _bootstrap_percentiles(
    *, differences: np.ndarray, months: np.ndarray, percentiles: tuple[float, float]
) -> tuple[float, float]:
    """Resample whole months and a seed, as `studies.bootstrap` does, at any two percentiles.

    Mirrors PR #885's `_bootstrap_percentiles`. The random stream has the same shape as
    `studies.bootstrap._resample_bounds`, and `_bonferroni_lines` asserts that this reproduces
    `bootstrap_difference`'s own 95% interval.

    Args:
        differences: Per-seed, per-row differences, shape (n_seeds, n_rows).
        months: Each row's month label.
        percentiles: The two percentiles to return.

    Returns:
        The two percentiles of the resampled mean.
    """
    unique_months, month_index = np.unique(months, return_inverse=True)
    rows_by_month = [np.flatnonzero(month_index == index) for index in range(len(unique_months))]
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    resampled = np.empty(N_BOOTSTRAP_RESAMPLES)
    for resample in range(N_BOOTSTRAP_RESAMPLES):
        seed_index = generator.integers(0, differences.shape[0])
        drawn = generator.integers(0, len(rows_by_month), size=len(rows_by_month))
        rows = np.concatenate([rows_by_month[index] for index in drawn])
        resampled[resample] = differences[seed_index, rows].mean()
    low, high = np.percentile(resampled, percentiles)
    return float(low), float(high)


def calendar_month_coverage(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Count, for each held-out calendar month, the training rows that carry it.

    Mirrors PR #885's `calendar_month_coverage`.

    Args:
        frame: The row set carrying `site`, `fold` and `time`.

    Returns:
        One row per (site, fold, calendar_month) with `n_scored`, `n_train`, `n_years` (how many
        distinct years of the site's rows carry that calendar month) and `covered`.
    """
    rows = frame.select(
        "site", "fold", calendar_month=pl.col("time").dt.month(), year=pl.col("time").dt.year()
    )
    totals = rows.group_by("site", "calendar_month").agg(
        n_total=pl.len(), n_years=pl.col("year").n_unique()
    )
    return (
        rows.group_by("site", "fold", "calendar_month")
        .agg(n_scored=pl.len())
        .join(totals, on=["site", "calendar_month"])
        .with_columns(n_train=pl.col("n_total") - pl.col("n_scored"))
        .with_columns(covered=pl.col("n_train") > 0)
        .drop("n_total")
        .sort("site", "fold", "calendar_month")
    )


def _raise_on_uncovered_months(*, coverage: pl.DataFrame) -> None:
    """Raise if a held-out calendar month that occurs in two years has no training row.

    Mirrors PR #885's `_raise_on_uncovered_months`.

    Args:
        coverage: `calendar_month_coverage`'s result.

    Raises:
        ValueError: Naming the number of failing cells, without their farm labels' row counts.
    """
    failures = coverage.filter(~pl.col("covered"), pl.col("n_years") > 1)
    if failures.height:
        months = sorted(failures["calendar_month"].unique().to_list())
        msg = (
            f"{failures.height} (farm, fold, calendar month) cells hold out a calendar month that "
            f"occurs in two years and leave no training row for it; calendar months {months}"
        )
        raise ValueError(msg)


# ---------------------------------------------------------------------------------------------
# The row set.
# ---------------------------------------------------------------------------------------------


class StationRows(TypedDict):
    """The station data the row set and the pre-fit checks are built from, all in memory only."""

    observed: pl.DataFrame
    chosen_k1: pl.DataFrame
    chosen_k3: pl.DataFrame
    window: pl.DataFrame


def station_observations() -> pl.DataFrame:
    """Read the candidate stations' wind, keeping the hours where speed and direction both exist.

    The calm flag (direction 0) is set before 360 is normalised to 0, and a calm row enters as sine
    and cosine both zero. North (360) enters as direction 0 degrees.

    Returns:
        One row per observed station hour with `src_id`, `time`, `speed`, `direction`, `sin`, `cos`,
        `calm`, `u` and `v` (the wind vector's components, zero for a calm hour).

    Raises:
        ValueError: If a row carries a unit code other than 4, or a direction outside 0 to 360.
    """
    raw = read_hourly_weather(
        path=HOURLY_WEATHER_PATH,
        columns=["wind_speed_m_s", "wind_direction", "wind_speed_unit_id"],
    ).filter(pl.col("src_id").is_in(CANDIDATE_STATIONS))
    other_units = raw.filter(
        pl.col("wind_speed_unit_id").is_not_null(),
        pl.col("wind_speed_unit_id") != EXPECTED_UNIT_CODE,
    )
    if other_units.height:
        msg = f"{other_units.height} wind rows carry a unit code other than {EXPECTED_UNIT_CODE}"
        raise ValueError(msg)
    observed = raw.filter(
        pl.col("wind_speed_m_s").is_not_null(), pl.col("wind_direction").is_not_null()
    )
    out_of_range = observed.filter(
        (pl.col("wind_direction") < 0.0) | (pl.col("wind_direction") > MAX_DIRECTION_DEG)
    )
    if out_of_range.height:
        msg = f"{out_of_range.height} rows carry a direction outside 0 to {MAX_DIRECTION_DEG}"
        raise ValueError(msg)
    calm = pl.col("wind_direction") == 0.0
    radians = (
        pl.when(pl.col("wind_direction") == MAX_DIRECTION_DEG)
        .then(0.0)
        .otherwise(pl.col("wind_direction"))
        .radians()
    )
    return observed.select(
        "src_id",
        "time",
        speed=pl.col("wind_speed_m_s"),
        direction=pl.col("wind_direction"),
        sin=pl.when(calm).then(0.0).otherwise(radians.sin()),
        cos=pl.when(calm).then(0.0).otherwise(radians.cos()),
        calm=calm,
    ).with_columns(u=pl.col("speed") * pl.col("sin"), v=pl.col("speed") * pl.col("cos"))


def window_rows(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return the page's rows restricted to the window, before the station rule.

    Args:
        sites: The wind roster.

    Returns:
        The page's `common_rows(joined(...))` between `WINDOW_START` and `WINDOW_END_EXCLUSIVE`.
    """
    return common_rows(frame=joined(sites=sites)).filter(
        pl.col("time") >= WINDOW_START, pl.col("time") < WINDOW_END_EXCLUSIVE
    )


def choose_stations(
    *, sites: pl.DataFrame, window: pl.DataFrame, observed: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Choose the nearest eligible station (k=1) and the three nearest (k=3) for each farm.

    Args:
        sites: The wind roster with `site`, `latitude` and `longitude`.
        window: The page's rows in the window, the hours the coverage rule is scored on.
        observed: `station_observations`'s output.

    Returns:
        The k=1 and the k=3 choices, each one row per farm and rank; rank 1 of the k=3 choice is
        asserted equal to the k=1 choice.

    Raises:
        ValueError: If a candidate station has no metadata, or the two rank-1 choices differ.
    """
    stations = read_station_metadata(path=STATION_METADATA_PATH).filter(
        pl.col("src_id").is_in(CANDIDATE_STATIONS)
    )
    if stations.height != len(CANDIDATE_STATIONS):
        msg = f"{stations.height} of {len(CANDIDATE_STATIONS)} candidate stations have metadata"
        raise ValueError(msg)
    required = window.select("site", "time")
    site_points = sites.select("site", "latitude", "longitude")
    chosen_k1 = select_nearest_stations(
        sites=site_points,
        stations=stations,
        observed=observed,
        required=required,
        k=1,
        min_coverage=MIN_COVERAGE,
    )
    chosen_k3 = select_nearest_stations(
        sites=site_points,
        stations=stations,
        observed=observed,
        required=required,
        k=3,
        min_coverage=MIN_COVERAGE,
    )
    if not chosen_k1.equals(chosen_k3.filter(pl.col("rank") == 1)):
        msg = "the k=1 choice differs from rank 1 of the k=3 choice"
        raise ValueError(msg)
    return chosen_k1, chosen_k3


def build_frame(
    *,
    window: pl.DataFrame,
    chosen_k1: pl.DataFrame,
    chosen_k3: pl.DataFrame,
    observed: pl.DataFrame,
) -> pl.DataFrame:
    """Join the nearest station's wind, and the k=3 wind, onto the window rows.

    A farm-hour with no observed nearest-station hour is dropped from every arm. No station
    identifier is kept in the frame.

    Args:
        window: The page's rows in the window.
        chosen_k1: The nearest eligible station per farm.
        chosen_k3: The three nearest eligible stations per farm.
        observed: `station_observations`'s output.

    Returns:
        The window rows with `STATION_COLUMNS` and `K3_COLUMNS`, time features, eras and folds.
    """
    nearest = observed.join(chosen_k1.select("site", "src_id"), on="src_id").select(
        "site",
        "time",
        station_speed=pl.col("speed"),
        station_sin=pl.col("sin"),
        station_cos=pl.col("cos"),
    )
    three = (
        observed.join(chosen_k3.select("site", "src_id"), on="src_id")
        .group_by("site", "time")
        .agg(
            k3_speed=pl.col("speed").mean(),
            mean_u=pl.col("u").mean(),
            mean_v=pl.col("v").mean(),
        )
        .with_columns(magnitude=(pl.col("mean_u") ** 2 + pl.col("mean_v") ** 2).sqrt())
        .select(
            "site",
            "time",
            "k3_speed",
            k3_sin=pl.when(pl.col("magnitude") > 0.0)
            .then(pl.col("mean_u") / pl.col("magnitude"))
            .otherwise(0.0),
            k3_cos=pl.when(pl.col("magnitude") > 0.0)
            .then(pl.col("mean_v") / pl.col("magnitude"))
            .otherwise(0.0),
        )
    )
    rows = window.join(nearest, on=["site", "time"], how="inner").join(
        three, on=["site", "time"], how="inner"
    )
    return with_eras(frame=_add_time_features(dataset=rows.sort("site", "time")))


# ---------------------------------------------------------------------------------------------
# Jobs.
# ---------------------------------------------------------------------------------------------


def arm_columns() -> dict[str, tuple[str, ...]]:
    """Return every fitted arm's feature columns.

    Returns:
        Arm name to its columns: the shared columns followed by the arm's wind columns.
    """
    arms: dict[str, tuple[str, ...]] = {
        "station_wind": (*SHARED_FEATURES, *STATION_COLUMNS),
        "era5_10m_wind": (*SHARED_FEATURES, *ERA5_10M_COLUMNS),
        "ukv_station_wind": (*SHARED_FEATURES, *_wind_columns(product="ukv"), *STATION_COLUMNS),
        "ukv_padded_wind": (*SHARED_FEATURES, *_wind_columns(product="ukv"), *UKV_PADDING_COLUMNS),
        "station_k3_wind": (*SHARED_FEATURES, *K3_COLUMNS),
    }
    arms.update(
        {
            f"{product}_wind": (*SHARED_FEATURES, *_wind_columns(product=product))
            for product in PRODUCTS
        }
    )
    return arms


def _raise_on_unequal_counts(*, arms: dict[str, tuple[str, ...]]) -> None:
    """Raise unless both arms of every planned contrast carry the same number of columns.

    Args:
        arms: `arm_columns()`.

    Raises:
        ValueError: Naming the contrast and the two counts.
    """
    for name, first, second in PLANNED_CONTRASTS:
        if len(arms[first]) != len(arms[second]):
            msg = f"{name}: {first} has {len(arms[first])} columns, {second} {len(arms[second])}"
            raise ValueError(msg)


def jobs() -> list[Job]:
    """Return every arm's job at the primary setting, and the sensitivity arms at the second.

    Returns:
        One job per arm at `PRIMARY_HYPER_PARAMETERS`, then one per `SENSITIVITY_ARMS`.
    """
    arms = arm_columns()
    _raise_on_unequal_counts(arms=arms)
    job_list: list[Job] = [
        (arm, "pooled", "power_mw", columns, PRIMARY_HYPER_PARAMETERS, False)
        for arm, columns in arms.items()
    ]
    job_list += [
        (arm, "sensitivity", "power_mw", arms[arm], SENSITIVITY_HYPER_PARAMETERS, False)
        for arm in SENSITIVITY_ARMS
    ]
    return job_list


# ---------------------------------------------------------------------------------------------
# Pre-fit checks. Every figure about stations is pooled over farms.
# ---------------------------------------------------------------------------------------------


class ChecksResult(TypedDict):
    """The pre-fit checks' raw results, computed once."""

    window_rows: int
    frame_rows: int
    distance_k1: tuple[float, float]
    distance_k3: tuple[float, float]
    coverage_k1: tuple[float, float]
    skipped_nearer: tuple[float, float]
    eligible_missing_share: tuple[float, float]
    n_eligible_pairs: int
    speed_range: tuple[float, float]
    speed_mean: float
    direction_facts: dict[str, float]
    unit_rows: dict[str, int]
    off_the_hour_rows: int
    correlations: dict[int, float]
    fold_months: list[list[str]]
    same_fold_layout: bool
    coverage: pl.DataFrame
    single_year_months: list[int]
    single_year_share: float


def _correlations(
    *, window: pl.DataFrame, chosen_k1: pl.DataFrame, observed: pl.DataFrame
) -> dict[int, float]:
    """Correlate the nearest station's speed with UKV's 10 m speed at each hour offset, pooled.

    An offset of +1 hour pairs UKV's speed at `t` with the station's reading stamped `t` + 1 hour.

    Args:
        window: The page's rows in the window.
        chosen_k1: The nearest eligible station per farm.
        observed: `station_observations`'s output.

    Returns:
        Offset to Pearson correlation over every farm-hour with both values.
    """
    station = observed.join(chosen_k1.select("site", "src_id"), on="src_id").select(
        "site", "time", "speed"
    )
    result: dict[int, float] = {}
    for offset in LAG_SCAN_HOURS:
        shifted = station.with_columns(time=pl.col("time") - pl.duration(hours=offset))
        paired = window.select("site", "time", "speed_10m_ukv").join(
            shifted, on=["site", "time"], how="inner"
        )
        result[offset] = float(paired.select(pl.corr("speed", "speed_10m_ukv")).item())
    return result


def _range(*, values: pl.Series) -> tuple[float, float]:
    """Return the smallest and largest value of a series.

    Args:
        values: A numeric series.

    Returns:
        The minimum and the maximum.
    """
    array = values.to_numpy()
    return float(array.min()), float(array.max())


def run_checks(
    *,
    window: pl.DataFrame,
    frame: pl.DataFrame,
    chosen_k1: pl.DataFrame,
    chosen_k3: pl.DataFrame,
    observed: pl.DataFrame,
) -> ChecksResult:
    """Compute every pre-fit check, reporting stations, hours and distances pooled over farms.

    Args:
        window: The page's rows in the window.
        frame: The row set every arm is fitted on.
        chosen_k1: The nearest eligible station per farm.
        chosen_k3: The three nearest eligible stations per farm.
        observed: `station_observations`'s output.

    Returns:
        The results.
    """
    required = window.select("site", "time")
    hours_by_farm = required.group_by("site").agg(n_required=pl.len())
    per_pair = (
        required.join(observed.select("src_id", "time"), on="time")
        .group_by("site", "src_id")
        .agg(n_covered=pl.len())
        .join(hours_by_farm, on="site")
        .with_columns(coverage=pl.col("n_covered") / pl.col("n_required"))
        .filter(pl.col("coverage") >= MIN_COVERAGE)
    )
    raw_units = read_hourly_weather(
        path=HOURLY_WEATHER_PATH, columns=["wind_speed_m_s", "wind_speed_unit_id"]
    ).filter(pl.col("src_id").is_in(CANDIDATE_STATIONS), pl.col("wind_speed_m_s").is_not_null())
    scored = observed.join(chosen_k1.select("site", "src_id"), on="src_id").join(
        frame.select("site", "time"), on=["site", "time"]
    )
    fold_months_by_farm = frame.select("site", "fold", "month").unique()
    layouts = [
        sorted(
            fold_months_by_farm.filter(pl.col("site") == site, pl.col("fold") == fold)[
                "month"
            ].to_list()
        )
        for site in sorted(frame["site"].unique().to_list())
        for fold in range(N_FOLDS)
    ]
    fold_months = [
        sorted(fold_months_by_farm.filter(pl.col("fold") == fold)["month"].unique().to_list())
        for fold in range(N_FOLDS)
    ]
    coverage = calendar_month_coverage(frame=frame)
    single_year = coverage.filter(pl.col("n_years") == 1)
    speeds = frame["station_speed"]
    return {
        "window_rows": window.height,
        "frame_rows": frame.height,
        "distance_k1": _range(values=chosen_k1["distance_km"]),
        "distance_k3": _range(values=chosen_k3.filter(pl.col("rank") == 3)["distance_km"]),
        "coverage_k1": _range(values=chosen_k1["coverage"]),
        "skipped_nearer": _range(values=chosen_k3["skipped_nearer"]),
        "eligible_missing_share": _range(values=1.0 - per_pair["coverage"]),
        "n_eligible_pairs": per_pair.height,
        "speed_range": _range(values=speeds),
        "speed_mean": float(speeds.to_numpy().mean()),
        "direction_facts": {
            "calm_share": float(scored["calm"].to_numpy().mean()),
            "calm_rows_with_speed": float(
                scored.filter(pl.col("calm"), pl.col("speed") > 0.0).height
            ),
            "multiples_of_ten_share": float(
                ((observed["direction"] % 10.0) == 0.0).to_numpy().mean()
            ),
            "whole_knot_share": float(
                (
                    ((observed["speed"] / MS_PER_KNOT).round(0) * MS_PER_KNOT - observed["speed"])
                    .abs()
                    .to_numpy()
                    < WHOLE_KNOT_TOLERANCE_M_S
                ).mean()
            ),
        },
        "unit_rows": {
            "with_unit_code_4": int(raw_units.filter(pl.col("wind_speed_unit_id") == 4).height),
            "with_no_unit_code": int(raw_units["wind_speed_unit_id"].null_count()),
            "with_another_code": int(
                raw_units.filter(
                    pl.col("wind_speed_unit_id").is_not_null(),
                    pl.col("wind_speed_unit_id") != EXPECTED_UNIT_CODE,
                ).height
            ),
        },
        "off_the_hour_rows": int(
            observed.filter(
                (pl.col("time").dt.minute() != 0) | (pl.col("time").dt.second() != 0)
            ).height
        ),
        "correlations": _correlations(window=window, chosen_k1=chosen_k1, observed=observed),
        "fold_months": fold_months,
        "same_fold_layout": all(
            layouts[fold] == layouts[fold + index * N_FOLDS]
            for index in range(1, frame["site"].n_unique())
            for fold in range(N_FOLDS)
        ),
        "coverage": coverage,
        "single_year_months": sorted(single_year["calendar_month"].unique().to_list()),
        "single_year_share": float(
            single_year["n_scored"].to_numpy().sum() / coverage["n_scored"].to_numpy().sum()
        ),
    }


def _raise_on_failed_checks(*, checks: ChecksResult, frame: pl.DataFrame) -> None:
    """Raise if a pre-fit check fails, before any arm is fitted.

    Args:
        checks: `run_checks`'s result.
        frame: The row set, whose arm columns must hold no missing value.

    Raises:
        ValueError: If a unit code is not 4, a timestamp is off the hour, the station-against-UKV
            correlation does not peak at zero offset, an uncovered calendar month occurs in two
            years, or an arm column holds a missing value.
    """
    if checks["unit_rows"]["with_another_code"]:
        msg = "a wind row carries a unit code other than 4"
        raise ValueError(msg)
    if checks["off_the_hour_rows"]:
        msg = f"{checks['off_the_hour_rows']} station rows are stamped off the hour"
        raise ValueError(msg)
    correlations = checks["correlations"]
    if max(correlations, key=correlations.__getitem__) != 0:
        msg = f"the station-against-UKV correlation does not peak at zero offset: {correlations}"
        raise ValueError(msg)
    _raise_on_uncovered_months(coverage=checks["coverage"])
    every_column = {column for columns in arm_columns().values() for column in columns}
    check_no_missing(frame=frame, columns=sorted(every_column))


def _shear_control(*, frame: pl.DataFrame) -> float:
    """Fit one fold of one farm on the raw and the 1/7-power-scaled station speed, and compare.

    A tree model is invariant to a monotone rescaling of one column, so the predictions should
    match. This is a control, not an arm: nothing here enters `losses.parquet`.

    Args:
        frame: The row set carrying `fold`, `power_mw` and the station columns.

    Returns:
        The largest absolute difference between the two fits' predictions, in MW.
    """
    site = min(frame["site"].unique().to_list())
    rows = frame.filter(pl.col("site") == site)
    scale = SHEAR_HEIGHT_RATIO**SHEAR_EXPONENT
    scaled = rows.with_columns(station_speed=pl.col("station_speed") * scale)
    predictions = []
    for source in (rows, scaled):
        prediction, _ = fit_one_fold(
            train=source.filter(pl.col("fold") != 0),
            test=source.filter(pl.col("fold") == 0),
            features=[*SHARED_FEATURES, *STATION_COLUMNS],
            target="power_mw",
            hyper_parameters=PRIMARY_HYPER_PARAMETERS,
            seed=SEEDS[0],
            with_quantiles=False,
        )
        predictions.append(prediction)
    return float(np.abs(predictions[0] - predictions[1]).max())


def _checks_lines(*, checks: ChecksResult) -> list[str]:
    """Render the pre-fit checks as markdown, pooled over farms.

    Args:
        checks: `run_checks`'s result.

    Returns:
        Markdown lines.
    """
    dropped = checks["window_rows"] - checks["frame_rows"]
    correlations = checks["correlations"]
    facts = checks["direction_facts"]
    lines = [
        "#### Row set and station checks, pooled over the three farms, run before any fit",
        "",
        (
            f"- Page rows in the window: {checks['window_rows']:,}. Rows after the station rule: "
            f"{checks['frame_rows']:,}. Rows dropped for want of an observed nearest-station hour: "
            f"{dropped:,}."
        ),
        (
            f"- Nearest eligible station's distance from its farm, range over the three farms: "
            f"{checks['distance_k1'][0]:.0f} to {checks['distance_k1'][1]:.0f} km. Third-nearest "
            f"eligible station: {checks['distance_k3'][0]:.0f} to "
            f"{checks['distance_k3'][1]:.0f} km."
        ),
        (
            "- Nearest eligible station's coverage of its farm's required hours, range over the "
            f"three farms: {checks['coverage_k1'][0]:.4f} to {checks['coverage_k1'][1]:.4f} "
            f"(the rule needs {MIN_COVERAGE})."
        ),
        (
            f"- Nearer stations skipped for failing the coverage rule, range over farms and ranks: "
            f"{checks['skipped_nearer'][0]:.0f} to {checks['skipped_nearer'][1]:.0f}."
        ),
        (
            "- Missing share of required hours, range over the "
            f"{checks['n_eligible_pairs']} eligible (farm, station) pairs: "
            f"{checks['eligible_missing_share'][0]:.4f} to "
            f"{checks['eligible_missing_share'][1]:.4f}."
        ),
        (
            f"- Nearest-station wind speed on the scored rows: {checks['speed_range'][0]:.2f} to "
            f"{checks['speed_range'][1]:.2f} m/s, mean {checks['speed_mean']:.2f} m/s."
        ),
        (
            "- Calm hours (direction 0) on the scored rows: "
            f"{facts['calm_share']:.4f} of rows; calm rows with a speed above zero: "
            f"{int(facts['calm_rows_with_speed'])}. A calm row enters as sine and cosine both "
            "zero; 360 is read as north (0 degrees)."
        ),
        (
            f"- Share of non-calm station directions that are multiples of 10 degrees: "
            f"{facts['multiples_of_ten_share']:.4f}. Share of station speeds that are whole knots: "
            f"{facts['whole_knot_share']:.4f}."
        ),
        (
            f"- Unit codes, over every candidate station's wind-speed rows: code 4 on "
            f"{checks['unit_rows']['with_unit_code_4']:,} rows, no code on "
            f"{checks['unit_rows']['with_no_unit_code']:,} rows, any other code on "
            f"{checks['unit_rows']['with_another_code']:,} rows."
        ),
        f"- Station rows stamped off the hour: {checks['off_the_hour_rows']}.",
        "- Correlation of the nearest station's speed with UKV's 10 m speed, pooled over farms, "
        "by hour offset (+1 pairs UKV at t with the station stamped t + 1 hour): "
        + ", ".join(f"{offset:+d} h: {correlations[offset]:.3f}" for offset in LAG_SCAN_HOURS)
        + ".",
        "",
        "#### Folds and calendar-month coverage, checked before any fit",
        "",
        (
            "Months in each fold (every farm holds the same layout: "
            f"{'yes' if checks['same_fold_layout'] else 'NO'}):"
        ),
        "",
    ]
    lines += [
        f"- Fold {fold}: {len(months)} months, {months[0]} to {months[-1]}"
        for fold, months in enumerate(checks["fold_months"])
    ]
    coverage = checks["coverage"]
    pooled = (
        coverage.group_by("fold", "calendar_month")
        .agg(
            n_scored=pl.col("n_scored").sum(),
            n_train=pl.col("n_train").sum(),
            n_years=pl.col("n_years").max(),
        )
        .sort("fold", "calendar_month")
    )
    uncovered = coverage.filter(~pl.col("covered"), pl.col("n_years") > 1).height
    lines += [
        "",
        (
            f"Cells (farm, fold, calendar month) checked: {coverage.height}. Cells with no "
            "training "
            f"row for a calendar month that occurs in two years: {uncovered}. Calendar months that "
            f"occur in one year only: {checks['single_year_months']}, which hold "
            f"{checks['single_year_share']:.1%} of the scored rows and are scored by models that "
            "never "
            "trained on that calendar month, so `day_of_year` extrapolates for every arm."
        ),
        "",
        (
            "| Fold | Calendar month | Rows scored, three farms | Training rows, three farms "
            "| Years of that month |"
        ),
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {row['fold']} | {row['calendar_month']} | {row['n_scored']:,} | {row['n_train']:,} "
        f"| {row['n_years']} |"
        for row in pooled.iter_rows(named=True)
    ]
    return lines


# ---------------------------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------------------------


def _arm_columns_lines(*, job_list: list[Job]) -> list[str]:
    """Render every fitted arm's feature columns and their count, once per arm.

    Args:
        job_list: Every job `jobs()` returns.

    Returns:
        Markdown lines.
    """
    seen: dict[str, tuple[str, ...]] = {}
    for arm, _, _, columns, _, _ in job_list:
        seen.setdefault(arm, columns)
    lines = ["#### Every arm's feature columns", ""]
    lines += [
        f"- `{arm}` ({len(columns)} columns, {len(columns) - len(SHARED_FEATURES)} wind): "
        f"{', '.join(f'`{column}`' for column in columns)}"
        for arm, columns in seen.items()
    ]
    return lines


def _absolute_record(
    *,
    section: str,
    setting: str,
    scope: str,
    arm: str,
    values: dict[str, float],
    n_rows: int | None,
    n_months: int | None,
) -> IntervalRecord:
    """Build the log record of one arm's absolute error and its interval.

    Args:
        section: The report section.
        setting: `pooled` or `sensitivity`.
        scope: `all`, or a farm label.
        arm: The arm.
        values: `value`, `lower` and `upper`, in percentage points.
        n_rows: The rows the interval rests on, or `None` for a per-farm interval.
        n_months: The months the interval rests on, or `None` for a per-farm interval.

    Returns:
        The record.
    """
    return {
        "section": section,
        "setting": setting,
        "scope": scope,
        "treatment": arm,
        "reference": None,
        "value": values["value"],
        "lower": values["lower"],
        "upper": values["upper"],
        "level": 95.0,
        "n_rows": n_rows,
        "n_months": n_months,
        "folds_agreeing": None,
        "n_folds": None,
    }


def _leaderboard_lines(
    *, losses: pl.DataFrame, setting: str, arms: list[str], log: IntervalLog
) -> list[str]:
    """Render every arm's absolute error, pooled with its 95% interval, then per farm.

    Per-farm rows carry no row count, and the interval rests on whole months and a seed.

    Args:
        losses: Per-row losses at one setting.
        setting: `pooled` or `sensitivity`.
        arms: The arms to show, in order.
        log: Where the intervals are recorded.

    Returns:
        Markdown lines.
    """
    columns = {arm: len(cols) - len(SHARED_FEATURES) for arm, cols in arm_columns().items()}
    lines = [
        "| Arm | Wind columns | MAE (pp of capacity) | 95% interval | Rows | Months |",
        "|---|---|---|---|---|---|",
    ]
    for arm in arms:
        interval = bootstrap_absolute(losses=losses, arm=arm, metric=METRIC)
        value, lower, upper = (
            interval[key] * PERCENTAGE_POINTS for key in ("value", "lower_95", "upper_95")
        )
        log.records.append(
            _absolute_record(
                section="leaderboard",
                setting=setting,
                scope="all",
                arm=arm,
                values={"value": value, "lower": lower, "upper": upper},
                n_rows=interval["n_rows"],
                n_months=interval["n_months"],
            )
        )
        lines.append(
            f"| `{arm}` | {columns[arm]} | {value:.3f} | [{lower:.3f}, {upper:.3f}] "
            f"| {interval['n_rows']:,} | {interval['n_months']} |"
        )
    lines += ["", "Per farm (95% interval from resampling whole months and a seed):", ""]
    lines += ["| Arm | Farm | MAE (pp of capacity) | 95% interval |", "|---|---|---|---|"]
    for arm in arms:
        for site in sorted(losses["site"].unique().to_list()):
            interval = bootstrap_absolute(
                losses=losses.filter(pl.col("site") == site), arm=arm, metric=METRIC
            )
            value, lower, upper = (
                interval[key] * PERCENTAGE_POINTS for key in ("value", "lower_95", "upper_95")
            )
            log.records.append(
                _absolute_record(
                    section="leaderboard",
                    setting=setting,
                    scope=site,
                    arm=arm,
                    values={"value": value, "lower": lower, "upper": upper},
                    n_rows=None,
                    n_months=None,
                )
            )
            lines.append(f"| `{arm}` | {site} | {value:.3f} | [{lower:.3f}, {upper:.3f}] |")
    return lines


def contrast_line(
    *,
    losses: pl.DataFrame,
    treatment: str,
    reference: str,
    label: str,
    section: str,
    setting: str,
    log: IntervalLog,
    per_farm: bool = False,
) -> str:
    """Return one markdown contrast row, and record its interval in `log`.

    Mirrors PR #885's `contrast_line`; a per-farm row carries no row or month count.

    Args:
        losses: Per-row losses holding both arms, restricted to the scope wanted.
        treatment: The arm whose error is being compared.
        reference: The arm it is compared against.
        label: The scope label for the first column.
        section: The report section, saved with the interval.
        setting: `pooled` or `sensitivity`, saved with the interval.
        log: Where the interval is recorded.
        per_farm: Whether the scope is one farm, in which case no count is printed or saved.

    Returns:
        The table row.
    """
    interval = bootstrap_difference(
        losses=losses, treatment=treatment, reference=reference, metric=METRIC
    )
    folds = per_fold_differences(
        losses=losses, treatment=treatment, reference=reference, metric=METRIC
    )
    agreeing = int(sum(np.sign(value) == np.sign(interval["difference"]) for value in folds))
    difference, lower, upper = (
        interval[key] * PERCENTAGE_POINTS for key in ("difference", "lower_95", "upper_95")
    )
    excludes = interval["lower_95"] > 0.0 or interval["upper_95"] < 0.0
    log.records.append(
        {
            "section": section,
            "setting": setting,
            "scope": label,
            "treatment": treatment,
            "reference": reference,
            "value": difference,
            "lower": lower,
            "upper": upper,
            "level": 95.0,
            "n_rows": None if per_farm else interval["n_rows"],
            "n_months": None if per_farm else interval["n_months"],
            "folds_agreeing": agreeing,
            "n_folds": len(folds),
        }
    )
    row = (
        f"| {label} | {treatment} − {reference} | {difference:+.3f} | "
        f"[{lower:+.3f}, {upper:+.3f}] | {'**yes**' if excludes else 'no'} | "
        f"{agreeing} of {len(folds)} |"
    )
    if per_farm:
        return row
    return f"{row} {interval['n_rows']:,} | {interval['n_months']} |"


def _contrast_table(
    *,
    losses: pl.DataFrame,
    pairs: tuple[tuple[str, str], ...],
    section: str,
    setting: str,
    log: IntervalLog,
    by_farm: bool,
) -> list[str]:
    """Render a contrast table over `pairs`, pooled or one row per farm and pair.

    Args:
        losses: Per-row losses holding every arm the pairs name.
        pairs: Each (treatment, reference).
        section: The report section, saved with each interval.
        setting: `pooled` or `sensitivity`.
        log: Where the intervals are recorded.
        by_farm: Whether to print one row per farm and pair, without any row count.

    Returns:
        Markdown lines, starting with the caveat line for a pooled table.
    """
    if by_farm:
        rows = [
            contrast_line(
                losses=losses.filter(pl.col("site") == site),
                treatment=treatment,
                reference=reference,
                label=site,
                section=section,
                setting=setting,
                log=log,
                per_farm=True,
            )
            for treatment, reference in pairs
            for site in sorted(losses["site"].unique().to_list())
        ]
        return [*FARM_CONTRAST_HEADER, *rows]
    rows = [
        contrast_line(
            losses=losses,
            treatment=treatment,
            reference=reference,
            label="all",
            section=section,
            setting=setting,
            log=log,
        )
        for treatment, reference in pairs
    ]
    return [*CONTRAST_HEADER, *rows]


def _pooled_caveat(*, losses: pl.DataFrame, arm: str) -> str:
    """Return the caveat line printed under every pooled interval.

    Args:
        losses: Per-row losses at one setting.
        arm: An arm whose rows and months the interval rests on.

    Returns:
        The line, with the pooled row and month counts.
    """
    interval = bootstrap_absolute(losses=losses, arm=arm, metric=METRIC)
    return POOLED_CAVEAT.format(rows=interval["n_rows"], months=interval["n_months"])


def _bonferroni_lines(*, losses: pl.DataFrame, setting: str, log: IntervalLog) -> list[str]:
    """Render the planned contrasts with intervals adjusted for the four planned intervals.

    Each interval is a percentile interval of the same month-and-seed resampling, at the level
    `BONFERRONI_LEVEL`. The function first asserts that its resampler reproduces
    `bootstrap_difference`'s own 95% interval on the first contrast.

    Args:
        losses: Per-row losses at one setting, holding the arms of the planned contrasts.
        setting: `pooled` or `sensitivity`.
        log: Where the intervals are recorded.

    Returns:
        Markdown lines.

    Raises:
        ValueError: If the resampler does not reproduce the 95% interval of `bootstrap_difference`.
    """
    tail = (100.0 - BONFERRONI_LEVEL) / 2.0
    lines = [
        (
            f"| Contrast | ΔMAE (pp of capacity) | {BONFERRONI_LEVEL:.2f}% interval "
            "| Excludes zero? "
            "| Months |"
        ),
        "|---|---|---|---|---|",
    ]
    for index, (name, treatment, reference) in enumerate(PLANNED_CONTRASTS):
        differences, months = paired_differences(
            losses=losses, treatment=treatment, reference=reference, metric=METRIC
        )
        if index == 0:
            own = bootstrap_difference(
                losses=losses, treatment=treatment, reference=reference, metric=METRIC
            )
            mine = _bootstrap_percentiles(
                differences=differences, months=months, percentiles=(2.5, 97.5)
            )
            if not np.allclose(mine, (own["lower_95"], own["upper_95"]), rtol=0.0, atol=1e-12):
                msg = f"the resampler gives {mine}, not bootstrap_difference's interval"
                raise ValueError(msg)
        lower, upper = (
            value * PERCENTAGE_POINTS
            for value in _bootstrap_percentiles(
                differences=differences, months=months, percentiles=(tail, 100.0 - tail)
            )
        )
        difference = float(differences.mean()) * PERCENTAGE_POINTS
        n_months = len(np.unique(months))
        log.records.append(
            {
                "section": "bonferroni",
                "setting": setting,
                "scope": "all",
                "treatment": treatment,
                "reference": reference,
                "value": difference,
                "lower": lower,
                "upper": upper,
                "level": BONFERRONI_LEVEL,
                "n_rows": differences.shape[1],
                "n_months": n_months,
                "folds_agreeing": None,
                "n_folds": None,
            }
        )
        excludes = lower > 0.0 or upper < 0.0
        lines.append(
            f"| {name}: {treatment} − {reference} | {difference:+.3f} "
            f"| [{lower:+.3f}, {upper:+.3f}] | {'**yes**' if excludes else 'no'} | {n_months} |"
        )
    return lines


def _august_to_december_lines(*, losses: pl.DataFrame, setting: str, log: IntervalLog) -> list[str]:
    """Render the planned contrasts scored on August to December only (exploratory).

    The models are the same ones fitted on every month. Only the scored rows are restricted, to the
    calendar months that occur in both years of the window.

    Args:
        losses: Per-row losses at one setting.
        setting: `pooled` or `sensitivity`.
        log: Where the intervals are recorded.

    Returns:
        Markdown lines.
    """
    restricted = losses.filter(pl.col("time").dt.month().is_in(AUGUST_TO_DECEMBER))
    return _contrast_table(
        losses=restricted,
        pairs=tuple((first, second) for _, first, second in PLANNED_CONTRASTS),
        section="august_to_december",
        setting=setting,
        log=log,
        by_farm=False,
    )


def _report(
    *,
    frame: pl.DataFrame,
    losses: pl.DataFrame,
    job_list: list[Job],
    checks: ChecksResult,
    shear_difference_mw: float,
    script_commit: str,
    fingerprint: str,
    log: IntervalLog,
) -> str:
    """Assemble the markdown report.

    Args:
        frame: The row set.
        losses: Every arm's losses at every setting.
        job_list: Every job `jobs()` returns.
        checks: `run_checks`'s result.
        shear_difference_mw: The shear control's largest prediction difference.
        script_commit: The commit of the script that fitted `losses`.
        fingerprint: The row-set fingerprint.
        log: Where every printed interval is recorded.

    Returns:
        The report.
    """
    pooled = losses.filter(pl.col("setting") == "pooled")
    sensitivity = losses.filter(pl.col("setting") == "sensitivity")
    arms = list(arm_columns())
    planned_pairs = tuple((first, second) for _, first, second in PLANNED_CONTRASTS)
    lines = [
        (
            f"### Nearest weather station's wind against gridded products, on {frame.height:,} "
            f"farm-hours ({frame['time'].min():%Y-%m-%d} to {frame['time'].max():%Y-%m-%d})"
        ),
        "",
        (
            f"Fitted by the script at commit `{script_commit}`; row-set fingerprint "
            f"`{fingerprint}`. MAE is the mean absolute error as a percentage of each row's own "
            "farm capacity, on the capped predictions. Every arm, including each gridded "
            "product's wind arm, is refitted on "
            "exactly these rows. Intervals resample whole calendar months and a fitting seed."
        ),
        "",
        *_checks_lines(checks=checks),
        "",
        *_arm_columns_lines(job_list=job_list),
        "",
        "#### Shear control (not an arm)",
        "",
        (
            "One fold (fold 0) of one farm is fitted on the raw station speed and on the speed "
            f"scaled by (100 / 10) ^ (1 / 7) = {SHEAR_HEIGHT_RATIO**SHEAR_EXPONENT:.4f}, seed "
            f"{SEEDS[0]}, at the "
            "primary setting. Largest absolute difference between the two fits' predictions: "
            f"{shear_difference_mw:.6g} MW. The exponent is a guess, not a fit."
        ),
        "",
        "#### Leaderboard at the primary setting",
        "",
        _pooled_caveat(losses=pooled, arm="station_wind"),
        "",
        *_leaderboard_lines(losses=pooled, setting="pooled", arms=arms, log=log),
        "",
        "#### Leaderboard at the second setting",
        "",
        _pooled_caveat(losses=sensitivity, arm="station_wind"),
        "",
        *_leaderboard_lines(
            losses=sensitivity, setting="sensitivity", arms=list(SENSITIVITY_ARMS), log=log
        ),
    ]
    for setting, scope in (("pooled", pooled), ("sensitivity", sensitivity)):
        heading = "primary" if setting == "pooled" else "second"
        lines += [
            "",
            f"#### Planned contrasts S1 and S2 at the {heading} setting (first minus second)",
            "",
            _pooled_caveat(losses=scope, arm="station_wind"),
            "",
            *_contrast_table(
                losses=scope,
                pairs=planned_pairs,
                section="planned",
                setting=setting,
                log=log,
                by_farm=False,
            ),
            "",
            (
                f"Adjusted for the {2 * len(PLANNED_CONTRASTS)} planned intervals (2 contrasts, "
                f"2 settings), at the {BONFERRONI_LEVEL:.2f}% level:"
            ),
            "",
            *_bonferroni_lines(losses=scope, setting=setting, log=log),
            "",
            f"By farm, {heading} setting:",
            "",
            *_contrast_table(
                losses=scope,
                pairs=planned_pairs,
                section="planned_by_farm",
                setting=setting,
                log=log,
                by_farm=True,
            ),
        ]
    lines += [
        "",
        "#### Exploratory results",
        "",
        (
            "The contrasts below were not planned. `ukv_station_wind − ukv_wind` compares 7 wind "
            "columns with 4, so it is not an equal-column-count contrast."
        ),
        "",
    ]
    for setting, scope in (("pooled", pooled), ("sensitivity", sensitivity)):
        heading = "primary" if setting == "pooled" else "second"
        lines += [
            f"Exploratory contrasts at the {heading} setting:",
            "",
            *_contrast_table(
                losses=scope,
                pairs=EXPLORATORY_CONTRASTS if setting == "pooled" else EXPLORATORY_CONTRASTS[:1],
                section="exploratory",
                setting=setting,
                log=log,
                by_farm=False,
            ),
            "",
            (
                f"S1 and S2 scored on August to December only, {heading} setting. The 10 scored "
                "months "
                "occur in both years of the window; the models are the ones fitted on every month."
            ),
            "",
            *_august_to_december_lines(losses=scope, setting=setting, log=log),
            "",
        ]
    return "\n".join(lines) + "\n"


def _readme_text() -> str:
    """Return the text of the README written beside the outputs.

    Returns:
        Markdown saying what each file holds.
    """
    return f"""# Nearby weather-station wind in the past-wind study

Outputs of `studies/beam_diffuse_split/station_wind_arms.py`. Wind farms appear only as `W1` to
`W3`. Nothing here carries a station identifier, a station coordinate, or a per-farm distance.

- `losses.parquet`: one row per (arm, setting, farm, hour, seed) with the out-of-fold absolute and
  signed errors in MW and as a fraction of the row's capacity, the fold, the month, and `actual_mw`,
  the metered power. `setting` is `pooled` (the primary hyperparameters) or `sensitivity` (the
  second setting).
- `losses.fingerprint`: a hash of the row set (floats cast to Float32), every arm's columns, the
  seeds and the hyperparameters. `--report-only` refuses to reuse `losses.parquet` if the hash
  changes.
- `script_commit.txt`: the commit of the script that fitted `losses.parquet`.
- `intervals.parquet`: every interval `report.md` prints, one row each, with its section, setting,
  scope, arms, value and bounds (percentage points of capacity), level, rows and months. A per-farm
  interval carries no row or month count.
- `report.md`: every table the docs page quotes, printed by the script and never transcribed.
- `superseded/`: outputs a later run replaced.

`SEEDS` is {list(SEEDS)}, and each interval resamples whole calendar months and one of those seeds
{N_BOOTSTRAP_RESAMPLES:,} times.
"""


def main() -> int:
    """Build the row set, run every check, fit every arm, and write the report.

    Every check runs before any arm is fitted, in a fresh run, `--report-only` and `--checks-only`.
    `--checks-only` prints the checks and stops, fitting and writing nothing.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Fit nothing; rebuild report.md from the saved losses.parquet alone.",
    )
    parser.add_argument(
        "--checks-only",
        action="store_true",
        help="Run every check before the fit, print the results, and stop.",
    )
    arguments = parser.parse_args()

    sites = _wind_sites()
    observed = station_observations()
    window = window_rows(sites=sites)
    chosen_k1, chosen_k3 = choose_stations(sites=sites, window=window, observed=observed)
    frame = build_frame(window=window, chosen_k1=chosen_k1, chosen_k3=chosen_k3, observed=observed)
    _LOG.info("window rows: %d; rows after the station rule: %d", window.height, frame.height)

    all_jobs = jobs()
    checks = run_checks(
        window=window, frame=frame, chosen_k1=chosen_k1, chosen_k3=chosen_k3, observed=observed
    )
    if arguments.checks_only:
        sys.stdout.write(
            "\n".join(
                [*_checks_lines(checks=checks), "", *_arm_columns_lines(job_list=all_jobs), ""]
            )
        )
    _raise_on_failed_checks(checks=checks, frame=frame)
    if arguments.checks_only:
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "losses.parquet"
    fingerprint_path = OUTPUT_DIR / "losses.fingerprint"
    intervals_path = OUTPUT_DIR / "intervals.parquet"
    report_path = OUTPUT_DIR / "report.md"
    readme_path = OUTPUT_DIR / "README.md"
    commit_path = OUTPUT_DIR / "script_commit.txt"

    fingerprint = _fingerprint(frame=frame, job_list=all_jobs)
    if arguments.report_only:
        saved = fingerprint_path.read_text().strip() if fingerprint_path.exists() else None
        if saved != fingerprint:
            msg = (
                f"--report-only: {path} was fitted on a different row set, column set, seed set, "
                "feature values or hyperparameter setting than this code now produces; re-run "
                "without --report-only"
            )
            raise ValueError(msg)
        losses = pl.read_parquet(path)
        script_commit = commit_path.read_text().strip()
    else:
        refuse_to_overwrite(
            paths=[path, fingerprint_path, intervals_path, report_path, commit_path]
        )
        script_commit = _script_commit()
        fitted = run_all(dataset=frame, jobs=all_jobs)
        losses = fitted.join(
            frame.select("site", "time", actual_mw=pl.col("power_mw").cast(pl.Float64)),
            on=["site", "time"],
            how="left",
        )
        losses.write_parquet(path)
        fingerprint_path.write_text(fingerprint)
        commit_path.write_text(script_commit)

    log = IntervalLog()
    report = _report(
        frame=frame,
        losses=losses,
        job_list=all_jobs,
        checks=checks,
        shear_difference_mw=_shear_control(frame=frame),
        script_commit=script_commit,
        fingerprint=fingerprint,
        log=log,
    )
    report_path.write_text(report)
    log.frame().write_parquet(intervals_path)
    readme_path.write_text(_readme_text())
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
