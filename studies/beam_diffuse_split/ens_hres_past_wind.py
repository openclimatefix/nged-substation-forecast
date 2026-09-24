"""Score ECMWF ENS and ECMWF IFS HRES as descriptions of past wind, refitting every arm.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/pull/885>, extending the five-product
past-wind comparison in `wind_products.py` (ERA5, UKV, ICON-D2, ICON-EU, ICON global) with two
ECMWF products: the deterministic IFS HRES forecast, and the mean of the 51-member ENS at day 0.
The past-solar study already scores both products; the past-wind page scored neither.

**The planned contrasts are in `plans/ens-hres-past-wind.md`, committed as `715681a7` before any fit
ran.** The three contrasts, and the second hyperparameter setting each is rerun at, are:

- **P1** `hres_wind − ukv_wind`: does ECMWF's deterministic forecast describe past wind better than
  UKV, the page's product for historical features?
- **P2** `ens_mean_day0_wind − ukv_wind`: does the ENS ensemble mean describe past wind better than
  UKV?
- **P3** `hres_wind − era5_wind`: does ECMWF's deterministic forecast describe past wind better than
  the reanalysis a training history could use?

Every other number this script prints is exploratory: ENS against HRES, ENS against ERA5, the ICON
contrasts, the three alternative ENS interpolations, the Bonferroni-adjusted intervals, the
period splits, the servable-hours split, and every by-farm row. **The ENS-against-ERA5 contrast
re-estimates a figure the ENS horizons study already published** (day 0 against ERA5 on 50,268 rows
from 12 August 2024), on a shorter row set, and is exploratory; the report prints the published
figure beside it.

**Departures from the maintainer's request, each written into the plan.** HRES is read from the
Open-Meteo Previous Runs file (`previous_runs/combined.parquet`, model `ecmwf_ifs`, the bare column
is the freshest run), not the 342-point grid file, because the Previous Runs file holds wind
direction and the grid file holds speed only. The grid file is used for one cross-check: at each
farm, some point among the five nearest to it must reproduce the Previous Runs speeds at 10 m and
100 m to within 0.05 km/h on every hour, because Open-Meteo picks its own model cell, which is not
always the nearest point. ENS day 0 is read from the horizons study's saved inputs
(`data/studies/ens_forecast_horizons/wind_inputs.parquet`), not from the `T+3` wind download, so no
new ENS code is written.

**Row set.** `common_rows(joined(sites=sites))` from `wind_products.py`, restricted to 1 December
2024 (the first whole month after IFS Cycle 49r1 on 12 November 2024) before eras and folds are
assigned, then inner-joined to ENS and HRES. Every arm, including the five original products, is
refit on this row set. The joins must lose no rows: the row set holds 43,555 rows (W1 14,489, W2
14,994, W3 14,072) and the script raises otherwise.

**Arms.** Every arm has the page's seven columns, `SHARED_FEATURES` plus `_wind_columns(product)`:
the hub-height speed, that height's direction as sine and cosine, and the 10 m speed. ERA5, UKV,
HRES and ENS are shown 100 m wind; the ICON products are shown 80 m wind, as the page does.
`colsample_bytree` is never set, so it is 1. HRES speeds are converted from km/h to m/s where the
frame is built; ENS speeds are already in m/s. XGBoost's split thresholds are invariant to a
rescaling of one column, so the ERA5, UKV and ICON columns stay in the km/h the page fitted them in.

- **Primary setting:** ERA5, UKV, ICON-D2, ICON-EU, ICON global, HRES, and `ens_mean_day0` (ENS
  ensemble mean at day 0, method `components`).
- **Second setting** (`SENSITIVITY_HYPER_PARAMETERS`): HRES, `ens_mean_day0`, UKV, ERA5.
- **Exploratory ENS variants, primary setting:** `ens_mean_day0_speed_components` (the horizons
  study's chosen combination), `ens_mean_day0_direction_components` and `ens_mean_day0_linear`.

**How the ENS wind is built.** In the horizons study's `components` combination, each member's
3-hourly speed and direction become eastward and northward components, the components are
interpolated linearly to hourly, the hourly speed is the magnitude of the interpolated vector, and
the hourly direction is that vector's bearing, entering the model as sine and cosine. The
members are then combined as `reduce_members` does: the mean of the members' speeds, and the
direction of the mean wind vector. Direction is never averaged or interpolated as a plain number in
`components`. The horizons study's own choice, `speed_components`, interpolates direction linearly
in degrees, which crosses north the wrong way on about 1% of day-0 hours, and runs here as an
exploratory arm.

**Eras and folds.** Three eras, cut where an input's version changes: before 1 October 2025 (the
date Open-Meteo's ECMWF archive source changes), 1 October 2025 to 20 January 2026, and from 1
February 2026 (the UKV upgrade; the rest of January 2026 is dropped as on the page). `era_code`
takes the values 0, 1 and 2 for every arm, and folds are cut inside each `(site, era)` with
`assign_folds`. IFS Cycle 50r1 (12 May 2026) is left to the period split.

**Before any fit**, the script prints and checks: the row counts at each join, the calendar-month
coverage of the folds (issue #868), the HRES cross-check against the grid file, the ENS power
against the page's power on the rows they share, and the HRES served-lead evidence (the hour-to-hour
change by UTC hour). The coverage check raises if a calendar month held out of a fold has no
training row from any era, except for a calendar month that occurs in one year of the row set only,
which no fold design can cover.

**What no planned contrast can separate.** Each contrast mixes served lead, step width, native and
served resolution, IFS cycle, the source of HRES's archive, height, how each value is read at a
farm, and, for the ENS-against-HRES contrast, ensemble averaging against a single run. The 3-hourly
steps, the single 00 UTC run, the missing direct-radiation fields and the roughly 09:00 UTC time by
which a 00 UTC run becomes available belong to ECMWF's open-data subset and Dynamical.org's archive,
not to ENS. ENS's native grid is about 9 km (O1280), served at 0.25 degrees.

Run it with `uv run python studies/beam_diffuse_split/ens_hres_past_wind.py`. `refuse_to_overwrite`
on a fresh run means `losses.parquet`, `losses.fingerprint`, `intervals.parquet` and `report.md`
each have to move to a `superseded/` subfolder before a re-run. `--report-only` rebuilds the report
from the saved `losses.parquet` alone, fitting nothing, but still raises if the saved fingerprint
(the row set with its Float32-cast floats, every job's columns, the seeds, and the hyperparameters)
no longer matches what this code would fit.
"""

import argparse
import hashlib
import importlib.util
import logging
import math
import re
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
    assign_folds,
)
from studies.guards import refuse_to_overwrite
from weather_products import METRIC, PERCENTAGE_POINTS, _mae
from wind_products import (
    SHARED_FEATURES,
    _hub_height_m,
    _wind_columns,
    common_rows,
    geometry_lines,
    joined,
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

OUTPUT_DIR_NAME: Final[str] = "ens_hres_past_wind"
"""The results directory under `sources.STUDY_DATA_DIR / 'past_weather_v2'`."""

OUTPUT_DIR: Final[Path] = STUDY_DATA_DIR / "past_weather_v2" / OUTPUT_DIR_NAME
"""Where this study writes its losses, intervals, report and README, and a `superseded/` folder."""

HRES_DIR: Final[Path] = WEATHER_DATA_DIR / "ECMWF-IFS-HRES"
"""Holds the HRES downloads this script reads."""

HRES_PREVIOUS_RUNS_PATH: Final[Path] = HRES_DIR / "previous_runs" / "combined.parquet"
"""Open-Meteo Previous Runs, model `ecmwf_ifs`, at each meter's own coordinates. Bare columns are
the freshest run, in km/h and degrees."""

HRES_GRID_PATH: Final[Path] = HRES_DIR / "ECMWF-IFS-HRES_2017-01-01_2026-09-22.parquet"
"""The 342-point, 0.05-degree grid of HRES speeds in km/h, read only for the cross-check."""

ENS_INPUTS_PATH: Final[Path] = (
    STUDY_DATA_DIR.parent / "ens_forecast_horizons" / "wind_inputs.parquet"
)
"""The horizons study's ENS wind inputs: the ensemble mean, hourly, in m/s, per interpolation."""

HORIZONS_REPORT_PATH: Final[Path] = ENS_INPUTS_PATH.parent / "report.md"
"""The horizons study's report, read for the published day-0 contrast against ERA5."""

TRIAL_AREA_PATHS_MODULE: Final[Path] = (
    Path(__file__).parent.parent / "weather_downloads" / "paths.py"
)
"""The module that holds the trial-area box, loaded by path because `studies/` is not a package."""

ROW_SET_START: Final[pl.Expr] = pl.datetime(2024, 12, 1, time_zone="UTC")
"""The first hour of the row set: the first whole month after IFS Cycle 49r1 (12 November 2024)."""

EXPECTED_ROWS: Final[dict[str, int]] = {"W1": 14_489, "W2": 14_994, "W3": 14_072}
"""The row count per farm the plan fixes; the script raises if the joined row set differs."""

HRES_ARCHIVE_CHANGE: Final[pl.Expr] = pl.datetime(2025, 10, 1, time_zone="UTC")
"""The date Open-Meteo's ECMWF archive source changes, and the first era boundary."""

IFS_CYCLE_50R1: Final[pl.Expr] = pl.datetime(2026, 5, 12, time_zone="UTC")
"""The date IFS Cycle 50r1 goes operational, the second period-split boundary."""

ERA_START_MONTHS: Final[tuple[str, str]] = ("2025-10", "2026-02")
"""The first month of the second and third eras, in the page's `%Y-%m` month label."""

ERA_FOLD_OFFSETS: Final[dict[int, int]] = {0: 0, 1: 0, 2: 2}
"""How far each era's fold numbers are rotated, modulo `N_FOLDS`, before any fit.

The calendar-month coverage check (`calendar_month_coverage`) fails with every offset at 0: July
is in fold 3 and September in fold 4 in both 2025 and 2026, so holding either fold out leaves no
training row for that season. A rotation of 2 for the third era puts every calendar month that
occurs in both years into two different folds.
"""

SERVABLE_LABEL_HOURS: Final[tuple[range, range]] = (range(9), range(10, 24))
"""Label hours (UTC) of the servable split: 00 to 08, and 10 to 23; label 09 is dropped."""

JUMP_RATIO_THRESHOLD: Final[float] = 1.15
"""A UTC hour whose mean absolute hour-to-hour change, over the median across hours, is at least
this is reported as a jump, the mark of a switch between two forecast runs in HRES's stitched
series."""

LEAD_PERIODS: Final[tuple[str, str]] = ("before 2025-10-01", "from 2025-10-01")
"""The two periods of the HRES served-lead table, split where the archive source changes."""

PLAN_EXPECTED_JUMP_HOURS: Final[dict[str, list[int]]] = {
    "before 2025-10-01": [1, 13],
    "from 2025-10-01": [0, 6, 12, 18],
}
"""The UTC hours at which the plan expected HRES's stitched series to hand over between runs."""

GRID_NEAREST_RANK: Final[int] = 5
"""How many of the nearest grid points the HRES cross-check tries for each farm."""

GRID_TOLERANCE_KMH: Final[float] = 0.05 + 1e-6
"""The largest hourly disagreement, in km/h, between the grid file and the Previous Runs file."""

KMH_PER_M_S: Final[float] = 3.6
"""Kilometres per hour in one metre per second."""

PRIMARY_PRODUCTS: Final[tuple[str, ...]] = (
    "era5",
    "ukv",
    "icon_d2",
    "icon_eu",
    "icon_global",
    "hres",
    "ens_mean_day0",
)
"""Every product scored at the primary setting, in the order the report lists them."""

ENS_VARIANT_PRODUCTS: Final[tuple[str, ...]] = (
    "ens_mean_day0_speed_components",
    "ens_mean_day0_direction_components",
    "ens_mean_day0_linear",
)
"""The exploratory ENS interpolations, scored at the primary setting only."""

SENSITIVITY_PRODUCTS: Final[tuple[str, ...]] = ("hres", "ens_mean_day0", "ukv", "era5")
"""The four products in the planned contrasts, rerun at the second hyperparameter setting."""

ENS_METHODS: Final[dict[str, str]] = {
    "ens_mean_day0": "components",
    "ens_mean_day0_speed_components": "speed_components",
    "ens_mean_day0_direction_components": "direction_components",
    "ens_mean_day0_linear": "linear",
}
"""Each ENS product key to its method name in the horizons study's `wind_inputs.parquet`."""

ALL_PRODUCTS: Final[tuple[str, ...]] = (*PRIMARY_PRODUCTS, *ENS_VARIANT_PRODUCTS)
"""Every product the row set carries columns for."""

KMH_PRODUCTS: Final[frozenset[str]] = frozenset(
    {"era5", "ukv", "icon_d2", "icon_eu", "icon_global"}
)
"""The products whose columns `wind_products.joined` leaves in Open-Meteo's km/h."""

PLANNED_CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("P1", "hres_wind", "ukv_wind"),
    ("P2", "ens_mean_day0_wind", "ukv_wind"),
    ("P3", "hres_wind", "era5_wind"),
)
"""The three planned contrasts, named in the plan before any fit, with their labels."""

EXPLORATORY_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("ens_mean_day0_wind", "hres_wind"),
    ("ens_mean_day0_wind", "era5_wind"),
)
"""The two exploratory contrasts between the two new products and the other products."""

ICON_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("ukv_wind", "era5_wind"),
    ("icon_d2_wind", "era5_wind"),
    ("icon_eu_wind", "era5_wind"),
    ("icon_global_wind", "era5_wind"),
)
"""UKV and the three ICON products against ERA5, exploratory."""

ENS_VARIANT_CONTRASTS: Final[tuple[tuple[str, str], ...]] = tuple(
    (f"{product}_wind", "ens_mean_day0_wind") for product in ENS_VARIANT_PRODUCTS
)
"""Each alternative ENS interpolation against the planned `components` arm, exploratory."""

PERIOD_SPLITS: Final[tuple[tuple[str, str, str], ...]] = (
    ("HRES archive source", "before 2025-10-01", "from 2025-10-01"),
    ("IFS Cycle 50r1", "before 2026-05-12", "from 2026-05-12"),
)
"""The two period splits: the name of the change, then the labels of the two periods."""

BONFERRONI_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / len(PLANNED_CONTRASTS))
"""The confidence level, in percent, of an interval adjusted for the three planned contrasts."""

CONTRAST_HEADER: Final[tuple[str, str]] = (
    (
        "| Scope | Contrast | ΔMAE (pp of capacity) | 95% interval | Excludes zero? "
        "| Folds agreeing | Rows | Months |"
    ),
    "|---|---|---|---|---|---|---|---|",
)
"""The header of every contrast table in the report."""

PUBLISHED_ENS_ERA5_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\| wind \| pooled \| all \| ens_mean_day0 − era5 \| ([+-][\d.]+) "
    r"\| \[([+-][\d.]+), ([+-][\d.]+)\] \|.*\| ([\d,]+) \|$",
    re.MULTILINE,
)
"""The horizons report's day-0 ENS-against-ERA5 wind contrast row."""


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
    n_rows: int
    n_months: int
    folds_agreeing: int | None
    n_folds: int | None


@dataclass
class IntervalLog:
    """Every interval the report prints, collected as the report is assembled."""

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


class ChecksResult(TypedDict):
    """Every pre-fit check's raw result, computed once by `run_checks`."""

    row_counts: list[tuple[str, int]]
    coverage: pl.DataFrame
    grid_check: dict[str, float]
    ens_power: dict[str, float]
    lead_table: pl.DataFrame


def _trial_area_box() -> object:
    """Load the trial-area box from `studies/weather_downloads/paths.py`, held in memory only.

    Returns:
        The box, whose `grid_points` method returns the 342-point grid's coordinates.
    """
    spec = importlib.util.spec_from_file_location("weather_download_paths", TRIAL_AREA_PATHS_MODULE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_trial_area_box()


def hres_frame(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return each wind farm's freshest-run HRES wind, hourly, in the page's four columns.

    Speeds are converted from Open-Meteo's km/h to m/s here. The direction becomes a sine and a
    cosine, as `wind_products.joined` does for every other product.

    Args:
        sites: The wind roster, carrying `site`.

    Returns:
        One row per (site, time) with `speed_hub_hres`, `direction_sin_hres`, `direction_cos_hres`
        and `speed_10m_hres`, where the hub height is 100 m.
    """
    speed, sine, cosine, surface = _wind_columns(product="hres")
    return (
        pl.read_parquet(HRES_PREVIOUS_RUNS_PATH)
        .filter(pl.col("site").is_in(sites["site"].to_list()))
        .select(
            "site",
            "time",
            pl.col("wind_speed_100m").truediv(KMH_PER_M_S).alias(speed),
            pl.col("wind_direction_100m").radians().sin().alias(sine),
            pl.col("wind_direction_100m").radians().cos().alias(cosine),
            pl.col("wind_speed_10m").truediv(KMH_PER_M_S).alias(surface),
        )
        .sort("site", "time")
    )


def ens_frame(*, product: str) -> pl.DataFrame:
    """Return the ENS ensemble-mean day-0 wind for one interpolation method, in the page's columns.

    Args:
        product: A key of `ENS_METHODS`.

    Returns:
        One row per (site, time) with `_wind_columns(product=product)`; speeds in m/s.
    """
    speed, sine, cosine, surface = _wind_columns(product=product)
    return (
        pl.read_parquet(ENS_INPUTS_PATH)
        .filter(pl.col("day") == 0, pl.col("method") == ENS_METHODS[product])
        .select(
            "site",
            "time",
            pl.col("speed_100m").alias(speed),
            pl.col("sin_100m").alias(sine),
            pl.col("cos_100m").alias(cosine),
            pl.col("speed_10m").alias(surface),
        )
        .sort("site", "time")
    )


def ens_power_check(*, base: pl.DataFrame) -> dict[str, float]:
    """Compare the horizons study's power with this row set's power on the rows they share.

    The ENS inputs carry the power the horizons study scored against. A disagreement would mean the
    two studies label the hour differently, so the ENS wind would sit an hour away from the power.

    Args:
        base: The row set before the ENS join, carrying `site`, `time` and `power_mw`.

    Returns:
        `n_shared_rows` and `max_abs_diff_mw`.
    """
    theirs = (
        pl.read_parquet(ENS_INPUTS_PATH)
        .filter(pl.col("day") == 0, pl.col("method") == "components")
        .drop_nulls("power_mw")
        .select("site", "time", theirs=pl.col("power_mw").cast(pl.Float64))
    )
    shared = base.select("site", "time", ours=pl.col("power_mw").cast(pl.Float64)).join(
        theirs, on=["site", "time"], how="inner"
    )
    return {
        "n_shared_rows": float(shared.height),
        "max_abs_diff_mw": float(
            shared.select((pl.col("ours") - pl.col("theirs")).abs().max()).item()
        ),
    }


MAX_ENS_POWER_DIFF_MW: Final[float] = 1e-3
"""`run_checks` fails if the horizons study's power and this row set's power differ by more."""


def _base_rows(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return the page's rows from 1 December 2024, before eras and folds are assigned.

    Args:
        sites: The wind roster.

    Returns:
        `common_rows(joined(...))` restricted to `ROW_SET_START` onward.
    """
    return common_rows(frame=joined(sites=sites)).filter(pl.col("time") >= ROW_SET_START)


def _count_by_site(*, frame: pl.DataFrame) -> dict[str, int]:
    """Return the number of rows per site.

    Args:
        frame: Rows carrying `site`.

    Returns:
        Each site's row count, in site order.
    """
    counted = frame.group_by("site").agg(n=pl.len()).sort("site")
    return dict(zip(counted["site"].to_list(), counted["n"].to_list(), strict=True))


def joined_row_set(*, sites: pl.DataFrame) -> tuple[pl.DataFrame, list[tuple[str, int]]]:
    """Return the row set with ENS and HRES joined, raising if a join loses a row.

    Args:
        sites: The wind roster.

    Returns:
        The row set carrying every product's columns, without eras or folds, and the row count
        after each step, for the report.

    Raises:
        ValueError: If a join loses rows, or the per-farm counts differ from `EXPECTED_ROWS`.
    """
    frame = _base_rows(sites=sites)
    counts = [("page rows from 2024-12-01", frame.height)]
    per_site = _count_by_site(frame=frame)
    if per_site != EXPECTED_ROWS:
        msg = f"the page's rows from 2024-12-01 are {per_site}, not the planned {EXPECTED_ROWS}"
        raise ValueError(msg)
    for name, addition in (
        *((f"{product} joined", ens_frame(product=product)) for product in ENS_METHODS),
        ("hres joined", hres_frame(sites=sites)),
    ):
        frame = frame.join(addition, on=["site", "time"], how="inner")
        counts.append((name, frame.height))
        if frame.height != counts[0][1]:
            msg = f"joining {name} lost {counts[0][1] - frame.height} rows"
            raise ValueError(msg)
    return frame.sort("site", "time"), counts


def with_three_eras(*, frame: pl.DataFrame, fold_offsets: dict[int, int]) -> pl.DataFrame:
    """Label each row's era, add the era feature, and cut folds inside each era.

    The date filter has already run, so the folds are cut on the row set the arms are fitted on.
    Each era's fold numbers are rotated by `fold_offsets` afterward, which leaves the folds
    contiguous within an era.

    Args:
        frame: The common rows, carrying `month`.
        fold_offsets: Each `era_code` to how far its fold numbers are rotated.

    Returns:
        The frame with `era`, `era_code` and `fold`.
    """
    second, third = ERA_START_MONTHS
    era_code = (pl.col("month") >= second).cast(pl.Int8) + (pl.col("month") >= third).cast(pl.Int8)
    labelled = frame.with_columns(era_code=era_code).with_columns(
        era=pl.col("era_code").cast(pl.String)
    )
    folded = assign_folds(dataset=labelled, by=("site", "era"))
    rotation = pl.col("era_code").replace_strict(fold_offsets, return_dtype=pl.Int32)
    return folded.with_columns(fold=(pl.col("fold") + rotation) % N_FOLDS)


def calendar_month_coverage(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Count, for each held-out calendar month, the training rows that carry it.

    For every (site, fold, calendar month) with rows in the fold, counts the rows of the same site
    and calendar month that lie in the other folds, from any era or year. A count of 0 means the
    fitted model has seen no row of that season.

    Args:
        frame: The row set carrying `site`, `fold` and `time`.

    Returns:
        One row per (site, fold, calendar_month) with `n_scored`, `n_train`, `n_years` (how many
        distinct years of the site's rows carry that calendar month) and `covered`.
    """
    rows = frame.select(
        "site",
        "fold",
        calendar_month=pl.col("time").dt.month(),
        year=pl.col("time").dt.year(),
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


def uncovered_months(*, coverage: pl.DataFrame) -> pl.DataFrame:
    """Return the rows of the coverage table that no fold design could fix, and the ones that fail.

    Args:
        coverage: `calendar_month_coverage`'s result.

    Returns:
        The rows with no training row for a calendar month that occurs in more than one year: the
        failures the fold design could have avoided.
    """
    return coverage.filter(~pl.col("covered"), pl.col("n_years") > 1)


def _raise_on_uncovered_months(*, coverage: pl.DataFrame) -> None:
    """Raise if a held-out calendar month that occurs in two years has no training row.

    Args:
        coverage: `calendar_month_coverage`'s result.

    Raises:
        ValueError: Naming the first failing (site, fold, calendar month) rows.
    """
    failures = uncovered_months(coverage=coverage)
    if failures.height:
        msg = (
            f"{failures.height} (site, fold, calendar month) cells hold out a calendar month that "
            f"occurs in two years and leave no training row for it: {failures.head(5).to_dicts()}"
        )
        raise ValueError(msg)


def _coverage_lines(*, coverage: pl.DataFrame) -> list[str]:
    """Render the calendar-month coverage table as markdown.

    Args:
        coverage: `calendar_month_coverage`'s result.

    Returns:
        Markdown lines: a summary, then one row per (site, fold, calendar month).
    """
    single_year = coverage.filter(pl.col("n_years") == 1)
    single_months = sorted(single_year["calendar_month"].unique().to_list())
    lines = [
        "#### Calendar-month coverage of the folds, checked before any fit",
        "",
        (
            f"Cells (site, fold, calendar month) checked: {coverage.height}. Cells with no "
            f"training row for a calendar month that occurs in two years: "
            f"{uncovered_months(coverage=coverage).height}. Calendar months that occur in one "
            f"year of the row set only, which no fold design can cover: {single_months}."
        ),
        "",
        "| Site | Fold | Calendar month | Rows scored | Training rows | Years of that month |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {row['site']} | {row['fold']} | {row['calendar_month']} | {row['n_scored']:,} "
        f"| {row['n_train']:,} | {row['n_years']} |"
        for row in coverage.iter_rows(named=True)
    ]
    return lines


def _nearest_grid_points(*, sites: pl.DataFrame, points: pl.DataFrame) -> dict[str, list[int]]:
    """Rank the grid points by distance from each farm, nearest first, keeping the nearest few.

    Args:
        sites: The wind roster, carrying `site`, `latitude`, `longitude`.
        points: The grid, carrying `point_id`, `latitude`, `longitude`.

    Returns:
        Each site to its `GRID_NEAREST_RANK` nearest `point_id`s.
    """
    ranked: dict[str, list[int]] = {}
    for site in sites.iter_rows(named=True):
        latitude = np.radians(points["latitude"].to_numpy())
        longitude = np.radians(points["longitude"].to_numpy())
        site_latitude = math.radians(site["latitude"])
        site_longitude = math.radians(site["longitude"])
        haversine = (
            np.sin((latitude - site_latitude) / 2) ** 2
            + np.cos(latitude)
            * math.cos(site_latitude)
            * np.sin((longitude - site_longitude) / 2) ** 2
        )
        order = np.argsort(haversine)[:GRID_NEAREST_RANK]
        ranked[site["site"]] = points["point_id"].to_numpy()[order].tolist()
    return ranked


def hres_grid_check(*, sites: pl.DataFrame) -> dict[str, float]:
    """Check the Previous Runs HRES speeds against the grid file's nearest points, pooled.

    For each farm, one of its `GRID_NEAREST_RANK` nearest grid points has to reproduce the Previous
    Runs bare `wind_speed_100m` and `wind_speed_10m` to within `GRID_TOLERANCE_KMH` on every hour,
    both before and from 1 October 2025. Open-Meteo picks its own model cell for a coordinate, so
    the exact match is not always the nearest grid point.

    Args:
        sites: The wind roster, carrying `site`, `latitude`, `longitude`.

    Returns:
        The pooled statistics: `max_rank` (1 is nearest), `n_hours`, and the largest hourly
        disagreement before and from 1 October 2025, in km/h. No coordinate or per-farm distance.

    Raises:
        ValueError: If any farm has no matching point among its nearest `GRID_NEAREST_RANK`.
    """
    points = _trial_area_box().grid_points(spacing_deg=0.05)  # ty: ignore[unresolved-attribute]
    ranked = _nearest_grid_points(sites=sites, points=points)
    candidates = sorted({point for ids in ranked.values() for point in ids})
    grid = (
        pl.scan_parquet(HRES_GRID_PATH)
        .filter(pl.col("point_id").is_in(candidates))
        .select("point_id", "time", "wind_speed_10m", "wind_speed_100m")
        .collect()
    )
    previous = pl.read_parquet(HRES_PREVIOUS_RUNS_PATH).select(
        "site", "time", "wind_speed_10m", "wind_speed_100m"
    )
    ranks: list[int] = []
    largest = {"before": 0.0, "from": 0.0}
    n_hours = 0
    for site, point_ids in ranked.items():
        mine = previous.filter(pl.col("site") == site)
        for rank, point_id in enumerate(point_ids, start=1):
            compared = mine.join(
                grid.filter(pl.col("point_id") == point_id), on="time", suffix="_grid"
            ).with_columns(
                difference=pl.max_horizontal(
                    (pl.col("wind_speed_10m") - pl.col("wind_speed_10m_grid")).abs(),
                    (pl.col("wind_speed_100m") - pl.col("wind_speed_100m_grid")).abs(),
                ),
                period=pl.when(pl.col("time") >= HRES_ARCHIVE_CHANGE)
                .then(pl.lit("from"))
                .otherwise(pl.lit("before")),
            )
            worst = {
                period: float(
                    compared.filter(pl.col("period") == period)
                    .select(pl.col("difference").max())
                    .item()
                )
                for period in ("before", "from")
            }
            if compared.height and max(worst.values()) <= GRID_TOLERANCE_KMH:
                ranks.append(rank)
                n_hours += compared.height
                largest = {period: max(largest[period], worst[period]) for period in largest}
                break
        else:
            msg = f"no grid point among the nearest {GRID_NEAREST_RANK} matches HRES at a farm"
            raise ValueError(msg)
    return {
        "max_rank": float(max(ranks)),
        "n_farms": float(len(ranks)),
        "n_hours": float(n_hours),
        "max_diff_before_kmh": largest["before"],
        "max_diff_from_kmh": largest["from"],
    }


def hres_lead_table(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Measure the hour-to-hour change of HRES's freshest-run wind by UTC hour of day.

    A stitched series of forecast runs jumps where one run hands over to the next, so the hours
    with the largest mean change mark where a new run starts.

    Args:
        sites: The wind roster.

    Returns:
        One row per (variable, period, hour) with `mean_abs_change_kmh` and `ratio`, the change
        over the median across the 24 hours of the same variable and period.
    """
    previous = (
        pl.read_parquet(HRES_PREVIOUS_RUNS_PATH)
        .filter(pl.col("site").is_in(sites["site"].to_list()), pl.col("time") >= ROW_SET_START)
        .sort("site", "time")
    )
    tables: list[pl.DataFrame] = []
    for variable in ("wind_speed_100m", "wind_speed_10m"):
        changes = (
            previous.with_columns(
                change=(pl.col(variable) - pl.col(variable).shift(1).over("site")).abs(),
                gap=pl.col("time") - pl.col("time").shift(1).over("site"),
            )
            .filter(pl.col("gap") == pl.duration(hours=1))
            .with_columns(
                period=pl.when(pl.col("time") >= HRES_ARCHIVE_CHANGE)
                .then(pl.lit("from 2025-10-01"))
                .otherwise(pl.lit("before 2025-10-01"))
            )
        )
        tables.append(
            changes.group_by("period", hour=pl.col("time").dt.hour())
            .agg(mean_abs_change_kmh=pl.col("change").mean())
            .with_columns(
                variable=pl.lit(variable),
                ratio=pl.col("mean_abs_change_kmh")
                / pl.col("mean_abs_change_kmh").median().over("period"),
            )
        )
    return pl.concat(tables).sort("variable", "period", "hour")


def _lead_lines(*, lead_table: pl.DataFrame) -> list[str]:
    """Render the HRES served-lead evidence and the statement it supports.

    Args:
        lead_table: `hres_lead_table`'s result.

    Returns:
        Markdown lines.
    """
    lines = [
        "#### HRES served-lead evidence: hour-to-hour change by UTC hour",
        "",
        (
            "Mean absolute hour-to-hour change in the freshest-run bare column at the arriving "
            "hour, over the median of the 24 hourly means for the same variable and period. A "
            "ratio well above 1 marks an hour where one forecast run hands over to the next. "
            f"Ratios of {JUMP_RATIO_THRESHOLD:.2f} or more are in bold."
        ),
        "",
    ]
    for variable in ("wind_speed_100m", "wind_speed_10m"):
        lines += [
            f"`{variable}`",
            "",
            "| UTC hour | Ratio before 2025-10-01 | Ratio from 2025-10-01 |",
            "|---|---|---|",
        ]
        for hour in range(24):
            cells: list[str] = []
            for period in LEAD_PERIODS:
                ratio = float(
                    lead_table.filter(
                        pl.col("variable") == variable,
                        pl.col("period") == period,
                        pl.col("hour") == hour,
                    )["ratio"].item()
                )
                cells.append(
                    f"**{ratio:.2f}**" if ratio >= JUMP_RATIO_THRESHOLD else f"{ratio:.2f}"
                )
            lines.append(f"| {hour:02d} | {cells[0]} | {cells[1]} |")
        lines.append("")
    for period in LEAD_PERIODS:
        jump_hours = sorted(
            set(
                lead_table.filter(
                    pl.col("period") == period, pl.col("ratio") >= JUMP_RATIO_THRESHOLD
                )["hour"].to_list()
            )
        )
        expected = PLAN_EXPECTED_JUMP_HOURS[period]
        reached = [hour for hour in expected if hour in jump_hours]
        lines.append(
            f"- {period}: UTC hours with a ratio of {JUMP_RATIO_THRESHOLD:.2f} or more in either "
            f"variable: {jump_hours}. Of the hours the plan expected, {expected}, "
            f"{len(reached)} of {len(expected)} reach the threshold: {reached}."
        )
    lines += [
        "",
        (
            "Lead statement. The plan expected run handovers at the hours listed above, which "
            "would put HRES's served lead at 1 to 12 hours before 1 October 2025 and at 0 to 5 "
            "hours from it. The hours that reach the threshold are what the data show. A large "
            "change at an hour can also come from the wind's own daily cycle, and the 10 m speed "
            "changes most in the evening in both periods, so an hour that reaches the threshold is "
            "not proof of a handover. The page states the served lead only as far as the hours "
            "that do reach the threshold support it."
        ),
    ]
    return lines


def run_checks(
    *, sites: pl.DataFrame, frame: pl.DataFrame, counts: list[tuple[str, int]]
) -> ChecksResult:
    """Run every pre-fit check once, before any arm is fitted.

    Args:
        sites: The wind roster.
        frame: The row set with eras and folds.
        counts: `joined_row_set`'s row counts.

    Returns:
        Every check's raw result.
    """
    return {
        "row_counts": counts,
        "coverage": calendar_month_coverage(frame=frame),
        "grid_check": hres_grid_check(sites=sites),
        "ens_power": ens_power_check(base=_base_rows(sites=sites)),
        "lead_table": hres_lead_table(sites=sites),
    }


def _raise_on_failed_checks(*, checks: ChecksResult) -> None:
    """Raise if any pre-fit check fails, before any arm is fitted on rows the checks distrust.

    Args:
        checks: `run_checks`'s result.

    Raises:
        ValueError: Naming every failed check.
    """
    _raise_on_uncovered_months(coverage=checks["coverage"])
    ens_power = checks["ens_power"]
    if not ens_power["n_shared_rows"] or ens_power["max_abs_diff_mw"] > MAX_ENS_POWER_DIFF_MW:
        msg = (
            f"the ENS inputs' power differs from this row set's power on {ens_power}; the two "
            "studies may label the hour differently"
        )
        raise ValueError(msg)


def _checks_lines(*, checks: ChecksResult) -> list[str]:
    """Render the row counts, the HRES cross-check and the ENS power check as markdown.

    Args:
        checks: `run_checks`'s result.

    Returns:
        Markdown lines.
    """
    grid = checks["grid_check"]
    power = checks["ens_power"]
    lines = ["#### Row counts at each join", "", "| Step | Rows |", "|---|---|"]
    lines += [f"| {name} | {count:,} |" for name, count in checks["row_counts"]]
    lines += [
        "",
        "#### Checks run before any fit",
        "",
        (
            f"- HRES cross-check against the grid file: at each of the {grid['n_farms']:.0f} "
            f"farms, one of the {GRID_NEAREST_RANK} nearest grid points reproduced the Previous "
            "Runs "
            f"bare `wind_speed_100m` and `wind_speed_10m` on every hour ({grid['n_hours']:,.0f} "
            f"farm-hours). The largest rank needed, pooled over farms, was {grid['max_rank']:.0f} "
            "(1 is the nearest point). The largest hourly disagreement was "
            f"{grid['max_diff_before_kmh']:.3f} km/h before 2025-10-01 and "
            f"{grid['max_diff_from_kmh']:.3f} km/h from it; the tolerance was 0.05 km/h."
        ),
        (
            f"- ENS power against this row set's power, on {power['n_shared_rows']:,.0f} shared "
            f"farm-hours: the largest absolute difference was {power['max_abs_diff_mw']:.6f} MW."
        ),
    ]
    return lines


def jobs() -> list[Job]:
    """Return every arm's job: the primary setting for every product, the second for four.

    Returns:
        One job per (arm, setting).
    """
    job_list: list[Job] = []
    for product in (*PRIMARY_PRODUCTS, *ENS_VARIANT_PRODUCTS):
        columns = (*SHARED_FEATURES, *_wind_columns(product=product))
        job_list.append(
            (f"{product}_wind", "pooled", "power_mw", columns, PRIMARY_HYPER_PARAMETERS, False)
        )
    for product in SENSITIVITY_PRODUCTS:
        columns = (*SHARED_FEATURES, *_wind_columns(product=product))
        job_list.append(
            (
                f"{product}_wind",
                "sensitivity",
                "power_mw",
                columns,
                SENSITIVITY_HYPER_PARAMETERS,
                False,
            )
        )
    return job_list


def _fingerprint(*, frame: pl.DataFrame, job_list: list[Job]) -> str:
    """Return a hash covering every row's values, every job's columns, and the seeds.

    `--report-only` refuses to reuse a saved `losses.parquet` when this does not match, so a code
    change that moves the row set, a feature's value, a fold, a column, a seed, or a hyperparameter
    setting cannot silently mix its fits with a previous run's. Every float column is cast to
    `Float32` before hashing, so the last-bit noise of a parallel mean over 51 members cannot flip
    the fingerprint; the saved `losses.parquet` keeps full precision.

    Args:
        frame: The row set every job is fitted on, including `power_mw` and `fold`.
        job_list: Every job this run means to fit.

    Returns:
        A hex digest.
    """
    ordered = frame.select(sorted(frame.columns)).sort("site", "time")
    float_columns = [name for name, dtype in ordered.schema.items() if dtype.is_float()]
    stable = ordered.cast(dict.fromkeys(float_columns, pl.Float32))
    row_hashes = stable.hash_rows(seed=0).to_list()
    payload = repr(
        (
            row_hashes,
            [
                (arm, setting, target, tuple(features), tuple(sorted(hyper_parameters.items())))
                for arm, setting, target, features, hyper_parameters, _ in job_list
            ],
            SEEDS,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _bootstrap_percentiles(
    *, differences: np.ndarray, months: np.ndarray, percentiles: tuple[float, float]
) -> tuple[float, float]:
    """Resample whole months and a seed, as `studies.bootstrap` does, at any two percentiles.

    The random stream has the same shape as `studies.bootstrap._resample_bounds`: a seed draw, then
    a month draw, per resample, from a stream seeded with `BOOTSTRAP_SEED`. `_bonferroni_lines`
    asserts that this reproduces `bootstrap_difference`'s own interval at 2.5 and 97.5.

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


def contrast_line(
    *,
    losses: pl.DataFrame,
    treatment: str,
    reference: str,
    label: str,
    section: str,
    setting: str,
    log: IntervalLog,
) -> str:
    """Return one markdown contrast row, and record its interval in `log`.

    Args:
        losses: Per-row losses holding both arms, restricted to the scope wanted.
        treatment: The arm whose error is being compared.
        reference: The arm it is compared against.
        label: The scope label for the first column.
        section: The report section, saved with the interval.
        setting: `pooled` or `sensitivity`, saved with the interval.
        log: Where the interval is recorded.

    Returns:
        The table row.
    """
    interval = bootstrap_difference(
        losses=losses, treatment=treatment, reference=reference, metric=METRIC
    )
    folds = per_fold_differences(
        losses=losses, treatment=treatment, reference=reference, metric=METRIC
    )
    agreeing = sum(np.sign(value) == np.sign(interval["difference"]) for value in folds)
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
            "n_rows": interval["n_rows"],
            "n_months": interval["n_months"],
            "folds_agreeing": int(agreeing),
            "n_folds": len(folds),
        }
    )
    return (
        f"| {label} | {treatment} − {reference} | {difference:+.3f} | "
        f"[{lower:+.3f}, {upper:+.3f}] | {'**yes**' if excludes else 'no'} | "
        f"{agreeing} of {len(folds)} | {interval['n_rows']:,} | {interval['n_months']} |"
    )


def _contrast_table(
    *,
    losses: pl.DataFrame,
    pairs: tuple[tuple[str, str], ...],
    label: str,
    section: str,
    setting: str,
    log: IntervalLog,
) -> list[str]:
    """Render a contrast table over `pairs`, on one scope of the losses.

    Args:
        losses: Per-row losses holding every arm the pairs name.
        pairs: Each (treatment, reference).
        label: The scope label.
        section: The report section, saved with each interval.
        setting: `pooled` or `sensitivity`.
        log: Where the intervals are recorded.

    Returns:
        Markdown lines: the header and one row per pair.
    """
    return [
        *CONTRAST_HEADER,
        *(
            contrast_line(
                losses=losses,
                treatment=treatment,
                reference=reference,
                label=label,
                section=section,
                setting=setting,
                log=log,
            )
            for treatment, reference in pairs
        ),
    ]


def _planned_pairs() -> tuple[tuple[str, str], ...]:
    """Return the planned contrasts as (treatment, reference) pairs.

    Returns:
        The pairs of `PLANNED_CONTRASTS` without their labels.
    """
    return tuple((treatment, reference) for _, treatment, reference in PLANNED_CONTRASTS)


def _leaderboard_lines(
    *, losses: pl.DataFrame, scope_label: str, section: str, setting: str, log: IntervalLog
) -> list[str]:
    """Render every arm's absolute error and its interval, for one scope of the losses.

    Args:
        losses: Per-row losses at one setting, restricted to the scope wanted.
        scope_label: The scope label, saved with the interval.
        section: The report section, saved with each interval.
        setting: `pooled` or `sensitivity`.
        log: Where the intervals are recorded.

    Returns:
        Markdown lines.
    """
    lines = [
        "| Arm | MAE (pp of capacity) | 95% interval | Rows | Months |",
        "|---|---|---|---|---|",
    ]
    for arm in sorted(
        losses["arm"].unique().to_list(), key=lambda name: _mae(losses=losses, arm=name)
    ):
        interval = bootstrap_absolute(losses=losses, arm=arm, metric=METRIC)
        value, lower, upper = (
            interval[key] * PERCENTAGE_POINTS for key in ("value", "lower_95", "upper_95")
        )
        log.records.append(
            {
                "section": section,
                "setting": setting,
                "scope": scope_label,
                "treatment": arm,
                "reference": None,
                "value": value,
                "lower": lower,
                "upper": upper,
                "level": 95.0,
                "n_rows": interval["n_rows"],
                "n_months": interval["n_months"],
                "folds_agreeing": None,
                "n_folds": None,
            }
        )
        lines.append(
            f"| {arm} | {value:.3f} | [{lower:.3f}, {upper:.3f}] | {interval['n_rows']:,} "
            f"| {interval['n_months']} |"
        )
    return lines


def _bonferroni_lines(*, losses: pl.DataFrame, setting: str, log: IntervalLog) -> list[str]:
    """Render the planned contrasts with intervals adjusted for testing three at once.

    Each interval is a percentile interval of the same month-and-seed resampling, at the level
    `BONFERRONI_LEVEL`. The function first asserts that its resampler reproduces
    `bootstrap_difference`'s own 95% interval on the first contrast, so the two intervals differ
    only in level.

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
        "| Contrast | ΔMAE (pp of capacity) | Adjusted interval | Excludes zero? | Months |",
        "|---|---|---|---|---|",
    ]
    for index, (name, treatment, reference) in enumerate(PLANNED_CONTRASTS):
        differences, months = paired_differences(
            losses=losses, treatment=treatment, reference=reference, metric=METRIC
        )
        if index == 0:
            reference_interval = bootstrap_difference(
                losses=losses, treatment=treatment, reference=reference, metric=METRIC
            )
            mine = _bootstrap_percentiles(
                differences=differences, months=months, percentiles=(2.5, 97.5)
            )
            expected = (reference_interval["lower_95"], reference_interval["upper_95"])
            if not np.allclose(mine, expected, rtol=0.0, atol=1e-12):
                msg = f"the resampler gives {mine}, not bootstrap_difference's {expected}"
                raise ValueError(msg)
        lower, upper = (
            value * PERCENTAGE_POINTS
            for value in _bootstrap_percentiles(
                differences=differences, months=months, percentiles=(tail, 100.0 - tail)
            )
        )
        difference = float(differences.mean()) * PERCENTAGE_POINTS
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
                "n_months": len(np.unique(months)),
                "folds_agreeing": None,
                "n_folds": None,
            }
        )
        excludes = lower > 0.0 or upper < 0.0
        lines.append(
            f"| {name}: {treatment} − {reference} | {difference:+.3f} "
            f"| [{lower:+.3f}, {upper:+.3f}] | {'**yes**' if excludes else 'no'} "
            f"| {len(np.unique(months))} |"
        )
    return lines


def _period_lines(*, pooled: pl.DataFrame, log: IntervalLog) -> list[str]:
    """Render the planned and ENS contrasts before and after each change, from the saved losses.

    Args:
        pooled: Per-row losses at the primary setting.
        log: Where the intervals are recorded.

    Returns:
        Markdown lines.
    """
    boundaries = {
        "HRES archive source": HRES_ARCHIVE_CHANGE,
        "IFS Cycle 50r1": IFS_CYCLE_50R1,
    }
    pairs = (*_planned_pairs(), *EXPLORATORY_CONTRASTS)
    lines = [
        "#### Period splits (exploratory)",
        "",
        (
            "Each row is a contrast on the rows of one period only, from the saved losses. Folds "
            "were cut before the split, so a period's models were trained on rows of both periods. "
            "The period from 2026-05-12 holds about 4 months, so its intervals under-cover: "
            "resampling 4 months cannot represent the month-to-month spread."
        ),
        "",
    ]
    for change, before_label, after_label in PERIOD_SPLITS:
        boundary = boundaries[change]
        before = pooled.filter(pl.col("time") < boundary)
        after = pooled.filter(pl.col("time") >= boundary)
        before_months = before.select(pl.col("month").n_unique()).item()
        after_months = after.select(pl.col("month").n_unique()).item()
        lines += [
            (
                f"`{change}`: {before_label} holds {before_months} calendar months and "
                f"{after_label} holds {after_months}."
            ),
            "",
        ]
        for label, part in ((before_label, before), (after_label, after)):
            lines += _contrast_table(
                losses=part,
                pairs=pairs,
                label=label,
                section=f"period: {change}",
                setting="pooled",
                log=log,
            )
            lines.append("")
    return lines


def _servable_lines(*, pooled: pl.DataFrame, log: IntervalLog) -> list[str]:
    """Render the ENS contrasts on morning and afternoon labels, exploratory.

    Wind power for label T covers T minus 30 minutes to T plus 30 minutes, so labels 00 to 08 UTC
    end before about 09:00 UTC, when the 00 UTC run is available from Dynamical.org's archive, and
    labels 10 to 23 do not. Label 09 is dropped.

    Args:
        pooled: Per-row losses at the primary setting.
        log: Where the intervals are recorded.

    Returns:
        Markdown lines.
    """
    pairs = (("ens_mean_day0_wind", "ukv_wind"), *EXPLORATORY_CONTRASTS)
    lines = [
        "#### ENS contrasts by label hour, the servable-hours split (exploratory)",
        "",
        (
            "Label hours 00 to 08 UTC hold power that ended before about 09:00 UTC, when the 00 "
            "UTC run is available from Dynamical.org's archive. Label hours 10 to 23 UTC hold "
            "power "
            "that ended after it. Label 09 is dropped. The split also separates ENS leads 0 to 8 "
            "hours from leads 10 to 23 hours, and morning from afternoon, so it is not a clean "
            "test of servability."
        ),
        "",
    ]
    for label, hours in zip(
        ("labels 00-08 UTC", "labels 10-23 UTC"), SERVABLE_LABEL_HOURS, strict=True
    ):
        part = pooled.filter(pl.col("time").dt.hour().is_in(list(hours)))
        lines += _contrast_table(
            losses=part, pairs=pairs, label=label, section="servable", setting="pooled", log=log
        )
        lines.append("")
    return lines


def _horizons_lines(*, pooled: pl.DataFrame, log: IntervalLog) -> list[str]:
    """Render the ENS-against-ERA5 contrast beside the horizons study's published figure.

    Args:
        pooled: Per-row losses at the primary setting.
        log: Where the intervals are recorded.

    Returns:
        Markdown lines. The published figure is re-derived from the horizons report's own table.

    Raises:
        ValueError: If the horizons report holds no day-0 ENS-against-ERA5 wind row.
    """
    match = PUBLISHED_ENS_ERA5_PATTERN.search(HORIZONS_REPORT_PATH.read_text())
    if match is None:
        msg = f"{HORIZONS_REPORT_PATH} holds no day-0 ENS-against-ERA5 wind contrast row"
        raise ValueError(msg)
    difference, lower, upper, rows = match.groups()
    return [
        "#### ENS against ERA5 beside the horizons study's published figure (exploratory)",
        "",
        (
            "The horizons study's report gives `ens_mean_day0 − era5` for wind, pooled setting, "
            f"as {difference} [{lower}, {upper}] pp of capacity on {rows} rows from 2024-08-12. "
            "The row below re-estimates it on this study's shorter row set."
        ),
        "",
        *_contrast_table(
            losses=pooled,
            pairs=(("ens_mean_day0_wind", "era5_wind"),),
            label="all",
            section="horizons",
            setting="pooled",
            log=log,
        ),
    ]


def _arm_columns_lines(*, job_list: list[Job]) -> list[str]:
    """Render every fitted arm's feature columns and their count, once per arm, as markdown.

    A reviewer checks this against the plan, since an arm can silently lose a column (see the
    `study` skill). Raises unless every arm holds the same number of columns.

    Args:
        job_list: Every job `jobs()` returns.

    Returns:
        Markdown lines.

    Raises:
        ValueError: If two arms hold different numbers of feature columns.
    """
    seen: dict[str, tuple[str, ...]] = {}
    for arm, _, _, columns, _, _ in job_list:
        seen.setdefault(arm, columns)
    counts = {len(columns) for columns in seen.values()}
    if len(counts) != 1:
        sizes = {arm: len(columns) for arm, columns in seen.items()}
        msg = f"arms hold different numbers of feature columns: {sizes}"
        raise ValueError(msg)
    lines = ["#### Every arm's feature columns", ""]
    lines += [
        f"- `{arm}` ({len(columns)} columns): {', '.join(f'`{column}`' for column in columns)}"
        for arm, columns in seen.items()
    ]
    return lines


def _mean_speed_lines(*, frame: pl.DataFrame) -> list[str]:
    """Render each product's mean hub-height and 10 m wind speed in m/s.

    Args:
        frame: The row set with every product's columns. Open-Meteo's km/h columns are converted;
            the HRES and ENS columns are already in m/s.

    Returns:
        Markdown lines.
    """
    lines = [
        "#### Mean wind speed of each product on the common rows (m/s)",
        "",
        "| Product | Hub height (m) | Mean hub-height speed | Mean 10 m speed |",
        "|---|---|---|---|",
    ]
    for product in ALL_PRODUCTS:
        speed, _, _, surface = _wind_columns(product=product)
        divisor = KMH_PER_M_S if product in KMH_PRODUCTS else 1.0
        hub_mean = float(frame.select(pl.col(speed).mean()).item()) / divisor
        surface_mean = float(frame.select(pl.col(surface).mean()).item()) / divisor
        height = _hub_height_m(product=product)
        lines.append(f"| {product} | {height} | {hub_mean:.2f} | {surface_mean:.2f} |")
    return lines


def _row_lines(*, frame: pl.DataFrame, fingerprint: str) -> list[str]:
    """Render the row set's size, months, farms and fingerprint.

    Args:
        frame: The row set with eras and folds.
        fingerprint: `_fingerprint`'s digest.

    Returns:
        Markdown lines.
    """
    per_site = (
        frame.group_by("site").agg(n=pl.len(), months=pl.col("month").n_unique()).sort("site")
    )
    lines = [
        f"Row fingerprint (Float32-cast floats): `{fingerprint}`.",
        "",
        "| Farm | Rows | Calendar months |",
        "|---|---|---|",
    ]
    lines += [
        f"| {row['site']} | {row['n']:,} | {row['months']} |"
        for row in per_site.iter_rows(named=True)
    ]
    eras = frame.group_by("era_code").agg(
        n=pl.len(), first=pl.col("month").min(), last=pl.col("month").max()
    )
    lines += ["", "| Era | Rows | First month | Last month |", "|---|---|---|---|"]
    lines += [
        f"| {row['era_code']} | {row['n']:,} | {row['first']} | {row['last']} |"
        for row in eras.sort("era_code").iter_rows(named=True)
    ]
    return lines


def _report(
    *,
    frame: pl.DataFrame,
    losses: pl.DataFrame,
    sites: pl.DataFrame,
    job_list: list[Job],
    checks: ChecksResult,
    fingerprint: str,
    log: IntervalLog,
    script_commit: str,
) -> str:
    """Assemble the markdown report.

    Args:
        frame: The row set with eras and folds.
        losses: Every arm's losses, at every setting.
        sites: The wind roster, for the geometry lines.
        job_list: Every job `jobs()` returns.
        checks: `run_checks`'s result.
        fingerprint: `_fingerprint`'s digest.
        log: Where every printed interval is recorded.
        script_commit: The commit at which the script was committed before its first fit.

    Returns:
        The report.
    """
    site_labels = sorted(frame["site"].unique().to_list())
    pooled = losses.filter(pl.col("setting") == "pooled")
    sensitivity = losses.filter(pl.col("setting") == "sensitivity")
    planned = _planned_pairs()
    lines = [
        (
            f"### ECMWF ENS and HRES wind against five other products, on {frame.height:,} common "
            f"farm-hours ({frame['time'].min():%Y-%m-%d} to {frame['time'].max():%Y-%m-%d})"
        ),
        "",
        (
            f"Script committed at `{script_commit}` before its first fit. Three wind farms are few "
            "independent sites, so every pooled interval below rests on three farms that share "
            "their weather, and resamples whole calendar months and one fitting seed."
        ),
        "",
        *_row_lines(frame=frame, fingerprint=fingerprint),
        "",
        *_arm_columns_lines(job_list=job_list),
        "",
        *_checks_lines(checks=checks),
        "",
        *_coverage_lines(coverage=checks["coverage"]),
        "",
        *_lead_lines(lead_table=checks["lead_table"]),
        "",
        *_mean_speed_lines(frame=frame),
        "",
        "#### Absolute error of every arm, pooled over three farms, primary setting",
        "",
        *_leaderboard_lines(
            losses=pooled, scope_label="all", section="leaderboard", setting="pooled", log=log
        ),
        "",
        (
            "Mean absolute error as a percentage of each site's capacity. The interval is a 95% "
            "bound from resampling whole months and a fitting seed."
        ),
        "",
        "#### Absolute error of every arm, pooled over three farms, second setting",
        "",
        *_leaderboard_lines(
            losses=sensitivity,
            scope_label="all",
            section="leaderboard",
            setting="sensitivity",
            log=log,
        ),
    ]
    for site in site_labels:
        lines += [
            "",
            f"#### Absolute error of every arm at farm {site}, primary setting",
            "",
            *_leaderboard_lines(
                losses=pooled.filter(pl.col("site") == site),
                scope_label=f"site {site}",
                section="leaderboard",
                setting="pooled",
                log=log,
            ),
        ]
    lines += [
        "",
        "#### Planned contrasts P1 to P3, pooled over three farms, primary setting",
        "",
        (
            "P1 is `hres_wind − ukv_wind`, P2 is `ens_mean_day0_wind − ukv_wind` and P3 is "
            "`hres_wind − era5_wind`. A positive difference means the first arm's error is larger."
        ),
        "",
        *_contrast_table(
            losses=pooled, pairs=planned, label="all", section="planned", setting="pooled", log=log
        ),
        "",
        "#### Planned contrasts P1 to P3, pooled over three farms, second setting",
        "",
        *_contrast_table(
            losses=sensitivity,
            pairs=planned,
            label="all",
            section="planned",
            setting="sensitivity",
            log=log,
        ),
        "",
        "#### Planned contrasts P1 to P3 by farm, primary setting (exploratory)",
        "",
    ]
    lines += list(CONTRAST_HEADER)
    for site in site_labels:
        lines += [
            contrast_line(
                losses=pooled.filter(pl.col("site") == site),
                treatment=treatment,
                reference=reference,
                label=f"site {site}",
                section="planned by farm",
                setting="pooled",
                log=log,
            )
            for treatment, reference in planned
        ]
    lines += [
        "",
        (
            f"#### Planned contrasts with {BONFERRONI_LEVEL:.2f}% intervals, adjusted for three "
            "contrasts (exploratory), primary setting"
        ),
        "",
        *_bonferroni_lines(losses=pooled, setting="pooled", log=log),
        "",
        (
            f"#### Planned contrasts with {BONFERRONI_LEVEL:.2f}% intervals, adjusted for three "
            "contrasts (exploratory), second setting"
        ),
        "",
        *_bonferroni_lines(losses=sensitivity, setting="sensitivity", log=log),
        "",
        (
            "#### ENS and HRES against each other and against ERA5, pooled, primary setting "
            "(exploratory)"
        ),
        "",
        *_contrast_table(
            losses=pooled,
            pairs=EXPLORATORY_CONTRASTS,
            label="all",
            section="exploratory",
            setting="pooled",
            log=log,
        ),
        "",
        "#### The same two contrasts, second setting (exploratory)",
        "",
        *_contrast_table(
            losses=sensitivity,
            pairs=EXPLORATORY_CONTRASTS,
            label="all",
            section="exploratory",
            setting="sensitivity",
            log=log,
        ),
        "",
        *_horizons_lines(pooled=pooled, log=log),
        "",
        "#### UKV and the ICON products against ERA5, pooled, primary setting (exploratory)",
        "",
        *_contrast_table(
            losses=pooled,
            pairs=ICON_CONTRASTS,
            label="all",
            section="icon",
            setting="pooled",
            log=log,
        ),
        "",
        (
            "#### Three alternative ENS interpolations against the planned `components` arm "
            "(exploratory)"
        ),
        "",
        *_contrast_table(
            losses=pooled,
            pairs=ENS_VARIANT_CONTRASTS,
            label="all",
            section="ens variants",
            setting="pooled",
            log=log,
        ),
        "",
        *_servable_lines(pooled=pooled, log=log),
        *_period_lines(pooled=pooled, log=log),
        *geometry_lines(sites=sites, noun="wind farms"),
    ]
    return "\n".join(lines) + "\n"


def _readme_text() -> str:
    """Return the text of the README written beside the outputs.

    Returns:
        Markdown saying what each file holds.
    """
    return f"""# ENS and HRES wind in the past-wind study

Outputs of `studies/beam_diffuse_split/ens_hres_past_wind.py`. Wind farms appear only as `W1` to
`W3`. Nothing here carries a coordinate or a generator name.

- `losses.parquet`: one row per (arm, setting, farm, hour, seed) with the out-of-fold absolute and
  signed errors in MW and as a fraction of the row's capacity, the fold, the month, and `actual_mw`,
  the metered power. `setting` is `pooled` (the primary hyperparameters) or `sensitivity` (the
  second setting).
- `losses.fingerprint`: a hash of the row set (floats cast to Float32), every arm's columns, the
  seeds and the hyperparameters. `--report-only` refuses to reuse `losses.parquet` if the hash
  changes.
- `intervals.parquet`: every interval `report.md` prints, one row each, with its section, setting,
  scope, arms, value and bounds (percentage points of capacity), level, rows and months.
- `report.md`: every table the docs page quotes, printed by the script and never transcribed.
- `superseded/`: outputs a later run replaced.

`SEEDS` is {list(SEEDS)}, and each interval resamples whole calendar months and one of those seeds
{N_BOOTSTRAP_RESAMPLES:,} times.
"""


def _script_commit() -> str:
    """Return the commit the script was committed at before its first fit, from the repository.

    Returns:
        The short hash of the last commit that changed this file, or `uncommitted` if none did.
    """
    result = subprocess.run(
        ["git", "log", "-1", "--format=%h", "--", str(Path(__file__).resolve())],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).parent,
    )
    return result.stdout.strip() or "uncommitted"


def main() -> int:
    """Build the row set, run every check, fit every arm, and write the report.

    Every check in `run_checks` runs before any arm is fitted, in both a fresh run and
    `--report-only`, and `_raise_on_failed_checks` raises rather than letting a bad row set reach
    `run_all`. `--checks-only` stops after the checks and prints them, fitting nothing and writing
    nothing.
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
    rows, counts = joined_row_set(sites=sites)
    frame = with_three_eras(frame=_add_time_features(dataset=rows), fold_offsets=ERA_FOLD_OFFSETS)
    _LOG.info("common rows: %d, %s to %s", frame.height, frame["time"].min(), frame["time"].max())

    checks = run_checks(sites=sites, frame=frame, counts=counts)
    if arguments.checks_only:
        sys.stdout.write(
            "\n".join(
                [
                    *_checks_lines(checks=checks),
                    "",
                    *_coverage_lines(coverage=checks["coverage"]),
                    "",
                    *_lead_lines(lead_table=checks["lead_table"]),
                    "",
                ]
            )
        )
    _raise_on_failed_checks(checks=checks)
    if arguments.checks_only:
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "losses.parquet"
    fingerprint_path = OUTPUT_DIR / "losses.fingerprint"
    intervals_path = OUTPUT_DIR / "intervals.parquet"
    report_path = OUTPUT_DIR / "report.md"
    readme_path = OUTPUT_DIR / "README.md"

    all_jobs = jobs()
    fingerprint = _fingerprint(frame=frame, job_list=all_jobs)

    if arguments.report_only:
        saved_fingerprint = (
            fingerprint_path.read_text().strip() if fingerprint_path.exists() else None
        )
        if saved_fingerprint != fingerprint:
            msg = (
                f"--report-only: {path} was fitted on a different row set, column set, seed set, "
                "feature values or hyperparameter setting than this code now produces; re-run "
                "without --report-only"
            )
            raise ValueError(msg)
        losses = pl.read_parquet(path)
    else:
        refuse_to_overwrite(paths=[path, fingerprint_path, intervals_path, report_path])
        fitted = run_all(dataset=frame, jobs=all_jobs)
        losses = fitted.join(
            frame.select("site", "time", actual_mw=pl.col("power_mw").cast(pl.Float64)),
            on=["site", "time"],
            how="left",
        )
        losses.write_parquet(path)
        fingerprint_path.write_text(fingerprint)

    log = IntervalLog()
    report = _report(
        frame=frame,
        losses=losses,
        sites=sites,
        job_list=all_jobs,
        checks=checks,
        fingerprint=fingerprint,
        log=log,
        script_commit=_script_commit(),
    )
    report_path.write_text(report)
    log.frame().write_parquet(intervals_path)
    readme_path.write_text(_readme_text())
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
