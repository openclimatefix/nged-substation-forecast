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
period splits, the lead and time-of-day split, the two post-review additions listed below,
and every by-farm row. **The ENS-against-ERA5 contrast
re-estimates a figure the ENS horizons study already published** (day 0 against ERA5 on 50,268 rows
from 12 August 2024), on a shorter row set, and is exploratory; the report prints the published
figure beside it.

**Departures from the maintainer's request, each written into the plan.** HRES is read from the
Open-Meteo Previous Runs file (`previous_runs/combined.parquet`, model `ecmwf_ifs`, the bare column
is the freshest run), not the 342-point grid file, because the Previous Runs file holds wind
direction and the grid file holds speed only. The fetch set no `cell_selection`, so Open-Meteo's
default (`land`) applies, and ERA5 is read at the nearest cell. The grid file is used for one
cross-check: at each farm, some point among the five nearest to it must reproduce the Previous Runs
speeds at 10 m and 100 m to within 0.05 km/h on every hour, because Open-Meteo picks its own model
cell, which is not always the nearest point. Both files are Open-Meteo downloads, so a match shows
that two downloads agree, not that the grid or the served lead is right. ENS day 0 is read from the
horizons study's saved inputs (`data/studies/ens_forecast_horizons/wind_inputs.parquet`), not from
the `T+3` wind download, so no new ENS code is written.

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
farm, and, for the ENS-against-HRES contrast, ensemble averaging against a single run. ENS itself
runs four times a day. The 3-hourly steps after ECMWF's hourly steps to T+90, the single 00 UTC run,
the missing direct-radiation fields and the roughly 09:00 UTC read time belong to ECMWF's open-data
subset and Dynamical.org's archive. The 09:00 UTC read time is this repository's
`NWP_PUBLICATION_DELAY_HOURS` assumption, not a documented Dynamical.org latency (ECMWF disseminates
ENS day 0 at about 06:40 UTC). ENS's native grid is about 9 km (O1280), served at 0.25 degrees. IFS
Cycle 50r1 went live with the 06 UTC run of 12 May 2026, so the 00 UTC ENS run of that day is still
49r1, and the period split's "from 2026-05-12" holds one day of 49r1 ENS data.

**Post-review additions (exploratory, added after the first results).** Listed in the plan before
they were fitted, and run by `--extra-fits`, which writes `losses_long_rows.parquet` and
`losses_fold_designs.parquet`, each with its own fingerprint, and leaves `losses.parquet` alone:

- The long-row-set reconciliation: ERA5, UKV, HRES and ENS refitted on the rows from 2024-08-12,
  under two designs (the horizons study's, and one with an extra era cut at 2024-12-01).
- The fold-design robustness table for P1 to P3 and ENS-HRES.
- The HRES served-lead table, now the ratio of each UTC hour's mean absolute hour-to-hour change to
  the mean of its two neighbours' changes, from the saved Previous Runs file.
- The lead and time-of-day split (labels 00-08 UTC against 10-23 UTC), which mixes ENS lead with
  time of day and does not test whether ENS could be read in time.

Run it with `uv run python studies/beam_diffuse_split/ens_hres_past_wind.py`. `refuse_to_overwrite`
on a fresh run means `losses.parquet`, `losses.fingerprint`, `intervals.parquet`, `report.md` and
`script_commit.txt` each have to move to a `superseded/` subfolder before a re-run. `--report-only`
rebuilds the report from the saved `losses.parquet` alone, fitting nothing, but still raises if
the saved fingerprint (the row set with its Float32-cast floats, every job's columns, the seeds,
and the hyperparameters) no longer matches what this code would fit.
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
from weather_products import METRIC, PERCENTAGE_POINTS, _mae, with_eras
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

SPLIT_LABEL_HOURS: Final[tuple[range, range]] = (range(9), range(10, 24))
"""Label hours (UTC) of the lead and time-of-day split: 00-08 and 10-23; label 09 is dropped."""

JUMP_RATIO_THRESHOLD: Final[float] = 1.10
"""A UTC hour whose mean absolute hour-to-hour change, over the mean of its two neighbouring hours'
changes, is at least this is reported as a jump, the mark of a switch between two forecast runs in
HRES's stitched series."""

LEAD_VARIABLES: Final[tuple[str, str, str]] = (
    "wind_speed_100m",
    "wind_speed_10m",
    "temperature_2m",
)
"""The Previous Runs columns of the HRES served-lead table."""

LONG_ROW_SET_START: Final[pl.Expr] = pl.datetime(2024, 8, 12, time_zone="UTC")
"""The first hour of the long row set: the page's own start, the rows common to every product."""

IFS_CYCLE_49R1_CUT_MONTH: Final[str] = "2024-12"
"""The first whole month after IFS Cycle 49r1 (12 November 2024), where the long row set's extra era
cut falls."""

UKV_UPGRADE_MONTH: Final[str] = "2026-02"
"""The first month of the second UKV era, the page's own cut."""

DESIGN_ARMS: Final[tuple[str, ...]] = ("era5", "ukv", "hres", "ens_mean_day0")
"""The four products the planned contrasts and ENS-HRES need, fitted under every extra design."""

LONG_ROW_ARMS: Final[tuple[str, ...]] = (
    "era5",
    "ukv",
    "hres",
    "ens_mean_day0",
    "ens_mean_day0_speed_components",
)
"""The arms of the long-row-set reconciliation, primary setting."""

LONG_ROW_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("ens_mean_day0_wind", "era5_wind"),
    ("ens_mean_day0_speed_components_wind", "era5_wind"),
    ("hres_wind", "era5_wind"),
    ("hres_wind", "ukv_wind"),
    ("ens_mean_day0_wind", "ukv_wind"),
    ("ens_mean_day0_speed_components_wind", "ukv_wind"),
    ("ens_mean_day0_wind", "hres_wind"),
    ("ens_mean_day0_speed_components_wind", "hres_wind"),
)
"""The contrasts printed for each long-row-set design and scope."""

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

FOLD_DESIGN_CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    *PLANNED_CONTRASTS,
    ("ENS-HRES", "ens_mean_day0_wind", "hres_wind"),
)
"""P1 to P3 and ENS-HRES, the contrasts the fold-design robustness table prints."""

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


class ExtraFits(TypedDict):
    """The post-review fits, read back with the frames they were fitted on."""

    long_frame: pl.DataFrame
    long_losses: pl.DataFrame
    design_losses: pl.DataFrame
    commit: str


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
    """Measure the hour-to-hour change of HRES's freshest-run series by UTC hour of day.

    A stitched series of forecast runs jumps where one run hands over to the next, so an hour whose
    mean change stands out from its two neighbours' marks where a new run starts. The statistic
    compares each hour with its neighbours, not with the median across the day, because a wind
    speed's own daily cycle makes some hours change more than others.

    Args:
        sites: The wind roster.

    Returns:
        One row per (variable, period, hour) with `mean_abs_change` (in the column's own unit, km/h
        or degrees Celsius) and `ratio`, that mean over the mean of the previous and the next
        hour's means, wrapping at midnight.
    """
    previous = (
        pl.read_parquet(HRES_PREVIOUS_RUNS_PATH)
        .filter(pl.col("site").is_in(sites["site"].to_list()), pl.col("time") >= ROW_SET_START)
        .sort("site", "time")
    )
    tables: list[pl.DataFrame] = []
    for variable in LEAD_VARIABLES:
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
        by_hour = changes.group_by("period", hour=pl.col("time").dt.hour()).agg(
            mean_abs_change=pl.col("change").mean()
        )
        for period in LEAD_PERIODS:
            means = (
                by_hour.filter(pl.col("period") == period).sort("hour")["mean_abs_change"].to_list()
            )
            tables.append(
                pl.DataFrame(
                    {
                        "variable": variable,
                        "period": period,
                        "hour": list(range(24)),
                        "mean_abs_change": means,
                        "ratio": [
                            means[hour] / ((means[hour - 1] + means[(hour + 1) % 24]) / 2.0)
                            for hour in range(24)
                        ],
                    },
                    schema={
                        "variable": pl.String,
                        "period": pl.String,
                        "hour": pl.Int32,
                        "mean_abs_change": pl.Float64,
                        "ratio": pl.Float64,
                    },
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
            "hour, over the mean of the same statistic at the previous and the next UTC hour "
            "(wrapping at midnight), for the same variable and period. A ratio well above 1 marks "
            "an hour where one forecast run hands over to the next. Ratios of "
            f"{JUMP_RATIO_THRESHOLD:.2f} or more are in bold. This statistic replaces the ratio to "
            "the median across the 24 hours, which the wind's own daily cycle swamps."
        ),
        "",
    ]
    for variable in LEAD_VARIABLES:
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
        expected = PLAN_EXPECTED_JUMP_HOURS[period]
        for variable in LEAD_VARIABLES:
            jump_hours = sorted(
                lead_table.filter(
                    pl.col("period") == period,
                    pl.col("variable") == variable,
                    pl.col("ratio") >= JUMP_RATIO_THRESHOLD,
                )["hour"].to_list()
            )
            reached = [hour for hour in expected if hour in jump_hours]
            lines.append(
                f"- {period}, `{variable}`: UTC hours with a ratio of "
                f"{JUMP_RATIO_THRESHOLD:.2f} or more: {jump_hours}. Of the expected handover "
                f"hours, {expected}, {len(reached)} of {len(expected)} reach the threshold: "
                f"{reached}."
            )
    lines += [
        "",
        (
            "Lead statement. HRES's served lead is 1 to 12 h before 1 October 2025 and 0 to 5 h "
            "from it, inferred from where the hour-to-hour jumps fall (not documented by "
            "Open-Meteo, whose documentation says only that each run's first few hours are "
            "stitched into a continuous series). The expected jumps before 1 October 2025 are at "
            "01 and 13 UTC, the arrival hours of the 00 and 12 UTC runs, and from 1 October 2025 "
            "at 00, 06, 12 and 18 UTC, the four daily runs. An hour that reaches the threshold is "
            "evidence of a handover, not proof of one, and the table can show further hours above "
            "the threshold that no run schedule explains."
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
            "- HRES fetch: no `cell_selection` was set, so Open-Meteo's default (`land`) applies, "
            "and ERA5 is read at the nearest cell."
        ),
        (
            f"- HRES cross-check against the grid file: at each of the {grid['n_farms']:.0f} "
            f"farms, one of the {GRID_NEAREST_RANK} nearest grid points reproduced the Previous "
            "Runs "
            f"bare `wind_speed_100m` and `wind_speed_10m` on every hour ({grid['n_hours']:,.0f} "
            f"farm-hours). The largest rank needed, pooled over farms, was {grid['max_rank']:.0f} "
            "(1 is the nearest point). The largest hourly disagreement was "
            f"{grid['max_diff_before_kmh']:.3f} km/h before 2025-10-01 and "
            f"{grid['max_diff_from_kmh']:.3f} km/h from it; the tolerance was 0.05 km/h. "
            "Both files are Open-Meteo downloads, so the match shows that two downloads agree, "
            "not that the grid or the served lead is right."
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


def _designs_fingerprint(*, designs: dict[str, pl.DataFrame], job_list: list[Job]) -> str:
    """Return one hash covering every design's frame and the jobs fitted under each.

    Args:
        designs: Each design's name to the frame it was fitted on, with `fold`.
        job_list: The jobs fitted under every design.

    Returns:
        A hex digest.
    """
    payload = repr(
        [(name, _fingerprint(frame=frame, job_list=job_list)) for name, frame in designs.items()]
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
            "The period from 2026-05-12 holds about 4 months of data, in 5 calendar-month labels "
            "because 12 May and 10 September fall inside months, so its intervals under-cover: "
            "resampling so few months cannot represent the month-to-month spread. IFS Cycle 50r1 "
            "went live with the 06 UTC run of 12 May 2026, so the 00 UTC ENS run of that day is "
            "still 49r1, and the period from 2026-05-12 holds one day of 49r1 ENS data, a "
            "trivial share."
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


def _lead_and_time_of_day_lines(*, pooled: pl.DataFrame, log: IntervalLog) -> list[str]:
    """Render the ENS contrasts on early and late label hours, exploratory.

    Wind power for label T covers T minus 30 minutes to T plus 30 minutes. Labels 00 to 08 UTC are
    the hours whose power ended before 09:00 UTC, and labels 10 to 23 UTC the hours after it. The
    split is one of ENS lead (0 to 8 hours against 10 to 23 hours from the 00 UTC run) and of time
    of day, and it is not a test of whether ENS could be served in time.

    Args:
        pooled: Per-row losses at the primary setting.
        log: Where the intervals are recorded.

    Returns:
        Markdown lines.
    """
    pairs = (("ens_mean_day0_wind", "ukv_wind"), *EXPLORATORY_CONTRASTS)
    lines = [
        "#### ENS contrasts by label hour, a lead and time-of-day split (exploratory)",
        "",
        (
            "Labels 00 to 08 UTC against labels 10 to 23 UTC; label 09 is dropped. The split "
            "separates ENS leads 0 to 8 h from leads 10 to 23 h from the 00 UTC run, and early "
            "hours of the day from late hours, so the two halves differ in lead and in time of "
            "day together. HRES's lead before 1 October 2025 is also mixed into the split. The "
            "split makes no claim about when a run can be read. Where ENS's deficit against UKV "
            "and HRES sits in the later hours, those hours are also the longer leads."
        ),
        "",
    ]
    for label, hours in zip(
        ("labels 00-08 UTC", "labels 10-23 UTC"), SPLIT_LABEL_HOURS, strict=True
    ):
        part = pooled.filter(pl.col("time").dt.hour().is_in(list(hours)))
        lines += _contrast_table(
            losses=part,
            pairs=pairs,
            label=label,
            section="lead and time of day",
            setting="pooled",
            log=log,
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


def long_row_frame(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return the long row set: the page's own rows from 2024-08-12, joined to ENS and HRES.

    The row set is the rows common to every product, without the 2024-12-01 restriction of the main
    row set. Both ENS combinations used by the long-row-set arms are joined.

    Args:
        sites: The wind roster.

    Returns:
        One row per (site, time), sorted, with time features and every product's columns.

    Raises:
        ValueError: If a join loses rows, or the rows start before `LONG_ROW_SET_START`.
    """
    frame = common_rows(frame=joined(sites=sites))
    start = frame["time"].min()
    if start < pl.select(LONG_ROW_SET_START).item():
        msg = f"the page's rows start at {start}, before the long row set's start"
        raise ValueError(msg)
    n_rows = frame.height
    for product in ("ens_mean_day0", "ens_mean_day0_speed_components"):
        frame = frame.join(ens_frame(product=product), on=["site", "time"], how="inner")
    frame = frame.join(hres_frame(sites=sites), on=["site", "time"], how="inner")
    if frame.height != n_rows:
        msg = f"joining ENS and HRES to the long row set lost {n_rows - frame.height} rows"
        raise ValueError(msg)
    return _add_time_features(dataset=frame).sort("site", "time")


def long_row_designs(*, frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return the long row set under its two era-and-fold designs.

    Args:
        frame: `long_row_frame`'s result.

    Returns:
        The horizons study's design (two UKV eras, no cut at IFS Cycle 49r1) and the same rows with
        one extra era cut at `IFS_CYCLE_49R1_CUT_MONTH`.
    """
    extra_cut = frame.with_columns(
        era_code=(
            (pl.col("month") >= IFS_CYCLE_49R1_CUT_MONTH).cast(pl.Int8)
            + (pl.col("month") >= UKV_UPGRADE_MONTH).cast(pl.Int8)
        )
    ).with_columns(era=pl.col("era_code").cast(pl.String))
    return {
        "two UKV eras, no cut at 49r1 (horizons design)": with_eras(frame=frame),
        "extra era cut at 2024-12-01": assign_folds(dataset=extra_cut, by=("site", "era")),
    }


def fold_designs(*, frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return the main row set under four era-and-fold designs other than the study's own.

    Args:
        frame: The main row set with time features, before eras and folds.

    Returns:
        Each design's name to its frame: three eras without fold rotation; the page's two UKV eras;
        the study's folds with a two-valued `era_code`; and an extra era cut at IFS Cycle 50r1 with
        the part-month of May 2026 dropped.
    """
    study_folds = with_three_eras(frame=frame, fold_offsets=ERA_FOLD_OFFSETS)
    rows_without_may = frame.filter(pl.col("month") != "2026-05")
    cut_50r1 = rows_without_may.with_columns(
        era_code=sum(
            (pl.col("month") >= month).cast(pl.Int8) for month in (*ERA_START_MONTHS, "2026-06")
        )
    ).with_columns(era=pl.col("era_code").cast(pl.String))
    folded_50r1 = assign_folds(dataset=cut_50r1, by=("site", "era"))
    rotation = pl.col("era_code").replace_strict({0: 0, 1: 0, 2: 2, 3: 4}, return_dtype=pl.Int32)
    return {
        "three eras, no fold rotation": with_three_eras(
            frame=frame, fold_offsets={0: 0, 1: 0, 2: 0}
        ),
        "two UKV eras (the page's design)": with_eras(frame=frame),
        "study folds, two-valued era_code": study_folds.with_columns(
            era_code=(pl.col("month") >= UKV_UPGRADE_MONTH).cast(pl.Int8)
        ),
        "extra era cut at IFS 50r1, May 2026 dropped": folded_50r1.with_columns(
            fold=(pl.col("fold") + rotation) % N_FOLDS
        ),
    }


def extra_jobs(*, arms: tuple[str, ...]) -> list[Job]:
    """Return one primary-setting job per arm, for the post-review fits.

    Args:
        arms: Product keys.

    Returns:
        One job per arm.
    """
    return [
        (
            f"{product}_wind",
            "pooled",
            "power_mw",
            (*SHARED_FEATURES, *_wind_columns(product=product)),
            PRIMARY_HYPER_PARAMETERS,
            False,
        )
        for product in arms
    ]


def fit_designs(*, designs: dict[str, pl.DataFrame], job_list: list[Job]) -> pl.DataFrame:
    """Fit every job under every design and stack the losses, labelled with the design.

    Args:
        designs: Each design's name to the frame to fit, carrying `fold`.
        job_list: The jobs to fit under each design.

    Returns:
        The stacked losses with a `design` column.
    """
    return pl.concat(
        [
            run_all(dataset=frame, jobs=job_list).with_columns(design=pl.lit(name))
            for name, frame in designs.items()
        ],
        how="diagonal_relaxed",
    )


def _long_row_lines(*, frame: pl.DataFrame, losses: pl.DataFrame, log: IntervalLog) -> list[str]:
    """Render the long-row-set reconciliation: contrasts per design and scope, and speed ratios.

    Args:
        frame: `long_row_frame`'s result.
        losses: The long-row-set losses, with `design`.
        log: Where the intervals are recorded.

    Returns:
        Markdown lines.
    """
    lines = [
        (
            "#### Long row set from 2024-08-12, ERA5, UKV, HRES and ENS refitted (exploratory, "
            "added after the first results)"
        ),
        "",
        (
            f"The rows are the page's own, from {frame['time'].min():%Y-%m-%d}: {frame.height:,} "
            f"farm-hours in {frame['month'].n_unique()} calendar months. Design one is the "
            "horizons study's: two UKV eras and no cut at IFS Cycle 49r1 (12 November 2024). "
            "Design two adds one era cut at 2024-12-01. `ens_mean_day0` is the `components` "
            "combination and `ens_mean_day0_speed_components` is the horizons study's own. Each "
            "design is scored on all rows and on the rows from 2024-12-01 only."
        ),
        "",
    ]
    for design in losses["design"].unique(maintain_order=True).to_list():
        for scope, part in (
            ("all rows", losses.filter(pl.col("design") == design)),
            (
                "rows from 2024-12-01",
                losses.filter(pl.col("design") == design, pl.col("time") >= ROW_SET_START),
            ),
        ):
            lines += [
                f"`{design}`, {scope}:",
                "",
                *_contrast_table(
                    losses=part,
                    pairs=LONG_ROW_CONTRASTS,
                    label=scope,
                    section=f"long rows: {design}",
                    setting="pooled",
                    log=log,
                ),
                "",
            ]
    ens_speed = _wind_columns(product="ens_mean_day0")[3]
    monthly = (
        frame.group_by("month")
        .agg(
            n=pl.len(),
            **{
                product: pl.col(_wind_columns(product=product)[3]).mean()
                / (KMH_PER_M_S if product in KMH_PRODUCTS else 1.0)
                for product in ("era5", "ukv", "hres")
            },
            ens=pl.col(ens_speed).mean(),
        )
        .sort("month")
    )
    lines += [
        "Monthly ratio of each product's mean 10 m wind speed to ERA5's, on the long row set:",
        "",
        "| Month | Rows | ENS / ERA5 | HRES / ERA5 | UKV / ERA5 |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {row['month']} | {row['n']:,} | {row['ens'] / row['era5']:.3f} "
        f"| {row['hres'] / row['era5']:.3f} | {row['ukv'] / row['era5']:.3f} |"
        for row in monthly.iter_rows(named=True)
    ]
    return lines


def _fold_design_lines(
    *, main_pooled: pl.DataFrame, design_losses: pl.DataFrame, log: IntervalLog
) -> list[str]:
    """Render P1 to P3 and ENS-HRES under the study's fold design and four others.

    Args:
        main_pooled: The main losses at the primary setting, the study's own design.
        design_losses: The other designs' losses, with `design`.
        log: Where the intervals are recorded.

    Returns:
        Markdown lines.
    """
    arms = [f"{product}_wind" for product in DESIGN_ARMS]
    lines = [
        (
            "#### Fold-design robustness of P1 to P3 and ENS-HRES (exploratory, added after the "
            "first results)"
        ),
        "",
        (
            "P1 is `hres_wind − ukv_wind`, P2 is `ens_mean_day0_wind − ukv_wind`, P3 is "
            "`hres_wind − era5_wind`, and ENS-HRES is `ens_mean_day0_wind − hres_wind`. Each "
            "design refits the four arms at the primary setting. The first two rows come from the "
            "main losses: the study's design, and the same losses without the rows of May 2026, "
            "the rows the last design drops."
        ),
        "",
    ]
    study = main_pooled.filter(pl.col("arm").is_in(arms))
    scored = [
        ("study design", study),
        ("study design, May 2026 rows removed", study.filter(pl.col("month") != "2026-05")),
        *(
            (name, design_losses.filter(pl.col("design") == name))
            for name in design_losses["design"].unique(maintain_order=True).to_list()
        ),
    ]
    for name, part in scored:
        lines += [
            f"`{name}`:",
            "",
            *_contrast_table(
                losses=part,
                pairs=tuple(
                    (treatment, reference) for _, treatment, reference in FOLD_DESIGN_CONTRASTS
                ),
                label=name,
                section="fold designs",
                setting="pooled",
                log=log,
            ),
            "",
        ]
    return lines


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
    extras: ExtraFits | None,
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
        extras: The post-review fits, or None if `--extra-fits` has not been run.

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
        *_lead_and_time_of_day_lines(pooled=pooled, log=log),
        *_period_lines(pooled=pooled, log=log),
    ]
    if extras is not None:
        lines += [
            "",
            (
                "Post-review fits, run by `--extra-fits` at script commit "
                f"`{extras['commit']}`, exploratory (added after the first results)."
            ),
            "",
            *_long_row_lines(frame=extras["long_frame"], losses=extras["long_losses"], log=log),
            "",
            *_fold_design_lines(main_pooled=pooled, design_losses=extras["design_losses"], log=log),
        ]
    lines += [
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
- `script_commit.txt`: the commit of the script that fitted `losses.parquet`, which a fresh run
  records after checking the script has no uncommitted changes.
- `intervals.parquet`: every interval `report.md` prints, one row each, with its section, setting,
  scope, arms, value and bounds (percentage points of capacity), level, rows and months.
- `losses_long_rows.parquet` and `losses_fold_designs.parquet`: the post-review fits
  (`--extra-fits`, exploratory, added after the first results), in the same layout as
  `losses.parquet` plus a `design` column. The first refits ERA5, UKV, HRES and ENS on the rows from
  2024-08-12 under two designs, and the second refits them on the main row set under four fold
  designs. Each has its own `.fingerprint` file, and `script_commit_extra_fits.txt` records the
  commit that fitted them. `losses.parquet` is not touched by them.
- `report.md`: every table the docs page quotes, printed by the script and never transcribed.
- `superseded/`: outputs a later run replaced.

`SEEDS` is {list(SEEDS)}, and each interval resamples whole calendar months and one of those seeds
{N_BOOTSTRAP_RESAMPLES:,} times.
"""


def _script_commit() -> str:
    """Return the commit the script is at, raising if the script has uncommitted changes.

    A fresh run records this hash beside the losses, so the report can say which committed script
    fitted them.

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


def _extra_fits(*, sites: pl.DataFrame, timed_rows: pl.DataFrame, fit: bool) -> ExtraFits | None:
    """Fit the post-review arms if asked, and read their saved losses back with their frames.

    Each saved losses file has its own fingerprint, which must match the frames this code builds,
    so the report never mixes the fits with a different row set or design. `losses.parquet` and its
    fingerprint are never read or written here.

    Args:
        sites: The wind roster.
        timed_rows: The main row set with time features, before eras and folds.
        fit: Whether to fit and write the losses; otherwise they are read if present.

    Returns:
        The saved fits, or None if the files are absent and `fit` is False.

    Raises:
        ValueError: If a saved fingerprint does not match, or only some of the files exist.
    """
    long_path = OUTPUT_DIR / "losses_long_rows.parquet"
    design_path = OUTPUT_DIR / "losses_fold_designs.parquet"
    commit_path = OUTPUT_DIR / "script_commit_extra_fits.txt"
    paths = [
        long_path,
        long_path.with_suffix(".fingerprint"),
        design_path,
        design_path.with_suffix(".fingerprint"),
        commit_path,
    ]
    long_frame = long_row_frame(sites=sites)
    long_designs = long_row_designs(frame=long_frame)
    long_jobs = extra_jobs(arms=LONG_ROW_ARMS)
    fold_frames = fold_designs(frame=timed_rows)
    fold_jobs = extra_jobs(arms=DESIGN_ARMS)
    fingerprints = {
        long_path: _designs_fingerprint(designs=long_designs, job_list=long_jobs),
        design_path: _designs_fingerprint(designs=fold_frames, job_list=fold_jobs),
    }
    if fit:
        refuse_to_overwrite(paths=paths)
        commit = _script_commit()
        for path, designs, job_list in (
            (long_path, long_designs, long_jobs),
            (design_path, fold_frames, fold_jobs),
        ):
            fit_designs(designs=designs, job_list=job_list).write_parquet(path)
            path.with_suffix(".fingerprint").write_text(fingerprints[path])
        commit_path.write_text(commit)
    elif not any(path.exists() for path in paths):
        return None
    elif not all(path.exists() for path in paths):
        msg = f"only some of the post-review outputs exist: {[p.name for p in paths if p.exists()]}"
        raise ValueError(msg)
    for path, fingerprint in fingerprints.items():
        if path.with_suffix(".fingerprint").read_text().strip() != fingerprint:
            msg = f"{path.name} was fitted on different frames, jobs or seeds than this code builds"
            raise ValueError(msg)
    return {
        "long_frame": long_frame,
        "long_losses": pl.read_parquet(long_path),
        "design_losses": pl.read_parquet(design_path),
        "commit": commit_path.read_text().strip(),
    }


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
    parser.add_argument(
        "--extra-fits",
        action="store_true",
        help=(
            "Fit the post-review arms (long row set, fold designs), write their losses, and "
            "rebuild report.md from the saved losses.parquet, fitting nothing else."
        ),
    )
    arguments = parser.parse_args()

    sites = _wind_sites()
    rows, counts = joined_row_set(sites=sites)
    timed_rows = _add_time_features(dataset=rows)
    frame = with_three_eras(frame=timed_rows, fold_offsets=ERA_FOLD_OFFSETS)
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
    commit_path = OUTPUT_DIR / "script_commit.txt"

    all_jobs = jobs()
    fingerprint = _fingerprint(frame=frame, job_list=all_jobs)

    if arguments.report_only or arguments.extra_fits:
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

    extras = _extra_fits(sites=sites, timed_rows=timed_rows, fit=arguments.extra_fits)
    log = IntervalLog()
    report = _report(
        frame=frame,
        losses=losses,
        sites=sites,
        job_list=all_jobs,
        checks=checks,
        fingerprint=fingerprint,
        log=log,
        script_commit=script_commit,
        extras=extras,
    )
    report_path.write_text(report)
    log.frame().write_parquet(intervals_path)
    readme_path.write_text(_readme_text())
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
