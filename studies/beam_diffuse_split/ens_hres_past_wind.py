"""Score ECMWF ENS and ECMWF IFS HRES as descriptions of past wind, refitting every arm.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/pull/885>, extending the five-product
past-wind comparison in `wind_products.py` (ERA5, UKV, ICON-D2, ICON-EU, ICON global) with two
ECMWF products: the deterministic IFS HRES forecast, and the mean of the 51-member ENS at day 0.
The past-solar study already scores both products; the past-wind page scored neither.

**The planned contrasts are in the plan file committed as `9bb0a6f7`, whose last revision before any
fit is `83dbbad3`.** The three contrasts, and the second hyperparameter setting each is rerun at,
are:

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
default (`land`) applies, as it does to every other product on the page (`fetch_wind_point.py` sets
`land`). The grid file is used for one cross-check: at each farm, some point among the five nearest
to it must reproduce the Previous Runs speeds at 10 m and 100 m to within 0.05 km/h on every hour,
because Open-Meteo picks its own model cell, which is not always the nearest point. Both files are
Open-Meteo downloads, so a match shows that two downloads agree, not that the grid or the served
lead is right. ENS day 0 is read from the horizons study's saved inputs
(`data/studies/ens_forecast_horizons/wind_inputs.parquet`), not from
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
`NWP_PUBLICATION_DELAY_HOURS` assumption, not a documented Dynamical.org latency. ECMWF's
dissemination schedule lists the 00 UTC ENS perturbation forecasts' hourly steps 0 to 90 at 06:40
to 06:55 UTC; open-data publication and Dynamical.org's ingest come later, and are not measured
here. ENS's native grid is about 9 km (O1280), served at 0.25 degrees. IFS
Cycle 50r1 went live with the 06 UTC run of 12 May 2026, so the 00 UTC ENS run of that day is still
49r1, and the period split's "from 2026-05-12" holds one day of 49r1 ENS data.

**Post-review additions (exploratory, added after the first results).** Listed in the plan before
they were fitted, and run by `--extra-fits`, which writes `losses_long_rows.parquet` and
`losses_fold_designs.parquet`, each with its own fingerprint, and leaves `losses.parquet` alone:

- The long-row-set reconciliation: ERA5, UKV, HRES and ENS refitted on the rows from 2024-08-12,
  under three designs (the horizons study's; the same eras with rotated folds; and one with an
  extra era cut at 2024-12-01).
- The fold-design robustness table for P1 to P3 and ENS-HRES.
- The HRES served-lead table: the ratio of each UTC hour's mean absolute hour-to-hour change to
  the mean of its two neighbours' changes, from the saved Previous Runs file.
- The lead and time-of-day split (labels 00-08 UTC against 10-23 UTC), which mixes ENS lead with
  time of day and does not test whether ENS could be read in time.
- The calendar-month coverage of every fold design and long-row design, and, for the long rows'
  extra-cut design, fold numbers rotated so that no calendar month is left without training rows.
- Control rows in the lead and time-of-day split and the period splits: UKV, HRES and ERA5
  contrasts that involve no ENS lead, and ERA5, an analysis, has no lead at all.
- Each product's 10 m speed over ERA5's averaged over August to October of two years, and the
  Previous Runs file's `_previous_day*` columns before and from the archive-source change.

**Further exploratory additions, computed from the saved losses and saved inputs with no refit.**

- Paired design differences: each contrast under one long-row design minus the same contrast under
  another, on the same rows, so the fold rotation and the era cut can each be separated.
- The change of each contrast between the early and the late label hours, between two farms, and
  between the two periods of each split, each interval resampling whole calendar months and one
  fitting seed (the halves of the day and the farms share one draw of months, and the periods draw
  their own).
- The range of P1 to P3 and ENS-HRES across every design scored on the rows from 2024-12-01.
- The 100 m ratio to ERA5 by month, and each product's ratio before and after the first hour of
  IFS Cycle 49r1 in its data.
- The 100 m hour-to-hour jump ratio at 07 and 19 UTC for ERA5, UKV and HRES, and the count of
  expected handover hours reaching the plan's threshold of 1.15.
- The fewest training rows in any covered fold cell, and the calendar months in which the long row
  set holds farm-hours that the horizons study's inputs hold no power for.
- A subset with fewer than `MIN_MONTHS_FOR_INTERVAL` calendar months prints `too few months` in the
  `Excludes zero?` column.

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
import re
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol, TypedDict

import numpy as np
import polars as pl
from build_dataset import _wind_sites
from ens_past_solar import _arm_columns_lines, _fingerprint
from run_experiment import Job, _add_time_features, run_all
from sources import STUDY_DATA_DIR, WEATHER_DATA_DIR
from studies.bootstrap import (
    BOOTSTRAP_SEED,
    MIN_MONTHS_FOR_INTERVAL,
    N_BOOTSTRAP_RESAMPLES,
    bootstrap_absolute,
    bootstrap_difference,
    bootstrap_difference_at_level,
    paired_differences,
    per_fold_differences,
)
from studies.charts import CONTRAST_COLUMNS
from studies.cross_validation import (
    ENS_HRES_WIND_ERA_FOLD_OFFSETS,
    ENS_HRES_WIND_ERA_START_MONTHS,
    PRIMARY_HYPER_PARAMETERS,
    SEEDS,
    SENSITIVITY_HYPER_PARAMETERS,
    UKV_UPGRADE_MONTH,
    HyperParameters,
    calendar_month_coverage,
    cut_eras,
    raise_on_uncovered_months,
    rotate_folds,
    uncovered_months,
)
from studies.grid_sampling import distance_matrix_km
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

ROW_SET_START_DATE: Final[datetime] = datetime(2024, 12, 1, tzinfo=UTC)
"""The first hour of the row set: the first whole month after IFS Cycle 49r1 (12 November 2024)."""

ROW_SET_START: Final[pl.Expr] = pl.lit(ROW_SET_START_DATE)
"""`ROW_SET_START_DATE` as a Polars expression."""

EXPECTED_ROWS: Final[Mapping[str, int]] = MappingProxyType(
    {"W1": 14_489, "W2": 14_994, "W3": 14_072}
)
"""The row count per farm the plan fixes; the script raises if the joined row set differs."""

HRES_ARCHIVE_CHANGE_DATE: Final[datetime] = datetime(2025, 10, 1, tzinfo=UTC)
"""The date Open-Meteo's ECMWF archive source changes, and the first era boundary."""

HRES_ARCHIVE_CHANGE: Final[pl.Expr] = pl.lit(HRES_ARCHIVE_CHANGE_DATE)
"""`HRES_ARCHIVE_CHANGE_DATE` as a Polars expression."""

IFS_CYCLE_50R1_DATE: Final[datetime] = datetime(2026, 5, 12, tzinfo=UTC)
"""The date IFS Cycle 50r1 goes operational, the second period-split boundary."""

IFS_CYCLE_50R1: Final[pl.Expr] = pl.lit(IFS_CYCLE_50R1_DATE)
"""`IFS_CYCLE_50R1_DATE` as a Polars expression."""

IFS_CYCLE_50R1_MONTH: Final[str] = f"{IFS_CYCLE_50R1_DATE:%Y-%m}"
"""The month label holding IFS Cycle 50r1's start, a part-month dropped by the 50r1 fold design."""

FIRST_MONTH_AFTER_50R1: Final[str] = (
    f"{IFS_CYCLE_50R1_DATE.replace(day=1) + timedelta(days=31):%Y-%m}"
)
"""The first whole month after IFS Cycle 50r1, where the 50r1 fold design's extra era begins."""

NO_FOLD_OFFSETS: Final[Mapping[int, int]] = MappingProxyType({0: 0, 1: 0, 2: 0})
"""No rotation of any of the three eras' fold numbers: the design the coverage check rejected."""

ERA_FOLD_OFFSETS_50R1: Final[Mapping[int, int]] = MappingProxyType({0: 0, 1: 0, 2: 2, 3: 4})
"""The fold rotation of the design with an extra era cut at IFS Cycle 50r1.

Era 3 (June to September 2026) is rotated by 4. Rotations of 0 and 4 both leave 0 uncovered cells;
4 is kept so that the design's folds, and the figures printed for them, do not move. These offsets
belong to this row set; `studies.cross_validation.search_fold_offsets` confirms coverage on another.
"""

HORIZONS_ROTATED_FOLD_OFFSETS: Final[Mapping[int, int]] = MappingProxyType({0: 0, 1: 2})
"""The fold rotation of the long row set's two-UKV-era design with no cut at IFS Cycle 49r1.

Rotations of 2, 3 and 4 of era 1 (from 2026-02) leave 0 uncovered cells, where the horizons study's
own fold layout leaves 6; 2 is the smallest. `studies.cross_validation.search_fold_offsets` finds
such rotations for a given row set.
"""

LONG_ROW_FOLD_OFFSETS: Final[Mapping[int, int]] = MappingProxyType({0: 0, 1: 2, 2: 0})
"""The fold rotation of the long row set's design with an extra era cut at 2024-12-01.

Found by searching every pair of rotations of eras 1 and 2 (era 0 fixed): six of the 25 pairs leave
0 uncovered cells, and this is the one with a single non-zero rotation, the smaller of the two such
pairs. `studies.cross_validation.search_fold_offsets` runs that search.
"""

SPLIT_LABEL_HOURS: Final[tuple[range, range]] = (range(9), range(10, 24))
"""Label hours (UTC) of the lead and time-of-day split: 00-08 and 10-23; label 09 is dropped."""

MAX_ENS_POWER_DIFF_MW: Final[float] = 1e-3
"""`run_checks` fails if the horizons study's power and this row set's power differ by more."""

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

LONG_ROW_SET_START: Final[datetime] = datetime(2024, 8, 12, tzinfo=UTC)
"""The first hour of the long row set: the page's own start, the rows common to every product."""

IFS_CYCLE_49R1_CUT_MONTH: Final[str] = f"{ROW_SET_START_DATE:%Y-%m}"
"""The first whole month after IFS Cycle 49r1 (12 November 2024), where the long row set's extra era
cut falls."""

HORIZONS_DESIGN: Final[str] = "two UKV eras, no cut at 49r1 (horizons design)"
"""The name of the long row set's first design: the horizons study's own eras and folds."""

ROTATED_DESIGN: Final[str] = "two UKV eras, no cut at 49r1, folds rotated"
"""The name of the long row set's second design: the horizons study's eras, folds rotated."""

CUT_DESIGN: Final[str] = "extra era cut at 2024-12-01"
"""The name of the long row set's third design: an extra era cut at 2024-12-01."""

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

LEAD_PERIODS: Final[tuple[str, str]] = (
    f"before {HRES_ARCHIVE_CHANGE_DATE:%Y-%m-%d}",
    f"from {HRES_ARCHIVE_CHANGE_DATE:%Y-%m-%d}",
)
"""The two periods of the HRES served-lead table, split where the archive source changes."""

PLAN_EXPECTED_JUMP_HOURS: Final[Mapping[str, tuple[int, ...]]] = MappingProxyType(
    {LEAD_PERIODS[0]: (1, 13), LEAD_PERIODS[1]: (0, 6, 12, 18)}
)
"""The UTC hours at which the plan expected HRES's stitched series to hand over between runs."""

PREVIOUS_DAY_VARIABLE: Final[str] = "wind_speed_100m_previous_day1"
"""The Previous Runs column of the run one day earlier, whose hour-to-hour jumps are tabulated."""

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

ENS_METHODS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ens_mean_day0": "components",
        "ens_mean_day0_speed_components": "speed_components",
        "ens_mean_day0_direction_components": "direction_components",
        "ens_mean_day0_linear": "linear",
    }
)
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

SPLIT_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("ens_mean_day0_wind", "ukv_wind"),
    *EXPLORATORY_CONTRASTS,
    ("hres_wind", "ukv_wind"),
    ("hres_wind", "era5_wind"),
    ("ukv_wind", "era5_wind"),
)
"""The contrasts of the lead and time-of-day split. The last three involve no ENS lead, and ERA5,
an analysis, has no lead at all, so they show how much of a change between the halves is time of
day."""

PERIOD_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    *((treatment, reference) for _, treatment, reference in PLANNED_CONTRASTS),
    *EXPLORATORY_CONTRASTS,
    ("ukv_wind", "era5_wind"),
)
"""The contrasts of the period splits: the planned ones, the two exploratory ones, and UKV against
ERA5, because UKV's own upgrade falls in the later period."""

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
    ("HRES archive source", *LEAD_PERIODS),
    (
        "IFS Cycle 50r1",
        f"before {IFS_CYCLE_50R1_DATE:%Y-%m-%d}",
        f"from {IFS_CYCLE_50R1_DATE:%Y-%m-%d}",
    ),
)
"""The two period splits: the name of the change, then the labels of the two periods."""

BONFERRONI_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / len(PLANNED_CONTRASTS))
"""The confidence level, in percent, of an interval adjusted for the three planned contrasts."""

CONTRAST_HEADER: Final[tuple[str, str]] = (
    "| " + " | ".join(CONTRAST_COLUMNS) + " |",
    "|" + "---|" * len(CONTRAST_COLUMNS),
)
"""The header of every contrast table in the report, the one `studies.charts.report_contrasts`
reads."""

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


IntervalLog = list[IntervalRecord]
"""Every interval the report prints, collected as the report is assembled."""

INTERVAL_SCHEMA: Final[Mapping[str, type[pl.DataType]]] = MappingProxyType(
    {
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
    }
)
"""The columns of `intervals.parquet`, one row per interval the report prints."""


class ExtraFits(TypedDict):
    """The post-review fits, read back with the frames they were fitted on."""

    long_frame: pl.DataFrame
    long_designs: dict[str, pl.DataFrame]
    long_losses: pl.DataFrame
    fold_frames: dict[str, pl.DataFrame]
    design_losses: pl.DataFrame
    commit: str


class PreviousDayEvidence(TypedDict):
    """What the Previous Runs file's `_previous_day*` columns show about the source change."""

    n_columns: int
    n_cells_before: int
    non_null_before: int
    last_non_null_before: str
    n_cells_from: int
    non_null_from: int
    jump_hours_from: list[int]
    ratios_from: list[float]


class ChecksResult(TypedDict):
    """Every pre-fit check's raw result, computed once by `run_checks`."""

    row_counts: list[tuple[str, int]]
    coverage: pl.DataFrame
    grid_check: dict[str, float]
    ens_power: dict[str, float]
    lead_table: pl.DataFrame
    previous_day: PreviousDayEvidence


class TrialAreaBox(Protocol):
    """The part of `weather_downloads.paths.TrialAreaBox` this script reads."""

    def grid_points(self, *, spacing_deg: float) -> pl.DataFrame:
        """Return the grid's `point_id`, `latitude` and `longitude`, one row per point."""
        ...


def _trial_area_box() -> TrialAreaBox:
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
        base: The row set, carrying `site`, `time` and `power_mw`.

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
    frame = common_rows(frame=joined(sites=sites)).filter(pl.col("time") >= ROW_SET_START)
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


def _fewest_training_rows(*, coverage: pl.DataFrame) -> str:
    """Describe the covered cell with the fewest training rows.

    Args:
        coverage: `calendar_month_coverage`'s result.

    Returns:
        The smallest count of training rows among the cells that have any, with its site, fold and
        calendar month.
    """
    covered = coverage.filter(pl.col("n_train") > 0).sort("n_train", "site", "fold")
    row = covered.row(0, named=True)
    return (
        f"{row['n_train']:,} training rows (site {row['site']}, fold {row['fold']}, calendar "
        f"month {row['calendar_month']}, where {row['n_scored']:,} rows are scored)"
    )


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
            f"year of the row set only, which no fold design can cover: {single_months}. The "
            f"covered cell with the fewest training rows holds "
            f"{_fewest_training_rows(coverage=coverage)}."
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
    order = np.argsort(distance_matrix_km(sites=sites, cells=points), axis=1)
    point_ids = points["point_id"].to_numpy()
    return {
        site: point_ids[order[index, :GRID_NEAREST_RANK]].tolist()
        for index, site in enumerate(sites["site"].to_list())
    }


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
    points = _trial_area_box().grid_points(spacing_deg=0.05)
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


def _jump_ratios(*, changes: pl.DataFrame) -> list[float]:
    """Return each UTC hour's mean change over the mean of its two neighbours' means.

    Args:
        changes: Hour-to-hour changes of one variable in one period, carrying `time` and `change`.

    Returns:
        24 ratios, hour 0 first, wrapping at midnight, so the neighbours of hour 0 are 23 and 1.
    """
    by_hour = changes.group_by(hour=pl.col("time").dt.hour()).agg(
        mean_abs_change=pl.col("change").mean()
    )
    means = by_hour.sort("hour")["mean_abs_change"].to_list()
    return [means[hour] / ((means[hour - 1] + means[(hour + 1) % 24]) / 2.0) for hour in range(24)]


def _hourly_changes(*, previous: pl.DataFrame, variable: str) -> pl.DataFrame:
    """Return each row's absolute change from the previous hour, for consecutive hours only.

    Args:
        previous: The Previous Runs rows of the wind roster, sorted by `site` and `time`.
        variable: The column whose change is measured.

    Returns:
        Rows with `time`, `change` and `period` (one of `LEAD_PERIODS`).
    """
    return (
        previous.with_columns(
            change=(pl.col(variable) - pl.col(variable).shift(1).over("site")).abs(),
            gap=pl.col("time") - pl.col("time").shift(1).over("site"),
        )
        .filter(pl.col("gap") == pl.duration(hours=1))
        .select(
            "time",
            "change",
            period=pl.when(pl.col("time") >= HRES_ARCHIVE_CHANGE)
            .then(pl.lit(LEAD_PERIODS[1]))
            .otherwise(pl.lit(LEAD_PERIODS[0])),
        )
    )


def _previous_runs_rows(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return the Previous Runs file's rows for the wind roster from `ROW_SET_START`, sorted.

    Args:
        sites: The wind roster.

    Returns:
        The rows, sorted by `site` and `time`.
    """
    return (
        pl.read_parquet(HRES_PREVIOUS_RUNS_PATH)
        .filter(pl.col("site").is_in(sites["site"].to_list()), pl.col("time") >= ROW_SET_START)
        .sort("site", "time")
    )


def hres_lead_table(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Measure the hour-to-hour change of HRES's freshest-run series by UTC hour of day.

    A stitched series of forecast runs jumps where one run hands over to the next, so an hour whose
    mean change stands out from its two neighbours' marks where a new run starts. The statistic
    compares each hour with its neighbours, not with the median across the day, because a wind
    speed's own daily cycle makes some hours change more than others.

    Args:
        sites: The wind roster.

    Returns:
        One row per (variable, period, hour) with `ratio`: that hour's mean absolute change over
        the mean of the previous and the next hour's means, wrapping at midnight.
    """
    previous = _previous_runs_rows(sites=sites)
    tables: list[pl.DataFrame] = []
    for variable in LEAD_VARIABLES:
        changes = _hourly_changes(previous=previous, variable=variable)
        for period in LEAD_PERIODS:
            ratios = _jump_ratios(changes=changes.filter(pl.col("period") == period))
            tables.append(
                pl.DataFrame(
                    {"variable": variable, "period": period, "hour": range(24), "ratio": ratios},
                    schema={
                        "variable": pl.String,
                        "period": pl.String,
                        "hour": pl.Int32,
                        "ratio": pl.Float64,
                    },
                )
            )
    return pl.concat(tables).sort("variable", "period", "hour")


def previous_day_evidence(*, sites: pl.DataFrame) -> PreviousDayEvidence:
    """Count the `_previous_day*` values before and from the archive-source change, and find jumps.

    Open-Meteo's Previous Runs file holds each variable's freshest-run value and, for each of the
    previous 7 days, the value the run started that many days earlier had for the hour. A value
    the source did not serve is null.

    Args:
        sites: The wind roster.

    Returns:
        The number of `_previous_day*` columns; how many cells they hold before and from
        `HRES_ARCHIVE_CHANGE_DATE` and how many of those are not null; the last date before the
        change on which any is not null; and the UTC hours at which `PREVIOUS_DAY_VARIABLE` jumps
        from that date, with all 24 ratios.
    """
    previous = _previous_runs_rows(sites=sites)
    columns = [name for name in previous.columns if "_previous_day" in name]
    before = previous.filter(pl.col("time") < HRES_ARCHIVE_CHANGE)
    after = previous.filter(pl.col("time") >= HRES_ARCHIVE_CHANGE)
    non_null_before = before.select(pl.sum_horizontal(pl.col(columns).is_not_null()).sum()).item()
    last_before = before.filter(pl.any_horizontal(pl.col(columns).is_not_null()))["time"].max()
    ratios = _jump_ratios(
        changes=_hourly_changes(previous=previous, variable=PREVIOUS_DAY_VARIABLE).filter(
            pl.col("period") == LEAD_PERIODS[1]
        )
    )
    return {
        "n_columns": len(columns),
        "n_cells_before": before.height * len(columns),
        "non_null_before": int(non_null_before),
        "last_non_null_before": f"{last_before:%Y-%m-%d}",
        "n_cells_from": after.height * len(columns),
        "non_null_from": int(
            after.select(pl.sum_horizontal(pl.col(columns).is_not_null()).sum()).item()
        ),
        "jump_hours_from": [
            hour for hour, ratio in enumerate(ratios) if ratio >= JUMP_RATIO_THRESHOLD
        ],
        "ratios_from": ratios,
    }


def _previous_day_lines(*, evidence: PreviousDayEvidence) -> list[str]:
    """Render the `_previous_day*` evidence for the archive-source change as markdown.

    Args:
        evidence: `previous_day_evidence`'s result.

    Returns:
        Markdown lines.
    """
    ratios = ", ".join(
        f"{hour:02d}: {ratio:.2f}" for hour, ratio in enumerate(evidence["ratios_from"])
    )
    return [
        "#### The Previous Runs file's `_previous_day*` columns and the archive-source change",
        "",
        (
            f"The wind farms' rows from {ROW_SET_START_DATE:%Y-%m-%d} hold "
            f"{evidence['n_cells_before']:,} cells in the {evidence['n_columns']} "
            f"`_previous_day*` columns before {HRES_ARCHIVE_CHANGE_DATE:%Y-%m-%d}, of which "
            f"{evidence['non_null_before']:,} are not null (the last on "
            f"{evidence['last_non_null_before']}), and {evidence['n_cells_from']:,} cells from "
            f"that date, of which {evidence['non_null_from']:,} are not null. From "
            f"{HRES_ARCHIVE_CHANGE_DATE:%Y-%m-%d} `{PREVIOUS_DAY_VARIABLE}`'s hour-to-hour "
            "change, over the mean of its two neighbours' changes, reaches "
            f"{JUMP_RATIO_THRESHOLD:.2f} or more at UTC hours {evidence['jump_hours_from']}. "
            f"All 24 ratios: {ratios}."
        ),
    ]


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
            f"{JUMP_RATIO_THRESHOLD:.2f} or more are in bold. The threshold was chosen after the "
            "first results."
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
        expected = list(PLAN_EXPECTED_JUMP_HOURS[period])
        for variable in LEAD_VARIABLES:
            jump_hours = sorted(
                lead_table.filter(
                    pl.col("period") == period,
                    pl.col("variable") == variable,
                    pl.col("ratio") >= JUMP_RATIO_THRESHOLD,
                )["hour"].to_list()
            )
            reached = [hour for hour in expected if hour in jump_hours]
            at_plan_threshold = sorted(
                lead_table.filter(
                    pl.col("period") == period,
                    pl.col("variable") == variable,
                    pl.col("ratio") >= PLAN_JUMP_RATIO_THRESHOLD,
                )["hour"].to_list()
            )
            reached_at_plan = [hour for hour in expected if hour in at_plan_threshold]
            lines.append(
                f"- {period}, `{variable}`: UTC hours with a ratio of "
                f"{JUMP_RATIO_THRESHOLD:.2f} or more: {jump_hours}. Of the expected handover "
                f"hours, {expected}, {len(reached)} of {len(expected)} reach the threshold: "
                f"{reached}. At the plan's threshold of {PLAN_JUMP_RATIO_THRESHOLD:.2f}, "
                f"{len(reached_at_plan)} of {len(expected)} reach it: {reached_at_plan}."
            )
    lines += [
        "",
        (
            "Lead statement, as the plan wrote it before the first fit and as the script prints "
            "it whatever the table shows. HRES's served lead is 1 to 12 h before 1 October 2025 "
            "and 0 to 5 h from it, inferred from where the hour-to-hour jumps fall (not "
            "documented by Open-Meteo, whose documentation says only that each run's first few "
            "hours are stitched into a continuous series). The expected jumps before 1 October "
            "2025 are at 01 and 13 UTC, the arrival hours of the 00 and 12 UTC runs, and from "
            "1 October 2025 at 00, 06, 12 and 18 UTC, the four daily runs. An hour that "
            "reaches the threshold is "
            "evidence of a handover, not proof of one, and the table can show further hours above "
            "the threshold that no run schedule explains. The plan fixed the threshold at "
            "1.15, and 1.10 was chosen after the first results."
        ),
    ]
    return lines


def run_checks(
    *, sites: pl.DataFrame, frame: pl.DataFrame, counts: list[tuple[str, int]]
) -> ChecksResult:
    """Run every pre-fit check once, before any arm is fitted.

    Args:
        sites: The wind roster.
        frame: The row set with eras and folds, carrying `power_mw`.
        counts: `joined_row_set`'s row counts.

    Returns:
        Every check's raw result.
    """
    return {
        "row_counts": counts,
        "coverage": calendar_month_coverage(frame=frame),
        "grid_check": hres_grid_check(sites=sites),
        "ens_power": ens_power_check(base=frame),
        "lead_table": hres_lead_table(sites=sites),
        "previous_day": previous_day_evidence(sites=sites),
    }


def _raise_on_failed_checks(*, checks: ChecksResult) -> None:
    """Raise if any pre-fit check fails, before any arm is fitted on rows the checks distrust.

    Args:
        checks: `run_checks`'s result.

    Raises:
        ValueError: Naming every failed check.
    """
    raise_on_uncovered_months(coverage=checks["coverage"])
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
            "- HRES fetch: no `cell_selection` was set, so Open-Meteo's default (`land`) applies. "
            "`fetch_wind_point.py` sets `land` for every other product, ERA5 included."
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


def _arm_jobs(
    *,
    products: tuple[str, ...],
    setting: str,
    hyper_parameters: HyperParameters,
) -> list[Job]:
    """Return one job per product at one hyperparameter setting.

    Args:
        products: Product keys.
        setting: `pooled` or `sensitivity`, the label saved with the losses.
        hyper_parameters: The XGBoost hyperparameters of the setting.

    Returns:
        One job per product, each with the page's seven columns.
    """
    return [
        (
            f"{product}_wind",
            setting,
            "power_mw",
            (*SHARED_FEATURES, *_wind_columns(product=product)),
            hyper_parameters,
            False,
        )
        for product in products
    ]


def jobs() -> list[Job]:
    """Return every arm's job: the primary setting for every product, the second for four.

    Returns:
        One job per (arm, setting).

    Raises:
        ValueError: If two arms hold different numbers of feature columns.
    """
    job_list = [
        *_arm_jobs(
            products=ALL_PRODUCTS, setting="pooled", hyper_parameters=PRIMARY_HYPER_PARAMETERS
        ),
        *_arm_jobs(
            products=SENSITIVITY_PRODUCTS,
            setting="sensitivity",
            hyper_parameters=SENSITIVITY_HYPER_PARAMETERS,
        ),
    ]
    counts = {len(features) for _, _, _, features, _, _ in job_list}
    if len(counts) != 1:
        msg = f"every arm should carry the same number of feature columns, found counts {counts}"
        raise ValueError(msg)
    return job_list


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
    excludes = _excludes_cell(
        lower=interval["lower_95"], upper=interval["upper_95"], n_months=interval["n_months"]
    )
    log.append(
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
        f"| {label} | {treatment} − {reference} | {difference:+.4f} | "
        f"[{lower:+.4f}, {upper:+.4f}] | {excludes} | "
        f"{agreeing} of {len(folds)} | {interval['n_rows']:,} |"
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
        log.append(
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
            f"| {arm} | {value:.4f} | [{lower:.4f}, {upper:.4f}] | {interval['n_rows']:,} "
            f"| {interval['n_months']} |"
        )
    return lines


def _bonferroni_lines(*, losses: pl.DataFrame, setting: str, log: IntervalLog) -> list[str]:
    """Render the planned contrasts with intervals adjusted for testing three at once.

    Each interval is a percentile interval of the same month-and-seed resampling as the 95%
    intervals, at the level `BONFERRONI_LEVEL`.

    Args:
        losses: Per-row losses at one setting, holding the arms of the planned contrasts.
        setting: `pooled` or `sensitivity`.
        log: Where the intervals are recorded.

    Returns:
        Markdown lines.
    """
    lines = [
        "| Contrast | ΔMAE (pp of capacity) | Adjusted interval | Excludes zero? | Months |",
        "|---|---|---|---|---|",
    ]
    for name, treatment, reference in PLANNED_CONTRASTS:
        interval = bootstrap_difference(
            losses=losses, treatment=treatment, reference=reference, metric=METRIC
        )
        lower, upper = (
            bound * PERCENTAGE_POINTS
            for bound in bootstrap_difference_at_level(
                losses=losses,
                treatment=treatment,
                reference=reference,
                metric=METRIC,
                level=BONFERRONI_LEVEL,
            )
        )
        difference = interval["difference"] * PERCENTAGE_POINTS
        log.append(
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
                "n_rows": interval["n_rows"],
                "n_months": interval["n_months"],
                "folds_agreeing": None,
                "n_folds": None,
            }
        )
        excludes = _excludes_cell(lower=lower, upper=upper, n_months=interval["n_months"])
        lines.append(
            f"| {name}: {treatment} − {reference} | {difference:+.4f} "
            f"| [{lower:+.4f}, {upper:+.4f}] | {excludes} "
            f"| {interval['n_months']} |"
        )
    tail_draws = N_BOOTSTRAP_RESAMPLES * (100.0 - BONFERRONI_LEVEL) / 200.0
    lines += [
        "",
        (
            "The adjustment covers the three planned contrasts of this table's own family, not "
            "every contrast in the report. Each tail of an interval at this level rests on about "
            f"{tail_draws:.0f} of the {N_BOOTSTRAP_RESAMPLES:,} resamples."
        ),
    ]
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
                pairs=PERIOD_CONTRASTS,
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
    lines = [
        "#### ENS contrasts by label hour, a lead and time-of-day split (exploratory)",
        "",
        (
            "Labels 00 to 08 UTC against labels 10 to 23 UTC; label 09 is dropped. The split "
            "separates ENS leads 0 to 8 h from leads 10 to 23 h from the 00 UTC run, and early "
            "hours of the day from late hours, so the two halves differ in lead and in time of "
            "day together. HRES's lead before 1 October 2025 is also mixed into the split. The "
            "split makes no claim about when a run can be read. The last three contrasts involve "
            "no ENS lead, and ERA5 is an analysis with no lead, so a change of those between the "
            "halves is a change with time of day."
        ),
        "",
    ]
    for label, hours in zip(
        ("labels 00-08 UTC", "labels 10-23 UTC"), SPLIT_LABEL_HOURS, strict=True
    ):
        part = pooled.filter(pl.col("time").dt.hour().is_in(list(hours)))
        lines += _contrast_table(
            losses=part,
            pairs=SPLIT_CONTRASTS,
            label=label,
            section="lead and time of day",
            setting="pooled",
            log=log,
        )
        lines.append("")
    return lines


def _horizons_lines() -> list[str]:
    """Render the horizons study's published ENS-against-ERA5 figure.

    Returns:
        Markdown lines. The published figure is re-derived from the horizons report's own table.
        This study's re-estimate of it is the `ens_mean_day0_wind − era5_wind` row of the
        exploratory table above.

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
            "The `ens_mean_day0_wind − era5_wind` row of the exploratory table above "
            "re-estimates it on this study's shorter row set."
        ),
    ]


CHANGE_HEADER: Final[tuple[str, str]] = (
    (
        "| Scope | Contrast | Change in the contrast (pp of capacity) | 95% interval "
        "| Excludes zero? | Rows | Months |"
    ),
    "|---|---|---|---|---|---|---|",
)
"""The header of every table that compares one contrast between two groups of rows."""

CUT_EFFECT_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("ens_mean_day0_wind", "era5_wind"),
    ("ens_mean_day0_speed_components_wind", "era5_wind"),
    ("hres_wind", "era5_wind"),
)
"""The long-row-set contrasts whose dependence on the fold design and the era cut is measured."""

FARM_PAIRS: Final[tuple[tuple[str, str], ...]] = (("W2", "W3"), ("W3", "W1"), ("W2", "W1"))
"""Each pair of farms, as (first, second): the printed difference is first minus second."""

STEP_SPLITS: Final[tuple[tuple[tuple[str, ...], datetime], ...]] = (
    (("hres", "ukv"), datetime(2024, 11, 12, 6, tzinfo=UTC)),
    (("ens_mean_day0",), datetime(2024, 11, 13, 0, tzinfo=UTC)),
)
"""Each group of products with the first hour of IFS Cycle 49r1 in its data: the 06 UTC HRES run of
12 November 2024, and the first 00 UTC ENS run, of 13 November 2024. UKV is a control at the HRES
split, since IFS Cycle 49r1 does not touch it."""

STEP_WINDOW_DAYS: Final[int] = 28
"""The days either side of a step split that the season-matched rows of the step table cover."""

JUMP_CONTROL_HOURS: Final[tuple[int, ...]] = (7, 19)
"""The UTC hours of the jump control: the hour of the jump in HRES's 100 m speed that no run
schedule explains, and the hour twelve hours later."""

JUMP_CONTROL_PRODUCTS: Final[tuple[str, ...]] = ("era5", "ukv", "hres")
"""The products whose hour-to-hour jump ratios are compared in the jump control."""

PLAN_JUMP_RATIO_THRESHOLD: Final[float] = 1.15
"""The jump threshold the plan fixed before the first results."""

DESIGN_RANGE_CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = FOLD_DESIGN_CONTRASTS
"""The contrasts whose range across every design scored on the rows from 2024-12-01 is printed."""


class ChangeInterval(TypedDict):
    """A difference between the mean of two groups of rows, and its interval."""

    change: float
    lower: float
    upper: float
    n_rows: int
    n_months: int


def _contrast_rows(
    *, losses: pl.DataFrame, treatment: str, reference: str
) -> tuple[pl.DataFrame, np.ndarray]:
    """Return one contrast's per-seed, per-row differences with each row's site, time and month.

    Args:
        losses: Per-row losses holding both arms.
        treatment: The arm whose error is being compared.
        reference: The arm it is compared against.

    Returns:
        The rows' `site`, `time` and `month`, in the order of the second value's columns, and the
        differences in percentage points of capacity, shape (n_seeds, n_rows).

    Raises:
        ValueError: If the keys and the differences disagree on the number of rows.
    """
    differences, _ = paired_differences(
        losses=losses, treatment=treatment, reference=reference, metric=METRIC
    )
    first_seed = losses["seed"].min()
    keys = (
        losses.filter(pl.col("arm") == reference, pl.col("seed") == first_seed)
        .select("site", "time", "month")
        .join(
            losses.filter(pl.col("arm") == treatment, pl.col("seed") == first_seed).select(
                "site", "time"
            ),
            on=["site", "time"],
            how="inner",
        )
        .sort("site", "time")
    )
    if keys.height != differences.shape[1]:
        msg = f"{keys.height} keyed rows but {differences.shape[1]} paired differences"
        raise ValueError(msg)
    return keys, differences * PERCENTAGE_POINTS


def _rows_by_month(*, months: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Group the masked rows' indices by month label.

    Args:
        months: Each row's month label.
        mask: Which rows to keep.

    Returns:
        Each month label to the indices of its kept rows.
    """
    indices = np.flatnonzero(mask)
    labels = months[indices]
    return {str(label): indices[labels == label] for label in np.unique(labels)}


def _draw_rows(*, by_month: dict[str, np.ndarray], generator: np.random.Generator) -> np.ndarray:
    """Draw as many whole months as the group holds, with replacement, and return their rows.

    Args:
        by_month: Each month label to the indices of its rows.
        generator: The random stream.

    Returns:
        The concatenated row indices of the drawn months.
    """
    labels = sorted(by_month)
    drawn = generator.integers(0, len(labels), size=len(labels))
    return np.concatenate([by_month[labels[index]] for index in drawn])


def _change_interval(
    *,
    values: np.ndarray,
    months: np.ndarray,
    minuend: np.ndarray,
    subtrahend: np.ndarray | None,
    joint_months: bool,
) -> ChangeInterval:
    """Interval the mean of one group of rows minus the mean of another, resampling months.

    Every resample draws one fitting seed and whole calendar months, as `bootstrap_difference`
    does. With `joint_months`, one draw of months serves both groups, which is right where both
    groups hold rows of the same months (the two halves of the day, two farms), because the swing
    the groups share then cancels. Without it, each group draws its own months from its own months
    alone, which is right where the groups hold different months (two periods). With no `subtrahend`
    group, the statistic is the mean of the `minuend` group alone.

    Args:
        values: Per-seed, per-row values, shape (n_seeds, n_rows).
        months: Each row's month label.
        minuend: Which rows form the group whose mean is read first.
        subtrahend: Which rows form the group subtracted from it, or None.
        joint_months: Whether both groups draw the same months. Both groups are then restricted to
            the months that both hold.

    Returns:
        The change, its 2.5th and 97.5th percentiles, and the rows and months it rests on. With two
        groups drawing their own months, the months are the smaller group's.
    """
    if subtrahend is not None and joint_months:
        shared = set(_rows_by_month(months=months, mask=minuend)) & set(
            _rows_by_month(months=months, mask=subtrahend)
        )
        keep = np.isin(months, sorted(shared))
        minuend, subtrahend = minuend & keep, subtrahend & keep
    minuend_rows = _rows_by_month(months=months, mask=minuend)
    subtrahend_rows = None if subtrahend is None else _rows_by_month(months=months, mask=subtrahend)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    resampled = np.empty(N_BOOTSTRAP_RESAMPLES)
    for resample in range(N_BOOTSTRAP_RESAMPLES):
        seed_index = generator.integers(0, values.shape[0])
        if subtrahend_rows is None:
            drawn = _draw_rows(by_month=minuend_rows, generator=generator)
            resampled[resample] = values[seed_index, drawn].mean()
            continue
        if joint_months:
            labels = sorted(minuend_rows)
            draw = generator.integers(0, len(labels), size=len(labels))
            minuend_drawn = np.concatenate([minuend_rows[labels[index]] for index in draw])
            subtrahend_drawn = np.concatenate([subtrahend_rows[labels[index]] for index in draw])
        else:
            minuend_drawn = _draw_rows(by_month=minuend_rows, generator=generator)
            subtrahend_drawn = _draw_rows(by_month=subtrahend_rows, generator=generator)
        resampled[resample] = (
            values[seed_index, minuend_drawn].mean() - values[seed_index, subtrahend_drawn].mean()
        )
    change = float(values[:, minuend].mean())
    if subtrahend is not None:
        change -= float(values[:, subtrahend].mean())
    n_months = (
        len(minuend_rows)
        if subtrahend_rows is None
        else min(len(minuend_rows), len(subtrahend_rows))
    )
    return {
        "change": change,
        "lower": float(np.percentile(resampled, 2.5)),
        "upper": float(np.percentile(resampled, 97.5)),
        "n_rows": int(minuend.sum() + (0 if subtrahend is None else subtrahend.sum())),
        "n_months": n_months,
    }


def _excludes_cell(*, lower: float, upper: float, n_months: int) -> str:
    """Return the "Excludes zero?" cell: `too few months` where the interval rests on too few.

    Args:
        lower: The interval's lower bound.
        upper: The interval's upper bound.
        n_months: The calendar months the interval resamples.

    Returns:
        `too few months` below `MIN_MONTHS_FOR_INTERVAL` months, else `**yes**` or `no`.
    """
    if n_months < MIN_MONTHS_FOR_INTERVAL:
        return "too few months"
    return "**yes**" if lower > 0.0 or upper < 0.0 else "no"


def _change_line(
    *,
    scope: str,
    contrast: tuple[str, str],
    interval: ChangeInterval,
    section: str,
    log: IntervalLog,
) -> str:
    """Return one markdown row of a change table, and record its interval in `log`.

    Args:
        scope: The scope cell, naming what changes between the two groups.
        contrast: The (treatment, reference) whose difference changes.
        interval: `_change_interval`'s result.
        section: The report section, saved with the interval.
        log: Where the interval is recorded.

    Returns:
        The table row.
    """
    treatment, reference = contrast
    log.append(
        {
            "section": section,
            "setting": "pooled",
            "scope": scope,
            "treatment": treatment,
            "reference": reference,
            "value": interval["change"],
            "lower": interval["lower"],
            "upper": interval["upper"],
            "level": 95.0,
            "n_rows": interval["n_rows"],
            "n_months": interval["n_months"],
            "folds_agreeing": None,
            "n_folds": None,
        }
    )
    excludes = _excludes_cell(
        lower=interval["lower"], upper=interval["upper"], n_months=interval["n_months"]
    )
    return (
        f"| {scope} | {treatment} − {reference} | {interval['change']:+.4f} "
        f"| [{interval['lower']:+.4f}, {interval['upper']:+.4f}] | {excludes} "
        f"| {interval['n_rows']:,} | {interval['n_months']} |"
    )


def _design_difference_lines(*, losses: pl.DataFrame, log: IntervalLog) -> list[str]:
    """Render how much the era cut and the fold rotation each move a contrast, from saved losses.

    The rotated-folds design differs from the horizons design only in its folds, and the design
    with the extra era cut differs from the rotated-folds design only in the cut (and the folds its
    new eras imply). The change is each contrast's value under one design minus its value under the
    other, on the same rows, so the interval resamples the months and the seed of that per-row
    difference. Nothing is refitted.

    Args:
        losses: The long-row-set losses, with `design`.
        log: Where the intervals are recorded.

    Returns:
        Markdown lines.
    """
    steps = (
        ("fold rotation (rotated folds minus horizons folds)", ROTATED_DESIGN, HORIZONS_DESIGN),
        ("extra era cut (cut design minus rotated folds)", CUT_DESIGN, ROTATED_DESIGN),
        ("both (cut design minus horizons folds)", CUT_DESIGN, HORIZONS_DESIGN),
    )
    lines = [
        "#### What the fold rotation and the era cut each change (exploratory)",
        "",
        (
            "Each row is a contrast's value under one long-row design minus its value under "
            "another, on the same rows and the same fitting seeds, from the saved losses, with "
            "nothing refitted. The interval resamples whole calendar months and one seed of the "
            "per-row difference. A positive change means the first design gives the larger "
            "contrast."
        ),
        "",
        *CHANGE_HEADER,
    ]
    for scope, cut in (("all rows", None), ("rows from 2024-12-01", ROW_SET_START_DATE)):
        for treatment, reference in CUT_EFFECT_CONTRASTS:
            per_design = {}
            for design in (HORIZONS_DESIGN, ROTATED_DESIGN, CUT_DESIGN):
                part = losses.filter(pl.col("design") == design)
                if cut is not None:
                    part = part.filter(pl.col("time") >= cut)
                per_design[design] = _contrast_rows(
                    losses=part, treatment=treatment, reference=reference
                )
            for name, plus, minus in steps:
                keys, values_plus = per_design[plus]
                keys_minus, values_minus = per_design[minus]
                if not keys.equals(keys_minus):
                    msg = f"designs {plus!r} and {minus!r} hold different rows"
                    raise ValueError(msg)
                interval = _change_interval(
                    values=values_plus - values_minus,
                    months=keys["month"].to_numpy(),
                    minuend=np.ones(keys.height, dtype=bool),
                    subtrahend=None,
                    joint_months=True,
                )
                lines.append(
                    _change_line(
                        scope=f"{scope}: {name}",
                        contrast=(treatment, reference),
                        interval=interval,
                        section="design differences",
                        log=log,
                    )
                )
    return lines


def _label_hour_change_lines(*, pooled: pl.DataFrame, log: IntervalLog) -> list[str]:
    """Render each contrast's change between the early and the late label hours, with intervals.

    The rows are the label hours of `SPLIT_LABEL_HOURS`. A contrast's change between the halves is
    late minus early. The two halves hold the same calendar months, so one draw of months serves
    both halves.

    Args:
        pooled: Per-row losses at the primary setting.
        log: Where the intervals are recorded.

    Returns:
        Markdown lines.
    """
    lines = [
        "#### Change of each contrast between the early and the late label hours (exploratory)",
        "",
        (
            "Each row is a contrast's mean over labels 10 to 23 UTC minus its mean over labels 00 "
            "to 08 UTC, from the saved losses. One draw of whole calendar months and one fitting "
            "seed serves both halves, because both halves hold the same months. The controls do "
            "not isolate time of day: HRES's served lead varies with the UTC hour before "
            "2025-10-01, and ERA5's assimilation windows change at 09 to 10 and 21 to 22 UTC."
        ),
        "",
        *CHANGE_HEADER,
    ]
    early_hours, late_hours = SPLIT_LABEL_HOURS
    for treatment, reference in SPLIT_CONTRASTS:
        keys, values = _contrast_rows(losses=pooled, treatment=treatment, reference=reference)
        hours = keys["time"].dt.hour().to_numpy()
        interval = _change_interval(
            values=values,
            months=keys["month"].to_numpy(),
            minuend=np.isin(hours, list(late_hours)),
            subtrahend=np.isin(hours, list(early_hours)),
            joint_months=True,
        )
        lines.append(
            _change_line(
                scope="labels 10-23 UTC minus labels 00-08 UTC",
                contrast=(treatment, reference),
                interval=interval,
                section="label-hour change",
                log=log,
            )
        )
    return lines


def _between_farm_lines(*, pooled: pl.DataFrame, log: IntervalLog) -> list[str]:
    """Render each planned contrast's difference between two farms, with intervals.

    Args:
        pooled: Per-row losses at the primary setting.
        log: Where the intervals are recorded.

    Returns:
        Markdown lines.
    """
    lines = [
        "#### Differences between farms in the planned contrasts (exploratory)",
        "",
        (
            "Each row is a contrast's mean at the first farm minus its mean at the second farm, "
            "from the saved losses. One draw of whole calendar months and one fitting seed serves "
            "both farms. Three farms make three pairs, and the pairs share farms."
        ),
        "",
        *CHANGE_HEADER,
    ]
    for _, treatment, reference in PLANNED_CONTRASTS:
        keys, values = _contrast_rows(losses=pooled, treatment=treatment, reference=reference)
        sites = keys["site"].to_numpy()
        for first_farm, second_farm in FARM_PAIRS:
            interval = _change_interval(
                values=values,
                months=keys["month"].to_numpy(),
                minuend=sites == first_farm,
                subtrahend=sites == second_farm,
                joint_months=True,
            )
            lines.append(
                _change_line(
                    scope=f"{first_farm} minus {second_farm}",
                    contrast=(treatment, reference),
                    interval=interval,
                    section="between farms",
                    log=log,
                )
            )
    return lines


def _period_change_lines(*, pooled: pl.DataFrame, log: IntervalLog) -> list[str]:
    """Render each contrast's change between the periods of each split, with intervals.

    The two periods hold different months, so each period draws its own months.

    Args:
        pooled: Per-row losses at the primary setting.
        log: Where the intervals are recorded.

    Returns:
        Markdown lines.
    """
    boundaries = {
        "HRES archive source": HRES_ARCHIVE_CHANGE_DATE,
        "IFS Cycle 50r1": IFS_CYCLE_50R1_DATE,
    }
    lines = [
        "#### Change of each contrast between the periods of each split (exploratory)",
        "",
        (
            "Each row is a contrast's mean in the later period minus its mean in the earlier "
            "period, from the saved losses. Each period draws its own whole calendar months, and "
            "one fitting seed serves both. A period of fewer than "
            f"{MIN_MONTHS_FOR_INTERVAL} calendar months is marked `too few months`."
        ),
        "",
        *CHANGE_HEADER,
    ]
    for change, before_label, after_label in PERIOD_SPLITS:
        boundary = boundaries[change]
        for treatment, reference in PERIOD_CONTRASTS:
            keys, values = _contrast_rows(losses=pooled, treatment=treatment, reference=reference)
            times = keys["time"].to_numpy()
            cut = np.datetime64(boundary.replace(tzinfo=None), "us")
            interval = _change_interval(
                values=values,
                months=keys["month"].to_numpy(),
                minuend=times >= cut,
                subtrahend=times < cut,
                joint_months=False,
            )
            lines.append(
                _change_line(
                    scope=f"{after_label} minus {before_label}",
                    contrast=(treatment, reference),
                    interval=interval,
                    section=f"period change: {change}",
                    log=log,
                )
            )
    return lines


def _jump_control_lines(*, frame: pl.DataFrame) -> list[str]:
    """Render the hour-to-hour jump ratio at chosen UTC hours for ERA5, UKV and HRES.

    HRES's 100 m speed jumps at 07 UTC by a ratio that no run schedule explains. ERA5 is an hourly
    analysis and UKV's served value is its analysis, so neither hands over between runs at 07 UTC.
    The same ratio for them shows how much of a jump at 07 UTC is the morning boundary layer and
    not a handover. The ratios use the main row set, from 2024-12-01.

    Args:
        frame: The main row set, carrying `time`, `site` and every product's columns.

    Returns:
        Markdown lines: the ratio at each control hour by period, then the 07 UTC ratio by
        calendar month.
    """
    changes = {
        product: _hourly_changes(
            previous=frame.select("site", "time", variable=_wind_columns(product=product)[0]).sort(
                "site", "time"
            ),
            variable="variable",
        )
        for product in JUMP_CONTROL_PRODUCTS
    }
    lines = [
        "#### Jump ratio of the 100 m speed at control hours, by product (exploratory)",
        "",
        (
            "The ratio is the one of the HRES served-lead table: an hour's mean absolute "
            "hour-to-hour change over the mean of its two neighbours' changes, from the main row "
            "set. ERA5 and UKV involve no handover between forecast runs, so their ratios show "
            "what a jump at these hours looks like without one."
        ),
        "",
        "| Product | Period | "
        + " | ".join(f"{hour:02d} UTC" for hour in JUMP_CONTROL_HOURS)
        + " |",
        "|---|---|" + "---|" * len(JUMP_CONTROL_HOURS),
    ]
    for product in JUMP_CONTROL_PRODUCTS:
        for period in LEAD_PERIODS:
            ratios = _jump_ratios(changes=changes[product].filter(pl.col("period") == period))
            lines.append(
                f"| {product} | {period} | "
                + " | ".join(f"{ratios[hour]:.2f}" for hour in JUMP_CONTROL_HOURS)
                + " |"
            )
    lines += [
        "",
        f"The {JUMP_CONTROL_HOURS[0]:02d} UTC ratio by calendar month, all years of the row set:",
        "",
        "| Calendar month | " + " | ".join(JUMP_CONTROL_PRODUCTS) + " |",
        "|---|" + "---|" * len(JUMP_CONTROL_PRODUCTS),
    ]
    for calendar_month in range(1, 13):
        cells = []
        for product in JUMP_CONTROL_PRODUCTS:
            ratios = _jump_ratios(
                changes=changes[product].filter(pl.col("time").dt.month() == calendar_month)
            )
            cells.append(f"{ratios[JUMP_CONTROL_HOURS[0]]:.2f}")
        lines.append(f"| {calendar_month} | " + " | ".join(cells) + " |")
    return lines


def _design_range_lines(*, log: IntervalLog) -> list[str]:
    """Render P1 to P3 and ENS-HRES under every design scored on the rows from 2024-12-01.

    The designs are the study's own, the four fold designs, and the three long-row designs scored
    on the rows from 2024-12-01. They share those months, so they are not independent confirmations.

    Args:
        log: The intervals recorded so far, which must hold the fold-design and long-row rows.

    Returns:
        Markdown lines: one row per design, then the range of each contrast's estimate and bounds.
    """
    scored = [
        record
        for record in log
        if record["section"] == "fold designs"
        or (
            record["section"].startswith("long rows: ")
            and record["scope"] == "rows from 2024-12-01"
        )
    ]
    designs = list(dict.fromkeys((record["section"], record["scope"]) for record in scored))
    lines = [
        (
            "#### P1 to P3 and ENS-HRES under every design scored on the rows from 2024-12-01 "
            "(exploratory)"
        ),
        "",
        (
            "Each cell is the estimate and the 95% interval, in percentage points of capacity. "
            "The designs share their months and their weather, so they are not independent "
            "confirmations of each other. The long-row designs train on the rows from 2024-08-12 "
            "and are scored on the rows from 2024-12-01 only. The row-count column gives the "
            "rows the design scores."
        ),
        "",
        "| Design | Rows | " + " | ".join(name for name, _, _ in DESIGN_RANGE_CONTRASTS) + " |",
        "|---|---|" + "---|" * len(DESIGN_RANGE_CONTRASTS),
    ]
    estimates: dict[str, list[tuple[float, float, float]]] = {
        name: [] for name, _, _ in DESIGN_RANGE_CONTRASTS
    }
    for section, scope in designs:
        cells = []
        n_rows = 0
        for name, treatment, reference in DESIGN_RANGE_CONTRASTS:
            matches = [
                record
                for record in scored
                if (record["section"], record["scope"]) == (section, scope)
                and (record["treatment"], record["reference"]) == (treatment, reference)
            ]
            if len(matches) != 1:
                msg = f"{(section, scope, name)} matches {len(matches)} recorded intervals"
                raise ValueError(msg)
            record = matches[0]
            n_rows = record["n_rows"]
            estimates[name].append((record["value"], record["lower"], record["upper"]))
            cells.append(f"{record['value']:+.2f} [{record['lower']:+.2f}, {record['upper']:+.2f}]")
        label = scope if section == "fold designs" else section.removeprefix("long rows: ")
        lines.append(f"| {label} | {n_rows:,} | " + " | ".join(cells) + " |")
    lines += [
        "",
        (
            "| Contrast | Smallest estimate | Largest estimate | Lowest lower bound "
            "| Highest upper bound |"
        ),
        "|---|---|---|---|---|",
    ]
    for name, values in estimates.items():
        lines.append(
            f"| {name} | {min(v[0] for v in values):+.4f} | {max(v[0] for v in values):+.4f} "
            f"| {min(v[1] for v in values):+.4f} | {max(v[2] for v in values):+.4f} |"
        )
    return lines


def _step_lines(*, frame: pl.DataFrame) -> list[str]:
    """Render each product's mean speed over ERA5's before and after IFS Cycle 49r1 first appears.

    The ratio is read at 100 m (the hub-height speed the arms are given) and at 10 m. HRES and UKV
    split at 06 UTC on 12 November 2024, the first 06 UTC HRES run of IFS Cycle 49r1, and ENS at
    00 UTC on 13 November 2024, the first ENS run of 49r1 at 00 UTC. UKV is unaffected by the
    cycle, so it shows what a split at that date does to a product with no cycle change. The first
    pair of columns uses every row of the long row set on each side, and the second pair uses the
    `STEP_WINDOW_DAYS` days on each side, so that the season is nearly the same.

    Args:
        frame: The long row set, carrying `time` and every product's columns.

    Returns:
        Markdown lines.
    """
    era5 = {
        height: _mean_speed_ms(product="era5", column=column) for height, column in RATIO_HEIGHTS
    }
    lines = [
        "#### Ratio to ERA5's mean speed before and after IFS Cycle 49r1 (exploratory)",
        "",
        (
            "The split is at the first hour of each product's data that comes from a 49r1 run: "
            "the 06 UTC HRES run of 2024-11-12, and the 00 UTC ENS run of 2024-11-13. UKV, which "
            "IFS Cycle 49r1 does not touch, is split at the HRES hour as a control. Speeds are "
            "means over the long row set. The first two ratio columns use every row on each "
            f"side of the split, and the last two use the {STEP_WINDOW_DAYS} days on each side."
        ),
        "",
        (
            "| Product | Height | Split (UTC) | Ratio before | Ratio after | "
            f"Ratio, {STEP_WINDOW_DAYS} days before | Ratio, {STEP_WINDOW_DAYS} days after |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for products, split in STEP_SPLITS:
        window = timedelta(days=STEP_WINDOW_DAYS)
        for product in products:
            for height, column in (("100 m", 0), ("10 m", 3)):
                ratio = _mean_speed_ms(product=product, column=column) / era5[height]
                cells = []
                for low, high in (
                    (None, split),
                    (split, None),
                    (split - window, split),
                    (split, split + window),
                ):
                    part = frame
                    if low is not None:
                        part = part.filter(pl.col("time") >= low)
                    if high is not None:
                        part = part.filter(pl.col("time") < high)
                    cells.append(f"{float(part.select(ratio).item()):.4f}")
                lines.append(
                    f"| {product} | {height} | {split:%Y-%m-%d %H:%M} | " + " | ".join(cells) + " |"
                )
    return lines


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
    n_early = frame.filter(pl.col("time") < LONG_ROW_SET_START).height
    if n_early:
        msg = f"{n_early} of the page's rows are before the long row set's start"
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
    """Return the long row set under its three era-and-fold designs.

    Args:
        frame: `long_row_frame`'s result.

    Returns:
        The horizons study's design (two UKV eras, no cut at IFS Cycle 49r1, its own fold layout);
        the same eras with era 1's folds rotated by `HORIZONS_ROTATED_FOLD_OFFSETS`; and the same
        rows with one extra era cut at `IFS_CYCLE_49R1_CUT_MONTH`, whose folds are rotated by
        `LONG_ROW_FOLD_OFFSETS`. The second design differs from the first only in the folds, and
        from the third only in the cut.
    """
    return {
        HORIZONS_DESIGN: with_eras(frame=frame),
        ROTATED_DESIGN: rotate_folds(
            frame=with_eras(frame=frame), fold_offsets=HORIZONS_ROTATED_FOLD_OFFSETS
        ),
        CUT_DESIGN: cut_eras(
            frame=frame,
            first_months=(IFS_CYCLE_49R1_CUT_MONTH, UKV_UPGRADE_MONTH),
            fold_offsets=LONG_ROW_FOLD_OFFSETS,
        ),
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
    study_folds = cut_eras(
        frame=frame,
        first_months=ENS_HRES_WIND_ERA_START_MONTHS,
        fold_offsets=ENS_HRES_WIND_ERA_FOLD_OFFSETS,
    )
    return {
        "three eras, no fold rotation": cut_eras(
            frame=frame, first_months=ENS_HRES_WIND_ERA_START_MONTHS, fold_offsets=NO_FOLD_OFFSETS
        ),
        "two UKV eras (the page's design)": with_eras(frame=frame),
        "study folds, two-valued era_code": study_folds.with_columns(
            era_code=(pl.col("month") >= UKV_UPGRADE_MONTH).cast(pl.Int8)
        ),
        "extra era cut at IFS 50r1, May 2026 dropped": cut_eras(
            frame=frame.filter(pl.col("month") != IFS_CYCLE_50R1_MONTH),
            first_months=(*ENS_HRES_WIND_ERA_START_MONTHS, FIRST_MONTH_AFTER_50R1),
            fold_offsets=ERA_FOLD_OFFSETS_50R1,
        ),
    }


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


def _design_coverage_lines(*, designs: Mapping[str, pl.DataFrame], heading: str) -> list[str]:
    """Render, for each design, how many folds hold out a calendar month with no training row.

    A design with a non-zero count leaves some (site, fold, calendar month) cell with no training
    row for a calendar month that occurs in two years (issue #868), so its scores rest on models
    that have never seen that season.

    Args:
        designs: Each design's name to its frame, carrying `site`, `fold` and `time`.
        heading: The heading line above the table.

    Returns:
        Markdown lines.
    """
    lines = [
        heading,
        "",
        (
            "| Design | Cells (site, fold, calendar month) | Cells with no training row for a "
            "month that occurs in two years | Calendar months affected |"
        ),
        "|---|---|---|---|",
    ]
    for name, frame in designs.items():
        coverage = calendar_month_coverage(frame=frame)
        failing = uncovered_months(coverage=coverage)
        months = sorted(failing["calendar_month"].unique().to_list())
        lines.append(f"| {name} | {coverage.height} | {failing.height} | {months or 'none'} |")
    lines += ["", "The covered cell with the fewest training rows, in each design:", ""]
    lines += [
        f"- {name}: {_fewest_training_rows(coverage=calendar_month_coverage(frame=frame))}."
        for name, frame in designs.items()
    ]
    return lines


def _mean_speed_ms(*, product: str, column: int) -> pl.Expr:
    """Return the mean wind speed of one product at one of its speed columns, in m/s.

    Args:
        product: A key of `ALL_PRODUCTS`.
        column: The index into `_wind_columns`: 0 for the hub-height speed, 3 for the 10 m speed.

    Returns:
        An aggregate expression; Open-Meteo's km/h columns are converted.
    """
    divisor = KMH_PER_M_S if product in KMH_PRODUCTS else 1.0
    return pl.col(_wind_columns(product=product)[column]).mean() / divisor


RATIO_PRODUCTS: Final[tuple[str, ...]] = ("ens_mean_day0", "hres", "ukv")
"""The products whose 10 m speed is compared with ERA5's, in the ratio tables' column order."""

RATIO_SEASON_MONTHS: Final[tuple[int, ...]] = (8, 9, 10)
"""The calendar months of the season-controlled ratio table: August to October."""


RATIO_HEIGHTS: Final[tuple[tuple[str, int], ...]] = (("10 m", 3), ("100 m", 0))
"""Each height of the ratio tables, with its index into `_wind_columns`. The 10 m table comes first
because `ens_hres_past_wind_charts.py` reads the first monthly table it finds."""


def _ratio_lines(*, frame: pl.DataFrame) -> list[str]:
    """Render each product's mean speed over ERA5's, by month and by August to October.

    Args:
        frame: The long row set, carrying `month`, `time` and every product's columns.

    Returns:
        Markdown lines: for each height in `RATIO_HEIGHTS`, the monthly table, then the table over
        August to October of each complete year. The long row set starts on 2024-08-12, so August
        2024 is a part-month.
    """
    header = ["| {} | Rows | ENS / ERA5 | HRES / ERA5 | UKV / ERA5 |", "|---|---|---|---|---|"]
    lines: list[str] = []
    for height, column in RATIO_HEIGHTS:
        era5 = _mean_speed_ms(product="era5", column=column)
        ratios = [
            pl.len().alias("n"),
            *(
                (_mean_speed_ms(product=product, column=column) / era5).alias(product)
                for product in RATIO_PRODUCTS
            ),
        ]
        monthly = frame.group_by("month").agg(ratios).sort("month")
        seasonal = (
            frame.filter(pl.col("time").dt.month().is_in(RATIO_SEASON_MONTHS))
            .group_by(year=pl.col("time").dt.year())
            .agg(*ratios, n_months=pl.col("month").n_unique())
            .filter(pl.col("n_months") == len(RATIO_SEASON_MONTHS))
            .sort("year")
        )
        lines += [
            "",
            (
                f"Monthly ratio of each product's mean {height} wind speed to ERA5's, on the long "
                "row set:"
            ),
            "",
            header[0].format("Month"),
            header[1],
        ]
        lines += [
            f"| {row['month']} | {row['n']:,} | "
            + " | ".join(f"{row[product]:.4f}" for product in RATIO_PRODUCTS)
            + " |"
            for row in monthly.iter_rows(named=True)
        ]
        lines += [
            "",
            (
                f"Season-controlled ratio of each product's mean {height} wind speed to ERA5's, "
                "pooled over August to October of each year that holds all three months, on the "
                "long row set:"
            ),
            "",
            header[0].format("Months"),
            header[1],
        ]
        lines += [
            f"| August to October {row['year']} | {row['n']:,} | "
            + " | ".join(f"{row[product]:.4f}" for product in RATIO_PRODUCTS)
            + " |"
            for row in seasonal.iter_rows(named=True)
        ]
    return lines[1:]


def _horizons_row_gap_line(*, frame: pl.DataFrame) -> str:
    """Say which of the long row set's rows the horizons study's inputs hold no power for.

    The horizons study scores rows that carry power, and the long row set is the page's own rows
    joined to the same ENS inputs, so the two counts differ by the rows without power there.

    Args:
        frame: The long row set, carrying `site` and `time`.

    Returns:
        One sentence with the count and the calendar months of the rows.
    """
    scored = (
        pl.read_parquet(ENS_INPUTS_PATH)
        .filter(pl.col("day") == 0, pl.col("method") == "speed_components")
        .drop_nulls("power_mw")
        .select("site", "time")
    )
    missing = frame.select("site", "time").join(scored, on=["site", "time"], how="anti")
    by_month = (
        missing.group_by(month=pl.col("time").dt.strftime("%Y-%m"))
        .agg(n=pl.len())
        .sort("month")
        .iter_rows(named=True)
    )
    months = ", ".join(f"{row['month']} ({row['n']})" for row in by_month)
    return (
        f"The long row set holds {frame.height:,} farm-hours and the horizons study's inputs hold "
        f"{scored.height:,} farm-hours with power. The {missing.height:,} farm-hours of the long "
        f"row set that the inputs hold no power for fall in {months}."
    )


def _long_row_lines(
    *,
    frame: pl.DataFrame,
    designs: Mapping[str, pl.DataFrame],
    losses: pl.DataFrame,
    log: IntervalLog,
) -> list[str]:
    """Render the long-row-set reconciliation: contrasts per design and scope, and speed ratios.

    Args:
        frame: `long_row_frame`'s result.
        designs: `long_row_designs`'s result.
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
            "horizons study's: two UKV eras, its own fold layout, and no cut at IFS Cycle 49r1 "
            "(12 November 2024). Design two keeps the two eras and rotates the folds of era 1 by "
            f"{HORIZONS_ROTATED_FOLD_OFFSETS[1]}, so that no calendar month is left without "
            "training rows. Design three adds one era cut at 2024-12-01, and rotates the folds "
            f"of era 1 by {LONG_ROW_FOLD_OFFSETS[1]} and of era 2 by {LONG_ROW_FOLD_OFFSETS[2]} "
            "for the same reason. `ens_mean_day0` is the `components` combination and "
            "`ens_mean_day0_speed_components` is the horizons study's own. Each design is scored "
            "on all rows and on the rows from 2024-12-01 only."
        ),
        "",
        _horizons_row_gap_line(frame=frame),
        "",
        *_design_coverage_lines(
            designs=designs,
            heading="Calendar-month coverage of each long-row design, checked before its fit:",
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
    return [*lines, *_ratio_lines(frame=frame)]


def _fold_design_lines(
    *,
    study_frame: pl.DataFrame,
    fold_frames: Mapping[str, pl.DataFrame],
    main_pooled: pl.DataFrame,
    design_losses: pl.DataFrame,
    log: IntervalLog,
) -> list[str]:
    """Render P1 to P3 and ENS-HRES under the study's fold design and four others.

    Args:
        study_frame: The main row set with the study's eras and folds.
        fold_frames: The other designs' frames, from `fold_designs`.
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
            "main losses: the study's design, and the same losses without the rows of "
            f"{IFS_CYCLE_50R1_MONTH}, the rows the last design drops."
        ),
        "",
        *_design_coverage_lines(
            designs={"study design": study_frame, **fold_frames},
            heading="Calendar-month coverage of each fold design, checked before its fit:",
        ),
        "",
    ]
    study = main_pooled.filter(pl.col("arm").is_in(arms))
    scored = [
        ("study design", study),
        (
            "study design, May 2026 rows removed",
            study.filter(pl.col("month") != IFS_CYCLE_50R1_MONTH),
        ),
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
        *_previous_day_lines(evidence=checks["previous_day"]),
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
        *_horizons_lines(),
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
        *_label_hour_change_lines(pooled=pooled, log=log),
        "",
        *_period_lines(pooled=pooled, log=log),
        *_period_change_lines(pooled=pooled, log=log),
        "",
        *_between_farm_lines(pooled=pooled, log=log),
        "",
        *_jump_control_lines(frame=frame),
    ]
    if extras is not None:
        lines += [
            "",
            (
                "Post-review fits, run by `--extra-fits` at script commit "
                f"`{extras['commit']}`, exploratory (added after the first results)."
            ),
            "",
            *_long_row_lines(
                frame=extras["long_frame"],
                designs=extras["long_designs"],
                losses=extras["long_losses"],
                log=log,
            ),
            "",
            *_design_difference_lines(losses=extras["long_losses"], log=log),
            "",
            *_fold_design_lines(
                study_frame=frame,
                fold_frames=extras["fold_frames"],
                main_pooled=pooled,
                design_losses=extras["design_losses"],
                log=log,
            ),
            "",
            *_design_range_lines(log=log),
            "",
            *_step_lines(frame=extras["long_frame"]),
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
- `script_commit.txt`: the commit of the script that fitted `losses.parquet`, which the first
  paragraph of `report.md` names.
- `intervals.parquet`: every interval `report.md` prints, one row each, with its section, setting,
  scope, arms, value and bounds (percentage points of capacity), level, rows and months.
- `losses_long_rows.parquet` and `losses_fold_designs.parquet`: the post-review fits
  (`--extra-fits`, exploratory, added after the first results), in the same layout as
  `losses.parquet` plus a `design` column. The first refits ERA5, UKV, HRES and ENS on the rows from
  2024-08-12 under three designs, and the second refits them on the main row set under four fold
  designs. Each has its own `.fingerprint` file, and `script_commit_extra_fits.txt` records the
  commit that fitted them. `losses.parquet` is not touched by them.
- `report.md`: every table the docs page quotes, printed by the script and never transcribed. The
  exploratory tables of changes between two groups of rows have their own header, and their
  intervals are in `intervals.parquet` under the sections `design differences`, `label-hour change`,
  `between farms` and `period change: ...`, with the change in `value`.
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
    long_jobs = _arm_jobs(
        products=LONG_ROW_ARMS, setting="pooled", hyper_parameters=PRIMARY_HYPER_PARAMETERS
    )
    fold_frames = fold_designs(frame=timed_rows)
    fold_jobs = _arm_jobs(
        products=DESIGN_ARMS, setting="pooled", hyper_parameters=PRIMARY_HYPER_PARAMETERS
    )
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
        "long_designs": long_designs,
        "long_losses": pl.read_parquet(long_path),
        "fold_frames": fold_frames,
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
    frame = cut_eras(
        frame=timed_rows,
        first_months=ENS_HRES_WIND_ERA_START_MONTHS,
        fold_offsets=ENS_HRES_WIND_ERA_FOLD_OFFSETS,
    )
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
                    *_previous_day_lines(evidence=checks["previous_day"]),
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
                f"{path} was fitted on a different row set, column set, seed set, feature values "
                "or hyperparameter setting than this code now produces; re-run without "
                "--report-only or --extra-fits"
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
    log: IntervalLog = []
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
    pl.DataFrame(log, schema=dict(INTERVAL_SCHEMA)).write_parquet(intervals_path)
    readme_path.write_text(_readme_text())
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
