"""Fit the exploratory extra-lead arms on the GPU, and write their losses and report once.

One-off throwaway script for the extra lead days of
<https://github.com/openclimatefix/nged-substation-forecast/issues/912>. It reads the published
shared rows and folds (`nwp_forecast_comparison.rows` on the published inputs), joins the columns
`build_forecast_inputs.py --extra-leads` wrote onto them, fits each arm at the primary setting only
(every arm here is exploratory), and writes `<domain>_losses.parquet`,
`<domain>_predictions.parquet`, and `report.md` to a new `--output-dir`. It never writes to the
published folder. It reuses a domain's saved losses rather
than refitting, and refuses to overwrite `report.md`.

The arms:

Four batches run into four folders (`--batch first` to `--batch fourth`). The second, third, and
fourth batches take one `--context-dir` for each earlier batch's folder whose arms their contrast
tables name.

The fourth batch fits six IFS HRES (9 km, Open-Meteo) arms (`FOURTH_NEW_PREFIXES`) and refits no
reference. Each arm is scored on the shared rows minus the target days whose serving run the
archive lacks (`drop_gap_rows`). Every contrast the report tabulates is computed on the rows both
arms score (`intersection_contrast_line`), with no refit of the other arm, whose fold training sets
therefore contained those gap days.

The third batch fits eight native GFS arms (`THIRD_NEW_PREFIXES`) and refits
no reference, because every arm it is compared with is already a GPU fit in an earlier batch; its
contrasts are against the ENS mean at the same day and against Open-Meteo's GFS at days 1, 2, 3, 5,
and 7. The second batch's arms are `SECOND_NEW_PREFIXES` (ENS mean at day 7, ENS control member at
days 2, 3, 5, 7, 10 and 14, GEFS mean at day 7) and `SECOND_REFERENCE_PREFIXES` (GPU refits of
the published arms the first batch left on the CPU), so that every mark on the leaderboard is a
GPU fit. The first batch's arms:

- **New arms:** ENS mean at days 5, 10 and 14; GEFS mean at days 0, 5, 10 and 14; IFS 0.25° and GFS
  at days 0, 5 and 7; ICON global at days 0 and 5; every other Previous Runs product at day 0
  (Open-Meteo's freshest run covering each hour); and the ENS control member at day 0, whose
  columns the published inputs already hold. ARPEGE Europe and AROME France are solar only.
- **References:** every published arm the new arms are compared with, refitted here on the same
  device, because a GPU fit is not bit-identical to a CPU fit and a contrast must not mix them.
  Each reference's difference from its published CPU fit is the device noise floor.

`--check` fits the batch's first new arm at one site twice, on the GPU, and stops unless the two
runs produce the same fingerprint. Run it before the full fit.

Every output carries only the anonymised `site` label.

Run it with `uv run python studies/nwp_forecast_comparison/fit_extra_leads.py --published-dir
PUBLISHED --output-dir DIR`, after `build_forecast_inputs.py --extra-leads --output-dir DIR` has
written the extra inputs there.
"""

import argparse
import concurrent.futures
import logging
import sys
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Final, NamedTuple

import polars as pl
from build_forecast_inputs import (
    GFS_NATIVE_DAYS,
    IFS_SINGLE_DAYS,
    PRODUCT_SLUGS,
    SOLAR_ONLY_PRODUCTS,
    ExtraBatchType,
    gfs_native_arm,
    ifs_single_arm,
)
from nwp_forecast_comparison import (
    METRIC,
    PERCENTAGE_POINTS,
    SETTINGS,
    TARGET,
    DomainType,
    arm_columns,
    assert_equal_rows,
    difference,
    fingerprint,
    leaderboard,
    losses_path,
    predictions_from_losses,
    predictions_path,
    rows,
)
from studies.bootstrap import bootstrap_absolute
from studies.cross_validation import DeviceType, out_of_fold_losses

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

DOMAINS: Final[tuple[DomainType, DomainType]] = ("solar", "wind")

DEVICE: Final[DeviceType] = "cuda"
"""The XGBoost device every fit here uses."""

SETTING: Final[str] = "primary"
"""The one hyperparameter setting fitted: every arm here is exploratory."""

NEW_PREFIXES: Final[tuple[str, ...]] = (
    "ens_mean_day5",
    "ens_mean_day10",
    "ens_mean_day14",
    "ens_control_day0",
    "gefs_mean_day0",
    "gefs_mean_day5",
    "gefs_mean_day10",
    "gefs_mean_day14",
    "ukv_day0",
    "ifs025_day0",
    "ifs025_day5",
    "ifs025_day7",
    "gfs_day0",
    "gfs_day5",
    "gfs_day7",
    "icon_global_day0",
    "icon_global_day5",
    "icon_d2_day0",
    "icon_eu_day0",
    "arpege_day0",
    "arome_day0",
    "knmi_harmonie_day0",
    "dmi_harmonie_day0",
)
"""The arms not fitted in the published run: `build_forecast_inputs.py --extra-leads` builds their
columns, except `ens_control_day0`, whose columns the published inputs already hold."""

REFERENCE_PREFIXES: Final[tuple[str, ...]] = (
    "ens_mean_day0",
    "ens_mean_day1",
    "ens_mean_day3",
    "ens_control_day1",
    "gefs_mean_day1",
    "gefs_mean_day3",
    "ifs025_day1",
    "ifs025_day3",
    "gfs_day1",
    "gfs_day3",
    "ukv_day1",
    "arpege_day1",
    "arome_day1",
    "knmi_harmonie_day1",
    "dmi_harmonie_day1",
    "icon_global_day1",
    "icon_global_day3",
    "icon_eu_day1",
    "icon_d2_day1",
)
"""The published arms refitted on the same device, as the new arms' references."""

SAME_PRODUCT_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("ens_mean_day5", "ens_mean_day3"),
    ("ens_mean_day10", "ens_mean_day3"),
    ("ens_mean_day14", "ens_mean_day3"),
    ("gefs_mean_day5", "gefs_mean_day3"),
    ("gefs_mean_day10", "gefs_mean_day3"),
    ("gefs_mean_day14", "gefs_mean_day3"),
    ("ifs025_day5", "ifs025_day3"),
    ("ifs025_day7", "ifs025_day3"),
    ("gfs_day5", "gfs_day3"),
    ("gfs_day7", "gfs_day3"),
    ("icon_global_day5", "icon_global_day3"),
    ("icon_d2_day0", "icon_d2_day1"),
    ("icon_eu_day0", "icon_eu_day1"),
    ("ens_control_day0", "ens_control_day1"),
    ("gefs_mean_day0", "gefs_mean_day1"),
    ("ukv_day0", "ukv_day1"),
    ("ifs025_day0", "ifs025_day1"),
    ("gfs_day0", "gfs_day1"),
    ("icon_global_day0", "icon_global_day1"),
    ("arpege_day0", "arpege_day1"),
    ("arome_day0", "arome_day1"),
    ("knmi_harmonie_day0", "knmi_harmonie_day1"),
    ("dmi_harmonie_day0", "dmi_harmonie_day1"),
)
"""Each new arm against the same product at the nearest lead already fitted (day 3 for days 5 to
14, day 1 for day 0), as (treatment, reference)."""

ENSEMBLE_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("gefs_mean_day5", "ens_mean_day5"),
    ("gefs_mean_day10", "ens_mean_day10"),
    ("gefs_mean_day14", "ens_mean_day14"),
    ("ifs025_day5", "ens_mean_day5"),
    ("gefs_mean_day0", "ens_mean_day0"),
    ("ens_control_day0", "ens_mean_day0"),
    ("ukv_day0", "ens_mean_day0"),
    ("ifs025_day0", "ens_mean_day0"),
    ("gfs_day0", "ens_mean_day0"),
    ("icon_global_day0", "ens_mean_day0"),
    ("arpege_day0", "ens_mean_day0"),
    ("arome_day0", "ens_mean_day0"),
    ("knmi_harmonie_day0", "ens_mean_day0"),
    ("dmi_harmonie_day0", "ens_mean_day0"),
)
"""The other products against ENS at the same day. GEFS and the ENS control member have ENS's exact
lead. A Previous Runs product's day-0 lead follows its own run cycle, so its contrast with ENS mean
at day 0 mixes weather models and leads. IFS 0.25° day 5 does not have ENS's lead either."""

NEAR_ANALYSIS_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("icon_d2_day0", "icon_eu_day0"),
    ("icon_d2_day1", "icon_eu_day1"),
)
"""ICON-D2 against ICON-EU at day 0 and day 1, also split by the hour of day modulo 3."""

ELSEWHERE_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("ens_mean_day10", "ens_mean_day5"),
    ("ens_mean_day14", "ens_mean_day5"),
    ("ens_mean_day14", "ens_mean_day10"),
    ("gefs_mean_day14", "gefs_mean_day10"),
    ("icon_d2_day0", "ens_mean_day0"),
    ("ens_mean_day0", "icon_eu_day0"),
)
"""Contrasts between two arms fitted here, as (treatment, reference): ENS's error rise from day 5 to
day 10 and day 14 and from day 10 to day 14, GEFS's fall from day 10 to day 14, ICON-D2 at day 0
against ENS at day 0, and ENS at day 0 against ICON-EU at day 0."""

CLIMATOLOGY_CONTRASTS: Final[tuple[str, ...]] = (
    "ens_mean_day10",
    "ens_mean_day14",
    "gefs_mean_day10",
    "gefs_mean_day14",
)
"""The arms compared with the no-weather climatology baseline, which is the published fit because
climatology involves no XGBoost model, so no device."""

HOUR_MODULO: Final[int] = 3
"""ICON-D2's and ICON-EU's runs start every 3 hours, so the freshest run's lead depends on the hour
of day modulo 3."""

MAX_MISSING_SHARE: Final[float] = 0.015
"""The largest share of rows with a missing weather value the report accepts for a new arm; the
published exploratory arms reach at most 1.48%."""


SECOND_NEW_PREFIXES: Final[tuple[str, ...]] = (
    "ens_mean_day7",
    "ens_control_day2",
    "ens_control_day3",
    "ens_control_day5",
    "ens_control_day7",
    "ens_control_day10",
    "ens_control_day14",
    "gefs_mean_day7",
)
"""The second batch's arms with no fit yet. `ens_control_day2` and `ens_control_day3` take their
columns from the published inputs; the rest take theirs from the second batch's build. The ENS
control member's day 0 and day 1 arms are not here: the first batch fits them (`ens_control_day0`,
`ens_control_day1`)."""

SECOND_REFERENCE_PREFIXES: Final[tuple[str, ...]] = (
    "ens_mean_day2",
    "gefs_mean_day2",
    "ifs025_day2",
    "gfs_day2",
    "icon_global_day2",
    "icon_eu_day2",
    "icon_eu_day3",
    "arpege_day2",
    "arpege_day3",
)
"""The published arms that the leaderboard draws and that the first batch did not refit on the GPU,
so that no mark on the leaderboard mixes devices. Climatology and the persistence baselines are not
XGBoost fits and have no device."""

SECOND_SAME_PRODUCT_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("ens_mean_day7", "ens_mean_day5"),
    ("ens_mean_day10", "ens_mean_day7"),
    ("gefs_mean_day7", "gefs_mean_day5"),
    ("gefs_mean_day10", "gefs_mean_day7"),
    ("ens_control_day3", "ens_control_day2"),
    ("ens_control_day5", "ens_control_day3"),
    ("ens_control_day7", "ens_control_day5"),
    ("ens_control_day10", "ens_control_day7"),
    ("ens_control_day14", "ens_control_day10"),
)
"""Each arm against the same product at the next shorter lead fitted, as (treatment, reference). The
first four take an arm of the first batch, read through `--first-batch-dir`."""

SECOND_ENSEMBLE_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("gefs_mean_day7", "ens_mean_day7"),
    ("ens_control_day2", "ens_mean_day2"),
    ("ens_control_day3", "ens_mean_day3"),
    ("ens_control_day5", "ens_mean_day5"),
    ("ens_control_day7", "ens_mean_day7"),
    ("ens_control_day10", "ens_mean_day10"),
    ("ens_control_day14", "ens_mean_day14"),
)
"""GEFS mean and the ENS control member against the ENS mean at the same day, which share ENS's
lead."""

SECOND_CLIMATOLOGY_CONTRASTS: Final[tuple[str, ...]] = (
    "ens_mean_day7",
    "gefs_mean_day7",
    "ens_control_day10",
    "ens_control_day14",
)
"""The second batch's arms compared with the published no-weather climatology baseline."""


THIRD_NEW_PREFIXES: Final[tuple[str, ...]] = tuple(
    gfs_native_arm(day=day) for day in GFS_NATIVE_DAYS
)
"""The third batch's arms: NOAA GFS read from Dynamical.org's native store at days 0, 1, 2, 3, 5, 7,
10 and 14, whose columns `build_forecast_inputs.py --batch third` builds."""

THIRD_REFERENCE_PREFIXES: Final[tuple[str, ...]] = ()
"""The third batch refits no reference: the arms it is compared with are GPU fits of the first two
batches, read through `--context-dir`."""

THIRD_SAME_PRODUCT_CONTRASTS: Final[tuple[tuple[str, str], ...]] = tuple(
    (gfs_native_arm(day=day), gfs_native_arm(day=earlier))
    for earlier, day in pairwise(GFS_NATIVE_DAYS)
)
"""Each native GFS arm against the native GFS arm at the next shorter lead fitted, as (treatment,
reference)."""

THIRD_ENSEMBLE_CONTRASTS: Final[tuple[tuple[str, str], ...]] = tuple(
    (gfs_native_arm(day=day), f"ens_mean_day{day}") for day in GFS_NATIVE_DAYS
)
"""Each native GFS arm against the ENS mean at the same day, from the first two batches."""

OPEN_METEO_GFS_DAYS: Final[tuple[int, ...]] = (1, 2, 3, 5, 7)
"""The days at which Open-Meteo's GFS-SEAMLESS arm (`gfs_day<N>`) is already fitted on the GPU (day
0 is a freshest-run nowcast whose lead differs, so it is left out)."""

THIRD_OPEN_METEO_GFS_CONTRASTS: Final[tuple[tuple[str, str], ...]] = tuple(
    (gfs_native_arm(day=day), f"gfs_day{day}") for day in OPEN_METEO_GFS_DAYS
)
"""Each native GFS arm against Open-Meteo's GFS-SEAMLESS arm at the same day. The two differ in
source (Dynamical.org's store against Open-Meteo's archive) and, at each day of at least 1, in
served lead: Open-Meteo serves the freshest run at least that many days old, which on most hours
is a shorter lead than the 00 UTC run N days before that the native GFS reads."""

THIRD_CLIMATOLOGY_CONTRASTS: Final[tuple[str, ...]] = (
    gfs_native_arm(day=10),
    gfs_native_arm(day=14),
)
"""The long-lead native GFS arms compared with the published no-weather climatology baseline."""

FOURTH_NEW_PREFIXES: Final[tuple[str, ...]] = tuple(
    ifs_single_arm(day=day) for day in IFS_SINGLE_DAYS
)
"""The fourth batch's arms: ECMWF IFS HRES read from Open-Meteo's Single Runs archive at days 0, 1,
2, 3, 5 and 7, whose columns `build_forecast_inputs.py --batch fourth` builds."""

FOURTH_REFERENCE_PREFIXES: Final[tuple[str, ...]] = ()
"""The fourth batch refits no reference: the arms it is compared with are GPU fits of the first
three batches, read through `--context-dir`."""

FOURTH_SAME_PRODUCT_CONTRASTS: Final[tuple[tuple[str, str], ...]] = tuple(
    (ifs_single_arm(day=day), ifs_single_arm(day=earlier))
    for earlier, day in pairwise(IFS_SINGLE_DAYS)
)
"""Each IFS HRES (9 km, Open-Meteo) arm against the same product's arm at the next shorter lead
fitted, as (treatment, reference)."""

FOURTH_ENSEMBLE_CONTRASTS: Final[tuple[tuple[str, str], ...]] = tuple(
    (ifs_single_arm(day=day), f"ens_mean_day{day}") for day in IFS_SINGLE_DAYS
)
"""Each IFS HRES (9 km, Open-Meteo) arm against the ENS mean at the same day, from the first two
batches."""

IFS_025_DAYS: Final[tuple[int, ...]] = (1, 2, 3, 5, 7)
"""The days at which Open-Meteo's IFS 0.25 degree Previous Runs arm (`ifs025_day<N>`) is fitted on
the GPU (day 0 is a freshest-run nowcast whose lead differs, so it is left out)."""

FOURTH_IFS_025_CONTRASTS: Final[tuple[tuple[str, str], ...]] = tuple(
    (ifs_single_arm(day=day), f"ifs025_day{day}") for day in IFS_025_DAYS
)
"""Each IFS HRES (9 km, Open-Meteo) arm against the IFS 0.25 degree Previous Runs arm at the same
day."""

ICON_EU_DAYS: Final[tuple[int, ...]] = (1, 2, 3)
"""The days at which ICON-EU's Previous Runs arm (`icon_eu_day<N>`) is fitted on the GPU and the
archive holds it (its archive ends at day 4)."""

FOURTH_ICON_EU_CONTRASTS: Final[tuple[tuple[str, str], ...]] = tuple(
    (ifs_single_arm(day=day), f"icon_eu_day{day}") for day in ICON_EU_DAYS
)
"""Each IFS HRES (9 km, Open-Meteo) arm against ICON-EU's arm at the same day, where the archive
holds it."""

ENSEMBLE_TITLE: Final[str] = (
    "Other products against ENS at the same day (Previous Runs day-0 rows mix "
    "weather models and leads; GEFS and the ENS control member share ENS's lead)"
)
"""The heading of the contrasts against ENS in the first and second batches' reports."""

THIRD_ENSEMBLE_TITLE: Final[str] = (
    "Native GFS against the ENS mean at the same day (the native GFS arm at day N reads a 00 UTC "
    "run's leads from 24 N hours, ENS's own lead; day 0 reads the freshest of four GFS runs a day)"
)
"""The heading of the third batch's contrasts against ENS."""

OPEN_METEO_GFS_TITLE: Final[str] = (
    "Native GFS against Open-Meteo's GFS at the same day (different source; Open-Meteo serves the "
    "freshest run at least N days old, a shorter lead)"
)
"""The heading of the third batch's contrasts against Open-Meteo's GFS."""

IFS_025_TITLE: Final[str] = (
    "IFS HRES (9 km, Open-Meteo) against IFS 0.25 degree Previous Runs at the same day (a coarser "
    "product; Previous Runs serves the freshest run at least N days old, a shorter lead)"
)
"""The heading of the fourth batch's contrasts against the IFS 0.25 degree arm."""

ICON_EU_TITLE: Final[str] = (
    "IFS HRES (9 km, Open-Meteo) against ICON-EU at the same day (a different weather model; "
    "ICON-EU serves the freshest run at least N days old, a shorter lead)"
)
"""The heading of the fourth batch's contrasts against ICON-EU."""

FOURTH_ENSEMBLE_TITLE: Final[str] = (
    "IFS HRES (9 km, Open-Meteo) against the ENS mean at the same day (both read a 00 UTC run's "
    "leads from 24 N hours, the same lead; day 0 is the 00 UTC run of the hour's own day)"
)
"""The heading of the fourth batch's contrasts against ENS."""

FIRST_BATCH_NOTE: Final[str] = (
    "Every day-0 arm is a nowcast, not a day-ahead forecast: a Previous Runs "
    "product's day 0 is the freshest run covering each hour, at a lead set by that "
    "product's own run cycle, and ENS's and GEFS's day 0 covers hours before the 00 "
    "UTC run is published. The day-0 served lead is measured only for ICON-D2 and "
    "ICON-EU; for every other product it is inferred from the run cycle."
)
"""The paragraph the first batch's report opens with, on what its day-0 arms are."""

THIRD_BATCH_NOTE: Final[str] = (
    "Every arm here reads NOAA GFS from Dynamical.org's native store, not Open-Meteo's "
    "GFS-SEAMLESS archive, at each generator's nearest 0.25 degree grid cell. Day N of 1 or more "
    "reads the 00 UTC run issued N days before the hour's own day, at leads 24 N + 1 to 24 N + 24 "
    "hours for solar (a solar hour is labelled by its end) and 24 N to 24 N + 23 hours for wind. "
    "Day 0 is a nowcast: it reads the freshest of the four runs a day (00, 06, 12, and 18 UTC) at "
    "a lead of 1 to 6 hours for solar and 0 to 5 hours for wind, which ignores the hours GFS takes "
    "to publish a run. GFS's radiation is a mean since the last 6-hourly reset, and its lead "
    "labels the window's end; each step's mean is recovered before use "
    "(`studies.gfs_native.step_means`). Days 0 to 4 read hourly leads directly. Days 5, 7, 10, and "
    "14 lie on 3-hourly leads and are upsampled to hourly as the ENS and GEFS arms are."
)
"""The paragraph the third batch's report opens with, on which run and lead each arm reads."""

FOURTH_BATCH_NOTE: Final[str] = (
    "Every arm here is IFS HRES (9 km, Open-Meteo): ECMWF's IFS HRES as Open-Meteo's Single Runs "
    "archive serves it (`ecmwf_ifs`, on ECMWF's O1280 grid), a finer product than the IFS 0.25° "
    "arm and not another version of it, with its own row. The archive holds one 00 UTC run a day "
    "and no other cycle, with hourly leads 0 to 240 hours, and Open-Meteo's processing of HRES "
    "has not been checked against a native archive. Each solar generator reads its nearest cell "
    "and each wind generator its nearest land cell (sites B and D share a source cell and carry "
    "identical series). Day N reads the 00 UTC run issued N days before the hour's own day, at "
    "leads 24 N + 1 to 24 N + 24 hours for solar and 24 N to 24 N + 23 hours for wind, the rule "
    "the ENS arms follow, so day 0 is the run of the hour's own day. Like ENS's day 0, it covers "
    "hours before the 00 UTC run is published, so it is not a forecast that could have been used "
    "in advance for those hours. Day 10 is "
    "absent because the runs end at lead 240 hours. Radiation is clipped at zero; wind is the "
    "served speed and the sine and cosine of the served direction, as the Open-Meteo Previous "
    "Runs arms are. IFS HRES publishes every 3 hours after lead 90 and every 6 hours after lead "
    "144, and Open-Meteo interpolates those steps to hourly, so hourly values at days 5 and 7, "
    "and at the last hours of day 3, are interpolated. The IFS model cycle changed inside the "
    "span (cycle 50r1 on 2026-05-12, from ECMWF's pages); this batch adds no era feature and "
    "uses the shared rows' `era_code`, which is cut on the target hour's month and has no "
    "boundary at that date. "
    "**Row set.** Each arm is scored on the shared rows minus the target days whose serving run "
    "the archive lacks, which are gaps and are never filled from another run. Every contrast "
    "below is computed on the rows both arms score, from the existing out-of-fold losses with no "
    "refit of the other arm, whose fold training sets contained the gap days; each prints its "
    "row and month counts and the absolute error of both arms on those rows. All are "
    "exploratory."
)
"""The paragraph the fourth batch's report opens with: what each arm reads and how it is scored."""


class ArmBatch(NamedTuple):
    """One fit batch's arms and the contrasts its report tabulates."""

    new_prefixes: tuple[str, ...]
    reference_prefixes: tuple[str, ...]
    same_product_contrasts: tuple[tuple[str, str], ...]
    ensemble_contrasts: tuple[tuple[str, str], ...]
    near_analysis_contrasts: tuple[tuple[str, str], ...]
    elsewhere_contrasts: tuple[tuple[str, str], ...]
    climatology_contrasts: tuple[str, ...]
    note: str = ""
    ensemble_title: str = ENSEMBLE_TITLE
    open_meteo_gfs_contrasts: tuple[tuple[str, str], ...] = ()
    ifs_025_contrasts: tuple[tuple[str, str], ...] = ()
    icon_eu_contrasts: tuple[tuple[str, str], ...] = ()
    drop_gap_rows: bool = False
    """Whether each arm is fitted and scored without the rows where its own weather columns are
    null, and every contrast is computed on the rows both arms score."""


BATCHES: Final[dict[ExtraBatchType, ArmBatch]] = {
    "first": ArmBatch(
        new_prefixes=NEW_PREFIXES,
        reference_prefixes=REFERENCE_PREFIXES,
        same_product_contrasts=SAME_PRODUCT_CONTRASTS,
        ensemble_contrasts=ENSEMBLE_CONTRASTS,
        near_analysis_contrasts=NEAR_ANALYSIS_CONTRASTS,
        elsewhere_contrasts=ELSEWHERE_CONTRASTS,
        climatology_contrasts=CLIMATOLOGY_CONTRASTS,
        note=FIRST_BATCH_NOTE,
    ),
    "second": ArmBatch(
        new_prefixes=SECOND_NEW_PREFIXES,
        reference_prefixes=SECOND_REFERENCE_PREFIXES,
        same_product_contrasts=SECOND_SAME_PRODUCT_CONTRASTS,
        ensemble_contrasts=SECOND_ENSEMBLE_CONTRASTS,
        near_analysis_contrasts=(),
        elsewhere_contrasts=(),
        climatology_contrasts=SECOND_CLIMATOLOGY_CONTRASTS,
    ),
    "third": ArmBatch(
        new_prefixes=THIRD_NEW_PREFIXES,
        reference_prefixes=THIRD_REFERENCE_PREFIXES,
        same_product_contrasts=THIRD_SAME_PRODUCT_CONTRASTS,
        ensemble_contrasts=THIRD_ENSEMBLE_CONTRASTS,
        near_analysis_contrasts=(),
        elsewhere_contrasts=(),
        climatology_contrasts=THIRD_CLIMATOLOGY_CONTRASTS,
        note=THIRD_BATCH_NOTE,
        ensemble_title=THIRD_ENSEMBLE_TITLE,
        open_meteo_gfs_contrasts=THIRD_OPEN_METEO_GFS_CONTRASTS,
    ),
    "fourth": ArmBatch(
        new_prefixes=FOURTH_NEW_PREFIXES,
        reference_prefixes=FOURTH_REFERENCE_PREFIXES,
        same_product_contrasts=FOURTH_SAME_PRODUCT_CONTRASTS,
        ensemble_contrasts=FOURTH_ENSEMBLE_CONTRASTS,
        near_analysis_contrasts=(),
        elsewhere_contrasts=(),
        climatology_contrasts=(),
        note=FOURTH_BATCH_NOTE,
        ensemble_title=FOURTH_ENSEMBLE_TITLE,
        ifs_025_contrasts=FOURTH_IFS_025_CONTRASTS,
        icon_eu_contrasts=FOURTH_ICON_EU_CONTRASTS,
        drop_gap_rows=True,
    ),
}
"""The four fit batches, by the name `--batch` takes."""


def batch_prefixes(*, batch: ArmBatch, domain: DomainType) -> tuple[str, ...]:
    """Return every arm a batch fits for one technology, new arms first.

    Args:
        batch: One of `BATCHES`.
        domain: `solar` or `wind`.

    Returns:
        The batch's new and reference arms, without the arms of solar-only products for wind.
    """
    return domain_prefixes(domain=domain, prefixes=(*batch.new_prefixes, *batch.reference_prefixes))


def joined_rows(*, published_dir: Path, output_dir: Path, domain: DomainType) -> pl.DataFrame:
    """Return the published shared rows with the extra-lead columns joined on.

    Args:
        published_dir: The folder holding the published inputs.
        output_dir: The folder holding `<domain>_extra_lead_inputs.parquet`.
        domain: `solar` or `wind`.

    Returns:
        `nwp_forecast_comparison.rows`'s result, with the extra columns added.

    Raises:
        ValueError: If a shared row has no row in the extra inputs, or the join changes the row
            count.
    """
    frame = rows(input_dir=published_dir, domain=domain)
    extra = pl.read_parquet(output_dir / f"{domain}_extra_lead_inputs.parquet")
    keys = ["site", "time"]
    unmatched = frame.select(keys).join(extra.select(keys), on=keys, how="anti").height
    if unmatched:
        msg = f"{domain}: {unmatched} shared rows are missing from the extra-lead inputs"
        raise ValueError(msg)
    joined = frame.join(extra, on=keys, how="left")
    if joined.height != frame.height:
        msg = f"{domain}: joining the extra columns changed {frame.height} rows to {joined.height}"
        raise ValueError(msg)
    return joined


def domain_prefixes(*, domain: DomainType, prefixes: tuple[str, ...]) -> tuple[str, ...]:
    """Drop the arms of solar-only products from a wind run.

    Args:
        domain: `solar` or `wind`.
        prefixes: Arm prefixes such as `arpege_day0`.

    Returns:
        `prefixes` unchanged for solar; for wind, without the arms of `SOLAR_ONLY_PRODUCTS`.
    """
    if domain == "solar":
        return prefixes
    solar_only = tuple(f"{PRODUCT_SLUGS[product]}_day" for product in SOLAR_ONLY_PRODUCTS)
    return tuple(prefix for prefix in prefixes if not prefix.startswith(solar_only))


def check_saved_losses_hold_arms(
    *, losses: pl.DataFrame, domain: DomainType, path: Path, batch: ArmBatch
) -> None:
    """Raise if saved losses lack any arm this run reports.

    Args:
        losses: A domain's saved per-row losses, with an `arm` column.
        domain: `solar` or `wind`.
        path: Where the losses were read from, named in the error.
        batch: The batch whose arms the losses must hold.

    Raises:
        ValueError: If an arm of `batch` for `domain` is absent.
    """
    expected = set(batch_prefixes(batch=batch, domain=domain))
    lacking = sorted(expected - set(losses["arm"].unique().to_list()))
    if lacking:
        msg = f"{path} lacks arms {lacking}; use a new --output-dir"
        raise ValueError(msg)


def missing_shares(*, frame: pl.DataFrame, domain: DomainType, batch: ArmBatch) -> pl.DataFrame:
    """Return each new arm's share of rows with any missing weather value.

    Args:
        frame: `joined_rows`'s result.
        domain: `solar` or `wind`.
        batch: The batch whose new arms are checked.

    Returns:
        One row per new arm present in `frame`, with `arm` and `share`.

    Raises:
        ValueError: If a new arm's share exceeds `MAX_MISSING_SHARE`, which would confound the
            lead with coverage.
    """
    records = []
    for prefix in domain_prefixes(domain=domain, prefixes=batch.new_prefixes):
        columns = arm_columns(domain=domain, prefixes=(prefix,))
        weather = [name for name in columns if name.startswith(f"{prefix}_")]
        if not weather or not all(name in frame.columns for name in weather):
            continue
        share = float(
            frame.select(
                pl.any_horizontal(pl.col(name).is_null() for name in weather).mean()
            ).item()
        )
        records.append({"arm": prefix, "share": share})
    result = pl.DataFrame(records, schema={"arm": pl.String, "share": pl.Float64})
    too_many = result.filter(pl.col("share") > MAX_MISSING_SHARE)
    if not too_many.is_empty():
        msg = f"{domain}: new arms with more than {MAX_MISSING_SHARE:.1%} missing: {too_many}"
        raise ValueError(msg)
    return result


def arm_rows(*, frame: pl.DataFrame, columns: Sequence[str], drop_gap_rows: bool) -> pl.DataFrame:
    """Return the rows one arm is fitted and scored on.

    Args:
        frame: `joined_rows`'s result.
        columns: The arm's feature columns.
        drop_gap_rows: Whether to drop the rows where any of `columns` is null: the target days
            whose serving run the archive lacks, which are gaps and are neither interpolated nor
            filled from another run.

    Returns:
        `frame` unchanged, or without the rows with a null in `columns`.
    """
    if not drop_gap_rows:
        return frame
    return frame.filter(pl.all_horizontal(pl.col(column).is_not_null() for column in columns))


def fit_arms(
    *,
    frame: pl.DataFrame,
    domain: DomainType,
    prefixes: tuple[str, ...],
    workers: int,
    drop_gap_rows: bool = False,
) -> pl.DataFrame:
    """Fit every arm at every site, out of fold, on the GPU, and stack the losses.

    Args:
        frame: `joined_rows`'s result.
        domain: `solar` or `wind`.
        prefixes: The arms to fit.
        workers: How many (arm, site) fits run at once.
        drop_gap_rows: Whether each arm drops its own null rows, as `arm_rows` does.

    Returns:
        Every fit's per-row losses, labelled with `arm`, `setting` and `device`.

    Raises:
        ValueError: If an arm's columns are absent from `frame`, which would otherwise drop the
            arm from the run silently.
    """
    sites = sorted(frame["site"].unique().to_list())
    outputs: list[pl.DataFrame] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for prefix in prefixes:
            columns = arm_columns(domain=domain, prefixes=(prefix,))
            absent = [name for name in columns if name not in frame.columns]
            if absent:
                msg = f"{domain}: {prefix} has no columns {absent} in the joined rows"
                raise ValueError(msg)
            for site in sites:
                future = pool.submit(
                    out_of_fold_losses,
                    site_rows=arm_rows(
                        frame=frame.filter(pl.col("site") == site),
                        columns=columns,
                        drop_gap_rows=drop_gap_rows,
                    ),
                    features=list(columns),
                    target=TARGET,
                    hyper_parameters=SETTINGS[SETTING],
                    with_quantiles=False,
                    device=DEVICE,
                )
                futures[future] = (prefix, site)
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            prefix, site = futures[future]
            outputs.append(
                future.result().with_columns(
                    arm=pl.lit(prefix), setting=pl.lit(SETTING), device=pl.lit(DEVICE)
                )
            )
            _LOG.info("%s: %d/%d done: %s / site %s", domain, done, len(futures), prefix, site)
    return pl.concat(outputs)


def check_determinism(
    *, published_dir: Path, output_dir: Path, prefix: str, drop_gap_rows: bool = False
) -> bool:
    """Fit one arm at one site twice on the GPU and compare the two runs' fingerprints.

    Args:
        published_dir: The folder holding the published inputs.
        output_dir: The folder holding the extra-lead inputs.
        prefix: The arm to fit, which must have wind columns in the joined rows.
        drop_gap_rows: Whether the arm drops its own null rows, as `arm_rows` does.

    Returns:
        Whether the two fingerprints agree.
    """
    frame = joined_rows(published_dir=published_dir, output_dir=output_dir, domain="wind")
    site = min(frame["site"].unique().to_list())
    fingerprints = []
    for _ in range(2):
        losses = fit_arms(
            frame=frame.filter(pl.col("site") == site),
            domain="wind",
            prefixes=(prefix,),
            workers=1,
            drop_gap_rows=drop_gap_rows,
        )
        fingerprints.append(fingerprint(frame=losses))
    _LOG.info("two GPU runs of one arm at one site: %s", fingerprints)
    return fingerprints[0] == fingerprints[1]


def interval_text(*, point: float, lower: float, upper: float) -> str:
    """Format a difference or an error as `point [lower, upper]`, in percentage points."""
    scaled = [value * PERCENTAGE_POINTS for value in (point, lower, upper)]
    return f"{scaled[0]:+.3f} [{scaled[1]:+.3f}, {scaled[2]:+.3f}]"


def error_text(*, value: float, lower: float, upper: float) -> str:
    """Format an absolute error as `value [lower, upper]`, in percent of capacity."""
    scaled = [x * PERCENTAGE_POINTS for x in (value, lower, upper)]
    return f"{scaled[0]:.3f} [{scaled[1]:.3f}, {scaled[2]:.3f}]"


def contrast_line(*, losses: pl.DataFrame, treatment: str, reference: str) -> str | None:
    """Format one paired contrast as a Markdown table row, or None if an arm is absent.

    Args:
        losses: Per-row losses at one setting.
        treatment: The arm whose error is compared.
        reference: The arm it is compared against.

    Returns:
        `| treatment − reference | difference [interval] | rows | months |`.
    """
    present = set(losses["arm"].unique().to_list())
    if treatment not in present or reference not in present:
        return None
    result = difference(losses=losses, treatment=treatment, reference=reference)
    text = interval_text(
        point=result["difference"], lower=result["lower_95"], upper=result["upper_95"]
    )
    return f"| {treatment} − {reference} | {text} | {result['n_rows']} | {result['n_months']} |"


INTERSECTION_HEADER: Final[tuple[str, str, str]] = (
    "",
    (
        "| Contrast, exploratory, on the rows both arms score (points) | Difference [95% interval] "
        "| Error of first arm (%) | Error of second arm (%) | Rows | Months |"
    ),
    "|---|---|---|---|---|---|",
)
"""The header of a contrast table computed on the rows two arms share."""


def shared_rows(*, losses: pl.DataFrame, treatment: str, reference: str) -> pl.DataFrame:
    """Restrict two arms' losses to the (site, time, seed) rows both hold.

    Args:
        losses: Per-row losses at one setting, carrying both arms.
        treatment: One arm's name.
        reference: The other arm's name.

    Returns:
        The two arms' rows whose (site, time, seed) is in both arms. `assert_equal_rows` holds on
        the result, so `difference` accepts it.
    """
    keys = ["site", "time", "seed"]
    both = pl.concat(
        [losses.filter(pl.col("arm") == arm).select(keys) for arm in (treatment, reference)]
    )
    shared = both.group_by(keys).len().filter(pl.col("len") == 2).select(keys)
    return losses.filter(pl.col("arm").is_in([treatment, reference])).join(
        shared, on=keys, how="semi"
    )


def intersection_contrast_line(
    *, losses: pl.DataFrame, treatment: str, reference: str
) -> str | None:
    """Format one paired contrast on the rows both arms score, or None if an arm is absent.

    Args:
        losses: Per-row losses at one setting.
        treatment: The arm whose error is compared.
        reference: The arm it is compared against.

    Returns:
        `| treatment − reference | difference [interval] | error | error | rows | months |`, where
        both errors are absolute errors on the shared rows, in percent of capacity.
    """
    present = set(losses["arm"].unique().to_list())
    if treatment not in present or reference not in present:
        return None
    shared = shared_rows(losses=losses, treatment=treatment, reference=reference)
    result = difference(losses=shared, treatment=treatment, reference=reference)
    text = interval_text(
        point=result["difference"], lower=result["lower_95"], upper=result["upper_95"]
    )
    errors = [
        bootstrap_absolute(losses=shared, arm=arm, metric=METRIC)["value"] * PERCENTAGE_POINTS
        for arm in (treatment, reference)
    ]
    return (
        f"| {treatment} − {reference} | {text} | {errors[0]:.3f} | {errors[1]:.3f} "
        f"| {result['n_rows']} | {result['n_months']} |"
    )


ROW_SET_REFERENCE_ARM: Final[str] = "ens_mean_day1"
"""The arm the row-set diagnostic scores on all shared rows and on the rows without a gap."""


def row_set_diagnostic(*, losses: pl.DataFrame, gap_arm: str) -> str | None:
    """Format how far dropping an arm's gap rows moves a reference arm's absolute error.

    The leaderboard mark of an arm scored without its gap rows averages over different hours from
    every other mark. This line measures that alone, on an arm that has no gap: the reference arm
    scored on every shared row and on the rows `gap_arm` also holds.

    Args:
        losses: Per-row losses at one setting, carrying `ROW_SET_REFERENCE_ARM` and `gap_arm`.
        gap_arm: An arm scored without its gap rows.

    Returns:
        `| reference | error on all rows | error without the gap rows | difference | rows | rows |`,
        in percent of capacity and percentage points, or None if either arm is absent.
    """
    present = set(losses["arm"].unique().to_list())
    if ROW_SET_REFERENCE_ARM not in present or gap_arm not in present:
        return None
    every_row = bootstrap_absolute(losses=losses, arm=ROW_SET_REFERENCE_ARM, metric=METRIC)
    kept = shared_rows(losses=losses, treatment=ROW_SET_REFERENCE_ARM, reference=gap_arm)
    without_gap = bootstrap_absolute(losses=kept, arm=ROW_SET_REFERENCE_ARM, metric=METRIC)
    moved = (without_gap["value"] - every_row["value"]) * PERCENTAGE_POINTS
    return (
        f"| {ROW_SET_REFERENCE_ARM} | {every_row['value'] * PERCENTAGE_POINTS:.3f} "
        f"| {without_gap['value'] * PERCENTAGE_POINTS:.3f} | {moved:+.3f} "
        f"| {every_row['n_rows']} | {without_gap['n_rows']} |"
    )


def row_set_diagnostic_lines(*, losses: pl.DataFrame, gap_arm: str) -> list[str]:
    """Write the row-set diagnostic's report section, or nothing if an arm is absent."""
    diagnostic = row_set_diagnostic(losses=losses, gap_arm=gap_arm)
    if diagnostic is None:
        return []
    return [
        "",
        (
            f"### Row-set diagnostic: {ROW_SET_REFERENCE_ARM} on all shared rows and without the "
            f"gap days of {gap_arm}"
        ),
        "",
        (
            "| Arm | Error on all rows (%) | Error without the gap rows (%) "
            "| Change (points) | Rows, all | Rows, without the gap rows |"
        ),
        "|---|---|---|---|---|---|",
        diagnostic,
    ]


def served_lead(*, domain: DomainType, remainder: int) -> int:
    """Return a 3-hourly model's served lead in hours where `hour % 3 == remainder`.

    Radiation is a mean over the hour before its label, so its served lead is `((h - 1) % 3) + 1`;
    wind is an instantaneous value at its label, so its served lead is `h % 3`.

    Args:
        domain: `solar` or `wind`.
        remainder: The hour of day modulo `HOUR_MODULO`.

    Returns:
        The served lead in hours.
    """
    return ((remainder - 1) % HOUR_MODULO) + 1 if domain == "solar" else remainder


def by_hour_modulo(
    *, domain: DomainType, losses: pl.DataFrame, treatment: str, reference: str
) -> list[str]:
    """Format one contrast on each class of the hour of day modulo `HOUR_MODULO`.

    Args:
        domain: `solar` or `wind`, which decides each class's served lead.
        losses: Per-row losses at one setting.
        treatment: The arm whose error is compared.
        reference: The arm it is compared against.

    Returns:
        One table row per class that holds both arms, labelled with the class and its served lead.
    """
    lines = []
    for remainder in range(HOUR_MODULO):
        subset = losses.filter(pl.col("time").dt.hour() % HOUR_MODULO == remainder)
        line = contrast_line(losses=subset, treatment=treatment, reference=reference)
        if line is not None:
            lead = served_lead(domain=domain, remainder=remainder)
            label = f"(hour mod {HOUR_MODULO} = {remainder}, lead {lead} h)"
            lines.append(line.replace("| ", f"| {label} ", 1))
    return lines


def noise_floor_lines(
    *,
    domain: DomainType,
    losses: pl.DataFrame,
    published: pl.DataFrame,
    batch: ArmBatch,
    header: list[str],
) -> list[str]:
    """Write the device noise floor: each refitted arm's GPU fit minus its published CPU fit.

    Args:
        domain: `solar` or `wind`.
        losses: The new fits' per-row losses at the primary setting.
        published: The published CPU fits' per-row losses at the primary setting.
        batch: The batch whose reference arms are compared.
        header: The contrast table's header lines.

    Returns:
        The section's Markdown lines, or none for a batch that refits no reference.
    """
    if not batch.reference_prefixes:
        return []
    noise = ["", "### Device noise floor: GPU fit minus published CPU fit, same arm", *header]
    for prefix in domain_prefixes(domain=domain, prefixes=batch.reference_prefixes):
        both = pl.concat(
            [
                losses.filter(pl.col("arm") == prefix).with_columns(arm=pl.lit("gpu")),
                published.filter(pl.col("arm") == prefix).with_columns(arm=pl.lit("cpu")),
            ],
            how="diagonal",
        )
        if (
            both.filter(pl.col("arm") == "cpu").is_empty()
            or both.filter(pl.col("arm") == "gpu").is_empty()
        ):
            continue
        assert_equal_rows(losses=both, treatment="gpu", reference="cpu")
        line = contrast_line(losses=both, treatment="gpu", reference="cpu")
        if line:
            noise.append(line.replace("| gpu − cpu", f"| {prefix} (GPU − CPU)", 1))
    return noise


def report_domain(
    *,
    domain: DomainType,
    losses: pl.DataFrame,
    published: pl.DataFrame,
    shares: pl.DataFrame,
    batch: ArmBatch,
    context: Sequence[pl.DataFrame] = (),
) -> list[str]:
    """Write one technology's report section.

    Args:
        domain: `solar` or `wind`.
        losses: The new fits' per-row losses at the primary setting.
        published: The published CPU fits' per-row losses at the primary setting.
        shares: `missing_shares`'s result.
        batch: The batch whose contrasts the report tabulates.
        context: Earlier batches' GPU losses at the primary setting, one frame per batch. The
            contrast tables read the arms of `losses` and `context` together, so a contrast can
            name an arm an earlier batch fitted.

    Returns:
        The section's Markdown lines.
    """
    arms = list(batch_prefixes(batch=batch, domain=domain))
    board = leaderboard(losses=losses, arms=arms)
    pooled = pl.concat(
        [losses, *(frame.drop("device", strict=False) for frame in context)],
        how="diagonal_relaxed",
    )
    lines = [
        f"## {domain.capitalize()}",
        "",
        "### Absolute error of every arm fitted here (GPU, primary setting)",
        "",
        "| Arm | Error (% of capacity) | 95% interval | Rows | Months |",
        "|---|---|---|---|---|",
    ]
    for row in board.iter_rows(named=True):
        text = error_text(value=row["value"], lower=row["lower_95"], upper=row["upper_95"])
        lines.append(
            f"| {row['arm']} | {text.split(' [')[0]} | [{text.split(' [')[1]} "
            f"| {row['n_rows']} | {row['n_months']} |"
        )
    lines += [
        "",
        "### Share of rows with a missing weather value, new arms",
        "",
        "| Arm | Share |",
        "|---|---|",
        *(f"| {row['arm']} | {row['share']:.4%} |" for row in shares.iter_rows(named=True)),
    ]
    header = [
        "",
        "| Contrast (points) | Difference [95% interval] | Rows | Months |",
        "|---|---|---|---|",
    ]
    if batch.drop_gap_rows:
        header = list(INTERSECTION_HEADER)
    line_of = intersection_contrast_line if batch.drop_gap_rows else contrast_line
    for title, pairs in (
        (
            "Change with lead: each arm minus the same product at the lead named",
            batch.same_product_contrasts,
        ),
        (batch.ensemble_title, batch.ensemble_contrasts),
        *(
            ((OPEN_METEO_GFS_TITLE, batch.open_meteo_gfs_contrasts),)
            if batch.open_meteo_gfs_contrasts
            else ()
        ),
        *(((IFS_025_TITLE, batch.ifs_025_contrasts),) if batch.ifs_025_contrasts else ()),
        *(((ICON_EU_TITLE, batch.icon_eu_contrasts),) if batch.icon_eu_contrasts else ()),
    ):
        table = [line_of(losses=pooled, treatment=t, reference=r) for t, r in pairs]
        lines += ["", f"### {title}", *header, *(line for line in table if line)]
    near = [*header]
    for treatment, reference in batch.near_analysis_contrasts:
        line = contrast_line(losses=pooled, treatment=treatment, reference=reference)
        if line:
            near.append(line)
            near.extend(
                by_hour_modulo(
                    domain=domain, losses=pooled, treatment=treatment, reference=reference
                )
            )
    if batch.near_analysis_contrasts:
        lines += ["", "### ICON-D2 against ICON-EU, whole and by hour of day modulo 3", *near]
    climatology = published.filter(pl.col("arm") == "climatology")
    with_climatology = pl.concat(
        [pooled.drop("device", strict=False), climatology], how="vertical_relaxed"
    )
    elsewhere = [
        contrast_line(losses=pooled, treatment=t, reference=r) for t, r in batch.elsewhere_contrasts
    ]
    elsewhere += [
        contrast_line(losses=with_climatology, treatment=arm, reference="climatology")
        for arm in batch.climatology_contrasts
    ]
    if batch.drop_gap_rows:
        lines += row_set_diagnostic_lines(losses=pooled, gap_arm=batch.new_prefixes[1])
    if any(elsewhere):
        lines += [
            "",
            "### Long leads against climatology, and other contrasts between arms fitted here",
            *header,
            *(line for line in elsewhere if line),
        ]
    icon_arms = ("icon_d2_day0", "icon_eu_day0") if batch.near_analysis_contrasts else ()
    if icon_arms:
        lines += [
            "",
            "### Absolute error by hour of day modulo 3, ICON-D2 and ICON-EU at day 0",
            "",
            (
                "| Arm | Hour of day mod 3 (served lead) | Error (% of capacity) "
                "| 95% interval | Rows |"
            ),
            "|---|---|---|---|---|",
        ]
    for arm in icon_arms:
        for remainder in range(HOUR_MODULO):
            subset = losses.filter(pl.col("time").dt.hour() % HOUR_MODULO == remainder)
            if arm not in set(subset["arm"].unique().to_list()):
                continue
            lead = served_lead(domain=domain, remainder=remainder)
            result = bootstrap_absolute(losses=subset, arm=arm, metric=METRIC)
            text = error_text(
                value=result["value"], lower=result["lower_95"], upper=result["upper_95"]
            )
            lines.append(
                f"| {arm} | {remainder} ({lead} h) | {text.split(' [')[0]} "
                f"| [{text.split(' [')[1]} | {result['n_rows']} |"
            )
    noise = noise_floor_lines(
        domain=domain, losses=losses, published=published, batch=batch, header=header
    )
    return [*lines, *noise, ""]


def require_arms(*, frame: pl.DataFrame, domain: DomainType, batch: ArmBatch) -> None:
    """Raise unless every arm to fit has all its columns in `frame`.

    Args:
        frame: `joined_rows`'s result.
        domain: `solar` or `wind`.
        batch: The batch whose arms are fitted.

    Raises:
        ValueError: If any arm is missing columns, which a GEFS gate that returned the keys
            unchanged would cause silently.
    """
    absent = [
        prefix
        for prefix in batch_prefixes(batch=batch, domain=domain)
        if not all(name in frame.columns for name in arm_columns(domain=domain, prefixes=(prefix,)))
    ]
    if absent:
        msg = f"{domain}: arms with no columns in the joined rows: {absent}"
        raise ValueError(msg)


def contrast_arms(*, batch: ArmBatch, domain: DomainType) -> set[str]:
    """Return every arm a batch's contrast tables name for one technology.

    Args:
        batch: One of `BATCHES`.
        domain: `solar` or `wind`.

    Returns:
        The arms of every contrast pair and every climatology contrast, without solar-only
        products for wind.
    """
    pairs = (
        *batch.same_product_contrasts,
        *batch.ensemble_contrasts,
        *batch.open_meteo_gfs_contrasts,
        *batch.ifs_025_contrasts,
        *batch.icon_eu_contrasts,
        *batch.near_analysis_contrasts,
        *batch.elsewhere_contrasts,
    )
    named = {arm for pair in pairs for arm in pair} | set(batch.climatology_contrasts)
    return set(domain_prefixes(domain=domain, prefixes=tuple(sorted(named))))


def check_context_arms(
    *, domain: DomainType, context_arms: Sequence[set[str]], batch: ArmBatch
) -> None:
    """Raise unless the batch's own arms and the earlier batches' cover every contrast arm once.

    Args:
        domain: `solar` or `wind`.
        context_arms: The arms each `--context-dir` folder's saved losses hold, one set per folder.
        batch: The batch being fitted.

    Raises:
        ValueError: If an arm is fitted by two batches, which would double its rows in the pooled
            losses, or a contrast names an arm no batch fits, which would drop its row silently.
    """
    seen = set(batch_prefixes(batch=batch, domain=domain))
    for arms in context_arms:
        twice = sorted(arms & seen)
        if twice:
            msg = f"{domain}: arms fitted twice across the batches: {twice}"
            raise ValueError(msg)
        seen |= arms
    lacking = sorted(contrast_arms(batch=batch, domain=domain) - seen)
    if lacking:
        msg = f"{domain}: contrast arms fitted by no batch: {lacking}"
        raise ValueError(msg)


def check_context_dirs(*, args: argparse.Namespace, batch: ArmBatch) -> None:
    """Raise unless `--context-dir` names finished earlier batches that complete the contrasts.

    Args:
        args: The parsed command line.
        batch: The chosen batch; the first batch reads no context.

    Raises:
        ValueError: If a later batch has no `--context-dir`, one names the output or published
            folder, or the arms do not fit together (see `check_context_arms`).
        FileNotFoundError: If an earlier batch has not written its losses.
    """
    if args.batch == "first":
        return
    if not args.context_dir:
        msg = f"--batch {args.batch} needs one --context-dir per earlier batch"
        raise ValueError(msg)
    forbidden = {args.output_dir.resolve(), args.published_dir.resolve()}
    if any(directory.resolve() in forbidden for directory in args.context_dir):
        msg = "--context-dir must name an earlier batch's own folder"
        raise ValueError(msg)
    for domain in DOMAINS:
        context_arms = []
        for directory in args.context_dir:
            path = losses_path(output_dir=directory, domain=domain)
            if not path.exists():
                msg = f"{path} does not exist; let that batch finish first"
                raise FileNotFoundError(msg)
            arms = (
                pl.scan_parquet(path)
                .filter(pl.col("setting") == SETTING)
                .select("arm")
                .unique()
                .collect()["arm"]
                .to_list()
            )
            context_arms.append(set(arms))
        check_context_arms(domain=domain, context_arms=context_arms, batch=batch)


def main() -> int:
    """Fit the arms for both technologies and write the losses and report once."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--published-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2, help="(arm, site) fits run at once.")
    parser.add_argument(
        "--batch",
        choices=tuple(BATCHES),
        default="first",
        help="Which fit batch: the first (the day-0 to day-14 arms), the second (ENS mean at "
        "day 7, the ENS control member, GEFS mean at day 7, and GPU refits of the arms the first "
        "batch left on the CPU), the third (native GFS at days 0, 1, 2, 3, 5, 7, 10, and 14), or "
        "the fourth (IFS HRES (9 km, Open-Meteo) at days 0, 1, 2, 3, 5, and 7).",
    )
    parser.add_argument(
        "--context-dir",
        type=Path,
        action="append",
        default=[],
        help="With --batch second, third, or fourth: an earlier batch's folder, whose losses the "
        "contrast tables read for arms only that batch fitted. Repeat it for each earlier batch "
        "the contrasts name (the third batch needs the first and the second, and so does the "
        "fourth).",
    )
    parser.add_argument("--check", action="store_true", help="Compare two GPU runs of one arm.")
    parser.add_argument(
        "--report-only", action="store_true", help="Write the report from the saved losses."
    )
    args = parser.parse_args()
    batch = BATCHES[args.batch]
    if args.output_dir.resolve() == args.published_dir.resolve():
        msg = "the output folder must not be the published folder"
        raise ValueError(msg)
    check_context_dirs(args=args, batch=batch)
    if args.check:
        for domain in DOMAINS:
            checked = joined_rows(
                published_dir=args.published_dir, output_dir=args.output_dir, domain=domain
            )
            require_arms(frame=checked, domain=domain, batch=batch)
            missing_shares(frame=checked, domain=domain, batch=batch)
        agree = check_determinism(
            published_dir=args.published_dir,
            output_dir=args.output_dir,
            prefix=domain_prefixes(domain="wind", prefixes=batch.new_prefixes)[0],
            drop_gap_rows=batch.drop_gap_rows,
        )
        sys.stdout.write(f"two GPU runs agree: {agree}\n")
        return 0 if agree else 1
    report_path = args.output_dir / "report.md"
    if report_path.exists():
        msg = f"{report_path} exists; the extra-lead report is write-once, move it first"
        raise FileExistsError(msg)
    frames = {
        domain: joined_rows(
            published_dir=args.published_dir, output_dir=args.output_dir, domain=domain
        )
        for domain in DOMAINS
    }
    shares = {
        domain: missing_shares(frame=frames[domain], domain=domain, batch=batch)
        for domain in DOMAINS
    }
    for domain in DOMAINS:
        require_arms(frame=frames[domain], domain=domain, batch=batch)
    report = [
        f"# Extra lead days, GPU fits, {args.batch} batch: report",
        "",
        (
            "Every arm is exploratory and fitted at the primary setting only, on the published "
            "shared rows and folds. Differences are first arm minus second, in percentage points "
            "of capacity; about 1 in 20 exploratory intervals reaches significance at the 5% level "
            "by chance."
        ),
        "",
    ]
    if batch.note:
        report += [batch.note, ""]
    for domain in DOMAINS:
        path = losses_path(output_dir=args.output_dir, domain=domain)
        if path.exists():
            _LOG.info("%s exists; reporting from the saved losses", path)
            losses = pl.read_parquet(path)
            check_saved_losses_hold_arms(losses=losses, domain=domain, path=path, batch=batch)
        elif args.report_only:
            msg = f"{path} does not exist; --report-only needs both domains' losses"
            raise FileNotFoundError(msg)
        else:
            losses = fit_arms(
                frame=frames[domain],
                domain=domain,
                prefixes=batch_prefixes(batch=batch, domain=domain),
                workers=args.workers,
                drop_gap_rows=batch.drop_gap_rows,
            )
            losses.write_parquet(path)
            predictions_from_losses(losses=losses, frame=frames[domain]).write_parquet(
                predictions_path(output_dir=args.output_dir, domain=domain)
            )
        published = pl.read_parquet(
            losses_path(output_dir=args.published_dir, domain=domain)
        ).filter(pl.col("setting") == SETTING)
        context = [
            pl.read_parquet(losses_path(output_dir=directory, domain=domain)).filter(
                pl.col("setting") == SETTING
            )
            for directory in args.context_dir
        ]
        report += report_domain(
            domain=domain,
            losses=losses,
            published=published,
            shares=shares[domain],
            batch=batch,
            context=context,
        )
    report_path.write_text("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
