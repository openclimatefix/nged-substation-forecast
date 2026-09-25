"""Build the per-technology weather-arm frames the matched-lead comparison fits on.

One-off throwaway script for the study in
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>. Reads:

- **Power, the site list, and the shared rows**: `ens_forecast_horizons.base_frame`, the same
  hourly rows (commissioning ramp, zero half-hours and outages already dropped) the ENS-horizons
  page builds its own inputs from, made public in that script for this reuse.
- **Previous Runs products**: each product's `previous_runs/combined.parquet`, already carrying the
  anonymised `site` label. UKV's radiation is rebuilt from its two label-adjacent snapshots
  (`studies.hourly_means.hourly_from_snapshots`), because V3 reads UKV as an instantaneous snapshot
  rather than an hour-ending mean; every other product's radiation is used as served.
- **ECMWF ENS**: built directly on days 0-3, at the upsampling combination the ENS horizons page's
  rule chose before this study existed (`clear_sky` for solar, radiation through the clear-sky
  index and temperature by straight line; `speed_components` for wind), read from
  `UPSAMPLING_METHODS` rather than re-running the rule. `_ens_frame` calls
  `ens_forecast_horizons.band_steps`, `.upsampled_fields`, `.combine`, and `.reduce_members`
  directly rather than going through `.build_inputs`/`.main_frame`, because those two also build
  every band from 0 to 14 under every upsampling combination and then drop, through `._complete`,
  every row missing any of them — including a day 0-3 row this study could otherwise score, whenever
  the missing band is one of days 5 to 14 (commonly true for 10-14 days after a power outage, when
  the run that would cover the later bands has not itself finished backfilling). Reading only the
  four bands and one combination this study actually uses also cuts the upsampling cost by about
  12x for solar and 8x for wind (`len(BAND_DAYS) * len(COMBINATIONS[domain])` down to
  `len(ENS_DAYS) * 1`).
- **NOAA GEFS**: `_gefs_frame` builds the mean of GEFS's 31 members the same way, reusing the same
  `ens_forecast_horizons` functions with `ensemble_size=31` and GEFS's own extract as `source`,
  gated on `GEFS_MONTH_CACHE_DIR` (the finished `GEFS_window_2024-11-01_None` download's
  `_month_cache/`) holding every month from 2024-11 to the month the study's rows actually end on,
  or on `--gefs-window-dir` for a `GEFS_window_*` test extract during development, which always
  writes to the given `--output-dir` rather than the gated production path.
  **Partial-month runs**: the download writes the month still being fetched as
  `<month>.partial.parquet`, and only that last month's partial file is accepted; a partial file
  for any earlier month does not count as covering it. After loading, every 00 UTC run the rows need
  (init dates from the rows' first date minus the largest day offset to their last date minus the
  smallest, clipped to the first cached month) must hold all 31 members at every 3-hourly lead from
  0 to 95 h at every grid cell (with `--extra-leads`, at every lead `gefs_band_leads` returns:
  3-hourly to 240 h and each band's 6-hourly leads beyond), or the build raises with the missing
  or incomplete runs listed. A run still arriving therefore stops the build instead of turning
  into null GEFS columns.

With `--extra-leads` the script instead builds the exploratory lead columns (ENS at days 5, 10 and
14, GEFS at days 0, 5, 10 and 14, Previous Runs at day 0 for every product, at day 5 for ICON
global, and at days 5 and 7 for IFS 0.25° and GFS) on the published inputs' own `(site, time)`
keys, into a new write-once `--output-dir`, and never writes to the published folder. The new days
are separate constants (`EXTRA_ENS_DAYS`, `EXTRA_GEFS_DAYS`), never added to `ENS_DAYS`, so the
shared rows cannot move. `--batch second` builds the second batch instead: ENS mean at day 7, the
ENS control member at days 5, 7, 10 and 14, and GEFS mean at day 7 (`EXTRA_LEAD_BUILDS`).
`--batch third` builds `gfs_native_day<N>_*` at days 0, 1, 2, 3, 5, 7, 10 and 14 from the native
Dynamical.org GFS store (`GFS_NATIVE_DIR_NAME`), which no other batch reads; see `_gfs_native_frame`
for which run and lead each day serves and how the store's radiation is converted.
`--batch fourth` builds `ifs_single_day<N>_*` at `IFS_SINGLE_DAYS` from Open-Meteo's Single Runs
archive of ECMWF IFS HRES (`IFS_SINGLE_DIR_NAME`); see `_ifs_single_frame`.

Every output row carries only the anonymised `site` label; no generator name, id or coordinate is
read from the private roster in this script, except inside `studies.grid_sampling` (GEFS's
nearest-cell match), which never prints what it reads.

Run it with `uv run python studies/nwp_forecast_comparison/build_forecast_inputs.py --output-dir
DIR`.
"""

import argparse
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Final, Literal, NamedTuple

import numpy as np
import polars as pl
from contracts.settings import PROJECT_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_extra_leads import (
    gefs_boundary_table,
    gefs_boundary_verdict,
    gefs_window_table,
    gefs_window_verdict,
)
from verify_previous_runs_leads import PRODUCT_DIRS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "beam_diffuse_split"))
import ens_forecast_horizons as efh
from studies.gfs_native import (
    HOURLY_SERVED_LAST_DAY,
    LAST_LEAD_HOURS,
    gfs_leads,
    served_init_time,
    served_lead_hours,
    step_means,
    three_hour_means,
)
from studies.grid_sampling import nearest_cells
from studies.hourly_means import hourly_from_snapshots
from studies.ifs_single_runs import clip_radiation, last_servable_day
from studies.ifs_single_runs import served_init_time as ifs_single_init_time
from studies.ifs_single_runs import served_lead_hours as ifs_single_lead_hours
from studies.resample import gefs_step_means

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

DomainType = Literal["solar", "wind"]

DEFAULT_OUTPUT_DIR_NAME: Final[str] = "nwp_forecast_comparison"
"""Under `data/studies/`, the default place this script writes to."""

GEFS_FIRST_MONTH: Final[str] = "2024-11"
"""The first month the study reads GEFS from (the plan reads GEFS from 2024-11-30 onwards)."""

UPSAMPLING_METHODS: Final[dict[DomainType, str]] = {
    "solar": "clear_sky",
    "wind": "speed_components",
}
"""The ENS upsampling combination per technology, as the ENS horizons page's report records its rule
choosing them."""

ENS_DAYS: Final[tuple[int, ...]] = (0, 1, 2, 3)
"""The bands this study reads ENS at (days 5, 7, 10 and 14 are the horizons page's territory)."""

PRODUCT_SLUGS: Final[dict[str, str]] = {
    "UKV": "ukv",
    "ICON-D2": "icon_d2",
    "ICON-EU": "icon_eu",
    "ICON global": "icon_global",
    "IFS 0.25°": "ifs025",
    "GFS": "gfs",
    "ARPEGE Europe": "arpege",
    "AROME France": "arome",
    "KNMI HARMONIE-AROME": "knmi_harmonie",
    "DMI HARMONIE-AROME": "dmi_harmonie",
}
"""Each Previous Runs product's name (a `PRODUCT_DIRS` key) to the slug its output columns use."""

PRODUCT_DAY_OFFSETS: Final[dict[str, tuple[int, ...]]] = {
    "UKV": (1,),
    "ICON-D2": (1,),
    "ICON-EU": (1, 2, 3),
    "ICON global": (1, 2, 3),
    "IFS 0.25°": (1, 2, 3),
    "GFS": (1, 2, 3),
    "ARPEGE Europe": (1, 2, 3),
    "AROME France": (1,),
    "KNMI HARMONIE-AROME": (1,),
    "DMI HARMONIE-AROME": (1,),
}
"""Each product's `previous_dayN` offsets this study reads, from the plan's product table."""

EXTRA_PRODUCT_DAY_OFFSETS: Final[dict[str, tuple[int, ...]]] = {
    "UKV": (0,),
    "ICON-D2": (0,),
    "ICON-EU": (0,),
    "ICON global": (0, 5),
    "IFS 0.25°": (0, 5, 7),
    "GFS": (0, 5, 7),
    "ARPEGE Europe": (0,),
    "AROME France": (0,),
    "KNMI HARMONIE-AROME": (0,),
    "DMI HARMONIE-AROME": (0,),
}
"""The exploratory `previous_dayN` offsets the extra-lead build reads on top of
`PRODUCT_DAY_OFFSETS`. Day 0 is Open-Meteo's unsuffixed series, the freshest run that covers each
hour, so its served lead depends on each product's run cycle; ICON-EU's archive ends at day 4,
ICON-D2's holds only days 0 and 1, and IFS 0.25°'s and GFS's end at day 7."""

EXTRA_ENS_DAYS: Final[tuple[int, ...]] = (5, 10, 14)
"""The ENS bands the extra-lead build adds. They never join `ENS_DAYS`, which also decides the
baseline columns every shared row must hold, so adding them there would move the shared rows."""

EXTRA_GEFS_DAYS: Final[tuple[int, ...]] = (0, 5, 10, 14)
"""The GEFS bands the extra-lead build adds."""

EXTRA_ENS_CONTROL_DAYS: Final[tuple[int, ...]] = (5, 7, 10, 14)
"""The ENS control-member bands the second extra-lead build adds. Days 0 to 3 are already in the
published inputs."""


ExtraBatchType = Literal["first", "second", "third", "fourth"]
"""Which extra-lead build: the first, second, third, or fourth batch's columns."""


class ExtraLeadBuild(NamedTuple):
    """The columns one extra-lead build adds on top of the published inputs."""

    product_day_offsets: Mapping[str, tuple[int, ...]]
    ens_mean_days: tuple[int, ...]
    ens_control_days: tuple[int, ...]
    gefs_days: tuple[int, ...]
    gfs_native_days: tuple[int, ...] = ()
    ifs_single_days: tuple[int, ...] = ()


GFS_NATIVE_DAYS: Final[tuple[int, ...]] = (0, 1, 2, 3, 5, 7, 10, 14)
"""The lead days the third batch reads the native GFS store at."""

IFS_SINGLE_DAYS: Final[tuple[int, ...]] = (0, 1, 2, 3, 5, 7)
"""The lead days the fourth batch reads Open-Meteo's IFS HRES (9 km) archive at. Day 10 is absent
because the archive's runs end at lead 240 hours, so day 10's leads (240 to 263 for wind, 241 to
264 for solar) are almost all beyond them; `studies.ifs_single_runs.last_servable_day` is 9."""

EXTRA_LEAD_BUILDS: Final[dict[ExtraBatchType, ExtraLeadBuild]] = {
    "first": ExtraLeadBuild(
        product_day_offsets=EXTRA_PRODUCT_DAY_OFFSETS,
        ens_mean_days=EXTRA_ENS_DAYS,
        ens_control_days=(),
        gefs_days=EXTRA_GEFS_DAYS,
    ),
    "second": ExtraLeadBuild(
        product_day_offsets={},
        ens_mean_days=(7,),
        ens_control_days=EXTRA_ENS_CONTROL_DAYS,
        gefs_days=(7,),
    ),
    "third": ExtraLeadBuild(
        product_day_offsets={},
        ens_mean_days=(),
        ens_control_days=(),
        gefs_days=(),
        gfs_native_days=GFS_NATIVE_DAYS,
    ),
    "fourth": ExtraLeadBuild(
        product_day_offsets={},
        ens_mean_days=(),
        ens_control_days=(),
        gefs_days=(),
        ifs_single_days=IFS_SINGLE_DAYS,
    ),
}
"""The first batch's columns (unchanged from its own build), the second batch's (ENS mean at day 7,
ENS control member at days 5, 7, 10 and 14, and GEFS mean at day 7, with no Previous Runs column
because the arms it refits take their columns from the published inputs), and the third batch's
(the native GFS store at `GFS_NATIVE_DAYS`, and nothing else), and the fourth batch's (Open-Meteo's
IFS HRES archive at `IFS_SINGLE_DAYS`, and nothing else)."""

SOLAR_ONLY_PRODUCTS: Final[frozenset[str]] = frozenset({"ARPEGE Europe", "AROME France"})
"""Products the plan scores for solar only: their 100 m wind offsets are missing on most rows."""

SNAPSHOT_RADIATION_PRODUCTS: Final[frozenset[str]] = frozenset({"UKV"})
"""Products whose Previous Runs radiation is an instantaneous snapshot (V3), rebuilt through
`hourly_from_snapshots` rather than used as served."""


def _repo_data_dir() -> Path:
    """Return the shared `data/` directory, resolving a linked worktree to the main checkout.

    See `verify_previous_runs_leads._repo_data_dir` for the reasoning; duplicated here because
    study scripts in different directories cannot import one another's private helpers.

    Returns:
        The directory holding `studies/`, `NGED/` and the rest of the shared downloads.
    """
    root = PROJECT_ROOT
    marker = root / ".git"
    if marker.is_file():
        pointer = marker.read_text().removeprefix("gitdir:").strip()
        git_dir = Path(pointer)
        if git_dir.parent.name == "worktrees":
            root = git_dir.parent.parent.parent
    return Path(os.environ.get("DATA_PATH_INTERNAL") or root / "data")


def _weather_dir() -> Path:
    """Return `data/studies/weather/`, where every downloaded weather product lives."""
    return _repo_data_dir() / "studies" / "weather"


def _previous_column(*, name: str, day: int) -> str:
    """Return Open-Meteo's column for a variable at a day offset.

    Args:
        name: The variable, such as `shortwave_radiation`.
        day: The offset: 0 for the unsuffixed series (the freshest run covering each hour), else N
            for `<name>_previous_day<N>`.

    Returns:
        The column's name.
    """
    return name if day == 0 else f"{name}_previous_day{day}"


def _solar_columns(*, frame: pl.DataFrame, day: int, snapshot: bool) -> pl.DataFrame:
    """Return one product's day-N radiation and temperature, on the rows both are present.

    Args:
        frame: The product's `previous_runs/combined.parquet`.
        day: The `previous_dayN` offset to read.
        snapshot: Whether the radiation column is an instantaneous snapshot that needs rebuilding
            through `hourly_from_snapshots` (UKV) rather than being used as served.

    Returns:
        `site`, `time`, `ghi`, `temp`.
    """
    ghi_column = _previous_column(name="shortwave_radiation", day=day)
    temp_column = _previous_column(name="temperature_2m", day=day)
    if snapshot:
        snapshots = frame.select(
            key=pl.col("site"), time="time", value=pl.col(ghi_column)
        ).drop_nulls()
        ghi = hourly_from_snapshots(
            frame=snapshots, value_columns=["value"], slot_offsets_minutes=(-60, 0)
        ).select(site=pl.col("key"), time="time", ghi=pl.col("value"))
    else:
        ghi = frame.select("site", "time", ghi=pl.col(ghi_column)).drop_nulls()
    temp = frame.select("site", "time", temp=pl.col(temp_column)).drop_nulls()
    return ghi.join(temp, on=["site", "time"], how="inner")


KMH_TO_MS: Final[float] = 1.0 / 3.6
"""Previous Runs serves wind speed in km/h; every other product (ENS, GEFS) is in m/s, and the plan
converts every speed to m/s so a wind arm's columns are on one unit regardless of its product."""


def _wind_columns(*, frame: pl.DataFrame, day: int) -> pl.DataFrame:
    """Return one product's day-N hub-height wind and 10 m wind, on the rows all three are present.

    Args:
        frame: The product's `previous_runs/combined.parquet`.
        day: The `previous_dayN` offset to read.

    Returns:
        `site`, `time`, `speed_100m`, `sin_100m`, `cos_100m`, `speed_10m`, speeds in m/s.
    """
    speed_100m_column = _previous_column(name="wind_speed_100m", day=day)
    direction_100m_column = _previous_column(name="wind_direction_100m", day=day)
    speed_10m_column = _previous_column(name="wind_speed_10m", day=day)
    return (
        frame.select(
            "site",
            "time",
            speed_100m=pl.col(speed_100m_column) * KMH_TO_MS,
            direction_100m=pl.col(direction_100m_column),
            speed_10m=pl.col(speed_10m_column) * KMH_TO_MS,
        )
        .drop_nulls()
        .with_columns(
            sin_100m=pl.col("direction_100m").radians().sin(),
            cos_100m=pl.col("direction_100m").radians().cos(),
        )
        .drop("direction_100m")
    )


def _previous_runs_frame(
    *,
    keys: pl.DataFrame,
    domain: DomainType,
    day_offsets: Mapping[str, tuple[int, ...]] = PRODUCT_DAY_OFFSETS,
) -> pl.DataFrame:
    """Join every Previous Runs product's arm columns onto the shared rows.

    Args:
        keys: `site`, `time` for every row the study might score.
        domain: `solar` or `wind`.
        day_offsets: Each product's `previous_dayN` offsets to read; a product it omits is skipped.

    Returns:
        `keys` with `<slug>_day<N>_<field>` for every product and offset that applies to `domain`,
        left-joined (a row missing a product's offset carries a null, not a dropped row: the row
        set is shared across the planned arms only, which `nwp_forecast_comparison.py` enforces).
    """
    frame = keys
    for product, dir_name in PRODUCT_DIRS.items():
        if product not in day_offsets or (domain == "wind" and product in SOLAR_ONLY_PRODUCTS):
            continue
        path = _weather_dir() / dir_name / "previous_runs" / "combined.parquet"
        if not path.exists():
            _LOG.warning("%s: %s not found, skipping", product, path)
            continue
        slug = PRODUCT_SLUGS[product]
        combined = pl.read_parquet(path)
        for day in day_offsets[product]:
            if domain == "solar":
                ghi_column = _previous_column(name="shortwave_radiation", day=day)
                if ghi_column not in combined.columns:
                    continue
                columns = _solar_columns(
                    frame=combined, day=day, snapshot=product in SNAPSHOT_RADIATION_PRODUCTS
                )
                rename = {"ghi": f"{slug}_day{day}_ghi", "temp": f"{slug}_day{day}_temp"}
            else:
                speed_column = _previous_column(name="wind_speed_100m", day=day)
                if speed_column not in combined.columns:
                    continue
                columns = _wind_columns(frame=combined, day=day)
                rename = {
                    "speed_100m": f"{slug}_day{day}_speed_100m",
                    "sin_100m": f"{slug}_day{day}_sin_100m",
                    "cos_100m": f"{slug}_day{day}_cos_100m",
                    "speed_10m": f"{slug}_day{day}_speed_10m",
                }
            frame = frame.join(columns.rename(rename), on=["site", "time"], how="left")
    return frame


def _ens_member_arms(
    *,
    extract: pl.DataFrame,
    domain: DomainType,
    days: tuple[int, ...],
    method: str,
    ensemble_size: int,
    arm_name: Callable[[str, int], str],
    ways: tuple[str, ...] = ("mean", "control"),
    fine_step_last_lead: int = efh.FINE_STEP_LAST_LEAD,
) -> list[pl.DataFrame]:
    """Upsample, combine and reduce one ensemble's members at several bands and ways.

    Shared by `_ens_frame` (ENS, both ways) and `_gefs_frame` (GEFS, mean only), whose only
    difference is the extract they read, its ensemble size, and the arm-name prefix.

    Args:
        extract: One row per (site, init_time, ensemble_member, lead_hours), as
            `ens_forecast_horizons.members` or `_gefs_members_frame` returns.
        domain: `solar` or `wind`.
        days: The bands to build.
        method: The upsampling combination (a key of `ens_forecast_horizons.COMBINATIONS[domain]`).
        ensemble_size: How many members a run must hold to be kept.
        arm_name: Given a way (`"mean"` or `"control"`) and a day, returns that arm's name.
        ways: Which reductions to build; ENS wants both, GEFS only the mean (no control-member arm
            is planned for GEFS).
        fine_step_last_lead: The last lead on 3-hour steps: 144 for ENS, 240 for GEFS.

    Returns:
        One frame per (day, way) with `site`, `time` and that arm's own weather columns.
    """
    clear_sky = efh.clear_sky_table(domain=domain)
    frames: list[pl.DataFrame] = []
    for day in days:
        steps = efh.band_steps(
            members=extract,
            day=day,
            domain=domain,
            ensemble_size=ensemble_size,
            fine_step_last_lead=fine_step_last_lead,
        )
        upsampled = efh.upsampled_fields(steps=steps, day=day, domain=domain, clear_sky=clear_sky)
        combined = efh.combine(
            steps=steps, upsampled=upsampled, day=day, domain=domain, method=method
        )
        for way in ways:
            reduced = efh.reduce_members(
                hourly=combined, domain=domain, way=way, ensemble_size=ensemble_size
            )
            frames.append(efh.prefixed(frame=reduced, arm=arm_name(way, day), domain=domain))
    return frames


def _ens_frame(*, keys: pl.DataFrame, domain: DomainType) -> pl.DataFrame:
    """Build ECMWF ENS's mean and control-member columns at `ENS_DAYS` directly on the base rows.

    Calls `ens_forecast_horizons.band_steps`, `.upsampled_fields`, `.combine` and `.reduce_members`
    directly, rather than going through `.build_inputs`/`.main_frame`, which build every band from
    0 to 14 under every upsampling combination and then drop, through `._complete`, every row
    missing any of them. That would silently cost this study a row it could otherwise score at
    days 0-3 whenever the band actually missing is one of days 5-14 (commonly true for 10-14 days
    after a power outage, before the later-band run has finished backfilling); see the module
    docstring's row-count comparison against the previous build.

    Args:
        keys: `site`, `time` for every row the study might score.
        domain: `solar` or `wind`.

    Returns:
        `keys` with `ens_mean_day<N>_<field>` and `ens_control_day<N>_<field>` for every `N` in
        `ENS_DAYS`, left-joined, plus `persistence_day<N>` and `diurnal_persistence_day<N>` and, for
        solar, `clear_sky_index_day<N>` and `clear_sky_w_m2` (the no-weather baselines' own inputs,
        including smart persistence's), which `with_baselines` computes on the way.
    """
    baselined = efh.with_baselines(frame=efh.base_frame(domain=domain), domain=domain)
    sites = sorted(baselined["site"].unique().to_list())
    extract = efh.members(sites=sites)
    arms = _ens_member_arms(
        extract=extract,
        domain=domain,
        days=ENS_DAYS,
        method=UPSAMPLING_METHODS[domain],
        ensemble_size=efh.ENSEMBLE_SIZE,
        arm_name=lambda way, day: efh.ens_arm(way=way, day=day),
    )
    baseline_columns = [
        efh.baseline_arm(name=name, day=day)
        for day in ENS_DAYS
        for name in ("persistence", "diurnal_persistence")
    ]
    if domain == "solar":
        baseline_columns += ["clear_sky_w_m2", *(f"clear_sky_index_day{day}" for day in ENS_DAYS)]
    frame = baselined.select("site", "time", *baseline_columns)
    for arm_frame in arms:
        frame = frame.join(arm_frame, on=["site", "time"], how="left")
    return keys.join(frame, on=["site", "time"], how="left")


def compare_ens_rebuild(*, domain: DomainType) -> float:
    """Check the rebuilt ENS columns against the old `build_inputs`/`main_frame` path.

    Run once by hand (not by `main`) to confirm `_ens_frame`'s direct build agrees with the
    previous implementation on the rows both hold, before relying on the direct build.

    Args:
        domain: `solar` or `wind`.

    Returns:
        The largest absolute difference between the two builds' `ens_mean_day<N>_<field>` and
        `ens_control_day<N>_<field>` columns, on the rows both builds hold (0.0 if identical).
    """
    inputs = efh.build_inputs(domain=domain)
    old = efh.main_frame(inputs=inputs, method=UPSAMPLING_METHODS[domain], domain=domain).frame
    wanted = [
        column
        for day in ENS_DAYS
        for way in ("mean", "control")
        for column in efh.ens_columns(arm=efh.ens_arm(way=way, day=day), domain=domain)
    ]
    keys = old.select("site", "time")
    new = _ens_frame(keys=keys, domain=domain)
    joined = old.select("site", "time", *wanted).join(
        new.select("site", "time", *wanted), on=["site", "time"], how="inner", suffix="_new"
    )
    if joined.is_empty():
        return 0.0
    differences = joined.select(
        pl.max_horizontal(
            (pl.col(column) - pl.col(f"{column}_new")).abs().max() for column in wanted
        ).alias("largest")
    )
    return float(differences.item() or 0.0)


GEFS_WINDOW_DIR_NAME: Final[str] = "GEFS_window_2024-11-01_None"
"""Under `data/studies/weather/`, the finished GEFS download: its `_month_cache/` and
`_grid_cells.parquet`."""

GEFS_MONTH_CACHE_DIR_NAME: Final[str] = "_month_cache"
"""Inside a GEFS download directory, where `fetch_dynamical_zarr.py` checkpoints each month."""

GEFS_PARTIAL_SUFFIX: Final[str] = ".partial"
"""Appended to a month's stem (`<month>.partial.parquet`) while its newest runs still arrive."""

GEFS_REQUIRED_MAX_LEAD_HOURS: Final[int] = 24 * 3 + 23
"""The last lead the study needs of a run: the end of day 3's band."""

GEFS_REQUIRED_LEAD_STEP_HOURS: Final[int] = 3
"""GEFS's lead spacing out to `GEFS_STEP_MEAN_MAX_LEAD_HOURS`."""


def _gefs_months_available(*, last_month: str) -> dict[str, Path]:
    """Return each calendar month the GEFS month cache holds, mapped to its file.

    Args:
        last_month: The last month the study needs, `%Y-%m`. Only this month's
            `<month>.partial.parquet` is accepted, and only if no complete file exists for it.

    Returns:
        Each cached month's `%Y-%m` label to its parquet path, sorted by month.
    """
    cache_dir = _weather_dir() / GEFS_WINDOW_DIR_NAME / GEFS_MONTH_CACHE_DIR_NAME
    if not cache_dir.exists():
        return {}
    months: dict[str, Path] = {}
    for path in sorted(cache_dir.glob("*.parquet")):
        if path.stem.endswith(GEFS_PARTIAL_SUFFIX):
            month = path.stem.removesuffix(GEFS_PARTIAL_SUFFIX)
            if month == last_month:
                months.setdefault(month, path)
        else:
            months[path.stem] = path
    return dict(sorted(months.items()))


def _gefs_span_complete(*, last_month: str) -> bool:
    """Whether the GEFS month cache covers every month from `GEFS_FIRST_MONTH` to `last_month`.

    Args:
        last_month: The last month the study needs, `%Y-%m`.

    Returns:
        Whether every month in that span is cached, with no gap.
    """
    available = set(_gefs_months_available(last_month=last_month))
    first_year, first_month = (int(part) for part in GEFS_FIRST_MONTH.split("-"))
    last_year, last_month_number = (int(part) for part in last_month.split("-"))
    expected = pl.date_range(
        pl.date(first_year, first_month, 1),
        pl.date(last_year, last_month_number, 1),
        interval="1mo",
        eager=True,
    )
    return all(f"{d.year:04d}-{d.month:02d}" in available for d in expected)


GEFS_ENSEMBLE_SIZE: Final[int] = 31
"""How many members a GEFS run holds (ENS holds 51)."""

GEFS_STEP_MEAN_MAX_LEAD_HOURS: Final[int] = 240
"""GEFS steps 3-hourly to this lead and 6-hourly beyond it. `gefs_step_means` inverts the
alternating windows up to this lead; beyond it each value is already a 6-hour mean and passes
through (`verify_extra_leads.py` checks that reading)."""

GEFS_DAYS: Final[dict[DomainType, tuple[int, ...]]] = {
    "solar": (1, 2, 3),
    "wind": (1, 2, 3),
}
"""The bands GEFS is built at (the plan's product table: day offsets 1-3, from 2024-11-30)."""


def gefs_band_leads(*, days: Sequence[int]) -> list[int]:
    """Return every lead of a GEFS run that the bands for `days` need, in hours.

    Radiation is inverted from windows to step means along a whole run, so every 3-hourly lead from
    0 to `GEFS_STEP_MEAN_MAX_LEAD_HOURS` is needed whatever the days. Beyond that lead GEFS steps
    every 6 hours, and a band reads the 6-hourly leads that fall in its window (`band_steps` reads
    from `24 * day - 6` to `24 * day + 30`).

    Args:
        days: The bands to build.

    Returns:
        The needed leads, sorted and without repeats.
    """
    leads = set(range(0, GEFS_STEP_MEAN_MAX_LEAD_HOURS + 1, GEFS_REQUIRED_LEAD_STEP_HOURS))
    for day in days:
        window = range(max(24 * day - 6, 0), 24 * day + 31)
        leads |= {lead for lead in window if lead > GEFS_STEP_MEAN_MAX_LEAD_HOURS and lead % 6 == 0}
    return sorted(leads)


def _gefs_cell_selection(
    *, grid_cells: pl.DataFrame, domain: DomainType, sites: list[str]
) -> dict[str, int]:
    """Return each site's nearest GEFS grid cell id.

    Args:
        grid_cells: `_grid_cells.parquet`'s rows, carrying `lat_index`, `lon_index`, `latitude` and
            `longitude`.
        domain: `solar` or `wind`.
        sites: The sites to match.

    Returns:
        Each site to its nearest cell id (`lat_index * 10 + lon_index`, the same convention V1
        uses). No coordinate is read outside `studies.grid_sampling`, and none is returned.
    """
    cells = grid_cells.with_columns(cell_id=pl.col("lat_index") * 10 + pl.col("lon_index"))
    roster = efh.site_roster(domain=domain).filter(pl.col("site").is_in(sites))
    nearest = nearest_cells(sites=roster, cells=cells)
    return dict(zip(nearest["site"].to_list(), nearest["cell_id"].to_list(), strict=True))


def _gefs_step_mean_radiation(*, radiation: pl.DataFrame) -> pl.DataFrame:
    """Convert one cell's GEFS radiation from alternating windows to plain step means.

    Runs `studies.resample.gefs_step_means` on each (run, member) series up to
    `GEFS_STEP_MEAN_MAX_LEAD_HOURS`, in lead order, on the whole run before any band slicing, as
    the plan requires. Beyond that lead GEFS steps every 6 hours and every value is already a plain
    6-hour mean (`verify_extra_leads.py` checks that reading), so those leads pass through
    unchanged.

    Args:
        radiation: `init_time`, `ensemble_member`, `lead_hours`, `ghi_raw`, with the null value at
            lead 0 already excluded.

    Returns:
        `init_time`, `ensemble_member`, `lead_hours`, `ghi_w_m2`: the same (run, member, lead)
        rows, radiation replaced by step means. A (run, member) missing any lead up to
        `GEFS_STEP_MEAN_MAX_LEAD_HOURS` is dropped whole, as `ens_forecast_horizons.band_steps`
        drops an incomplete run.
    """
    fine_radiation = radiation.filter(pl.col("lead_hours") <= GEFS_STEP_MEAN_MAX_LEAD_HOURS)
    schema = {
        "init_time": pl.Datetime("us", "UTC"),
        "ensemble_member": pl.Int8,
        "lead_hours": pl.Int32,
        "ghi_w_m2": pl.Float64,
    }
    leads = np.sort(fine_radiation["lead_hours"].unique().to_numpy())
    lead_columns = [str(int(lead)) for lead in leads]
    wide = fine_radiation.pivot(
        on="lead_hours", index=["init_time", "ensemble_member"], values="ghi_raw", sort_columns=True
    ).drop_nulls(lead_columns)
    if wide.is_empty():
        return pl.DataFrame(schema=schema)
    values = wide.select(lead_columns).to_numpy()
    step_means = gefs_step_means(values=values, leads=leads.astype(np.float64))
    fine = (
        wide.select("init_time", "ensemble_member")
        .with_columns(
            [pl.Series(lead_columns[index], step_means[:, index]) for index in range(len(leads))]
        )
        .unpivot(
            index=["init_time", "ensemble_member"],
            on=lead_columns,
            variable_name="lead_hours",
            value_name="ghi_w_m2",
        )
        .with_columns(lead_hours=pl.col("lead_hours").cast(pl.Int32))
    )
    coarse = (
        radiation.filter(pl.col("lead_hours") > GEFS_STEP_MEAN_MAX_LEAD_HOURS)
        .join(wide.select("init_time", "ensemble_member"), on=["init_time", "ensemble_member"])
        .select("init_time", "ensemble_member", "lead_hours", ghi_w_m2="ghi_raw")
        .cast(
            {
                "ensemble_member": fine.schema["ensemble_member"],
                "ghi_w_m2": fine.schema["ghi_w_m2"],
            }
        )
    )
    return pl.concat([fine, coarse], how="vertical")


def _gefs_missing_runs(
    *,
    files: list[Path],
    first_init: date,
    last_init: date,
    leads: Sequence[int] | None = None,
) -> list[str]:
    """Return the 00 UTC runs in `first_init` to `last_init` that are absent or incomplete.

    A run is complete when every grid cell holds all `GEFS_ENSEMBLE_SIZE` members at every lead in
    `leads`, by default 0 to `GEFS_REQUIRED_MAX_LEAD_HOURS` in steps of
    `GEFS_REQUIRED_LEAD_STEP_HOURS`.

    Args:
        files: The month parquets (or the extract's `GEFS.parquet`) to read.
        first_init: The first init date the study needs.
        last_init: The last init date the study needs.
        leads: The leads every run must hold, in hours; `None` reads the default above.

    Returns:
        One `YYYY-MM-DD: <reason>` line per missing or incomplete run, sorted by date.
    """
    if leads is None:
        leads = range(0, GEFS_REQUIRED_MAX_LEAD_HOURS + 1, GEFS_REQUIRED_LEAD_STEP_HOURS)
    expected = GEFS_ENSEMBLE_SIZE * len(leads)
    lead_hours = (pl.col("lead_time").dt.total_minutes() / 60).cast(pl.Int32)
    present = (
        pl.scan_parquet(files)
        .with_columns(lead_hours=lead_hours, init_date=pl.col("init_time").dt.date())
        .filter(
            pl.col("init_time").dt.hour() == 0,
            pl.col("init_date").is_between(first_init, last_init),
            pl.col("lead_hours").is_in(list(leads)),
        )
        .group_by("init_date", "lat_index", "lon_index")
        .agg(n=pl.struct("ensemble_member", "lead_hours").n_unique())
        .group_by("init_date")
        .agg(worst=pl.col("n").min())
        .collect()
    )
    worst_by_date = dict(
        zip(present["init_date"].to_list(), present["worst"].to_list(), strict=True)
    )
    missing: list[str] = []
    for day in pl.date_range(first_init, last_init, interval="1d", eager=True):
        worst = worst_by_date.get(day, 0)
        if worst < expected:
            missing.append(f"{day}: {worst} of {expected} member-lead pairs in the sparsest cell")
    return missing


def _gefs_members_frame(
    *,
    path: Path,
    files: list[Path],
    domain: DomainType,
    sites: list[str],
    max_lead_hours: int = GEFS_STEP_MEAN_MAX_LEAD_HOURS,
) -> pl.DataFrame:
    """Read one GEFS extract and reshape it to `ens_forecast_horizons.band_steps`'s input shape.

    Each site reads its nearest 0.25 degree grid cell. Wind u/v components become speed and
    direction (GEFS already gives both heights in m/s, unlike Previous Runs' km/h). Radiation is
    converted through `_gefs_step_mean_radiation`.

    Args:
        path: The download's directory, holding `_grid_cells.parquet`.
        files: The parquets holding the runs: the extract's `GEFS.parquet`, or the month cache's
            files.
        domain: `solar` or `wind`.
        sites: The sites to build.
        max_lead_hours: The longest lead to keep.

    Returns:
        One row per (site, init_time, ensemble_member, lead_hours), with `ghi_w_m2`, `temp_c`,
        `speed_100m`, `direction_100m`, `speed_10m`, `direction_10m`.
    """
    grid_cells = pl.read_parquet(path / "_grid_cells.parquet")
    cell_by_site = _gefs_cell_selection(grid_cells=grid_cells, domain=domain, sites=sites)
    raw = (
        pl.read_parquet(files)
        .with_columns(
            lead_hours=(pl.col("lead_time").dt.total_minutes() / 60).cast(pl.Int32),
            cell=pl.col("lat_index") * 10 + pl.col("lon_index"),
            init_time=pl.col("init_time").dt.replace_time_zone("UTC"),
        )
        .filter(pl.col("lead_hours") <= max_lead_hours)
    )
    frames: list[pl.DataFrame] = []
    for site, cell_id in cell_by_site.items():
        cell_rows = raw.filter(pl.col("cell") == cell_id)
        if cell_rows.is_empty():
            continue
        wind = cell_rows.select(
            "init_time",
            "ensemble_member",
            "lead_hours",
            speed_100m=(pl.col("wind_u_100m") ** 2 + pl.col("wind_v_100m") ** 2).sqrt(),
            direction_100m=(
                pl.arctan2(-pl.col("wind_u_100m"), -pl.col("wind_v_100m")).degrees() % 360.0
            ),
            speed_10m=(pl.col("wind_u_10m") ** 2 + pl.col("wind_v_10m") ** 2).sqrt(),
            direction_10m=(
                pl.arctan2(-pl.col("wind_u_10m"), -pl.col("wind_v_10m")).degrees() % 360.0
            ),
            temp_c="temperature_2m",
        )
        radiation_input = (
            cell_rows.filter(pl.col("lead_hours") > 0)
            .select(
                "init_time",
                "ensemble_member",
                "lead_hours",
                ghi_raw="downward_short_wave_radiation_flux_surface",
            )
            .drop_nulls()
            .filter(pl.col("ghi_raw").is_not_nan())
        )
        radiation = _gefs_step_mean_radiation(radiation=radiation_input)
        merged = wind.join(radiation, on=["init_time", "ensemble_member", "lead_hours"], how="left")
        frames.append(merged.with_columns(site=pl.lit(site)))
    if not frames:
        return pl.DataFrame(
            schema={
                "site": pl.String,
                "init_time": pl.Datetime("us", "UTC"),
                "ensemble_member": pl.Int8,
                "lead_hours": pl.Int32,
                "ghi_w_m2": pl.Float64,
                "temp_c": pl.Float32,
                "speed_100m": pl.Float64,
                "direction_100m": pl.Float64,
                "speed_10m": pl.Float64,
                "direction_10m": pl.Float64,
            }
        )
    return pl.concat(frames, how="diagonal")


def _gefs_frame(
    *,
    keys: pl.DataFrame,
    domain: DomainType,
    window_dir: Path | None,
    days: Sequence[int] | None = None,
) -> pl.DataFrame:
    """Build NOAA GEFS's mean columns, gated on a complete month cache or a test window extract.

    In production (`window_dir=None`) this only runs once `GEFS_WINDOW_DIR_NAME`'s `_month_cache/`
    covers every month from `GEFS_FIRST_MONTH` to the month `keys`'s own rows end on -- not "the
    month before today", which in January gives an invalid `YYYY-00` month and, every other month,
    checks a span the rows may not even reach. With `--gefs-window-dir`, a `GEFS_window_*` test
    extract is read unconditionally, for development. In production, `_gefs_missing_runs` must find
    every needed run complete, or this raises; a test extract is not checked, since it covers only
    part of the rows' span.

    Args:
        keys: `site`, `time` for every row the study might score.
        domain: `solar` or `wind`.
        window_dir: A `GEFS_window_*` test extract, for development only. `None` in production.
        days: The bands to build. `None` builds `GEFS_DAYS[domain]`, reading leads to 240 h; any
            other days read every lead they need (`gefs_band_leads`) and check every run holds
            them.

    Returns:
        `keys` with `gefs_mean_day<N>_<field>` for every `N` in `days`, left-joined.
        Unchanged (no GEFS columns) while the production gate does not pass.

    Raises:
        RuntimeError: If any needed 00 UTC run is missing or incomplete.
    """
    default_days = days is None
    build_days = tuple(GEFS_DAYS[domain] if days is None else days)
    required_leads = None if default_days else gefs_band_leads(days=build_days)
    max_lead_hours = (
        GEFS_STEP_MEAN_MAX_LEAD_HOURS if required_leads is None else max(required_leads)
    )
    time_range = keys.select(first=pl.col("time").min(), last=pl.col("time").max()).row(
        0, named=True
    )
    if window_dir is None:
        last_needed = time_range["last"].strftime("%Y-%m")
        months = _gefs_months_available(last_month=last_needed)
        if not months:
            _LOG.info("GEFS: %s holds no months, no GEFS columns written.", GEFS_WINDOW_DIR_NAME)
            return keys
        if not _gefs_span_complete(last_month=last_needed):
            _LOG.info(
                "GEFS: month cache holds %d months but does not yet cover %s to %s (the rows' "
                "last month), no GEFS columns written.",
                len(months),
                GEFS_FIRST_MONTH,
                last_needed,
            )
            return keys
        path = _weather_dir() / GEFS_WINDOW_DIR_NAME
        files = list(months.values())
        first_init = max(
            time_range["first"].date() - timedelta(days=max(build_days)),
            date.fromisoformat(f"{GEFS_FIRST_MONTH}-01"),
        )
        last_init = time_range["last"].date() - timedelta(days=min(build_days))
        missing = _gefs_missing_runs(
            files=files, first_init=first_init, last_init=last_init, leads=required_leads
        )
        if missing:
            raise RuntimeError(
                f"GEFS: {len(missing)} of the 00 UTC runs the rows need are missing or incomplete "
                "(partial-month gaps would silently become null GEFS columns):\n"
                + "\n".join(missing)
            )
    else:
        path = window_dir
        files = [path / "GEFS.parquet"]
    sites = sorted(keys["site"].unique().to_list())
    extract = _gefs_members_frame(
        path=path, files=files, domain=domain, sites=sites, max_lead_hours=max_lead_hours
    )
    if extract.is_empty():
        _LOG.warning("GEFS: no rows matched at %s, no GEFS columns written.", path)
        return keys
    arms = _ens_member_arms(
        extract=extract,
        domain=domain,
        days=build_days,
        method=UPSAMPLING_METHODS[domain],
        ensemble_size=GEFS_ENSEMBLE_SIZE,
        arm_name=lambda way, day: f"gefs_{way}_day{day}",
        ways=("mean",),
        fine_step_last_lead=GEFS_STEP_MEAN_MAX_LEAD_HOURS,
    )
    frame = keys
    for arm_frame in arms:
        frame = frame.join(arm_frame, on=["site", "time"], how="left")
    return frame


GFS_NATIVE_DIR_NAME: Final[str] = "GFS"
"""Under `data/studies/weather/`, the native Dynamical.org GFS store: `GFS.parquet` (every run at
00, 06, 12 and 18 UTC) and `_grid_cells.parquet`."""

GFS_NATIVE_ARM_PREFIX: Final[str] = "gfs_native"
"""The column prefix of the native GFS arms, `gfs_native_day<N>_<field>`."""

GFS_NATIVE_MAX_MISSING_SHARE: Final[float] = 0.015
"""The largest share of rows with a null in one native GFS arm's columns the build accepts, the same
limit `fit_extra_leads.MAX_MISSING_SHARE` applies to the shared rows."""

GFS_NATIVE_RUN_MARGIN_DAYS: Final[int] = 2
"""How many days before the largest served day the store is read from, for the run cycles and the
hour a solar label is taken from. The read never starts before `efh.SPAN`'s first hour, which is
where the clear-sky table the band days' upsampling reads begins."""


def gfs_native_arm(*, day: int) -> str:
    """Return the column prefix of the native GFS arm at one lead day."""
    return f"{GFS_NATIVE_ARM_PREFIX}_day{day}"


def _long_radiation(*, index: pl.DataFrame, leads: np.ndarray, array: np.ndarray) -> pl.DataFrame:
    """Turn radiation arrays over (series, lead) into one row per cell, run and lead.

    Args:
        index: `cell` and `init_time`, one row per array row.
        leads: The array's columns' leads in hours.
        array: Shape (index.height, len(leads)), NaN where missing.

    Returns:
        `cell`, `init_time`, `lead_hours`, `ghi_w_m2`, with each NaN replaced by a null.
    """
    return (
        index[np.repeat(np.arange(index.height), len(leads))]
        .with_columns(
            lead_hours=pl.Series(np.tile(leads, index.height)).cast(pl.Int32),
            ghi_w_m2=pl.Series(array.reshape(-1)),
        )
        .with_columns(pl.col("ghi_w_m2").fill_nan(None))
    )


def _gfs_native_step_radiation(*, radiation: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Convert the store's since-reset radiation to step means, per cell and run.

    Args:
        radiation: `cell`, `init_time`, `lead_hours` (1 or more) and `ghi_raw`, the store's mean
            since the last reset, in W/m2.

    Returns:
        The mean over each step on the store's own steps (1 hour to lead 120, 3 hours beyond), and
        the mean over each 3 hours at every multiple of 3 hours, both as `_long_radiation` returns
        them. A run's missing lead leaves a null at its own step and at the next step of its reset
        window, and nowhere else (`studies.gfs_native.step_means`).
    """
    leads = gfs_leads()
    wide = radiation.pivot(on="lead_hours", index=["cell", "init_time"], values="ghi_raw")
    values = np.full((wide.height, len(leads)), np.nan)
    for position, lead in enumerate(leads):
        if (name := str(int(lead))) in wide.columns:
            values[:, position] = wide[name].cast(pl.Float64).fill_null(float("nan")).to_numpy()
    index = wide.select("cell", "init_time")
    hourly = step_means(values=values, leads=leads)
    three_leads, three = three_hour_means(hourly=hourly, leads=leads)
    return (
        _long_radiation(index=index, leads=leads, array=hourly),
        _long_radiation(index=index, leads=three_leads, array=three),
    )


def _gfs_native_weather(*, raw: pl.DataFrame) -> pl.DataFrame:
    """Return the store's temperature and wind per cell, run and lead, in the GEFS extract's units.

    Args:
        raw: The store's rows with `cell`, `init_time` and `lead_hours` added.

    Returns:
        `cell`, `init_time`, `lead_hours`, `temp_c`, `speed_100m`, `direction_100m`, `speed_10m`,
        `direction_10m` (m/s and degrees the wind blows from), with each NaN replaced by a null.
    """
    u100, v100 = pl.col("wind_u_100m").cast(pl.Float64), pl.col("wind_v_100m").cast(pl.Float64)
    u10, v10 = pl.col("wind_u_10m").cast(pl.Float64), pl.col("wind_v_10m").cast(pl.Float64)
    return raw.select(
        "cell",
        "init_time",
        "lead_hours",
        temp_c=pl.col("temperature_2m").cast(pl.Float64),
        speed_100m=(u100**2 + v100**2).sqrt(),
        direction_100m=pl.arctan2(-u100, -v100).degrees() % 360.0,
        speed_10m=(u10**2 + v10**2).sqrt(),
        direction_10m=pl.arctan2(-u10, -v10).degrees() % 360.0,
    ).with_columns(pl.col(pl.Float64).fill_nan(None))


def _gfs_native_direct_arm(
    *, keys: pl.DataFrame, extract: pl.DataFrame, domain: DomainType, day: int
) -> pl.DataFrame:
    """Read one lead day straight from the store's hourly leads, with no upsampling.

    Each target hour reads the run and lead `studies.gfs_native.served_init_time` and
    `served_lead_hours` give. A solar hour's radiation is the store's step mean at that lead, which
    is the mean over the hour ending at the label, and its temperature is read at the hour's
    midpoint, the mean of the instantaneous values at the hour's two ends (the ENS arms read
    temperature at the midpoint too). A wind hour reads the instantaneous values at its label.

    Args:
        keys: `site`, `time` for every row the study scores.
        extract: `_gfs_native_extract`'s hourly-lead frame.
        domain: `solar` or `wind`.
        day: The lead day, at most `HOURLY_SERVED_LAST_DAY`.

    Returns:
        `site`, `time` and the arm's columns; a row whose run or lead the store lacks carries nulls.
    """
    targets = keys.select("site", "time").with_columns(
        init_time=served_init_time(time=pl.col("time"), day=day, domain=domain),
        lead_hours=served_lead_hours(time=pl.col("time"), day=day, domain=domain),
    )
    columns = efh.ens_columns(arm=gfs_native_arm(day=day), domain=domain)
    join_keys = ["site", "init_time", "lead_hours"]
    if domain == "solar":
        earlier = extract.select(
            "site",
            "init_time",
            lead_hours=pl.col("lead_hours") + 1,
            temp_earlier="temp_c",
        )
        joined = targets.join(
            extract.select(*join_keys, "ghi_w_m2", "temp_c"), on=join_keys, how="left"
        ).join(earlier, on=join_keys, how="left")
        return joined.select(
            "site",
            "time",
            pl.col("ghi_w_m2").alias(columns[0]),
            ((pl.col("temp_c") + pl.col("temp_earlier")) / 2.0).alias(columns[1]),
        )
    joined = targets.join(
        extract.select(*join_keys, "speed_100m", "direction_100m", "speed_10m"),
        on=join_keys,
        how="left",
    )
    direction = pl.col("direction_100m").radians()
    return joined.select(
        "site",
        "time",
        pl.col("speed_100m").alias(columns[0]),
        direction.sin().alias(columns[1]),
        direction.cos().alias(columns[2]),
        pl.col("speed_10m").alias(columns[3]),
    )


def _gfs_native_frame(
    *,
    keys: pl.DataFrame,
    domain: DomainType,
    days: Sequence[int],
    gfs_dir: Path | None = None,
) -> pl.DataFrame:
    """Build the native GFS arms' columns from Dynamical.org's GFS store, at the given lead days.

    **Which run and lead each day serves.** Day 0 reads the freshest run at or before the instant
    each hour describes, one of the four runs a day (00, 06, 12, and 18 UTC), at a lead of 1 to 6
    hours for a solar hour (labelled by its end, so the window ends at the label) and 0 to 5 hours
    for a wind hour. Days 1 and above read the 00 UTC run issued that many days before the label's
    own day (a solar hour's own day is the day of the instant an hour before its label), at a lead
    of `24 * day + 1` to `24 * day + 24` for solar and `24 * day` to `24 * day + 23` for wind. The
    served rule is `studies.gfs_native.served_init_time` and `served_lead_hours`. The rule is the
    ENS and GEFS arms' (day `N` is the 00 UTC run's leads from `24 * N`), not the Previous Runs
    archive's freshest run at least `N` days old.

    **Radiation.** The store's value is the mean since the last 6-hourly reset, its lead labelling
    the window's end. `studies.gfs_native.step_means` inverts the windows to the mean over each
    step: exactly the hour before the label up to lead 120, and the 3 hours before it beyond (the
    store steps 3-hourly from lead 123). Days 0 to 4 read those hourly means directly. Days 5, 7, 10
    and 14 lie on the 3-hourly leads, so the same 3-hour means (the mean of three hourly steps to
    lead 120, the store's own step beyond it) go through the ENS and GEFS arms' upsampling to hourly
    (`clear_sky` for solar radiation, its straight-line temperature, `speed_components` for wind),
    which reads a one-member ensemble. That upsampling is the only difference between days 0 to 4
    and days 5 and above.

    **Space.** Each site reads its nearest 0.25 degree grid cell by great-circle distance, as the
    GEFS arms do (`nearest_cells`), not an H3 area mean. Wind is the speed and direction of the
    10 m and 100 m components; temperature is at 2 m.

    Args:
        keys: `site`, `time` for every row the study scores.
        domain: `solar` or `wind`.
        days: The lead days to build.
        gfs_dir: The store's folder; `None` reads `GFS_NATIVE_DIR_NAME` under the weather folder.

    Returns:
        `keys` with `gfs_native_day<N>_<field>` for every `N` in `days`, left-joined.

    Raises:
        RuntimeError: If any arm has a null in more than `GFS_NATIVE_MAX_MISSING_SHARE` of the rows.
    """
    directory = _weather_dir() / GFS_NATIVE_DIR_NAME if gfs_dir is None else gfs_dir
    sites = sorted(keys["site"].unique().to_list())
    cell_by_site = _gefs_cell_selection(
        grid_cells=pl.read_parquet(directory / "_grid_cells.parquet"), domain=domain, sites=sites
    )
    site_cells = pl.DataFrame(
        {"site": list(cell_by_site), "cell": list(cell_by_site.values())},
        schema={"site": pl.String, "cell": pl.Int32},
    )
    time_range = keys.select(first=pl.col("time").min(), last=pl.col("time").max()).row(
        0, named=True
    )
    time_dtype = keys.schema["time"]
    first_init = max(
        time_range["first"] - timedelta(days=max(days) + GFS_NATIVE_RUN_MARGIN_DAYS),
        efh.SPAN[0],
    )
    raw = (
        pl.scan_parquet(directory / "GFS.parquet")
        .with_columns(
            lead_hours=(pl.col("lead_time").dt.total_minutes() / 60).cast(pl.Int32),
            cell=(pl.col("lat_index") * 10 + pl.col("lon_index")).cast(pl.Int32),
            init_time=pl.col("init_time").dt.replace_time_zone("UTC").cast(time_dtype),
        )
        .filter(
            pl.col("cell").is_in(site_cells["cell"].unique().to_list()),
            pl.col("init_time") >= first_init,
            pl.col("init_time") <= time_range["last"],
        )
        .collect()
    )
    hourly_radiation, three_radiation = _gfs_native_step_radiation(
        radiation=raw.filter(pl.col("lead_hours") > 0).select(
            "cell",
            "init_time",
            "lead_hours",
            ghi_raw=pl.col("downward_short_wave_radiation_flux_surface")
            .cast(pl.Float64)
            .fill_nan(None),
        )
    )
    weather = _gfs_native_weather(raw=raw)
    cell_keys = ["cell", "init_time", "lead_hours"]
    frame = keys
    direct_days = [day for day in days if day <= HOURLY_SERVED_LAST_DAY]
    if direct_days:
        hourly_extract = weather.join(hourly_radiation, on=cell_keys, how="left").join(
            site_cells, on="cell"
        )
        for day in direct_days:
            frame = frame.join(
                _gfs_native_direct_arm(keys=keys, extract=hourly_extract, domain=domain, day=day),
                on=["site", "time"],
                how="left",
            )
    band_days = tuple(day for day in days if day > HOURLY_SERVED_LAST_DAY)
    if band_days:
        band_extract = (
            weather.filter(pl.col("lead_hours") % 3 == 0, pl.col("init_time").dt.hour() == 0)
            .join(three_radiation, on=cell_keys, how="left")
            .join(site_cells, on="cell")
            .with_columns(ensemble_member=pl.lit(0, dtype=pl.Int8))
        )
        for arm_frame in _ens_member_arms(
            extract=band_extract,
            domain=domain,
            days=band_days,
            method=UPSAMPLING_METHODS[domain],
            ensemble_size=1,
            arm_name=lambda _way, day: gfs_native_arm(day=day),
            ways=("mean",),
            fine_step_last_lead=LAST_LEAD_HOURS,
        ):
            frame = frame.join(arm_frame, on=["site", "time"], how="left")
    too_many = {}
    for day in days:
        columns = efh.ens_columns(arm=gfs_native_arm(day=day), domain=domain)
        in_span = frame.filter(
            served_init_time(time=pl.col("time"), day=day, domain=domain) >= first_init
        )
        share = float(
            in_span.select(pl.any_horizontal(pl.col(c).is_null() for c in columns).mean()).item()
        )
        _LOG.info(
            "%s: %s has a null in %.3f%% of the %d rows whose run is in the store's read span "
            "(%d earlier rows have no run by construction)",
            domain,
            gfs_native_arm(day=day),
            100 * share,
            in_span.height,
            frame.height - in_span.height,
        )
        if share > GFS_NATIVE_MAX_MISSING_SHARE:
            too_many[gfs_native_arm(day=day)] = share
    if too_many:
        msg = (
            f"{domain}: native GFS arms with a null in more than "
            f"{GFS_NATIVE_MAX_MISSING_SHARE:.1%} of rows: {too_many}"
        )
        raise RuntimeError(msg)
    return frame


IFS_SINGLE_DIR_NAME: Final[str] = "ECMWF-IFS-SINGLE-RUNS"
"""Under `data/studies/weather/`, Open-Meteo's Single Runs archive of ECMWF IFS HRES."""

IFS_SINGLE_FILE_NAME: Final[str] = "ECMWF-IFS-SINGLE-RUNS.parquet"
"""Inside `IFS_SINGLE_DIR_NAME`, the combined file: one 00 UTC run a day, hourly leads 0 to 240."""

IFS_SINGLE_ARM_PREFIX: Final[str] = "ifs_single"
"""The column prefix of the IFS HRES (9 km, Open-Meteo) arms, `ifs_single_day<N>_<field>`."""

IFS_SINGLE_MAX_MISSING_SHARE: Final[float] = 0.015
"""The largest share of rows with a null in one IFS HRES (9 km, Open-Meteo) arm's columns the build
accepts. The build measures it over every published row (about 1% at most), whereas
`fit_extra_leads.MAX_MISSING_SHARE`, the same limit, is measured over the shared rows only, where
the seven gap run days fall on long summer days and reach 1.45% for solar day 0. The nulls are the
days whose run the archive lacks (each arm loses the target days those runs would serve)."""


def ifs_single_arm(*, day: int) -> str:
    """Return the column prefix of the IFS HRES (9 km, Open-Meteo) arm at one lead day."""
    return f"{IFS_SINGLE_ARM_PREFIX}_day{day}"


def _ifs_single_extract(
    *, raw: pl.DataFrame, domain: DomainType, time_dtype: pl.DataType
) -> pl.DataFrame:
    """Turn the archive's rows into the fields an arm reads, per site, run and lead.

    Args:
        raw: The archive's rows: `site`, `init_time`, `lead_hours`, and the weather columns, with
            the times timezone-naive UTC.
        domain: `solar` or `wind`.
        time_dtype: The dtype of the study's `time` column, which `init_time` is cast to.

    Returns:
        `site`, `init_time`, `lead_hours` and, for solar, `ghi` (clipped at zero) and `temp`, or
        for wind, `speed_100m` and `speed_10m` in m/s and `sin_100m` and `cos_100m` of the 100 m
        direction, as `_wind_columns` builds them from the Previous Runs archive.
    """
    init_time = pl.col("init_time").dt.replace_time_zone("UTC").cast(time_dtype)
    if domain == "solar":
        return raw.select(
            "site",
            init_time=init_time,
            lead_hours=pl.col("lead_hours"),
            ghi=clip_radiation(radiation=pl.col("shortwave_radiation")),
            temp=pl.col("temperature_2m"),
        )
    direction = pl.col("wind_direction_100m").radians()
    return raw.select(
        "site",
        init_time=init_time,
        lead_hours=pl.col("lead_hours"),
        speed_100m=pl.col("wind_speed_100m") * KMH_TO_MS,
        sin_100m=direction.sin(),
        cos_100m=direction.cos(),
        speed_10m=pl.col("wind_speed_10m") * KMH_TO_MS,
    )


def _ifs_single_arm_columns(
    *, keys: pl.DataFrame, extract: pl.DataFrame, domain: DomainType, day: int
) -> pl.DataFrame:
    """Read one lead day for every target hour from the run and lead its rule serves.

    Args:
        keys: `site`, `time` for every row the study scores.
        extract: `_ifs_single_extract`'s result.
        domain: `solar` or `wind`.
        day: The lead day, at most `last_servable_day`.

    Returns:
        `site`, `time` and the arm's columns; a target hour whose run the archive lacks carries
        nulls and keeps its row, which `fit_extra_leads` drops from this arm's fit and score.

    Raises:
        ValueError: If `day` is beyond `last_servable_day`.
    """
    if day > last_servable_day(domain=domain):
        msg = f"day {day} is beyond the archive's 240-hour runs for {domain}"
        raise ValueError(msg)
    targets = keys.select("site", "time").with_columns(
        init_time=ifs_single_init_time(time=pl.col("time"), day=day, domain=domain),
        lead_hours=ifs_single_lead_hours(time=pl.col("time"), day=day, domain=domain),
    )
    columns = efh.ens_columns(arm=ifs_single_arm(day=day), domain=domain)
    fields = efh.fields(domain=domain)
    joined = targets.join(extract, on=["site", "init_time", "lead_hours"], how="left")
    return joined.select(
        "site",
        "time",
        *(pl.col(field).alias(column) for field, column in zip(fields, columns, strict=True)),
    )


def _ifs_single_frame(
    *,
    keys: pl.DataFrame,
    domain: DomainType,
    days: Sequence[int],
    ifs_single_dir: Path | None = None,
) -> pl.DataFrame:
    """Build the IFS HRES (9 km, Open-Meteo) arms' columns from Open-Meteo's Single Runs archive.

    **Which run and lead each day serves.** Day `N` reads the 00 UTC run issued `N` days before the
    hour's own day (a solar hour's own day is the day of the instant an hour before its label), at a
    lead of `24 * N + 1` to `24 * N + 24` for solar and `24 * N` to `24 * N + 23` for wind, the ENS
    and GEFS arms' rule (`studies.ifs_single_runs.served_init_time`). Day 0 is the run of the
    hour's own day, so it is the 00 UTC run and not the freshest run. Day 0 covers
    hours before the 00 UTC run is published, as ENS's day 0 does, so day 0 is not a forecast that
    could have been used in advance for those hours.

    **The values.** Every value is used as the archive serves it, with no upsampling, because the
    archive is already hourly. Radiation is the mean over the hour ending at the label, clipped at
    zero, and temperature is read at the label, as the Previous Runs IFS 0.25 degree arm reads
    them. Wind is the speed and the sine and cosine of the direction at 100 m and the speed at 10 m,
    as `_wind_columns` builds the Previous Runs arms' wind, not `speed_components`, because the
    archive serves speed and direction and no components.

    **Gap days.** The archive lacks whole runs (four run days, and three whose runs were
    incomplete and are absent). A target hour whose serving run is absent carries nulls in the arm's
    columns, and never a value from another run.

    Args:
        keys: `site`, `time` for every row the study scores.
        domain: `solar` or `wind`.
        days: The lead days to build.
        ifs_single_dir: The archive's folder; `None` reads `IFS_SINGLE_DIR_NAME` under the weather
            folder.

    Returns:
        `keys` with `ifs_single_day<N>_<field>` for every `N` in `days`, left-joined.

    Raises:
        RuntimeError: If any arm has a null in more than `IFS_SINGLE_MAX_MISSING_SHARE` of the rows.
    """
    directory = _weather_dir() / IFS_SINGLE_DIR_NAME if ifs_single_dir is None else ifs_single_dir
    time_dtype = keys.schema["time"]
    time_range = keys.select(first=pl.col("time").min(), last=pl.col("time").max()).row(
        0, named=True
    )
    first_init = time_range["first"] - timedelta(days=max(days) + 1)
    raw = (
        pl.scan_parquet(directory / IFS_SINGLE_FILE_NAME)
        .filter(
            pl.col("site").is_in(keys["site"].unique().to_list()),
            pl.col("init_time").dt.replace_time_zone("UTC").cast(time_dtype) >= first_init,
            pl.col("init_time").dt.replace_time_zone("UTC").cast(time_dtype) <= time_range["last"],
        )
        .collect()
    )
    extract = _ifs_single_extract(raw=raw, domain=domain, time_dtype=time_dtype)
    frame = keys
    too_many = {}
    for day in days:
        arm = _ifs_single_arm_columns(keys=keys, extract=extract, domain=domain, day=day)
        frame = frame.join(arm, on=["site", "time"], how="left")
        columns = efh.ens_columns(arm=ifs_single_arm(day=day), domain=domain)
        share = float(
            frame.select(pl.any_horizontal(pl.col(c).is_null() for c in columns).mean()).item()
        )
        _LOG.info(
            "%s: %s has a null in %.3f%% of rows", domain, ifs_single_arm(day=day), 100 * share
        )
        if share > IFS_SINGLE_MAX_MISSING_SHARE:
            too_many[ifs_single_arm(day=day)] = share
    if too_many:
        msg = (
            f"{domain}: IFS HRES (9 km, Open-Meteo) arms with a null in more than "
            f"{IFS_SINGLE_MAX_MISSING_SHARE:.1%} of rows: {too_many}"
        )
        raise RuntimeError(msg)
    return frame


def build_domain(*, domain: DomainType, output_dir: Path, gefs_window_dir: Path | None) -> Path:
    """Build one technology's arm-input parquet and write it under `output_dir`.

    Args:
        domain: `solar` or `wind`.
        output_dir: Where `<domain>_forecast_inputs.parquet` is written.
        gefs_window_dir: A `GEFS_window_*` test extract, or `None`.

    Returns:
        The written file's path.
    """
    base = efh.base_frame(domain=domain)
    reference_columns = (
        ["ghi_era5", "ghi_cams", "temp_c", "solar_elevation_deg", "solar_azimuth_deg"]
        if domain == "solar"
        else ["speed_hub_era5", "speed_10m_era5"]
    )
    keys = base.select(
        "site",
        "time",
        "power_mw",
        "cap_mw",
        "constrained",
        "effective_capacity_mw",
        "hour_of_day",
        "day_of_year",
        *[column for column in reference_columns if column in base.columns],
    )
    frame = _previous_runs_frame(keys=keys, domain=domain)
    frame = _ens_frame(keys=frame, domain=domain)
    frame = _gefs_frame(keys=frame, domain=domain, window_dir=gefs_window_dir)
    output_path = output_dir / f"{domain}_forecast_inputs.parquet"
    frame.write_parquet(output_path)
    _LOG.info("%s: wrote %d rows, %d columns to %s", domain, frame.height, frame.width, output_path)
    return output_path


def _ens_extra_frame(
    *,
    keys: pl.DataFrame,
    domain: DomainType,
    mean_days: tuple[int, ...],
    control_days: tuple[int, ...],
) -> pl.DataFrame:
    """Build ECMWF ENS's mean and control-member columns at the given days on `keys`.

    Args:
        keys: `site`, `time` for every row the study scores.
        domain: `solar` or `wind`.
        mean_days: The days to build the ENS mean at.
        control_days: The days to build the ENS control member at.

    Returns:
        `keys` with `ens_mean_day<N>_<field>` for every `N` in `mean_days` and
        `ens_control_day<N>_<field>` for every `N` in `control_days`, left-joined. A row missing a
        band carries nulls. `keys` unchanged if both are empty.
    """
    if not (mean_days or control_days):
        return keys
    sites = sorted(keys["site"].unique().to_list())
    extract = efh.members(sites=sites)
    arms: list[pl.DataFrame] = []
    for day in sorted({*mean_days, *control_days}):
        arms += _ens_member_arms(
            extract=extract,
            domain=domain,
            days=(day,),
            method=UPSAMPLING_METHODS[domain],
            ensemble_size=efh.ENSEMBLE_SIZE,
            arm_name=lambda way, band: efh.ens_arm(way=way, day=band),
            ways=extra_ens_ways(day=day, mean_days=mean_days, control_days=control_days),
        )
    frame = keys
    for arm_frame in arms:
        frame = frame.join(arm_frame, on=["site", "time"], how="left")
    return frame


def extra_ens_ways(
    *, day: int, mean_days: tuple[int, ...], control_days: tuple[int, ...]
) -> tuple[str, ...]:
    """Return which ENS reductions one day's band is built for.

    Args:
        day: The band.
        mean_days: The days the ENS mean is wanted at.
        control_days: The days the ENS control member is wanted at.

    Returns:
        `"mean"` if `day` is in `mean_days`, then `"control"` if it is in `control_days`.
    """
    return tuple(
        way for way, days in (("mean", mean_days), ("control", control_days)) if day in days
    )


def build_extra_leads(
    *,
    domain: DomainType,
    published_dir: Path,
    output_dir: Path,
    gefs_window_dir: Path | None,
    batch: ExtraBatchType = "first",
    gfs_dir: Path | None = None,
    ifs_single_dir: Path | None = None,
) -> Path:
    """Build the exploratory lead columns on the published inputs' own `(site, time)` keys.

    The published `<domain>_forecast_inputs.parquet` is read for its keys only, so the new columns
    sit on exactly the rows the published study scored, whatever has changed on disk since.
    Nothing is written to `published_dir`, and the output is write-once.

    Args:
        domain: `solar` or `wind`.
        published_dir: The folder holding the published `<domain>_forecast_inputs.parquet`.
        output_dir: The new folder to write `<domain>_extra_lead_inputs.parquet` into.
        gefs_window_dir: A `GEFS_window_*` test extract, or `None` for the month cache.
        batch: Which extra-lead build (`EXTRA_LEAD_BUILDS`).
        gfs_dir: The native GFS store's folder for the third batch; `None` reads the shared one.
        ifs_single_dir: The IFS HRES archive's folder for the fourth batch; `None` reads the
            shared one.

    Returns:
        The written file's path.

    Raises:
        ValueError: If `output_dir` is `published_dir`.
        FileExistsError: If the output file already exists.
    """
    if output_dir.resolve() == published_dir.resolve():
        msg = f"the extra-lead output must not be the published folder {published_dir}"
        raise ValueError(msg)
    output_path = output_dir / f"{domain}_extra_lead_inputs.parquet"
    if output_path.exists():
        msg = f"{output_path} exists; the extra-lead inputs are write-once, move it first"
        raise FileExistsError(msg)
    keys = pl.read_parquet(published_dir / f"{domain}_forecast_inputs.parquet").select(
        "site", "time"
    )
    build = EXTRA_LEAD_BUILDS[batch]
    if max(gefs_band_leads(days=build.gefs_days), default=0) > GEFS_STEP_MEAN_MAX_LEAD_HOURS:
        last_month = keys.select(pl.col("time").max().dt.strftime("%Y-%m")).item()
        cache_files = list(_gefs_months_available(last_month=last_month).values())
        failures = [
            *gefs_window_verdict(table=gefs_window_table(files=cache_files)),
            *gefs_boundary_verdict(table=gefs_boundary_table(files=cache_files)),
        ]
        if failures:
            msg = f"GEFS beyond 240 h is not a 6-hour window mean: {failures}"
            raise RuntimeError(msg)
    frame = (
        _previous_runs_frame(keys=keys, domain=domain, day_offsets=build.product_day_offsets)
        if build.product_day_offsets
        else keys
    )
    frame = _ens_extra_frame(
        keys=frame,
        domain=domain,
        mean_days=build.ens_mean_days,
        control_days=build.ens_control_days,
    )
    if build.gefs_days:
        frame = _gefs_frame(
            keys=frame, domain=domain, window_dir=gefs_window_dir, days=build.gefs_days
        )
    if build.gfs_native_days:
        frame = _gfs_native_frame(
            keys=frame, domain=domain, days=build.gfs_native_days, gfs_dir=gfs_dir
        )
    if build.ifs_single_days:
        frame = _ifs_single_frame(
            keys=frame,
            domain=domain,
            days=build.ifs_single_days,
            ifs_single_dir=ifs_single_dir,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path)
    _LOG.info("%s: wrote %d rows, %d columns to %s", domain, frame.height, frame.width, output_path)
    return output_path


def main() -> int:
    """Build the solar and wind arm-input frames and write them under `--output-dir`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_data_dir() / "studies" / DEFAULT_OUTPUT_DIR_NAME,
        help="Directory the two arm-input parquets are written to.",
    )
    parser.add_argument(
        "--extra-leads",
        action="store_true",
        help="Build the exploratory lead columns (ENS at days 5, 10 and 14, GEFS at days 0, 5, 10 "
        "and 14, Previous Runs day 0 for every product, day 5, and day 7 for IFS 0.25° and GFS) on "
        "the published inputs' keys, into a new --output-dir.",
    )
    parser.add_argument(
        "--batch",
        choices=tuple(EXTRA_LEAD_BUILDS),
        default="first",
        help="With --extra-leads: which extra-lead build (the second adds ENS mean at day 7, the "
        "ENS control member at days 5, 7, 10 and 14, and GEFS mean at day 7; the third adds the "
        "native GFS store at days 0, 1, 2, 3, 5, 7, 10 and 14; the fourth adds Open-Meteo's IFS "
        "Single Runs archive at days 0, 1, 2, 3, 5 and 7).",
    )
    parser.add_argument(
        "--gfs-dir",
        type=Path,
        default=None,
        help="With --batch third: the native GFS store's folder, for development; production "
        "reads the shared one.",
    )
    parser.add_argument(
        "--ifs-single-dir",
        type=Path,
        default=None,
        help="With --batch fourth: the IFS HRES archive's folder, for development; "
        "production reads the shared one.",
    )
    parser.add_argument(
        "--published-dir",
        type=Path,
        default=_repo_data_dir() / "studies" / DEFAULT_OUTPUT_DIR_NAME,
        help="With --extra-leads: the folder holding the published arm-input parquets.",
    )
    parser.add_argument(
        "--gefs-window-dir",
        type=Path,
        default=None,
        help="A GEFS_window_* test extract, for development only; production reads the month "
        "cache once it is complete.",
    )
    args = parser.parse_args()
    if args.extra_leads:
        for domain in ("solar", "wind"):
            build_extra_leads(
                domain=domain,
                published_dir=args.published_dir,
                output_dir=args.output_dir,
                gefs_window_dir=args.gefs_window_dir,
                batch=args.batch,
                gfs_dir=args.gfs_dir,
                ifs_single_dir=args.ifs_single_dir,
            )
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for domain in ("solar", "wind"):
        build_domain(
            domain=domain, output_dir=args.output_dir, gefs_window_dir=args.gefs_window_dir
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
