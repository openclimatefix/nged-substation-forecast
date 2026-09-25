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

With `--extra-leads` the script instead builds the exploratory lead columns (ENS at days 5 and 14,
GEFS at days 5, 10 and 14, Previous Runs at day 0 for ICON-D2 and ICON-EU and at day 5 for ICON
global, IFS 0.25° and GFS) on the published inputs' own `(site, time)` keys, into a new write-once
`--output-dir`, and never writes to the published folder. The new days are separate constants
(`EXTRA_ENS_DAYS`, `EXTRA_GEFS_DAYS`), never added to `ENS_DAYS`, so the shared rows cannot move.

With `--aifs` the script instead builds the AIFS inputs (issue #923) on the published inputs' own
`(site, time)` keys, into a new write-once `--output-dir`: ECMWF AIFS Single and AIFS ENS at days 1
and 2, and ENS's mean and control member on the same 6-hourly steps as the AIFS reads. AIFS is read
as ENS is: the 00 UTC run of day `D - d` for a target hour on day `D`, so the lead is `24d + h`.
Each site's value is the H3 resolution-5 overlap-weighted mean of the crop's 0.25 degree cells, the
read ENS's stored table has, plus (AIFS Single only) a nearest-cell arm for the spatial-read
sensitivity. Every AIFS arm also carries its run's `init_time`, which `fit_aifs.py` checks against
each AIFS version era.

Every output row carries only the anonymised `site` label; no generator name, id or coordinate is
read from the private roster in this script, except inside `studies.grid_sampling` (GEFS's and
AIFS's nearest-cell match) and the H3 cell lookup of `_aifs_site_weights`, which never print what
they read.

Run it with `uv run python studies/nwp_forecast_comparison/build_forecast_inputs.py --output-dir
DIR`.
"""

import argparse
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

import h3.api.basic_int as h3
import numpy as np
import polars as pl
from contracts.settings import PROJECT_ROOT
from geo.h3 import compute_h3_grid_weights

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
from fetch_ens_forecast_horizons import H3_RESOLUTION
from studies.grid_sampling import nearest_cells
from studies.guards import refuse_to_overwrite
from studies.hourly_means import hourly_from_snapshots
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
    "ICON-D2": (0,),
    "ICON-EU": (0,),
    "ICON global": (5,),
    "IFS 0.25°": (5,),
    "GFS": (5,),
}
"""The exploratory `previous_dayN` offsets the extra-lead build reads on top of
`PRODUCT_DAY_OFFSETS`. Day 0 is Open-Meteo's unsuffixed series, the freshest run that covers each
hour; ICON-EU's archive ends at day 4, and ICON-D2's holds only days 0 and 1."""

EXTRA_ENS_DAYS: Final[tuple[int, ...]] = (5, 14)
"""The ENS bands the extra-lead build adds. They never join `ENS_DAYS`, which also decides the
baseline columns every shared row must hold, so adding them there would move the shared rows."""

EXTRA_GEFS_DAYS: Final[tuple[int, ...]] = (5, 10, 14)
"""The GEFS bands the extra-lead build adds."""

AIFS_DAYS: Final[tuple[int, ...]] = (1, 2)
"""The bands the AIFS build reads: day 1 (the day-ahead product) and day 2."""

AIFS_SINGLE_DIR_NAME: Final[str] = "ECMWF-AIFS"
AIFS_ENS_DIR_NAME: Final[str] = "ECMWF-AIFS-ENS"
"""The two AIFS downloads' folders under `data/studies/weather/`. Each holds `<name>.parquet` and
`_grid_cells.parquet`."""

AIFS_SINGLE_FIRST_INIT: Final[datetime] = datetime(2025, 2, 26, tzinfo=UTC)
"""The first 00 UTC AIFS Single run read: the first after v1.0 went operational (2025-02-25 06 UTC).
The store's radiation and 100 m wind are `NaN` before the 2025-02-24 06 UTC run."""

AIFS_ENS_FIRST_INIT: Final[datetime] = datetime(2025, 7, 2, tzinfo=UTC)
"""The first 00 UTC AIFS ENS run in the store."""

AIFS_CROP_DEGREES: Final[float] = 0.25
"""The AIFS grid's cell size, as `compute_h3_grid_weights` bins by."""

AIFS_WEIGHT_TOLERANCE: Final[float] = 1e-6
"""How far a site's H3 weights over the crop may sit from summing to 1."""

AIFS_VALUE_COLUMNS: Final[tuple[str, ...]] = (
    "downward_short_wave_radiation_flux_surface",
    "temperature_2m",
    "wind_u_10m",
    "wind_v_10m",
    "wind_u_100m",
    "wind_v_100m",
)
"""The store columns the AIFS extract reads."""

SpatialReadType = Literal["h3", "nearest"]

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
    six_hourly: bool = False,
    keep_init_time: bool = False,
) -> list[pl.DataFrame]:
    """Upsample, combine and reduce one ensemble's members at several bands and ways.

    Shared by `_ens_frame` (ENS, both ways), `_gefs_frame` (GEFS, mean only) and `_aifs_frame`
    (AIFS and ENS on 6-hourly steps), whose only difference is the extract they read, its ensemble
    size, and the arm-name prefix.

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
        fine_step_last_lead: The last lead on 3-hour steps: 144 for ENS, 240 for GEFS, 0 for a
            product with 6-hourly steps throughout (AIFS).
        six_hourly: Whether to emulate 6-hourly steps from a 3-hourly extract (ENS at AIFS's steps).
        keep_init_time: Whether each frame also carries the run that fed each hour, as
            `<arm>_init_time`.

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
            six_hourly=six_hourly,
        )
        upsampled = efh.upsampled_fields(steps=steps, day=day, domain=domain, clear_sky=clear_sky)
        combined = efh.combine(
            steps=steps, upsampled=upsampled, day=day, domain=domain, method=method
        )
        for way in ways:
            reduced = efh.reduce_members(
                hourly=combined, domain=domain, way=way, ensemble_size=ensemble_size
            )
            arm = arm_name(way, day)
            arm_frame = efh.prefixed(frame=reduced, arm=arm, domain=domain)
            if keep_init_time:
                runs = combined.select("site", "time", **{f"{arm}_init_time": "init_time"}).unique(
                    subset=["site", "time"]
                )
                arm_frame = arm_frame.join(runs, on=["site", "time"], how="left")
            frames.append(arm_frame)
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


def _ens_extra_frame(*, keys: pl.DataFrame, domain: DomainType) -> pl.DataFrame:
    """Build ECMWF ENS's mean columns at `EXTRA_ENS_DAYS` on `keys`.

    Args:
        keys: `site`, `time` for every row the study scores.
        domain: `solar` or `wind`.

    Returns:
        `keys` with `ens_mean_day<N>_<field>` for every `N` in `EXTRA_ENS_DAYS`, left-joined. A
        row missing a band carries nulls.
    """
    sites = sorted(keys["site"].unique().to_list())
    arms = _ens_member_arms(
        extract=efh.members(sites=sites),
        domain=domain,
        days=EXTRA_ENS_DAYS,
        method=UPSAMPLING_METHODS[domain],
        ensemble_size=efh.ENSEMBLE_SIZE,
        arm_name=lambda way, day: efh.ens_arm(way=way, day=day),
        ways=("mean",),
    )
    frame = keys
    for arm_frame in arms:
        frame = frame.join(arm_frame, on=["site", "time"], how="left")
    return frame


def build_extra_leads(
    *, domain: DomainType, published_dir: Path, output_dir: Path, gefs_window_dir: Path | None
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
    last_month = keys.select(pl.col("time").max().dt.strftime("%Y-%m")).item()
    cache_files = list(_gefs_months_available(last_month=last_month).values())
    failures = [
        *gefs_window_verdict(table=gefs_window_table(files=cache_files)),
        *gefs_boundary_verdict(table=gefs_boundary_table(files=cache_files)),
    ]
    if failures:
        msg = f"GEFS beyond 240 h is not a 6-hour window mean: {failures}"
        raise RuntimeError(msg)
    frame = _previous_runs_frame(keys=keys, domain=domain, day_offsets=EXTRA_PRODUCT_DAY_OFFSETS)
    frame = _ens_extra_frame(keys=frame, domain=domain)
    frame = _gefs_frame(keys=frame, domain=domain, window_dir=gefs_window_dir, days=EXTRA_GEFS_DAYS)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path)
    _LOG.info("%s: wrote %d rows, %d columns to %s", domain, frame.height, frame.width, output_path)
    return output_path


def _h3_crop_weights(*, site_cells: Mapping[str, int], grid_cells: pl.DataFrame) -> pl.DataFrame:
    """Return each site's H3 area weights over the AIFS crop's grid cells.

    Args:
        site_cells: Each site to the H3 resolution-5 cell it sits in.
        grid_cells: The crop's `lat_index`, `lon_index`, `latitude` and `longitude`.

    Returns:
        One row per (site, crop cell the site's H3 cell overlaps), with `site`, `lat_index`,
        `lon_index` and `weight`.

    Raises:
        ValueError: If a site's weights over the crop do not sum to 1, which means the H3 cell
            reaches beyond the crop.
    """
    h3_weights = compute_h3_grid_weights(
        nwp_grid_size_degrees=AIFS_CROP_DEGREES, h3_index=sorted(set(site_cells.values()))
    )
    cells = grid_cells.select(
        "lat_index",
        "lon_index",
        nwp_lat=pl.col("latitude").round(4),
        nwp_lon=pl.col("longitude").round(4),
    )
    by_cell = h3_weights.select(
        "h3_index",
        "proportion",
        nwp_lat=pl.col("nwp_lat").round(4),
        nwp_lon=pl.col("nwp_lon").round(4),
    ).join(cells, on=["nwp_lat", "nwp_lon"])
    sites = pl.DataFrame(
        {"site": list(site_cells), "h3_index": list(site_cells.values())},
        schema={"site": pl.String, "h3_index": pl.UInt64},
    )
    weights = sites.join(by_cell, on="h3_index").select(
        "site", "lat_index", "lon_index", weight=pl.col("proportion").cast(pl.Float64)
    )
    sums = weights.group_by("site").agg(total=pl.col("weight").sum())
    bad = set(site_cells) - set(
        sums.filter((pl.col("total") - 1.0).abs() <= AIFS_WEIGHT_TOLERANCE)["site"].to_list()
    )
    if bad:
        msg = f"H3 weights over the AIFS crop do not sum to 1 for {len(bad)} sites"
        raise ValueError(msg)
    return weights


def _aifs_site_weights(
    *, path: Path, domain: DomainType, sites: list[str], spatial: SpatialReadType
) -> pl.DataFrame:
    """Return each site's cell weights over one AIFS download's crop.

    Args:
        path: The download's directory, holding `_grid_cells.parquet`.
        domain: `solar` or `wind`.
        sites: The sites to read.
        spatial: `h3` for the overlap-weighted mean of the cells under the site's H3 resolution-5
            cell (the read ENS's stored table has), or `nearest` for the one nearest cell.

    Returns:
        `site`, `lat_index`, `lon_index` and `weight`. No coordinate or cell id is printed.
    """
    grid_cells = pl.read_parquet(path / "_grid_cells.parquet")
    if spatial == "nearest":
        nearest = _gefs_cell_selection(grid_cells=grid_cells, domain=domain, sites=sites)
        return pl.DataFrame(
            {
                "site": list(nearest),
                "lat_index": [cell // 10 for cell in nearest.values()],
                "lon_index": [cell % 10 for cell in nearest.values()],
                "weight": [1.0] * len(nearest),
            },
            schema={
                "site": pl.String,
                "lat_index": pl.Int16,
                "lon_index": pl.Int16,
                "weight": pl.Float64,
            },
        )
    roster = efh.site_roster(domain=domain).filter(pl.col("site").is_in(sites))
    site_cells = {
        site: h3.latlng_to_cell(latitude, longitude, H3_RESOLUTION)
        for site, latitude, longitude in roster.iter_rows()
    }
    return _h3_crop_weights(site_cells=site_cells, grid_cells=grid_cells)


def _aifs_members_frame(
    *,
    store: Path,
    weights: pl.DataFrame,
    ensemble: bool,
    first_init: datetime,
    max_lead_hours: int = 24 * max(AIFS_DAYS) + 30,
) -> pl.DataFrame:
    """Read one AIFS store's 00 UTC runs and reshape them to `band_steps`'s input shape.

    A sibling of `_gefs_members_frame`. Each site's value is the weighted mean of the crop cells in
    `weights`, taken on the wind components and then turned into speed and from-direction, as
    `dynamical_data`'s H3 aggregation does. The scan is lazy because the AIFS ENS file holds 50
    million rows.

    Args:
        store: The store's parquet.
        weights: `_aifs_site_weights`'s result.
        ensemble: Whether the store has an `ensemble_member` column (AIFS ENS). AIFS Single gets
            member 0.
        first_init: The first run kept; earlier runs hold `NaN` radiation and 100 m wind.
        max_lead_hours: The longest lead kept.

    Returns:
        One row per (site, init_time, ensemble_member, lead_hours), with `ghi_w_m2` (null at lead
        0), `temp_c`, `speed_100m`, `direction_100m`, `speed_10m` and `direction_10m`.

    Raises:
        ValueError: If any value the study reads is `NaN` after the run filter.
    """
    scan = pl.scan_parquet(store).filter(
        pl.col("init_time").dt.hour() == 0,
        pl.col("init_time") >= first_init.replace(tzinfo=None),
        pl.col("lead_time") <= pl.duration(hours=max_lead_hours),
    )
    if not ensemble:
        scan = scan.with_columns(ensemble_member=pl.lit(0, dtype=pl.Int8))
    keys = ["site", "init_time", "ensemble_member", "lead_time"]
    weighted = (
        scan.join(weights.lazy(), on=["lat_index", "lon_index"])
        .group_by(keys)
        .agg(
            ((pl.col(column) * pl.col("weight")).sum() / pl.col("weight").sum()).alias(column)
            for column in AIFS_VALUE_COLUMNS
        )
        .collect()
    )
    lead_hours = pl.col("lead_time").dt.total_hours().cast(pl.Int32)
    frame = weighted.select(
        "site",
        init_time=pl.col("init_time").dt.replace_time_zone("UTC"),
        ensemble_member=pl.col("ensemble_member").cast(pl.Int8),
        lead_hours=lead_hours,
        ghi_w_m2=pl.when(lead_hours > 0)
        .then(pl.col("downward_short_wave_radiation_flux_surface"))
        .cast(pl.Float64),
        temp_c=pl.col("temperature_2m").cast(pl.Float32),
        speed_100m=(pl.col("wind_u_100m") ** 2 + pl.col("wind_v_100m") ** 2)
        .sqrt()
        .cast(pl.Float64),
        direction_100m=pl.arctan2(-pl.col("wind_u_100m"), -pl.col("wind_v_100m")).degrees() % 360.0,
        speed_10m=(pl.col("wind_u_10m") ** 2 + pl.col("wind_v_10m") ** 2).sqrt().cast(pl.Float64),
        direction_10m=pl.arctan2(-pl.col("wind_u_10m"), -pl.col("wind_v_10m")).degrees() % 360.0,
    )
    not_a_number = {
        column: int(frame[column].is_nan().sum())
        for column in ("ghi_w_m2", "temp_c", "speed_100m", "speed_10m")
        if frame[column].is_nan().any()
    }
    if not_a_number:
        msg = f"{store.name}: NaN after the run filter: {not_a_number}"
        raise ValueError(msg)
    return frame


def _aifs_frame(*, keys: pl.DataFrame, domain: DomainType, weather_dir: Path) -> pl.DataFrame:
    """Build the AIFS arms and their like-for-like ENS references on `keys`.

    Args:
        keys: `site`, `time` for every row the study might score.
        domain: `solar` or `wind`.
        weather_dir: The folder holding the two AIFS downloads' directories.

    Returns:
        `keys` with `aifs_single_day<N>`, `aifs_ens_mean_day<N>`, `ens_mean6_day<N>` and
        `ens_control6_day<N>` columns for every `N` in `AIFS_DAYS`, `aifs_single_nearest_day1`, and
        `<arm>_init_time` for each AIFS arm, left-joined.

    Raises:
        ValueError: If a written column is not one of those.
    """
    sites = sorted(keys["site"].unique().to_list())
    method = UPSAMPLING_METHODS[domain]
    single_dir = weather_dir / AIFS_SINGLE_DIR_NAME
    ens_dir = weather_dir / AIFS_ENS_DIR_NAME
    arm_frames: list[pl.DataFrame] = []
    for spatial, days, name in (
        ("h3", AIFS_DAYS, "aifs_single"),
        ("nearest", (1,), "aifs_single_nearest"),
    ):
        extract = _aifs_members_frame(
            store=single_dir / f"{AIFS_SINGLE_DIR_NAME}.parquet",
            weights=_aifs_site_weights(
                path=single_dir, domain=domain, sites=sites, spatial=spatial
            ),
            ensemble=False,
            first_init=AIFS_SINGLE_FIRST_INIT,
        )
        arm_frames += _ens_member_arms(
            extract=extract,
            domain=domain,
            days=days,
            method=method,
            ensemble_size=1,
            arm_name=lambda way, day, name=name: f"{name}_day{day}",
            ways=("control",),
            fine_step_last_lead=0,
            keep_init_time=True,
        )
    ens_extract = _aifs_members_frame(
        store=ens_dir / f"{AIFS_ENS_DIR_NAME}.parquet",
        weights=_aifs_site_weights(path=ens_dir, domain=domain, sites=sites, spatial="h3"),
        ensemble=True,
        first_init=AIFS_ENS_FIRST_INIT,
    )
    arm_frames += _ens_member_arms(
        extract=ens_extract,
        domain=domain,
        days=AIFS_DAYS,
        method=method,
        ensemble_size=efh.ENSEMBLE_SIZE,
        arm_name=lambda way, day: f"aifs_ens_{way}_day{day}",
        ways=("mean",),
        fine_step_last_lead=0,
        keep_init_time=True,
    )
    arm_frames += _ens_member_arms(
        extract=efh.members(sites=sites),
        domain=domain,
        days=AIFS_DAYS,
        method=method,
        ensemble_size=efh.ENSEMBLE_SIZE,
        arm_name=lambda way, day: f"ens_{way}6_day{day}",
        six_hourly=True,
    )
    frame = keys
    for arm_frame in arm_frames:
        frame = frame.join(arm_frame, on=["site", "time"], how="left")
    allowed = ("aifs_single_day", "aifs_single_nearest_day", "aifs_ens_mean_day", "ens_mean6_day")
    allowed += ("ens_control6_day",)
    unexpected = [
        column for column in frame.columns[keys.width :] if not column.startswith(allowed)
    ]
    if unexpected:
        msg = f"unexpected columns in the AIFS inputs: {unexpected}"
        raise ValueError(msg)
    return frame


def build_aifs(
    *, domain: DomainType, published_dir: Path, output_dir: Path, weather_dir: Path
) -> Path:
    """Build the AIFS columns on the published inputs' own `(site, time)` keys.

    Args:
        domain: `solar` or `wind`.
        published_dir: The folder holding the published `<domain>_forecast_inputs.parquet`.
        output_dir: The new folder to write `<domain>_aifs_inputs.parquet` into.
        weather_dir: The folder holding the two AIFS downloads' directories.

    Returns:
        The written file's path.

    Raises:
        ValueError: If `output_dir` is `published_dir`.
        FileExistsError: If the output file already exists.
    """
    if output_dir.resolve() == published_dir.resolve():
        msg = f"the AIFS output must not be the published folder {published_dir}"
        raise ValueError(msg)
    output_path = output_dir / f"{domain}_aifs_inputs.parquet"
    refuse_to_overwrite(paths=[output_path])
    keys = pl.read_parquet(published_dir / f"{domain}_forecast_inputs.parquet").select(
        "site", "time"
    )
    frame = _aifs_frame(keys=keys, domain=domain, weather_dir=weather_dir)
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
        help="Build the exploratory lead columns (ENS at days 5 and 14, GEFS at days 5, 10 and 14, "
        "Previous Runs day 0 and day 5) on the published inputs' keys, into a new --output-dir.",
    )
    parser.add_argument(
        "--aifs",
        action="store_true",
        help="Build the AIFS Single and AIFS ENS columns and ENS's 6-hourly references on the "
        "published inputs' keys, into a new --output-dir.",
    )
    parser.add_argument(
        "--aifs-weather-dir",
        type=Path,
        default=_weather_dir(),
        help="With --aifs: the folder holding the ECMWF-AIFS and ECMWF-AIFS-ENS downloads.",
    )
    parser.add_argument(
        "--published-dir",
        type=Path,
        default=_repo_data_dir() / "studies" / DEFAULT_OUTPUT_DIR_NAME,
        help="With --extra-leads or --aifs: the folder holding the published arm-input parquets.",
    )
    parser.add_argument(
        "--gefs-window-dir",
        type=Path,
        default=None,
        help="A GEFS_window_* test extract, for development only; production reads the month "
        "cache once it is complete.",
    )
    args = parser.parse_args()
    if args.aifs:
        for domain in ("solar", "wind"):
            build_aifs(
                domain=domain,
                published_dir=args.published_dir,
                output_dir=args.output_dir,
                weather_dir=args.aifs_weather_dir,
            )
        return 0
    if args.extra_leads:
        for domain in ("solar", "wind"):
            build_extra_leads(
                domain=domain,
                published_dir=args.published_dir,
                output_dir=args.output_dir,
                gefs_window_dir=args.gefs_window_dir,
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
