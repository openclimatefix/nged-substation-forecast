"""The feature-tensor cache of the differentiable-physics (DP) series.

Part 1 of the series (experiment 103) decided which rows each substation may train on. This module
turns those decisions, ERA5 temperature, and CAMS irradiance into one set of arrays per substation
on a common half-hourly grid, built once and cached on disk under a hash of the input files, so
that every later part is a thin configuration plus its arms.

**One array per input family, indexed by the grid.** The arrays are:

- the target `y`, power in points of the substation's effective capacity (percent), `NaN` where the
  meter has no reading;
- the row masks of experiment 103: `present`, `in_pool`, `default_rows`, `switching_mask`,
  `lockdown`, and `dark`;
- ERA5 temperature at the half-hour, and its trailing means over 6, 24, 72, and 168 hours;
- the sun's elevation at the half-hour's midpoint;
- CAMS global horizontal irradiance, and the unit PV shape `pv_unit_shape` (output per MW of AC
  capacity of a south-facing fleet, tilt 35 degrees, DC to AC ratio 1.15, from the Erbs split of
  CAMS irradiance), before temperature derating;
- the calendar columns of `studies.dp_calendar`, which do not depend on the substation.

**Design hooks.** `SiteTable` carries the coordinates of each substation for a later
geographic-proximity prior, and `FeatureCache.node_ids` gives every row the list of nodes it
observes (its own substation for now), which a later part replaces with electricity-network
topology.

**Timestamps.** The cleaned power `time` marks the end of the half-hour, and NGED's late stamps
before 2026-03-26 08:30 UTC are already corrected at ingest, so nothing here shifts them again.
The grid ends before 2026-07-01 00:00 UTC, the start of the held-out period, so no array holds a
valid time on or after it.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

from studies.dp_calendar import (
    SMOOTHING_HOURS,
    CalendarColumns,
    calendar_columns,
    trailing_mean_temperatures,
)
from studies.experiment_store import HOLDOUT_START
from studies.pv_physics import (
    PlantParameters,
    half_hour_sun_and_sky,
    plant_power_mw,
    solar_position,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

CACHE_VERSION: Final[int] = 1
"""Bump this when a column's definition changes, so an old cache file is not reused."""

GRID_START: Final[datetime] = datetime(2019, 9, 1, 0, 30, tzinfo=UTC)
"""The end of the first half-hour on the grid."""

GRID_END: Final[datetime] = datetime(2026, 6, 30, 23, 30, tzinfo=UTC)
"""The end of the last half-hour on the grid, one half-hour before the held-out period."""

FLEET: Final[PlantParameters] = PlantParameters(
    tilt_deg=35.0, azimuth_deg=180.0, dc_capacity_mw=1.15, ac_capacity_mw=1.0
)
"""The generic south-facing fleet whose output per MW of AC capacity is the unit PV shape.

It is experiment 64's `FLEET`, so the convex model and the XGBoost solar offset share one shape.
"""

PART_1_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00103-differentiable-physics-1-data-look"
"""The private folder that holds experiment 103's row sets and site table."""

ERA5_PATH: Final[Path] = (
    REPO_DATA_DIR / "studies/per_study/stage1_pilot/demand/inputs/era5_features.parquet"
)
"""ERA5 weather at each series' location on a half-hourly grid; the temperature is `temperature_2m`.

The reanalysis is hourly. The :30 value of each half-hour is the mean of the two neighbouring hourly
values, so a half-hour ending at :30 uses the next hour's value (weather is an exogenous input).
"""

CAMS_PATH: Final[Path] = (
    REPO_DATA_DIR / "studies/downloads/reanalysis/CAMS/demand_points/cams_demand_points.parquet"
)
"""CAMS hourly irradiance at each series' location, stamped with the end of the hour."""

SERIES_TABLE_NAME: Final[str] = "series_table.parquet"
ROWS_NAME: Final[str] = "rows.parquet"
SITES_NAME: Final[str] = "sites.parquet"


@dataclass(frozen=True)
class DpInputPaths:
    """The files the cache is built from.

    Attributes:
        rows: Experiment 103's `rows.parquet`.
        series_table: Its `series_table.parquet` (coordinates, units, effective capacity).
        sites: Its `sites.parquet` (role and first usable date).
        era5: ERA5 weather per `time_series_id` on a half-hourly grid.
        cams: CAMS hourly irradiance per site label.
    """

    rows: Path
    series_table: Path
    sites: Path
    era5: Path
    cams: Path


def default_input_paths() -> DpInputPaths:
    """Return the paths of experiment 103's outputs and the weather files."""
    return DpInputPaths(
        rows=PART_1_DIR / ROWS_NAME,
        series_table=PART_1_DIR / SERIES_TABLE_NAME,
        sites=PART_1_DIR / SITES_NAME,
        era5=ERA5_PATH,
        cams=CAMS_PATH,
    )


@dataclass(frozen=True)
class SiteTable:
    """One substation's identity and location.

    Attributes:
        site: The anonymised label, such as `D10`.
        time_series_id: NGED's series identifier.
        units: `MW` or `MVA`.
        role: Experiment 103's role: `mw_long`, `mw_short`, `mva_dark`, or `generator_case`.
        latitude: Degrees north.
        longitude: Degrees east.
        effective_capacity_mw: The series' effective capacity, which converts MW to points.
        first_usable: The first usable instant after the early-calibration cut.
    """

    site: str
    time_series_id: int
    units: str
    role: str
    latitude: float
    longitude: float
    effective_capacity_mw: float
    first_usable: datetime


@dataclass(frozen=True)
class SiteFeatures:
    """One substation's arrays on the grid; every array has one entry per grid half-hour.

    Attributes:
        table: The substation's identity and location.
        y: Power in points of effective capacity, float64, `NaN` where there is no reading.
        present: True where the meter has a reading.
        in_pool: True where experiment 103's pool contains the row.
        default_rows: True where the row is in the pool, outside the lockdown window, and outside
            the switching masks.
        switching_mask: True within the padded hand-filtered switching events.
        lockdown: True inside the fixed lockdown window.
        dark: True where CAMS irradiance is below 1 W/m².
        temperature_c: ERA5 temperature interpolated to the half-hour, float64.
        smoothed_temperature_c: Trailing means over `SMOOTHING_HOURS`, shape (T, 4), float64.
        ghi_w_m2: CAMS irradiance at the half-hour, float64, `NaN` where CAMS is missing.
        elevation_deg: The sun's elevation at the half-hour's midpoint, float64.
        pv_unit_shape: The unit PV shape per MW of AC capacity before derating, float64, `NaN`
            where CAMS is missing.
    """

    table: SiteTable
    y: np.ndarray
    present: np.ndarray
    in_pool: np.ndarray
    default_rows: np.ndarray
    switching_mask: np.ndarray
    lockdown: np.ndarray
    dark: np.ndarray
    temperature_c: np.ndarray
    smoothed_temperature_c: np.ndarray
    ghi_w_m2: np.ndarray
    elevation_deg: np.ndarray
    pv_unit_shape: np.ndarray


@dataclass(frozen=True)
class FeatureCache:
    """All substations' arrays on one grid.

    Attributes:
        key: The hash of the inputs and `CACHE_VERSION` that the cache was built from.
        end_times: The UTC end of each half-hour, as `datetime64[us]`, strictly increasing.
        calendar: The calendar columns, which do not depend on the substation.
        sites: Each substation's arrays, keyed by label.
        node_ids: For each row, the labels of the nodes whose readings the row observes; the
            substation itself, for now.
    """

    key: str
    end_times: np.ndarray
    calendar: CalendarColumns
    sites: dict[str, SiteFeatures]
    node_ids: dict[str, tuple[str, ...]]

    def times_series(self) -> pl.Series:
        """Return the grid as a polars UTC datetime series named `time`."""
        return pl.Series("time", self.end_times).dt.replace_time_zone("UTC")


def naive_utc(*, moment: datetime) -> np.datetime64:
    """Return a timezone-aware datetime as a naive UTC `datetime64[us]`.

    Args:
        moment: An aware datetime in any time zone.

    Returns:
        The same instant as UTC microseconds.

    Raises:
        ValueError: If `moment` has no time zone, since a naive wall time could be local.
    """
    if moment.tzinfo is None:
        msg = "a cut or window bound must carry a time zone"
        raise ValueError(msg)
    return np.datetime64(moment.astimezone(UTC).replace(tzinfo=None), "us")


def grid_end_times() -> pl.Series:
    """Return the cache's half-hourly grid as a polars UTC datetime series."""
    return pl.datetime_range(
        GRID_START, GRID_END, interval="30m", time_zone="UTC", eager=True
    ).alias("time")


def input_key(*, paths: DpInputPaths) -> str:
    """Return the hash of the input files' bytes and `CACHE_VERSION`.

    Args:
        paths: The input files.

    Returns:
        A 16-character hexadecimal string.
    """
    digest = hashlib.sha256(str(CACHE_VERSION).encode())
    for path in (paths.rows, paths.series_table, paths.sites, paths.era5, paths.cams):
        with path.open("rb") as handle:
            while block := handle.read(1 << 24):
                digest.update(block)
    return digest.hexdigest()[:16]


def _interpolate_to_grid(
    *, reading_stamps: np.ndarray, values: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Interpolate readings linearly onto the grid, `NaN` outside their range.

    The readings are already on the grid today, so this returns them unchanged; it keeps a coarser
    series (hourly, say) usable.
    """
    seconds_hourly = reading_stamps.astype("datetime64[s]").astype(np.int64)
    seconds_grid = grid.astype("datetime64[s]").astype(np.int64)
    known = np.isfinite(values)
    return np.interp(
        seconds_grid, seconds_hourly[known], values[known], left=np.nan, right=np.nan
    ).astype(np.float64)


def _mask_column(*, joined: pl.DataFrame, column: str) -> np.ndarray:
    """Return a boolean mask column of the joined frame, False where the row is missing."""
    return joined[column].fill_null(False).to_numpy()


def _site_table(*, row: Mapping[str, Any], sites_row: Mapping[str, Any]) -> SiteTable:
    """Combine one series-table row and one sites-table row into a `SiteTable`."""
    return SiteTable(
        site=row["site"],
        time_series_id=int(row["time_series_id"]),
        units=str(row["units"]),
        role=sites_row["role"],
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"]),
        effective_capacity_mw=float(row["effective_capacity_mw"]),
        first_usable=sites_row["first_usable"],
    )


def unit_pv_shape(
    *, cams: pl.DataFrame, table: SiteTable, grid: pl.Series, plant: PlantParameters = FLEET
) -> np.ndarray:
    """Return the unit PV shape of the site's CAMS irradiance on the grid, `NaN` if missing.

    Args:
        cams: Columns `site`, `time` (the end of the hour), and `ghi_w_m2`.
        table: The substation's table, which gives the site label and the coordinates.
        grid: The half-hour end stamps of the feature cache's time grid.
        plant: The plant whose output is the shape; the generic south-facing fleet by default.

    Returns:
        The unit PV shape on the grid.
    """
    hourly = (
        cams.filter(pl.col("site") == table.site)
        .select("time", "ghi_w_m2")
        .drop_nulls()
        .sort("time")
    )
    sky = half_hour_sun_and_sky(
        hourly=hourly.with_columns(ghi_w_m2=pl.col("ghi_w_m2").cast(pl.Float64)),
        latitude=table.latitude,
        longitude=table.longitude,
    )
    shape = pl.DataFrame(
        {
            "time": pl.Series(sky.half_hour_end_time).dt.replace_time_zone("UTC"),
            "pv_unit_shape": plant_power_mw(sky=sky, parameters=plant),
        }
    )
    joined = pl.DataFrame({"time": grid}).join(
        shape, on="time", how="left", validate="1:1", maintain_order="left"
    )
    return joined["pv_unit_shape"].to_numpy().astype(np.float64)


def _site_features(
    *,
    table: SiteTable,
    rows: pl.DataFrame,
    era5: pl.DataFrame,
    cams: pl.DataFrame,
    grid: pl.Series,
) -> SiteFeatures:
    """Build one substation's arrays on the grid."""
    joined = pl.DataFrame({"time": grid}).join(
        rows.filter(pl.col("site") == table.site).drop("site", "time_series_id"),
        on="time",
        how="left",
        validate="1:1",
        maintain_order="left",
    )
    readings = trailing_mean_temperatures(
        series=era5.filter(pl.col("time_series_id") == table.time_series_id).select(
            "valid_time", temperature_c=pl.col("temperature_2m").cast(pl.Float64)
        )
    )
    stamps = readings["valid_time"].dt.replace_time_zone(None).to_numpy()
    grid_numpy = grid.dt.replace_time_zone(None).to_numpy()
    smoothed = np.stack(
        [
            _interpolate_to_grid(
                reading_stamps=stamps,
                values=readings[f"temperature_c_k{k}"].to_numpy(),
                grid=grid_numpy,
            )
            for k in SMOOTHING_HOURS
        ],
        axis=1,
    )
    temperature = _interpolate_to_grid(
        reading_stamps=stamps, values=readings["temperature_c"].to_numpy(), grid=grid_numpy
    )
    midpoints = grid.dt.offset_by("-15m")
    zenith, _ = solar_position(stamps=midpoints, latitude=table.latitude, longitude=table.longitude)
    half_hour_cams = pl.DataFrame({"time": grid}).join(
        _cams_on_half_hours(cams=cams.filter(pl.col("site") == table.site)).select(
            "time", ghi_w_m2=pl.col("ghi_w_m2").cast(pl.Float64)
        ),
        on="time",
        how="left",
        validate="1:1",
        maintain_order="left",
    )
    power = joined["power"].to_numpy().astype(np.float64)
    return SiteFeatures(
        table=table,
        y=100.0 * power / table.effective_capacity_mw,
        present=_mask_column(joined=joined, column="present"),
        in_pool=_mask_column(joined=joined, column="in_pool"),
        default_rows=_mask_column(joined=joined, column="default_rows"),
        switching_mask=_mask_column(joined=joined, column="switching_mask"),
        lockdown=_mask_column(joined=joined, column="lockdown"),
        dark=_mask_column(joined=joined, column="dark"),
        temperature_c=temperature,
        smoothed_temperature_c=smoothed,
        ghi_w_m2=half_hour_cams["ghi_w_m2"].to_numpy().astype(np.float64),
        elevation_deg=90.0 - zenith,
        pv_unit_shape=unit_pv_shape(cams=cams, table=table, grid=grid),
    )


def build_feature_cache(*, paths: DpInputPaths) -> FeatureCache:
    """Build every substation's arrays from experiment 103's outputs and the weather files.

    Half-hour CAMS irradiance is the hour's value for both of the hour's half-hours; the unit PV
    shape shares an hour's irradiance between its half-hours by the cosine of the zenith.

    Args:
        paths: The input files.

    Returns:
        The cache, keyed by `input_key(paths=paths)`.
    """
    grid = grid_end_times()
    rows = pl.read_parquet(paths.rows).filter(pl.col("time") < HOLDOUT_START)
    series = pl.read_parquet(paths.series_table)
    sites_table = pl.read_parquet(paths.sites).filter(pl.col("time_series_id").is_not_null())
    era5 = pl.read_parquet(paths.era5, columns=["time_series_id", "valid_time", "temperature_2m"])
    cams = pl.read_parquet(paths.cams, columns=["site", "time", "ghi_w_m2"])
    sites: dict[str, SiteFeatures] = {}
    wanted = set(rows["site"].unique().to_list())
    for row in series.iter_rows(named=True):
        if row["site"] not in wanted:
            continue
        sites_row = sites_table.filter(pl.col("site") == row["site"]).row(0, named=True)
        table = _site_table(row=row, sites_row=sites_row)
        sites[table.site] = _site_features(table=table, rows=rows, era5=era5, cams=cams, grid=grid)
    return FeatureCache(
        key=input_key(paths=paths),
        end_times=grid.dt.replace_time_zone(None).to_numpy(),
        calendar=calendar_columns(end_times=grid),
        sites=sites,
        node_ids={site: (site,) for site in sites},
    )


def _cams_on_half_hours(*, cams: pl.DataFrame) -> pl.DataFrame:
    """Repeat each CAMS hour-ending value on the two half-hours that end inside the hour.

    Args:
        cams: Columns `site`, `time` (the end of the hour), and `ghi_w_m2`.

    Returns:
        Columns `site`, `time`, and `ghi_w_m2` with one row per half-hour end.
    """
    first_half = cams.with_columns(time=pl.col("time").dt.offset_by("-30m"))
    return pl.concat([first_half, cams]).sort("site", "time")


_SITE_ARRAYS: Final[tuple[str, ...]] = (
    "y",
    "present",
    "in_pool",
    "default_rows",
    "switching_mask",
    "lockdown",
    "dark",
    "temperature_c",
    "smoothed_temperature_c",
    "ghi_w_m2",
    "elevation_deg",
    "pv_unit_shape",
)
_CALENDAR_ARRAYS: Final[tuple[str, ...]] = (
    "half_hour",
    "day_type",
    "band",
    "holiday_period",
    "tent",
    "signed_tent",
    "harmonics",
    "local_date",
    "season_year",
    "day_of_year",
    "quarter",
)


def save_feature_cache(*, cache: FeatureCache, path: Path) -> None:
    """Write the cache to one compressed `.npz` file, atomically.

    Args:
        cache: The cache.
        path: The destination; the file is written beside it and renamed into place.
    """
    arrays: dict[str, Any] = {"end_times": cache.end_times}
    for name in _CALENDAR_ARRAYS:
        arrays[f"calendar/{name}"] = getattr(cache.calendar, name)
    for site, features in cache.sites.items():
        for name in _SITE_ARRAYS:
            arrays[f"{site}/{name}"] = getattr(features, name)
        table = features.table
        arrays[f"{site}/table"] = np.array(
            [
                site,
                str(table.time_series_id),
                table.units,
                table.role,
                repr(table.latitude),
                repr(table.longitude),
                repr(table.effective_capacity_mw),
                table.first_usable.isoformat(),
            ]
        )
    arrays["key"] = np.array([cache.key])
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".staging.npz")
    np.savez_compressed(staging, **arrays)
    staging.rename(path)


def load_feature_cache(*, path: Path) -> FeatureCache:
    """Read a cache written by `save_feature_cache`.

    Args:
        path: The `.npz` file.

    Returns:
        The cache.
    """
    stored = np.load(path, allow_pickle=False)
    calendar = CalendarColumns(**{name: stored[f"calendar/{name}"] for name in _CALENDAR_ARRAYS})
    sites: dict[str, SiteFeatures] = {}
    for name in {key.split("/")[0] for key in stored.files if key.endswith("/table")}:
        fields = stored[f"{name}/table"].tolist()
        table = SiteTable(
            site=fields[0],
            time_series_id=int(fields[1]),
            units=fields[2],
            role=fields[3],
            latitude=float(fields[4]),
            longitude=float(fields[5]),
            effective_capacity_mw=float(fields[6]),
            first_usable=datetime.fromisoformat(fields[7]),
        )
        sites[name] = SiteFeatures(
            table=table, **{array: stored[f"{name}/{array}"] for array in _SITE_ARRAYS}
        )
    return FeatureCache(
        key=str(stored["key"][0]),
        end_times=stored["end_times"],
        calendar=calendar,
        sites=sites,
        node_ids={site: (site,) for site in sites},
    )


def load_or_build_feature_cache(*, paths: DpInputPaths, cache_dir: Path) -> FeatureCache:
    """Return the cache for these inputs, building and saving it on the first call.

    Args:
        paths: The input files.
        cache_dir: The folder of cache files, one per input hash.

    Returns:
        The cache.
    """
    key = input_key(paths=paths)
    path = cache_dir / f"dp_features_{key}.npz"
    if path.exists():
        return load_feature_cache(path=path)
    cache = build_feature_cache(paths=paths)
    save_feature_cache(cache=cache, path=path)
    return cache
