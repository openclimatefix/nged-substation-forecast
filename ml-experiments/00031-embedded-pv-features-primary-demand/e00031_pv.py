"""Per-unit PV output, the daily clear-sky index, and the capacity features of experiment 00031.

The per-unit PV output `p_unit` is the AC output per MW of AC capacity of a generic fleet: tilt 35
degrees, azimuth 180 degrees, DC 1.15 MW, AC 1.0 MW, times a 0.86 system derate. At lag times the
irradiance is CAMS; at valid times it is the ENS member's `shortwave`. Every capacity feature at a
forecast origin reads power and CAMS values up to 48 hours before the origin only.
"""

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00031_common as common
import holidays
import numpy as np
import polars as pl
import pvlib
from studies.pv_physics import (
    PlantParameters,
    SunAndSky,
    half_hour_sun_and_sky,
    plant_power_mw,
    solar_position,
)
from studies.sources import STUDIES_DATA_DIR

CAMS_PATH: Final[Path] = (
    STUDIES_DATA_DIR
    / "downloads"
    / "reanalysis"
    / "CAMS"
    / "demand_points"
    / "cams_demand_points.parquet"
)
"""CAMS hourly GHI and clear-sky GHI at the 22 demand points, read only."""
FLEET: Final[PlantParameters] = PlantParameters(
    tilt_deg=35.0, azimuth_deg=180.0, dc_capacity_mw=1.15, ac_capacity_mw=1.0
)
SYSTEM_DERATE: Final[float] = 0.86
"""Inverter, wiring, soiling, and degradation losses; the temperature effect is ignored."""
MAX_DAYLIGHT_ZENITH_DEG: Final[float] = 75.0
"""Daylight half-hours have the sun at least 15 degrees above the horizon."""
FEATURE_DELAY: Final[timedelta] = timedelta(hours=48)
FEATURE_DELAY_STEPS: Final[int] = int(FEATURE_DELAY / common.STEP)
WINDOW_DAYS: Final[int] = 28
YEAR_EARLIER_DAYS: Final[int] = 364
MIN_DAYS: Final[int] = 8
MIN_SUNNY_AND_CLOUDY_DAYS: Final[int] = 3
SUNNY_INDEX: Final[float] = 0.7
CLOUDY_INDEX: Final[float] = 0.3
MIN_VARIANCE: Final[float] = 1e-6
"""Smallest sum of squared demeaned `p_unit` (per daylight half-hour) that identifies a slope."""


@dataclass(frozen=True)
class SiteGrid:
    """One site's sun, CAMS-driven PV output, and sky clearness on the dense half-hourly grid.

    Attributes:
        zenith_deg: Solar zenith at each half-hour's midpoint.
        azimuth_deg: Solar azimuth at each half-hour's midpoint.
        day_of_year: Day of year of each half-hour's midpoint.
        daylight: True where the sun is at least 15 degrees above the horizon.
        cams_unit_power: `p_unit` driven by CAMS, NaN where CAMS has no value.
        daily_clear_sky_index: CAMS GHI over clear-sky GHI summed over each local day, repeated at
            every half-hour of the day; NaN where CAMS has no value.
    """

    zenith_deg: np.ndarray
    azimuth_deg: np.ndarray
    day_of_year: np.ndarray
    daylight: np.ndarray
    cams_unit_power: np.ndarray
    daily_clear_sky_index: np.ndarray


def unit_power(
    *,
    ghi_w_m2: np.ndarray,
    zenith_deg: np.ndarray,
    azimuth_deg: np.ndarray,
    day_of_year: np.ndarray,
) -> np.ndarray:
    """Return `p_unit` for half-hours with a known global horizontal irradiance.

    Args:
        ghi_w_m2: Global horizontal irradiance, W/m².
        zenith_deg: Solar zenith in degrees.
        azimuth_deg: Solar azimuth in degrees clockwise from north.
        day_of_year: Day of year, for the Erbs split into direct and diffuse light.

    Returns:
        The fleet's AC output per MW of AC capacity, NaN where the irradiance is NaN.
    """
    ghi = np.clip(np.nan_to_num(ghi_w_m2.astype(np.float64)), 0.0, None)
    split = pvlib.irradiance.erbs(ghi=ghi, zenith=zenith_deg, datetime_or_doy=day_of_year)
    sky = SunAndSky(
        half_hour_end_time=np.zeros(ghi.shape),
        ghi_w_m2=ghi,
        dni_w_m2=np.nan_to_num(np.asarray(split["dni"]).astype(np.float64)),
        dhi_w_m2=np.nan_to_num(np.asarray(split["dhi"]).astype(np.float64)),
        zenith_deg=zenith_deg,
        sun_azimuth_deg=azimuth_deg,
    )
    power = SYSTEM_DERATE * plant_power_mw(sky=sky, parameters=FLEET) / FLEET.ac_capacity_mw
    return np.where(np.isnan(ghi_w_m2), np.nan, power)


def local_cells() -> tuple[np.ndarray, np.ndarray]:
    """Return, for each grid half-hour, its local hour-by-day-type cell and local date number.

    The cell is the local half-hour of the day (0 to 47) plus 48 on weekends and bank holidays.

    Returns:
        The cell and the local date's ordinal, both integer arrays on the grid.
    """
    local = common.grid_times().dt.offset_by("-15m").dt.convert_time_zone("Europe/London")
    calendar = holidays.country_holidays("GB", subdiv="ENG", years=common.HOLIDAY_YEARS)
    dates = local.dt.date()
    off_day = (local.dt.weekday() >= 6) | dates.is_in(pl.Series(sorted(calendar)).cast(pl.Date))
    half_hour = local.dt.hour() * 2 + local.dt.minute() // 30
    return (
        (half_hour + 48 * off_day.cast(pl.Int32)).to_numpy().astype(np.int64),
        dates.dt.ordinal_day().to_numpy().astype(np.int64)
        + 366 * (dates.dt.year().to_numpy().astype(np.int64) - 2018),
    )


def site_grid(*, site: str, latitude: float, longitude: float) -> SiteGrid:
    """Build one site's `SiteGrid` from its location and the CAMS table."""
    times = common.grid_times()
    midpoints = times.dt.offset_by("-15m")
    zenith_deg, azimuth_deg = solar_position(
        stamps=midpoints, latitude=latitude, longitude=longitude
    )
    day_of_year = midpoints.dt.ordinal_day().to_numpy()
    cams = pl.read_parquet(CAMS_PATH).filter(pl.col("site") == site).sort("time")
    sky = half_hour_sun_and_sky(
        hourly=cams.select("time", "ghi_w_m2").cast({"ghi_w_m2": pl.Float64}),
        latitude=latitude,
        longitude=longitude,
    )
    cams_power = SYSTEM_DERATE * plant_power_mw(sky=sky, parameters=FLEET) / FLEET.ac_capacity_mw
    ends = pl.Series(sky.half_hour_end_time).dt.cast_time_unit("us").dt.replace_time_zone("UTC")
    index = np.asarray(common.grid_index(times=ends))
    unit = np.full(times.len(), np.nan)
    inside = (index >= 0) & (index < times.len())
    unit[index[inside]] = cams_power[inside]
    local_day = (
        cams.with_columns(
            day=(pl.col("time") - pl.duration(minutes=30))
            .dt.convert_time_zone("Europe/London")
            .dt.date()
        )
        .filter(pl.col("clear_sky_ghi_w_m2") > 0)
        .group_by("day")
        .agg(index=pl.col("ghi_w_m2").sum() / pl.col("clear_sky_ghi_w_m2").sum())
    )
    grid_days = times.dt.offset_by("-15m").dt.convert_time_zone("Europe/London").dt.date()
    daily = (
        pl.DataFrame({"day": grid_days})
        .join(local_day, on="day", how="left", maintain_order="left")["index"]
        .to_numpy()
        .astype(np.float64)
    )
    return SiteGrid(
        zenith_deg=zenith_deg,
        azimuth_deg=azimuth_deg,
        day_of_year=day_of_year,
        daylight=zenith_deg <= MAX_DAYLIGHT_ZENITH_DEG,
        cams_unit_power=unit,
        daily_clear_sky_index=daily,
    )


@dataclass(frozen=True)
class WindowFit:
    """The slope estimate over one 28-day window.

    Attributes:
        capacity: Minus the fixed-effects slope of power on `p_unit`, in shares of effective
            capacity.
        standard_error: Its standard error, clustered by local day.
        sunny_minus_cloudy: Minus the matched sunny-minus-cloudy power difference over the matching
            `p_unit` difference.
    """

    capacity: float
    standard_error: float
    sunny_minus_cloudy: float


def _slope(
    *, y: np.ndarray, x: np.ndarray, cell: np.ndarray, day: np.ndarray
) -> tuple[float, float]:
    """Return the within-cell OLS slope of y on x and its day-clustered standard error."""
    size = int(cell.max()) + 1
    counts = np.bincount(cell, minlength=size)
    safe = np.maximum(counts, 1)
    x_tilde = x - (np.bincount(cell, weights=x, minlength=size) / safe)[cell]
    y_tilde = y - (np.bincount(cell, weights=y, minlength=size) / safe)[cell]
    denominator = float((x_tilde**2).sum())
    if denominator < MIN_VARIANCE * x.size:
        return np.nan, np.nan
    slope = float((x_tilde * y_tilde).sum()) / denominator
    score = x_tilde * (y_tilde - slope * x_tilde)
    _, inverse = np.unique(day, return_inverse=True)
    per_day = np.bincount(inverse, weights=score)
    return slope, float(np.sqrt((per_day**2).sum())) / denominator


def _sunny_minus_cloudy(
    *, y: np.ndarray, x: np.ndarray, cell: np.ndarray, day: np.ndarray, index: np.ndarray
) -> float:
    """Return minus the cell-matched sunny-minus-cloudy power difference over the `p_unit` one."""
    sunny = index >= SUNNY_INDEX
    cloudy = index <= CLOUDY_INDEX
    if (
        np.unique(day[sunny]).size < MIN_SUNNY_AND_CLOUDY_DAYS
        or np.unique(day[cloudy]).size < MIN_SUNNY_AND_CLOUDY_DAYS
    ):
        return np.nan
    size = int(cell.max()) + 1
    numerator = denominator = 0.0
    parts = {}
    for name, mask in (("sunny", sunny), ("cloudy", cloudy)):
        count = np.bincount(cell[mask], minlength=size)
        safe = np.maximum(count, 1)
        parts[name] = (
            count,
            np.bincount(cell[mask], weights=y[mask], minlength=size) / safe,
            np.bincount(cell[mask], weights=x[mask], minlength=size) / safe,
        )
    both = (parts["sunny"][0] > 0) & (parts["cloudy"][0] > 0)
    weight = np.minimum(parts["sunny"][0], parts["cloudy"][0])[both]
    numerator = float((weight * (parts["sunny"][1] - parts["cloudy"][1])[both]).sum())
    denominator = float((weight * (parts["sunny"][2] - parts["cloudy"][2])[both]).sum())
    if weight.sum() == 0 or denominator <= MIN_VARIANCE * weight.sum():
        return np.nan
    return -numerator / denominator


def _complete_day_index(
    *, site: SiteGrid, day: np.ndarray, window: slice, keep: np.ndarray, end_index: int
) -> np.ndarray:
    """Return the daily clear-sky index of the kept half-hours, NaN on the window's last day.

    The index of a day sums CAMS over the whole local day, so the last day, which the window may
    cut short, would read CAMS values newer than the window's end.
    """
    index = site.daily_clear_sky_index[window].copy()
    index[day[window] >= day[end_index]] = np.nan
    return index[keep]


def window_fit(
    *,
    power: np.ndarray,
    site: SiteGrid,
    cell: np.ndarray,
    day: np.ndarray,
    end_index: int,
) -> WindowFit:
    """Fit one 28-day window whose last half-hour is at grid index `end_index`.

    Args:
        power: Observed power over effective capacity on the grid, NaN where unobserved.
        site: The site's grid.
        cell: `local_cells()`' cell array.
        day: `local_cells()`' date array.
        end_index: Grid index of the newest half-hour the window may read.

    Returns:
        The window's estimates, all NaN when fewer than `MIN_DAYS` days have daylight data.
    """
    start = end_index - WINDOW_DAYS * common.STEPS_PER_DAY + 1
    if start < 0:
        return WindowFit(np.nan, np.nan, np.nan)
    window = slice(start, end_index + 1)
    y = power[window].astype(np.float64)
    x = site.cams_unit_power[window]
    keep = site.daylight[window] & np.isfinite(y) & np.isfinite(x)
    if np.unique(day[window][keep]).size < MIN_DAYS:
        return WindowFit(np.nan, np.nan, np.nan)
    y, x, cell_w, day_w = y[keep], x[keep], cell[window][keep], day[window][keep]
    slope, error = _slope(y=y, x=x, cell=cell_w, day=day_w)
    return WindowFit(
        capacity=-slope,
        standard_error=error,
        sunny_minus_cloudy=_sunny_minus_cloudy(
            y=y,
            x=x,
            cell=cell_w,
            day=day_w,
            index=_complete_day_index(
                site=site, day=day, window=window, keep=keep, end_index=end_index
            ),
        ),
    )


def origin_features(
    *,
    power: np.ndarray,
    site: SiteGrid,
    cell: np.ndarray,
    day: np.ndarray,
    origins: pl.Series,
) -> pl.DataFrame:
    """Return F1 to F3 for each forecast origin, from data at least 48 hours old.

    Args:
        power: Observed power over effective capacity on the grid, NaN where unobserved.
        site: The site's grid.
        cell: `local_cells()`' cell array.
        day: `local_cells()`' date array.
        origins: UTC forecast origins.

    Returns:
        One row per origin: `power_fcst_init_time`, `cap_slope_28d`, `cap_slope_se`,
        `sunny_minus_cloudy_28d`, and `cap_change_yoy`.
    """
    rows = []
    for origin, index in zip(origins, np.asarray(common.grid_index(times=origins)), strict=True):
        end = int(index) - FEATURE_DELAY_STEPS
        now = window_fit(power=power, site=site, cell=cell, day=day, end_index=end)
        earlier = window_fit(
            power=power,
            site=site,
            cell=cell,
            day=day,
            end_index=end - YEAR_EARLIER_DAYS * common.STEPS_PER_DAY,
        )
        rows.append(
            {
                "power_fcst_init_time": origin,
                "cap_slope_28d": now.capacity,
                "cap_slope_se": now.standard_error,
                "sunny_minus_cloudy_28d": now.sunny_minus_cloudy,
                "cap_change_yoy": now.capacity - earlier.capacity,
            }
        )
    return pl.DataFrame(rows, schema_overrides={"power_fcst_init_time": origins.dtype})
