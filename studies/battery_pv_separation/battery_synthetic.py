"""Build the synthetic solar-plus-battery aggregates and the price regressors of rung 4.

Each aggregate is the real settled output of one or more solar BMUs plus the real settled output
of one battery, each scaled so that the aggregate's 99th-percentile output is 100 MW, with the
solar part holding a stated share of it. This module copies only three parts of the solar
disaggregation study's design: the same solar halves, the same window, and the same regional sky.
This module reads the solar study's saved `stage1_fits.parquet` for the direct-current to
alternating-current (DC:AC) ratio the fit's starting capacity uses, and imports none of that study's
scripts.

Half-hours are labelled by their end time, as B1610 labels them and as the solar study's grid
does. The price files label each period by its start, so a price is looked up at the half-hour end
time minus 30 minutes.
"""

from datetime import UTC, datetime, timedelta
from typing import Final

import numpy as np
import polars as pl
from battery_inputs import B1610_DIR, B1610_SUFFIX, WINDOW_START, market_frame
from studies.pv_physics import SunAndSky, half_hour_sun_and_sky
from studies.sources import (
    REANALYSIS_DOWNLOADS_DIR,
    SOLAR_BMU_CENSUS_DIR,
    SOLAR_BMU_DISAGGREGATION_DIR,
)

SOLAR_SETS: Final[dict[str, tuple[str, ...]]] = {
    "pure_burwell": ("T_BURWS-1",),
    "pure_bishampton": ("C__ESTAT019",),
    "pure_litchardon": ("C__LSTAT020",),
    "pure_three_sites": ("T_BURWS-1", "C__ESTAT019", "C__LSTAT020"),
}
"""The solar halves: BMUs the census calls pure PV, singly and together."""
SOLAR_SET_LABELS: Final[dict[str, str]] = {
    "pure_burwell": "Burwell",
    "pure_bishampton": "Bishampton",
    "pure_litchardon": "Litchardon",
    "pure_three_sites": "All three solar sites",
}
SHARES: Final[tuple[float, ...]] = (0.0, 0.1, 0.25, 0.5)
"""The solar part's 99th-percentile output over the aggregate's 100 MW."""
AGGREGATE_P99_MW: Final[float] = 100.0
NEAREST_POINTS: Final[int] = 3
"""The regional sky is the mean of this many CAMS grid points nearest to the solar centroid."""
WINDOW_END: Final[datetime] = datetime(2026, 9, 1, tzinfo=UTC)
CAMS_PATH = REANALYSIS_DOWNLOADS_DIR / "CAMS_public_points" / "cams_public_points.parquet"
MIN_CAMS_RELIABILITY: Final[float] = 0.9
"""Point-hours that CAMS flags below this fraction of reliable inputs are dropped."""
REFERENCE_LATITUDE: Final[float] = 53.0
REFERENCE_LONGITUDE: Final[float] = -1.5
"""The point whose sun position stands for Great Britain when the sky is the GB mean."""
HALF_HOURS_PER_WEEK: Final[int] = 7 * 48
SPREAD_COLUMN: Final[str] = "imbalance_spread"


def window_half_hours() -> pl.Series:
    """Return the end time of every half-hour of the window.

    Returns:
        The end times as UTC datetimes.
    """
    return pl.datetime_range(
        WINDOW_START + timedelta(minutes=30),
        WINDOW_END,
        interval="30m",
        time_unit="us",
        eager=True,
    ).alias("half_hour_end_time")


def grid_index(*, half_hour_end_time: np.ndarray) -> np.ndarray:
    """Return each half-hour's row in the window grid; -1 where it is outside the window.

    Args:
        half_hour_end_time: The end time of each half-hour.

    Returns:
        Each half-hour's row index in the window grid.
    """
    start = np.datetime64(WINDOW_START.replace(tzinfo=None), "us")
    steps = (half_hour_end_time.astype("datetime64[us]") - start) // np.timedelta64(30, "m") - 1
    inside = (steps >= 0) & (steps < len(window_half_hours()))
    return np.where(inside, steps, -1).astype(int)


def on_grid(*, half_hour_end_time: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Place values given at some half-hours onto the window grid, NaN elsewhere.

    Args:
        half_hour_end_time: The end time of each half-hour that has a value.
        values: One value (or row of values) per half-hour.

    Returns:
        An array with one row per half-hour of the window grid.
    """
    index = grid_index(half_hour_end_time=half_hour_end_time)
    grid = np.full((len(window_half_hours()), *values.shape[1:]), np.nan)
    grid[index[index >= 0]] = values[index >= 0]
    return grid


def output_on_grid(*, bmu_id: str) -> np.ndarray:
    """Return a BMU's settled output in megawatts at every window half-hour.

    Args:
        bmu_id: The Elexon BMU identifier.

    Returns:
        The output, NaN where B1610 published none.
    """
    output = pl.read_parquet(B1610_DIR / f"{bmu_id}{B1610_SUFFIX}").select(
        half_hour_end_time=pl.col("half_hour_end_time").dt.cast_time_unit("us"),
        output_mw=pl.col("output_mwh") * 2.0,
    )
    grid = pl.DataFrame({"half_hour_end_time": window_half_hours()})
    return grid.join(output, on="half_hour_end_time", how="left")["output_mw"].to_numpy()


def site_coordinates(*, bmu_ids: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    """Return each BMU's latitude and longitude, from the census table.

    Args:
        bmu_ids: The BMUs to look up.

    Returns:
        Latitude and longitude in degrees, keyed by BMU id.
    """
    table = pl.read_parquet(SOLAR_BMU_CENSUS_DIR / "solar_bmus.parquet").filter(
        pl.col("elexon_bmu_id").is_in(bmu_ids)
    )
    return {
        row["elexon_bmu_id"]: (row["latitude"], row["longitude"])
        for row in table.iter_rows(named=True)
    }


def hourly_cams(*, point_ids: list[str], column: str = "ghi_w_m2") -> pl.DataFrame:
    """Return the mean CAMS irradiance of some points in the window.

    Args:
        point_ids: The CAMS points to average.
        column: `ghi_w_m2` for all-sky irradiance or `clear_sky_ghi_w_m2` for a cloudless sky.

    Returns:
        Columns `time` (UTC, the end of the hour) and `ghi_w_m2`, the mean over the points that
        are reliable in that hour.
    """
    cams = pl.read_parquet(CAMS_PATH).filter(
        pl.col("point_id").is_in(point_ids)
        & (pl.col("reliability") >= MIN_CAMS_RELIABILITY)
        & (pl.col("time") > WINDOW_START)
        & (pl.col("time") <= WINDOW_END)
    )
    return (
        cams.group_by("time")
        .agg(pl.col(column).mean().alias("ghi_w_m2"))
        .sort("time")
        .with_columns(pl.col("time").dt.cast_time_unit("us"))
    )


def grid_point_positions() -> pl.DataFrame:
    """Return the 18 grid points' identifiers and positions.

    Returns:
        One row per grid point.
    """
    return (
        pl.read_parquet(CAMS_PATH, columns=["point_id", "latitude", "longitude"])
        .filter(pl.col("point_id").str.starts_with("gb_"))
        .unique()
        .sort("point_id")
    )


def nearest_grid_points(*, latitude: float, longitude: float, count: int) -> list[str]:
    """Return the `count` grid points nearest to a position, nearest first.

    Args:
        latitude: Degrees north.
        longitude: Degrees east.
        count: How many points to return.

    Returns:
        The points' identifiers, by great-circle distance.
    """
    points = grid_point_positions()
    lat1, lon1 = np.radians(latitude), np.radians(longitude)
    lat2 = np.radians(points["latitude"].to_numpy())
    lon2 = np.radians(points["longitude"].to_numpy())
    haversine = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return [points["point_id"].to_list()[index] for index in np.argsort(haversine)[:count]]


def sky_at(*, hourly: pl.DataFrame, latitude: float, longitude: float) -> SunAndSky:
    """Return the half-hourly sun and sky for hourly irradiance at one place.

    Args:
        hourly: Hourly irradiance, as `hourly_cams` returns it.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        The sun and sky at each half-hour.
    """
    return half_hour_sun_and_sky(hourly=hourly, latitude=latitude, longitude=longitude)


def stage1_ratio(*, excluded: tuple[str, ...]) -> float:
    """Return the median fitted DC:AC ratio of the solar study's single-site fits.

    Args:
        excluded: The BMUs in the aggregate, whose fits are left out.

    Returns:
        The median ratio of the free-orientation fits to all the data.
    """
    fits = pl.read_parquet(SOLAR_BMU_DISAGGREGATION_DIR / "stage1_fits.parquet").filter(
        (pl.col("arm") == "free") & (pl.col("fold") == -1) & ~pl.col("bmu").is_in(list(excluded))
    )
    return float(fits["dc_ac_ratio"].median())  # ty: ignore[invalid-argument-type]


def price_regressors() -> tuple[np.ndarray, dict[str, int]]:
    """Return the price regressors at every window half-hour.

    The columns are, in order: the day-ahead price minus its UTC-day mean, divided by the standard
    deviation of that difference over the window; the within-day rank of the day-ahead price
    (ties averaged) rescaled to between -1 and 1; and the system price minus the day-ahead price,
    divided by its standard deviation over the window. The hourly day-ahead price applies to both
    half-hours of its hour. A half-hour without a price gets 0 in every column.

    Returns:
        The array, with one row per half-hour of the window, and the number of half-hours that
        had no price, keyed by column name.
    """
    grid = pl.DataFrame({"half_hour_end_time": window_half_hours()}).with_columns(
        time=pl.col("half_hour_end_time").dt.offset_by("-30m")
    )
    prices = (
        grid.join(market_frame(), on="time", how="left")
        .with_columns(date=pl.col("time").dt.date())
        .with_columns(
            level=pl.col("day_ahead_gbp_per_mwh")
            - pl.col("day_ahead_gbp_per_mwh").mean().over("date"),
            rank=(
                (pl.col("day_ahead_gbp_per_mwh").rank("average").over("date") - 0.5)
                / pl.col("day_ahead_gbp_per_mwh").count().over("date")
            )
            * 2.0
            - 1.0,
            spread=pl.col("system_price_gbp_per_mwh") - pl.col("day_ahead_gbp_per_mwh"),
        )
    )
    level = prices["level"].to_numpy()
    spread = prices["spread"].to_numpy()
    columns = {
        "day_ahead_level": level / np.nanstd(level),
        "day_ahead_rank": prices["rank"].to_numpy(),
        SPREAD_COLUMN: spread / np.nanstd(spread),
    }
    missing = {name: int(np.isnan(values).sum()) for name, values in columns.items()}
    stacked = np.column_stack(list(columns.values()))
    return np.nan_to_num(stacked, nan=0.0), missing


def day_ahead_on_grid() -> np.ndarray:
    """Return the day-ahead price at every window half-hour, NaN where there is none.

    Returns:
        The price in pounds per megawatt-hour, one value per half-hour of the window grid, which
        starts at the first half-hour of a UTC day.
    """
    grid = pl.DataFrame({"half_hour_end_time": window_half_hours()}).with_columns(
        time=pl.col("half_hour_end_time").dt.offset_by("-30m")
    )
    return grid.join(market_frame(), on="time", how="left")["day_ahead_gbp_per_mwh"].to_numpy()
