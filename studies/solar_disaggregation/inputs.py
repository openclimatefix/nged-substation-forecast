"""Load the inputs that every script of the solar-disaggregation study shares.

The census study (`studies/solar_bmu_census`) wrote the BMUs' half-hourly settled output, the
census table, and the BMU register under `data/studies/per_study/solar_bmu_census/`. The CAMS
public points are under `data/studies/downloads/reanalysis/CAMS_public_points/`. This module reads
those files and builds the sun-and-sky frames. It runs no fit.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.pv_fit import change_variance_explained, fit_plant_to_changes
from studies.pv_physics import SunAndSky, half_hour_sun_and_sky, plant_power_mw
from studies.sources import (
    REANALYSIS_DOWNLOADS_DIR,
    SOLAR_BMU_CENSUS_DIR,
    SOLAR_BMU_CENSUS_INPUTS_DIR,
    SOLAR_BMU_DISAGGREGATION_DIR,
)

OUTPUT_DIR: Final[Path] = SOLAR_BMU_DISAGGREGATION_DIR
"""Where this study's results live."""
B1610_DIR: Final[Path] = SOLAR_BMU_CENSUS_INPUTS_DIR / "b1610"
"""The census's half-hourly output, one parquet file per BMU."""
B1610_SUFFIX: Final[str] = "_20250901_20260901.parquet"
CAMS_PATH: Final[Path] = (
    REANALYSIS_DOWNLOADS_DIR / "CAMS_public_points" / "cams_public_points.parquet"
)
"""Hourly CAMS irradiance at the single-site BMUs' positions and at 18 grid points."""
MIN_CAMS_RELIABILITY: Final[float] = 0.9
"""Point-hours that CAMS flags below this fraction of reliable inputs are dropped."""
WINDOW_START: Final[datetime] = datetime(2025, 9, 1, tzinfo=UTC)
WINDOW_END: Final[datetime] = datetime(2026, 9, 1, tzinfo=UTC)
"""The study window `(start, end]` of half-hour end times, the census's window."""
REFERENCE_LATITUDE: Final[float] = 53.0
REFERENCE_LONGITUDE: Final[float] = -1.5
"""The point whose sun position stands for Great Britain when the irradiance is a regional mean."""
VALIDATION_BMUS: Final[tuple[str, ...]] = (
    "C__ESTAT019",
    "C__LSTAT020",
    "T_BLPFS-1",
    "T_BRCHS-1",
    "T_BURWS-1",
    "T_CLVHS-1",
    "T_CLVHS-2",
    "T_LARKS-1",
    "T_SUTBS-1",
    "T_TEBWS-1",
    "T_TYLNS-1",
)
"""The single-site BMUs whose output follows the sun and has a CAMS point. Kincraig is left out:
B1610 holds no output for it."""
PURE_PV_BMUS: Final[tuple[str, ...]] = ("C__ESTAT019", "C__LSTAT020", "T_BURWS-1")
"""The census's pure-PV BMUs among `VALIDATION_BMUS`: no registered or REPD storage at the site."""


def census_table() -> pl.DataFrame:
    """Return the census table written by `studies/solar_bmu_census/collate.py`.

    Returns:
        The census table, one row per BMU.
    """
    return pl.read_parquet(SOLAR_BMU_CENSUS_DIR / "solar_bmus.parquet")


def census_classes() -> pl.DataFrame:
    """Return the census's per-BMU classes (behaviour, correlation with the sun).

    Returns:
        The per-BMU classes, one row per BMU.
    """
    return pl.read_parquet(SOLAR_BMU_CENSUS_DIR / "classes.parquet")


def bmu_register() -> pl.DataFrame:
    """Return the BMU register that the census fetched, one row per BMU.

    Returns:
        The register, one row per BMU.
    """
    raw = json.loads((SOLAR_BMU_CENSUS_INPUTS_DIR / "raw" / "bmunits_all.json").read_text())
    return pl.DataFrame(json.loads(raw["body"]), infer_schema_length=None)


def output_by_bmu(*, bmu_id: str) -> pl.DataFrame:
    """Return a BMU's half-hourly output in megawatts.

    Args:
        bmu_id: The Elexon BMU identifier.

    Returns:
        Columns `half_hour_end_time` (UTC) and `output_mw`. A half-hour that B1610 did not publish
        has no row.
    """
    frame = pl.read_parquet(B1610_DIR / f"{bmu_id}{B1610_SUFFIX}")
    return frame.select(
        half_hour_end_time=pl.col("half_hour_end_time").dt.cast_time_unit("us"),
        output_mw=pl.col("output_mwh") * 2.0,
    )


def hourly_cams(*, point_ids: list[str], column: str = "ghi_w_m2") -> pl.DataFrame:
    """Return the mean CAMS irradiance of some points, in the study window.

    Args:
        point_ids: The CAMS points to average, such as `bmu_T_BRCHS-1` or `gb_07`.
        column: `ghi_w_m2` for the all-sky irradiance, or `clear_sky_ghi_w_m2` for the irradiance
            CAMS computes for a cloudless sky.

    Returns:
        Columns `time` (UTC, the end of the hour) and `ghi_w_m2`, which holds the chosen column's
        mean over the points that are reliable in that hour. An hour with no reliable point is
        absent.
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


def grid_point_ids() -> list[str]:
    """Return the identifiers of the 18 grid points.

    Returns:
        The grid point identifiers.
    """
    ids = pl.read_parquet(CAMS_PATH, columns=["point_id"])["point_id"].unique().to_list()
    return sorted(point for point in ids if point.startswith("gb_"))


def sky_at(*, hourly: pl.DataFrame, latitude: float, longitude: float) -> SunAndSky:
    """Return the half-hourly sun and sky for hourly irradiance at one place.

    Args:
        hourly: Hourly irradiance, as `hourly_cams` returns it.
        latitude: The place's latitude, in degrees north.
        longitude: The place's longitude, in degrees east.

    Returns:
        The sun's position and the sky's irradiance at each half-hour.
    """
    return half_hour_sun_and_sky(hourly=hourly, latitude=latitude, longitude=longitude)


def align_output(*, sky: SunAndSky, output: pl.DataFrame) -> np.ndarray:
    """Return a BMU's output in megawatts at the half-hours of `sky`, NaN where B1610 has none.

    Args:
        sky: The sun and sky whose half-hours to align to.
        output: `output_by_bmu`'s output.

    Returns:
        One value per half-hour of `sky`.
    """
    times = pl.Series("half_hour_end_time", sky.half_hour_end_time).cast(pl.Datetime("us"))
    grid = pl.DataFrame({"half_hour_end_time": times.dt.replace_time_zone("UTC")})
    joined = grid.join(output, on="half_hour_end_time", how="left")
    return joined["output_mw"].to_numpy()


def site_coordinates() -> dict[str, tuple[float, float]]:
    """Return each validation BMU's position, from the census table.

    Returns:
        Each validation BMU's latitude and longitude in degrees, keyed by BMU id.
    """
    table = census_table().filter(pl.col("elexon_bmu_id").is_in(VALIDATION_BMUS))
    return {
        row["elexon_bmu_id"]: (row["latitude"], row["longitude"])
        for row in table.iter_rows(named=True)
    }


def site_cams_point(*, bmu_id: str) -> str:
    """Return the CAMS point id for a validation BMU (Cleve Hill's two BMUs share one point).

    Args:
        bmu_id: The validation BMU's identifier.

    Returns:
        The identifier of the BMU's CAMS point.
    """
    return "bmu_T_CLVHS-1" if bmu_id.startswith("T_CLVHS") else f"bmu_{bmu_id}"


COMMISSIONING_SKIP_DAYS: Final[int] = 30
"""A BMU that starts generating after the window's first week is judged this many days later."""
RUNNING_AT_START_DAYS: Final[int] = 7
POSITIVE_FLOOR_MW: Final[float] = 0.02
"""Output above this, in megawatts, counts as the BMU generating."""


def settled_mask(*, sky: SunAndSky, output_mw: np.ndarray) -> np.ndarray:
    """Return the half-hours a fit may use, by the census's rules.

    A BMU that was already generating in the window's first week is judged on every half-hour. Any
    other BMU is judged from `COMMISSIONING_SKIP_DAYS` after its first output above
    `POSITIVE_FLOOR_MW`. Exact zeros while the sun is up are dropped, because a solar plant does
    not output exactly zero at midday and an exact zero is a metering fault or an outage.

    Args:
        sky: The sun and sky at the BMU's half-hours.
        output_mw: The BMU's output at those half-hours, NaN where B1610 has none.

    Returns:
        True for each half-hour a fit may use.
    """
    times = sky.half_hour_end_time.astype("datetime64[us]")
    start = np.datetime64(WINDOW_START.replace(tzinfo=None), "us")
    positive = np.isfinite(output_mw) & (output_mw > POSITIVE_FLOOR_MW)
    mask = np.isfinite(output_mw)
    if positive.any():
        first = times[positive].min()
        if first >= start + np.timedelta64(RUNNING_AT_START_DAYS, "D"):
            mask &= times >= first + np.timedelta64(COMMISSIONING_SKIP_DAYS, "D")
    sun_up = (90.0 - sky.zenith_deg) > 5.0
    return mask & ~(sun_up & (output_mw == 0))


def clear_sky_peak_w_m2(*, point_id: str) -> float:
    """Return the highest clear-sky irradiance of any hour in a CAMS point's record.

    Args:
        point_id: The CAMS point's identifier.

    Returns:
        The highest hourly clear-sky irradiance, in watts per square metre.
    """
    frame = pl.read_parquet(CAMS_PATH, columns=["point_id", "clear_sky_ghi_w_m2"]).filter(
        pl.col("point_id") == point_id
    )
    return float(frame["clear_sky_ghi_w_m2"].max())  # ty: ignore[invalid-argument-type]


def grid_point_positions() -> pl.DataFrame:
    """Return the 18 grid points' identifiers and positions.

    Returns:
        One row per grid point, with its identifier, latitude, and longitude.
    """
    return (
        pl.read_parquet(CAMS_PATH, columns=["point_id", "latitude", "longitude"])
        .filter(pl.col("point_id").str.starts_with("gb_"))
        .unique()
        .sort("point_id")
    )


def nearest_grid_points(*, latitude: float, longitude: float, count: int) -> list[str]:
    """Return the identifiers of the `count` grid points nearest to a position.

    Args:
        latitude: Degrees north.
        longitude: Degrees east.
        count: How many points to return.

    Returns:
        The points' identifiers, nearest first, by great-circle distance.
    """
    points = grid_point_positions()
    lat1, lon1 = np.radians(latitude), np.radians(longitude)
    lat2 = np.radians(points["latitude"].to_numpy())
    lon2 = np.radians(points["longitude"].to_numpy())
    haversine = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    order = np.argsort(haversine)[:count]
    return [points["point_id"].to_list()[index] for index in order]


def window_half_hours() -> pl.Series:
    """Return the end time of every half-hour of the study window, as UTC datetimes.

    Returns:
        The half-hour end times of the window, as a UTC datetime series.
    """
    return pl.datetime_range(
        WINDOW_START + timedelta(minutes=30),
        WINDOW_END,
        interval="30m",
        time_unit="us",
        eager=True,
    ).alias("half_hour_end_time")


def grid_index(*, half_hour_end_time: np.ndarray) -> np.ndarray:
    """Return each half-hour's row in `window_half_hours()`; -1 where it is outside the window.

    Args:
        half_hour_end_time: The end time of each half-hour, as UTC datetimes.

    Returns:
        Each half-hour's row index in the window grid, -1 where outside the window.
    """
    start = np.datetime64(WINDOW_START.replace(tzinfo=None), "us")
    steps = (half_hour_end_time.astype("datetime64[us]") - start) // np.timedelta64(30, "m") - 1
    inside = (steps >= 0) & (steps < len(window_half_hours()))
    return np.where(inside, steps, -1).astype(int)


def on_grid(*, half_hour_end_time: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Place values given at some half-hours onto the full window grid, NaN elsewhere.

    Args:
        half_hour_end_time: The end time of each half-hour that has a value.
        values: One value (or one row of values) per half-hour in `half_hour_end_time`.

    Returns:
        An array with one row per half-hour of the window grid, NaN where no value was given.
    """
    index = grid_index(half_hour_end_time=half_hour_end_time)
    grid = np.full((len(window_half_hours()), *values.shape[1:]), np.nan)
    grid[index[index >= 0]] = values[index >= 0]
    return grid


def output_on_grid(*, bmu_id: str) -> np.ndarray:
    """Return a BMU's output in megawatts at every window half-hour, NaN where B1610 has none.

    Args:
        bmu_id: The BMU's identifier.

    Returns:
        The BMU's output in megawatts on the window grid.
    """
    grid = pl.DataFrame({"half_hour_end_time": window_half_hours()})
    joined = grid.join(output_by_bmu(bmu_id=bmu_id), on="half_hour_end_time", how="left")
    return joined["output_mw"].to_numpy()


INPUTS_DIR: Final[Path] = OUTPUT_DIR.joinpath("inputs")
"""The weather downloads that the study fits against."""
WEATHER_PATH: Final[Path] = INPUTS_DIR / "open_meteo_era5_gb_points.parquet"
"""Hourly ERA5 wind speed at 100 m and air temperature at 2 m at the 18 grid points."""
RATED_WIND_M_S: Final[float] = 12.0
"""Wind speed above which a turbine's power stops rising, so the covariate stops rising too."""
HEATING_THRESHOLD_C: Final[float] = 15.5
"""Air temperature below which the heating-degree covariate is positive."""
WEATHER_COLUMNS: Final[tuple[str, ...]] = ("wind", "wind_cubed", "temperature", "heating_degrees")


def weather_covariates_on_grid(*, point_ids: list[str] | None = None) -> np.ndarray:
    """Return four weather covariates at every half-hour of the window.

    The covariates are the GB-mean wind speed at 100 m scaled by `RATED_WIND_M_S` and capped at 1,
    its cube, the GB-mean air temperature, and the heating degrees below `HEATING_THRESHOLD_C`.
    An hour's values apply to both half-hours of the hour that ends at its label.

    Args:
        point_ids: The grid points to average. Defaults to all 18.

    Returns:
        An array with one row per half-hour of the window and the columns of `WEATHER_COLUMNS`.
    """
    weather = pl.read_parquet(WEATHER_PATH)
    if point_ids is not None:
        weather = weather.filter(pl.col("point_id").is_in(point_ids))
    hourly = (
        weather.group_by("time")
        .agg(pl.col("wind_speed_100m_m_s").mean(), pl.col("temperature_2m_c").mean())
        .sort("time")
    )
    grid = pl.DataFrame({"half_hour_end_time": window_half_hours()}).with_columns(
        time=pl.col("half_hour_end_time").dt.offset_by("-1us").dt.truncate("1h").dt.offset_by("1h")
    )
    joined = grid.join(hourly, on="time", how="left")
    wind = np.minimum(joined["wind_speed_100m_m_s"].to_numpy() / RATED_WIND_M_S, 1.0)
    temperature = joined["temperature_2m_c"].to_numpy()
    return np.column_stack(
        [wind, wind**3, temperature, np.maximum(HEATING_THRESHOLD_C - temperature, 0.0)]
    )


def clear_sky_variance_explained(
    *,
    point_ids: list[str],
    latitude: float,
    longitude: float,
    output: np.ndarray,
    ac_guess_mw: float,
    lag_half_hours: int,
) -> float:
    """Fit a plant to a cloudless sky and return the share of the output's changes it explains.

    The fit is the same kind as the all-sky fit: `fit_plant_to_changes` with a free orientation
    and no priors. Only the sky differs: it is CAMS's clear-sky irradiance, so the plant can follow
    the daily shape of the output but not cloud.

    Args:
        point_ids: The CAMS points whose clear-sky irradiance is averaged.
        latitude: The latitude of the sun, in degrees north.
        longitude: The longitude of the sun, in degrees east.
        output: The output on the window grid, in megawatts.
        ac_guess_mw: The starting AC capacity.
        lag_half_hours: The lag of the differences.

    Returns:
        The share of the variance of the output's changes explained, or NaN when no fit exists.
    """
    clear_sky = sky_at(
        hourly=hourly_cams(point_ids=point_ids, column="clear_sky_ghi_w_m2"),
        latitude=latitude,
        longitude=longitude,
    )
    on_sky = output[grid_index(half_hour_end_time=clear_sky.half_hour_end_time)]
    fit = fit_plant_to_changes(
        sky=clear_sky,
        output_mw=on_sky,
        orientation="free",
        ac_guess_mw=ac_guess_mw,
        lag_half_hours=lag_half_hours,
    )
    if fit is None:
        return float("nan")
    return change_variance_explained(
        sky=clear_sky,
        output_mw=on_sky,
        power_mw=plant_power_mw(sky=clear_sky, parameters=fit.parameters),
        lag_half_hours=lag_half_hours,
    )
