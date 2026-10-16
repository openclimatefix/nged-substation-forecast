"""Paths, series, lead bands, and the irradiance chain shared by the scripts of experiment 00053.

The experiment asks whether the Met Office UKV forecast tracks the CAMS solar anomaly better than
the ECMWF ensemble (ENS) does (`docs/ml-experiments/00053-ukv-vs-ens-solar-anomaly/result.md`).
The helpers are copied from experiment 00046's `e00046_common.py` and `e00046_solar_inputs.py`;
an experiment may not import another experiment.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import pvlib
from studies.pv_physics import PlantParameters, SunAndSky, plant_power_mw, solar_position
from studies.sources import ML_EXPERIMENTS_DATA_DIR, study_dir_for

EXPERIMENT_ID: Final[int] = 53
SLUG: Final[str] = "ukv-vs-ens-solar-anomaly"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""
RESULTS_DIR: Final[Path] = WORK_DIR / "results"
E46_WORK: Final[Path] = next(ML_EXPERIMENTS_DATA_DIR.glob("00046-*")) / "work"
"""Experiment 00046's private work folder, read only."""
E46_SOLAR: Final[Path] = E46_WORK / "solar"
STUDY_WEATHER_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR.parent / "studies" / "weather"
CAMS_FRAME_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00039-pv-disaggregation-cams-injection" / "work" / "frame.parquet"
)
SERIES_PATH: Final[Path] = (
    study_dir_for(study="stage1_pilot") / "demand" / "inputs" / "series.parquet"
)
METADATA_PATH: Final[Path] = (
    study_dir_for(study="quick_wins_fixed_pipeline_1156")
    / "frames"
    / "combined"
    / "fold_a"
    / "metadata.parquet"
)
ENS_ARCHIVE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR.parent / "NWP" / "nwp_model_id=ECMWF_ENS_0_25_degree"
)
MOGREPS_DIR: Final[Path] = Path("/mnt/wd_18tb/data/nwp-archive/MOGREPS-UK")

SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
VERDICT_SITES: Final[tuple[str, ...]] = ("D9", "D10", "D11", "D14", "D16")
DROPPED_SITES: Final[frozenset[str]] = frozenset({"D7", "D20"})
LAST_TRAINING_VALID: Final[datetime] = datetime(2025, 6, 30, 23, 30, tzinfo=UTC)
FIRST_TRAINING_ORIGIN: Final[datetime] = datetime(2024, 4, 1, 9, tzinfo=UTC)
VALIDATION_START: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
PUBLICATION_DELAY: Final[timedelta] = timedelta(hours=9)
"""The forecast origin of a daily 00 UTC ENS run, as in production."""
UKV_LATENCY: Final[timedelta] = timedelta(hours=3)
"""Assumed delay from a UKV run's initialisation to its availability at the forecast origin."""
SCORED_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
DAYLIGHT_THRESHOLD: Final[float] = 0.02
MIN_CLIMATOLOGY_DAYS: Final[int] = 30
DAYLIGHT_FLOOR_W_M2: Final[float] = 20.0
MAX_CLEAR_SKY_INDEX: Final[float] = 1.5
KEYS: Final[tuple[str, ...]] = ("site", "power_fcst_init_time", "valid_time")

BAND_EDGES: Final[tuple[float, ...]] = (0.0, 6.0, 12.0, 24.0, 48.0, 54.0, 120.0)
BAND_NAMES: Final[tuple[str, ...]] = (
    "0-6 h",
    "6-12 h",
    "12-24 h",
    "24-48 h",
    "48-54 h",
    "54-120 h",
)

FLEET: Final[PlantParameters] = PlantParameters(
    tilt_deg=35.0, azimuth_deg=180.0, dc_capacity_mw=1.15, ac_capacity_mw=1.0
)
"""The generic fleet whose output per MW of AC capacity is the unit PV shape `u`."""


def band_index(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the index into `BAND_NAMES` of each lead (hours from the forecast origin).

    Args:
        lead_hours: Hours from the forecast origin to the valid time.

    Returns:
        Integer band indices; leads of 120 h or more get index 6 and leads of 0 h or less -1.
    """
    return np.searchsorted(np.array(BAND_EDGES[1:]), lead_hours, side="right").astype(np.int64)


def write_atomically(*, frame: pl.DataFrame, target: Path) -> None:
    """Write a parquet file through a temporary name so a crash leaves no half-written file.

    Args:
        frame: The table.
        target: The final path.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    frame.write_parquet(temporary)
    temporary.rename(target)


def latest_time(*, times: pl.Series) -> datetime:
    """Return the latest timestamp of a datetime series.

    Args:
        times: A non-empty datetime series.

    Returns:
        Its maximum.
    """
    latest = times.max()
    assert isinstance(latest, datetime)
    return latest


def with_cell(*, frame: pl.DataFrame, time_column: str) -> pl.DataFrame:
    """Add the climatology cell of each row: the local calendar month and the UTC half-hour of day.

    Args:
        frame: Rows with a UTC datetime column.
        time_column: The name of that column (a period-ending stamp).

    Returns:
        The frame with `month` (Int8, Europe/London calendar month) and `hh` (Int8, 0 to 47).
    """
    stamp = pl.col(time_column)
    return frame.with_columns(
        month=stamp.dt.convert_time_zone("Europe/London").dt.month().cast(pl.Int8),
        hh=(stamp.dt.hour() * 2 + stamp.dt.minute() // 30).cast(pl.Int8),
    )


def series_table() -> pl.DataFrame:
    """Return the six sites' label, coordinates, and H3 cell; the coordinates stay in the process.

    Returns:
        One row per site in `SITES` with `time_series_id`, `site`, `latitude`, `longitude`,
        `h3_res_5`.
    """
    table = pl.read_parquet(SERIES_PATH)
    metadata = pl.read_parquet(METADATA_PATH, columns=["time_series_id", "h3_res_5"])
    return (
        table.filter(pl.col("site").is_in(SITES))
        .join(metadata, on="time_series_id", how="left", validate="1:1")
        .select("time_series_id", "site", "latitude", "longitude", "h3_res_5")
        .sort(pl.col("site").str.slice(1).cast(pl.Int32))
    )


def u_from_half_hour_ghi(
    *, stamps: pl.Series, ghi: np.ndarray, latitude: float, longitude: float
) -> np.ndarray:
    """Return the unit PV shape of half-hour mean irradiance, the chain CAMS goes through.

    The sun is at the midpoint of the half-hour ending at each stamp, the Erbs correlation splits
    the irradiance, and the generic fleet turns it into power per MW of AC capacity.

    Args:
        stamps: Distinct period-ending UTC stamps, shape (n,).
        ghi: Half-hour mean global horizontal irradiance in W/m2, shape (k, n); NaN where unknown.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        `u`, shape (k, n), NaN where `ghi` is NaN.
    """
    copies = ghi.shape[0]
    midpoints = stamps.dt.offset_by("-15m")
    zenith, sun_azimuth = solar_position(stamps=midpoints, latitude=latitude, longitude=longitude)
    day = midpoints.dt.ordinal_day().to_numpy()
    unknown = ~np.isfinite(ghi)
    flat = np.clip(np.where(unknown, 0.0, ghi), 0.0, None).ravel().astype(np.float64)
    split = pvlib.irradiance.erbs(
        ghi=flat, zenith=np.tile(zenith, copies), datetime_or_doy=np.tile(day, copies)
    )
    sky = SunAndSky(
        half_hour_end_time=np.tile(stamps.to_numpy(), copies),
        ghi_w_m2=flat,
        dni_w_m2=np.nan_to_num(np.asarray(split["dni"]).astype(np.float64)),
        dhi_w_m2=np.nan_to_num(np.asarray(split["dhi"]).astype(np.float64)),
        zenith_deg=np.tile(zenith, copies),
        sun_azimuth_deg=np.tile(sun_azimuth, copies),
    )
    shape = plant_power_mw(sky=sky, parameters=FLEET).reshape(copies, -1)
    return np.where(unknown, np.nan, shape)


def clear_sky_ghi(*, instants: pl.Series, latitude: float, longitude: float) -> np.ndarray:
    """Return the Haurwitz clear-sky global horizontal irradiance at each instant.

    Args:
        instants: UTC datetimes.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        Irradiance in W/m2, zero when the sun is below the horizon.
    """
    zenith, _ = solar_position(stamps=instants, latitude=latitude, longitude=longitude)
    cosine = np.cos(np.radians(zenith))
    positive = np.clip(cosine, 1e-6, None)
    return np.where(cosine > 0, 1098.0 * positive * np.exp(-0.059 / positive), 0.0)


def half_hour_ghi_from_instants(
    *,
    node_times: np.ndarray,
    node_ghi: np.ndarray,
    stamps: pl.Series,
    latitude: float,
    longitude: float,
) -> np.ndarray:
    """Turn instantaneous irradiance at steps into half-hour means in clear-sky-index space.

    The clear-sky index (irradiance over Haurwitz clear-sky irradiance at the same instant) is
    formed at each step whose clear-sky irradiance is at least `DAYLIGHT_FLOOR_W_M2`, interpolated
    linearly in time to the midpoint of each half-hour, and multiplied by the clear-sky irradiance
    at that midpoint. A midpoint more than 3 h from a usable step on either side takes the nearest
    step's index, so nothing is interpolated across a night.

    Args:
        node_times: Instants of the steps, sorted, shape (m,) as datetime64.
        node_ghi: Instantaneous irradiance at those steps in W/m2, NaN where unknown, shape (m,).
        stamps: Period-ending half-hour stamps as a UTC datetime series, shape (n,).
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        Half-hour mean irradiance in W/m2, shape (n,); NaN where no step is usable.
    """
    nodes = pl.Series(node_times).dt.replace_time_zone("UTC")
    node_clear = clear_sky_ghi(instants=nodes, latitude=latitude, longitude=longitude)
    usable = np.isfinite(node_ghi) & (node_clear >= DAYLIGHT_FLOOR_W_M2)
    midpoints = stamps.dt.offset_by("-15m")
    mid_clear = clear_sky_ghi(instants=midpoints, latitude=latitude, longitude=longitude)
    if not usable.any():
        return np.where(mid_clear > 0, np.nan, 0.0)
    index = np.clip(node_ghi[usable] / node_clear[usable], 0.0, MAX_CLEAR_SKY_INDEX)
    seconds = node_times[usable].astype("datetime64[s]").astype(np.int64)
    target = (
        midpoints.dt.replace_time_zone(None).to_numpy().astype("datetime64[s]").astype(np.int64)
    )
    position = np.searchsorted(seconds, target)
    left = np.clip(position - 1, 0, len(seconds) - 1)
    right = np.clip(position, 0, len(seconds) - 1)
    span = 3 * 3600
    near_left = np.abs(target - seconds[left]) <= span
    near_right = np.abs(seconds[right] - target) <= span
    interpolated = np.interp(target, seconds, index)
    nearest = np.where(
        np.abs(target - seconds[left]) <= np.abs(seconds[right] - target), index[left], index[right]
    )
    chosen = np.where(near_left & near_right, interpolated, nearest)
    return np.where(mid_clear > 0, chosen * mid_clear, 0.0)


def climatology(*, frame: pl.DataFrame, value: str, time_column: str, pool: bool) -> pl.DataFrame:
    """Return the month-by-half-hour mean of a column over training-window valid times.

    Args:
        frame: Columns `site`, a UTC time column, and the value column.
        value: The value column.
        time_column: The time column.
        pool: Pool each local month with its two neighbours when true.

    Returns:
        Columns `site`, `month`, `hh`, `clim` (the mean), and `days` (distinct UTC dates behind it).

    Raises:
        ValueError: If a row is later than the last training valid time.
    """
    training = frame.filter(pl.col(value).is_not_null() & pl.col(value).is_not_nan())
    if latest_time(times=training[time_column]) > LAST_TRAINING_VALID:
        msg = "climatology rows must be training-window valid times only"
        raise ValueError(msg)
    cells = with_cell(frame=training, time_column=time_column).with_columns(
        date=pl.col(time_column).dt.date()
    )
    if pool:
        cells = pl.concat(
            [
                cells.with_columns(month=((pl.col("month") - 1 + shift) % 12 + 1).cast(pl.Int8))
                for shift in (-1, 0, 1)
            ]
        )
    return cells.group_by("site", "month", "hh").agg(
        clim=pl.col(value).mean(), days=pl.col("date").n_unique()
    )
