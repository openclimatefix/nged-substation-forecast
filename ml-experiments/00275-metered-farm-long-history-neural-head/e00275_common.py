# ruff: noqa: E501
"""Shared constants and data preparation for experiment 275 (long past-weather history for H5).

Each farm has two kinds of rows. Ensemble rows come from experiment 224's band-0 tables (ECMWF
ensemble-mean weather at 12 to 36 h leads) and carry era 0. Past-weather rows are built here from
the farm's whole cleaned history, with CAMS irradiance and ERA5 temperature standing in for a
forecast, and carry era 1. Both kinds hold the same columns, so one network reads either. Farms
appear only as labels A to F.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import pvlib
from studies.ens_members import MIXTURE_LEVELS
from studies.pv_physics import SunAndSky, plane_of_array_w_m2
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00275-metered-farm-long-history-neural-head"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
PRED_DIR: Final[Path] = DATA_DIR / "predictions"
E224_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00224-metered-farm-dayahead-two-number"
FARM_ROWS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00218-metered-solar-farm-plant-fit" / "farm_rows_cams.parquet"
)
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
LEVELS: Final[np.ndarray] = np.array(MIXTURE_LEVELS)
SCORED_START: Final[datetime] = datetime(2025, 9, 1, tzinfo=UTC)
SCORED_END: Final[datetime] = datetime(2026, 9, 1, tzinfo=UTC)
VALIDATION_START: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
"""The last two tuning months select the training epoch."""
CEILING_QUANTILE: Final[float] = 0.995
N_BINS: Final[int] = 4
MIN_BIN_ROWS: Final[int] = 50
MIN_FORECAST_POWER: Final[float] = 0.02
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
SETTINGS: Final[tuple[str, ...]] = ("own_history", "zero_history")
NEURAL_ARMS: Final[tuple[str, ...]] = ("l0", "l1", "l2", "l3", "control")
POSTHOC_ARMS: Final[tuple[str, ...]] = ("l2_ens_offsets", "l2_pre_ens")
"""Post hoc explanation tests: l2 with offsets from ensemble rows only, and l2 with past rows before the ensemble era only."""
ENS_START: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
XGB_ARMS: Final[tuple[str, ...]] = ("x0", "x2")
HEAT_PER_KELVIN: Final[float] = 0.004
FLEET_TILT_DEG: Final[float] = 35.0
FLEET_AZIMUTH_DEG: Final[float] = 180.0
CELL_TEMPERATURE_RISE_K: Final[float] = 25.0
REFERENCE_CELL_TEMPERATURE_C: Final[float] = 25.0
MIN_ELEVATION_DEG: Final[float] = 0.0
RAW_FEATURES: Final[tuple[str, ...]] = (
    "ghi_mean",
    "ghi_std",
    "dni_mean",
    "dhi_mean",
    "temperature_mean",
    "elevation_deg",
    "azimuth_sin",
    "azimuth_cos",
    "lead_hours",
    "csi",
    "lag_power",
    "lag_missing",
)
"""The era flag is appended as the last column of every feature matrix, and dropped by arms without it."""


def unit_response(
    *,
    ghi: np.ndarray,
    temperature_c: np.ndarray,
    zenith: np.ndarray,
    azimuth: np.ndarray,
    doy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the fleet-shape unit response, beam, and diffuse (copied from experiment 224)."""
    split = pvlib.irradiance.erbs(ghi=ghi, zenith=zenith, datetime_or_doy=doy)
    dni = np.nan_to_num(np.asarray(split["dni"]))
    dhi = np.nan_to_num(np.asarray(split["dhi"]))
    sky = SunAndSky(
        half_hour_end_time=np.zeros(ghi.size),
        ghi_w_m2=ghi,
        dni_w_m2=dni,
        dhi_w_m2=dhi,
        zenith_deg=zenith,
        sun_azimuth_deg=azimuth,
    )
    poa = plane_of_array_w_m2(
        sky=sky, tilt_deg=FLEET_TILT_DEG, azimuth_deg=FLEET_AZIMUTH_DEG, tracker=False
    )
    unit = poa / 1000.0
    cell = temperature_c + CELL_TEMPERATURE_RISE_K * unit
    derating = np.clip(1.0 - HEAT_PER_KELVIN * (cell - REFERENCE_CELL_TEMPERATURE_C), 0.0, None)
    return unit * derating, dni, dhi


@dataclass
class Farm:
    """One farm's rows (ensemble and past weather), masks, and the plant scale and ceiling."""

    site: str
    frame: pl.DataFrame
    ens_train: np.ndarray
    ens_val: np.ndarray
    past_train: np.ndarray
    scored: np.ndarray
    own_scale: float
    own_ceiling: float


def _add_lags(table: pl.DataFrame) -> pl.DataFrame:
    """Add the farm's own power one day earlier (two when the lead is over 24 h) and sun columns."""
    power = table.select("time", "output")
    lag_frames = [
        power.select(
            time=pl.col("time") + pl.duration(days=days), **{f"lag{days}": pl.col("output")}
        )
        for days in (1, 2)
    ]
    table = table.join(lag_frames[0], on="time", how="left").join(
        lag_frames[1], on="time", how="left"
    )
    return table.with_columns(
        lag_power=pl.when(pl.col("lead_hours") <= 24).then(pl.col("lag1")).otherwise(pl.col("lag2"))
    ).with_columns(
        lag_missing=pl.col("lag_power").is_null().cast(pl.Float64),
        lag_power=pl.col("lag_power").fill_null(0.0),
        elevation_deg=90.0 - pl.col("zenith_deg"),
        azimuth_sin=(pl.col("sun_azimuth_deg") * np.pi / 180).sin(),
        azimuth_cos=(pl.col("sun_azimuth_deg") * np.pi / 180).cos(),
    )


def load_farm(site: str) -> Farm:
    """Return a farm's ensemble and past-weather rows with features, and the plant scale and ceiling."""
    sky = pl.read_parquet(FARM_ROWS).filter(pl.col("site") == site)
    ens = pl.read_parquet(E224_DIR / "members" / f"{site}_b0.parquet").join(
        sky.select("time", "clear_sky"), on="time", how="left"
    )
    ens = _add_lags(ens.sort("time"))
    # Past-weather rows: the lead is the one the ensemble run for that valid time would have.
    past = (
        sky.filter(
            pl.col("output").is_not_null()
            & (pl.col("elevation_deg") > MIN_ELEVATION_DEG)
            & (pl.col("time") < SCORED_START)
        )
        .sort("time")
        .with_columns(
            init_time=(pl.col("time") - pl.duration(hours=12)).dt.truncate("1d"),
        )
        .with_columns(lead_hours=(pl.col("time") - pl.col("init_time")).dt.total_minutes() / 60.0)
    )
    doy = past["time"].dt.ordinal_day().to_numpy()
    unit, dni, dhi = unit_response(
        ghi=past["ghi_w_m2"].to_numpy(),
        temperature_c=past["temperature_c"].to_numpy(),
        zenith=past["zenith_deg"].to_numpy(),
        azimuth=past["sun_azimuth_deg"].to_numpy(),
        doy=doy,
    )
    past = past.with_columns(
        ghi_mean=pl.col("ghi_w_m2"),
        temperature_mean=pl.col("temperature_c"),
        dni_mean=pl.Series(dni),
        dhi_mean=pl.Series(dhi),
        unit_mean=pl.Series(unit),
        ghi_std=pl.lit(float("nan")),
    ).select(
        "time",
        "init_time",
        "output",
        "lead_hours",
        "zenith_deg",
        "sun_azimuth_deg",
        "ghi_mean",
        "ghi_std",
        "temperature_mean",
        "dni_mean",
        "dhi_mean",
        "unit_mean",
        "clear_sky",
    )
    past = _add_lags(past)
    ens = ens.select(past.columns).with_columns(era=pl.lit(0.0))
    past = past.with_columns(era=pl.lit(1.0))
    table = pl.concat([ens, past]).with_columns(
        csi=(pl.col("ghi_mean") / pl.col("clear_sky").clip(lower_bound=1.0))
        .clip(0.0, 1.5)
        .fill_null(0.0)
    )
    table = fill_spread(table)
    time = table["time"]
    is_ens = (table["era"] == 0.0).to_numpy()
    ens_train = is_ens & (time < VALIDATION_START).to_numpy()
    ens_val = is_ens & ((time >= VALIDATION_START) & (time < SCORED_START)).to_numpy()
    past_train = ~is_ens & (time < VALIDATION_START).to_numpy()
    scored = is_ens & ((time >= SCORED_START) & (time < SCORED_END)).to_numpy()
    tuning = ens_train | ens_val
    output = table["output"].to_numpy().astype(np.float64)
    x = table["unit_mean"].to_numpy()
    scale = float(np.sum(output[tuning] * x[tuning]) / np.sum(x[tuning] ** 2))
    ceiling = float(np.quantile(output[tuning], CEILING_QUANTILE))
    return Farm(site, table, ens_train, ens_val, past_train, scored, scale, ceiling)


def fill_spread(table: pl.DataFrame) -> pl.DataFrame:
    """Give each past-weather row the median ensemble-era spread of its farm, month, half-hour, and cloud band.

    The medians use ensemble rows before the scored year only. The cloud band is the tercile of the
    clear-sky index among those rows. Cells with no ensemble row fall back to the farm's month and
    half-hour median, then to the overall median.
    """
    tuning = (table["era"] == 0.0) & (table["time"] < SCORED_START)
    ens = table.filter(tuning)
    edges = np.quantile(ens["csi"].to_numpy(), [1 / 3, 2 / 3])

    def keyed(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            month=pl.col("time").dt.month(),
            half_hour=pl.col("time").dt.hour() * 2 + pl.col("time").dt.minute() // 30,
            cloud_band=pl.Series(np.digitize(frame["csi"].to_numpy(), edges)),
        )

    ens_keyed = keyed(ens)
    full = ens_keyed.group_by("month", "half_hour", "cloud_band").agg(
        full_std=pl.col("ghi_std").median()
    )
    coarse = ens_keyed.group_by("month", "half_hour").agg(coarse_std=pl.col("ghi_std").median())
    overall = float(np.median(ens["ghi_std"].to_numpy()))
    keyed_all = (
        keyed(table.with_row_index("row"))
        .join(full, on=["month", "half_hour", "cloud_band"], how="left")
        .join(coarse, on=["month", "half_hour"], how="left")
        .sort("row")
    )
    filled = (
        pl.when(pl.col("era") == 0.0)
        .then(pl.col("ghi_std"))
        .otherwise(pl.coalesce(pl.col("full_std"), pl.col("coarse_std"), pl.lit(overall)))
    )
    return keyed_all.with_columns(ghi_std=filled).select(table.columns)


def plant_forecast(*, unit: np.ndarray, scale: float, ceiling: float) -> np.ndarray:
    """Return the capped plant forecast: scale times the unit response, at most the ceiling."""
    return np.minimum(scale * unit, ceiling)


def bin_offsets(*, mu: np.ndarray, output: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the bin edges of the forecast and the (bins, 11) residual quantiles per bin."""
    usable = mu > MIN_FORECAST_POWER
    edges = np.quantile(mu[usable], np.linspace(0, 1, N_BINS + 1)[1:-1])
    bins = np.digitize(mu, edges)
    residual = output - mu
    overall = np.quantile(residual, LEVELS)
    table = np.empty((N_BINS, len(LEVELS)))
    for index in range(N_BINS):
        rows = bins == index
        table[index] = (
            np.quantile(residual[rows], LEVELS) if rows.sum() >= MIN_BIN_ROWS else overall
        )
    return edges, table


def offsets_for(*, mu: np.ndarray, edges: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Return each row's 11 offsets from its bin of the forecast."""
    return table[np.digitize(mu, edges)]
