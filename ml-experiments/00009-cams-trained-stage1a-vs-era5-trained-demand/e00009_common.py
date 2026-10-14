"""Paths, series, columns, and helpers shared by the scripts of experiment 00009.

The experiment compares a stage 1a XGBoost demand model trained on ERA5 shortwave with the same
model trained on CAMS irradiance, both applied to ENS members
(`docs/ml-experiments/00009-cams-trained-stage1a-vs-era5-trained-demand/result.md`). Everything
runs on one dense half-hourly grid per series, so a valid time is an index.
"""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

import holidays
import numpy as np
import polars as pl
import xgboost as xgb
from studies.solar import azimuth, zenith
from studies.sources import ML_EXPERIMENTS_DATA_DIR, PER_STUDY_DIR, STUDIES_DATA_DIR

EXPERIMENT_ID: Final[int] = 9
SLUG: Final[str] = "cams-trained-stage1a-vs-era5-trained-demand"
ROW_SET_ID: Final[str] = "fold_a_every5d_v1"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
ArmType = Literal["era5", "cams"]
ARMS: Final[tuple[ArmType, ...]] = ("era5", "cams")

EXPERIMENT_00005_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00005-lag-placement-stage1b-vs-stage2-demand"
)
EXPERIMENT_00006_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00006-stage1a-demand-input-feature-sweep"
)
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables and boosters; private and never committed."""
DEMAND_DIR: Final[Path] = PER_STUDY_DIR / "stage1_pilot" / "demand"
"""Experiment 00003's inputs, read only."""
DEMAND_SERIES_PATH: Final[Path] = DEMAND_DIR / "inputs" / "series.parquet"
ERA5_PATH: Final[Path] = DEMAND_DIR / "inputs" / "era5_features.parquet"
OBSERVED_PATH: Final[Path] = DEMAND_DIR / "inputs" / "observed.parquet"
VALIDATION_DIR: Final[Path] = DEMAND_DIR / "ens_every5d" / "val"
SERIES_LABELS_PATH: Final[Path] = EXPERIMENT_00006_DIR / "inputs" / "series.parquet"
FEATURE_SET_PATH: Final[Path] = EXPERIMENT_00005_DIR / "work" / "feature_set.json"
CAMS_PATH: Final[Path] = (
    STUDIES_DATA_DIR / "downloads" / "reanalysis" / "CAMS" / "demand_points"
) / "cams_demand_points.parquet"

GRID_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
GRID_END: Final[datetime] = datetime(2026, 7, 31, 23, 30, tzinfo=UTC)
STEP: Final[timedelta] = timedelta(minutes=30)
TRAIN_START: Final[datetime] = datetime(2023, 9, 1, tzinfo=UTC)
FINAL_TRAIN_CUTOFF: Final[datetime] = datetime(2025, 6, 17, tzinfo=UTC)
"""Training uses valid times before this: 14 days before the first validation run."""
VALIDATION_START: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)

LEVELS: Final[np.ndarray] = np.array([0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95])
LEVEL_COLUMNS: Final[tuple[str, ...]] = (
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
)
ATOM_MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (LEVELS[:-1] + LEVELS[1:]) / 2, [1.0]])
)
"""Probability mass of each level: the interval between the mid-points to its neighbours."""

DERIVED: Final[tuple[str, ...]] = (
    "solar_zenith",
    "solar_azimuth",
    "hour_local",
    "day_of_week_local",
    "covid_indicator",
    "days_to_nearest_holiday",
    "days_to_christmas",
)
"""Stage 1a columns computed from the timestamp and the location, not read from a weather file."""
LOCKDOWN_START: Final[datetime] = datetime(2020, 3, 23, tzinfo=UTC)
LOCKDOWN_END: Final[datetime] = datetime(2021, 7, 19, tzinfo=UTC)
HOLIDAY_YEARS: Final[range] = range(2018, 2029)
HOLIDAY_CLIP_DAYS: Final[int] = 7
CHRISTMAS_CLIP_DAYS: Final[int] = 14
MAX_ZENITH_DEGREES: Final[float] = 80.0
"""Daylight valid times have a solar zenith below this."""

XGBOOST_PARAMETERS: Final[dict[str, float | int | str]] = {
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "colsample_bynode": 1.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cuda",
    "nthread": 10,
}
"""Experiment 00005's settings with the thread count of this experiment's slot."""
N_ROUNDS: Final[int] = 500


def feature_columns() -> list[str]:
    """Return stage 1a's input columns, from the set experiment 00005 froze after 00006."""
    return json.loads(FEATURE_SET_PATH.read_text())["columns"]


def series_table() -> pl.DataFrame:
    """Return the 20 series with their demand label (D1 to D22), anonymous label, and capacity.

    The coordinates never leave this process.

    Returns:
        One row per series with `time_series_id`, `site` (the D label, which names the CAMS
        download), `label` (P1 to P16, B1, B2, G1, or G2), `substation_type`, `latitude`,
        `longitude`, and `effective_capacity_mw`.

    Raises:
        ValueError: If the table does not hold 20 series.
    """
    labels = pl.read_parquet(SERIES_LABELS_PATH).select(
        "time_series_id", "substation_type", label="site"
    )
    table = (
        pl.read_parquet(DEMAND_SERIES_PATH)
        .select("time_series_id", "site", "latitude", "longitude", "effective_capacity_mw")
        .join(labels, on="time_series_id", how="inner", validate="1:1")
    )
    if table.height != 20:
        msg = f"expected the 20 Primary, BSP, and GSP series, found {table.height}"
        raise ValueError(msg)
    return table.sort(pl.col("site").str.slice(1).cast(pl.Int32))


def grid_times() -> pl.Series:
    """Return the dense half-hourly grid every per-series array is indexed on."""
    return pl.datetime_range(
        GRID_START, GRID_END, interval="30m", time_unit="us", time_zone="UTC", eager=True
    )


def grid_index(*, times: datetime | pl.Series) -> int | np.ndarray:
    """Return the grid index of one timestamp, or of a series of timestamps."""
    if isinstance(times, datetime):
        return int((times - GRID_START) / STEP)
    return ((times.dt.epoch("us") - int(GRID_START.timestamp() * 1e6)) // 1_800_000_000).to_numpy()


def holiday_gaps() -> pl.DataFrame:
    """Return, for each calendar date, the signed days to the nearest holiday and to Christmas."""
    calendar = holidays.country_holidays("GB", subdiv="ENG", years=HOLIDAY_YEARS)
    holiday_dates = np.array(sorted(calendar), dtype="datetime64[D]")
    days = pl.date_range(
        date(HOLIDAY_YEARS.start, 1, 1), date(HOLIDAY_YEARS.stop - 1, 12, 31), eager=True
    )
    gaps = (holiday_dates[None, :] - days.to_numpy()[:, None]).astype("timedelta64[D]").astype(int)
    nearest = gaps[np.arange(days.len()), np.abs(gaps).argmin(axis=1)]
    christmas = np.array(
        [np.datetime64(f"{year}-12-25") for year in HOLIDAY_YEARS], dtype="datetime64[D]"
    )
    to_christmas = (
        (christmas[None, :] - days.to_numpy()[:, None]).astype("timedelta64[D]").astype(int)
    )
    nearest_christmas = to_christmas[np.arange(days.len()), np.abs(to_christmas).argmin(axis=1)]
    return pl.DataFrame(
        {
            "date": days,
            "days_to_nearest_holiday": np.clip(nearest, -HOLIDAY_CLIP_DAYS, HOLIDAY_CLIP_DAYS),
            "days_to_christmas": np.clip(
                nearest_christmas, -CHRISTMAS_CLIP_DAYS, CHRISTMAS_CLIP_DAYS
            ),
        }
    )


def derived_columns(*, valid_time: pl.Series, latitude: float, longitude: float) -> pl.DataFrame:
    """Return the timestamp-and-location columns of stage 1a for period-ending UTC stamps.

    The sun position is at the midpoint of the half-hour ending at each stamp. The local hour,
    weekday, and holiday distance use the stamp in Europe/London time.
    """
    midpoint = valid_time.dt.offset_by("-15m")
    local = valid_time.dt.convert_time_zone("Europe/London")
    frame = pl.DataFrame(
        {
            "solar_zenith": zenith(stamps=midpoint, latitude=latitude, longitude=longitude),
            "solar_azimuth": azimuth(stamps=midpoint, latitude=latitude, longitude=longitude),
            "hour_local": local.dt.hour().cast(pl.Float64)
            + local.dt.minute().cast(pl.Float64) / 60.0,
            "day_of_week_local": local.dt.weekday().cast(pl.Float64),
            "covid_indicator": (
                (valid_time >= LOCKDOWN_START) & (valid_time < LOCKDOWN_END + timedelta(days=1))
            ).cast(pl.Float64),
            "date": local.dt.date(),
        }
    )
    return (
        frame.join(holiday_gaps(), on="date", how="left", validate="m:1")
        .select(DERIVED)
        .cast(dict.fromkeys(DERIVED, pl.Float32))
    )


def with_derived(*, frame: pl.DataFrame, latitude: float, longitude: float) -> pl.DataFrame:
    """Add the `DERIVED` columns to a frame with a `valid_time` column, replacing any it has."""
    times = frame["valid_time"].unique().sort()
    table = derived_columns(valid_time=times, latitude=latitude, longitude=longitude).with_columns(
        valid_time=times
    )
    return frame.drop(DERIVED, strict=False).join(
        table, on="valid_time", how="left", validate="m:1"
    )


def cams_half_hourly(*, site: str) -> pl.DataFrame:
    """Return one site's CAMS irradiance on the half-hourly grid as `valid_time`, `cams`.

    A CAMS value is the mean over the hour ending at its stamp, so both half-hours of that hour
    (the one ending at the stamp and the one ending 30 minutes earlier) take the value.
    """
    hourly = pl.read_parquet(CAMS_PATH).filter(pl.col("site") == site).select("time", "ghi_w_m2")
    return (
        pl.DataFrame({"valid_time": grid_times()})
        .with_columns(
            hour=(pl.col("valid_time") - pl.duration(microseconds=1)).dt.truncate("1h")
            + pl.duration(hours=1)
        )
        .join(hourly, left_on="hour", right_on="time", how="left", validate="m:1")
        .select("valid_time", cams=pl.col("ghi_w_m2"))
    )


def fit_booster(*, features: np.ndarray, target: np.ndarray, seed: int) -> xgb.Booster:
    """Fit a squared-error XGBoost model with experiment 00003's hyperparameters."""
    return xgb.train(
        {**XGBOOST_PARAMETERS, "seed": seed},
        xgb.DMatrix(features, label=target),
        num_boost_round=N_ROUNDS,
    )


def pinball_quantile_score(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return twice the mass-weighted mean pinball loss of the 9 levels, per case."""
    difference = actual[:, None] - quantiles
    pinball = np.where(difference >= 0, LEVELS * difference, (LEVELS - 1) * difference)
    return 2.0 * (pinball * ATOM_MASSES).sum(axis=1)


def model_path(*, arm: str, seed: int, site: str) -> Path:
    """Return where one fitted booster is stored."""
    return WORK_DIR / "models" / arm / f"seed{seed}" / f"stage1a_{site}.ubj"
