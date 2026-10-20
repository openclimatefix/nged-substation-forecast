"""Training and prediction rows for experiment 111's XGBoost arms.

The rows follow SIMP1's no-lag recipe (experiment 102) and BG2's lagged member (experiment 87):

- **Training rows** are the ENS control member's rows of every daily run from 2024-04-01 to
  2026-06-29, with observed power over effective capacity as the label.
- **Features** are experiment 6's final Primary columns (seven weather columns and the solar
  zenith and azimuth, local hour and weekday, the lockdown indicator, and the days to Christmas)
  plus `lead_hours`; the lagged arms add BG2's three origin lags.
- **Start cuts.** Rows before a site's `first_usable` instant (experiment 103's start cut, which
  removes D18's commissioning months) never train.

The frames are the 821 daily runs of `e00111_common.FRAME_FOLDERS`. Everything is read once per
site and cached under `work/xgb/`.
"""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import e00111_common as common
import holidays
import numpy as np
import polars as pl
from studies.solar import azimuth, zenith
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

XGB_DIR: Final[Path] = common.WORK_DIR / "xgb"
FEATURE_SET_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00005-lag-placement-stage1b-vs-stage2-demand"
    / "work"
    / "feature_set.json"
)
OBSERVED_PATH: Final[Path] = (
    REPO_DATA_DIR / "studies/per_study/stage1_pilot/demand/inputs/observed.parquet"
)
GRID_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
STEP: Final[timedelta] = timedelta(minutes=30)
STEPS_PER_DAY: Final[int] = 48
NEWEST_AGE: Final[int] = 7
"""Half-hours between the origin and the newest usable observation: 3 hours plus the half-hour."""
LAG_COLUMNS: Final[tuple[str, ...]] = (
    "lag_origin",
    "lag_same_time_latest_day",
    "lag_same_time_latest_week",
)
WEATHER: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
)
DERIVED: Final[tuple[str, ...]] = (
    "solar_zenith",
    "solar_azimuth",
    "hour_local",
    "day_of_week_local",
    "covid_indicator",
    "days_to_nearest_holiday",
    "days_to_christmas",
)
LOCKDOWN_START: Final[datetime] = datetime(2020, 3, 23, tzinfo=UTC)
LOCKDOWN_END: Final[datetime] = datetime(2021, 7, 19, tzinfo=UTC)
HOLIDAY_YEARS: Final[range] = range(2018, 2029)
HOLIDAY_CLIP_DAYS: Final[int] = 7
CHRISTMAS_CLIP_DAYS: Final[int] = 14
KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
MAX_LEAD_HOURS: Final[float] = 168.0
FRAME_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id",
    "power_fcst_init_time",
    "valid_time",
    "ensemble_member",
    "power",
    *WEATHER,
    "lead_hours",
)


def feature_columns() -> list[str]:
    """Return experiment 6's final Primary input columns (weather and derived, no lead)."""
    columns: list[str] = json.loads(FEATURE_SET_PATH.read_text())["columns"]
    return columns


def input_columns(*, lagged: bool) -> list[str]:
    """Return the input columns in matrix order: features, `lead_hours`, and the lags if lagged."""
    return [*feature_columns(), "lead_hours", *(LAG_COLUMNS if lagged else ())]


def holiday_gaps() -> pl.DataFrame:
    """Return, per date, the signed days to the nearest bank holiday and to Christmas."""
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
            "days_to_christmas": np.where(
                np.abs(nearest_christmas) <= CHRISTMAS_CLIP_DAYS, nearest_christmas, np.nan
            ),
        }
    )


def derived_columns(*, valid_time: pl.Series, latitude: float, longitude: float) -> pl.DataFrame:
    """Return the timestamp-and-location columns for period-ending UTC stamps, float32."""
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
        frame.join(holiday_gaps(), on="date", how="left", validate="m:1", maintain_order="left")
        .select(DERIVED)
        .cast(dict.fromkeys(DERIVED, pl.Float32))
    )


def with_derived(*, frame: pl.DataFrame, latitude: float, longitude: float) -> pl.DataFrame:
    """Add the `DERIVED` columns, computed once per distinct valid time."""
    times = frame["valid_time"].unique().sort()
    table = derived_columns(valid_time=times, latitude=latitude, longitude=longitude).with_columns(
        valid_time=times
    )
    return frame.drop(DERIVED, strict=False).join(
        table, on="valid_time", how="left", validate="m:1", maintain_order="left"
    )


def grid_index(*, times: pl.Series) -> np.ndarray:
    """Return the dense-grid index of period-ending UTC stamps (grid start 2019-09-01 00:00)."""
    return ((times.dt.epoch("us") - int(GRID_START.timestamp() * 1e6)) // 1_800_000_000).to_numpy()


def take(*, values: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Return `values[index]`, NaN where the index falls off either end of the grid."""
    inside = (index >= 0) & (index < values.size)
    out = np.full(index.shape, np.nan, dtype=np.float32)
    out[inside] = values[index[inside]]
    return out


def origin_lags(*, power: np.ndarray, target: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Return BG2's three lags measured from the forecast origin, using only usable observations.

    Args:
        power: Observed power over effective capacity on the grid, NaN where unobserved.
        target: Grid index of each case's valid time.
        origin: Grid index of each case's initialisation time.

    Returns:
        Shape (cases, 3) in `LAG_COLUMNS` order: the newest usable value, the value at the latest
        earlier time with the valid time's half-hour of day, and with its weekday and half-hour.

    Raises:
        ValueError: If a valid time is not after its origin, or a lag is younger than the newest
            usable observation.
    """
    if (target <= origin).any():
        msg = "every valid time must be after its initialisation time"
        raise ValueError(msg)
    newest = origin - NEWEST_AGE
    gap = target - newest
    day = target - STEPS_PER_DAY * np.ceil(gap / STEPS_PER_DAY).astype(int)
    week = target - 7 * STEPS_PER_DAY * np.ceil(gap / (7 * STEPS_PER_DAY)).astype(int)
    if (day > newest).any() or (week > newest).any():
        msg = "a lag index is younger than the newest usable observation"
        raise ValueError(msg)
    return np.stack(
        [
            take(values=power, index=newest),
            take(values=power, index=day),
            take(values=power, index=week),
        ],
        axis=1,
    )


def power_array(*, time_series_id: int, effective_capacity_mw: float) -> np.ndarray:
    """Return a series' observed power over effective capacity on the dense grid."""
    n = int((datetime(2026, 7, 31, 23, 30, tzinfo=UTC) - GRID_START) / STEP) + 1
    grid = np.full(n, np.nan)
    observed = (
        pl.scan_parquet(OBSERVED_PATH)
        .filter(pl.col("time_series_id") == time_series_id)
        .select("valid_time", "power")
        .collect()
    )
    index = grid_index(times=observed["valid_time"])
    keep = (index >= 0) & (index < n) & observed["power"].is_finite().to_numpy()
    grid[index[keep]] = observed["power"].to_numpy()[keep] / effective_capacity_mw
    return grid


def control_rows(*, table_row: dict) -> pl.DataFrame:
    """Return one site's control-member rows of every daily run, at all leads, with derived columns.

    Args:
        table_row: The site's `time_series_id`, `latitude`, and `longitude`.

    Returns:
        Rows keyed by `KEYS` with `power` in MW (null where unobserved), the weather columns,
        `lead_hours`, and the derived columns.
    """
    pieces = [
        pl.read_parquet(common.frame_path(run=run), columns=list(FRAME_COLUMNS)).filter(
            (pl.col("time_series_id") == table_row["time_series_id"])
            & (pl.col("ensemble_member") == 0)
        )
        for run in common.run_dates()
    ]
    rows = pl.concat(pieces)
    return with_derived(
        frame=rows, latitude=table_row["latitude"], longitude=table_row["longitude"]
    )


def run_rows(*, run: datetime, table_row: dict) -> pl.DataFrame:
    """Return one run's rows for a site: the 51 members, leads up to 168 hours, sorted.

    Args:
        run: The run's 00:00 UTC initialisation time.
        table_row: The site's `time_series_id`, `latitude`, and `longitude`.

    Returns:
        Rows sorted by valid time and member, with the weather, `lead_hours`, and derived columns.

    Raises:
        ValueError: If a valid time lacks a member.
    """
    frame = (
        pl.read_parquet(common.frame_path(run=run), columns=list(FRAME_COLUMNS))
        .filter(
            (pl.col("time_series_id") == table_row["time_series_id"])
            & (pl.col("lead_hours") <= MAX_LEAD_HOURS)
        )
        .sort("valid_time", "ensemble_member")
    )
    members = frame["ensemble_member"].n_unique()
    if frame.height != frame["valid_time"].n_unique() * members:
        msg = f"the {run:%Y-%m-%d} run does not hold every member at every valid time"
        raise ValueError(msg)
    return with_derived(
        frame=frame, latitude=table_row["latitude"], longitude=table_row["longitude"]
    )
