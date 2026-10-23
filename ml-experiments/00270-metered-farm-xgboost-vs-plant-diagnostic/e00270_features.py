"""Feature groups for experiment 270's ladder, built on experiment 243's farm rows.

Every column exists at forecast time: ECMWF ensemble (00 UTC run, leads 12 to 36 h) weather at the
farm's nearest cell, the plant forecast, the farm's own power one or two days earlier, and the
clock. The groups are cumulative rungs of the ladder.
"""

from functools import cache
from typing import Final

import e00270_common as common
import numpy as np
import polars as pl

MIN_MU: Final[float] = 0.02
RATIO_CLIP: Final[float] = 1.5
PLANT_INPUTS: Final[tuple[str, ...]] = (
    "ghi_mean",
    "dni_mean",
    "dhi_mean",
    "temperature_mean",
    "elevation_deg",
    "azimuth_sin",
    "azimuth_cos",
)
GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "plant_forecast": ("mu",),
    "spread": ("ghi_std", "plant_spread"),
    "lag": ("lag_power", "lag_missing", "avail_ratio"),
    "calendar": ("hour", "doy"),
    "weather": ("relative_humidity", "longwave", "precipitation", "pressure_surface"),
    "rest": ("wind10", "wind100", "pressure_msl", "z500", "csi", "lead_hours"),
}
RUNGS: Final[tuple[str, ...]] = ("plant_forecast", "spread", "lag", "calendar", "weather", "rest")
"""Rung `Lk` holds the plant inputs plus the first k groups of `RUNGS`."""


def rung_columns(*, rung: int) -> list[str]:
    """Return the columns of ladder rung `rung` (1 to 6)."""
    columns = list(PLANT_INPUTS)
    for name in RUNGS[:rung]:
        columns += GROUPS[name]
    return columns


@cache
def members_unit(site: str) -> np.ndarray:
    """Return the 51 members' unit plant responses, rows aligned with experiment 243's frame."""
    unit = np.load(common.E224_DIR / "members" / f"{site}_b0_unit.npy")
    raw = pl.read_parquet(common.E224_DIR / "members" / f"{site}_b0.parquet")
    if not raw["time"].is_sorted():
        msg = f"farm {site}: member rows are not sorted by time"
        raise ValueError(msg)
    if np.abs(unit.mean(axis=1) - raw["unit_mean"].to_numpy()).mean() > 0.02:
        msg = f"farm {site}: member responses are not aligned with unit_mean"
        raise ValueError(msg)
    return unit


def availability(*, frame: pl.DataFrame, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the availability ratio (lagged power over the plant forecast then) and its flag."""
    t = frame["time"].dt.epoch("us").to_numpy()
    days = np.where(frame["lead_hours"].to_numpy() <= 24, 1, 2)
    lag_t = t - days * 86_400_000_000
    idx = np.minimum(np.searchsorted(t, lag_t), len(t) - 1)
    found = (t[idx] == lag_t) & (frame["lag_missing"].to_numpy() == 0)
    mu_lag = mu[idx]
    usable = found & (mu_lag >= MIN_MU)
    ratio = np.where(
        usable,
        np.clip(frame["lag_power"].to_numpy() / np.maximum(mu_lag, MIN_MU), 0.0, RATIO_CLIP),
        1.0,
    )
    return ratio, (~usable).astype(np.float64)


@cache
def extra_weather(site: str) -> pl.DataFrame:
    """Return the farm's extra ensemble variables keyed by (init_time, period_end)."""
    return pl.read_parquet(common.DATA_DIR / f"extra_{site}.parquet").rename(
        {"valid_time": "period_end"}
    )


def feature_frame(*, farm: common.Farm, scale: float, ceiling: float) -> pl.DataFrame:
    """Return the farm's rows with every ladder column at the setting's plant scale and ceiling."""
    frame = farm.frame
    mu = common.plant_forecast(farm=farm, scale=scale, ceiling=ceiling)
    ratio, _ = availability(frame=frame, mu=mu)
    plant_spread = np.minimum(scale * members_unit(farm.site).std(axis=1), ceiling)
    out = (
        frame.with_columns(
            period_end=(pl.col("time") - pl.duration(microseconds=1)).dt.truncate("3h")
            + pl.duration(hours=3),
            mu=pl.Series(mu),
            plant_spread=pl.Series(plant_spread),
            avail_ratio=pl.Series(ratio),
            hour=(pl.col("time").dt.hour() + pl.col("time").dt.minute() / 60.0).cast(pl.Float64),
            doy=pl.col("time").dt.ordinal_day().cast(pl.Float64),
        )
        .join(extra_weather(farm.site), on=["init_time", "period_end"], how="left")
        .sort("time")
    )
    if out.height != frame.height:
        msg = f"farm {farm.site}: the weather join changed the row count"
        raise ValueError(msg)
    all_columns = [c for name in RUNGS for c in GROUPS[name]] + list(PLANT_INPUTS)
    nulls = out.select(pl.col(all_columns).null_count()).row(0, named=True)
    if any(n > 0 for n in nulls.values()):
        msg = f"farm {farm.site}: nulls in ladder columns {nulls}"
        raise ValueError(msg)
    return out
