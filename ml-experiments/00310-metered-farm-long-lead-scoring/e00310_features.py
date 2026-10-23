# ruff: noqa: E501
"""Load a farm's rows for one day band and build the feature columns of experiment 300's rung six.

Every column exists at forecast time: ECMWF ensemble (00 UTC run) weather at the farm's nearest cell,
the plant forecast, the farm's own power `ceil(lead / 24)` days before the half-hour (so before the
run), and the clock. A band pools its days; a half-hour then appears once per day of the band, with
that day's run.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from typing import Final

import e00310_common as common
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
LAG_COLUMNS: Final[tuple[str, ...]] = ("lag_power", "lag_missing", "avail_ratio")
RUNG6: Final[tuple[str, ...]] = (
    *PLANT_INPUTS,
    "mu",
    "ghi_std",
    "plant_spread",
    *LAG_COLUMNS,
    "hour",
    "doy",
    "relative_humidity",
    "longwave",
    "precipitation",
    "pressure_surface",
    "wind10",
    "wind100",
    "pressure_msl",
    "z500",
    "csi",
    "lead_hours",
)
EXTRA_COLUMNS: Final[tuple[str, ...]] = (
    "relative_humidity",
    "longwave",
    "precipitation",
    "pressure_surface",
    "wind10",
    "wind100",
    "pressure_msl",
    "z500",
)


@dataclass
class BandFarm:
    """One farm's rows for one day band, before any plant scale is applied."""

    site: str
    band: str
    frame: pl.DataFrame
    unit_members: np.ndarray


@cache
def output_series(site: str) -> pl.DataFrame:
    """Return the farm's observed output over capacity and the clear-sky irradiance, by time."""
    return (
        pl.read_parquet(common.FARM_ROWS)
        .filter(pl.col("site") == site)
        .select("time", "output", "clear_sky")
        .sort("time")
    )


@cache
def extra_weather(site: str) -> pl.DataFrame:
    """Return the farm's extra ensemble variables keyed by (init_time, period_end)."""
    return pl.read_parquet(common.E270_DIR / f"extra_{site}.parquet").rename(
        {"valid_time": "period_end"}
    )


@cache
def load_band_farm(site: str, band: str) -> BandFarm:
    """Return a farm's day-band rows with lagged power, clear-sky index, and sun angles."""
    tables, units = [], []
    for day in common.BANDS[band]:
        tables.append(pl.read_parquet(common.table_path(site=site, day=day)))
        units.append(np.load(common.unit_path(site=site, day=day)))
    table = pl.concat(tables)
    unit = np.concatenate(units)
    series = output_series(site)
    table = table.join(series.select("time", "clear_sky"), on="time", how="left")
    table = table.with_columns(
        lag_days=(pl.col("lead_hours") / 24.0).ceil().cast(pl.Int64),
    ).with_columns(lag_time=pl.col("time") - pl.duration(days=1) * pl.col("lag_days"))
    lag = series.select(lag_time="time", lag_power="output")
    table = table.join(lag, on="lag_time", how="left")
    table = table.with_columns(
        lag_missing=pl.col("lag_power").is_null().cast(pl.Float64),
        lag_power=pl.col("lag_power").fill_null(0.0),
        elevation_deg=90.0 - pl.col("zenith_deg"),
        azimuth_sin=(pl.col("sun_azimuth_deg") * np.pi / 180).sin(),
        azimuth_cos=(pl.col("sun_azimuth_deg") * np.pi / 180).cos(),
        csi=(pl.col("ghi_mean") / pl.col("clear_sky").clip(lower_bound=1.0))
        .clip(0.0, 1.5)
        .fill_null(0.0),
    )
    return BandFarm(site=site, band=band, frame=table, unit_members=unit)


def scale_and_ceiling(*, farm: BandFarm, cutoff: datetime) -> tuple[float, float]:
    """Return the plant scale and output ceiling fitted on the farm's rows before `cutoff`."""
    rows = (farm.frame["time"] < cutoff).to_numpy()
    output = farm.frame["output"].to_numpy().astype(np.float64)[rows]
    x = farm.frame["unit_mean"].to_numpy()[rows]
    return float(np.sum(output * x) / np.sum(x**2)), float(
        np.quantile(output, common.CEILING_QUANTILE)
    )


def feature_frame(*, farm: BandFarm, scale: float, ceiling: float) -> pl.DataFrame:
    """Return the farm's rows with every rung-six column at the setting's plant scale and ceiling."""
    frame = farm.frame
    unit_mean = frame["unit_mean"].to_numpy()
    mu = np.minimum(scale * unit_mean, ceiling)
    # Plant forecast at the lagged time, from this band's own rows (any day's run of that half-hour).
    by_time = frame.group_by("time").agg(unit_lag=pl.col("unit_mean").mean())
    lagged = frame.select("lag_time").join(
        by_time.rename({"time": "lag_time"}), on="lag_time", how="left"
    )
    mu_lag = np.minimum(scale * lagged["unit_lag"].fill_null(0.0).to_numpy(), ceiling)
    usable = (frame["lag_missing"].to_numpy() == 0) & (mu_lag >= MIN_MU)
    ratio = np.where(
        usable,
        np.clip(frame["lag_power"].to_numpy() / np.maximum(mu_lag, MIN_MU), 0.0, RATIO_CLIP),
        1.0,
    )
    plant_spread = np.minimum(scale * farm.unit_members.std(axis=1), ceiling)
    out = frame.with_columns(
        mu=pl.Series(mu),
        plant_spread=pl.Series(plant_spread),
        avail_ratio=pl.Series(ratio),
        hour=(pl.col("time").dt.hour() + pl.col("time").dt.minute() / 60.0).cast(pl.Float64),
        doy=pl.col("time").dt.ordinal_day().cast(pl.Float64),
    )
    n = out.height
    out = out.join(extra_weather(farm.site), on=["init_time", "period_end"], how="left")
    if out.height != n:
        msg = f"farm {farm.site}: the weather join changed the row count"
        raise ValueError(msg)
    nulls = out.select(pl.col(list(RUNG6)).null_count()).row(0, named=True)
    if any(v > 0 for v in nulls.values()):
        msg = f"farm {farm.site} band {farm.band}: nulls in feature columns {nulls}"
        raise ValueError(msg)
    return out


def utc(year: int, month: int) -> datetime:
    """Return the first day of a month in UTC."""
    return datetime(year, month, 1, tzinfo=UTC)
