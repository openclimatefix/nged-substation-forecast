"""Known-answer injections for experiment 152: a metered farm's output hidden in a five-MW Primary.

Copied from experiment 110's injection helpers. **Privacy.** The farms A to F are metered
generators, so their output enters only as a series scaled to a known capacity, and only
ratios or scores may leave the private data folder.
"""

from dataclasses import replace
from pathlib import Path
from typing import Final

import e00152_common as common
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache

INJECTION_FRAME_PATH: Final[Path] = (
    common.ML_EXPERIMENTS_DATA_DIR
    / "00039-pv-disaggregation-cams-injection"
    / "work"
    / "frame.parquet"
)
FARM_OF_SITE: Final[dict[str, str]] = {
    "D5": "A",
    "D10": "B",
    "D11": "C",
    "D14": "D",
    "D16": "F",
}
"""Each decision Primary gets a different farm, so no two tests share an injected series."""
FARM_MAX_ZENITH_DEG: Final[float] = 75.0
FARM_DROPOUT_GHI_W_M2: Final[float] = 100.0
SUN_UP_UNIT_SHAPE: Final[float] = 0.01
MIDDAY_HOURS_UTC: Final[tuple[int, int]] = (10, 14)
TAN_PHI_PF97: Final[float] = float(np.tan(np.arccos(0.97)))
"""The ratio of reactive to real power at a power factor of 0.97."""


def with_reading(*, cache: FeatureCache, site: str, y: np.ndarray) -> FeatureCache:
    """Return the cache with one substation's reading replaced."""
    return replace(cache, sites={**cache.sites, site: replace(cache.sites[site], y=y)})


def farm_norm(*, cache: FeatureCache, farm: str) -> np.ndarray:
    """Return a farm's power divided by its 99th percentile, on the cache's grid.

    `NaN` marks a half-hour where the farm's reading is missing or dropped out while the sun is up,
    so the substation's row leaves training; other negative readings are set to zero.
    """
    everything = pl.read_parquet(INJECTION_FRAME_PATH).filter(
        pl.col("site") == farm, pl.col("kind") == "farm"
    )
    frame = everything.filter(pl.col("zenith") < FARM_MAX_ZENITH_DEG)
    power = frame.filter(pl.col("power").is_not_null())
    p99 = float(np.nanpercentile(power["power"].to_numpy(), 99))
    dropout = (pl.col("power") <= 0.0) & (pl.col("ghi_u") > FARM_DROPOUT_GHI_W_M2)
    daylight = frame.select(
        "time",
        norm=pl.when(pl.col("power").is_null() | dropout)
        .then(float("nan"))
        .otherwise(pl.col("power").clip(lower_bound=0.0) / p99),
    )
    grid = pl.DataFrame({"time": cache.times_series()})
    joined = grid.join(daylight, on="time", how="left", validate="1:1", maintain_order="left")
    norm = joined["norm"].fill_null(0.0).to_numpy().astype(np.float64)
    in_frame = grid.join(everything.select("time"), on="time", how="semi")["time"]
    missing = ~np.isin(cache.times_series().to_numpy(), in_frame.to_numpy())
    sun_up = np.max(
        [np.nan_to_num(site.pv_unit_shape, nan=0.0) for site in cache.sites.values()], axis=0
    )
    norm[missing & (sun_up > SUN_UP_UNIT_SHAPE)] = np.nan
    return norm


def midday_median_absolute(*, cache: FeatureCache, site: str) -> float:
    """Return the median absolute reading at midday in the year before the scored year, in points.

    Midday is the half-hours that start from 10:00 to before 14:00 UTC. This is experiment 79's
    injection level 1.0.
    """
    features = cache.sites[site]
    starts = cache.end_times - np.timedelta64(30, "m")
    hours = starts.astype("datetime64[h]").astype(np.int64) % 24
    stop = np.datetime64("2025-07-01T00:00", "us")
    start = np.datetime64("2024-07-01T00:00", "us")
    keep = (
        np.isfinite(features.y)
        & (hours >= MIDDAY_HOURS_UTC[0])
        & (hours < MIDDAY_HOURS_UTC[1])
        & (cache.end_times > start)
        & (cache.end_times <= stop)
    )
    return float(np.median(np.abs(features.y[keep])))


def signed_and_folded(
    *, cache: FeatureCache, site: str, level: float, power_factor: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return the signed series with the farm hidden, and the apparent power a meter would read.

    The hidden farm's capacity is `level` times the site's midday median absolute reading. The
    apparent power is `|P|` for zero reactive power, or `sqrt(P^2 + (tan_phi * D)^2)` with the
    pre-injection reading as the demand `D` when `power_factor` is set.
    """
    original = cache.sites[site].y
    capacity = level * midday_median_absolute(cache=cache, site=site)
    signed = original - capacity * farm_norm(cache=cache, farm=FARM_OF_SITE[site])
    if power_factor:
        folded = np.sqrt(signed**2 + (TAN_PHI_PF97 * np.abs(original)) ** 2)
    else:
        folded = np.abs(signed)
    return signed, folded
