"""Known-answer injections for experiment 110: PV, farms, generators, clips, and the meter.

Each function returns a new cache whose power for one substation has been changed by a quantity
whose size is known, so a fit that recovers the quantity shows that the model can see it.

**Privacy.** The farms A to F are metered generators, so their output enters here only as a series
scaled to a known capacity, and only ratios of recovered to injected capacity may leave the private
data folder.
"""

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Final

import e00110_common as common
import numpy as np
import polars as pl
from studies.dp_features import CAMS_PATH, FeatureCache
from studies.dp_gam import GamFit, clipped_pv, generator_path
from studies.pv_physics import PlantParameters, half_hour_sun_and_sky, plant_power_mw

FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "F")
"""The farms with data from 2019-09; farm E starts in 2024-02 and is left out."""
FARM_MAX_ZENITH_DEG: Final[float] = 75.0
FARM_DROPOUT_GHI_W_M2: Final[float] = 100.0
"""A farm half-hour at or below zero power while the irradiance exceeds this is a dropout."""
MIDDAY_HOURS_UTC: Final[tuple[int, int]] = (10, 14)

EAST_WEST_PLANTS: Final[tuple[PlantParameters, ...]] = (
    PlantParameters(tilt_deg=15.0, azimuth_deg=90.0, dc_capacity_mw=0.65, ac_capacity_mw=0.5),
    PlantParameters(tilt_deg=15.0, azimuth_deg=270.0, dc_capacity_mw=0.65, ac_capacity_mw=0.5),
)
"""An east-west mix: two halves of one AC megawatt, DC:AC 1.3, clipped at the inverter."""
SOUTH_WEST_PLANTS: Final[tuple[PlantParameters, ...]] = (
    PlantParameters(tilt_deg=25.0, azimuth_deg=225.0, dc_capacity_mw=1.3, ac_capacity_mw=1.0),
)
"""A plant facing south-west at tilt 25 degrees, DC:AC 1.3, clipped at the inverter."""


def epoch_day(*, cache: FeatureCache) -> np.ndarray:
    """Return the day of each half-hour's start, in days since 1970-01-01."""
    seconds = cache.end_times.astype("datetime64[s]").astype(np.int64)
    return np.floor(seconds / 86400.0 - 1.0 / 48.0).astype(np.int64)


def day_of(*, moment: datetime) -> int:
    """Return the days since 1970-01-01 of a date."""
    return int(np.datetime64(moment.replace(tzinfo=None), "D").astype(np.int64))


def with_reading(*, cache: FeatureCache, site: str, y: np.ndarray) -> FeatureCache:
    """Return the cache with one substation's reading replaced."""
    return replace(cache, sites={**cache.sites, site: replace(cache.sites[site], y=y)})


def plant_shape(
    *, cache: FeatureCache, site: str, plants: tuple[PlantParameters, ...]
) -> np.ndarray:
    """Return a plant's AC output per MW of AC capacity at the substation, on the cache's grid.

    The output uses the substation's CAMS irradiance, the plant's own tilt, azimuth, DC:AC ratio,
    and inverter clip, and no temperature derating. Several plants are summed, so their AC
    capacities should add to 1 MW.

    Args:
        cache: The cache, for the grid.
        site: The substation's label.
        plants: The plant, or the parts of a mix.

    Returns:
        The shape on the grid, `NaN` where CAMS is missing.
    """
    table = cache.sites[site].table
    hourly = (
        pl.read_parquet(CAMS_PATH)
        .filter(pl.col("site") == table.site)
        .select("time", pl.col("ghi_w_m2").cast(pl.Float64))
        .drop_nulls()
        .sort("time")
    )
    sky = half_hour_sun_and_sky(hourly=hourly, latitude=table.latitude, longitude=table.longitude)
    total = sum(plant_power_mw(sky=sky, parameters=plant) for plant in plants)
    frame = pl.DataFrame(
        {
            "time": pl.Series(sky.half_hour_end_time).dt.replace_time_zone("UTC"),
            "shape": np.asarray(total, dtype=np.float64),
        }
    )
    grid = pl.DataFrame({"time": cache.times_series()})
    joined = grid.join(frame, on="time", how="left", validate="1:1", maintain_order="left")
    return joined["shape"].to_numpy().astype(np.float64)


def inject_pv(
    *, cache: FeatureCache, site: str, shape: np.ndarray, capacity_points: float
) -> FeatureCache:
    """Subtract `capacity_points` times a per-MW-AC shape from a substation's power.

    A `NaN` in the shape makes the power `NaN` there, so the row drops out of training.
    """
    features = cache.sites[site]
    return with_reading(cache=cache, site=site, y=features.y - capacity_points * shape)


def farm_norm(*, cache: FeatureCache, farm: str) -> np.ndarray:
    """Return a farm's power divided by its 99th percentile, on the cache's grid.

    The 99th percentile is over the farm's daylight half-hours in all history, as in experiment 39.
    A daylight half-hour with no reading, or at or below zero power while the irradiance exceeds
    `FARM_DROPOUT_GHI_W_M2` (a dropout), returns `NaN`, so a substation's power becomes `NaN` there
    and the row drops out of training instead of carrying a farm that produced nothing. Remaining
    negative readings are set to zero. A half-hour outside the daylight rows of the farm's frame
    returns 0, because the farm injects nothing in the dark.

    Args:
        cache: The cache, for the grid.
        farm: The farm label (A to F).

    Returns:
        The normalised power on the grid.
    """
    frame = pl.read_parquet(common.INJECTION_FRAME_PATH).filter(
        pl.col("site") == farm, pl.col("kind") == "farm", pl.col("zenith") < FARM_MAX_ZENITH_DEG
    )
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
    return joined["norm"].fill_null(0.0).to_numpy().astype(np.float64)


def inject_farm(
    *, cache: FeatureCache, site: str, farm: str, capacity_points: float
) -> FeatureCache:
    """Subtract a farm's normalised output times `capacity_points` from a substation's power."""
    return inject_pv(
        cache=cache,
        site=site,
        shape=farm_norm(cache=cache, farm=farm),
        capacity_points=capacity_points,
    )


def midday_median_absolute(*, cache: FeatureCache, site: str, before: datetime) -> float:
    """Return the median absolute power at midday in the year before `before`, in points.

    Midday is the half-hours that start from 10:00 to before 14:00 UTC. This is experiment 79's
    injection level 1.0.
    """
    features = cache.sites[site]
    starts = cache.end_times - np.timedelta64(30, "m")
    hours = starts.astype("datetime64[h]").astype(np.int64) % 24
    start = common.to_64(moment=before.replace(year=before.year - 1))
    keep = (
        np.isfinite(features.y)
        & (hours >= MIDDAY_HOURS_UTC[0])
        & (hours < MIDDAY_HOURS_UTC[1])
        & (cache.end_times > start)
        & (cache.end_times <= common.to_64(moment=before))
    )
    return float(np.median(np.abs(features.y[keep])))


def generator_level(
    *,
    cache: FeatureCache,
    level: float,
    step_day: int | None = None,
    step_size: float = 0.0,
    outages: tuple[tuple[int, int], ...] = (),
) -> np.ndarray:
    """Return a flat generator's output on the grid: a level, one step, and outages.

    Args:
        cache: The cache, for the grid.
        level: The output before the step, in points.
        step_day: The day (days since 1970-01-01) from which the output is `level + step_size`.
        step_size: The step, in points.
        outages: Each outage as `(first_day, n_days)`; the output is zero on those days.

    Returns:
        The output at each half-hour.
    """
    day = epoch_day(cache=cache)
    output = np.full(len(day), level, dtype=np.float64)
    if step_day is not None:
        output = np.where(day >= step_day, level + step_size, output)
    for first, length in outages:
        output = np.where((day >= first) & (day < first + length), 0.0, output)
    return output


def inject_generator(*, cache: FeatureCache, site: str, output: np.ndarray) -> FeatureCache:
    """Subtract a generator's output from a substation's power."""
    return with_reading(cache=cache, site=site, y=cache.sites[site].y - output)


def inject_meter(
    *,
    cache: FeatureCache,
    site: str,
    signed: np.ndarray,
    demand: np.ndarray,
    tan_phi: float,
) -> FeatureCache:
    """Replace a substation's reading with `sqrt(P^2 + Q^2)`, `Q = tan_phi * demand`.

    Args:
        cache: The cache.
        site: The substation's label.
        signed: The signed real power `P` on the grid, `NaN` where there is none.
        demand: The demand `D` on the grid, whose multiple is the reactive power.
        tan_phi: The ratio of reactive power to demand.

    Returns:
        The cache with the apparent power as the reading.
    """
    reading = np.sqrt(signed**2 + (tan_phi * demand) ** 2)
    return with_reading(cache=cache, site=site, y=reading)


def clipped_total(
    *, pinned_pv: np.ndarray, scale: float, share_of_peak: float, beta: float
) -> tuple[np.ndarray, float]:
    """Return the clipped, scaled rooftop output, and the clip level in points.

    The truth has the model's own form: the pinned rooftop output times a capacity multiplier
    `scale`, clipped smoothly at `share_of_peak` times its peak, which falls in summer. Because the
    multiplier is inside the profile's grid, the profile can represent the truth exactly.

    Args:
        pinned_pv: The model's own rooftop solar output (the pinned capacity times the shape).
        scale: The capacity multiplier, such as 1.25.
        share_of_peak: The clip as a share of the scaled output's peak.
        beta: The smooth clip's sharpness, per point.

    Returns:
        The clipped output on the grid and the clip level.
    """
    total = scale * np.nan_to_num(pinned_pv, nan=0.0)
    level = share_of_peak * float(np.max(total))
    return clipped_pv(unclipped=total, p_max=level, beta=beta), level


@dataclass(frozen=True)
class NodeReadout:
    """What a node's level path says about an injected step and outage.

    Attributes:
        step_error_days: The estimated step day minus the true step day, in days.
        step_size: The estimated step: the mean level in the 20 days after minus the 20 before.
        outage_found: True if the level during the outage's interior days is at most half the
            level just before it.
        outage_level: The level just before the outage minus the level during it, in points.
    """

    step_error_days: float
    step_size: float
    outage_found: bool
    outage_level: float


def _level_at(*, path: np.ndarray, first_day: int, block_days: int, day: int) -> float:
    block = min(max((day - first_day) // block_days, 0), len(path) - 1)
    return float(path[block])


def node_readout(
    *,
    fit: GamFit,
    step_day: int,
    outage: tuple[int, int],
    window_days: int = 60,
    flank_days: int = 20,
) -> NodeReadout:
    """Read the node's level path around an injected step and an injected outage.

    The step is placed at the block with the largest positive change in the path within
    `window_days` of the true step day. The outage is read from the days strictly inside it, so a
    block that straddles the outage's edge does not count.

    Args:
        fit: A fit with a node.
        step_day: The true step day, in days since 1970-01-01.
        outage: The true outage as `(first_day, n_days)`.
        window_days: The search window for the step, either side of the true day.
        flank_days: The days before and after the step that the step size is read from.

    Returns:
        The readout.
    """
    node = fit.layout.spec.node
    if node is None:
        msg = "the fit has no node"
        raise ValueError(msg)
    path = generator_path(fit=fit)
    first_day = int(fit.layout.node_first_day)
    block = node.step_days

    def level(day: int) -> float:
        return _level_at(path=path, first_day=first_day, block_days=block, day=day)

    candidates = range(step_day - window_days, step_day + window_days + 1)
    changes = {day: level(day) - level(day - 1) for day in candidates}
    placed = max(changes, key=lambda day: changes[day])
    before = np.mean([level(step_day - k) for k in range(2, flank_days + 2)])
    after = np.mean([level(step_day + k) for k in range(2, flank_days + 2)])
    first, length = outage
    interior = range(first + (1 if length > 2 else 0), first + length - (1 if length > 2 else 0))
    during = float(np.mean([level(day) for day in interior]))
    just_before = float(np.mean([level(first - k) for k in range(2, 9)]))
    return NodeReadout(
        step_error_days=float(placed - step_day),
        step_size=float(after - before),
        outage_found=during <= 0.5 * just_before,
        outage_level=just_before - during,
    )
