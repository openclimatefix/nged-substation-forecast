"""Load the scored rows of experiment 350: every arm's quantiles, the references, and the groups.

Rows are experiment 317's: six Primaries, rows with a reading outside the hand-filtered events,
and a stored forecast of both references (the manual heuristic and climatology), to 240 hours.
"""

from typing import Final

import e00317_score as s317  # ty: ignore[unresolved-import]
import e00350_common as common
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache

ROW_KEYS: Final[list[str]] = ["site", "power_fcst_init_time", "valid_time"]
SUN_ELEVATION_DEG: Final[float] = 5.0
SUNNY_QUANTILE: Final[float] = 0.75
"""A sunny half-hour is in the top quartile of its Primary's daylight half-hours by
ensemble-mean shortwave."""
E111_STORE = s317.common.E136_PREDS.parent.parent.parent / (
    "00111-differentiable-physics-6-ens-forecast-engine-against-bg2/work/ens_store"
)


def load_arm(*, arm: str, period: str) -> pl.DataFrame:
    """Return an arm's chain rows at the six Primaries for one period, outside switching masks."""
    pieces = [
        pl.read_parquet(common.CHAIN_DIR / arm / f"{site}.parquet")
        .filter(pl.col("period") == period)
        .with_columns(site=pl.lit(site))
        for site in common.SCORED_SITES
    ]
    frame = pl.concat(pieces)
    return frame.filter(~pl.col("switching_mask") & pl.col("actual").is_finite()).sort(ROW_KEYS)


def ensemble_mean_shortwave(*, frame: pl.DataFrame, site: str) -> np.ndarray:
    """Return each row's raw ensemble-mean shortwave (W/m2) from experiment 111's store."""
    with np.load(E111_STORE / f"{site}.npz") as stored:
        origins = stored["origins"]
        shortwave = stored["shortwave"]
    run = frame["run"].to_numpy().astype(np.int64)
    origin = frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
    if not (origins[run] == origin).all():
        msg = f"{site}: the run column does not index experiment 111's store"
        raise ValueError(msg)
    valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    position = ((valid - origin) // np.timedelta64(30, "m")).astype(np.int64) - 1
    out = np.full(frame.height, np.nan)
    for start in range(0, frame.height, 200_000):
        part = slice(start, start + 200_000)
        block = shortwave[run[part], :, position[part]].astype(np.float64)
        with np.errstate(all="ignore"):
            out[part] = np.nanmean(block, axis=1)
    return out


def add_sun_flags(*, frame: pl.DataFrame, cache: FeatureCache) -> pl.DataFrame:
    """Add `elevation_deg`, `ens_mean_shortwave`, `daylight`, and `sunny_daylight` to a frame.

    The sunny threshold is each Primary's 75th percentile of the ensemble-mean shortwave over its
    own daylight half-hours in the frame.
    """
    pieces = []
    for site in common.SCORED_SITES:
        part = frame.filter(pl.col("site") == site)
        positions = np.searchsorted(
            cache.end_times, part["valid_time"].dt.replace_time_zone(None).to_numpy()
        )
        if not (
            cache.end_times[positions] == part["valid_time"].dt.replace_time_zone(None).to_numpy()
        ).all():
            msg = f"{site}: a valid time is not on the cache's grid"
            raise ValueError(msg)
        elevation = cache.sites[site].elevation_deg[positions]
        shortwave = ensemble_mean_shortwave(frame=part, site=site)
        daylight = elevation > SUN_ELEVATION_DEG
        threshold = np.nanquantile(shortwave[daylight], SUNNY_QUANTILE)
        sunny = daylight & (shortwave >= threshold)
        pieces.append(
            part.with_columns(
                elevation_deg=pl.Series(elevation),
                ens_mean_shortwave=pl.Series(shortwave),
                daylight=pl.Series(daylight),
                sunny_daylight=pl.Series(sunny),
                sunny_threshold_w_m2=pl.lit(float(threshold)),
            )
        )
    return pl.concat(pieces).sort(ROW_KEYS)
