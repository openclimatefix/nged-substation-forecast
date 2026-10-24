# ruff: noqa: E501
"""Constants and the row assembly shared by the scripts of experiment 329.

Every input is a stored file of an earlier experiment (317, 319, 322), read and never written.
This experiment writes only under its own `WORK_DIR`.

Run every script with the folders of experiments 184, 202, 317, 319, 320, and 322 on `PYTHONPATH`.
"""

from pathlib import Path
from typing import Final

import e00317_score as s317  # ty: ignore[unresolved-import]
import e00319_arms as a319  # ty: ignore[unresolved-import]
import e00319_common as c319  # ty: ignore[unresolved-import]
import e00322_common as c322  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.baselines import haurwitz_w_m2
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 329
SLUG: Final[str] = "primary-demand-default-3-member-solar-distribution-at-7-to-10-days"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
ARMS_DIR: Final[Path] = WORK_DIR / "arms"
REPORT_DIR: Final[Path] = WORK_DIR / "report"
KEYS: Final[list[str]] = ["site", "power_fcst_init_time", "valid_time"]
QUANTILES: Final[tuple[str, ...]] = c319.QUANTILES
LEVELS: Final[np.ndarray] = c319.LEVELS
SITES: Final[tuple[str, ...]] = c319.SCORED_SITES
BAND_LABELS: Final[tuple[str, ...]] = ("168-192 h", "192-216 h", "216-240 h")
CSI_FORECAST_SUNNY: Final[float] = 0.9
SEED: Final[int] = 329


def load_default3(*, period: str) -> tuple[pl.DataFrame, np.ndarray]:
    """Return the long-lead rows of a period with Primary demand research default 3, and the member responses.

    The default's quantiles are experiment 322's `d1_*` columns. The frame also holds the chain's
    mean and quantiles (`mean`, `q01`..`q99`), experiment 319's arm `x1_*`, the flags `sunny`,
    `daylight`, `sunny_daylight`, `cold_winter_night`, and the row `period`.
    """
    frame, delta = a319.load_rows()
    if period == "scored":
        stored = pl.read_parquet(c322.ARMS_DIR / "scored_long_lead.parquet")
    else:
        stored = pl.read_parquet(c322.ARMS_DIR / "replicate_long_lead.parquet")
    stored = stored.select(
        *KEYS, "p_share", *[f"d1_{q}" for q in QUANTILES], *[f"x1_{q}" for q in QUANTILES]
    )
    keep = (frame["period"] == period).to_numpy()
    sub = frame.filter(pl.Series(keep)).with_row_index("i_row")
    delta = delta[keep]
    joined = sub.join(stored, on=KEYS, how="inner", validate="1:1")
    joined = s317.flags(rows=joined)
    order = joined["i_row"].to_numpy()
    return joined.drop("i_row"), delta[order]


def forecast_clear_sky_index(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the ensemble-mean shortwave over the Haurwitz clear-sky value, NaN at night."""
    import e00184_common as c184  # ty: ignore[unresolved-import]

    cache = c184.load_cache()
    out = np.full(frame.height, np.nan)
    for site in SITES:
        index = np.flatnonzero((frame["site"] == site).to_numpy())
        sub = frame[index]
        stamps = sub["valid_time"].dt.replace_time_zone(None).to_numpy()
        position = np.searchsorted(cache.end_times, stamps)
        elevation = cache.sites[site].elevation_deg[position]
        with np.load(c322.E111_STORE / f"{site}.npz") as stored:
            origins = stored["origins"]
            shortwave = stored["shortwave"]
        run = sub["run"].to_numpy().astype(np.int64)
        origin = sub["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
        if not (origins[run] == origin).all():
            msg = f"{site}: run does not index the ensemble store"
            raise ValueError(msg)
        lead = ((stamps - origin) // np.timedelta64(30, "m")).astype(np.int64) - 1
        ens = np.nanmean(shortwave[run, :, lead].astype(np.float64), axis=1)
        clear = haurwitz_w_m2(apparent_zenith_deg=90.0 - elevation)
        out[index] = np.where(clear > 20.0, ens / np.where(clear > 20.0, clear, 1.0), np.nan)
    return out


def pinball(*, actual: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return the quantile score of each row: twice the mean pinball loss over 11 levels."""
    return 2.0 * s317.pin(actual=actual, q=q).mean(axis=1)
