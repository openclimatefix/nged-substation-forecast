# ruff: noqa: E501
"""Load arm A's forecasts joined to the member deltas and row labels, for one period.

Arm A is experiment 164's fix O (experiment 169's arm A). The member deltas come from
`e00192_members.py`. Each row carries its site, solar elevation, unit PV shape, lead, local month,
and the member and spread columns.
"""

from pathlib import Path
from typing import Final

import e00192_common as common
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache
from studies.sources import ML_EXPERIMENTS_DATA_DIR

E164_PREDS_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00164-calibration-cold-night-tail-under-coverage"
    / "work"
    / "dp_preds"
)
MEMBERS_DIR: Final[Path] = common.E184_WORK_DIR / "members"
QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
HORIZON_HOURS: Final[float] = 168.0
DAYLIGHT_ELEVATION_DEG: Final[float] = 5.0
"""A half-hour is daylight when the sun's elevation at its midpoint is at or above this."""
NIGHT_ELEVATION_DEG: Final[float] = 0.0
"""A half-hour is night when the sun's elevation is under this; between the two is twilight."""
SUN_UP_UNIT_SHAPE: Final[float] = 0.01
"""Experiment 178's limit on the unit PV shape for a sunny-daylight row."""
STAT_COLUMNS: Final[tuple[str, ...]] = ("z_std", "z_range", "z_bimodal")


NEW_MEMBERS_DIR: Final[Path] = common.WORK_DIR / "members"
DELTAS: Final[tuple[str, ...]] = ("delta_m", "delta_t", "delta_st")
"""M: solar only (experiment 184's array); T: temperature only; ST: every input (184's `delta_f`)."""
COLD_C: Final[float] = 2.0
COLD_NIGHT_FIRST_HOUR: Final[int] = 17
COLD_NIGHT_LAST_HOUR: Final[int] = 8


def load_site(
    *, site: str, period: str, cache: FeatureCache
) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Return arm A's rows of one site and period, with labels, and the three member deltas.

    Args:
        site: The substation's label.
        period: `tuning`, `replicate`, or `scored`.
        cache: The feature cache.

    Returns:
        The frame and `delta_m`, `delta_t`, and `delta_st` (rows, 51), in the frame's row order.
    """
    a = pl.read_parquet(E164_PREDS_DIR / f"o_{period}_{site}.parquet").select(
        "run",
        "valid_time",
        "lead_hours",
        *QUANTILES,
        "actual",
        "mean",
        "hour_band",
        "quarter",
        "switching_mask",
    )
    with np.load(MEMBERS_DIR / f"{site}_{period}.npz") as stored:
        old = {name: stored[name] for name in ("run", "valid_time", "delta_m", "delta_f")}
    with np.load(NEW_MEMBERS_DIR / f"{site}_{period}.npz") as stored:
        new = {
            name: stored[name]
            for name in ("run", "valid_time", "delta_t", "delta_st", "t_mean", "t_member_std")
        }
    if not (
        np.array_equal(old["run"], new["run"])
        and np.array_equal(old["valid_time"], new["valid_time"])
    ):
        msg = f"{site} {period}: experiment 184's and this experiment's member rows differ"
        raise ValueError(msg)
    if not np.allclose(old["delta_f"], new["delta_st"], equal_nan=True, atol=1e-4):
        msg = f"{site} {period}: all-input responses differ from experiment 184's"
        raise ValueError(msg)
    keys = pl.DataFrame(
        {
            "run": new["run"].astype(np.int32),
            "valid_time": pl.Series(new["valid_time"]).dt.replace_time_zone("UTC"),
            "row": np.arange(len(new["run"])),
            "t_mean": new["t_mean"],
            "t_member_std": new["t_member_std"],
        }
    )
    joined = a.join(keys, on=["run", "valid_time"], how="inner", validate="1:1")
    row = joined["row"].to_numpy()
    index = np.searchsorted(
        cache.end_times, joined["valid_time"].dt.replace_time_zone(None).to_numpy()
    )
    features = cache.sites[site]
    local = joined["valid_time"].dt.convert_time_zone("Europe/London")
    joined = joined.with_columns(
        site=pl.lit(site),
        elevation=pl.Series(features.elevation_deg[index]),
        unit=pl.Series(features.pv_unit_shape[index]),
        month=local.dt.strftime("%Y-%m"),
        local_hour=local.dt.hour().cast(pl.Int32),
        hour=((local.dt.hour().cast(pl.Int32) * 60 + local.dt.minute().cast(pl.Int32) - 15) // 60),
    ).drop("row")
    deltas = {
        "delta_m": old["delta_m"][row],
        "delta_t": new["delta_t"][row],
        "delta_st": new["delta_st"][row],
    }
    return joined, deltas


def load_period(
    *, period: str, cache: FeatureCache, sites: tuple[str, ...] = common.SCORED_SITES
) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Return all sites' rows of a period as one frame and the stacked delta arrays."""
    frames, deltas = [], {name: [] for name in DELTAS}
    for site in sites:
        frame, d = load_site(site=site, period=period, cache=cache)
        frames.append(frame)
        for name in DELTAS:
            deltas[name].append(d[name])
    frame = pl.concat(frames)
    kind = (
        pl.when(pl.col("elevation") >= DAYLIGHT_ELEVATION_DEG)
        .then(pl.lit("daylight"))
        .when(pl.col("elevation") < NIGHT_ELEVATION_DEG)
        .then(pl.lit("night"))
        .otherwise(pl.lit("twilight"))
    )
    evening_to_morning = (pl.col("local_hour") >= COLD_NIGHT_FIRST_HOUR) | (
        pl.col("local_hour") < COLD_NIGHT_LAST_HOUR
    )
    frame = frame.with_columns(
        kind=kind,
        evening_to_morning=evening_to_morning,
        cold_night=evening_to_morning & (pl.col("t_mean") < COLD_C),
    )
    return frame, {name: np.concatenate(parts) for name, parts in deltas.items()}


def usable_rows(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the rows a score may use: a reading, outside switching events, leads to 168 hours."""
    return (
        ~frame["switching_mask"]
        & frame["actual"].is_finite()
        & (frame["lead_hours"] <= HORIZON_HOURS)
    ).to_numpy()
