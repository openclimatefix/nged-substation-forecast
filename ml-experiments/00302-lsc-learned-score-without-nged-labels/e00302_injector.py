# ruff: noqa: E501
"""The injector: draws transfers from distributions fitted to the real events, outside a held-out fold.

A drawn event has a pair of Primaries (a donor and its receivers), a duration, a depth in noise
units, a shape, an onset hour and weekday, and the receivers' opposite share. The fit uses the
66 labelled real transfers, the logged closures, and D8's 29 drops, and leaves out every event
whose Primary pair is in `held_out_pairs`. Nothing here reads a window or a network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import polars as pl
from e00302_common import (
    ADJACENCY,
    D8_DROPS,
    D8_SERIES,
    DAY,
    LOGGED_PATH,
    MAD_TO_STD,
    real_transfers,
    stamp_hour_weekday,
)

Shape = Literal["constant", "demand", "pv", "ramp"]
SHAPES: Final[tuple[Shape, ...]] = ("constant", "demand", "pv", "ramp")
SHAPE_PROBABILITIES: Final[tuple[float, ...]] = (0.35, 0.30, 0.15, 0.20)
"""Experiment 162 found the real transfers' daily shape no better fitted than quiet stretches', so the
shape mix is a prior spread over the four shapes, and the realism check tests it."""
DEPTH_BANDWIDTH: Final[float] = 0.30
"""Standard deviation, in natural-log units, of the jitter added to a resampled real depth."""
DURATION_BANDWIDTH: Final[float] = 0.25
P_LOG_UNIFORM_DURATION: Final[float] = 0.20
P_LOGGED_DURATION: Final[float] = 0.15
"""Share of events whose duration is resampled from the logged closures; the rest come from the labelled transfers and D8's drops."""
"""Share of events whose duration is log-uniform between 6 hours and 19 days (D8-sized)."""
DURATION_RANGE_H: Final[tuple[float, float]] = (6.0, 19 * 24.0)
P_EXTRA_NEIGHBOUR: Final[float] = 0.35
LATITUDE_DEG: Final[float] = 52.5


@dataclass(frozen=True)
class InjectorFit:
    """The empirical distributions an event is drawn from."""

    pairs: tuple[tuple[int, ...], ...]
    """Series tuples of the real transfers outside the held-out fold (with repeats)."""
    log_depth_z: np.ndarray
    """Natural log of each real depth divided by the noise unit of its Primaries (donor unknown)."""
    durations_h: np.ndarray
    """Durations of the logged closures (2 hours to 30 days) outside the fold."""
    long_durations_h: np.ndarray
    """Durations of the labelled transfers and of D8's drops, in hours."""
    onset_hours: np.ndarray
    onset_weekdays: np.ndarray
    p_conserved: float
    n_real: int


def _pair_key(*, series: tuple[int, ...]) -> str:
    return ",".join(str(s) for s in sorted(series))


def fit_injector(*, held_out_pairs: frozenset[str], sigma: dict[int, float]) -> InjectorFit:
    """Fit the injector on every real event whose pair is not in `held_out_pairs`."""
    transfers = real_transfers().filter(~pl.col("pair").is_in(list(held_out_pairs)))
    pairs, log_depth = [], []
    for row in transfers.iter_rows(named=True):
        series = tuple(s for s in row["series"] if np.isfinite(sigma.get(s, np.nan)))
        if not series or row["moved_mw_p50"] is None:
            continue
        pairs.append(tuple(row["series"]))
        unit = float(np.mean([sigma[s] for s in series]))
        log_depth.append(np.log(max(row["moved_mw_p50"], 0.05) / unit))
    drops = pl.read_parquet(D8_DROPS)
    held = real_transfers().filter(pl.col("pair").is_in(list(held_out_pairs)))
    overlaps = [
        any(
            r["onset"] <= d["last_stamp"] and d["first_stamp"] <= r["end"]
            for r in held.iter_rows(named=True)
        )
        for d in drops.iter_rows(named=True)
    ]
    drops = drops.filter(~pl.Series(overlaps, dtype=pl.Boolean))
    log_depth += [np.log(depth / sigma[D8_SERIES]) for depth in drops["depth_mw"].to_list()]
    long_durations = transfers["hours"].to_list() + (drops["duration_days"] * 24).to_list()
    logged = pl.read_parquet(LOGGED_PATH).filter(pl.col("hours").is_between(2, 30 * 24))
    keep = []
    for row in logged.iter_rows(named=True):
        sides = tuple(sorted(int(s) for s in (row["side_a"], row["side_b"]) if str(s).isdigit()))
        keep.append(_pair_key(series=sides) not in held_out_pairs)
    logged = logged.filter(pl.Series(keep))
    hours, weekdays = stamp_hour_weekday()
    stamps = logged["first_stamp"].to_numpy()
    return InjectorFit(
        pairs=tuple(pairs),
        log_depth_z=np.array(log_depth),
        durations_h=logged["hours"].to_numpy(),
        long_durations_h=np.array(long_durations),
        onset_hours=hours[np.clip(stamps, 0, len(hours) - 1)],
        onset_weekdays=weekdays[np.clip(stamps, 0, len(weekdays) - 1)],
        p_conserved=float((transfers["visibility"] == "conserved").mean()),  # ty: ignore[invalid-argument-type]
        n_real=transfers.height,
    )


@dataclass(frozen=True)
class DrawnEvent:
    """One drawn transfer, before it is placed in time."""

    donor: int
    receivers: tuple[int, ...]
    weights: tuple[float, ...]
    """Share of the moved MW each metered receiver takes; the rest goes to unmetered load."""
    hours: float
    depth_z: float
    """Mean moved MW divided by the donor's noise unit."""
    shape: Shape
    ramp_hours: float
    ramp_down: bool
    onset_hour: float
    onset_weekday: int


def draw_event(
    *, fit: InjectorFit, sigma: dict[int, float], rng: np.random.Generator
) -> DrawnEvent | None:
    """Draw one event, or None when the drawn pair has no series with a noise unit."""
    pair = fit.pairs[int(rng.integers(len(fit.pairs)))]
    sides = [s for s in pair if s in ADJACENCY]
    if not sides:
        return None
    donor = int(rng.choice(sides))
    if not np.isfinite(sigma.get(donor, np.nan)):
        return None
    receivers = [s for s in sides if s != donor]
    extras = [n for n in ADJACENCY[donor] if n not in receivers and n != donor]
    rng.shuffle(extras)
    if rng.random() < P_EXTRA_NEIGHBOUR and extras:
        receivers += extras[: int(rng.integers(1, 3))]
    receivers = [r for r in receivers if np.isfinite(sigma.get(r, np.nan))]
    if receivers and rng.random() < fit.p_conserved:
        weights = rng.dirichlet(np.ones(len(receivers)))
    elif receivers:
        weights = rng.dirichlet(np.ones(len(receivers))) * rng.uniform(0.0, 0.7)
    else:
        weights = np.array([])
    u = rng.random()
    if u < P_LOG_UNIFORM_DURATION:
        low, high = np.log(DURATION_RANGE_H[0]), np.log(DURATION_RANGE_H[1])
        hours = float(np.exp(rng.uniform(low, high)))
    elif u < P_LOG_UNIFORM_DURATION + P_LOGGED_DURATION:
        hours = float(rng.choice(fit.durations_h) * np.exp(rng.normal(0, DURATION_BANDWIDTH)))
    else:
        hours = float(rng.choice(fit.long_durations_h) * np.exp(rng.normal(0, DURATION_BANDWIDTH)))
    hours = float(np.clip(hours, 2.0, 30 * 24.0))
    depth_z = float(np.exp(rng.choice(fit.log_depth_z) + rng.normal(0, DEPTH_BANDWIDTH)))
    j = int(rng.integers(len(fit.onset_hours)))
    return DrawnEvent(
        donor=donor,
        receivers=tuple(receivers),
        weights=tuple(float(w) for w in weights),
        hours=hours,
        depth_z=depth_z,
        shape=SHAPES[int(rng.choice(len(SHAPES), p=SHAPE_PROBABILITIES))],
        ramp_hours=float(rng.uniform(1.0, 24.0)),
        ramp_down=bool(rng.random() < 0.5),
        onset_hour=float(fit.onset_hours[j]),
        onset_weekday=int(fit.onset_weekdays[j]),
    )


def _pv_profile(*, local_hours: np.ndarray, day_of_year: np.ndarray) -> np.ndarray:
    """A daylight bell: zero before sunrise and after sunset, one at solar noon."""
    declination = np.radians(23.44) * np.sin(2 * np.pi * (day_of_year - 81) / 365.0)
    latitude = np.radians(LATITUDE_DEG)
    cos_h0 = np.clip(-np.tan(latitude) * np.tan(declination), -1.0, 1.0)
    day_length = 2 * np.degrees(np.arccos(cos_h0)) / 15.0
    return np.clip(np.cos(np.pi * (local_hours - 12.0) / np.maximum(day_length, 1.0)), 0.0, None)


def moved_profile(
    *,
    event: DrawnEvent,
    length: int,
    donor_power: np.ndarray,
    local_hours: np.ndarray,
    day_of_year: np.ndarray,
    sigma_donor: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """The MW moved at each of `length` stamps; its mean is `depth_z` noise units of the donor."""
    t = np.arange(length)
    if event.shape == "constant":
        base = np.ones(length)
    elif event.shape == "demand":
        if np.isfinite(donor_power).any():
            filled = np.nan_to_num(donor_power, nan=np.nanmedian(donor_power))
            base = np.clip(filled, 0.05 * max(float(np.mean(filled)), 1e-6), None)
        else:
            base = np.ones(length)
    elif event.shape == "pv":
        base = _pv_profile(local_hours=local_hours, day_of_year=day_of_year) + 0.02
    else:
        up = np.clip((t + 1) / max(event.ramp_hours * 2, 1.0), 0.0, 1.0)
        down = np.clip((length - t) / max(event.ramp_hours * 2, 1.0), 0.0, 1.0)
        base = np.minimum(up, down) if event.ramp_down else up
    drift = np.exp(np.clip(np.cumsum(rng.normal(0, 0.01, length)), np.log(0.7), np.log(1.4)))
    base = base * drift
    return event.depth_z * sigma_donor * base / max(float(base.mean()), 1e-9)


def robust_scale(*, values: np.ndarray) -> float:
    """1.4826 times the median absolute deviation of the finite values."""
    chosen = values[np.isfinite(values)]
    return float(MAD_TO_STD * np.median(np.abs(chosen - np.median(chosen))))


def stamps_per_day() -> int:
    """Half-hours in a day, exported for callers that size windows in days."""
    return DAY
