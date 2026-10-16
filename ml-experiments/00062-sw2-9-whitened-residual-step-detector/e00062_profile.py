"""The five injection shapes and the profile shares fitted to the labelled real transfers.

Copied from experiment 60, without its profile-weighted detector.
"""

from __future__ import annotations

import json
from typing import Final, Literal

import numpy as np
import polars as pl
from e00062_common import GRID_LENGTH, LABELLED_PATH, stamp_of
from e00062_inject import Event, Injection, _share

Shape = Literal["S1", "S2", "S3", "S4", "S5"]
SHAPES: Final[tuple[Shape, ...]] = ("S1", "S2", "S3", "S4", "S5")
SHAPE_NAMES: Final[dict[str, str]] = {
    "S1": "donor profile",
    "S2": "constant MW",
    "S3": "receiver profile",
    "S4": "fitted mix",
    "S5": "profile shifted 12 h",
}
NON_MATCHING: Final[tuple[Shape, ...]] = ("S2", "S3", "S4")
MIN_FITS: Final[int] = 10
"""Fewer usable fitted real transfers than this and S4 draws its profile share uniformly."""
MIN_FIT_HOURS: Final[float] = 6.0


def fit_profile_shares(*, panel: dict) -> list[float]:
    """The profile share lambda of each usable labelled real transfer, from the donor's residual.

    Least squares of the donor's residual during the transfer on a constant and its expected power
    relative to the transfer's mean; lambda is the profile term's share of the fitted mean, clipped
    to 0 to 1. Transfers under 6 hours, without a known donor series, or with a fitted mean that is
    not positive are dropped.
    """
    frame = pl.read_parquet(LABELLED_PATH).filter(
        ~pl.col("is_sentinel"),
        pl.col("kind") == "real_operation",
        pl.col("visibility").is_in(["conserved", "one_sided"]),
        pl.col("donor").is_not_null(),
    )
    shares: list[float] = []
    for row in frame.iter_rows(named=True):
        donor = int(row["donor"])
        onset, end = stamp_of(when=row["onset_utc"]), stamp_of(when=row["end_utc"]) + 1
        if donor not in panel["residual"] or (end - onset) / 2 < MIN_FIT_HOURS:
            continue
        y = panel["residual"][donor][onset:end]
        e = panel["expected"][donor][onset:end]
        ok = np.isfinite(y) & np.isfinite(e)
        if ok.sum() < 0.75 * (end - onset) or e[ok].mean() <= 0:
            continue
        sign = 1.0 if y[ok].mean() >= 0 else -1.0
        design = np.column_stack([np.ones(ok.sum()), e[ok] / e[ok].mean()])
        constant, slope = np.linalg.lstsq(design, sign * y[ok], rcond=None)[0]
        if constant + slope <= 0:
            continue
        shares.append(float(np.clip(slope / (constant + slope), 0.0, 1.0)))
    return shares


def inject_shaped(
    *,
    power: dict[int, np.ndarray],
    events: list[Event],
    seed: int,
    shape: Shape,
    shares: list[float],
) -> Injection:
    """Inject the events with one of the five shapes; the random draws match `inject_power`.

    Every shape uses the share and drift of experiment 58 (S1 is exactly its injection) and is
    rescaled to S1's mean MW moved per transfer. S2 moves the donor's mean power. S3 gives each
    metered receiver its weight times that level times its own power relative to its mean, and the
    donor loses the total plus the unmetered remainder, flat. S4 mixes S2 and S1, with a profile
    share drawn from `shares` per transfer (uniform on 0 to 1 when `shares` has under
    `MIN_FITS`). S5 is the donor's profile 12 hours ahead, a guard shape far from the donor's own.
    """
    rng = np.random.default_rng(seed)
    mix_rng = np.random.default_rng((seed, 1))
    out = {s: v.copy() for s, v in power.items()}
    delta = {s: np.zeros(GRID_LENGTH) for s in power}
    for event in events:
        share, donor = _share(event=event, power=power, rng=rng)
        stamps = np.arange(event.onset, event.end)
        level = share * float(donor.mean())
        gains: dict[int, np.ndarray] = {}
        if shape == "S1":
            moved = share * donor
        elif shape == "S2":
            moved = level
        elif shape == "S3":
            moved = level * max(1.0 - sum(event.weights), 0.0)
            for receiver, weight in zip(event.receivers, event.weights, strict=True):
                p = np.nan_to_num(power[receiver][stamps], nan=np.nanmedian(power[receiver]))
                relative = p / p.mean() if p.mean() > 0 else np.ones(len(stamps))
                gains[receiver] = weight * level * relative
                moved = moved + gains[receiver]
        elif shape == "S5":
            ahead = np.clip(stamps + 24, 0, GRID_LENGTH - 1)
            shifted = np.nan_to_num(power[event.donor][ahead], nan=np.nanmedian(power[event.donor]))
            moved = level * shifted / max(float(shifted.mean()), 1e-9)
        else:
            lam = (
                float(mix_rng.choice(shares))
                if len(shares) >= MIN_FITS
                else float(mix_rng.uniform(0.0, 1.0))
            )
            moved = level * ((1.0 - lam) + lam * donor / max(float(donor.mean()), 1e-9))
        target = float((share * donor).mean())
        if shape != "S1" and moved.mean() > 0:
            scale = target / float(moved.mean())
            moved, gains = moved * scale, {r: g * scale for r, g in gains.items()}
        out[event.donor][stamps] -= moved
        delta[event.donor][stamps] -= moved
        for receiver, weight in zip(event.receivers, event.weights, strict=True):
            gain = gains.get(receiver, weight * moved)
            out[receiver][stamps] += gain
            delta[receiver][stamps] += gain
    return Injection(power=out, delta=delta, events=events)


def shares_json(*, shares: list[float]) -> str:
    """The fitted profile shares as JSON, for the run log."""
    return json.dumps([round(x, 3) for x in shares])
