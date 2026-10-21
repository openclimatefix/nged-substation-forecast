"""Injections whose moved power follows the donor's convex-model demand or solar term.

Every arm uses the draws of experiment 58's injection (the same share and drift, so the events and
random numbers match `inject_power`) and rescales the moved power at each event to the mean MW the
share-of-net-power arm (`SHARE`, exactly `inject_power`) moves. The arms differ only in the daily
shape of the moved power:

- `SHARE`: a drifting share of the donor's net power (experiment 58, as in experiments 141 and 156).
- `CONST`: a constant.
- `DEMAND`: proportional to the donor's demand term of the convex model.
- `SOLAR`: proportional to the donor's solar output (zero at night).
- `MIX`: half the mean MW demand-shaped and half solar-shaped.

A donor without convex-model terms (series 12 and 13) falls back to its own net power as the shape
for `DEMAND`, `SOLAR`, and `MIX`; the count of such events is stored beside each world.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from e00162_h_common import GRID_LENGTH, WORK_DIR
from e00162_h_inject import Event, Injection, _share, inject_power

ARMS: Final[tuple[str, ...]] = ("SHARE", "CONST", "DEMAND", "SOLAR", "MIX")
TERMS_DIR: Final = WORK_DIR / "terms"
_TERMS: dict[int, dict[str, np.ndarray]] = {}


def terms(*, series: int) -> dict[str, np.ndarray] | None:
    """Demand term and solar output (both positive MW) of a series, or None when it has none."""
    if series not in _TERMS:
        path = TERMS_DIR / f"terms_{series}.npz"
        if not path.exists():
            return None
        with np.load(path) as stored:
            _TERMS[series] = {"demand": stored["demand"], "solar": -stored["solar"]}
    return _TERMS[series]


def _unit(*, values: np.ndarray) -> np.ndarray | None:
    """`values` over its mean with gaps filled by the median; None if unusable."""
    if not np.isfinite(values).any():
        return None
    filled = np.nan_to_num(values, nan=float(np.nanmedian(values)))
    mean = float(filled.mean())
    return filled / mean if mean > 1e-9 else None


def shape_of(
    *, arm: str, donor: int, stamps: np.ndarray, net: np.ndarray
) -> tuple[np.ndarray, bool]:
    """The unit-mean daily shape of the moved power, and whether the net-power fallback was used."""
    ones = np.ones(len(stamps))
    if arm == "CONST":
        return ones, False
    t = terms(series=donor)
    demand = _unit(values=t["demand"][stamps]) if t else None
    solar = _unit(values=t["solar"][stamps]) if t else None
    fallback = _unit(values=net) if arm != "SHARE" else None
    if arm == "DEMAND":
        out = demand
    elif arm == "SOLAR":
        out = solar
    else:
        out = 0.5 * (demand + solar) if demand is not None and solar is not None else None
    if out is None:
        return (fallback if fallback is not None else ones), True
    return out, False


def inject_arm(
    *, power: dict[int, np.ndarray], events: list[Event], seed: int, arm: str
) -> tuple[Injection, int]:
    """Inject the events with one arm; returns the injection and the count of fallback events."""
    if arm == "SHARE":
        return inject_power(power=power, events=events, seed=seed), 0
    rng = np.random.default_rng(seed)
    out = {s: v.copy() for s, v in power.items()}
    delta = {s: np.zeros(GRID_LENGTH) for s in power}
    fallbacks = 0
    for event in events:
        share, donor_power = _share(event=event, power=power, rng=rng)
        stamps = np.arange(event.onset, event.end)
        target = float((share * donor_power).mean())
        unit, used_fallback = shape_of(arm=arm, donor=event.donor, stamps=stamps, net=donor_power)
        fallbacks += int(used_fallback)
        moved = target * unit
        out[event.donor][stamps] -= moved
        delta[event.donor][stamps] -= moved
        for receiver, weight in zip(event.receivers, event.weights, strict=True):
            out[receiver][stamps] += weight * moved
            delta[receiver][stamps] += weight * moved
    return Injection(power=out, delta=delta, events=events), fallbacks
