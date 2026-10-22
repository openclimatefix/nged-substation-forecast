"""The gated level boost of experiment 190: the two-state filter with a run counter.

`gated_stream` runs the four hour bands of a residual stream in time order, so that one counter can
read the one-step innovations of all bands together. With the boost off (`Gate.mode == "none"`) it
reproduces experiment 136's `filter_stream` with a constant observation variance.

**The counter.** After each reading's innovation is known, the counter is on when the last `k`
innovations (readings, which skip missing half-hours), pooled over the bands, share one sign and their mean exceeds `z` times the root mean
square of the 48 innovations before those `k`. It is off when fewer than 48 earlier innovations
exist, when the sign flips, or when the run is shorter than `k`.

**The boost.** While the counter is on, the level's process-noise increment since the band's
previous reading is multiplied by `m`. The mode `always` multiplies at every reading. The mode
`shared` also adds the level increment of the band that was updated to the other three bands'
levels while the counter is on.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Final, Literal

import numpy as np
from e00190_filters import FilterSettings, StreamRun, _update
from studies.dp_assimilation import (
    INITIAL_LEVEL_VARIANCE,
    MIN_SIGMA_EPS,
    CorrectionTrace,
)
from studies.dp_calendar import HALF_HOURS_PER_DAY
from studies.dp_gam import N_BANDS

SPREAD_READINGS: Final[int] = 48
"""The innovations before the run that give the day-before spread (one day of half-hours)."""

ModeType = Literal["none", "gated", "always", "shared"]


@dataclass(frozen=True)
class Gate:
    """One setting of the gated boost.

    Attributes:
        mode: `none` (no boost), `gated`, `always`, or `shared` (gated, with a shared level).
        m: The multiplier on the level's process noise while the boost is on.
        k: The number of consecutive same-sign innovations (half-hours) that switch the counter on.
        z: The run's mean must exceed this multiple of the day-before innovation spread.
    """

    mode: ModeType = "none"
    m: float = 5.0
    k: int = 6
    z: float = 1.0


@dataclass(frozen=True)
class GatedRun:
    """A gated filter's output.

    Attributes:
        stream: The correction trace and the (unused here) per-band arrays of the stream run.
        trigger_positions: The grid positions at which the counter switched from off to on.
        on_positions: The grid positions of the readings processed while the counter was on.
    """

    stream: StreamRun
    trigger_positions: np.ndarray
    on_positions: np.ndarray


def counter_is_on(*, history: list[float], gate: Gate) -> bool:
    """Return True when the newest `k` innovations form a same-sign run above the threshold."""
    k = gate.k
    if len(history) < k + SPREAD_READINGS:
        return False
    recent = np.asarray(history[-k:])
    if not (np.all(recent > 0.0) or np.all(recent < 0.0)):
        return False
    before = np.asarray(history[-k - SPREAD_READINGS : -k])
    spread = sqrt(float(np.mean(before * before)))
    return abs(float(recent.mean())) > gate.z * spread


def gated_stream(  # noqa: PLR0912, PLR0915 (one loop holds the four bands' state)
    *, residual: np.ndarray, band: np.ndarray, settings: FilterSettings, gate: Gate
) -> GatedRun:
    """Run the filter over a site's residual stream in time order and return the trace.

    Args:
        residual: Observed minus forecast on the grid, in points, `NaN` where there is no reading.
        band: The hour band of each grid row.
        settings: The filter's setting (`rho_a` must be 0: the observation variance is constant).
        gate: The gate's setting.

    Returns:
        The run, with the trace and the counter's trigger positions.

    Raises:
        ValueError: If `settings.rho_a` is not 0.
    """
    if settings.rho_a != 0.0:
        msg = "the gated stream holds the observation variance constant"
        raise ValueError(msg)
    phi = settings.phi
    stationary = settings.sigma_z**2 / (1.0 - phi**2)
    noise = max(settings.sigma_eps**2, MIN_SIGMA_EPS**2)
    q_step = settings.q / HALF_HOURS_PER_DAY
    positions_all = np.flatnonzero(np.isfinite(residual))
    level = np.zeros(N_BANDS)
    anomaly = np.zeros(N_BANDS)
    covariance = [(INITIAL_LEVEL_VARIANCE, 0.0, stationary) for _ in range(N_BANDS)]
    previous = [-1] * N_BANDS
    shared = gate.mode == "shared"
    # Per-band traces: own readings only, or every reading's position for the shared mode.
    own_positions: list[list[int]] = [[] for _ in range(N_BANDS)]
    own_level: list[list[float]] = [[] for _ in range(N_BANDS)]
    own_anomaly: list[list[float]] = [[] for _ in range(N_BANDS)]
    history: list[float] = []
    was_on = False
    triggers: list[int] = []
    on_positions: list[int] = []
    last_time = None
    for position in positions_all.tolist():
        b = int(band[position])
        if shared and last_time is not None:
            # Decay every band's transient to this position (the stored value is at `position`).
            step = position - last_time
            anomaly *= phi**step
        gap = position - previous[b] if previous[b] >= 0 else 0
        p_ll, p_lz, p_zz = covariance[b]
        if gap and not shared:
            decay = phi**gap
            anomaly[b] *= decay
            p_lz *= decay
            p_zz = decay * decay * p_zz + stationary * (1.0 - decay * decay)
        elif gap:
            decay = phi**gap
            p_lz *= decay
            p_zz = decay * decay * p_zz + stationary * (1.0 - decay * decay)
        innovation = float(residual[position]) - level[b] - anomaly[b]
        history.append(innovation)
        on = gate.mode == "always" or (
            gate.mode in ("gated", "shared") and counter_is_on(history=history, gate=gate)
        )
        if gate.mode in ("gated", "shared"):
            if on and not was_on:
                triggers.append(position)
            was_on = on
        if on:
            on_positions.append(position)
        multiplier = gate.m if on else 1.0
        p_ll += multiplier * q_step * gap
        d_level, d_anomaly, covariance[b] = _update(
            covariance=(p_ll, p_lz, p_zz), noise=noise, innovation=innovation
        )
        level[b] += d_level
        anomaly[b] += d_anomaly
        if shared and on:
            for other in range(N_BANDS):
                if other != b:
                    level[other] += d_level
        previous[b] = position
        last_time = position
        if shared:
            for j in range(N_BANDS):
                own_positions[j].append(position)
                own_level[j].append(float(level[j]))
                own_anomaly[j].append(float(anomaly[j]))
        else:
            own_positions[b].append(position)
            own_level[b].append(float(level[b]))
            own_anomaly[b].append(float(anomaly[b]))
    trace = CorrectionTrace(
        kind="kalman",
        positions=tuple(np.asarray(p, dtype=np.int64) for p in own_positions),
        level=tuple(np.asarray(v) for v in own_level),
        anomaly=tuple(np.asarray(v) for v in own_anomaly),
        phi=phi,
        log_likelihood=0.0,
    )
    stream = StreamRun(
        trace=trace,
        log_variance=tuple(np.zeros(len(p)) for p in own_positions),
        innovation_rms=tuple(np.zeros(len(p)) for p in own_positions),
    )
    return GatedRun(
        stream=stream,
        trigger_positions=np.asarray(triggers, dtype=np.int64),
        on_positions=np.asarray(on_positions, dtype=np.int64),
    )
