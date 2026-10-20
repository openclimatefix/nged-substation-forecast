"""A batched Kalman filter grid for the DP series: every filter setting at once, on the GPU.

**`studies.dp_assimilation.run_kalman` filters one site's residual stream with one setting of the
four noise hyperparameters, in a Python loop.** Tuning the filter means running it for every site
and every candidate setting, so the loop dominates the run time. This module runs all of them in
one vectorised pass: the loop over readings stays, but each step updates every stream, every
setting, and every hour band together, so a GPU (or the CPU, as a fallback) does the work of
thousands of `run_kalman` calls per step.

**The arithmetic is the same as `run_kalman`'s** (the exact discretisation of the level's random
walk and the transient's AR(1) over the gap between readings, with the same variance floors), in
`float64`, so the two agree to about 1e-12 on the test cases. `torch` is the only dependency
beyond `studies.dp_assimilation`'s constants.

**Rolling-origin cuts need no extra runs.** The filter is causal, so its state after a reading
depends on that reading and the earlier ones only. A forecast at any initialisation therefore
looks up the state after the newest reading that the information cut allows, which
`correction_at_cutoffs` does for a whole batch of cuts at once.

**Shapes.** `S` streams (sites), `H` settings, `B` hour bands, `K` readings per band (the longest
band, with shorter bands padded).
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch

from studies.dp_assimilation import (
    INITIAL_LEVEL_VARIANCE,
    VARIANCE_FLOOR,
)
from studies.dp_calendar import HALF_HOURS_PER_DAY

DEFAULT_DEVICE: Final[str] = "cuda"
"""The device that `resolve_device` prefers; it falls back to the CPU without CUDA."""

_LOG_TWO_PI: Final[float] = float(np.log(2.0 * np.pi))

PADDING_POSITION: Final[int] = np.iinfo(np.int64).max
"""The position of a padded slot: later than any cutoff, so a look-up never reaches it."""


def resolve_device(*, device: str | None = None) -> torch.device:
    """Return the device to use, falling back to the CPU when CUDA is unavailable.

    Args:
        device: A torch device name, or `None` for CUDA when it is available and the CPU otherwise.

    Returns:
        The device.
    """
    if device is None:
        device = DEFAULT_DEVICE
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


@dataclass(frozen=True)
class BandSlots:
    """The grid positions of each hour band's readings, padded to one length.

    Attributes:
        positions: Shape `(B, K)`, int64. The ascending grid positions of each band's rows, padded
            at the end with `PADDING_POSITION`.
        n_rows: Shape `(B,)`. How many real rows each band has.
    """

    positions: np.ndarray
    n_rows: np.ndarray


def band_slots(*, band: np.ndarray, n_bands: int) -> BandSlots:
    """Group the grid rows by hour band, ascending, padded to the longest band.

    Args:
        band: The hour band of each grid row.
        n_bands: The number of bands.

    Returns:
        The slots.
    """
    rows = [np.flatnonzero(band == index) for index in range(n_bands)]
    n_rows = np.array([len(row) for row in rows])
    width = max(int(n_rows.max()), 1)
    positions = np.full((n_bands, width), PADDING_POSITION, dtype=np.int64)
    for index, row in enumerate(rows):
        positions[index, : len(row)] = row
    return BandSlots(positions=positions, n_rows=n_rows)


@dataclass(frozen=True)
class GridTrace:
    """The filter's state after every slot, for every stream, setting, and band.

    A slot with no reading (padding, or a missing residual) carries the previous state forward, so
    the state at a slot is the state after the band's newest reading at or before that slot.

    Attributes:
        slots: The band slots the trace is laid out on.
        level: Shape `(S, H, B, K)`. The level after each slot.
        anomaly: Shape `(S, H, B, K)`. The transient after each slot, at the last reading's time.
        last_reading: Shape `(S, B, K)`, int64. The grid position of the newest reading at or
            before each slot, or -1 where the stream has none yet.
        log_likelihood: Shape `(S, H)`. The marginal log likelihood of each stream's readings,
            summed over the bands.
        phi: Shape `(H, B)`. The transient's AR(1) factor per half-hour.
    """

    slots: BandSlots
    level: torch.Tensor
    anomaly: torch.Tensor
    last_reading: torch.Tensor
    log_likelihood: torch.Tensor
    phi: torch.Tensor


def _per_band(*, hyper: np.ndarray, n_bands: int) -> np.ndarray:
    """Return the settings as shape `(H, B, 4)`, repeating a shared setting across the bands."""
    hyper = np.asarray(hyper, dtype=np.float64)
    if hyper.ndim == 2:
        hyper = np.repeat(hyper[:, None, :], n_bands, axis=1)
    if hyper.ndim != 3 or hyper.shape[1:] != (n_bands, 4):
        msg = f"hyper must have shape (H, 4) or (H, {n_bands}, 4), not {hyper.shape}"
        raise ValueError(msg)
    return hyper


@dataclass(frozen=True)
class _Constants:
    """The per-setting, per-band constants of the filter, each of shape `(1, H, B)`."""

    q_step: torch.Tensor
    phi: torch.Tensor
    stationary: torch.Tensor
    noise: torch.Tensor


@dataclass(frozen=True)
class _State:
    """The filter's state for every stream, setting, and band, each of shape `(S, H, B)`."""

    level: torch.Tensor
    anomaly: torch.Tensor
    p_ll: torch.Tensor
    p_lz: torch.Tensor
    p_zz: torch.Tensor


def _constants(*, params: torch.Tensor) -> _Constants:
    """Convert the unconstrained settings, shape `(H, B, 4)`, into the filter's constants."""
    phi = torch.sigmoid(params[..., 1])[None]
    return _Constants(
        q_step=(torch.exp(params[..., 0]) / HALF_HOURS_PER_DAY)[None],
        phi=phi,
        stationary=torch.exp(2.0 * params[..., 2])[None] / (1.0 - phi**2),
        noise=torch.exp(2.0 * params[..., 3])[None],
    )


def _initial_state(
    *, shape: tuple[int, int, int], constants: _Constants, device: torch.device
) -> _State:
    """Return the state before any reading: a vague level and a stationary transient."""
    zeros = torch.zeros(shape, dtype=torch.float64, device=device)
    return _State(
        level=zeros,
        anomaly=zeros.clone(),
        p_ll=torch.full_like(zeros, INITIAL_LEVEL_VARIANCE),
        p_lz=zeros.clone(),
        p_zz=constants.stationary.expand(shape).clone(),
    )


def _update_state(
    *,
    state: _State,
    reading: torch.Tensor,
    gap: torch.Tensor,
    here: torch.Tensor,
    constants: _Constants,
) -> tuple[_State, torch.Tensor]:
    """Move every filter over `gap` half-hours and fold in a reading where `here` is true.

    Where `here` is false the state is unchanged, which is how a missing reading is skipped.

    Args:
        state: The state after the previous reading.
        reading: The residual at this slot, shape `(S, 1, B)` (anything where `here` is false).
        gap: Half-hours since the previous reading, shape `(S, 1, B)`, int64.
        here: Whether the slot holds a reading, shape `(S, 1, B)`.
        constants: The filter's constants.

    Returns:
        The new state, and the reading's log likelihood where `here` is true, else 0.
    """
    decay = constants.phi ** gap.to(torch.float64)
    anomaly = state.anomaly * decay
    p_ll = state.p_ll + constants.q_step * gap
    p_lz = state.p_lz * decay
    p_zz = decay * decay * state.p_zz + constants.stationary * (1.0 - decay * decay)
    variance = p_ll + 2.0 * p_lz + p_zz + constants.noise
    innovation = reading - state.level - anomaly
    step_ll = -0.5 * (_LOG_TWO_PI + torch.log(variance) + innovation**2 / variance)
    gain_l = (p_ll + p_lz) / variance
    gain_z = (p_lz + p_zz) / variance
    updated = _State(
        level=state.level + gain_l * innovation,
        anomaly=anomaly + gain_z * innovation,
        p_ll=torch.clamp(p_ll - gain_l * gain_l * variance, min=VARIANCE_FLOOR),
        p_lz=p_lz - gain_l * gain_z * variance,
        p_zz=torch.clamp(p_zz - gain_z * gain_z * variance, min=VARIANCE_FLOOR),
    )
    kept = _State(
        level=torch.where(here, updated.level, state.level),
        anomaly=torch.where(here, updated.anomaly, state.anomaly),
        p_ll=torch.where(here, updated.p_ll, state.p_ll),
        p_lz=torch.where(here, updated.p_lz, state.p_lz),
        p_zz=torch.where(here, updated.p_zz, state.p_zz),
    )
    return kept, torch.where(here, step_ll, 0.0)


def run_kalman_grid(
    *,
    residuals: np.ndarray,
    band: np.ndarray,
    hyper: np.ndarray,
    n_bands: int,
    device: str | None = None,
    keep_states: bool = True,
) -> GridTrace:
    """Run the two-state filter for every stream and setting in one vectorised pass.

    Args:
        residuals: Shape `(S, T)`. Observed minus forecast on the grid, in points, `NaN` where
            there is no reading. A rolling-origin cut can be a stream with the later residuals set
            to `NaN`, though `correction_at_cutoffs` makes that unnecessary.
        band: Shape `(T,)`. The hour band of each grid row.
        hyper: The unconstrained hyperparameters (log `q`, logit `phi`, log `sigma_z`, log
            `sigma_eps`) of each setting. Shape `(H, 4)` shares a setting across the bands, and
            shape `(H, B, 4)` gives each band its own, which is how the decay can differ by hour
            band.
        n_bands: The number of hour bands.
        device: A torch device name, or `None` for CUDA when available, otherwise the CPU.
        keep_states: If false, `level` and `anomaly` are empty `(S, H, B, 0)` tensors and only the
            log likelihood is kept, which needs far less memory for a tuning grid.

    Returns:
        The trace, on the chosen device.
    """
    target = resolve_device(device=device)
    slots = band_slots(band=band, n_bands=n_bands)
    settings = _per_band(hyper=hyper, n_bands=n_bands)
    n_streams, n_settings = residuals.shape[0], settings.shape[0]
    width = slots.positions.shape[1]

    params = torch.as_tensor(settings, device=target)  # (H, B, 4)
    constants = _constants(params=params)
    positions = torch.as_tensor(slots.positions, device=target)  # (B, K)
    padded = torch.as_tensor(
        np.concatenate([residuals, np.full((n_streams, 1), np.nan)], axis=1), device=target
    )
    safe_index = positions.clamp(max=residuals.shape[1])
    values = padded[:, safe_index]  # (S, B, K); padded slots read the NaN column
    present = torch.isfinite(values)

    shape = (n_streams, n_settings, n_bands)
    state = _initial_state(shape=shape, constants=constants, device=target)
    log_likelihood = torch.zeros((n_streams, n_settings), dtype=torch.float64, device=target)
    previous = torch.full((n_streams, 1, n_bands), -1, dtype=torch.int64, device=target)

    states_shape = (n_streams, n_settings, n_bands, width if keep_states else 0)
    levels = torch.zeros(states_shape, dtype=torch.float64, device=target)
    anomalies = torch.zeros_like(levels)
    last_reading = torch.full((n_streams, n_bands, width), -1, dtype=torch.int64, device=target)

    for k in range(width):
        here = present[:, None, :, k]  # (S, 1, B)
        position = positions[None, None, :, k]  # (1, 1, B)
        state, step_ll = _update_state(
            state=state,
            reading=torch.where(here, values[:, None, :, k], 0.0),
            gap=torch.where(previous >= 0, position - previous, torch.zeros_like(previous)),
            here=here,
            constants=constants,
        )
        log_likelihood = log_likelihood + step_ll.sum(dim=2)
        # A reading's state is stored at the reading's own time, so the anomaly is not decayed to
        # the next slot here: `correction_at_cutoffs` decays it from `last_reading` to the target.
        previous = torch.where(here[:, :1], position.expand_as(previous), previous)
        last_reading[:, :, k] = previous[:, 0, :]
        if keep_states:
            levels[:, :, :, k] = state.level
            anomalies[:, :, :, k] = state.anomaly
    return GridTrace(
        slots=slots,
        level=levels,
        anomaly=anomalies,
        last_reading=last_reading,
        log_likelihood=log_likelihood,
        phi=constants.phi[0],
    )


def correction_at_cutoffs(
    *, trace: GridTrace, cutoff: np.ndarray, target: np.ndarray, band: np.ndarray
) -> torch.Tensor:
    """Look up the correction for a batch of forecast cases, for every stream and setting.

    Each case uses the state after its band's newest reading at or before `cutoff`, as
    `studies.dp_assimilation.correction_at` does, with the transient decayed to the target. A case
    with no such reading gets a correction of zero.

    Args:
        trace: The grid trace, which must have been run with `keep_states=True`.
        cutoff: Shape `(C,)`. Each case's newest usable grid position.
        target: Shape `(C,)`. Each case's grid position of the valid time.
        band: Shape `(C,)`. Each case's hour band (the target's).

    Returns:
        Shape `(S, H, C)`: the level plus the decayed transient, in points.
    """
    device = trace.level.device
    n_streams, n_settings = trace.level.shape[:2]
    out = torch.zeros((n_streams, n_settings, len(cutoff)), dtype=torch.float64, device=device)
    for band_index in range(trace.level.shape[2]):
        selected = np.flatnonzero(band == band_index)
        if not len(selected):
            continue
        slot_positions = trace.slots.positions[band_index, : trace.slots.n_rows[band_index]]
        found = np.searchsorted(slot_positions, cutoff[selected], side="right") - 1
        valid = found >= 0
        rows, slot = selected[valid], found[valid]
        if not len(rows):
            continue
        slot_tensor = torch.as_tensor(slot, device=device)
        last = trace.last_reading[:, band_index, :][:, slot_tensor]  # (S, C')
        known = last >= 0
        ahead = (torch.as_tensor(target[rows], device=device)[None] - last).clamp(min=0)
        decay = trace.phi[:, band_index][None, :, None] ** ahead[:, None, :].to(torch.float64)
        total = (
            trace.level[:, :, band_index, :][:, :, slot_tensor]
            + decay * trace.anomaly[:, :, band_index, :][:, :, slot_tensor]
        )
        out[:, :, torch.as_tensor(rows, device=device)] = torch.where(known[:, None, :], total, 0.0)
    return out
