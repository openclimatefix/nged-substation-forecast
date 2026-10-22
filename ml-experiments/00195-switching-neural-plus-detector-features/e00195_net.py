# ruff: noqa: E501, PLR0912, PLR0915
"""The window network, the window inputs, and the training-window sampler.

A window is `WINDOW` half-hours (7 days) of one Primary's residual and its up to four most-read
connected neighbours', each divided by its noise unit, with a present-or-not channel for each,
the sine and cosine of the local hour, and a bank-holiday flag. The network is a stack of dilated
one-dimensional convolutions with four outputs: a window-level onset alarm, a window-level end
alarm, and per-half-hour onset and end maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
import torch
from e00195_common import (
    ADJACENCY,
    D8_DROPS,
    D8_SERIES,
    D9_SERIES,
    GRID_LENGTH,
    SERIES_IDS,
    TUNING_MONTHS,
    WINDOW,
    explained_masks,
    holiday_flag,
    stamp_months,
)
from e00195_injector import InjectorFit
from e00195_worlds import Calendar, sample_placed
from torch import nn

N_CHANNELS: Final[int] = 13
REGION: Final[tuple[int, int]] = (48, 240)
"""An onset (or end) counts as inside a window when it falls in this span of the window's stamps."""
CLIP: Final[float] = 12.0
SERIES_INDEX: Final[dict[int, int]] = {s: i for i, s in enumerate(SERIES_IDS)}
EVEN_MONTHS: Final[frozenset[int]] = frozenset(set(range(1, 13)) - TUNING_MONTHS)
HARD_SHARES: Final[tuple[float, float, float]] = (0.2, 0.2, 0.2)
"""Shares of negative windows near a bank holiday, on D8's drops, and on D9."""


@dataclass
class Context:
    """The standardised panel and the lookups a window needs."""

    series: tuple[int, ...]
    z: np.ndarray
    """Residual in noise units, shape (series, grid), NaN where missing."""
    sigma: dict[int, float]
    neighbours: np.ndarray
    """Shape (series, 4): index into `series` of each neighbour, -1 where none."""
    hour_sin: np.ndarray
    hour_cos: np.ndarray
    holiday: np.ndarray
    calendar: Calendar
    explained: np.ndarray
    """Shape (series, grid), True within 24 hours of a logged closure or inside a labelled one."""
    even: np.ndarray
    """True on every stamp in an even-numbered month."""


def build_context(*, residual: dict[int, np.ndarray], sigma: dict[int, float]) -> Context:
    """Standardise the residuals and fix each series' neighbour order (most readings first)."""
    calendar = Calendar()
    z = np.full((len(SERIES_IDS), GRID_LENGTH), np.nan)
    for s, i in SERIES_INDEX.items():
        if np.isfinite(sigma.get(s, np.nan)):
            z[i] = residual[s] / sigma[s]
    available = np.isfinite(z).mean(axis=1)
    neighbours = np.full((len(SERIES_IDS), 4), -1, dtype=int)
    for s, i in SERIES_INDEX.items():
        ordered = sorted(ADJACENCY[s], key=lambda n: (-available[SERIES_INDEX[n]], n))
        for k, n in enumerate(ordered[:4]):
            neighbours[i, k] = SERIES_INDEX[n]
    masks = explained_masks()
    explained = np.stack([masks[s] for s in SERIES_IDS])
    months = stamp_months()
    angle = 2 * np.pi * calendar.hour / 24.0
    return Context(
        series=SERIES_IDS,
        z=z,
        sigma=sigma,
        neighbours=neighbours,
        hour_sin=np.sin(angle).astype(np.float32),
        hour_cos=np.cos(angle).astype(np.float32),
        holiday=holiday_flag(),
        calendar=calendar,
        explained=explained,
        even=np.isin(months, list(EVEN_MONTHS)),
    )


def window_inputs(
    *, ctx: Context, targets: np.ndarray, starts: np.ndarray, z: np.ndarray | None = None
) -> np.ndarray:
    """Assemble the (N, 13, WINDOW) input of windows starting at `starts` for series index `targets`.

    `z` replaces the context's residual (the injected world's), shape (series, grid).
    """
    z = ctx.z if z is None else z
    n = len(targets)
    offsets = starts[:, None] + np.arange(WINDOW)[None, :]
    out = np.zeros((n, N_CHANNELS, WINDOW), dtype=np.float32)
    chosen = [targets] + [ctx.neighbours[targets, k] for k in range(4)]
    for k, rows in enumerate(chosen):
        valid = rows >= 0
        values = np.zeros((n, WINDOW), dtype=np.float32)
        values[valid] = z[rows[valid][:, None], offsets[valid]]
        present = np.isfinite(values)
        out[:, k, :] = np.where(present, np.clip(np.nan_to_num(values), -CLIP, CLIP), 0.0)
        out[:, 5 + k, :] = present
    out[:, 10, :] = ctx.hour_sin[offsets]
    out[:, 11, :] = ctx.hour_cos[offsets]
    out[:, 12, :] = ctx.holiday[offsets]
    return out


class WindowNet(nn.Module):
    """Dilated convolutions over a window; outputs two alarm logits and two per-stamp maps."""

    def __init__(
        self, *, width: int = 32, dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    ) -> None:
        """Build the stem, the dilated blocks, and the two heads."""
        super().__init__()
        self.stem = nn.Conv1d(N_CHANNELS, width, kernel_size=5, padding=2)
        self.blocks = nn.ModuleList(
            nn.Conv1d(width, width, kernel_size=5, padding=2 * d, dilation=d) for d in dilations
        )
        self.norms = nn.ModuleList(nn.GroupNorm(4, width) for _ in dilations)
        self.maps = nn.Conv1d(width, 2, kernel_size=1)
        self.alarms = nn.Linear(2 * width, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return window alarm logits (N, 2: onset, end) and per-stamp map logits (N, 2, WINDOW)."""
        h = torch.nn.functional.gelu(self.stem(x))
        for conv, norm in zip(self.blocks, self.norms, strict=True):
            h = h + torch.nn.functional.gelu(norm(conv(h)))
        pooled = torch.cat([h.amax(dim=2), h.mean(dim=2)], dim=1)
        return self.alarms(pooled), self.maps(h)


def n_parameters(*, model: nn.Module) -> int:
    """Number of trainable weights."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class Batch:
    """Training windows: inputs, alarm labels, and the onset and end positions (-1 if absent)."""

    x: np.ndarray
    onset_alarm: np.ndarray
    end_alarm: np.ndarray
    onset_pos: np.ndarray
    end_pos: np.ndarray


def _region(*, position: int) -> bool:
    return REGION[0] <= position < REGION[1]


def _window_ok(*, ctx: Context, target: int, start: int, even_only: bool) -> bool:
    """A training window lies in even months and clear of logged and labelled events."""
    stop = start + WINDOW
    if start < 0 or stop > GRID_LENGTH:
        return False
    if even_only and not (ctx.even[start] and ctx.even[stop - 1]):
        return False
    rows = [target] + [r for r in ctx.neighbours[target] if r >= 0]
    return not ctx.explained[rows, start:stop].any()


def _holiday_starts(*, ctx: Context) -> np.ndarray:
    """Window starts whose region holds a bank holiday stamp."""
    return np.flatnonzero(ctx.holiday > 0)


def _d8_edges() -> np.ndarray:
    """Stamps of the first and last half-hour of each of D8's drops."""
    drops = pl.read_parquet(D8_DROPS)
    return np.concatenate([drops["first_stamp"].to_numpy(), drops["last_stamp"].to_numpy()])


def make_batch(
    *,
    n: int,
    fit: InjectorFit,
    residual: dict[int, np.ndarray],
    power: dict[int, np.ndarray],
    ctx: Context,
    rng: np.random.Generator,
    positive_share: float = 0.5,
    allowed_months: frozenset[int] = EVEN_MONTHS,
) -> Batch:
    """Draw `n` windows: injected-transfer windows, and real stretches with no logged event.

    Positive windows anchor the window on an injected event's onset or its end (half each), so the
    onset or the end falls in `REGION`. Negative windows are real: a fifth near a bank holiday, a
    fifth around an edge of one of D8's drops, a fifth on D9, the rest anywhere in `allowed_months`.
    """
    even_only = allowed_months == EVEN_MONTHS
    holiday_stamps = _holiday_starts(ctx=ctx)
    d8_edges = _d8_edges()
    explained = {s: ctx.explained[i] for s, i in SERIES_INDEX.items()}
    targets, starts, z_list = [], [], []
    labels = []
    sigma = ctx.sigma
    while len(targets) < n:
        if rng.random() < positive_share:
            placed = sample_placed(
                fit=fit,
                residual=residual,
                power=power,
                sigma=sigma,
                calendar=ctx.calendar,
                explained=explained,
                rng=rng,
                allow_months=allowed_months,
            )
            usable = [s for s in placed.series if np.isfinite(ctx.z[SERIES_INDEX[s]]).any()]
            if not usable:
                continue
            target = SERIES_INDEX[int(rng.choice(usable))]
            anchor = placed.onset if rng.random() < 0.5 else placed.end
            start = anchor - int(rng.integers(REGION[0], REGION[1]))
            if not _window_ok(ctx=ctx, target=target, start=start, even_only=even_only):
                continue
            o, e = placed.onset - start, placed.end - start
            labels.append((_region(position=o), _region(position=e), o, e))
            targets.append(target)
            starts.append(start)
            z_list.append(placed)
        else:
            u = rng.random()
            if u < HARD_SHARES[0]:
                target = int(rng.integers(len(SERIES_IDS)))
                start = int(rng.choice(holiday_stamps)) - int(rng.integers(REGION[0], REGION[1]))
            elif u < sum(HARD_SHARES[:2]):
                target = SERIES_INDEX[D8_SERIES]
                start = int(rng.choice(d8_edges)) - int(rng.integers(REGION[0], REGION[1]))
            elif u < sum(HARD_SHARES):
                target = SERIES_INDEX[D9_SERIES]
                start = int(rng.integers(0, GRID_LENGTH - WINDOW))
            else:
                target = int(rng.integers(len(SERIES_IDS)))
                start = int(rng.integers(0, GRID_LENGTH - WINDOW))
            if not np.isfinite(ctx.z[target, start : start + WINDOW]).mean() > 0.5:
                continue
            if not _window_ok(ctx=ctx, target=target, start=start, even_only=even_only):
                continue
            labels.append((False, False, -1, -1))
            targets.append(target)
            starts.append(start)
            z_list.append(None)
    t_arr, s_arr = np.array(targets), np.array(starts)
    x = window_inputs(ctx=ctx, targets=t_arr, starts=s_arr)
    for i, placed in enumerate(z_list):
        if placed is None:
            continue
        rows = [t_arr[i]] + [r for r in ctx.neighbours[t_arr[i]] if r >= 0]
        for k, row in enumerate(rows):
            series = SERIES_IDS[row]
            if series not in placed.delta:
                continue
            lo, hi = max(placed.onset, s_arr[i]), min(placed.end, s_arr[i] + WINDOW)
            if hi <= lo:
                continue
            piece = placed.delta[series][lo - placed.onset : hi - placed.onset] / sigma[series]
            channel = 0 if k == 0 else k
            present = x[i, 5 + channel, lo - s_arr[i] : hi - s_arr[i]] > 0
            x[i, channel, lo - s_arr[i] : hi - s_arr[i]] = np.where(
                present,
                np.clip(x[i, channel, lo - s_arr[i] : hi - s_arr[i]] + piece, -CLIP, CLIP),
                0.0,
            )
    arr = np.array([(a, b, c, d) for a, b, c, d in labels])
    return Batch(
        x=x,
        onset_alarm=arr[:, 0].astype(np.float32),
        end_alarm=arr[:, 1].astype(np.float32),
        onset_pos=np.where(arr[:, 0] > 0, arr[:, 2], -1).astype(np.int64),
        end_pos=np.where(arr[:, 1] > 0, arr[:, 3], -1).astype(np.int64),
    )
