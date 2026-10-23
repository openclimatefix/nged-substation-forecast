# ruff: noqa: ANN202
"""Causal detectors on the daily mean residual: variants A to F of experiment 298.

Every array is on one day grid (`DAY0` plus the index). A detector run reads the days one by one
and never looks ahead, except that the start of a span is moved back, in hindsight, to the first
day of the candidate window that stood outside the band.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Final

import numpy as np
import polars as pl

DAY0: Final[np.datetime64] = np.datetime64("2019-12-01", "D")
N_DAYS: Final[int] = int((np.datetime64("2026-07-01", "D") - DAY0).astype(int))
NOISE_DAYS: Final[int] = 730
LEVEL_DAYS: Final[int] = 365
MIN_NOISE_DAYS: Final[int] = 180
BAND: Final[float] = 1.5
"""A daily mean inside this many noise units ends a span."""
DEPTH: Final[float] = 0.10
"""The least mean residual, as a share of the Primary's load, that counts as deep."""
MIN_LENGTH: Final[int] = 2
SMALL_SUM: Final[float] = 0.5
"""Variant D: the node plus its matched neighbours must sum to under this share of the node."""
MIN_MATCH: Final[float] = 0.30
"""Variant D: a matched neighbour moves at least this share of the node's size."""
F_SHARE: Final[float] = 0.70
F_MIN_PARTNER: Final[float] = 0.10
PAIR_SUM_BAND: Final[float] = 1.5
MAX_COVER_PARTNERS: Final[int] = 3
CUSUM_ALLOWANCE: Final[float] = 0.5


def day_index(*, days: np.ndarray) -> np.ndarray:
    """Return the integer grid position of each `datetime64[D]` day."""
    return (days.astype("datetime64[D]") - DAY0).astype(int)


def to_grid(*, days: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Place `values` on the day grid, `NaN` where there is no day."""
    out = np.full(N_DAYS, np.nan)
    out[day_index(days=days)] = values
    return out


def panel_arrays(*, panel: pl.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Return {site: {residual, power}} on the day grid from the clean panel."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for (site,), group in panel.sort("day").group_by("site"):
        days = group["day"].to_numpy()
        out[str(site)] = {
            "residual": to_grid(days=days, values=group["residual_mw"].to_numpy()),
            "power": to_grid(days=days, values=group["power_mw"].to_numpy()),
        }
    return out


def rolling_mad_noise(*, series: np.ndarray, first: int = 0) -> np.ndarray:
    """Return the robust sd (1.4826 MAD) of `series` over the previous `NOISE_DAYS` days.

    Entry d uses days d - NOISE_DAYS to d - 1 only, so the noise is causal. `NaN` before
    `MIN_NOISE_DAYS` valid days exist.
    """
    out = np.full(len(series), np.nan)
    for d in range(max(first, 1), len(series)):
        window = series[max(0, d - NOISE_DAYS) : d]
        window = window[np.isfinite(window)]
        if len(window) >= MIN_NOISE_DAYS:
            out[d] = 1.4826 * np.median(np.abs(window - np.median(window)))
    return out


def rolling_level(*, power: np.ndarray) -> np.ndarray:
    """Return the median power over the previous `LEVEL_DAYS` days (causal)."""
    out = np.full(len(power), np.nan)
    for d in range(30, len(power)):
        window = power[max(0, d - LEVEL_DAYS) : d]
        window = window[np.isfinite(window)]
        if len(window) >= 30:
            out[d] = float(np.median(window))
    return out


@dataclass
class Stream:
    """One site's daily residual, its noise unit, and its load level."""

    residual: np.ndarray
    noise: np.ndarray
    level: np.ndarray
    z: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Derive the z-score of the daily mean in noise units."""
        self.z = self.residual / self.noise


def make_stream(*, residual: np.ndarray, level: np.ndarray, first: int = 0) -> Stream:
    """Build a stream from a residual and a load level on the grid."""
    return Stream(
        residual=residual, noise=rolling_mad_noise(series=residual, first=first), level=level
    )


@dataclass(frozen=True)
class Span:
    """One alarm span, in grid days. `end` is inclusive; `fire` is the day the alarm is known."""

    start: int
    end: int
    fire: int
    sign: int
    size_mw: float
    explained: float = float("nan")
    partners: tuple[str, ...] = ()


@dataclass(frozen=True)
class Variant:
    """A detector variant. `kind` is A, B, C, D, E or F; D and F wrap `base` (A or C)."""

    kind: str
    k: float
    w: int = 3
    base: str = "A"

    @property
    def name(self) -> str:
        """A short label, such as `A3` or `D-A3`."""
        core = f"A{self.w}" if self.base == "A" else "C"
        if self.kind in "AC":
            return f"A{self.w}" if self.kind == "A" else "C"
        if self.kind == "B":
            return "B"
        if self.kind == "E":
            return f"E{self.w}"
        return f"{self.kind}-{core}"


def _window_stat(*, stream: Stream, d: int, variant: Variant) -> tuple[float, int]:
    """Return (signed z-statistic, window length) of the A, B or C candidate window at day d."""
    z = stream.z
    if variant.kind == "C" or (variant.kind in "DF" and variant.base == "C"):
        recent = z[d - 2 : d + 1]
        base = z[d - 30 : d - 2]
        if np.isfinite(recent).sum() < 2 or np.isfinite(base).sum() < 14:
            return np.nan, 3
        return float(np.nanmean(recent) - np.nanmean(base)), 3
    w = variant.w
    window = z[d - w + 1 : d + 1]
    if np.isfinite(window).sum() < w:
        return np.nan, w
    return float(np.mean(window)), w


def _depth_ok(*, stream: Stream, d: int, w: int, sign: int) -> bool:
    """True when the mean residual over the last `w` days is at least `DEPTH` of the load."""
    window = stream.residual[d - w + 1 : d + 1]
    level = stream.level[d]
    if not np.isfinite(level) or level <= 0 or np.isfinite(window).sum() < max(1, w - 1):
        return False
    return sign * float(np.nanmean(window)) >= DEPTH * level


def _hindsight_start(*, stream: Stream, d: int, w: int, sign: int) -> int:
    """Return the first day of the last `w` days whose z stood outside the band, else d - w + 1."""
    for day in range(d - w + 1, d + 1):
        if np.isfinite(stream.z[day]) and sign * stream.z[day] >= BAND:
            return day
    return d - w + 1


def _confirm_d(
    *, node: Stream, others: dict[str, Stream], d: int, w: int, sign: int
) -> tuple[bool, tuple[str, ...], float]:
    """Variant D: a metered neighbour with the opposite sign, and a small summed residual."""
    mean_node = float(np.nanmean(node.residual[d - w + 1 : d + 1]))
    matched, total = [], mean_node
    for name, other in others.items():
        window = other.residual[d - w + 1 : d + 1]
        if np.isfinite(window).sum() < max(1, w - 1):
            continue
        mean_other = float(np.nanmean(window))
        if sign * mean_other < 0 and abs(mean_other) >= MIN_MATCH * abs(mean_node):
            matched.append(name)
            total += mean_other
    ok = bool(matched) and abs(total) <= SMALL_SUM * abs(mean_node)
    return ok, tuple(matched), 1.0 - abs(total) / abs(mean_node)


def _confirm_f(
    *, node: Stream, others: dict[str, Stream], d: int, w: int, sign: int
) -> tuple[bool, tuple[str, ...], float]:
    """Variant F: up to 3 neighbours whose summed mean cancels the node's mean."""
    mean_node = float(np.nanmean(node.residual[d - w + 1 : d + 1]))
    means = {}
    for name, other in others.items():
        window = other.residual[d - w + 1 : d + 1]
        if np.isfinite(window).sum() >= max(1, w - 1):
            mean_other = float(np.nanmean(window))
            if sign * mean_other < 0 and abs(mean_other) >= F_MIN_PARTNER * abs(mean_node):
                means[name] = mean_other
    best: tuple[float, tuple[str, ...]] = (abs(mean_node), ())
    for r in range(1, MAX_COVER_PARTNERS + 1):
        for subset in combinations(means, r):
            remaining = abs(mean_node + sum(means[n] for n in subset))
            if remaining < best[0]:
                best = (remaining, subset)
    share = 1.0 - best[0] / abs(mean_node)
    return share >= F_SHARE, best[1], share


def _run_machine(
    *,
    stream: Stream,
    trigger,  # noqa: ANN001  (day -> (sign, w, start, extra) | None)
    first: int,
    last: int,
) -> list[Span]:
    """Run the span state machine over days `first`..`last` with a per-day trigger function."""
    spans: list[Span] = []
    active: dict | None = None
    last_end = -1
    for d in range(first, last + 1):
        z = stream.z[d]
        if active is not None:
            if np.isfinite(z) and active["sign"] * z < BAND:
                end = d - 1
                if end - active["start"] + 1 >= MIN_LENGTH:
                    spans.append(Span(end=end, **{k: v for k, v in active.items() if k != "peak"}))
                last_end = end
                active = None
            else:
                if np.isfinite(z):
                    active["peak"] = max(active["peak"], active["sign"] * stream.residual[d])
                continue
        if active is None:
            hit = trigger(d, last_end)
            if hit is not None:
                sign, start, extra = hit
                active = {
                    "start": start,
                    "fire": d,
                    "sign": sign,
                    "size_mw": sign * float(np.nanmean(stream.residual[start : d + 1])),
                    "peak": 0.0,
                    **extra,
                }
    if active is not None:
        spans.append(Span(end=last, **{k: v for k, v in active.items() if k != "peak"}))
    return spans


def detect(
    *,
    variant: Variant,
    stream: Stream,
    others: dict[str, Stream],
    first: int,
    last: int,
    pair_noise: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> list[Span]:
    """Return the alarm spans of `variant` on `stream` for days `first`..`last` (inclusive).

    Args:
        variant: The detector variant.
        stream: The node's stream.
        others: The metered neighbours' streams, keyed by label (used by D, E and F).
        first: The first day to read; at least 31 days after the grid start.
        last: The last day to read.
        pair_noise: Optional precomputed (difference, sum) noise units per neighbour for variant E,
            to avoid recomputing them from the stream.

    Returns:
        The alarm spans, in order of start day.
    """
    pair_stats = (
        _pair_stats(stream=stream, others=others, pair_noise=pair_noise)
        if variant.kind == "E"
        else {}
    )

    def trigger(d: int, last_end: int):
        if variant.kind == "B":
            return _cusum_trigger(stream=stream, d=d, last_end=last_end, variant=variant)
        if variant.kind == "E":
            return _pair_trigger(
                stream=stream, pair_stats=pair_stats, d=d, last_end=last_end, variant=variant
            )
        stat, w = _window_stat(stream=stream, d=d, variant=variant)
        if not np.isfinite(stat) or abs(stat) < variant.k:
            return None
        sign = 1 if stat > 0 else -1
        if d - w + 1 <= last_end or not _depth_ok(stream=stream, d=d, w=w, sign=sign):
            return None
        extra: dict = {}
        if variant.kind in "DF":
            confirm = _confirm_d if variant.kind == "D" else _confirm_f
            ok, partners, share = confirm(node=stream, others=others, d=d, w=w, sign=sign)
            if not ok:
                return None
            extra = {"partners": partners, "explained": share}
        return sign, _hindsight_start(stream=stream, d=d, w=w, sign=sign), extra

    return _run_machine(stream=stream, trigger=trigger, first=first, last=last)


_CUSUM_STATE: dict[int, dict] = {}


def _cusum_trigger(*, stream: Stream, d: int, last_end: int, variant: Variant):
    """Variant B: a two-sided CUSUM on the daily z, restarted after each span."""
    state = _CUSUM_STATE.setdefault(
        id(stream), {"last": -1, "pos": 0.0, "neg": 0.0, "p0": 0, "n0": 0}
    )
    if state["last"] >= d:  # a new run over the same stream
        state.update(last=-1, pos=0.0, neg=0.0, p0=0, n0=0)
    z = stream.z[d]
    if np.isfinite(z):
        state["pos"] = max(0.0, state["pos"] + z - CUSUM_ALLOWANCE)
        state["neg"] = max(0.0, state["neg"] - z - CUSUM_ALLOWANCE)
        if state["pos"] == 0.0:
            state["p0"] = d
        if state["neg"] == 0.0:
            state["n0"] = d
    state["last"] = d
    for sign, key, zero in ((1, "pos", "p0"), (-1, "neg", "n0")):
        if state[key] >= variant.k:
            start = max(state[zero] + 1, last_end + 1)
            span = max(1, d - start + 1)
            if start > d or not _depth_ok(stream=stream, d=d, w=span, sign=sign):
                continue
            state.update(pos=0.0, neg=0.0)
            return sign, start, {}
    return None


def pair_noise_of(*, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the causal noise units of the difference and of the sum of two residuals."""
    return rolling_mad_noise(series=a - b), rolling_mad_noise(series=a + b)


def _pair_stats(
    *,
    stream: Stream,
    others: dict[str, Stream],
    pair_noise: dict[str, tuple[np.ndarray, np.ndarray]] | None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {neighbour: (z of the difference, z of the sum)} with their own noise units."""
    out = {}
    for name, other in others.items():
        diff = stream.residual - other.residual
        total = stream.residual + other.residual
        noise_diff, noise_sum = (
            pair_noise[name]
            if pair_noise and name in pair_noise
            else pair_noise_of(a=stream.residual, b=other.residual)
        )
        out[name] = (diff / noise_diff, total / noise_sum)
    return out


def _pair_trigger(*, stream: Stream, pair_stats: dict, d: int, last_end: int, variant: Variant):
    """Variant E: the difference of a connected pair shifts while its sum stays flat."""
    w = variant.w
    if d - w + 1 <= last_end:
        return None
    best = None
    for name, (zdiff, zsum) in pair_stats.items():
        a, b = zdiff[d - w + 1 : d + 1], zsum[d - w + 1 : d + 1]
        if np.isfinite(a).sum() < w or np.isfinite(b).sum() < w:
            continue
        stat, flat = float(np.mean(a)), abs(float(np.mean(b)))
        if abs(stat) >= variant.k and flat <= PAIR_SUM_BAND:
            sign = 1 if stat > 0 else -1
            if _depth_ok(stream=stream, d=d, w=w, sign=sign) and (
                best is None or abs(stat) > best[0]
            ):
                best = (abs(stat), sign, name)
    if best is None:
        return None
    _, sign, name = best
    return sign, _hindsight_start(stream=stream, d=d, w=w, sign=sign), {"partners": (name,)}


def flagged_days(
    *, spans: list[Span], cut_day: int, pre: int = 2, post: int = 1, cap: int = 60
) -> np.ndarray:
    """Return the grid days to exclude from the refit at `cut_day` (known before the cut).

    A span counts when its alarm fired before `cut_day`. It covers `pre` days before its start and
    `post` days after its end, never beyond `cut_day - 1`. A run of flagged days is capped at
    `cap` days (the first `cap` are kept), so that a permanent change is eventually learned.
    """
    mark = np.zeros(N_DAYS, dtype=bool)
    for s in spans:
        if s.fire < cut_day:
            mark[max(0, s.start - pre) : min(s.end + post, cut_day - 1) + 1] = True
    run = 0
    for d in range(N_DAYS):
        run = run + 1 if mark[d] else 0
        if run > cap:
            mark[d] = False
    return mark
