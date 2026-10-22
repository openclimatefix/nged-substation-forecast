"""Known-answer injections: G2's recorded output subtracted from D8's reading through the meter."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
from e00199_common import G2_FIRST, G2_LAST, N_UNITS, Q_FLOOR, UNIT_MW, stamp

K2_FIRST_DESTINATION = datetime(2022, 1, 3, tzinfo=UTC)
K2_PITCH_DAYS = 49
N_K2_WINDOWS = 9


@dataclass
class Injection:
    """An injected reading and what was injected, on the full grid."""

    y: np.ndarray
    output: np.ndarray
    """Injected output in MW; NaN where the record has no reading (not scored)."""
    window_id: np.ndarray
    """-1 outside the injected windows; 0 for K1; 1 to 9 for the K2 windows; 10 to 19 synthetic."""

    @property
    def truth_units(self) -> np.ndarray:
        """Whole units running, from the injected output."""
        return np.clip(np.rint(np.nan_to_num(self.output) / UNIT_MW), 0, N_UNITS).astype(int)

    @property
    def scored(self) -> np.ndarray:
        """True where a known answer exists."""
        return (self.window_id >= 0) & np.isfinite(self.output)


def through_meter(*, y: np.ndarray, output: np.ndarray, q: float = Q_FLOOR) -> np.ndarray:
    """Subtract `output` (MW) from the real power behind the reading and read it back as |S|."""
    real = np.sqrt(np.maximum(y**2 - q**2, 0.0))
    return np.sqrt((real - output) ** 2 + q**2)


def _bst_shift(*, when: datetime) -> int:
    """Half-hours to add so a summer (UTC+1) clock pattern keeps its local time at `when`."""
    summer = (
        datetime(when.year, 3, 25, tzinfo=UTC) <= when < datetime(when.year, 10, 25, tzinfo=UTC)
    )
    return 0 if summer else 2


def build(
    *, y: np.ndarray, g2: np.ndarray, with_k1: bool = True, with_k2: bool = True, q: float = Q_FLOOR
) -> Injection:
    """K1 (exact replay) and K2 (nine whole-week shifts into 2022-01 to 2023-05) in one copy."""
    n = len(y)
    out = np.full(n, np.nan)
    window = np.full(n, -1)
    lo, hi = stamp(G2_FIRST), stamp(G2_LAST)
    record = np.clip(g2[lo:hi], 0.0, None)
    y_new = y.copy()
    if with_k1:
        out[lo:hi] = record
        window[lo:hi] = 0
    if with_k2:
        for i in range(N_K2_WINDOWS):
            start = K2_FIRST_DESTINATION + timedelta(days=K2_PITCH_DAYS * i)
            mid = start + timedelta(days=22)
            a = stamp(start) + _bst_shift(when=mid)
            out[a : a + len(record)] = record
            window[a : a + len(record)] = i + 1
    inject = np.isfinite(out)
    # unscored half-hours inside a window (no G2 reading) keep the original reading
    y_new[inject] = through_meter(y=y[inject], output=out[inject], q=q)
    in_window = window >= 0
    return Injection(y=y_new, output=np.where(in_window, out, np.nan), window_id=window)


def synthetic(
    *, y: np.ndarray, g2: np.ndarray, rate: float, seed: int, q: float = Q_FLOOR
) -> Injection:
    """Spells resampled from G2's record at `rate` times its running rate, in the K2 windows."""
    rng = np.random.default_rng(seed)
    lo, hi = stamp(G2_FIRST), stamp(G2_LAST)
    record = np.clip(np.nan_to_num(g2[lo:hi]), 0.0, None)
    running = record > 0.5 * UNIT_MW
    edges = np.flatnonzero(np.diff(np.concatenate([[0], running.astype(int), [0]])))
    spells = [(a, b) for a, b in zip(edges[::2], edges[1::2], strict=True)]
    n = len(y)
    out = np.full(n, np.nan)
    window = np.full(n, -1)
    n_spells = round(len(spells) * rate)
    for i in range(N_K2_WINDOWS):
        a0 = stamp(K2_FIRST_DESTINATION + timedelta(days=K2_PITCH_DAYS * i))
        length = hi - lo
        sig = np.zeros(length)
        for _ in range(n_spells):
            a, b = spells[rng.integers(len(spells))]
            # hour-of-day of the start is kept; the day is uniform
            day = rng.integers(length // 48)
            start = day * 48 + (a % 48)
            piece = record[a:b]
            end = min(start + len(piece), length)
            sig[start:end] = np.maximum(sig[start:end], piece[: end - start])
        out[a0 : a0 + length] = sig
        window[a0 : a0 + length] = 10 + i
    inject = np.isfinite(out)
    y_new = y.copy()
    y_new[inject] = through_meter(y=y[inject], output=out[inject], q=q)
    return Injection(y=y_new, output=out, window_id=window)
