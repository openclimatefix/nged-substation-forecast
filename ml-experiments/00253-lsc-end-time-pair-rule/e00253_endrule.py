"""End rules for an alarm interval: the lookahead rule of experiment 148 and the pair rules.

The alarms stay as they are. Nothing here reads a switching log. R0 is experiment 148's lookahead
rule at 24 hours, applied to each side alone. P1, P2, and P3 add the pair: the neighbouring metered
substations that stepped the opposite way at the onset (the partners) must show an equal and
opposite step for a candidate end to count.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from typing import Any, Final

import numpy as np
from e00253_common import ADJACENCY, GRID_LENGTH
from e00253_detectors import REFIT_PERIOD_WINDOW, DetectorInputs, refit_stamps
from e00253_intervals import BLOCK, Interval

RETURN_Z_MIN: Final[float] = 1.5
"""A return step must reach this many noise units in the opposite direction to count."""
MIN_DURATION: Final[int] = BLOCK
"""Shortest duration searched, in half-hours (24 hours)."""
FIRST_SHARE: Final[float] = 0.5
"""A return step must reach this share of the onset step's size to count."""
LOOKAHEAD_HOURS: Final[int] = 24
"""Hours of data after the candidate end that may fix the end."""
BALANCE_TOLERANCE: Final[float] = 0.5
"""A candidate is balanced when the pair's summed step is under this share of the side's step."""
PARTNER_ONSET_STAMPS: Final[int] = 4
"""A partner's onset step may lie this many half-hours (2 hours) either side of the alarm."""
RULES: Final[tuple[str, ...]] = ("OFF", "R0", "P1", "P2", "P3", "P1G", "P2G")
"""OFF is experiment 130's offline rule (a reference), R0 is the 24-hour lookahead rule on each
side alone, and P1, P2, and P3 are the pair rules. P1G and P2G are post hoc: P1 and P2 over the
group of every neighbour that stepped at the onset, of either sign, so that a transfer with two
metered receivers balances."""
FEATURE_NAMES: Final[tuple[str, ...]] = (
    "own_return",
    "partner_min_return",
    "partner_mean_return",
    "imbalance",
    "days_since_onset",
    "own_return_z",
    "own_peak_share",
)
"""The columns of P3's feature matrix."""
SCORER: dict[str, Any] = {}
"""P3's fitted scorer, set once per process by `set_scorer`."""


def set_scorer(*, scorer: Any) -> None:
    """Install the fitted scorer that rule P3 reads."""
    SCORER["model"] = scorer


def blocked_by_refit(*, inputs: DetectorInputs) -> np.ndarray:
    """Stamps where a return step is ignored: near a refit, unless `inputs` never crosses one."""
    if not inputs.suppress_refits:
        return np.zeros(GRID_LENGTH, dtype=bool)
    return near_refit()


@cache
def near_refit() -> np.ndarray:
    """True within 24 hours of a monthly refit, where the residual model's jump can mimic a step."""
    mask = np.zeros(GRID_LENGTH, dtype=bool)
    for stamp in refit_stamps():
        mask[max(stamp - REFIT_PERIOD_WINDOW + 1, 0) : stamp + REFIT_PERIOD_WINDOW] = True
    return mask


def find_partners(
    *, inputs: DetectorInputs, series: int, onset: int, group: bool = False
) -> dict[int, float]:
    """The neighbouring metered series that stepped the opposite way at the onset, and their steps.

    A neighbour is a partner when its largest opposite-signed step within
    `PARTNER_ONSET_STAMPS` of `onset` reaches `RETURN_Z_MIN` noise units. The value is that step
    in MW, with its sign. With `group`, a neighbour that stepped the same way also counts, so the
    group is every neighbour with a step of at least `RETURN_Z_MIN` noise units at the onset.
    Only data from the alarm itself is read.
    """
    sign = np.sign(inputs.shift[series][onset])
    partners: dict[int, float] = {}
    for neighbour in ADJACENCY.get(series, ()):
        if neighbour not in inputs.shift:
            continue
        segment = inputs.shift[neighbour][
            max(onset - PARTNER_ONSET_STAMPS, 0) : onset + PARTNER_ONSET_STAMPS + 1
        ]
        if not np.isfinite(segment).any():
            continue
        if group:
            step = float(segment[np.nanargmax(np.abs(segment))])
            big = abs(step) / inputs.scale[neighbour] >= RETURN_Z_MIN
        else:
            step = float(segment[np.nanargmax(-sign * segment)])
            big = -sign * step / inputs.scale[neighbour] >= RETURN_Z_MIN
        if big:
            partners[neighbour] = step
    return partners


def lookahead_end(*, usable: np.ndarray, first: int, base: int, window: int) -> tuple[int, int]:
    """The end fixed with a lookahead of `LOOKAHEAD_HOURS`, and the stamp at which it is called.

    From the first qualifying stamp, the candidate end is the stamp of the largest score so far
    (`usable`, with -inf where a stamp does not qualify). The candidate is fixed once none of the
    next `LOOKAHEAD_HOURS` hours of stamps has a larger score. The step at the last of those
    stamps needs `window` more stamps, so the call is made `lookahead + window` stamps after the
    candidate. When the search window ends first, the candidate is fixed at the end of it.
    """
    look = LOOKAHEAD_HOURS * 2
    candidate, peak, k = first, usable[first], first
    while k - candidate < look and k < usable.size - 1:
        k += 1
        if usable[k] > peak:
            candidate, peak = k, usable[k]
    return base + candidate, base + k + window


class Candidates:
    """The return steps of one interval's side and its partners over the search window."""

    def __init__(
        self, *, inputs: DetectorInputs, iv: Interval, cap_days: int, group: bool = False
    ) -> None:
        """Read the steps of the side and its partners from 24 hours after the onset."""
        self.sign = float(np.sign(inputs.shift[iv.series][iv.onset]))
        self.onset_step = abs(float(inputs.shift[iv.series][iv.onset]))
        self.base = iv.onset + MIN_DURATION
        stop = min(iv.onset + cap_days * BLOCK + 1, GRID_LENGTH)
        own = inputs.shift[iv.series][self.base : stop]
        self.own_step = own
        self.ret = -self.sign * own
        self.z_ret = self.ret / inputs.scale[iv.series]
        self.blocked = blocked_by_refit(inputs=inputs)[self.base : stop]
        found = find_partners(inputs=inputs, series=iv.series, onset=iv.onset, group=group)
        self.partner_onset = {p: abs(step) for p, step in found.items()}
        self.partner_ret = {
            p: -np.sign(step) * inputs.shift[p][self.base : stop] for p, step in found.items()
        }
        partner_sum = sum(inputs.shift[p][self.base : stop] for p in self.partner_onset)
        self.pair_sum = own + (partner_sum if self.partner_onset else 0.0)
        onset_z = abs(inputs.shift[iv.series][iv.onset] / inputs.scale[iv.series])
        self.need_z = max(RETURN_Z_MIN, FIRST_SHARE * onset_z)

    @property
    def paired(self) -> bool:
        """True when at least one partner stepped at the onset."""
        return bool(self.partner_onset)

    def qualifies(self) -> np.ndarray:
        """Stamps whose own opposite step is big enough, outside a refit."""
        return np.isfinite(self.z_ret) & (self.z_ret >= self.need_z) & ~self.blocked

    def balanced(self) -> np.ndarray:
        """Stamps where the pair's summed step is under `BALANCE_TOLERANCE` of the side's step."""
        with np.errstate(invalid="ignore"):
            return np.isfinite(self.pair_sum) & (
                np.abs(self.pair_sum) <= BALANCE_TOLERANCE * np.abs(self.own_step)
            )

    def partner_returns(self) -> np.ndarray:
        """Each partner's opposite step as a share of its onset step; rows are partners."""
        rows = [self.partner_ret[p] / max(self.partner_onset[p], 1e-9) for p in self.partner_onset]
        return np.vstack(rows) if rows else np.empty((0, self.ret.size))

    def joint_score(self) -> np.ndarray:
        """The smallest return share over the side and its partners at every stamp."""
        own = self.ret / max(self.onset_step, 1e-9)
        shares = np.vstack([own, self.partner_returns()])
        with np.errstate(invalid="ignore"):
            return np.where(np.isfinite(shares).all(axis=0), shares.min(axis=0), -np.inf)

    def features(self) -> np.ndarray:
        """P3's feature matrix, one row per stamp of the window (columns in `FEATURE_NAMES`)."""
        n = self.ret.size
        partners = self.partner_returns()
        own = self.ret / max(self.onset_step, 1e-9)
        with np.errstate(invalid="ignore", divide="ignore"):
            running_peak = np.maximum.accumulate(np.nan_to_num(self.ret, nan=0.0))
            peak_share = np.where(
                running_peak > 0, np.nan_to_num(self.ret, nan=0.0) / running_peak, 0.0
            )
            imbalance = np.abs(self.pair_sum) / np.maximum(np.abs(self.own_step), 1e-9)
        return np.column_stack(
            [
                own,
                partners.min(axis=0) if partners.size else np.zeros(n),
                partners.mean(axis=0) if partners.size else np.zeros(n),
                np.minimum(imbalance, 5.0),
                np.arange(n) / BLOCK,
                self.z_ret,
                peak_share,
            ]
        )


def live_end_intervals(
    *, intervals: list[Interval], inputs: DetectorInputs, rule: str, cap_days: int
) -> list[Interval]:
    """End each interval by one of the rules, with `known_from` the stamp at which it is called.

    The step statistic at stamp t averages the `inputs.window` stamps from t, so it can be computed
    at t + window. The first qualifying stamp is the first whose opposite step is at least
    `FIRST_SHARE` of the onset step (and `RETURN_Z_MIN` noise units). Then:

    - OFF (offline): the end is the largest opposite step within one window after the first
      qualifying stamp, which is known at that stamp plus two windows.
    - R0: the end is the largest opposite step so far, fixed by a lookahead of 24 hours.
    - P1: R0 over the balanced stamps only.
    - P2: R0 on the joint score of the side and its partners, over the balanced stamps.
    - P3: R0 on the scorer's probability over the balanced stamps.

    A side with no partner, or with no balanced qualifying stamp, keeps R0's end. For the second
    case the end is called only when the search window has ended, because a balanced stamp could
    still come. When no stamp
    qualifies, the interval keeps its tracked end and the stamp it was known from.
    """
    out: list[Interval] = []
    window = inputs.window
    for iv in intervals:
        cand = Candidates(inputs=inputs, iv=iv, cap_days=cap_days, group=rule.endswith("G"))
        qualifies = cand.qualifies()
        hits = np.flatnonzero(qualifies)
        if hits.size == 0:
            out.append(iv)
            continue
        plain = np.where(qualifies, cand.ret, -np.inf)
        if rule == "OFF":
            first = int(hits[0])
            best = first + int(np.argmax(plain[first : first + window + 1]))
            end, known = cand.base + best, cand.base + first + 2 * window
        else:
            usable = plain
            fallback = False
            kind = rule.removesuffix("G")
            if kind != "R0" and cand.paired:
                keep = qualifies & cand.balanced()
                score = {
                    "P1": cand.ret,
                    "P2": cand.joint_score(),
                    "P3": _scorer_probability(features=cand.features()),
                }[kind]
                if kind == "P2":
                    keep &= score >= FIRST_SHARE
                if keep.any():
                    usable = np.where(keep, score, -np.inf)
                else:
                    fallback = True
            first = int(np.flatnonzero(np.isfinite(usable))[0])
            end, known = lookahead_end(usable=usable, first=first, base=cand.base, window=window)
            if fallback:
                # A live service cannot know that no balanced stamp will come until the search
                # window has ended.
                known = max(known, iv.onset + cap_days * BLOCK + window)
        out.append(replace(iv, end=int(end), known_from=int(known)))
    return out


def _scorer_probability(*, features: np.ndarray) -> np.ndarray:
    """P3's probability that each stamp is the true end; 0 where a feature is missing."""
    finite = np.isfinite(features).all(axis=1)
    out = np.zeros(features.shape[0])
    if finite.any():
        out[finite] = SCORER["model"].predict_proba(features[finite])[:, 1]
    return out
