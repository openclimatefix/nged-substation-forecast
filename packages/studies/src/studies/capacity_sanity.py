"""Four hard checks on a labeller's knots, for the study in issue #1147.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>. The
checks are a study prototype of a rule a production capacity estimator would need: an estimate that
is physically implausible, unsupported by the evidence, or worse than no estimate is not used.

**The policy differs for the two roles, and it is policy rather than a check.** A causal estimate
that fails any check is replaced by the static 99th-percentile power, and the failing checks are
recorded. A hindsight estimate loses only the failing knots, so that the hits of one plant are not
correlated by a whole-series fallback. Knot order is enforced by the answer schema, not here.

1. `bounds`: 0.2 to 1.1 times the static power. There is no clip at the fitted alternating-current
   capacity, because the static power exceeds that capacity at four plants.
2. `step_within_evidence`: a step is a change of at least `STEP_MINIMUM` between consecutive knots'
   levels. It passes if a changepoint candidate lies within `CANDIDATE_WINDOW_DAYS` of the knot and
   the step differs from the candidate's size by at most the larger of `STEP_TOLERANCE_POINTS` and
   `STEP_TOLERANCE_SHARE` of the candidate's size. A step with no candidate fails.
3. `reproduces_the_power`: the variance of the corrected clear-sky ratio after dividing by the
   estimate must not exceed its variance without the estimate, each series first divided by its own
   median, so a correct constant offset (an opening level of 0.9 times the static power) passes.
   An empty knot list passes.
4. `anm_consistency`: a knot whose cause is not `curtailment` must not sit inside a run of
   `CAP_RUN_DAYS` or more cap-binding days, and a `curtailment` knot needs a cap-binding day within
   `CANDIDATE_WINDOW_DAYS`.
"""

from dataclasses import dataclass, replace
from datetime import date
from itertools import pairwise
from typing import Final, Literal

import numpy as np

from studies.capacity_evidence import Evidence
from studies.capacity_labeller import Answer, Knot, capacity_series

CheckType = Literal["bounds", "step_within_evidence", "reproduces_the_power", "anm_consistency"]
"""The four checks."""

LOWER_BOUND_SHARE: Final[float] = 0.2
"""The least capacity a knot may state, as a share of the static power."""

UPPER_BOUND_SHARE: Final[float] = 1.1
"""The most capacity a knot may state, as a share of the static power."""

STEP_MINIMUM: Final[float] = 0.03
"""A change between consecutive knots smaller than this is not a step."""

CANDIDATE_WINDOW_DAYS: Final[int] = 14
"""How far a changepoint candidate or a binding cap day may lie from a knot."""

STEP_TOLERANCE_POINTS: Final[float] = 0.05
"""The least tolerance on a step's size against its candidate, as a fraction."""

STEP_TOLERANCE_SHARE: Final[float] = 0.4
"""The tolerance on a step's size as a share of the candidate's size."""

CAP_BINDING_DAY_SHARE: Final[float] = 0.5
"""A day is cap-binding when the cap binds on at least this share of its daylight hours."""

CAP_RUN_DAYS: Final[int] = 3
"""A run of this many cap-binding days in a row is a curtailment run."""

VARIANCE_TOLERANCE: Final[float] = 1e-9
"""A variance ratio this close to 1 is a tie, which passes, so rounding cannot reject a constant."""

CAUSAL_WINDOW_DAYS: Final[int] = 180
"""The causal role's variance check covers this many of the latest days."""


@dataclass(frozen=True)
class Failure:
    """One failed check.

    Attributes:
        check: Which check failed.
        day: The knot's date, or `None` for a check on the whole estimate.
    """

    check: CheckType
    day: date | None


@dataclass(frozen=True)
class SanityOutcome:
    """The result of applying the checks under one role's policy.

    Attributes:
        answer: The answer to use: the input with failing knots dropped (hindsight), or with no
            knots (causal, after any failure).
        failures: Every failed check, with the knot's date where it applies.
        fell_back: Whether the causal policy replaced the whole estimate by the static power.
    """

    answer: Answer
    failures: tuple[Failure, ...]
    fell_back: bool


def check_bounds(*, knots: tuple[Knot, ...], static_p99_mw: float) -> list[Failure]:
    """Return a failure for each knot outside 0.2 to 1.1 times the static power."""
    return [
        Failure(check="bounds", day=knot.day)
        for knot in knots
        if not LOWER_BOUND_SHARE * static_p99_mw
        <= knot.capacity_mw
        <= UPPER_BOUND_SHARE * static_p99_mw
    ]


def local_boundary_days(*, evidence: Evidence) -> list[date]:
    """Return the dates on which a local-scan episode starts or ends."""
    last = len(evidence.dates) - 1
    return [
        evidence.dates[index].astype(date)
        for episode in evidence.local_episodes
        for index in (episode.first_day, min(episode.last_day + 1, last))
    ]


def check_steps(
    *, knots: tuple[Knot, ...], evidence: Evidence, local_scan_candidates: bool = False
) -> list[Failure]:
    """Return a failure for each step that no candidate supports.

    **The amended check (round 3) also accepts a step when a local-scan episode starts or ends
    within `CANDIDATE_WINDOW_DAYS` of it, with no test of size.** It was written down before round 3
    ran, and a step the changepoint list supports still needs the size tolerance.

    Args:
        knots: The knots in date order.
        evidence: The evidence, whose candidates are the support.
        local_scan_candidates: Whether local-scan episode boundaries count as candidates.

    Returns:
        One failure per unsupported step, dated at the knot that ends the step.
    """
    failures = []
    boundaries = local_boundary_days(evidence=evidence) if local_scan_candidates else []
    for before, after in pairwise(knots):
        step = after.capacity_mw / before.capacity_mw - 1.0
        if abs(step) < STEP_MINIMUM:
            continue
        if any(abs((day - after.day).days) <= CANDIDATE_WINDOW_DAYS for day in boundaries):
            continue
        near = [
            change
            for change in evidence.changepoints
            if abs((change.day - after.day).days) <= CANDIDATE_WINDOW_DAYS
        ]
        if not near:
            failures.append(Failure(check="step_within_evidence", day=after.day))
            continue
        nearest = min(near, key=lambda change: abs((change.day - after.day).days))
        tolerance = max(STEP_TOLERANCE_POINTS, STEP_TOLERANCE_SHARE * abs(nearest.size))
        if abs(step - nearest.size) > tolerance:
            failures.append(Failure(check="step_within_evidence", day=after.day))
    return failures


def variance_ratio(
    *, knots: tuple[Knot, ...], evidence: Evidence, window_days: int | None
) -> float:
    """Return the variance of the corrected ratio with the estimate over the variance without.

    Args:
        knots: The knots.
        evidence: The evidence.
        window_days: How many of the latest days to use, or `None` for all.

    Returns:
        The ratio, or 1.0 for an empty knot list or fewer than 3 finite days. A series with no
        variance gives 1.0 if the estimate leaves it so and infinity if the estimate adds variance.
    """
    if not knots:
        return 1.0
    series = evidence.clear_corrected
    capacity = capacity_series(
        knots=knots, dates=evidence.dates, static_p99_mw=evidence.static_p99_mw
    )
    if window_days is not None:
        series, capacity = series[-window_days:], capacity[-window_days:]
    relative = capacity / float(np.median(capacity))
    usable = np.isfinite(series) & np.isfinite(relative)
    if usable.sum() < 3:
        return 1.0
    plain = series[usable] / float(np.median(series[usable]))
    divided = series[usable] / relative[usable]
    divided = divided / float(np.median(divided))
    with_estimate, without = float(np.var(divided)), float(np.var(plain))
    if without == 0.0:
        return 1.0 if with_estimate == 0.0 else float("inf")
    return with_estimate / without


def check_reproduces(
    *, knots: tuple[Knot, ...], evidence: Evidence, window_days: int | None
) -> list[Failure]:
    """Return a failure if dividing by the estimate raises the ratio's variance (ties pass)."""
    ratio = variance_ratio(knots=knots, evidence=evidence, window_days=window_days)
    if ratio > 1.0 + VARIANCE_TOLERANCE:
        return [Failure(check="reproduces_the_power", day=None)]
    return []


def binding_days(*, evidence: Evidence) -> np.ndarray:
    """Return which days are cap-binding, shape (days,); all False without an overlay."""
    share = evidence.cap_binding_share
    if share is None:
        return np.zeros(len(evidence.dates), dtype=bool)
    return np.nan_to_num(share, nan=0.0) >= CAP_BINDING_DAY_SHARE


def _inside_run(*, binding: np.ndarray, index: int) -> bool:
    """Say whether day `index` lies in a run of at least `CAP_RUN_DAYS` binding days."""
    if not binding[index]:
        return False
    start = index
    while start > 0 and binding[start - 1]:
        start -= 1
    stop = index
    while stop + 1 < len(binding) and binding[stop + 1]:
        stop += 1
    return stop - start + 1 >= CAP_RUN_DAYS


def check_anm(*, knots: tuple[Knot, ...], evidence: Evidence) -> list[Failure]:
    """Return a failure for each knot whose cause contradicts the export cap.

    Args:
        knots: The knots.
        evidence: The evidence, whose cap overlay says when the cap binds.

    Returns:
        A failure for a non-curtailment knot inside a binding run, and for a curtailment knot
        with no binding day within `CANDIDATE_WINDOW_DAYS`.
    """
    binding = binding_days(evidence=evidence)
    failures = []
    for knot in knots:
        index = int((np.datetime64(knot.day) - evidence.dates[0]).astype(int))
        index = min(max(index, 0), len(binding) - 1)
        if knot.cause == "curtailment":
            window = binding[
                max(index - CANDIDATE_WINDOW_DAYS, 0) : index + CANDIDATE_WINDOW_DAYS + 1
            ]
            if not window.any():
                failures.append(Failure(check="anm_consistency", day=knot.day))
        elif _inside_run(binding=binding, index=index):
            failures.append(Failure(check="anm_consistency", day=knot.day))
    return failures


def _drop(*, answer: Answer, days: set[date]) -> Answer:
    """Return the answer without the knots dated in `days`."""
    kept = tuple(knot for knot in answer.knots if knot.day not in days)
    return replace(answer, knots=kept, no_change=not kept)


def apply_hindsight(
    *, answer: Answer, evidence: Evidence, local_scan_candidates: bool = False
) -> SanityOutcome:
    """Drop the failing knots of a hindsight answer, and record each failure.

    Checks 1, 2, and 4 mark knots; every marked knot is dropped. Check 3 then drops, one at a time,
    the knot whose removal lowers the variance ratio most, until the ratio is at most 1.

    Args:
        answer: The labeller's answer.
        evidence: The evidence it was written from.
        local_scan_candidates: Whether the amended step check applies (see `check_steps`).

    Returns:
        The outcome, with `fell_back` False.
    """
    failures = [
        *check_bounds(knots=answer.knots, static_p99_mw=evidence.static_p99_mw),
        *check_steps(
            knots=answer.knots, evidence=evidence, local_scan_candidates=local_scan_candidates
        ),
        *check_anm(knots=answer.knots, evidence=evidence),
    ]
    current = _drop(answer=answer, days={f.day for f in failures if f.day is not None})
    while check_reproduces(knots=current.knots, evidence=evidence, window_days=None):
        trials = [
            (
                variance_ratio(
                    knots=_drop(answer=current, days={knot.day}).knots,
                    evidence=evidence,
                    window_days=None,
                ),
                knot,
            )
            for knot in current.knots
        ]
        _, worst = min(trials, key=lambda trial: trial[0])
        failures.append(Failure(check="reproduces_the_power", day=worst.day))
        current = _drop(answer=current, days={worst.day})
    return SanityOutcome(answer=current, failures=tuple(failures), fell_back=False)


def apply_causal(
    *, answer: Answer, evidence: Evidence, local_scan_candidates: bool = False
) -> SanityOutcome:
    """Replace a causal answer by the static power if any check fails.

    Args:
        answer: The labeller's answer.
        evidence: The evidence it was written from.
        local_scan_candidates: Whether the amended step check applies (see `check_steps`).

    Returns:
        The outcome. On any failure the answer has no knots and `fell_back` is True.
    """
    failures = [
        *check_bounds(knots=answer.knots, static_p99_mw=evidence.static_p99_mw),
        *check_steps(
            knots=answer.knots, evidence=evidence, local_scan_candidates=local_scan_candidates
        ),
        *check_reproduces(knots=answer.knots, evidence=evidence, window_days=CAUSAL_WINDOW_DAYS),
        *check_anm(knots=answer.knots, evidence=evidence),
    ]
    if not failures:
        return SanityOutcome(answer=answer, failures=(), fell_back=False)
    return SanityOutcome(
        answer=replace(answer, knots=(), no_change=True), failures=tuple(failures), fell_back=True
    )
