"""Q2: does the interval builder recover injected transfers, their ends, and their splits?

600 synthetic transfers go into quiet periods: the moved load is a fraction `phi` of the donor's own
power (so it follows the donor's daily shape), split across 1 to 3 receivers with fractions that
vary through the day in half the events. Events are injected in batches so that no two events in a
batch lie within 7 days of each other on a shared series. Writes `injection_results.parquet`.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from e00038_base import (
    ADJACENCY,
    GRID_LENGTH,
    SOURCE_DIR,
    WORK_DIR,
    DetectorInputs,
    alarms,
    level_shift,
    prepare,
)
from e00038_detect_run import load_arrays
from e00038_intervals import (
    BLOCK,
    Interval,
    build_intervals,
    relative_residual,
    split_fractions,
)

N_EVENTS: Final[int] = 600
PHIS: Final[tuple[float, ...]] = (0.05, 0.15, 0.3)
QUIET_STAMPS: Final[int] = 7 * BLOCK
HIT_TOLERANCE: Final[int] = 4
WINDOW: Final[int] = 8
THRESHOLD: Final[float] = 4.0


@dataclass(frozen=True)
class Event:
    """One synthetic transfer."""

    event_id: int
    donor: int
    receivers: tuple[int, ...]
    onset: int
    duration: int
    phi: float
    base_fractions: tuple[float, ...]
    swing: tuple[float, ...]
    """Amplitude `b_k` of each receiver's daily sine; all zero for a constant split."""


def fractions_at(*, event: Event, stamps: np.ndarray) -> np.ndarray:
    """Return an array `(n_receivers, len(stamps))` of the split fractions, summing to 1."""
    hour = (stamps % BLOCK) / 2.0
    raw = np.stack(
        [
            np.clip(a + b * np.sin(2 * np.pi * hour / 24.0), 0.0, None)
            for a, b in zip(event.base_fractions, event.swing, strict=True)
        ]
    )
    return raw / np.maximum(raw.sum(axis=0), 1e-9)


def quiet_masks(*, oracle: pl.DataFrame, d1_alarms: pl.DataFrame) -> dict[int, np.ndarray]:
    """True within 7 days of a logged closure or a real D1 alarm on the series (not quiet)."""
    out = {}
    for sid in ADJACENCY:
        mark = np.zeros(GRID_LENGTH, dtype=bool)
        rows = oracle.filter((pl.col("series_a") == sid) | (pl.col("series_b") == sid))
        for first, last in rows.select("first_stamp", "last_stamp_exclusive").iter_rows():
            mark[max(first - QUIET_STAMPS, 0) : min(last + QUIET_STAMPS, GRID_LENGTH)] = True
        for stamp in d1_alarms.filter(pl.col("series") == sid)["stamp"]:
            mark[max(stamp - QUIET_STAMPS, 0) : min(stamp + QUIET_STAMPS, GRID_LENGTH)] = True
        out[sid] = mark
    return out


def draw_events(*, quiet: dict[int, np.ndarray], durations: np.ndarray) -> list[Event]:
    """Draw `N_EVENTS` events whose donor and receivers are quiet over the whole event."""
    rng = np.random.default_rng(38)
    events: list[Event] = []
    donors = list(ADJACENCY)
    while len(events) < N_EVENTS:
        donor = int(rng.choice(donors))
        n_receivers = int(rng.integers(1, 4))
        if len(ADJACENCY[donor]) < n_receivers:
            continue
        receivers = tuple(int(r) for r in rng.choice(ADJACENCY[donor], n_receivers, replace=False))
        duration = int(rng.choice(durations) * 2)
        onset = int(rng.integers(14 * BLOCK, 4 * 365 * BLOCK))
        if onset + duration >= GRID_LENGTH - BLOCK:
            continue
        span = slice(onset - QUIET_STAMPS, onset + duration + QUIET_STAMPS)
        if any(quiet[s][span].any() for s in (donor, *receivers)):
            continue
        base = rng.dirichlet(np.ones(n_receivers))
        varies = rng.random() < 0.5
        swing = rng.uniform(0.0, 0.3, n_receivers) if varies else np.zeros(n_receivers)
        events.append(
            Event(
                len(events),
                donor,
                receivers,
                onset,
                duration,
                float(rng.choice(PHIS)),
                tuple(float(x) for x in base),
                tuple(float(x) for x in swing),
            )
        )
    return events


def batches(*, events: list[Event]) -> list[list[Event]]:
    """Greedily split events into batches with no shared-series overlap within 7 days."""
    out: list[list[Event]] = []
    for event in events:
        for batch in out:
            if all(
                {event.donor, *event.receivers}.isdisjoint((o.donor, *o.receivers))
                or event.onset - QUIET_STAMPS > o.onset + o.duration
                or o.onset - QUIET_STAMPS > event.onset + event.duration
                for o in batch
            ):
                batch.append(event)
                break
        else:
            out.append([event])
    return out


def inject(
    *,
    batch: list[Event],
    residual: dict[int, np.ndarray],
    power: dict[int, np.ndarray],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Return copies of the residual and power with the batch's transfers applied."""
    res = {s: v.copy() for s, v in residual.items()}
    pw = {s: v.copy() for s, v in power.items()}
    for event in batch:
        stamps = np.arange(event.onset, event.onset + event.duration)
        moved = event.phi * power[event.donor][stamps]
        res[event.donor][stamps] -= moved
        pw[event.donor][stamps] -= moved
        shares = fractions_at(event=event, stamps=stamps)
        for receiver, share in zip(event.receivers, shares, strict=True):
            res[receiver][stamps] += share * moved
            pw[receiver][stamps] += share * moved
    return res, pw


def evaluate(
    *, event: Event, intervals: list[Interval], fractions: pl.DataFrame, variant: str
) -> dict:
    """Score one event: hit, overlap, end error, split error, and the within-day range."""
    true_end = event.onset + event.duration
    members = {event.donor, *event.receivers}
    near = [
        iv
        for iv in intervals
        if iv.role == "alarm"
        and iv.alarm_series in members
        and abs(iv.onset - event.onset) <= HIT_TOLERANCE
    ]
    record = {
        "variant": variant,
        "event_id": event.event_id,
        "donor": event.donor,
        "n_receivers": len(event.receivers),
        "phi": event.phi,
        "hours": event.duration / 2,
        "varies": max(event.swing) > 0,
        "hit": bool(near),
    }
    if not near:
        return record | dict.fromkeys(
            ("iou", "end_error_hours", "split_mae", "est_range", "true_range"), None
        )
    best = min(near, key=lambda iv: abs(iv.onset - event.onset))
    own = [iv for iv in intervals if iv.series == event.donor and iv.onset == best.onset]
    donor_iv = own[0] if own else best
    overlap = max(0, min(donor_iv.end, true_end) - max(donor_iv.onset, event.onset))
    union = max(donor_iv.end, true_end) - min(donor_iv.onset, event.onset)
    mean_true = fractions_at(event=event, stamps=np.arange(event.onset, true_end)).mean(axis=1)
    sub = fractions.filter(
        pl.col("alarm_series") == best.alarm_series, pl.col("onset") == best.onset
    )
    est_mean = {p: g["fraction"].mean() for (p,), g in sub.group_by("partner")}
    errors = [
        abs(est_mean.get(r, 0.0) - m) for r, m in zip(event.receivers, mean_true, strict=True)
    ]
    per_block_true = np.stack(
        [
            fractions_at(
                event=event, stamps=np.array([event.onset - event.onset % BLOCK + 12 * b + 6])
            )[:, 0]
            for b in range(4)
        ]
    )
    true_range = float((per_block_true.max(axis=0) - per_block_true.min(axis=0)).max())
    ranges = [
        float(g["fraction"].max() - g["fraction"].min())  # ty: ignore[unsupported-operator]
        for (p,), g in sub.group_by("partner")
        if p in event.receivers
    ]
    return record | {
        "iou": overlap / union if union else 0.0,
        "end_error_hours": (donor_iv.end - true_end) / 2,
        "split_mae": float(np.mean(errors)) if sub.height else None,
        "est_range": max(ranges) if ranges else None,
        "true_range": true_range,
    }


def main() -> None:
    """Inject, detect, build intervals with both builders, and score."""
    panel = pl.read_parquet(SOURCE_DIR / "panel.parquet")
    residual = load_arrays(panel=panel, column="res_mw")
    expected = load_arrays(panel=panel, column="expected_mw")
    power = load_arrays(panel=panel, column="obs_mw")
    oracle = pl.read_parquet(SOURCE_DIR / "oracle_real.parquet").filter("primary")
    d1_alarms = pl.read_parquet(WORK_DIR / "alarms_d1.parquet")
    durations = oracle.filter("confirmed", pl.col("hours") >= 6)["hours"].to_numpy()
    events = draw_events(quiet=quiet_masks(oracle=oracle, d1_alarms=d1_alarms), durations=durations)
    scale = prepare(residuals=residual, window=WINDOW).scale
    results = []
    for batch in batches(events=events):
        res, pw = inject(batch=batch, residual=residual, power=power)
        shift = {s: level_shift(values=v, window=WINDOW) for s, v in res.items()}
        table = alarms(
            inputs=DetectorInputs(window=WINDOW, shift=shift, scale=scale),
            detector="D1",
            threshold=THRESHOLD,
        )
        relative = {s: relative_residual(residual=res[s], expected=expected[s]) for s in res}
        for variant, fixed in (("spec", False), ("fixed", True)):
            intervals = build_intervals(
                alarm_table=table, relative=relative, window=WINDOW, detector="D1", fixed=fixed
            )
            fractions = split_fractions(power=pw, interval_set=intervals)
            results.extend(
                evaluate(event=e, intervals=intervals, fractions=fractions, variant=variant)
                for e in batch
            )
        print(f"batch of {len(batch)} done", flush=True)
    frame = pl.DataFrame(results, infer_schema_length=None)
    frame.write_parquet(WORK_DIR / "injection_results.parquet")
    print(
        frame.group_by("variant", "phi")
        .agg(n=pl.len(), recall=pl.col("hit").mean(), iou=pl.col("iou").mean())
        .sort("variant", "phi")
    )


if __name__ == "__main__":
    main()
