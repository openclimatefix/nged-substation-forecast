r"""Print every number of the experiment 138 write-up, and the verdict the decision rule gives.

Reads the tables that `e00138_run.py` wrote. The arms are `B` (RO-NC on the frozen convex
residual, the control), `W1`, `W2`, and `W3`. The planned contrasts are each W arm minus `B`.

    uv run --no-sync python ml-experiments/00138-*/e00138_contrasts.py
"""

from __future__ import annotations

import itertools
import json
from typing import Final

import numpy as np
import polars as pl
from e00138_arms import ARMS
from e00138_common import EXPERIMENT_130_RUN_DIR, WORK_DIR, stamp_months, stamp_year_months
from e00138_profile import NON_MATCHING, SHAPES

RUN_DIR: Final = WORK_DIR / "run"
N_RESAMPLES: Final[int] = 20_000
LEVEL: Final[float] = 1 - 0.05 / 3
"""The Bonferroni level for three planned contrasts: 98.33%."""
SEED: Final[int] = 138
THRESHOLD: Final[float] = 0.03
REFUTED_BELOW: Final[float] = 0.02
CONTROL: Final[str] = "B"
WEAK_ARMS: Final[tuple[str, ...]] = tuple(a for a in ARMS if a != CONTROL)
STEP_EDGES: Final[tuple[float, ...]] = (0.0, 2.0, 4.0, 6.0, 10.0, float("inf"))
BLANK_SERIES: Final[int] = 12
REPRODUCTION_TOLERANCE: Final[float] = 0.005
EXPERIMENT_130_RECALL: Final[float] = 0.794


def populations() -> pl.DataFrame:
    """A row per transfer, shape, and arm for the fresh `base` and `long` draws."""
    events = pl.read_parquet(RUN_DIR / "events.parquet")
    onset = events["onset"].to_numpy()
    month = stamp_months()[onset]
    events = events.with_columns(
        year_month=pl.Series(stamp_year_months()[onset]),
        parity=pl.Series(np.where(month % 2 == 0, "even", "odd")),
    ).filter(pl.col("donor") != BLANK_SERIES)
    results = pl.read_parquet(RUN_DIR / "event_results.parquet")
    weight = (
        results.filter(pl.col("shape") == "S1", pl.col("setting") == CONTROL)
        .select("event_id", weight=pl.col("moved_mw"))
        .unique("event_id")
    )
    return results.join(events, on="event_id", validate="m:1").join(
        weight, on="event_id", validate="m:1"
    )


def wide(*, data: pl.DataFrame, shape: str, population: str, parity: str) -> pl.DataFrame:
    """One row per transfer with every arm's `found` (columns named by arm) for one shape."""
    one = data.filter(
        pl.col("shape") == shape, pl.col("population") == population, pl.col("parity") == parity
    )
    frame = one.filter(pl.col("setting") == CONTROL).select(
        "event_id",
        "weight",
        "year_month",
        "metered_receivers",
        "donor_step_z",
        "hours",
        **{CONTROL: "found", f"onset_{CONTROL}": "onset_found"},
    )
    for arm in WEAK_ARMS:
        part = one.filter(pl.col("setting") == arm)
        frame = frame.join(
            part.select("event_id", **{arm: "found", f"onset_{arm}": "onset_found"}),
            on="event_id",
            validate="1:1",
        )
    return frame.sort("event_id")


def pooled(*, data: pl.DataFrame, population: str, parity: str) -> list[pl.DataFrame]:
    """The three non-matching shapes' wide frames."""
    return [wide(data=data, shape=s, population=population, parity=parity) for s in NON_MATCHING]


def month_sums(*, frame: pl.DataFrame, found: str) -> tuple[np.ndarray, np.ndarray]:
    """Per calendar month: the weight-times-found sum and the weight sum."""
    grouped = (
        frame.with_columns(wf=pl.col("weight") * pl.col(found).cast(pl.Float64))
        .group_by("year_month")
        .agg(pl.col("wf").sum(), pl.col("weight").sum())
        .sort("year_month")
    )
    return grouped["wf"].to_numpy(), grouped["weight"].to_numpy()


def stat(
    *, frames: list[pl.DataFrame], arm: str, base: str | None, picks: np.ndarray | None
) -> np.ndarray:
    """MW-weighted recall of `arm` (minus `base`) over shapes; `picks` resamples months."""
    total = 0.0
    for frame in frames:
        wf_a, w = month_sums(frame=frame, found=arm)
        wf_b = month_sums(frame=frame, found=base)[0] if base else np.zeros_like(wf_a)
        if picks is None:
            total = total + (wf_a.sum() - wf_b.sum()) / w.sum()
        else:
            total = total + (wf_a[picks].sum(axis=1) - wf_b[picks].sum(axis=1)) / w[picks].sum(
                axis=1
            )
    return np.atleast_1d(total / len(frames))


def contrast(
    *, frames: list[pl.DataFrame], arm: str, base: str | None = None, level: float = LEVEL
) -> tuple[float, float, float]:
    """Point value and interval of the shape-averaged MW-weighted recall (difference)."""
    months = np.unique(frames[0]["year_month"].to_numpy())
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(months), size=(N_RESAMPLES, len(months)))
    point = float(stat(frames=frames, arm=arm, base=base, picks=None)[0])
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(
        stat(frames=frames, arm=arm, base=base, picks=picks), [tail, 100 - tail]
    )
    return point, float(low), float(high)


def fmt(point: float, low: float, high: float) -> str:
    """A point value with its interval."""
    return f"{point:+.3f} ({low:+.3f} to {high:+.3f})"


def decide(*, contrasts: dict[str, tuple[float, float, float]]) -> str:
    """Apply the pre-registered verdict rule to the planned contrasts (point, low, high)."""
    if all(c[2] < REFUTED_BELOW for c in contrasts.values()):
        return "refuted"
    best = max(contrasts.values(), key=lambda c: c[1])
    if best[1] > THRESHOLD:
        return "supported"
    if best[1] > 0:
        return "small gain"
    return "inconclusive"


def print_recall(*, data: pl.DataFrame, population: str, parity: str) -> None:
    """Absolute MW-weighted recall per arm with its 95% interval."""
    frames = pooled(data=data, population=population, parity=parity)
    print(f"\n== {population} transfers, {parity} months: MW-weighted recall (S2, S3, S4 pooled)")
    print(f"transfers per shape: {frames[0].height}; MW weight {frames[0]['weight'].sum():.1f}")
    for arm in ARMS:
        value, low, high = contrast(frames=frames, arm=arm, level=0.95)
        print(f"{arm:4s} {value:.3f} ({low:.3f} to {high:.3f})")


def print_control(*, data: pl.DataFrame) -> None:
    """Arm B must reproduce experiment 130's arm B, row by row where the stored file exists."""
    stored = pl.read_parquet(EXPERIMENT_130_RUN_DIR / "event_results.parquet").filter(
        pl.col("setting") == "XL"
    )
    mine = data.filter(pl.col("setting") == CONTROL)
    joined = mine.join(stored, on=["event_id", "shape"], suffix="_130", validate="1:1")
    differ = joined.filter(pl.col("found") != pl.col("found_130")).height
    print(f"\nControl: {joined.height} rows of arm B joined to experiment 130's, {differ} differ")
    value = contrast(frames=pooled(data=data, population="base", parity="even"), arm=CONTROL)[0]
    print(f"  arm B recall {value:.3f} against {EXPERIMENT_130_RECALL}")
    if abs(value - EXPERIMENT_130_RECALL) > REPRODUCTION_TOLERANCE:
        print("  CONTROL FAILED: arm B does not reproduce experiment 130 within 0.005")


def print_by_step(*, data: pl.DataFrame) -> None:
    """Transfers found by donor step in noise units, split by metered receiver (S2 to S4 summed)."""
    print("\nTransfers found by donor step (base, even months): n, then found per arm")
    for flag, label in ((True, "metered"), (False, "unmetered")):
        for low, high in itertools.pairwise(STEP_EDGES):
            n, counts = 0, dict.fromkeys(ARMS, 0)
            for frame in pooled(data=data, population="base", parity="even"):
                part = frame.filter(
                    (pl.col("metered_receivers") > 0) == flag,
                    pl.col("donor_step_z").abs().is_between(low, high, "left"),
                )
                n += part.height
                for arm in ARMS:
                    counts[arm] += int(part[arm].sum())
            print(
                f"{label:9s} [{low:g},{high:g}) n={n}: "
                + " ".join(f"{a} {counts[a]}" for a in ARMS)
            )


def print_real() -> None:
    """The labelled real transfers of more than 1 day, found by each arm."""
    real = pl.read_parquet(RUN_DIR / "real_results.parquet").filter(pl.col("hours") > 24)
    print("\nLabelled real transfers of more than 1 day (descriptive)")
    for arm in ARMS:
        part = real.filter(pl.col("setting") == arm)
        print(f"  {arm}: found {int(part['found'].sum())} of {part.height}")


def print_alarms() -> None:
    """Thresholds, and unexplained alarms per odd month and in total, for each arm."""
    alarms = pl.read_parquet(RUN_DIR / "alarms.parquet")
    print(f"\nThresholds {json.loads((RUN_DIR / 'thresholds.json').read_text())}")
    for arm in ARMS:
        odd = alarms.filter(
            pl.col("setting") == arm, ~pl.col("logged_matched"), pl.col("month") % 2 == 1
        ).height
        even = alarms.filter(
            pl.col("setting") == arm, ~pl.col("logged_matched"), pl.col("month") % 2 == 0
        ).height
        print(f"  {arm}: unexplained alarms on odd months {odd}, on even months {even}")


def main() -> None:
    """Print the tables and the verdict."""
    data = populations()
    print_control(data=data)
    for population, parity in itertools.product(("base", "long"), ("even", "odd")):
        print_recall(data=data, population=population, parity=parity)
    base = pooled(data=data, population="base", parity="even")
    contrasts = {arm: contrast(frames=base, arm=arm, base=CONTROL) for arm in WEAK_ARMS}
    print("\nPlanned contrasts, each arm minus B (98.33% interval, base, even months)")
    for arm, value in contrasts.items():
        print(f"  {arm} minus B: {fmt(*value)}")
    print("Each arm minus B by shape, base, even months")
    for shape in SHAPES:
        frame = wide(data=data, shape=shape, population="base", parity="even")
        for arm in WEAK_ARMS:
            print(f"  {shape} {arm}: {fmt(*contrast(frames=[frame], arm=arm, base=CONTROL))}")
    long = pooled(data=data, population="long", parity="even")
    print("Long transfers, each arm minus B")
    for arm in WEAK_ARMS:
        print(f"  {arm}: {fmt(*contrast(frames=long, arm=arm, base=CONTROL))}")
    print_by_step(data=data)
    print_real()
    print_alarms()
    print(f"VERDICT: {decide(contrasts=contrasts)}")


if __name__ == "__main__":
    main()
