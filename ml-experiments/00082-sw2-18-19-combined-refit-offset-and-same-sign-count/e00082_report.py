"""Print every number the write-up of experiment 82 quotes, and the verdict the decision rule gives.

Reads the tables that `e00082_run.py` wrote: MW-weighted recall of PF14, X, LOGIT_SAME, and XL on
the fresh headline-band and long draws, by injection shape, with month-resampled intervals; the
planned contrasts C1, C2, and C3; recall by donor step; the labelled real transfers; and the alarm
counts.

    uv run python ml-experiments/00070-*/e00082_report.py
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00082_common import WORK_DIR, stamp_months, stamp_year_months
from e00082_profile import NON_MATCHING, SHAPE_NAMES, SHAPES
from e00082_run import ARMS, TARGET

N_RESAMPLES: Final[int] = 20_000
LEVEL: Final[float] = 1 - 0.05 / 3
"""Two-sided level of the intervals: Bonferroni across the three planned contrasts (98.33%)."""
SEED: Final[int] = 8200
THRESHOLD: Final[float] = 0.02
STEP_EDGES: Final[tuple[float, ...]] = (0.0, 2.0, 4.0, 6.0, 10.0, float("inf"))
DAY_EDGES: Final[tuple[float, ...]] = (1.0, 2.0, 7.0, 14.0, 15.6, 30.0, 61.0)
EXPECTED_ROWS: Final[dict[str, int]] = {"base": 1160, "long": 400}


def populations(*, out: Path) -> pl.DataFrame:
    """A row per transfer, shape, and arm for the fresh `base` and `long` draws.

    The weight of a transfer is its mean moved MW under S1. `parity` is `even` or `odd` by the UTC
    month of the onset.
    """
    events = pl.read_parquet(out / "events.parquet")
    for population, n in EXPECTED_ROWS.items():
        if events.filter(pl.col("population") == population).height != n:
            raise ValueError(f"the {population} draw does not hold {n} transfers")
    if events["event_id"].n_unique() != events.height:
        raise ValueError("event ids repeat")
    onset = events["onset"].to_numpy()
    month = stamp_months()[onset]
    events = events.with_columns(
        year_month=pl.Series(stamp_year_months()[onset]),
        parity=pl.Series(np.where(month % 2 == 0, "even", "odd")),
    )
    results = pl.read_parquet(out / "event_results.parquet")
    weight = (
        results.filter(pl.col("shape") == "S1", pl.col("setting") == "PF14")
        .select("event_id", weight=pl.col("moved_mw"))
        .unique("event_id")
    )
    return (
        results.join(events, on="event_id", validate="m:1")
        .join(weight, on="event_id", validate="m:1")
        .sort("shape", "setting", "event_id")
    )


def wide(*, data: pl.DataFrame, shape: str, population: str, parity: str) -> pl.DataFrame:
    """One row per transfer with every arm's `found` (columns named by setting) for one shape."""
    one = data.filter(
        pl.col("shape") == shape, pl.col("population") == population, pl.col("parity") == parity
    )
    frame = one.filter(pl.col("setting") == "PF14").select(
        "event_id",
        "weight",
        "year_month",
        "metered_receivers",
        "donor_step_z",
        "hours",
        PF14="found",
        onset_PF14="onset_found",
    )
    for setting in ARMS:
        if setting == "PF14":
            continue
        part = one.filter(pl.col("setting") == setting)
        frame = frame.join(
            part.select("event_id", **{setting: "found", f"onset_{setting}": "onset_found"}),
            on="event_id",
            validate="1:1",
        )
    return frame.sort("event_id")


def interval(*, resampled: np.ndarray) -> tuple[float, float]:
    """The two-sided interval of resampled statistics at `LEVEL`."""
    tail = 100 * (1 - LEVEL) / 2
    low, high = np.percentile(resampled, [tail, 100 - tail])
    return float(low), float(high)


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
    """MW-weighted recall of `arm` (minus `base`), averaged over shapes; `picks` resamples months.

    Every frame must have the same months, so one set of picks serves all shapes.
    """
    total = 0.0
    for frame in frames:
        wf_a, w = month_sums(frame=frame, found=arm)
        if len(w) != len(np.unique(frames[0]["year_month"].to_numpy())):
            raise ValueError("the shapes cover different months")
        wf_b = month_sums(frame=frame, found=base)[0] if base else np.zeros_like(wf_a)
        if picks is None:
            total = total + (wf_a.sum() - wf_b.sum()) / w.sum()
        else:
            total = total + (wf_a[picks].sum(axis=1) - wf_b[picks].sum(axis=1)) / w[picks].sum(
                axis=1
            )
    return np.atleast_1d(total / len(frames))


def contrast(
    *, frames: list[pl.DataFrame], arm: str, base: str | None = None
) -> tuple[float, float, float]:
    """Point value and interval of the shape-averaged MW-weighted recall (difference)."""
    months = np.unique(frames[0]["year_month"].to_numpy())
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(months), size=(N_RESAMPLES, len(months)))
    point = float(stat(frames=frames, arm=arm, base=base, picks=None)[0])
    low, high = interval(resampled=stat(frames=frames, arm=arm, base=base, picks=picks))
    return point, low, high


def fmt(point: float, low: float, high: float) -> str:
    """A point value with its interval."""
    return f"{point:+.3f} ({low:+.3f} to {high:+.3f})"


def pooled(*, data: pl.DataFrame, population: str, parity: str) -> list[pl.DataFrame]:
    """The three non-matching shapes' wide frames."""
    return [wide(data=data, shape=s, population=population, parity=parity) for s in NON_MATCHING]


def print_recall(*, data: pl.DataFrame, population: str, parity: str) -> dict[str, float]:
    """Absolute MW-weighted recall per arm and shape, with the pooled interval and the missed MW."""
    frames = pooled(data=data, population=population, parity=parity)
    print(f"\n== {population} transfers, {parity} months: MW-weighted recall (S2, S3, S4 pooled)")
    if frames[0]["weight"].null_count():
        raise ValueError("a transfer has no weight")
    print(f"transfers per shape: {frames[0].height}; MW weight {frames[0]['weight'].sum():.1f}")
    point: dict[str, float] = {}
    for setting in ARMS:
        value, low, high = contrast(frames=frames, arm=setting)
        point[setting] = value
        by_shape = " ".join(
            f"{s} {contrast(frames=[f], arm=setting)[0]:.3f}"
            for s, f in zip(NON_MATCHING, frames, strict=True)
        )
        no_start = np.mean(
            [
                float((f["weight"] * (~f[f"onset_{setting}"]).cast(pl.Float64)).sum())
                / float(f["weight"].sum())
                for f in frames
            ]
        )
        wrong_end = np.mean(
            [
                float((f["weight"] * (f[f"onset_{setting}"] & ~f[setting]).cast(pl.Float64)).sum())
                / float(f["weight"].sum())
                for f in frames
            ]
        )
        unweighted = np.mean([f[setting].cast(pl.Float64).mean() for f in frames])
        print(
            f"{setting:9s} {value:.3f} ({low:.3f} to {high:.3f}) [{by_shape}] "
            f"missed {1 - value:.3f} = no start {no_start:.3f} + wrong end {wrong_end:.3f}; "
            f"unweighted {unweighted:.3f}"
        )
    return point


def print_all_shapes(*, data: pl.DataFrame, population: str) -> None:
    """Recall of every arm on S1 and S5 too (outside the headline), even months."""
    for shape in ("S1", "S5"):
        frame = wide(data=data, shape=shape, population=population, parity="even")
        print(
            f"{population} even {shape} {SHAPE_NAMES[shape]}: "
            + " ".join(f"{s} {contrast(frames=[frame], arm=s)[0]:.3f}" for s in ARMS)
        )


def print_by_step(*, data: pl.DataFrame, population: str) -> None:
    """Standard table: recall by donor step in noise units, split by metered receiver (S2-S4)."""
    print(
        f"\nRecall by donor step ({population}, even months): found/transfers, shapes S2-S4 summed"
    )
    for flag, label in ((True, "metered"), (False, "unmetered")):
        for low, high in itertools.pairwise(STEP_EDGES):
            n = 0
            counts = dict.fromkeys(ARMS, 0)
            for frame in pooled(data=data, population=population, parity="even"):
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


def print_by_duration(*, data: pl.DataFrame) -> None:
    """MW-weighted recall by duration in days, both populations pooled, S2-S4, even months."""
    print("\nRecall by duration in days (base and long transfers together, even months)")
    frames = [
        pl.concat([wide(data=data, shape=s, population=p, parity="even") for p in ("base", "long")])
        for s in NON_MATCHING
    ]
    for low, high in itertools.pairwise(DAY_EDGES):
        cells = []
        n = 0
        for arm in ARMS:
            vals = []
            for frame in frames:
                part = frame.filter((pl.col("hours") / 24).is_between(low, high, "left"))
                n = part.height
                w = part["weight"].to_numpy()
                vals.append(
                    (w * part[arm].cast(pl.Float64).to_numpy()).sum() / w.sum()
                    if len(w)
                    else np.nan
                )
            cells.append(f"{arm} {np.mean(vals):.3f}")
        print(f"[{low:g},{high:g}) days n={n} per shape: " + " ".join(cells))


def print_real(*, out: Path) -> None:
    """The labelled real transfers of more than 1 day, found by each arm."""
    real = pl.read_parquet(out / "real_results.parquet").filter(pl.col("hours") > 24)
    n = real.filter(pl.col("setting") == "PF14").height
    print(f"\nLabelled real transfers of more than 1 day: {n}")
    for setting in ARMS:
        part = real.filter(pl.col("setting") == setting)
        w = part["moved_mw_p50"].fill_null(0).to_numpy()
        f = part["found"].to_numpy().astype(float)
        print(
            f"  {setting:9s} found {int(f.sum())}, onset found {part['onset_found'].sum()}, "
            f"MW-weighted recall {(w * f).sum() / w.sum():.3f}"
        )


def print_alarms(*, out: Path) -> None:
    """Unexplained alarm counts by even and odd month for each arm."""
    alarms = pl.read_parquet(out / "alarms.parquet")
    print(f"\nthresholds {json.loads((out / 'thresholds.json').read_text())}")
    for arm in ARMS:
        part = alarms.filter(pl.col("setting") == arm, ~pl.col("logged_matched"))
        even = part.filter(pl.col("month") % 2 == 0).height
        odd = part.filter(pl.col("month") % 2 == 1).height
        print(f"{arm}: unexplained alarms even months {even}, odd {odd}")
        if odd > TARGET:
            raise ValueError(f"{arm} has {odd} odd-month unexplained alarms")


def decide(
    *,
    c1: tuple[float, float, float],
    c2: tuple[float, float, float],
    c3: tuple[float, float, float],
) -> str:
    """Apply the pre-registered verdict rule to C1, C2, and C3 (point, lower end, upper end)."""
    if c1[2] < THRESHOLD:
        return "refuted"
    if c1[1] > 0 and c1[0] >= THRESHOLD and c2[0] > 0 and c3[0] > 0:
        return "supported"
    if c1[1] > 0:
        return "small gain"
    return "inconclusive"


def main() -> None:
    """Print the tables and the verdict."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    out = parser.parse_args().out
    data = populations(out=out)
    for population, parity in itertools.product(("base", "long"), ("even", "odd")):
        print_recall(data=data, population=population, parity=parity)
    for population in ("base", "long"):
        print_all_shapes(data=data, population=population)
    base = pooled(data=data, population="base", parity="even")
    long = pooled(data=data, population="long", parity="even")
    c = {
        "C1": contrast(frames=base, arm="XL", base="PF14"),
        "C2": contrast(frames=base, arm="XL", base="X"),
        "C3": contrast(frames=base, arm="XL", base="LOGIT_SAME"),
    }
    print("\nPlanned contrasts, XL minus PF14 / X / LOGIT_SAME (98.33% interval, even months)")
    for name, value in c.items():
        print(f"  {name}: {fmt(*value)}")
    print("Descriptive contrasts")
    for label, frames, arm, ref in (
        ("base, X minus PF14", base, "X", "PF14"),
        ("base, LOGIT_SAME minus PF14", base, "LOGIT_SAME", "PF14"),
        ("long, XL minus PF14", long, "XL", "PF14"),
        ("long, XL minus X", long, "XL", "X"),
        ("long, XL minus LOGIT_SAME", long, "XL", "LOGIT_SAME"),
    ):
        print(f"  {label}: {fmt(*contrast(frames=frames, arm=arm, base=ref))}")
    print("XL minus PF14 by shape, base, even months")
    for shape in SHAPES:
        frame = wide(data=data, shape=shape, population="base", parity="even")
        print(f"  {shape}: {fmt(*contrast(frames=[frame], arm='XL', base='PF14'))}")
    print("Odd-month check (not used for any decision)")
    for population in ("base", "long"):
        frames = pooled(data=data, population=population, parity="odd")
        print(
            f"  {population} odd, XL minus PF14: "
            f"{fmt(*contrast(frames=frames, arm='XL', base='PF14'))}"
        )
    print_by_step(data=data, population="base")
    print_by_duration(data=data)
    print_real(out=out)
    print_alarms(out=out)
    print(f"VERDICT: {decide(c1=c['C1'], c2=c['C2'], c3=c['C3'])}")


if __name__ == "__main__":
    main()
