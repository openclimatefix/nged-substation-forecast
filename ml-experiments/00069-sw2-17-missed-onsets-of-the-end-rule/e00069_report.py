"""Print every number the write-up of experiment 69 quotes, and the verdict the decision rule gives.

Reads the tables that `e00069_run.py` wrote: MW-weighted recall of D0M and the end rule at caps of
14, 30, and 60 days, for the 24-hour window and two 4-hour alarm settings, on experiment 58's 620
even-month transfers and on the long set, by injection shape, with month-resampled intervals; the
planned contrasts C1 to C5; recall by donor step, by duration, and on the labelled real transfers.

    uv run python ml-experiments/00069-*/e00069_report.py
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00069_common import ML_EXPERIMENTS_DATA_DIR, WORK_DIR, stamp_months, stamp_year_months
from e00069_inject import HEADLINE_BAND
from e00069_profile import NON_MATCHING, SHAPE_NAMES
from e00069_run import ALARM_SETTINGS, SETTINGS

EXPERIMENT_61_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00061-sw2-6-paired-step-intervals"
N_RESAMPLES: Final[int] = 20_000
LEVEL: Final[float] = 1 - 0.05 / 5
"""Two-sided level of the intervals: Bonferroni across the five planned contrasts (99%)."""
SEED: Final[int] = 6600
THRESHOLD: Final[float] = 0.01
GUARD: Final[float] = -0.01
STEP_EDGES: Final[tuple[float, ...]] = (0.0, 2.0, 4.0, 6.0, 10.0, float("inf"))
DAY_EDGES: Final[tuple[float, ...]] = (1.0, 2.0, 7.0, 14.0, 15.6, 30.0, 61.0)
BENCHMARK: Final[str] = "PE14_48"
REAL_LONG_SHARE: Final[float] = 25 / 151
"""Labelled real operations of 15.6 days or more, as a share of those over 1 day."""


def populations(*, out: Path) -> pl.DataFrame:
    """A row per transfer, shape, and arm: the 620-style base transfers and the long set.

    The weight of a transfer is its mean moved MW under S1. `population` is `base` (headline band,
    not the holiday set) or `long`; `parity` is `even` or `odd` by the UTC month of the onset.
    """
    events = pl.read_parquet(out / "events.parquet")
    onset = events["onset"].to_numpy()
    month = stamp_months()[onset]
    events = events.with_columns(
        year_month=pl.Series(stamp_year_months()[onset]),
        parity=pl.Series(np.where(month % 2 == 0, "even", "odd")),
        population=pl.when(pl.col("set") == "long")
        .then(pl.lit("long"))
        .when(
            (pl.col("size_band") == HEADLINE_BAND[0])
            & (pl.col("duration_band") == HEADLINE_BAND[1])
            & (pl.col("set") != "holiday")
        )
        .then(pl.lit("base"))
        .otherwise(pl.lit("other")),
    ).filter(pl.col("population") != "other")
    results = pl.read_parquet(out / "event_results.parquet")
    weight = (
        results.filter(pl.col("shape") == "S1", pl.col("setting") == "D0M_48")
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
    frame = one.filter(pl.col("setting") == "D0M_48").select(
        "event_id",
        "weight",
        "year_month",
        "metered_receivers",
        "donor_step_z",
        "hours",
        D0M_48="found",
        onset_D0M_48="onset_found",
    )
    for setting in SETTINGS:
        if setting == "D0M_48":
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
    expected = {("base", "even"): 620, ("long", "even"): None}.get((population, parity))
    if expected is not None and frames[0].height != expected:
        raise ValueError(f"{frames[0].height} transfers, the plan says {expected}")
    if frames[0]["weight"].null_count():
        raise ValueError("a transfer has no weight")
    print(f"transfers per shape: {frames[0].height}; MW weight {frames[0]['weight'].sum():.1f}")
    point: dict[str, float] = {}
    for setting in SETTINGS:
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
        wrong_end_n = np.mean([(f[f"onset_{setting}"] & ~f[setting]).sum() for f in frames])
        print(
            f"{setting:9s} {value:.3f} ({low:.3f} to {high:.3f}) [{by_shape}] "
            f"missed {1 - value:.3f} = no start {no_start:.3f} + wrong end {wrong_end:.3f}; "
            f"unweighted {unweighted:.3f}; start found, end wrong: {wrong_end_n:.1f} transfers"
        )
    return point


def print_all_shapes(*, data: pl.DataFrame, population: str, parity: str) -> None:
    """Recall of the benchmark and the chosen arms on S1 and S5 too (outside the headline)."""
    for shape in ("S1", "S5"):
        frame = wide(data=data, shape=shape, population=population, parity=parity)
        print(
            f"{population} {parity} {shape} {SHAPE_NAMES[shape]}: "
            + " ".join(f"{s} {contrast(frames=[frame], arm=s)[0]:.3f}" for s in SETTINGS)
        )


def print_by_step(*, data: pl.DataFrame, population: str) -> None:
    """Standard table: recall by donor step in noise units, split by metered receiver (S2-S4)."""
    print(
        f"\nRecall by donor step ({population}, even months): found/transfers, shapes S2-S4 summed"
    )
    arms = [a for a in SETTINGS if a.endswith(("_48", "_8m"))]
    for flag, label in ((True, "metered"), (False, "unmetered")):
        for low, high in itertools.pairwise(STEP_EDGES):
            cells = []
            n = 0
            counts = dict.fromkeys(arms, 0)
            for frame in pooled(data=data, population=population, parity="even"):
                part = frame.filter(
                    (pl.col("metered_receivers") > 0) == flag,
                    pl.col("donor_step_z").abs().is_between(low, high, "left"),
                )
                n += part.height
                for arm in arms:
                    counts[arm] += int(part[arm].sum())
            cells = " ".join(f"{a} {counts[a]}" for a in arms)
            print(f"{label:9s} [{low:g},{high:g}) n={n}: {cells}")


def recall_by_duration(*, data: pl.DataFrame, arms: list[str]) -> list[dict]:
    """MW-weighted recall by duration bin, both populations pooled, S2-S4 averaged, even months."""
    frames = [
        pl.concat([wide(data=data, shape=s, population=p, parity="even") for p in ("base", "long")])
        for s in NON_MATCHING
    ]
    rows = []
    for low, high in itertools.pairwise(DAY_EDGES):
        for arm in arms:
            vals = []
            n = 0
            for frame in frames:
                part = frame.filter((pl.col("hours") / 24).is_between(low, high, "left"))
                n = part.height
                w = part["weight"].to_numpy()
                vals.append(
                    (w * part[arm].cast(pl.Float64).to_numpy()).sum() / w.sum()
                    if len(w)
                    else np.nan
                )
            rows.append(
                {"low": low, "high": high, "arm": arm, "n": n, "recall": float(np.mean(vals))}
            )
    return rows


def print_by_duration(*, data: pl.DataFrame) -> None:
    """MW-weighted recall by duration in days, both populations pooled, S2-S4, even months."""
    print("\nRecall by duration in days (base and long transfers together, even months)")
    arms = [a for a in SETTINGS if a.endswith(("_48", "_8m"))]
    frames = [
        pl.concat([wide(data=data, shape=s, population=p, parity="even") for p in ("base", "long")])
        for s in NON_MATCHING
    ]
    for low, high in itertools.pairwise(DAY_EDGES):
        cells = []
        n = 0
        for arm in arms:
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
    n = real.filter(pl.col("setting") == "D0M_48").height
    print(f"\nLabelled real transfers of more than 1 day: {n}")
    for setting in SETTINGS:
        part = real.filter(pl.col("setting") == setting)
        w = part["moved_mw_p50"].fill_null(0).to_numpy()
        f = part["found"].to_numpy().astype(float)
        longer = part.filter(pl.col("hours") > 14 * 24)
        print(
            f"  {setting:9s} found {int(f.sum())}, onset found {part['onset_found'].sum()}, "
            f"MW-weighted recall {(w * f).sum() / w.sum():.3f}; over 14 days: "
            f"{longer['found'].sum()} of {longer.height}"
        )


def print_alarms(*, out: Path) -> None:
    """Unexplained alarm counts by even and odd month for each alarm setting."""
    alarms = pl.read_parquet(out / "alarms.parquet")
    print(f"\nthresholds {json.loads((out / 'thresholds.json').read_text())}")
    odd_limits = {"48": 80, "8": 120, "8m": 80}
    for tag in ALARM_SETTINGS:
        part = alarms.filter(pl.col("setting") == f"D0M_{tag}", ~pl.col("logged_matched"))
        even = part.filter(pl.col("month") % 2 == 0).height
        odd = part.filter(pl.col("month") % 2 == 1).height
        print(f"D0M_{tag}: unexplained alarms even months {even}, odd {odd}")
        if odd > odd_limits[tag]:
            raise ValueError(f"D0M_{tag} has {odd} odd-month unexplained alarms")


def compare_with_61(*, data: pl.DataFrame) -> None:
    """The 14-day end rule at 24 hours must equal experiment 61's PE on every shape."""
    old = pl.read_parquet(EXPERIMENT_61_DIR / "event_results.parquet").filter(
        pl.col("setting") == "PE_48"
    )
    mine = data.filter(pl.col("setting") == BENCHMARK, pl.col("population") == "base")
    joined = mine.join(old, on=["event_id", "shape"], suffix="_61", validate="1:1")
    different = joined.filter(pl.col("found") != pl.col("found_61")).height
    print(f"\n{BENCHMARK} against experiment 61's PE: {joined.height} rows, {different} differ")


def print_mixture(*, long: list[pl.DataFrame], base: list[pl.DataFrame]) -> None:
    """Post hoc: the gain over PE14_48 if transfers over 15.6 days were as common as in the logs.

    Of the 151 labelled real operations lasting more than 1 day, 25 last 15.6 days or more
    (`REAL_LONG_SHARE`). The point values of the long-set and base contrasts are mixed at that
    share.
    """
    print(f"\nPOST HOC mixture: share of transfers over 15.6 days {REAL_LONG_SHARE:.3f}")
    for arm in ("PE30_48", "PE60_48", "PF14_48", "PF30_48", "PF60_48"):
        on_long = contrast(frames=long, arm=arm, base=BENCHMARK)[0]
        on_base = contrast(frames=base, arm=arm, base=BENCHMARK)[0]
        mixed = REAL_LONG_SHARE * on_long + (1 - REAL_LONG_SHARE) * on_base
        print(
            f"  {arm} minus {BENCHMARK}: long {on_long:+.3f}, base {on_base:+.3f}, mix {mixed:+.3f}"
        )


def decide(*, c: dict[str, tuple[float, float, float]], long_recall: dict[str, float]) -> str:
    """Apply the pre-registered decision rule to the five contrasts."""
    window4 = c["C5"][1] > 0 and c["C5"][0] >= THRESHOLD
    passes = {
        30: c["C1"][1] > 0 and c["C1"][0] >= THRESHOLD and c["C3"][0] >= GUARD,
        60: c["C2"][1] > 0 and c["C2"][0] >= THRESHOLD and c["C4"][0] >= GUARD,
    }
    cap = 14
    if passes[30]:
        cap = 30
    if passes[60] and (
        not passes[30] or long_recall["PE60_48"] - long_recall["PE30_48"] >= THRESHOLD
    ):
        cap = 60
    print(f"window changes to 4 hours: {window4}; caps passing: {passes}; cap chosen: {cap}")
    if window4 or cap != 14:
        return "supported"
    if all(c[k][2] < THRESHOLD for k in ("C1", "C2", "C5")):
        return "refuted"
    return "inconclusive"


def main() -> None:
    """Print the tables and the verdict."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    out = parser.parse_args().out
    data = populations(out=out)
    compare_with_61(data=data)
    recalls = {}
    for population, parity in itertools.product(("base", "long"), ("even", "odd")):
        recalls[(population, parity)] = print_recall(
            data=data, population=population, parity=parity
        )
    for population in ("base", "long"):
        print_all_shapes(data=data, population=population, parity="even")
    base = pooled(data=data, population="base", parity="even")
    long = pooled(data=data, population="long", parity="even")
    c = {
        "C1": contrast(frames=long, arm="PE30_48", base=BENCHMARK),
        "C2": contrast(frames=long, arm="PE60_48", base=BENCHMARK),
        "C3": contrast(frames=base, arm="PE30_48", base=BENCHMARK),
        "C4": contrast(frames=base, arm="PE60_48", base=BENCHMARK),
        "C5": contrast(frames=base, arm="PE14_8m", base=BENCHMARK),
    }
    print("\nPlanned contrasts (99% interval, even months, S2-S4 pooled)")
    for name, value in c.items():
        print(f"  {name}: {fmt(*value)}")
    print("Descriptive contrasts")
    for label, frames, arm, ref in (
        ("long, PE60_48 - PE30_48", long, "PE60_48", "PE30_48"),
        ("long, PE30_48 - D0M_48", long, "PE30_48", "D0M_48"),
        ("base, PE14_48 - D0M_48", base, BENCHMARK, "D0M_48"),
        ("base, PE14_8 - PE14_48 (4 h at 6.0)", base, "PE14_8", BENCHMARK),
        ("base, D0M_8m - D0M_48", base, "D0M_8m", "D0M_48"),
        ("long, PE30_8m - PE14_8m", long, "PE30_8m", "PE14_8m"),
        ("long, PE60_8m - PE14_8m", long, "PE60_8m", "PE14_8m"),
        ("POST HOC long, PF14_48 - PE14_48", long, "PF14_48", BENCHMARK),
        ("POST HOC long, PF30_48 - PE14_48", long, "PF30_48", BENCHMARK),
        ("POST HOC long, PF60_48 - PE14_48", long, "PF60_48", BENCHMARK),
        ("POST HOC long, PF60_48 - PE30_48", long, "PF60_48", "PE30_48"),
        ("POST HOC base, PF14_48 - PE14_48", base, "PF14_48", BENCHMARK),
        ("POST HOC base, PF30_48 - PE14_48", base, "PF30_48", BENCHMARK),
        ("POST HOC base, PF60_48 - PE14_48", base, "PF60_48", BENCHMARK),
        ("base, PE30_8m - PE14_8m", base, "PE30_8m", "PE14_8m"),
        ("base, PE60_8m - PE14_8m", base, "PE60_8m", "PE14_8m"),
    ):
        print(f"  {label}: {fmt(*contrast(frames=frames, arm=arm, base=ref))}")
    print_mixture(long=long, base=base)
    print("POST HOC odd-month check of PF against PE14_48 (not used to choose)")
    for population in ("base", "long"):
        frames = pooled(data=data, population=population, parity="odd")
        for arm in ("PF14_48", "PF30_48", "PE30_48"):
            value = contrast(frames=frames, arm=arm, base=BENCHMARK)
            print(f"  {population} odd, {arm} - {BENCHMARK}: {fmt(*value)}")
    print_by_step(data=data, population="base")
    print_by_duration(data=data)
    print_real(out=out)
    print_alarms(out=out)
    verdict = decide(c=c, long_recall=recalls[("long", "even")])
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
