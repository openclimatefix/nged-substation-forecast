"""Print every number the write-up of experiment 71 quotes, and the verdict the decision rule gives.

Reads the tables that `e00071_run.py` wrote: MW-weighted recall of the five arms (PF14, X, G, XG,
and the control L) on the fresh headline-band, long, and zone-oversampled draws, by injection shape,
with month-resampled intervals; the planned contrasts C1 to C3 and the zone split of C1; recall by
donor step; the labelled real transfers; the alarm counts; and the splice diagnostics.

    uv run python ml-experiments/00071-*/e00071_report.py
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00071_common import WORK_DIR, stamp_months, stamp_year_months
from e00071_profile import NON_MATCHING, SHAPE_NAMES
from e00071_run import ARMS
from e00071_variants import refit_zone, second_refit_stamps

N_RESAMPLES: Final[int] = 20_000
LEVEL: Final[float] = 1 - 0.05 / 3
"""Two-sided level of the intervals: Bonferroni across the three planned contrasts (98.33%)."""
SEED: Final[int] = 7100
THRESHOLD: Final[float] = 0.01
STEP_EDGES: Final[tuple[float, ...]] = (0.0, 2.0, 4.0, 6.0, 10.0, float("inf"))
EXPECTED_ROWS: Final[dict[str, int]] = {"base": 1160, "long": 400, "zone": 300}
BENCHMARK: Final[str] = "PF14_48"


def populations(*, out: Path) -> pl.DataFrame:
    """A row per transfer, shape, and arm for the `base`, `long`, and `zone` draws.

    The weight of a transfer is its mean moved MW under S1. `parity` is `even` or `odd` by the UTC
    month of the onset, and `in_zone` says whether the onset or the end lies within 24 hours of a
    refit of the first model.
    """
    events = pl.read_parquet(out / "events.parquet")
    for population, n in EXPECTED_ROWS.items():
        if events.filter(pl.col("population") == population).height != n:
            raise ValueError(f"the {population} draw does not hold {n} transfers")
    if events["event_id"].n_unique() != events.height:
        raise ValueError("event ids repeat")
    onset, end = events["onset"].to_numpy(), events["end"].to_numpy()
    zone = refit_zone()
    month = stamp_months()[onset]
    events = events.with_columns(
        year_month=pl.Series(stamp_year_months()[onset]),
        parity=pl.Series(np.where(month % 2 == 0, "even", "odd")),
        in_zone=pl.Series(zone[onset] | zone[np.minimum(end, len(zone) - 1)]),
    )
    results = pl.read_parquet(out / "event_results.parquet")
    weight = (
        results.filter(pl.col("shape") == "S1", pl.col("setting") == BENCHMARK)
        .select("event_id", weight=pl.col("moved_mw"))
        .unique("event_id")
    )
    return (
        results.join(events, on="event_id", validate="m:1")
        .join(weight, on="event_id", validate="m:1")
        .sort("shape", "setting", "event_id")
    )


def wide(
    *, data: pl.DataFrame, shape: str, population: str, parity: str, zone: bool | None = None
) -> pl.DataFrame:
    """One row per transfer with every arm's `found` (columns named by setting) for one shape.

    `onset_<arm>` holds the onset flag, and `scaled_<arm>` the found flag under the scaled onset
    tolerance. `zone` keeps only transfers in (True) or out of (False) the refit zones.
    """
    one = data.filter(
        pl.col("shape") == shape, pl.col("population") == population, pl.col("parity") == parity
    )
    if zone is not None:
        one = one.filter(pl.col("in_zone") == zone)
    frame = one.filter(pl.col("setting") == BENCHMARK).select(
        "event_id",
        "weight",
        "year_month",
        "metered_receivers",
        "donor_step_z",
        "hours",
        "in_zone",
        **{
            BENCHMARK: "found",
            f"onset_{BENCHMARK}": "onset_found",
            f"scaled_{BENCHMARK}": "found_scaled",
        },
    )
    for setting in ARMS:
        if setting == BENCHMARK:
            continue
        part = one.filter(pl.col("setting") == setting)
        frame = frame.join(
            part.select(
                "event_id",
                **{
                    setting: "found",
                    f"onset_{setting}": "onset_found",
                    f"scaled_{setting}": "found_scaled",
                },
            ),
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


def pooled(
    *, data: pl.DataFrame, population: str, parity: str, zone: bool | None = None
) -> list[pl.DataFrame]:
    """The three non-matching shapes' wide frames."""
    return [
        wide(data=data, shape=s, population=population, parity=parity, zone=zone)
        for s in NON_MATCHING
    ]


def print_recall(*, data: pl.DataFrame, population: str, parity: str) -> None:
    """Absolute MW-weighted recall per arm and shape, with the pooled interval and the missed MW."""
    frames = pooled(data=data, population=population, parity=parity)
    print(f"\n== {population} transfers, {parity} months: MW-weighted recall (S2, S3, S4 pooled)")
    if frames[0]["weight"].null_count():
        raise ValueError("a transfer has no weight")
    print(f"transfers per shape: {frames[0].height}; MW weight {frames[0]['weight'].sum():.1f}")
    for setting in ARMS:
        value, low, high = contrast(frames=frames, arm=setting)
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
        scaled = contrast(frames=frames, arm=f"scaled_{setting}")[0]
        print(
            f"{setting:8s} {value:.3f} ({low:.3f} to {high:.3f}) [{by_shape}] "
            f"missed {1 - value:.3f} = no start {no_start:.3f} + wrong end {wrong_end:.3f}; "
            f"unweighted {unweighted:.3f}; scaled onset tolerance {scaled:.3f}"
        )


def print_by_step(*, data: pl.DataFrame, population: str) -> None:
    """Standard table: recall by donor step in noise units, split by metered receiver (S2-S4)."""
    print(f"\nRecall by donor step ({population}, even months): found/transfers, S2-S4 summed")
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


def print_zone_split(*, data: pl.DataFrame) -> None:
    """C1 read separately for transfers in and out of the refit zones, with the counts."""
    print("\nZone split of C1 (X minus PF14, even months, S2-S4 pooled, 98.33% interval)")
    for population in ("base", "long"):
        for zone, label in ((True, "in a zone"), (False, "outside the zones")):
            frames = pooled(data=data, population=population, parity="even", zone=zone)
            if not frames[0].height:
                print(f"  {population}, {label}: no transfers")
                continue
            low_months = len(np.unique(frames[0]["year_month"].to_numpy()))
            print(
                f"  {population}, {label}: {frames[0].height} transfers per shape, "
                f"{low_months} months, PF14 {contrast(frames=frames, arm=BENCHMARK)[0]:.3f}, "
                f"X minus PF14 {fmt(*contrast(frames=frames, arm='X_48', base=BENCHMARK))}"
            )
    zone_all = pooled(data=data, population="zone", parity="even")
    odd_all = pooled(data=data, population="zone", parity="odd")
    print("Zone-oversampled draw (300 transfers with onsets in a zone)")
    for label, frames in (("even months", zone_all), ("odd months", odd_all)):
        print(f"  {label}: {frames[0].height} transfers per shape")
        for arm in ARMS:
            print(f"    {arm}: {contrast(frames=frames, arm=arm)[0]:.3f}")
        for arm in ("X_48", "XG_48", "L_48"):
            print(f"    {arm} minus PF14: {fmt(*contrast(frames=frames, arm=arm, base=BENCHMARK))}")


def print_real(*, out: Path) -> None:
    """The labelled real transfers of more than 1 day, found by each arm."""
    real = pl.read_parquet(out / "real_results.parquet").filter(pl.col("hours") > 24)
    n = real.filter(pl.col("setting") == BENCHMARK).height
    print(f"\nLabelled real transfers of more than 1 day: {n}")
    for setting in ARMS:
        part = real.filter(pl.col("setting") == setting)
        w = part["moved_mw_p50"].fill_null(0).to_numpy()
        f = part["found"].to_numpy().astype(float)
        print(
            f"  {setting:8s} found {int(f.sum())}, onset found {part['onset_found'].sum()}, "
            f"MW-weighted recall {(w * f).sum() / w.sum():.3f}"
        )
    classes = pl.read_parquet(out / "classes.parquet")
    print(f"\nClasses of the {classes.height} transfers with no PF14 alarm at the start")
    by_arm = {
        arm: real.filter(pl.col("setting") == arm).select(
            "card_id", **{f"onset_{arm}": "onset_found", f"found_{arm}": "found"}
        )
        for arm in ARMS
    }
    table = classes.select("card_id", "class", "peak_near_z")
    for frame in by_arm.values():
        table = table.join(frame, on="card_id", validate="1:1")
    for label in sorted(set(table["class"])):
        part = table.filter(pl.col("class") == label)
        print(
            f"  {label}: {part.height}; "
            + "; ".join(
                f"{a} onset {int(part[f'onset_{a}'].sum())} found {int(part[f'found_{a}'].sum())}"
                for a in ARMS
            )
        )
    print("Each transfer of the refit and timing classes:")
    for row in table.filter(
        pl.col("class").is_in(["monthly refit", "alarm outside the tolerance"])
    ).iter_rows(named=True):
        flags = " ".join(f"{a} {row[f'onset_{a}']}/{row[f'found_{a}']}" for a in ARMS)
        print(f"  {row['card_id']} {row['class']}: peak z {row['peak_near_z']:.2f}; {flags}")


def print_alarms(*, out: Path) -> None:
    """Thresholds and unexplained alarm counts by even and odd month, per arm."""
    alarms = pl.read_parquet(out / "alarms.parquet")
    print(f"\nthresholds {json.loads((out / 'thresholds.json').read_text())}")
    for arm in ARMS:
        part = alarms.filter(pl.col("setting") == arm, ~pl.col("logged_matched"))
        even = part.filter(pl.col("month") % 2 == 0).height
        odd = part.filter(pl.col("month") % 2 == 1).height
        print(f"{arm}: unexplained alarms even months {even}, odd {odd}")
        if odd > 80:
            print(f"WARNING: {arm} has {odd} odd-month unexplained alarms, above the target 80")


def print_diagnostics(*, out: Path, data: pl.DataFrame) -> None:
    """The agreement of the two models, the refit profile, and the gaps of the panel."""
    splice = pl.read_parquet(out / "splice_agreement.parquet")
    print("\nStatistic of the 1st-refit model minus the 16th-refit model, in noise units, where")
    print("both are clean (no refit of either within 24 hours):")
    print(
        f"  median |difference| over series {splice['median_abs_difference_z'].median():.3f}; "
        f"largest series median {splice['median_abs_difference_z'].max():.3f}; "
        f"99th percentile, median over series {splice['p99_abs_difference_z'].median():.3f}"
    )
    profile = pl.read_parquet(out / "refit_profile.parquet")
    summary = (
        profile.group_by("refit", "offset_half_hours")
        .agg(pl.col("mean_abs_difference_share").median())
        .sort("refit", "offset_half_hours")
    )
    print(
        "Median over series of mean |expected(1st) - expected(16th)| as a share of median |power|,"
    )
    print("by half-hours from the refit:")
    for refit in ("1st", "16th"):
        part = summary.filter(pl.col("refit") == refit)
        print(
            f"  {refit}: "
            + " ".join(
                f"{o}:{v:.3f}"
                for o, v in part.select(
                    "offset_half_hours", "mean_abs_difference_share"
                ).iter_rows()
            )
        )
    gaps = pl.read_parquet(out / "gaps.parquet")
    print(
        f"\nInner gaps of the residual panel: {gaps.height}; 13 half-hours or longer "
        f"{gaps.filter(pl.col('length') >= 13).height}; a day or longer "
        f"{gaps.filter(pl.col('length') >= 48).height}"
    )
    events = pl.read_parquet(out / "events.parquet").filter(pl.col("population") == "base")
    hit = 0
    for row in events.iter_rows(named=True):
        series = {row["donor"], *(int(r) for r in row["receivers"].split(",") if r)}
        inside = gaps.filter(
            pl.col("series").is_in(list(series)),
            pl.col("length") >= 13,
            pl.col("start") <= row["onset"],
            pl.col("start") + pl.col("length") > row["onset"],
        )
        hit += inside.height > 0
    print(f"Base-draw transfers whose onset lies inside a gap of 13 half-hours or longer: {hit}")
    print_splice_and_gap_effects(out=out, data=data)


def print_splice_and_gap_effects(*, out: Path, data: pl.DataFrame) -> None:
    """Alarms near the splice stamps, and the transfers whose outcome the gap bridging changes."""
    zone = refit_zone()
    edges = np.flatnonzero(np.diff(zone.astype(int)) != 0) + 1
    edges = np.concatenate([edges, second_refit_stamps()])
    alarms = pl.read_parquet(out / "alarms.parquet")
    print("\nReal alarms within 48 half-hours of a splice stamp (zone edges and 16th refits):")
    for arm in ARMS:
        stamps = alarms.filter(pl.col("setting") == arm)["stamp"].to_numpy()
        near = sum(bool(np.abs(edges - s).min() <= 48) for s in stamps)
        print(f"  {arm}: {near} of {len(stamps)}")
    for arm in ("G_48", "XG_48"):
        reference = "PF14_48" if arm == "G_48" else "X_48"
        mine = data.filter(pl.col("setting") == arm, pl.col("shape") == "S1")
        theirs = data.filter(pl.col("setting") == reference, pl.col("shape") == "S1")
        joined = mine.join(theirs, on="event_id", suffix="_ref", validate="1:1")
        changed = joined.filter(
            (pl.col("found") != pl.col("found_ref"))
            | (pl.col("onset_found") != pl.col("onset_found_ref"))
        )
        print(
            f"  {arm} differs from {reference} in the outcome of {changed.height} of "
            f"{joined.height} transfers (S1, all draws)"
        )


def decide(*, c1: tuple[float, float, float]) -> str:
    """Apply the pre-registered verdict rule to C1 (point, lower end, upper end)."""
    if c1[1] > 0 and c1[0] >= THRESHOLD:
        return "supported"
    if c1[1] > 0:
        return "small gain"
    if c1[2] < THRESHOLD:
        return "refuted"
    return "inconclusive"


def main() -> None:
    """Print the tables and the verdict."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    out = parser.parse_args().out
    data = populations(out=out)
    for population, parity in itertools.product(("base", "long"), ("even", "odd")):
        print_recall(data=data, population=population, parity=parity)
    base = pooled(data=data, population="base", parity="even")
    long = pooled(data=data, population="long", parity="even")
    c = {
        "C1 X minus PF14": contrast(frames=base, arm="X_48", base=BENCHMARK),
        "C2 G minus PF14": contrast(frames=base, arm="G_48", base=BENCHMARK),
        "C3 XG minus PF14": contrast(frames=base, arm="XG_48", base=BENCHMARK),
    }
    print("\nPlanned contrasts (98.33% interval, headline band, even months, S2-S4 pooled)")
    for name, value in c.items():
        print(f"  {name}: {fmt(*value)}")
    print("Descriptive contrasts")
    for label, frames, arm in (
        ("long, X minus PF14", long, "X_48"),
        ("long, G minus PF14", long, "G_48"),
        ("long, XG minus PF14", long, "XG_48"),
        ("base, L (suppression lifted, no splice) minus PF14", base, "L_48"),
        ("long, L minus PF14", long, "L_48"),
        ("base, X minus L (what the splice adds)", base, "X_48"),
    ):
        ref = "L_48" if "minus L" in label else BENCHMARK
        print(f"  {label}: {fmt(*contrast(frames=frames, arm=arm, base=ref))}")
    for shape in ("S1", "S5"):
        frame = wide(data=data, shape=shape, population="base", parity="even")
        print(
            f"base even {shape} {SHAPE_NAMES[shape]}: "
            + " ".join(f"{s} {contrast(frames=[frame], arm=s)[0]:.3f}" for s in ARMS)
        )
    print("Odd-month check (not used for any decision)")
    odd = pooled(data=data, population="base", parity="odd")
    print(f"  base odd, X minus PF14: {fmt(*contrast(frames=odd, arm='X_48', base=BENCHMARK))}")
    print_zone_split(data=data)
    print_by_step(data=data, population="base")
    print_real(out=out)
    print_alarms(out=out)
    print_diagnostics(out=out, data=data)
    print(f"\nVERDICT: {decide(c1=c['C1 X minus PF14'])}")


if __name__ == "__main__":
    main()
