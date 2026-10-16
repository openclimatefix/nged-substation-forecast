r"""Print every number the write-up of experiment 59 quotes.

Reads the tables `e00059_run.py` wrote: the matched thresholds, the real alarm counts, the outcome
of every injected transfer in each shape, and the labelled real transfers. Prints the best single
window (chosen on odd months), the two planned contrasts with their intervals, the verdict the
decision rule gives, recall by donor step, and why large transfers are missed.

    uv run python ml-experiments/00059-*/e00059_report.py
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00059_common import WORK_DIR, stamp_months, stamp_year_months
from e00059_inject import HEADLINE_BAND, NON_MATCHING_SHAPES, SHAPES
from e00059_windows import COMBINED_NAMES, DETECTOR_NAMES, SINGLE_NAMES

N_RESAMPLES: Final[int] = 20_000
LEVEL: Final[float] = 1 - 0.05 / 2
"""Two-sided level of the intervals: Bonferroni across the two contrasts C1 and C2 (97.5%)."""
SEED: Final[int] = 5900
SUPPORT_AT_LEAST: Final[float] = 0.02
ALARM_MARGIN: Final[int] = 10
STEP_EDGES: Final[tuple[float, ...]] = (0.0, 2.0, 4.0, 8.0, float("inf"))
"""Bins of donor step in 24-hour noise units."""
LARGE_STEP_Z: Final[float] = 4.0


def event_data(*, out: Path, variant: str = "plain") -> pl.DataFrame:
    """Return every (transfer, shape, detector) row of one interval variant, with month, weight."""
    events = pl.read_parquet(out / "events.parquet")
    onset = events["onset"].to_numpy()
    events = events.with_columns(
        month=pl.Series(stamp_months()[onset]),
        year_month=pl.Series(stamp_year_months()[onset]),
        headline=(pl.col("size_band") == HEADLINE_BAND[0])
        & (pl.col("duration_band") == HEADLINE_BAND[1]),
    ).with_columns(even=pl.col("month") % 2 == 0)
    return (
        pl.read_parquet(out / "event_results.parquet")
        .filter(pl.col("variant") == variant)
        .join(events.drop("onset", "end"), on="event_id", validate="m:1")
        .with_columns(weight=pl.col("donor_step_mw").abs())
    )


def headline(*, data: pl.DataFrame, even: bool, shapes: tuple[str, ...]) -> pl.DataFrame:
    """Headline-band transfers outside the holiday set, in odd or even months, in given shapes."""
    return data.filter(
        pl.col("headline"),
        pl.col("set") != "holiday",
        pl.col("even") == even,
        pl.col("shape").is_in(shapes),
    )


def share(*, rows: pl.DataFrame, weighted: bool, column: str = "found") -> float:
    """The (MW-weighted) share of rows whose `column` is true."""
    found = rows[column].cast(pl.Float64).to_numpy()
    weight = rows["weight"].to_numpy() if weighted else np.ones(rows.height)
    return float((weight * found).sum() / weight.sum())


def choose_best_single(*, data: pl.DataFrame) -> str:
    """The single window with the highest MW-weighted share found on odd-month transfers."""
    odd = headline(data=data, even=False, shapes=NON_MATCHING_SHAPES)
    scores: dict[str, float] = {
        name: share(rows=odd.filter(pl.col("detector") == name), weighted=True)
        for name in SINGLE_NAMES.values()
    }
    for name, value in scores.items():
        print(f"odd-month MW-weighted share found, {name}: {value:.4f}")
    return max(scores, key=scores.__getitem__)


def paired_difference(
    *, new: pl.DataFrame, reference: pl.DataFrame, weighted: bool = True
) -> tuple[float, float, float]:
    """`new - reference` in the share found, with a month-resampled interval at `LEVEL`.

    Rows pair on the transfer and its shape. A resample draws whole calendar months, so both
    shapes of a transfer are drawn together, and the weighted share is a ratio of sums.
    """
    joined = new.join(reference, on=["event_id", "shape"], suffix="_ref").sort("event_id", "shape")
    weight = joined["weight"].to_numpy() if weighted else np.ones(joined.height)
    found = joined["found"].cast(pl.Float64).to_numpy()
    found_ref = joined["found_ref"].cast(pl.Float64).to_numpy()
    months = joined["year_month"].to_numpy()
    labels = np.unique(months)
    top = np.array([(weight * (found - found_ref))[months == m].sum() for m in labels])
    bottom = np.array([weight[months == m].sum() for m in labels])
    point = float(top.sum() / bottom.sum())
    picks = np.random.default_rng(SEED).integers(0, len(labels), size=(N_RESAMPLES, len(labels)))
    resampled = top[picks].sum(axis=1) / bottom[picks].sum(axis=1)
    tail = 100 * (1 - LEVEL) / 2
    low, high = np.percentile(resampled, [tail, 100 - tail])
    return point, float(low), float(high)


def share_interval(*, rows: pl.DataFrame) -> tuple[float, float, float]:
    """The MW-weighted share found with a 95% interval from resampling whole calendar months."""
    weight = rows["weight"].to_numpy()
    found = rows["found"].cast(pl.Float64).to_numpy()
    months = rows["year_month"].to_numpy()
    labels = np.unique(months)
    top = np.array([(weight * found)[months == m].sum() for m in labels])
    bottom = np.array([weight[months == m].sum() for m in labels])
    picks = np.random.default_rng(SEED).integers(0, len(labels), size=(N_RESAMPLES, len(labels)))
    low, high = np.percentile(top[picks].sum(axis=1) / bottom[picks].sum(axis=1), [2.5, 97.5])
    return float(top.sum() / bottom.sum()), float(low), float(high)


def print_shares(*, data: pl.DataFrame) -> None:
    """Print each detector's shares found by shape, even months, and the MW share missed."""
    print("\nEven-month headline-band transfers (original and extra sets)")
    for shape in (*SHAPES, "pooled"):
        shapes = NON_MATCHING_SHAPES if shape == "pooled" else (shape,)
        rows = headline(data=data, even=True, shapes=shapes)
        print(f"  shape {shape}: {rows.filter(pl.col('detector') == 'W8').height} rows")
        for name in DETECTOR_NAMES:
            part = rows.filter(pl.col("detector") == name)
            weighted = share(rows=part, weighted=True)
            print(
                f"    {name}: weighted {weighted:.4f} (MW missed {1 - weighted:.4f}), "
                f"unweighted {share(rows=part, weighted=False):.4f}, "
                f"onset found (weighted) "
                f"{share(rows=part, weighted=True, column='onset_found'):.4f}"
            )


def print_intervals(*, data: pl.DataFrame) -> None:
    """Print each detector's MW-weighted share found with a 95% interval, pooled shapes."""
    rows = headline(data=data, even=True, shapes=NON_MATCHING_SHAPES)
    print("\nMW-weighted share found, pooled non-matching shapes, 95% month-resampled interval")
    for name in DETECTOR_NAMES:
        point, low, high = share_interval(rows=rows.filter(pl.col("detector") == name))
        print(f"  {name}: {point:.4f} ({low:.4f} to {high:.4f})")


def print_contrasts(*, data: pl.DataFrame, best: str, alarms: pl.DataFrame) -> None:
    """Print C1 and C2, the even-month alarm counts, and the verdict the decision rule gives."""
    rows = headline(data=data, even=True, shapes=NON_MATCHING_SHAPES)
    reference = rows.filter(pl.col("detector") == best)
    verdicts = {}
    print(f"\nBest single window (odd months): {best}")
    for label, name in (("C1", "MAX"), ("C2", "SUM")):
        new = rows.filter(pl.col("detector") == name)
        point, low, high = paired_difference(new=new, reference=reference)
        unweighted = paired_difference(new=new, reference=reference, weighted=False)
        count = alarm_counts(alarms=alarms)
        alarm_ok = count[name]["even"] <= count[best]["even"] + ALARM_MARGIN
        if low > 0 and point >= SUPPORT_AT_LEAST and alarm_ok:
            verdicts[label] = "supported"
        elif high < SUPPORT_AT_LEAST:
            verdicts[label] = "refuted"
        else:
            verdicts[label] = "inconclusive"
        print(
            f"{label}: {name} - {best} weighted {point:+.4f} ({low:+.4f} to {high:+.4f}); "
            f"unweighted {unweighted[0]:+.4f} ({unweighted[1]:+.4f} to {unweighted[2]:+.4f}); "
            f"alarm count ok {alarm_ok}; {verdicts[label]}"
        )
    if "supported" in verdicts.values():
        verdict = "supported"
    elif all(v == "refuted" for v in verdicts.values()):
        verdict = "refuted"
    else:
        verdict = "inconclusive"
    print(f"Verdict: {verdict}")
    # Continuity with experiment 58: the donor-profile shape, and the holiday set.
    donor = headline(data=data, even=True, shapes=("donor_profile",))
    for label, name in (("C1", "MAX"), ("C2", "SUM")):
        point, low, high = paired_difference(
            new=donor.filter(pl.col("detector") == name),
            reference=donor.filter(pl.col("detector") == best),
        )
        print(f"  donor-profile shape, {label}: {point:+.4f} ({low:+.4f} to {high:+.4f})")
    holiday = data.filter(
        pl.col("headline"),
        pl.col("set") == "holiday",
        pl.col("even"),
        pl.col("shape").is_in(NON_MATCHING_SHAPES),
    )
    for label, name in (("C1", "MAX"), ("C2", "SUM")):
        point, low, high = paired_difference(
            new=holiday.filter(pl.col("detector") == name),
            reference=holiday.filter(pl.col("detector") == best),
        )
        print(f"  holiday set, {label}: {point:+.4f} ({low:+.4f} to {high:+.4f})")


def alarm_counts(*, alarms: pl.DataFrame) -> dict[str, dict[str, int]]:
    """Unexplained-alarm counts by detector, for odd and even months and for all months."""
    out = {}
    for name in DETECTOR_NAMES:
        part = alarms.filter(pl.col("detector") == name)
        unexplained = part.filter(~pl.col("explained"))
        out[name] = {
            "all": part.height,
            "odd": unexplained.filter(pl.col("month") % 2 == 1).height,
            "even": unexplained.filter(pl.col("month") % 2 == 0).height,
        }
    return out


def print_by_step(*, data: pl.DataFrame, best: str) -> None:
    """Print the MW-weighted and unweighted share found by donor step and metered receiver."""
    rows = headline(data=data, even=True, shapes=NON_MATCHING_SHAPES)
    print("\nShare found by donor step (24-hour noise units), pooled non-matching shapes")
    for name in (best, *COMBINED_NAMES):
        for metered in (True, False):
            part = rows.filter(
                pl.col("detector") == name, (pl.col("metered_receivers") > 0) == metered
            )
            cells = []
            for low, high in itertools.pairwise(STEP_EDGES):
                bin_rows = part.filter(pl.col("donor_step_z").is_between(low, high, closed="left"))
                if bin_rows.is_empty():
                    cells.append(f"{low:g}-{high:g}: -")
                    continue
                cells.append(
                    f"{low:g}-{high:g}: {bin_rows['found'].sum()}/{bin_rows.height} "
                    f"(w {share(rows=bin_rows, weighted=True):.2f})"
                )
            print(f"  {name} {'metered' if metered else 'unmetered'}: " + ", ".join(cells))


def print_misses(*, data: pl.DataFrame, best: str) -> None:
    """Print why transfers with a metered receiver and a donor step of 4.0 or more were missed."""
    print("\nWhy large transfers with a metered receiver are not found (even months, headline)")
    for shape in (*SHAPES, "pooled"):
        shapes = NON_MATCHING_SHAPES if shape == "pooled" else (shape,)
        rows = headline(data=data, even=True, shapes=shapes).filter(
            pl.col("metered_receivers") > 0, pl.col("donor_step_z") >= LARGE_STEP_Z
        )
        for name in (best, *COMBINED_NAMES):
            part = rows.filter(pl.col("detector") == name)
            counts = dict(part.group_by("reason").len().sort("reason").iter_rows())
            weighted_found = share(rows=part, weighted=True) if part.height else float("nan")
            no_alarm = part.filter(pl.col("reason") == "no_alarm")["score_share"]
            median = f"{no_alarm.median():.2f}" if no_alarm.len() else "-"
            print(
                f"  {shape} {name}: n {part.height}, {counts}, "
                f"weighted share found {weighted_found:.3f}, "
                f"no-alarm median score/threshold {median}"
            )


def print_post_hoc(*, out: Path, best: str) -> None:
    """Print the post hoc arm whose interval builder tracks the end more patiently."""
    data = event_data(out=out, variant="fixed")
    rows = headline(data=data, even=True, shapes=NON_MATCHING_SHAPES)
    print("\nPost hoc: fixed interval variant, even months, pooled non-matching shapes")
    for name in DETECTOR_NAMES:
        part = rows.filter(pl.col("detector") == name)
        print(
            f"  {name}: weighted {share(rows=part, weighted=True):.4f}, "
            f"unweighted {share(rows=part, weighted=False):.4f}, "
            f"onset found (weighted) {share(rows=part, weighted=True, column='onset_found'):.4f}"
        )
    reference = rows.filter(pl.col("detector") == best)
    for label, name in (("C1", "MAX"), ("C2", "SUM")):
        point, low, high = paired_difference(
            new=rows.filter(pl.col("detector") == name), reference=reference
        )
        print(f"  {label} {name} - {best}: {point:+.4f} ({low:+.4f} to {high:+.4f})")
    large = rows.filter(pl.col("metered_receivers") > 0, pl.col("donor_step_z") >= LARGE_STEP_Z)
    for name in (best, *COMBINED_NAMES):
        part = large.filter(pl.col("detector") == name)
        counts = dict(part.group_by("reason").len().sort("reason").iter_rows())
        print(f"  large metered {name}: n {part.height}, {counts}")


def print_end_errors(*, data: pl.DataFrame, best: str) -> None:
    """Print the end error of large transfers whose onset was found and whose end was missed."""
    rows = headline(data=data, even=True, shapes=NON_MATCHING_SHAPES).filter(
        pl.col("metered_receivers") > 0,
        pl.col("donor_step_z") >= LARGE_STEP_Z,
        pl.col("reason") == "end_outside_tolerance",
    )
    print("\nEnd error of large metered transfers with a found onset and a missed end")
    for name in (best, *COMBINED_NAMES):
        part = rows.filter(pl.col("detector") == name).with_columns(
            error_hours=pl.col("end_error") / 2,
            error_share=pl.col("end_error") / 2 / pl.col("hours"),
        )
        quartiles = part["error_hours"].quantile(0.25), part["error_hours"].median()
        print(
            f"  {name}: n {part.height}, end error hours 25%/50%/75% "
            f"{quartiles[0]:.1f}/{quartiles[1]:.1f}/{part['error_hours'].quantile(0.75):.1f}, "
            f"median share of duration {part['error_share'].median():.2f}, "
            f"median duration {part['hours'].median():.1f} h"
        )


def print_real(*, out: Path) -> None:
    """Print the labelled real transfers over 1 day found by each detector."""
    real = pl.read_parquet(out / "real_results.parquet").filter(pl.col("hours") > 24)
    print("\nLabelled real transfers over 1 day")
    for name in DETECTOR_NAMES:
        part = real.filter(pl.col("detector") == name)
        print(f"  {name}: found {part['found'].sum()} of {part.height}")


def main() -> None:
    """Print every number the write-up quotes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    out = parser.parse_args().out
    print("thresholds:", json.loads((out / "thresholds.json").read_text()))
    alarms = pl.read_parquet(out / "real_alarms.parquet")
    for name, counts in alarm_counts(alarms=alarms).items():
        print(f"  alarms {name}: {counts}")
    data = event_data(out=out)
    best = choose_best_single(data=data)
    print_shares(data=data)
    print_intervals(data=data)
    print_contrasts(data=data, best=best, alarms=alarms)
    print_by_step(data=data, best=best)
    print_misses(data=data, best=best)
    print_end_errors(data=data, best=best)
    print_real(out=out)
    print_post_hoc(out=out, best=best)


if __name__ == "__main__":
    main()
