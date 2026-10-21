r"""Print every number of the experiment 148 write-up, and the verdicts the decision rule gives.

Reads the tables that `e00148_run.py` wrote. Settings are the end rules R0 (offline), R2 (running
peak), H14 (hindsight), and L0, L6, L24, L48 (lookahead in hours), each alone and with a merge of
overlapping alarms within 24 hours, `<rule>_M48`.

    uv run --no-sync python ml-experiments/00148-*/e00148_contrasts.py
"""

from __future__ import annotations

import json
from typing import Final

import numpy as np
import polars as pl
from e00148_common import WORK_DIR, stamp_months, stamp_year_months
from e00148_profile import NON_MATCHING

RUN_DIR: Final = WORK_DIR / "run"
N_RESAMPLES: Final[int] = 20_000
LEVEL: Final[float] = 1 - 0.05 / 4
"""The Bonferroni level for four planned contrasts: 98.75%."""
SEED: Final[int] = 148
LOOKAHEADS: Final[tuple[int, ...]] = (0, 6, 24, 48)
POST_HOC: Final[tuple[int, ...]] = (1, 2, 3, 4)
"""Post hoc lookaheads in hours, scored with 95% intervals that no correction adjusts."""
RULES: Final[tuple[str, ...]] = (
    "R0",
    "R2",
    "H14",
    *(f"L{h}" for h in sorted(LOOKAHEADS + POST_HOC)),
)
SETTINGS: Final[tuple[str, ...]] = tuple(s for r in RULES for s in (r, f"{r}_M48"))
MARGIN: Final[float] = -0.02
MAX_FALSE_END: Final[float] = 0.10
MAX_LATE_END: Final[float] = 0.10
BLANK_SERIES: Final[int] = 12
STAMPS_PER_HOUR: Final[int] = 2


def populations() -> pl.DataFrame:
    """A row per transfer, shape, and setting, with the calendar month and parity of the start."""
    events = pl.read_parquet(RUN_DIR / "events.parquet")
    onset = events["onset"].to_numpy()
    month = stamp_months()[onset]
    events = events.with_columns(
        year_month=pl.Series(stamp_year_months()[onset]),
        parity=pl.Series(np.where(month % 2 == 0, "even", "odd")),
    ).filter(pl.col("donor") != BLANK_SERIES)
    results = pl.read_parquet(RUN_DIR / "event_results.parquet").filter(pl.col("kind") == "event")
    weight = (
        results.filter(pl.col("shape") == "S1", pl.col("setting") == "R0")
        .select("event_id", weight=pl.col("moved_mw"))
        .unique("event_id")
    )
    return results.join(events, on="event_id", validate="m:1").join(
        weight, on="event_id", validate="m:1"
    )


def wide(*, data: pl.DataFrame, shape: str, population: str, parity: str) -> pl.DataFrame:
    """One row per transfer with every setting's `found` (columns named by setting)."""
    one = data.filter(
        pl.col("shape") == shape, pl.col("population") == population, pl.col("parity") == parity
    )
    frame = one.filter(pl.col("setting") == "R0").select(
        "event_id", "weight", "year_month", R0="found"
    )
    for setting in SETTINGS:
        if setting == "R0":
            continue
        frame = frame.join(
            one.filter(pl.col("setting") == setting).select("event_id", **{setting: "found"}),
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


def contrast(
    *, frames: list[pl.DataFrame], arm: str, base: str | None = None, level: float = LEVEL
) -> tuple[float, float, float]:
    """Point value and interval of the shape-averaged MW-weighted share found (difference)."""
    months = np.unique(frames[0]["year_month"].to_numpy())
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(months), size=(N_RESAMPLES, len(months)))
    point = 0.0
    boot = np.zeros(N_RESAMPLES)
    for frame in frames:
        wf_a, w = month_sums(frame=frame, found=arm)
        wf_b = month_sums(frame=frame, found=base)[0] if base else np.zeros_like(wf_a)
        point += (wf_a.sum() - wf_b.sum()) / w.sum()
        boot += (wf_a[picks].sum(axis=1) - wf_b[picks].sum(axis=1)) / w[picks].sum(axis=1)
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(boot / len(frames), [tail, 100 - tail])
    return point / len(frames), float(low), float(high)


def fmt(point: float, low: float, high: float) -> str:
    """A point value with its interval."""
    return f"{point:+.3f} ({low:+.3f} to {high:+.3f})"


def event_stats(*, data: pl.DataFrame, population: str, parity: str) -> pl.DataFrame:
    """Per setting, over the three shapes pooled: false-end, late-end, lag, and interval counts."""
    rows = data.filter(
        pl.col("shape").is_in(list(NON_MATCHING)),
        pl.col("population") == population,
        pl.col("parity") == parity,
    )
    onset = rows.filter(pl.col("onset_found"))
    found = rows.filter(pl.col("found"))
    return (
        rows.group_by("setting")
        .agg(
            n=pl.len(),
            share_2plus=(pl.col("n_intervals") >= 2).mean(),
            ratio_median=pl.col("detected_ratio").median(),
            end_error_median_h=pl.col("end_error").median() / STAMPS_PER_HOUR,
        )
        .join(
            onset.group_by("setting").agg(
                n_onset=pl.len(),
                false_end=pl.col("premature").mean(),
                late_end=pl.col("late").mean(),
            ),
            on="setting",
        )
        .join(
            found.group_by("setting").agg(
                lag_median_h=pl.col("lag").median() / STAMPS_PER_HOUR,
                lag_p90_h=pl.col("lag").quantile(0.9) / STAMPS_PER_HOUR,
            ),
            on="setting",
        )
        .sort("setting")
    )


def precision() -> pl.DataFrame:
    """Share of intervals in the injected batches that match an injected transfer."""
    batch = pl.read_parquet(RUN_DIR / "event_results.parquet").filter(pl.col("kind") == "batch")
    return (
        batch.group_by("setting")
        .agg(n=pl.col("n_intervals").sum(), matched=pl.col("n_matched").sum())
        .with_columns(precision=pl.col("matched") / pl.col("n"))
        .sort("setting")
    )


def print_partner_check(*, data: pl.DataFrame) -> None:
    """Whether the paired series' step reverses at the stated end, and the end-time error."""
    rows = data.filter(
        pl.col("shape").is_in(list(NON_MATCHING)),
        pl.col("population") == "base",
        pl.col("parity") == "even",
        pl.col("onset_found"),
        pl.col("setting").is_in(list(RULES)),
    )
    print("\nEnd-time accuracy on transfers whose onset is found (base, even, S2 to S4 pooled)")
    print("  setting: n, with a partner, partner reverses (0.5 to 2 times its onset step),")
    print("  median ratio, end error in hours (median, 10th and 90th percentile; late is positive)")
    for setting in RULES:
        part = rows.filter(pl.col("setting") == setting)
        paired = part.filter(pl.col("partner_reversal").is_not_nan())
        error = part["end_error"].drop_nans().to_numpy() / STAMPS_PER_HOUR
        print(
            f"  {setting}: {part.height}, {paired.height}, "
            f"{paired['partner_reversal'].to_numpy().mean():.3f}, "
            f"{np.median(paired['partner_ratio'].to_numpy()):.2f}, "
            f"{np.median(error):+.2f} "
            f"({np.percentile(error, 10):+.2f}, {np.percentile(error, 90):+.2f})"
        )
    for label, flag in (("within", True), ("outside", False)):
        part = rows.filter(pl.col("found") == flag, pl.col("setting") == "R2")
        values = part["partner_reversal"].drop_nans().to_numpy()
        share = values.mean()
        print(f"  R2, end {label} the tolerance: partner reverses {share:.3f} of {len(values)}")


def print_real() -> None:
    """The labelled real transfers of more than 1 day, per setting."""
    real = pl.read_parquet(RUN_DIR / "real_results.parquet").filter(pl.col("hours") > 24)
    print("\nLabelled real transfers over 1 day (descriptive): setting, found, onset found,")
    print("  false ends, 2 or more intervals, median lag in hours of found")
    for setting in SETTINGS:
        part = real.filter(pl.col("setting") == setting)
        found = part.filter(pl.col("found"))
        lag = (
            float(found["lag"].to_numpy().mean()) / STAMPS_PER_HOUR
            if found.height
            else float("nan")
        )
        print(
            f"  {setting:7s} found {int(part['found'].sum())} of {part.height}, onset "
            f"{int(part['onset_found'].sum())}, false end {int(part['premature'].sum())}, "
            f"2+ intervals {int((part['n_intervals'] >= 2).sum())}, mean lag {lag:.1f}, partner "
            f"reverses {int(part['partner_reversal'].drop_nans().sum())} of "
            f"{part['partner_reversal'].drop_nans().len()} with a partner"
        )


def main() -> None:
    """Print the tables and the verdicts."""
    data = populations()
    base = pooled(data=data, population="base", parity="even")
    print(
        f"transfers per shape (base, even): {base[0].height}; "
        f"MW weight {base[0]['weight'].sum():.1f}"
    )
    print("\nMW-weighted share found (95% interval), base, even months")
    shares = {}
    for setting in SETTINGS:
        value, low, high = contrast(frames=base, arm=setting, level=0.95)
        shares[setting] = (value, low, high)
        print(f"  {setting:7s} {value:.3f} ({low:.3f} to {high:.3f})")
    contrasts = {f"L{h}": contrast(frames=base, arm=f"L{h}", base="R0") for h in LOOKAHEADS}
    print("\nPlanned contrasts, minus R0 (98.75% interval)")
    for name, value in contrasts.items():
        print(f"  {name}: {fmt(*value)}")
    print("\nReference rows and post hoc lookaheads minus R0 (95% interval)")
    for name in ("R2", "H14", *(f"L{h}" for h in POST_HOC)):
        print(f"  {name}: {fmt(*contrast(frames=base, arm=name, base='R0', level=0.95))}")
    print("\nMerged (M48) minus the same rule alone (95% interval)")
    for name in RULES[1:]:
        print(f"  {name}: {fmt(*contrast(frames=base, arm=f'{name}_M48', base=name, level=0.95))}")
    pl.Config.set_tbl_rows(60).set_tbl_cols(12)
    for population, parity in (("base", "even"), ("long", "even")):
        print(f"\nEvent statistics, {population}, {parity} months, S2 to S4 pooled")
        print(event_stats(data=data, population=population, parity=parity))
    print("\nIntervals matching an injected transfer, all injected batches")
    print(precision())
    long = pooled(data=data, population="long", parity="even")
    print("\nLong transfers, share found minus R0")
    for setting in RULES[1:]:
        print(f"  {setting}: {fmt(*contrast(frames=long, arm=setting, base='R0'))}")
    print_partner_check(data=data)
    print_real()
    by = {
        r["setting"]: r for r in event_stats(data=data, population="base", parity="even").to_dicts()
    }
    summary = {
        "share": [{"rule": r, "share": v[0], "low": v[1], "high": v[2]} for r, v in shares.items()],
        "contrast": [
            {"rule": r, "diff": v[0], "low": v[1], "high": v[2]} for r, v in contrasts.items()
        ],
        "stats": [by[r] for r in RULES],
    }
    (RUN_DIR / "chart_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    adequate = []
    for h in LOOKAHEADS:
        name = f"L{h}"
        ok = (
            contrasts[name][1] > MARGIN
            and by[name]["false_end"] <= MAX_FALSE_END
            and by[name]["late_end"] <= MAX_LATE_END
        )
        print(f"LOOKAHEAD {h} h: {'adequate' if ok else 'not adequate'}")
        if ok:
            adequate.append(h)
    if adequate and min(adequate) <= 24:
        verdict = "supported"
    elif adequate:
        verdict = "partly supported"
    elif all(v[2] < MARGIN for v in contrasts.values()):
        verdict = "refuted"
    else:
        verdict = "inconclusive"
    print(f"RECOMMENDED lookahead: {min(adequate) if adequate else None}; VERDICT: {verdict}")


if __name__ == "__main__":
    main()
