"""Print every number the write-up of experiment 47 quotes.

Reads the per-arm tables that `e00047_run.py` and `e00047_diagnostics.py` wrote, and prints the
matched thresholds, the contrasts C1 to C4 with their intervals, the verdict the decision rule
gives, and the diagnostics.

    uv run python ml-experiments/00047-*/e00047_report.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00047_common import EXPERIMENT_43_DIR, WORK_DIR, stamp_months, stamp_year_months
from e00047_detect import SAMPLE_NAMES, SETTINGS, holiday_distance_by_stamp
from e00047_inject import DURATION_BANDS, HEADLINE_BAND, SIZE_BANDS
from e00047_score import event_f1, month_bootstrap

N_RESAMPLES: Final[int] = 20_000
LEVEL: Final[float] = 0.9875
"""Two-sided level of the intervals: Bonferroni across the four contrasts C1 to C4."""
SEED: Final[int] = 4700
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
REFUTED_BELOW: Final[float] = 0.03
LABELS: Final[dict[str, str]] = {
    "D0_8": "D0, 4 h window",
    "D1_8": "D1, 4 h window",
    "D0_48": "D0, 24 h window",
    "D1_48": "D1, 24 h window",
}
CONTRASTS: Final[dict[str, str]] = {"C1": "D0_48", "C2": "D0_8", "C3": "D1_48", "C4": "D1_8"}
"""Contrast to detector setting: C1 is D0 at 24 hours, C2 D0 at 4, C3 D1 at 24, C4 D1 at 4."""


def load_thresholds(*, out: Path, name: str) -> dict[str, float]:
    """Return an arm-seed's matched thresholds."""
    return json.loads((out / f"thresholds_{name}.json").read_text())


def event_table(*, out: Path) -> pl.DataFrame:
    """Return the injected events with their onset month, parity, and year-and-month label."""
    events = pl.read_parquet(out / "events.parquet")
    onset = events["onset"].to_numpy()
    return events.with_columns(
        month=pl.Series(stamp_months()[onset]),
        year_month=pl.Series(stamp_year_months()[onset]),
        headline=(pl.col("size_band") == HEADLINE_BAND[0])
        & (pl.col("duration_band") == HEADLINE_BAND[1]),
    ).with_columns(even=pl.col("month") % 2 == 0)


def outcomes(*, out: Path, name: str, events: pl.DataFrame, variant: str = "plain") -> pl.DataFrame:
    """Return an arm-seed's per-transfer outcomes joined to the event table."""
    results = pl.read_parquet(out / f"event_results_{name}.parquet").filter(
        pl.col("variant") == variant
    )
    return results.join(events, on="event_id", validate="m:1").sort("setting", "event_id")


def precision_bounds(*, out: Path, name: str) -> dict[str, tuple[float, float]]:
    """Return per setting the lower and upper precision from even-month alarms.

    The lower bound counts the alarms within 24 hours of a logged closure. The upper bound adds the
    unmatched alarms times the share of the SW2-0b sample's alarms that were visible transfers.
    """
    alarms = pl.read_parquet(out / f"alarms_matched_{name}.parquet").filter(
        pl.col("month") % 2 == 0
    )
    sample = pl.read_parquet(EXPERIMENT_43_DIR / "alarm_sample.parquet")
    result = {}
    for setting in SETTINGS:
        row = sample.filter(pl.col("setting") == SAMPLE_NAMES[setting])
        rate = row["visible"].item() / row["sampled"].item()
        part = alarms.filter(pl.col("setting") == setting)
        n, matched = part.height, int(part["logged_matched"].sum())
        if n == 0:
            result[setting] = (float("nan"), float("nan"))
        else:
            result[setting] = (matched / n, (matched + (n - matched) * rate) / n)
    return result


def seed_mean_found(
    *, out: Path, arm: str, events: pl.DataFrame, setting: str, seeds: tuple[int, ...] = SEEDS
) -> pl.DataFrame:
    """Return per transfer the share of seeds in which the arm found it, for one setting."""
    frames = [
        outcomes(out=out, name=f"{arm}_s{seed}", events=events)
        .filter(pl.col("setting") == setting)
        .select("event_id", found=pl.col("found").cast(pl.Float64), seed=pl.lit(seed))
        for seed in seeds
    ]
    return pl.concat(frames).group_by("event_id").agg(found=pl.col("found").mean()).sort("event_id")


def paired_interval(
    *, population: pl.DataFrame, new: pl.DataFrame, reference: pl.DataFrame, seed: int
) -> tuple[float, float, float]:
    """Return the mean paired difference `new - reference` and its Bonferroni interval.

    `population` holds `event_id` and `year_month`; whole calendar months are resampled.
    """
    joined = (
        population.join(new, on="event_id")
        .join(reference, on="event_id", suffix="_ref")
        .sort("event_id")
    )
    if joined.height != population.height:
        raise SystemExit("STOP: a transfer is missing from a result file")
    diff = (joined["found"] - joined["found_ref"]).to_numpy()
    months = joined["year_month"].to_numpy()
    resampled = month_bootstrap(months=months, values=diff, n_resamples=N_RESAMPLES, seed=seed)
    tail = 100 * (1 - LEVEL) / 2
    low, high = np.percentile(resampled, [tail, 100 - tail])
    return float(diff.mean()), float(low), float(high)


def mean_f1(
    *,
    out: Path,
    arm: str,
    events: pl.DataFrame,
    setting: str,
    seeds: tuple[int, ...],
    bound: int,
    population: pl.Expr,
) -> float:
    """Return the event F1 of an arm at a precision bound, averaged over seeds."""
    values = []
    for seed in seeds:
        name = f"{arm}_s{seed}"
        part = outcomes(out=out, name=name, events=events).filter(
            pl.col("setting") == setting, population
        )
        precision = precision_bounds(out=out, name=name)[setting][bound]
        recall = float(part["found"].mean())  # ty: ignore[invalid-argument-type]
        values.append(event_f1(recall=recall, precision=precision))
    return float(np.mean(values))


def print_thresholds(*, out: Path, names: list[str]) -> None:
    """Print matched thresholds, odd-month counts, and the change in noise scale."""
    print("== Matched thresholds (z units) and odd-month unexplained counts")
    for name in names:
        thresholds = load_thresholds(out=out, name=name)
        odd = json.loads((out / f"odd_counts_matched_{name}.json").read_text())
        old = json.loads((out / f"odd_counts_old_{name}.json").read_text())
        print(
            f"  {name:8s} "
            + ", ".join(
                f"{s} {thresholds[s]:.2f} (count {odd[s]}, at 43's threshold {old[s]})"
                for s in SETTINGS
            )
        )
    print("\n== Median across series of the change in odd-month noise scale, new vs H0.5")
    for window in (8, 48):
        for seed in SEEDS:
            scales = {
                arm: pl.read_parquet(out / f"scales_{arm}_s{seed}.parquet")
                .filter(pl.col("window") == window)
                .sort("series")["scale"]
                for arm in ("H0.5", "HOL")
            }
            change = (scales["HOL"] / scales["H0.5"] - 1).median()
            print(f"  window {window // 2} h, seed {seed}: {change:+.3%}")
    print("\n== Matched-threshold change, HOL minus H0.5, mean over seeds")
    for setting in SETTINGS:
        diffs = [
            load_thresholds(out=out, name=f"HOL_s{s}")[setting]
            - load_thresholds(out=out, name=f"H0.5_s{s}")[setting]
            for s in SEEDS
        ]
        print(
            f"  {LABELS[setting]:16s} {np.mean(diffs):+.3f} "
            f"(by seed {[round(d, 2) for d in diffs]})"
        )


def print_holiday_alarms(*, out: Path, names: list[str]) -> None:
    """Print unexplained 24-hour D0 alarms within 1 day of a bank holiday outside Christmas."""
    share_of_days = float((np.abs(holiday_distance_by_stamp()) <= 1).mean())
    print(f"\nShare of half-hours within 1 day of such a holiday: {share_of_days:.3f}")
    print("\n== Unexplained D0 24 h alarms, all months, within 1 day of a bank holiday")
    for label in ("old", "matched"):
        for name in names:
            alarms = pl.read_parquet(out / f"alarms_{label}_{name}.parquet").filter(
                pl.col("setting") == "D0_48", ~pl.col("logged_matched")
            )
            near = alarms.filter(pl.col("holiday_distance").abs() <= 1).height
            print(
                f"  {label:7s} threshold {name:8s} unexplained {alarms.height:4d}, "
                f"within 1 day of a holiday {near}"
            )
    print("\n== Unexplained D0 24 h alarms at the matched threshold, by distance in days")
    for name in names:
        alarms = pl.read_parquet(out / f"alarms_matched_{name}.parquet").filter(
            pl.col("setting") == "D0_48", ~pl.col("logged_matched")
        )
        counts = (
            alarms.group_by(pl.col("holiday_distance").abs().clip(0, 7))
            .len()
            .sort("holiday_distance")
        )
        print(f"  {name:8s} {dict(zip(counts['holiday_distance'], counts['len'], strict=True))}")


def print_contrasts(*, out: Path, events: pl.DataFrame) -> dict[str, tuple[float, float, float]]:
    """Print C1 to C4, the splits, the F1 differences, and return the contrasts."""
    results: dict[str, tuple[float, float, float]] = {}
    even_headline = events.filter(pl.col("headline"), pl.col("even"))
    print(
        f"\n== Contrasts, new minus H0.5, seeds 0 to 2, headline band, even months "
        f"({even_headline.height} transfers; {int(LEVEL * 10000) / 100}% interval)"
    )
    for index, (label, setting) in enumerate(CONTRASTS.items()):
        new = seed_mean_found(out=out, arm="HOL", events=events, setting=setting)
        reference = seed_mean_found(out=out, arm="H0.5", events=events, setting=setting)
        mean, low, high = paired_interval(
            population=even_headline.select("event_id", "year_month"),
            new=new,
            reference=reference,
            seed=SEED + index,
        )
        results[label] = (mean, low, high)
        f1 = {
            bound: mean_f1(
                out=out,
                arm="HOL",
                events=events,
                setting=setting,
                seeds=SEEDS,
                bound=bound,
                population=pl.col("headline") & pl.col("even"),
            )
            - mean_f1(
                out=out,
                arm="H0.5",
                events=events,
                setting=setting,
                seeds=SEEDS,
                bound=bound,
                population=pl.col("headline") & pl.col("even"),
            )
            for bound in (0, 1)
        }
        print(
            f"  {label} {LABELS[setting]:16s} {mean:+.3f} ({low:+.3f} to {high:+.3f}); "
            f"F1 difference {f1[0]:+.3f} at the lower and {f1[1]:+.3f} at the upper precision bound"
        )
        for name, keep in (
            ("metered receiver", pl.col("metered_receivers") > 0),
            ("no metered receiver", pl.col("metered_receivers") == 0),
        ):
            part = even_headline.filter(keep)
            m, lo, hi = paired_interval(
                population=part.select("event_id", "year_month"),
                new=new,
                reference=reference,
                seed=SEED + index,
            )
            print(f"      {name:20s} {m:+.3f} ({lo:+.3f} to {hi:+.3f}) over {part.height}")
    return results


def print_verdict(*, out: Path, events: pl.DataFrame, c1: tuple[float, float, float]) -> None:
    """Apply the decision rule to C1 and the F1 differences."""
    mean, low, high = c1
    f1_positive = all(
        mean_f1(
            out=out,
            arm="HOL",
            events=events,
            setting="D0_48",
            seeds=SEEDS,
            bound=bound,
            population=pl.col("headline") & pl.col("even"),
        )
        > mean_f1(
            out=out,
            arm="H0.5",
            events=events,
            setting="D0_48",
            seeds=SEEDS,
            bound=bound,
            population=pl.col("headline") & pl.col("even"),
        )
        for bound in (0, 1)
    )
    if low > 0 and f1_positive:
        verdict = "supported"
    elif high < REFUTED_BELOW:
        verdict = "refuted"
    else:
        verdict = "inconclusive"
    print(
        f"\n== Verdict by the decision rule: {verdict} (C1 {mean:+.3f}, interval {low:+.3f} to "
        f"{high:+.3f}; event F1 positive at both bounds: {f1_positive})"
    )


def print_reported(*, out: Path, events: pl.DataFrame) -> None:
    """Print placebo contrasts, all months, odd months, and every band (seed-mean recall)."""
    print("\n== Reported beside the contrasts, headline band, D0 24 h (seed 0 for the placebo)")
    pairs = {
        "placebo minus H0.5": ("PLA", (0,), "H0.5"),
        "new arm minus placebo": ("HOL", (0,), "PLA"),
    }
    even_headline = events.filter(pl.col("headline"), pl.col("even"))
    for label, (arm, seeds, ref) in pairs.items():
        for setting in ("D0_48", "D0_8", "D1_48", "D1_8"):
            new = seed_mean_found(out=out, arm=arm, events=events, setting=setting, seeds=seeds)
            reference = seed_mean_found(
                out=out, arm=ref, events=events, setting=setting, seeds=(0,)
            )
            m, lo, hi = paired_interval(
                population=even_headline.select("event_id", "year_month"),
                new=new,
                reference=reference,
                seed=SEED + 10,
            )
            print(f"  {label:22s} {LABELS[setting]:16s} {m:+.3f} ({lo:+.3f} to {hi:+.3f})")
    print("\n== Share found by arm, headline band, by month parity (mean over seeds)")
    for setting in SETTINGS:
        for arm, seeds in (("H0.5", SEEDS), ("HOL", SEEDS), ("PLA", (0,))):
            found = seed_mean_found(out=out, arm=arm, events=events, setting=setting, seeds=seeds)
            table = events.join(found, on="event_id").filter(pl.col("headline"))
            even = table.filter(pl.col("even"))["found"].mean()
            odd = table.filter(~pl.col("even"))["found"].mean()
            print(
                f"  {LABELS[setting]:16s} {arm:5s} even months {even:.3f} "
                f"({table.filter(pl.col('even')).height}), odd months {odd:.3f} "
                f"({table.filter(~pl.col('even')).height})"
            )
    print("\n== Share found by band, even months, mean over seeds (D0 24 h)")
    for arm, seeds in (("H0.5", SEEDS), ("HOL", SEEDS), ("PLA", (0,))):
        found = seed_mean_found(out=out, arm=arm, events=events, setting="D0_48", seeds=seeds)
        table = events.join(found, on="event_id").filter(pl.col("even"))
        parts = []
        for size in SIZE_BANDS:
            for duration in DURATION_BANDS:
                band = table.filter(
                    pl.col("size_band") == size, pl.col("duration_band") == duration
                )
                parts.append(f"{size}/{duration} {band['found'].mean():.2f} ({band.height})")
        print(f"  {arm:5s} " + "; ".join(parts))


def print_precision(*, out: Path, names: list[str]) -> None:
    """Print each arm-seed's precision bounds from even-month alarms."""
    print("\n== Precision bounds from even-month alarms (lower to upper)")
    for name in names:
        bounds = precision_bounds(out=out, name=name)
        print(
            f"  {name:8s} "
            + ", ".join(f"{LABELS[s]} {lo:.2f} to {hi:.2f}" for s, (lo, hi) in bounds.items())
        )


def print_real(*, out: Path, names: list[str]) -> None:
    """Print how many labelled real transfers each arm-seed finds (plain intervals)."""
    print("\n== Real labelled transfers found (plain intervals), by setting")
    for name in names:
        real = pl.read_parquet(out / f"real_results_{name}.parquet").filter(
            pl.col("variant") == "plain"
        )
        primary = real.filter(pl.col("share") >= 0.25, pl.col("hours") > 24)
        for label, frame in (
            ("primary", primary),
            ("all over 1 day", real.filter(pl.col("hours") > 24)),
        ):
            parts = []
            for setting in SETTINGS:
                part = frame.filter(pl.col("setting") == setting)
                parts.append(f"{LABELS[setting]} {int(part['found'].sum())}/{part.height}")
            print(f"  {name:8s} {label:15s} " + ", ".join(parts))


def print_diagnostics(*, out: Path) -> None:
    """Print the spread, autocorrelation, shares, and median-error tables when they exist."""
    if (out / "diag_spread.parquet").exists():
        spread = pl.read_parquet(out / "diag_spread.parquet")
        print("\n== Residual spread and autocorrelation (mean over the 16 series)")
        summary = (
            spread.group_by("name")
            .agg(
                pl.col("mad_points_of_capacity").mean(),
                pl.col("lag_1").mean(),
                pl.col("lag_48").mean(),
                pl.col("daily_lag_1").mean(),
            )
            .sort("name")
        )
        print(summary)
    if (out / "diag_shares.parquet").exists():
        print("\n== Share of squared residual against share of half-hours")
        print(pl.read_parquet(out / "diag_shares.parquet").sort("name", "region"))
    if (out / "diag_mae.parquet").exists():
        mae = pl.read_parquet(out / "diag_mae.parquet")
        print("\n== Mean absolute error of the median, points of capacity")
        print(mae.sort("name", "scope"))
        base = mae.filter(pl.col("name") == "H0.5_s0").select("scope", base="mae_points")
        worst = (
            mae.filter(pl.col("name").str.starts_with("HOL"))
            .join(base, on="scope")
            .with_columns(change=pl.col("mae_points") - pl.col("base"))
        )
        print(worst.select("name", "scope", "change"))
        all_rows = float(worst.filter(pl.col("scope") == "all rows")["change"].mean())  # ty: ignore[invalid-argument-type]
        flag = "STOP and look for a bug" if all_rows > 0.10 else "within the stop rule"
        print(f"  mean change over the whole period for the new arm: {all_rows:+.3f} ({flag})")
    check = out / "injection_check.parquet"
    if check.exists():
        joined = pl.read_parquet(check)
        agree = (pl.col("found") == pl.col("found_refit")) & (
            pl.col("onset_found") == pl.col("onset_found_refit")
        )
        print(
            f"\n== Injection check: {joined.height} transfer-setting pairs, agreement "
            f"{float(joined.select(agree.mean()).item()):.3f} (passes at 0.95 or more)"
        )


def print_scored_operating_points(*, out: Path, names: list[str], events: pl.DataFrame) -> None:
    """Print even-month alarm counts, odd-month recall, and event F1 of every arm.

    Matching fixes only the odd-month count, so the even months the contrasts are scored on can
    hold a different number of unexplained alarms for each arm.
    """
    print("\n== Unexplained D0 24 h alarms in even months (scored) and odd months (matched)")
    for name in names:
        alarms = pl.read_parquet(out / f"alarms_matched_{name}.parquet").filter(
            pl.col("setting") == "D0_48", ~pl.col("logged_matched")
        )
        even = alarms.filter(pl.col("month") % 2 == 0).height
        print(f"  {name:8s} even months {even}, odd months {alarms.height - even}")
    print("\n== Event F1, D0 24 h, headline band, even months (mean over the arm's seeds)")
    for arm, seeds in (("H0.5", SEEDS), ("HOL", SEEDS), ("PLA", (0,)), ("H0.5", (0,))):
        values = [
            mean_f1(
                out=out,
                arm=arm,
                events=events,
                setting="D0_48",
                seeds=seeds,
                bound=bound,
                population=pl.col("headline") & pl.col("even"),
            )
            for bound in (0, 1)
        ]
        print(f"  {arm:5s} seeds {seeds}: {values[0]:.3f} at the lower and {values[1]:.3f} upper")
    print("\n== Odd-month share found, D0 24 h, headline band: new arm minus H0.5")
    odd = events.filter(pl.col("headline"), ~pl.col("even"))
    new = seed_mean_found(out=out, arm="HOL", events=events, setting="D0_48")
    reference = seed_mean_found(out=out, arm="H0.5", events=events, setting="D0_48")
    odd_diff = odd.join(new, on="event_id").join(reference, on="event_id", suffix="_ref")
    odd_change = float((odd_diff["found"] - odd_diff["found_ref"]).mean())  # ty: ignore[invalid-argument-type]
    print(f"  odd months ({odd.height} transfers): {odd_change:+.3f}")
    even = events.filter(pl.col("headline"), pl.col("even"))
    even_diff = even.join(new, on="event_id").join(reference, on="event_id", suffix="_ref")
    even_change = float((even_diff["found"] - even_diff["found_ref"]).mean())  # ty: ignore[invalid-argument-type]
    pooled = (odd_change * odd.height + even_change * even.height) / (odd.height + even.height)
    print(f"  pooled over {odd.height + even.height} transfers: {pooled:+.3f}")


def main() -> None:
    """Print all the sections."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    out = parser.parse_args().out
    names = sorted(p.stem.removeprefix("thresholds_") for p in out.glob("thresholds_*.json"))
    events = event_table(out=out)
    print_thresholds(out=out, names=names)
    print_holiday_alarms(out=out, names=names)
    contrasts = print_contrasts(out=out, events=events)
    print_verdict(out=out, events=events, c1=contrasts["C1"])
    print_reported(out=out, events=events)
    print_precision(out=out, names=names)
    print_real(out=out, names=names)
    print_scored_operating_points(out=out, names=names, events=events)
    print_diagnostics(out=out)


if __name__ == "__main__":
    main()
