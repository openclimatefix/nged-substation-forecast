"""Print every number the write-up quotes: the scoreboard for experiment 43.

uv run python ml-experiments/00043-sw2-1-detector-scoreboard/e00043_report.py
"""

from __future__ import annotations

from math import sqrt
from typing import Final

import numpy as np
import polars as pl
from e00043_common import WORK_DIR, stamp_year_months
from e00043_inject import DURATION_BANDS, HEADLINE_BAND, SIZE_BANDS
from e00043_score import event_f1, month_bootstrap

LABELS: Final[dict[str, str]] = {
    "alarms_D0_8_6.0": "D0, 4 h window",
    "alarms_D1_8_3.75": "D1, 4 h window",
    "alarms_D0_48_4.0": "D0, 24 h window",
    "alarms_D1_48_2.5": "D1, 24 h window",
}
PAIRS: Final[dict[str, tuple[str, str]]] = {
    "4 h": ("alarms_D0_8_6.0", "alarms_D1_8_3.75"),
    "24 h": ("alarms_D0_48_4.0", "alarms_D1_48_2.5"),
}
N_RESAMPLES: Final[int] = 4000
SEED: Final[int] = 4300


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for k successes in n trials."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def precision_bracket() -> pl.DataFrame:
    """Per detector setting: alarms, logged-matched, and the lower and upper precision."""
    base = pl.read_parquet(WORK_DIR / "baseline_summary.parquet")
    sample = pl.read_parquet(WORK_DIR / "alarm_sample.parquet")
    joined = base.join(sample, on="setting").with_columns(
        visible_rate=pl.col("visible") / pl.col("sampled"),
        lower=pl.col("logged_matched") / pl.col("alarms"),
    )
    return joined.with_columns(
        upper=(pl.col("logged_matched") + pl.col("unexplained") * pl.col("visible_rate"))
        / pl.col("alarms")
    )


def _print_bracket(bracket: pl.DataFrame) -> None:
    print("== Precision bracket over all alarms")
    for row in bracket.iter_rows(named=True):
        print(
            f"  {LABELS[row['setting']]:16s} alarms {row['alarms']:4d}, "
            f"logged-matched {row['logged_matched']:4d}, unexplained {row['unexplained']:4d}, "
            f"sample visible {row['visible']}/{row['sampled']}, "
            f"precision {row['lower']:.3f} to {row['upper']:.3f}"
        )


def _f1_range(*, recall: float, low: float, high: float) -> str:
    first = event_f1(recall=recall, precision=low)
    return f"{first:.2f} to {event_f1(recall=recall, precision=high):.2f}"


def _print_scoreboard(*, plain: pl.DataFrame, bracket: pl.DataFrame) -> None:
    print("\n== Injected events: recall, onset-only recall, end error (plain intervals)")
    for setting, label in LABELS.items():
        sub = plain.filter(pl.col("setting") == setting)
        row = bracket.filter(pl.col("setting") == setting)
        low, high = row["lower"].item(), row["upper"].item()
        k, n = int(sub["found"].sum()), sub.height
        lo, hi = wilson(k, n)
        errors = sub.filter(pl.col("onset_found"))["end_error"].drop_nans() / 2
        print(
            f"  {label:16s} all bands: found {k}/{n} = {k / n:.3f} ({lo:.2f} to {hi:.2f}), "
            f"onset-only {sub['onset_found'].mean():.3f}, median end error "
            f"{errors.median():.1f} h, F1 {_f1_range(recall=k / n, low=low, high=high)}"
        )
        for size in SIZE_BANDS:
            for duration in DURATION_BANDS:
                band = sub.filter(pl.col("size_band") == size, pl.col("duration_band") == duration)
                kb = int(band["found"].sum())
                lo_b, hi_b = wilson(kb, band.height)
                marker = " (headline)" if (size, duration) == HEADLINE_BAND else ""
                print(
                    f"      {size:12s} {duration:9s} {kb:3d}/{band.height:3d} = "
                    f"{kb / band.height:.2f} ({lo_b:.2f} to {hi_b:.2f}) onset-only "
                    f"{band['onset_found'].mean():.2f} "
                    f"F1 {_f1_range(recall=kb / band.height, low=low, high=high)}{marker}"
                )


def _headline(*, plain: pl.DataFrame, bracket: pl.DataFrame) -> None:
    print("\n== Headline contrast: D1 minus D0, 25% or more and over 1 day, plain intervals")
    months = stamp_year_months()
    band = plain.filter(
        pl.col("size_band") == HEADLINE_BAND[0], pl.col("duration_band") == HEADLINE_BAND[1]
    )
    for window, (d0, d1) in PAIRS.items():
        found = {s: band.filter(pl.col("setting") == s).sort("event_id") for s in (d0, d1)}
        month = months[found[d0]["onset"].to_numpy()]
        resampled = [
            month_bootstrap(
                months=month,
                values=found[s]["found"].to_numpy().astype(float),
                n_resamples=N_RESAMPLES,
                seed=SEED,
            )
            for s in (d0, d1)
        ]
        point = found[d1]["found"].to_numpy().mean() - found[d0]["found"].to_numpy().mean()
        lo, hi = np.percentile(resampled[1] - resampled[0], [2.5, 97.5])
        print(
            f"  {window}: recall difference {point:+.3f} ({lo:+.3f} to {hi:+.3f}) "
            f"over {found[d0].height} events"
        )
        for bound in ("lower", "upper"):
            p = [float(bracket.filter(pl.col("setting") == s)[bound].item()) for s in (d0, d1)]
            r = [float(found[s]["found"].to_numpy().mean()) for s in (d0, d1)]
            diff = event_f1(recall=r[1], precision=p[1]) - event_f1(recall=r[0], precision=p[0])
            print(f"      F1 difference at the {bound} precision bound: {diff:+.3f}")
    print("\n== D1 minus D0 on the headline events that have a metered receiver")
    metered = band.filter(pl.col("metered_receivers") > 0)
    for window, (d0, d1) in PAIRS.items():
        found = {s: metered.filter(pl.col("setting") == s).sort("event_id") for s in (d0, d1)}
        month = months[found[d0]["onset"].to_numpy()]
        resampled = [
            month_bootstrap(
                months=month,
                values=found[s]["found"].to_numpy().astype(float),
                n_resamples=N_RESAMPLES,
                seed=SEED,
            )
            for s in (d0, d1)
        ]
        point = found[d1]["found"].to_numpy().mean() - found[d0]["found"].to_numpy().mean()
        lo, hi = np.percentile(resampled[1] - resampled[0], [2.5, 97.5])
        print(
            f"  {window}: recall difference {point:+.3f} ({lo:+.3f} to {hi:+.3f}) "
            f"over {found[d0].height} events"
        )
    print("\n== Headline band by whether a metered receiver exists")
    for setting, label in LABELS.items():
        for name, keep in (
            ("metered receiver", band["metered_receivers"] > 0),
            ("none", band["metered_receivers"] == 0),
        ):
            part = band.filter(keep, pl.col("setting") == setting)
            print(f"  {label:16s} {name:17s} found {int(part['found'].sum())}/{part.height}")


def _print_real() -> None:
    print("\n== Real labelled events")
    real = pl.read_parquet(WORK_DIR / "real_results.parquet").filter(pl.col("variant") == "plain")
    primary = real.filter(pl.col("share") >= 0.25, pl.col("hours") > 24)
    secondary = real.filter(pl.col("hours") > 24)
    one = secondary.filter(pl.col("setting") == next(iter(LABELS)))
    print(f"  primary set: {primary['card_id'].n_unique()} events")
    print(
        f"  visible operations over 1 day: {one.height}; share 25% or more "
        f"{one.filter(pl.col('share') >= 0.25).height}, under 25% "
        f"{one.filter(pl.col('share') < 0.25).height}, share unknown "
        f"{one.filter(pl.col('share').is_null()).height}"
    )
    for setting, label in LABELS.items():
        for name, frame in (("primary", primary), ("all over 1 day", secondary)):
            sub = frame.filter(pl.col("setting") == setting)
            for visibility in ("conserved", "one_sided"):
                part = sub.filter(pl.col("visibility") == visibility)
                print(
                    f"  {label:16s} {name:15s} {visibility:10s} found "
                    f"{int(part['found'].sum())}/{part.height}, onset-only "
                    f"{int(part['onset_found'].sum())}/{part.height}"
                )


def main() -> None:
    """Print the precision bracket, scoreboard, headline contrast, and real events."""
    bracket = precision_bracket()
    _print_bracket(bracket)
    events = pl.read_parquet(WORK_DIR / "events.parquet")
    print("\n== Events per band")
    print(events.group_by("size_band", "duration_band").len().sort("size_band", "duration_band"))
    results = pl.read_parquet(WORK_DIR / "event_results.parquet").join(events, on="event_id")
    plain = results.filter(pl.col("variant") == "plain")
    _print_scoreboard(plain=plain, bracket=bracket)
    _headline(plain=plain, bracket=bracket)
    print("\n== Interval variants (all bands, recall)")
    for variant in ("plain", "fixed"):
        sub = results.filter(pl.col("variant") == variant)
        recalls = ", ".join(
            f"{LABELS[s]} {sub.filter(pl.col('setting') == s)['found'].mean():.3f}" for s in LABELS
        )
        print(f"  {variant}: {recalls}")
    print("\n== Delay from the true end to the end being known (hours, found events, plain)")
    for setting, label in LABELS.items():
        delay = plain.filter(pl.col("setting") == setting, pl.col("found"))["known_delay"]
        delay = delay.drop_nans() / 2
        print(f"  {label:16s} median {delay.median():.1f} h over {delay.len()} events")
    _print_real()


if __name__ == "__main__":
    main()
