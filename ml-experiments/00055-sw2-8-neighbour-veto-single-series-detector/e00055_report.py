"""Print every number the write-up of experiment 55 quotes.

Reads the tables that `e00055_run.py` wrote: the veto strength chosen on odd months, the matched
thresholds, the contrasts C1 to C4 with their intervals, the verdict the decision rule gives, and
the reported checks.

    uv run python ml-experiments/00055-*/e00055_report.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00055_common import (
    EXPERIMENT_43_DIR,
    LABELLED_META_PATH,
    LABELLED_PATH,
    WORK_DIR,
    stamp_months,
    stamp_year_months,
)
from e00055_detect import VETO_ZS, neighbour_check_for_cards
from e00055_inject import HEADLINE_BAND
from e00055_run import vz_tag
from e00055_score import event_f1, month_bootstrap

N_RESAMPLES: Final[int] = 20_000
LEVEL: Final[float] = 0.9875
"""Two-sided level of the intervals: Bonferroni across the four contrasts C1 to C4."""
SEED: Final[int] = 5500
SUPPORT_AT_LEAST: Final[float] = 0.01
ALARM_TOLERANCE: Final[float] = 1.10
SAMPLE_NAMES: Final[dict[str, str]] = {"8": "alarms_D0_8_6.0", "48": "alarms_D0_48_4.0"}


def event_table(*, out: Path) -> pl.DataFrame:
    """Return the injected events with onset month, parity, and the headline flag."""
    events = pl.read_parquet(out / "events.parquet")
    onset = events["onset"].to_numpy()
    return events.with_columns(
        month=pl.Series(stamp_months()[onset]),
        year_month=pl.Series(stamp_year_months()[onset]),
        headline=(pl.col("size_band") == HEADLINE_BAND[0])
        & (pl.col("duration_band") == HEADLINE_BAND[1]),
    ).with_columns(even=pl.col("month") % 2 == 0)


def results(*, out: Path, veto_z: float) -> pl.DataFrame:
    """Return per-transfer outcomes at one veto strength, joined to the event table."""
    return (
        pl.read_parquet(out / f"event_results_{vz_tag(veto_z)}.parquet")
        .join(event_table(out=out), on="event_id", validate="m:1")
        .sort("setting", "event_id")
    )


def choose_veto(*, out: Path) -> float:
    """Pick the veto strength with the highest odd-month share found; ties go to 2.0."""
    best: dict[float, float] = {}
    for veto_z in VETO_ZS:
        part = results(out=out, veto_z=veto_z).filter(
            pl.col("setting") == "D2_48",
            pl.col("headline"),
            ~pl.col("even"),
            pl.col("set") != "holiday",
        )
        best[veto_z] = float(part["found"].mean())  # ty: ignore[invalid-argument-type]
        print(f"veto {vz_tag(veto_z)}: odd-month D2 24 h share found {best[veto_z]:.4f}")
    top = max(best.values())
    tied = [z for z, v in best.items() if abs(v - top) < 1e-12]
    return 2.0 if 2.0 in tied else tied[0]


def paired(*, new: pl.DataFrame, reference: pl.DataFrame, seed: int) -> tuple[float, float, float]:
    """Mean paired difference `new - reference` of `found` with a Bonferroni interval."""
    joined = new.join(reference, on="event_id", suffix="_ref").sort("event_id")
    diff = (joined["found"].cast(pl.Float64) - joined["found_ref"].cast(pl.Float64)).to_numpy()
    resampled = month_bootstrap(
        months=joined["year_month"].to_numpy(), values=diff, n_resamples=N_RESAMPLES, seed=seed
    )
    tail = 100 * (1 - LEVEL) / 2
    low, high = np.percentile(resampled, [tail, 100 - tail])
    return float(diff.mean()), float(low), float(high)


def alarms_of(*, out: Path, veto_z: float) -> pl.DataFrame:
    """Return the real alarms of every setting at one veto strength."""
    return pl.read_parquet(out / f"alarms_{vz_tag(veto_z)}.parquet")


def print_thresholds(*, out: Path) -> None:
    """Print the matched thresholds and the odd-month counts."""
    matched = json.loads((out / "thresholds.json").read_text())
    print("\nmatched thresholds (window in half-hours)")
    for tag, per_window in matched.items():
        print(f"  veto {tag}: {per_window}")
    curves = pl.read_parquet(out / "curves.parquet")
    print(curves.group_by("veto_z", "window").agg(points=pl.len()).sort("veto_z", "window"))


def print_ratio(*, out: Path, veto_z: float) -> None:
    """Print the veto's removal rate on unexplained and logged-explained alarms."""
    for window in (8, 48):
        table = alarms_of(out=out, veto_z=veto_z).filter(pl.col("setting") == f"D2_{window}")
        vetoed = pl.col("tier") == "vetoed"
        odd = pl.col("month") % 2 == 1
        explained = pl.col("logged_matched")
        for label, mask in (
            ("unexplained, odd months", ~explained & odd),
            ("unexplained, all months", ~explained),
            ("logged-explained, all months", explained),
        ):
            part = table.filter(mask)
            print(
                f"window {window} veto {vz_tag(veto_z)}: {label}: vetoed "
                f"{part.filter(vetoed).height} of {part.height}"
            )
        by_series = table.filter(vetoed).group_by("series").len().sort("series")
        print("  vetoed alarms by series:", dict(by_series.iter_rows()))
        print(
            "  vetoed logged-explained by series:",
            dict(
                table.filter(vetoed, explained).group_by("series").len().sort("series").iter_rows()
            ),
        )


def print_tiers(*, out: Path, veto_z: float) -> None:
    """Print the logged-explained share per tier and the labelled-visible rate per tier."""
    table = alarms_of(out=out, veto_z=veto_z).filter(pl.col("setting").str.starts_with("D2"))
    print("\nlogged-explained share of D2 alarms by tier")
    print(
        table.group_by("setting", "tier")
        .agg(alarms=pl.len(), logged=pl.col("logged_matched").sum())
        .with_columns(share=pl.col("logged") / pl.col("alarms"))
        .sort("setting", "tier")
    )
    meta = json.loads(LABELLED_META_PATH.read_text())
    labelled = pl.read_parquet(LABELLED_PATH).filter(pl.col("kind") == "alarm")
    cards = pl.DataFrame(
        [
            {"card_id": c, "detector": m["detector"], "stamp": m["alarm_stamp"]}
            for c, m in meta.items()
            if m["kind"] == "alarm"
        ]
    ).join(labelled.select("card_id", series="side_a", visibility="visibility"), on="card_id")
    tiers = neighbour_check_for_cards(cards=cards, veto_z=veto_z)
    print("\nexperiment 41's 148 labelled alarms by tier (visible = conserved or one_sided)")
    print(
        tiers.with_columns(visible=pl.col("visibility").is_in(["conserved", "one_sided"]))
        .group_by("tier")
        .agg(alarms=pl.len(), visible=pl.col("visible").sum())
        .sort("tier")
    )


def print_contrasts(*, out: Path, veto_z: float) -> tuple[float, float, float]:
    """Print C1 to C4 and return C1's mean and interval."""
    data = results(out=out, veto_z=veto_z)
    base = data.filter(pl.col("headline"), pl.col("even"), pl.col("set") != "holiday")
    print(f"\nC1 to C4 population: {base.filter(pl.col('setting') == 'D2_48').height} transfers")
    specs = {
        "C1": ("48", None),
        "C2": ("8", None),
        "C3": ("48", pl.col("metered_receivers") > 0),
        "C4": ("48", pl.col("metered_receivers") == 0),
    }
    c1 = (0.0, 0.0, 0.0)
    for name, (window, mask) in specs.items():
        part = base if mask is None else base.filter(mask)
        new = part.filter(pl.col("setting") == f"D2_{window}")
        ref = part.filter(pl.col("setting") == f"D0M_{window}")
        old = part.filter(pl.col("setting") == f"D0_{window}")
        mean, low, high = paired(new=new, reference=ref, seed=SEED)
        print(
            f"{name}: n {new.height}, D2 {new['found'].mean():.3f}, D0M {ref['found'].mean():.3f}, "
            f"D0 at experiment 43's threshold {old['found'].mean():.3f}; "
            f"D2 - D0M {mean:+.4f} ({low:+.4f} to {high:+.4f})"
        )
        if name == "C1":
            c1 = (mean, low, high)
    return c1


def print_verdict(*, out: Path, veto_z: float, c1: tuple[float, float, float]) -> None:
    """Apply the decision rule."""
    mean, low, high = c1
    table = alarms_of(out=out, veto_z=veto_z)
    even = pl.col("month") % 2 == 0

    def count(setting: str) -> int:
        return table.filter(
            pl.col("setting") == setting,
            even,
            ~pl.col("logged_matched"),
            pl.col("tier") != "vetoed",
        ).height

    d2, d0m = count("D2_48"), count("D0M_48")
    print(
        f"\neven-month unexplained 24 h alarms: D2 {d2}, D0M {d0m} (limit {ALARM_TOLERANCE:.2f}x)"
    )
    alarm_ok = d2 <= ALARM_TOLERANCE * d0m
    if low > 0 and mean >= SUPPORT_AT_LEAST and alarm_ok:
        verdict = "supported"
    elif high < SUPPORT_AT_LEAST:
        verdict = "refuted"
    else:
        verdict = "inconclusive"
    print(f"C1 {mean:+.4f} ({low:+.4f} to {high:+.4f}); alarm count ok {alarm_ok}; {verdict}")


def print_reported(*, out: Path, veto_z: float) -> None:
    """Print the by-band recall, the holiday set, receiver naming, and precision."""
    data = results(out=out, veto_z=veto_z)
    print("\nshare found by set, 24 h window, headline band (all months)")
    for name in ("original", "extra", "holiday"):
        for setting in ("D0_48", "D0M_48", "D2_48", "D1_48"):
            part = data.filter(
                pl.col("set") == name, pl.col("headline"), pl.col("setting") == setting
            )
            print(f"  {name:9s} {setting:7s} {part['found'].sum():4d} of {part.height}")
    holiday = data.filter(pl.col("set") == "holiday", pl.col("setting").is_in(["D0M_48", "D2_48"]))
    wide = holiday.pivot(on="setting", index="event_id", values="found")
    print(
        "holiday set: D0M found",
        int(wide["D0M_48"].sum()),
        "D2 found",
        int(wide["D2_48"].sum()),
        "lost to the veto",
        wide.filter(pl.col("D0M_48"), ~pl.col("D2_48")).height,
        "gained",
        wide.filter(~pl.col("D0M_48"), pl.col("D2_48")).height,
    )
    events = event_table(out=out)
    holiday_events = events.filter(pl.col("set") == "holiday")
    print(
        "holiday onsets the alarm part of the mask would reject:",
        holiday_events.filter(~pl.col("alarm_quiet")).height,
        "of",
        holiday_events.height,
    )
    named = data.filter(
        pl.col("setting") == "D2_48", pl.col("found"), pl.col("metered_receivers") > 0
    )
    print(
        "receiver named, D2 24 h found transfers with a metered receiver:",
        int(named["named"].sum()),
        "of",
        named.height,
    )
    print("\nprecision of real alarms (even months), lower bound = logged-explained share")
    table = alarms_of(out=out, veto_z=veto_z).filter(pl.col("month") % 2 == 0)
    sample = pl.read_parquet(EXPERIMENT_43_DIR / "alarm_sample.parquet")
    for window, name in SAMPLE_NAMES.items():
        row = sample.filter(pl.col("setting") == name)
        rate = row["visible"].item() / row["sampled"].item()
        for setting in (f"D0_{window}", f"D0M_{window}", f"D2_{window}"):
            part = table.filter(pl.col("setting") == setting, pl.col("tier") != "vetoed")
            n, matched = part.height, int(part["logged_matched"].sum())
            if n == 0:
                continue
            low, high = matched / n, (matched + (n - matched) * rate) / n
            found = data.filter(
                pl.col("setting") == setting,
                pl.col("headline"),
                pl.col("even"),
                pl.col("set") != "holiday",
            )["found"].mean()
            print(
                f"  {setting}: alarms {n}, precision {low:.3f} to {high:.3f}, "
                f"F1 {event_f1(recall=float(found), precision=low):.3f} to "  # ty: ignore[invalid-argument-type]
                f"{event_f1(recall=float(found), precision=high):.3f}"  # ty: ignore[invalid-argument-type]
            )


def print_real(*, out: Path, veto_z: float) -> None:
    """Print the labelled real transfers found, as counts."""
    real = pl.read_parquet(out / f"real_results_{vz_tag(veto_z)}.parquet")
    print(
        "\nlabelled real transfers (25% or more of the receiving load, over 1 day; and all >1 day)"
    )
    for setting in ("D0_48", "D0M_48", "D2_48", "D0_8", "D0M_8", "D2_8"):
        part = real.filter(pl.col("setting") == setting)
        big = part.filter(pl.col("share") >= 0.25, pl.col("hours") > 24)
        long = part.filter(pl.col("hours") > 24)
        print(
            f"  {setting}: {int(big['found'].sum())} of {big.height}; "
            f"{int(long['found'].sum())} of {long.height}"
        )


def print_ceiling(*, out: Path) -> None:
    """Print the ceiling check."""
    path = out / "ceiling.parquet"
    if path.exists():
        part = pl.read_parquet(path)
        for setting in ("D0_48", "D0M_48", "D0_8", "D0M_8"):
            rows = part.filter(pl.col("setting") == setting)
            print(f"ceiling {setting}: found {rows['found'].sum()} of {rows.height}")


def main() -> None:
    """Print every number the write-up quotes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    args = parser.parse_args()
    print_thresholds(out=args.out)
    print_ceiling(out=args.out)
    veto_z = choose_veto(out=args.out)
    print(f"\nchosen veto strength: {vz_tag(veto_z)}")
    print_ratio(out=args.out, veto_z=veto_z)
    print_tiers(out=args.out, veto_z=veto_z)
    c1 = print_contrasts(out=args.out, veto_z=veto_z)
    print_verdict(out=args.out, veto_z=veto_z, c1=c1)
    print_reported(out=args.out, veto_z=veto_z)
    print_real(out=args.out, veto_z=veto_z)


if __name__ == "__main__":
    main()
