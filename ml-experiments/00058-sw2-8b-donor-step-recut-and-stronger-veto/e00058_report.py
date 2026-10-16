"""Print every number the write-up of experiment 58 quotes.

Reads the tables that `e00058_run.py` and `e00058_steps.py` wrote: recall by donor step in noise
units and by metered receiver (C1), the veto strength chosen on odd months, C2 and C3 with their
intervals, and the verdict the decision rule gives.

    uv run python ml-experiments/00058-*/e00058_report.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00058_common import WORK_DIR, stamp_months, stamp_year_months
from e00058_detect import VETO_ZS
from e00058_inject import HEADLINE_BAND
from e00058_run import vz_tag
from e00058_score import month_bootstrap

N_RESAMPLES: Final[int] = 20_000
LEVEL: Final[float] = 1 - 0.05 / 3 / 1
"""Two-sided level of the intervals: Bonferroni across the three contrasts C1 to C3 (98.33%)."""
SEED: Final[int] = 5800
N_BINS: Final[int] = 4
HALF_GAP: Final[float] = 0.165
"""Half of experiment 55's raw gap, 0.33 (0.50 minus 0.17)."""
SUPPORT_AT_LEAST: Final[float] = 0.01
ALARM_TOLERANCE: Final[float] = 1.10


def event_table(*, out: Path) -> pl.DataFrame:
    """Return the injected events with onset month, parity, donor step, and the headline flag."""
    events = pl.read_parquet(out / "events.parquet")
    onset = events["onset"].to_numpy()
    return (
        events.with_columns(
            month=pl.Series(stamp_months()[onset]),
            year_month=pl.Series(stamp_year_months()[onset]),
            headline=(pl.col("size_band") == HEADLINE_BAND[0])
            & (pl.col("duration_band") == HEADLINE_BAND[1]),
        )
        .with_columns(even=pl.col("month") % 2 == 0)
        .join(pl.read_parquet(out / "steps.parquet"), on="event_id", validate="1:1")
    )


def results(*, out: Path, veto_z: float) -> pl.DataFrame:
    """Return per-transfer outcomes at one veto strength, joined to the event table."""
    return (
        pl.read_parquet(out / f"event_results_{vz_tag(veto_z)}.parquet")
        .join(event_table(out=out), on="event_id", validate="m:1")
        .sort("setting", "event_id")
    )


def population(*, data: pl.DataFrame, setting: str) -> pl.DataFrame:
    """The 24-hour headline-band transfers that start in even months, one detector's rows."""
    return data.filter(
        pl.col("setting") == setting,
        pl.col("headline"),
        pl.col("even"),
        pl.col("set") != "holiday",
    ).sort("event_id")


def choose_veto(*, out: Path) -> float:
    """Pick the veto strength with the highest odd-month share found; ties go to the weaker."""
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
    return max(z for z, v in best.items() if abs(v - top) < 1e-12)


def interval(*, resampled: np.ndarray) -> tuple[float, float]:
    """The two-sided interval of resampled statistics at `LEVEL`."""
    tail = 100 * (1 - LEVEL) / 2
    low, high = np.percentile(resampled, [tail, 100 - tail])
    return float(low), float(high)


def paired(*, new: pl.DataFrame, reference: pl.DataFrame) -> tuple[float, float, float]:
    """Mean paired difference `new - reference` of `found` with a Bonferroni interval."""
    joined = new.join(reference, on="event_id", suffix="_ref").sort("event_id")
    diff = (joined["found"].cast(pl.Float64) - joined["found_ref"].cast(pl.Float64)).to_numpy()
    resampled = month_bootstrap(
        months=joined["year_month"].to_numpy(), values=diff, n_resamples=N_RESAMPLES, seed=SEED
    )
    low, high = interval(resampled=resampled)
    return float(diff.mean()), low, high


def matched_gap(
    *,
    step: str,
    rows: pl.DataFrame,
    edges: np.ndarray,
    months: np.ndarray,
    picks: np.ndarray | None,
) -> float:
    """Recall gap, metered minus unmetered, averaged over step bins weighted by unmetered count.

    `picks` is a (resamples, months) array of month indices; None computes the point value.
    """
    labels = np.unique(months)
    z = rows[step].to_numpy()
    metered = (rows["metered_receivers"] > 0).to_numpy()
    found = rows["found"].to_numpy().astype(float)
    bin_of = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, N_BINS - 1)
    stat = np.zeros((len(labels), N_BINS, 2, 2))  # month, bin, group, (found sum, count)
    for i, label in enumerate(labels):
        inside = months == label
        for b in range(N_BINS):
            for g, flag in enumerate((False, True)):
                pick = inside & (bin_of == b) & (metered == flag)
                stat[i, b, g] = (found[pick].sum(), pick.sum())
    chosen = stat.sum(axis=0)[None] if picks is None else stat[picks].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = chosen[..., 0] / chosen[..., 1]
    weight = chosen[:, :, 0, 1]
    gap = (recall[:, :, 1] - recall[:, :, 0]) * weight
    valid = np.isfinite(gap)
    total = np.where(valid, weight, 0).sum(axis=1)
    value = np.where(valid, gap, 0).sum(axis=1) / total
    return float(value[0]) if picks is None else value  # ty: ignore[invalid-return-type]


def print_recut(*, data: pl.DataFrame, step: str) -> None:
    """Print recall by donor-step bin and group, and C1 with its interval."""
    for setting in ("D0M_48", "D2_48"):
        rows = population(data=data, setting=setting)
        metered = rows.filter(pl.col("metered_receivers") > 0)[step]
        unmetered = rows.filter(pl.col("metered_receivers") == 0)[step]
        print(
            f"\n[{step}] {setting}: n {rows.height}, metered {metered.len()}, "
            f"unmetered {unmetered.len()}"
        )
        print(
            "  donor step (noise units) quantiles 10/50/90: metered "
            f"{metered.quantile(0.1):.1f}/{metered.median():.1f}/{metered.quantile(0.9):.1f}, "
            f"unmetered {unmetered.quantile(0.1):.1f}/{unmetered.median():.1f}/"
            f"{unmetered.quantile(0.9):.1f}"
        )
        low = float(max(metered.to_numpy().min(), unmetered.to_numpy().min()))
        high = float(min(metered.to_numpy().max(), unmetered.to_numpy().max()))
        overlap = rows.filter(pl.col(step).is_between(low, high))
        print(f"  overlap range {low:.1f} to {high:.1f}: {overlap.height} of {rows.height}")
        edges = np.quantile(overlap[step].to_numpy(), np.linspace(0, 1, N_BINS + 1))
        edges[0], edges[-1] = low, high
        months = overlap["year_month"].to_numpy()
        for b in range(N_BINS):
            part = overlap.filter(
                pl.col(step) >= edges[b],
                (pl.col(step) < edges[b + 1]) if b < N_BINS - 1 else pl.lit(True),
            )
            cells = []
            for name, mask in (("metered", pl.col("metered_receivers") > 0), ("unmetered", None)):
                g = (
                    part.filter(mask)
                    if mask is not None
                    else part.filter(pl.col("metered_receivers") == 0)
                )
                cells.append(f"{name} {g['found'].sum()}/{g.height}")
            print(f"  bin {b + 1} ({edges[b]:.1f} to {edges[b + 1]:.1f}): " + ", ".join(cells))
        point = matched_gap(step=step, rows=overlap, edges=edges, months=months, picks=None)
        labels = np.unique(months)
        rng = np.random.default_rng(SEED)
        picks = rng.integers(0, len(labels), size=(N_RESAMPLES, len(labels)))
        resampled = matched_gap(step=step, rows=overlap, edges=edges, months=months, picks=picks)
        raw = rows.group_by(pl.col("metered_receivers") > 0).agg(r=pl.col("found").mean())
        print(f"  raw recall by metered receiver: {dict(raw.iter_rows())}")
        low_i, high_i = interval(resampled=np.asarray(resampled))
        reading = (
            "step size explains at least half the gap"
            if high_i < HALF_GAP
            else "step size explains less than half the gap"
            if low_i > HALF_GAP
            else "mixed"
        )
        print(f"  C1 matched gap {point:+.3f} ({low_i:+.3f} to {high_i:+.3f}): {reading}")
        print(f"  logistic check (found on log step and metered): {logistic(rows=rows, step=step)}")


def logistic(*, rows: pl.DataFrame, step: str) -> dict[str, float]:
    """Newton fit of found on [1, log donor step, metered receiver]; return the coefficients."""
    x = np.column_stack(
        [
            np.ones(rows.height),
            np.log(np.clip(rows[step].to_numpy(), 1e-6, None)),
            (rows["metered_receivers"] > 0).to_numpy().astype(float),
        ]
    )
    y = rows["found"].to_numpy().astype(float)
    beta = np.zeros(3)
    for _ in range(50):
        p = 1 / (1 + np.exp(-x @ beta))
        beta = beta + np.linalg.solve(
            x.T @ (x * (p * (1 - p))[:, None]) + 1e-8 * np.eye(3), x.T @ (y - p)
        )
    return {"intercept": float(beta[0]), "log_step": float(beta[1]), "metered": float(beta[2])}


def print_contrasts(*, out: Path, veto_z: float) -> tuple[float, float, float]:
    """Print C2 and C3 and return C2's mean and interval."""
    data = results(out=out, veto_z=veto_z)
    c2 = (0.0, 0.0, 0.0)
    for name, mask in (
        ("C2", pl.lit(True)),
        ("C3", pl.col("metered_receivers") == 0),
    ):
        new = population(data=data, setting="D2_48").filter(mask)
        ref = population(data=data, setting="D0M_48").filter(mask)
        mean, low, high = paired(new=new, reference=ref)
        print(
            f"{name}: n {new.height}, D2 {new['found'].mean():.3f}, D0M {ref['found'].mean():.3f}; "
            f"D2 - D0M {mean:+.4f} ({low:+.4f} to {high:+.4f})"
        )
        if name == "C2":
            c2 = (mean, low, high)
    return c2


def print_verdict(*, out: Path, veto_z: float, c2: tuple[float, float, float]) -> None:
    """Apply the decision rule."""
    mean, low, high = c2
    table = pl.read_parquet(out / f"alarms_{vz_tag(veto_z)}.parquet")

    def count(setting: str) -> int:
        return table.filter(
            pl.col("setting") == setting,
            pl.col("month") % 2 == 0,
            ~pl.col("logged_matched"),
            pl.col("tier") != "vetoed",
        ).height

    d2, d0m = count("D2_48"), count("D0M_48")
    print(f"even-month unexplained 24 h alarms: D2 {d2}, D0M {d0m} (limit {ALARM_TOLERANCE:.2f}x)")
    alarm_ok = d2 <= ALARM_TOLERANCE * d0m
    if low > 0 and mean >= SUPPORT_AT_LEAST and alarm_ok:
        verdict = "supported"
    elif high < SUPPORT_AT_LEAST:
        verdict = "refuted"
    else:
        verdict = "inconclusive"
    print(f"C2 {mean:+.4f} ({low:+.4f} to {high:+.4f}); alarm count ok {alarm_ok}; {verdict}")


def main() -> None:
    """Print every number the write-up quotes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    args = parser.parse_args()
    out = args.out
    print("matched thresholds:", json.loads((out / "thresholds.json").read_text()))
    veto_z = choose_veto(out=out)
    print(f"chosen veto strength: {vz_tag(veto_z)}")
    data = results(out=out, veto_z=veto_z)
    for step in ("largest_step_z", "donor_step_z"):
        print_recut(data=data, step=step)
    c2 = print_contrasts(out=out, veto_z=veto_z)
    wide = population(data=data, setting="D2_48").join(
        population(data=data, setting="D0M_48"), on="event_id", suffix="_ref"
    )
    print(
        "C2 lost to the veto",
        wide.filter(~pl.col("found"), pl.col("found_ref")).height,
        "gained",
        wide.filter(pl.col("found"), ~pl.col("found_ref")).height,
    )
    print_verdict(out=out, veto_z=veto_z, c2=c2)


if __name__ == "__main__":
    main()
