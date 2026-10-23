# ruff: noqa: E501, PLR0912, PLR0915
"""Shares of the perfect-knowledge benefit captured by class-restricted oracles, with intervals.

Reads `event_sums.parquet` and `ordinary_sums.parquet` from `e00296_sums.py`, builds the three
frequency scenarios from the logged operations, and prints every number the page quotes. The benefit
of an oracle in one shift is the sum over the shift's rows (within 7 days of the shift) of the K0
score minus the oracle's score. An oracle that knows a set of shifts earns the benefit of PERFECT on
those shifts and zero on the others (checked below against the explicitly run PERFECT_LD and
PERFECT_SMALL arms). Annual benefit weights each shift by the shifts per series-year of its cell.
Intervals resample whole shifts within each (shape, duration, depth) cell, 20,000 draws.

    uv run --no-sync python e00296_report.py
"""

from __future__ import annotations

import json
from typing import Any, Final

import numpy as np
import polars as pl
from e00296_common import LABELLED_PATH, SERIES_IDS, WORK_DIR

RESULTS: Final = WORK_DIR / "results"
SCORES: Final = WORK_DIR / "scores"
N_DRAWS: Final[int] = 20_000
CHUNK: Final[int] = 2_000
DEPTHS: Final[tuple[float, ...]] = (0.05, 0.10, 0.25, 0.50)
BINS: Final[tuple[str, ...]] = ("2-6 h", "6-24 h", "1-3 d", "3-7 d", "7-30 d", "over 30 d")
BIN_EDGES_H: Final[tuple[float, ...]] = (2.0, 6.0, 24.0, 72.0, 168.0, 720.0, 1.0e9)
LD_HOURS: Final[float] = 48.0
LD_DEPTH: Final[float] = 0.10
SHORT_HOURS: Final[float] = 12.0
PHASES: Final[dict[str, tuple[str, ...]]] = {
    "G2 before alarm": ("BEFORE_ALARM",),
    "G3 during": ("DURING",),
    "G4 after": ("AFTER",),
    "G5 near": ("BEFORE_ALARM", "DURING", "AFTER", "MARGIN"),
}
SCENARIOS: Final[dict[str, dict]] = {
    "low": {"mix": (0.10, 0.15, 0.35, 0.40), "short_multiplier": 1.0},
    "central": {"mix": (0.25, 0.25, 0.30, 0.20), "short_multiplier": 1.0},
    "high": {"mix": (0.45, 0.30, 0.18, 0.07), "short_multiplier": 3.0},
}
"""The depth mix is the share of shifts at 5%, 10%, 25%, and 50% of load; `high` also multiplies
the frequency of the two shortest duration bins by 3 because the log lists short shifts least."""
# (headline) contrast definitions: (label, arm prefix, metric)
MASK_METRICS: Final[tuple[str, ...]] = (
    "tw95_nra",
    "tw99_nra",
    "qs_nra",
    "ae_median_nra",
    "ae_mean_nra",
)
ADD_METRICS: Final[tuple[str, ...]] = ("cost0.5", "cost0.75", "cost0.9", "tw95_obs", "qs_obs")
MASK_COST: Final[tuple[str, ...]] = ("cost0.5",)


def logged_frequency() -> tuple[pl.DataFrame, float]:
    """Shifts per series-year in each duration bin, from the logged operations of 2 h to 30 d+."""
    lab = pl.read_parquet(LABELLED_PATH).filter(pl.col("kind") == "real_operation")
    lab = lab.with_columns(
        hours=(pl.col("logged_open_utc") - pl.col("logged_close_utc")).dt.total_minutes() / 60
    )
    span = lab.select(
        (pl.col("logged_close_utc").max() - pl.col("logged_close_utc").min()).dt.total_seconds()
    ).item()
    years = span / (365.25 * 86400)
    sides = np.concatenate(
        [
            lab.filter(pl.col("side_a").is_in(list(SERIES_IDS)))["hours"].to_numpy(),
            lab.filter(pl.col("side_b").is_in(list(SERIES_IDS)))["hours"].to_numpy(),
        ]
    )
    counts, _ = np.histogram(sides, bins=BIN_EDGES_H)
    frame = pl.DataFrame(
        {
            "duration_bin": list(BINS),
            "logged_sides": counts,
            "per_series_year": counts / (years * len(SERIES_IDS)),
        }
    )
    return frame, years


def scenario_frequency(*, base: pl.DataFrame, name: str) -> dict[tuple[str, float], float]:
    """Shifts per series-year of every (duration bin, depth) cell under one scenario."""
    spec = SCENARIOS[name]
    out = {}
    for row in base.iter_rows(named=True):
        multiplier = spec["short_multiplier"] if row["duration_bin"] in BINS[:2] else 1.0
        for depth, share in zip(DEPTHS, spec["mix"], strict=True):
            out[(row["duration_bin"], depth)] = row["per_series_year"] * multiplier * share
    return out


def event_table() -> pl.DataFrame:
    """One row per shift and arm and row group: benefit over K0 for every metric."""
    sums = pl.read_parquet(RESULTS / "event_sums.parquet")
    metric_cols = [c for c in sums.columns if c.startswith("sum_")]
    parts = []
    for group, phases in PHASES.items():
        grouped = (
            sums.filter(pl.col("phase").is_in(phases))
            .group_by("site", "world", "event", "arm")
            .agg(*[pl.col(c).sum() for c in metric_cols])
            .with_columns(group=pl.lit(group))
        )
        parts.append(grouped)
    long = pl.concat(parts)
    info = sums.group_by("site", "world", "event").agg(
        pl.col("event_hours").first(),
        pl.col("fraction").first().alias("depth"),
        pl.col("sign").first(),
        pl.col("shape").first(),
        pl.col("duration_bin").first(),
    )
    k0 = long.filter(pl.col("arm") == "K0").drop("arm")
    k0 = k0.rename({c: f"k0_{c}" for c in metric_cols})
    joined = long.filter(pl.col("arm") != "K0").join(k0, on=["site", "world", "event", "group"])
    joined = joined.with_columns(
        [(pl.col(f"k0_{c}") - pl.col(c)).alias(f"ben_{c[4:]}") for c in metric_cols]
    )
    return joined.join(info, on=["site", "world", "event"])


def shares(
    *,
    benefit: np.ndarray,
    weight: np.ndarray,
    flags: dict[str, np.ndarray],
    cell: np.ndarray,
    seed: int,
) -> dict[str, tuple[float, float, float]]:
    """Share of the total weighted benefit in each flagged class: point, 2.5%, 97.5%."""
    total = float((weight * benefit).sum())
    point = {k: float((weight * benefit * f).sum()) / total for k, f in flags.items()}
    rng = np.random.default_rng(seed)
    cells = [np.flatnonzero(cell == c) for c in np.unique(cell)]
    draws = {k: [] for k in flags}
    for _ in range(N_DRAWS // CHUNK):
        idx = np.concatenate([rng.choice(ix, size=(CHUNK, len(ix))) for ix in cells], axis=1)
        denominator = (weight * benefit)[idx].sum(1)
        for k, f in flags.items():
            draws[k].append((weight * benefit * f)[idx].sum(1) / denominator)
    out = {}
    for k in flags:
        d = np.concatenate(draws[k])
        d = d[np.isfinite(d)]
        if not len(d):
            out[k] = (point[k], float("nan"), float("nan"))
            continue
        out[k] = (point[k], float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)))
    return out


def prepare(*, events: pl.DataFrame, arm: str, group: str, freq: dict) -> dict[str, Any]:
    """The arrays of one arm and row group: benefit columns, weights, cell labels, class columns."""
    sub = events.filter((pl.col("arm") == arm) & (pl.col("group") == group)).sort(
        "site", "world", "event"
    )
    depth = sub["depth"].to_numpy().astype(np.float64).round(2)
    bins = sub["duration_bin"].to_list()
    count: dict[tuple[str, float], int] = {}
    for b, d in zip(bins, depth, strict=True):
        count[(b, float(d))] = count.get((b, float(d)), 0) + 1
    weight = np.array(
        [freq[(b, float(d))] / count[(b, float(d))] for b, d in zip(bins, depth, strict=True)]
    )
    cell = np.array(
        [f"{s}|{b}|{d}" for s, b, d in zip(sub["shape"].to_list(), bins, depth, strict=True)]
    )
    return {
        "sub": sub,
        "hours": sub["event_hours"].to_numpy().astype(np.float64),
        "depth": depth,
        "bins": np.array(bins),
        "site": sub["site"].to_numpy(),
        "sign": sub["sign"].to_numpy(),
        "weight": weight,
        "cell": cell,
    }


def class_flags(*, hours: np.ndarray, depth: np.ndarray) -> dict[str, np.ndarray]:
    """The class of each shift: long and deep, short or small, and the others."""
    ld = (hours >= LD_HOURS) & (depth >= LD_DEPTH - 1e-9)
    return {
        "LD": ld,
        "OTHER": ~ld,
        "SHORT_OR_SMALL": (hours < SHORT_HOURS) | (depth < LD_DEPTH - 1e-9),
        "SHORT": hours < SHORT_HOURS,
        "SMALL": depth < LD_DEPTH - 1e-9,
    }


def fmt(v: tuple[float, float, float]) -> str:
    """A share with its interval, in percent."""
    return f"{v[0] * 100:5.1f}% ({v[1] * 100:5.1f} to {v[2] * 100:5.1f})"


HEADLINES: Final[tuple[tuple[str, str, str], ...]] = (
    ("MASK", "", "tw95_nra"),
    ("ADD", "_ADD", "cost0.5"),
)


def main() -> None:
    """Print every number the page quotes and write the figures' source tables."""
    base, years = logged_frequency()
    print(f"Logged operations: {years:.2f} years, {len(SERIES_IDS)} series")
    print(base)
    events = event_table()
    out: dict = {"frequency_base": base.to_dicts(), "years": years}
    perfect = events.filter((pl.col("arm") == "PERFECT") & (pl.col("group") == "G5 near"))
    census = perfect.group_by("duration_bin", "depth").len()
    print(
        "Shifts scored:",
        perfect.height,
        "; per cell:",
        census["len"].min(),
        "to",
        census["len"].max(),
        "; cells:",
        census.height,
    )
    results: dict = {}
    for scenario in SCENARIOS:
        freq = scenario_frequency(base=base, name=scenario)
        for arms_label, prefix, metrics in (
            ("MASK", "", MASK_METRICS + MASK_COST),
            ("ADD", "_ADD", ADD_METRICS),
        ):
            for group in PHASES:
                a = prepare(events=events, arm=f"PERFECT{prefix}", group=group, freq=freq)
                flags = class_flags(hours=a["hours"], depth=a["depth"])
                for metric in metrics:
                    benefit = a["sub"][f"ben_{metric}"].to_numpy()
                    if not np.isfinite(benefit).all() or abs((a["weight"] * benefit).sum()) < 1e-12:
                        continue
                    results[(scenario, arms_label, group, metric)] = shares(
                        benefit=benefit, weight=a["weight"], flags=flags, cell=a["cell"], seed=296
                    )
    out["shares"] = {"|".join(map(str, k)): v for k, v in results.items()}
    for key, res in results.items():
        scenario, arms_label, group, metric = key
        print(
            f"{scenario:8s} {arms_label:4s} {group:16s} {metric:14s} "
            + " | ".join(
                f"{n} {fmt(v)}" for n, v in res.items() if n in {"LD", "SHORT_OR_SMALL", "OTHER"}
            )
        )

    # Ladder, heat maps, break-even, explicit-arm check, leave-one-site-out, all at the central scenario.
    central = scenario_frequency(base=base, name="central")
    out["frequency_central"] = {f"{b}|{d}": v for (b, d), v in central.items()}
    for scenario in SCENARIOS:
        out[f"frequency_{scenario}"] = {
            f"{b}|{d}": v for (b, d), v in scenario_frequency(base=base, name=scenario).items()
        }
    ladder: dict = {}
    for arms_label, prefix, metric in HEADLINES:
        for scenario in SCENARIOS:
            freq = scenario_frequency(base=base, name=scenario)
            a = prepare(events=events, arm=f"PERFECT{prefix}", group="G5 near", freq=freq)
            benefit = a["sub"][f"ben_{metric}"].to_numpy()
            grid = {}
            for hours_min in (2.0, 6.0, 12.0, 24.0, 48.0, 168.0, 720.0):
                for depth_min in DEPTHS:
                    grid[f"min {hours_min:g} h, {depth_min:.0%}"] = (
                        a["hours"] >= hours_min - 1e-9
                    ) & (a["depth"] >= depth_min - 1e-9)
            base_ld = (a["hours"] >= LD_HOURS) & (a["depth"] >= LD_DEPTH - 1e-9)
            extra = {
                "depth 5% added to LD": (
                    (a["hours"] >= LD_HOURS) & (a["depth"] >= 0.05 - 1e-9)
                ).astype(float)
                - base_ld,
                "duration 6 h added to LD": (
                    (a["hours"] >= 6.0) & (a["depth"] >= LD_DEPTH - 1e-9)
                ).astype(float)
                - base_ld,
                "duration 12 h added to LD": (
                    (a["hours"] >= 12.0) & (a["depth"] >= LD_DEPTH - 1e-9)
                ).astype(float)
                - base_ld,
            }
            res = shares(
                benefit=benefit,
                weight=a["weight"],
                flags={**grid, **extra},
                cell=a["cell"],
                seed=297,
            )
            ladder[f"{scenario}|{arms_label}|{metric}"] = res
            if scenario == "central":
                print(
                    f"\nLadder ({arms_label}, {metric}, central, G5): share of PERFECT benefit by minimum duration and depth"
                )
                for k, v in res.items():
                    print(f"  {k:28s} {fmt(v)}")
    out["ladder"] = dict(ladder)
    # Per-cell benefit, frequency, and share of the ceiling (central), G5, and relative gain per cell.
    cells = []
    for arms_label, prefix, metric in HEADLINES:
        a = prepare(events=events, arm=f"PERFECT{prefix}", group="G5 near", freq=central)
        sub = a["sub"]
        benefit = sub[f"ben_{metric}"].to_numpy()
        k0_score = sub[f"k0_sum_{metric}"].to_numpy()
        total = float((a["weight"] * benefit).sum())
        for shape in ("all", "constant", "demand-shaped"):
            for b in BINS:
                for d in DEPTHS:
                    sel = (a["bins"] == b) & (a["depth"] == d)
                    if shape != "all":
                        sel &= sub["shape"].to_numpy() == shape
                    if not sel.any():
                        continue
                    cells.append(
                        {
                            "arms": arms_label,
                            "metric": metric,
                            "shape": shape,
                            "duration_bin": b,
                            "depth": d,
                            "n": int(sel.sum()),
                            "mean_benefit": float(benefit[sel].mean()),
                            "relative_gain_pct": 100.0
                            * float(benefit[sel].sum() / k0_score[sel].sum()),
                            "share_of_ceiling_pct": 100.0
                            * float((a["weight"] * benefit)[sel].sum() / total),
                            "frequency_per_series_year": central[(b, d)],
                        }
                    )
    cell_frame = pl.DataFrame(cells)
    cell_frame.write_parquet(RESULTS / "cells.parquet")
    show = cell_frame.filter(pl.col("shape") == "all").sort(
        "arms", "metric", "share_of_ceiling_pct", descending=[False, False, True]
    )
    print("\nCells carrying at least 5% of the ceiling on their own (central, both shapes):")
    print(
        show.filter(pl.col("share_of_ceiling_pct") >= 5.0).select(
            "arms",
            "metric",
            "duration_bin",
            "depth",
            "n",
            "mean_benefit",
            "relative_gain_pct",
            "share_of_ceiling_pct",
            "frequency_per_series_year",
        )
    )
    # Explicit PERFECT_LD / PERFECT_SMALL arms against the selection.
    print("\nExplicit arm against selection (central, G5): share of PERFECT benefit")
    for arms_label, prefix, metric in HEADLINES:
        a = prepare(events=events, arm=f"PERFECT{prefix}", group="G5 near", freq=central)
        ld = prepare(events=events, arm=f"PERFECT_LD{prefix}", group="G5 near", freq=central)
        small = prepare(events=events, arm=f"PERFECT_SMALL{prefix}", group="G5 near", freq=central)
        total = float((a["weight"] * a["sub"][f"ben_{metric}"].to_numpy()).sum())
        flags = class_flags(hours=a["hours"], depth=a["depth"])
        sel_ld = (
            float((a["weight"] * a["sub"][f"ben_{metric}"].to_numpy() * flags["LD"]).sum()) / total
        )
        exp_ld = float((ld["weight"] * ld["sub"][f"ben_{metric}"].to_numpy()).sum()) / total
        exp_small = (
            float((small["weight"] * small["sub"][f"ben_{metric}"].to_numpy()).sum()) / total
        )
        print(
            f"  {arms_label} {metric}: selection LD {sel_ld * 100:.1f}%, explicit LD {exp_ld * 100:.1f}%, explicit SMALL {exp_small * 100:.1f}% (sum {(exp_ld + exp_small) * 100:.1f}%)"
        )
        out[f"explicit|{arms_label}|{metric}"] = [sel_ld, exp_ld, exp_small]
    # Break-even frequency multiplier on the shifts outside the long-and-deep class.
    print(
        "\nMultiplier on the frequency of every non-LD cell that brings the LD share down to 70% (central, G5):"
    )
    for arms_label, prefix, metric in HEADLINES:
        a = prepare(events=events, arm=f"PERFECT{prefix}", group="G5 near", freq=central)
        benefit = a["sub"][f"ben_{metric}"].to_numpy()
        flags = class_flags(hours=a["hours"], depth=a["depth"])
        big = float((a["weight"] * benefit * flags["LD"]).sum())
        rest = float((a["weight"] * benefit * flags["OTHER"]).sum())
        mult = big * (1 / 0.7 - 1) / rest if rest > 0 else float("inf")
        print(f"  {arms_label} {metric}: {mult:.1f}x (LD {big:.3g}, other {rest:.3g})")
        out[f"breakeven|{arms_label}|{metric}"] = mult
    # Leave one site out of the headline shares.
    print("\nLong-and-deep share with one site left out (central, G5):")
    for arms_label, prefix, metric in HEADLINES:
        a = prepare(events=events, arm=f"PERFECT{prefix}", group="G5 near", freq=central)
        benefit = a["sub"][f"ben_{metric}"].to_numpy()
        flags = class_flags(hours=a["hours"], depth=a["depth"])
        row = {}
        for site in sorted(set(a["site"])):
            keep = a["site"] != site
            r = shares(
                benefit=benefit[keep],
                weight=a["weight"][keep],
                flags={"LD": flags["LD"][keep], "SHORT_OR_SMALL": flags["SHORT_OR_SMALL"][keep]},
                cell=a["cell"][keep],
                seed=298,
            )
            row[site] = r
            print(
                f"  {arms_label} {metric} without {site}: LD {fmt(r['LD'])}, short or small {fmt(r['SHORT_OR_SMALL'])}"
            )
        out[f"loso|{arms_label}|{metric}"] = row
    # Relative gain of PERFECT over K0 per row group (unweighted, all shifts), for comparison with experiment 283.
    print(
        "\nRelative gain of PERFECT over K0, all scored shifts, by row group (percent of K0 score):"
    )
    for arms_label, prefix, metric in (
        *HEADLINES,
        ("MASK", "", "qs_nra"),
        ("MASK", "", "cost0.5"),
        ("ADD", "_ADD", "tw95_obs"),
    ):
        for group in PHASES:
            sub = events.filter((pl.col("arm") == f"PERFECT{prefix}") & (pl.col("group") == group))
            gain = 100.0 * float(sub[f"ben_{metric}"].sum()) / float(sub[f"k0_sum_{metric}"].sum())
            print(f"  {arms_label} {metric} {group}: {gain:.2f}%")
    # Class shares by shape of the shift.
    print("\nLong-and-deep share by shape of the shift (central, G5):")
    for arms_label, prefix, metric in HEADLINES:
        a = prepare(events=events, arm=f"PERFECT{prefix}", group="G5 near", freq=central)
        benefit = a["sub"][f"ben_{metric}"].to_numpy()
        flags = class_flags(hours=a["hours"], depth=a["depth"])
        for shape in ("constant", "demand-shaped"):
            keep = a["sub"]["shape"].to_numpy() == shape
            r = shares(
                benefit=benefit[keep],
                weight=a["weight"][keep],
                flags={"LD": flags["LD"][keep], "SHORT_OR_SMALL": flags["SHORT_OR_SMALL"][keep]},
                cell=a["cell"][keep],
                seed=300,
            )
            print(
                f"  {arms_label} {metric} {shape}: LD {fmt(r['LD'])}, short or small {fmt(r['SHORT_OR_SMALL'])}"
            )
            out[f"shape|{arms_label}|{metric}|{shape}"] = r
    # Absolute scores and tail calibration.
    print(
        "\nAbsolute scores per row (points of effective capacity) and tail coverage, all shifts, by row group:"
    )
    sums_all = pl.read_parquet(RESULTS / "event_sums.parquet")
    for group, phases in PHASES.items():
        sub = (
            sums_all.filter(pl.col("phase").is_in(phases))
            .group_by("arm")
            .agg(
                pl.all()
                .exclude(
                    "site",
                    "world",
                    "event",
                    "phase",
                    "arm",
                    "event_hours",
                    "fraction",
                    "sign",
                    "shape",
                    "duration_bin",
                )
                .sum()
            )
        )
        for row in sub.sort("arm").iter_rows(named=True):
            if row["arm"] not in {"K0", "PERFECT", "PERFECT_LD", "PERFECT_SMALL", "PERFECT_ADD"}:
                continue
            n = row["n_qs_nra"]
            target = "obs" if row["arm"].endswith("_ADD") else "nra"
            print(
                f"  {group:16s} {row['arm']:14s} CRPS {row[f'sum_qs_{target}'] / n:6.3f} MAE-median {row[f'sum_ae_median_{target}'] / n:6.3f} tw95 {row[f'sum_tw95_{target}'] / n:6.3f}"
                f" below q05 {100 * row[f'sum_low_{target}'] / n:5.1f}% above q95 {100 * row[f'sum_high_{target}'] / n:5.1f}%"
                f" pinball05 {row[f'sum_pin05_{target}'] / n:6.3f} pinball95 {row[f'sum_pin95_{target}'] / n:6.3f} ({target})"
            )
    # Cost by sign of the shift.
    print(
        "\nCost0.5 benefit of PERFECT_ADD by sign of the shift (all shifts, percent of K0 score, G5):"
    )
    sub = events.filter((pl.col("arm") == "PERFECT_ADD") & (pl.col("group") == "G5 near"))
    for sign in (1.0, -1.0):
        part = sub.filter(pl.col("sign") == sign)
        print(
            f"  sign {sign:+.0f}: {100 * float(part['ben_cost0.5'].sum()) / float(part['k0_sum_cost0.5'].sum()):.2f}% (n={part.height})"
        )
    # Ordinary rows: month-resampled K0 against PERFECT.
    ordinary = pl.read_parquet(RESULTS / "ordinary_sums.parquet")
    rng = np.random.default_rng(299)
    months = np.sort(ordinary["month"].unique().to_numpy())
    for metric in ("qs_nra", "tw95_nra"):
        wide = (
            ordinary.group_by("month", "arm")
            .agg(pl.col(f"sum_{metric}").sum(), pl.col("n_qs_nra").sum())
            .pivot(on="arm", index="month", values=[f"sum_{metric}", "n_qs_nra"])
        )
        wide = wide.sort("month")
        k0 = wide[f"sum_{metric}_K0"].to_numpy()
        pf = wide[f"sum_{metric}_PERFECT"].to_numpy()
        point = 100.0 * (k0.sum() - pf.sum()) / k0.sum()
        boot = []
        for _ in range(N_DRAWS):
            idx = rng.integers(0, len(k0), len(k0))
            boot.append(100.0 * (k0[idx].sum() - pf[idx].sum()) / k0[idx].sum())
        lo, hi = np.quantile(boot, [0.025, 0.975])
        print(
            f"\nOrdinary rows (G1), {metric}: PERFECT gain {point:.3f}% ({lo:.3f} to {hi:.3f}); months {len(months)}; rows {int(wide['n_qs_nra_K0'].sum())}"
        )
        out[f"g1|{metric}"] = [point, float(lo), float(hi)]
    # Median forecast minus mean forecast per site (checks).
    print(
        "\nMAE of the median minus MAE of the mean forecast, per site, K0 and PERFECT (all event rows, points):"
    )
    sums = pl.read_parquet(RESULTS / "event_sums.parquet").filter(
        pl.col("arm").is_in(["K0", "PERFECT"])
    )
    agg = sums.group_by("site", "arm").agg(
        pl.col("sum_ae_median_nra").sum(), pl.col("sum_ae_mean_nra").sum(), pl.col("n_qs_nra").sum()
    )
    for row in agg.sort("site", "arm").iter_rows(named=True):
        print(
            f"  {row['site']} {row['arm']}: {(row['sum_ae_median_nra'] - row['sum_ae_mean_nra']) / row['n_qs_nra']:+.4f}"
        )
    # Mean-forecast error beside the quantile-score contrast.
    print(
        "\nPERFECT gain over K0 on the MAE of the mean and of the median, by row group (percent, all shifts):"
    )
    for group in PHASES:
        sub = events.filter((pl.col("arm") == "PERFECT") & (pl.col("group") == group))
        print(
            "  "
            + group
            + ": "
            + ", ".join(
                f"{m} {100 * float(sub[f'ben_{m}'].sum()) / float(sub[f'k0_sum_{m}'].sum()):.2f}%"
                for m in ("qs_nra", "ae_mean_nra", "ae_median_nra")
            )
        )
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "report_numbers.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
