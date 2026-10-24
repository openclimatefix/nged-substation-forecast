# ruff: noqa: E501
"""Contrasts of the injected-world curtailment cost, with intervals that resample whole worlds.

uv run --no-sync python e00332_inj_report.py
"""

from __future__ import annotations

import json
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final = ML_EXPERIMENTS_DATA_DIR / "00332-lsc-curtailment-cost-of-label-free-mask"
CONTRASTS: Final = (
    ("A", "C"),
    ("B", "C"),
    ("D", "C"),
    ("A", "B"),
    ("A_plus", "C"),
    ("B_plus", "C"),
    ("D_plus", "C"),
)
N_DRAWS: Final = 20_000
POWER: Final = 2.8
GROUPS: Final = (
    "DA_during_after",
    "G3_during",
    "G4_after",
    "G2_before_alarm",
    "G5_near",
    "G1_ordinary",
    "G0_all",
)


def per_world(
    *, long: pl.DataFrame, x: str, y: str, sites: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-world sums of the cost difference x minus y, of the cost of y, and of the row count."""
    if not {x, y} <= set(long["arm"].unique().to_list()):
        empty = np.zeros(0)
        return empty, empty, empty
    wide = (
        long.filter(pl.col("site").is_in(sites), pl.col("arm").is_in([x, y]))
        .pivot(on="arm", index=["world", "site", "n"], values="cost")
        .drop_nulls()
        .group_by("world")
        .agg(
            diff=(pl.col(x) - pl.col(y)).sum(),
            base=pl.col(y).sum(),
            n=pl.col("n").sum(),
        )
        .sort("world")
    )
    return wide["diff"].to_numpy(), wide["base"].to_numpy(), wide["n"].to_numpy().astype(float)


def interval(*, diff: np.ndarray, n: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    """Mean difference per half-hour with a 95% interval from resampling whole worlds."""
    k = len(diff)
    draws = rng.integers(0, k, size=(N_DRAWS, k))
    boot = diff[draws].sum(axis=1) / n[draws].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "est": float(diff.sum() / n.sum()),
        "lo": float(lo),
        "hi": float(hi),
        "se": float(boot.std()),
        "mde": float(POWER * boot.std()),
    }


def main() -> None:
    """Print and store every contrast."""
    long = pl.read_parquet(OUT / "inj_cost_long.parquet")
    all_sites = sorted(long["site"].unique().to_list())
    rng = np.random.default_rng(332)
    result: dict[str, dict] = {}
    for season in ("all", "winter"):
        for target in ("obs", "nra"):
            for fraction in (0.5, 0.75, 0.9):
                sub = long.filter(
                    pl.col("season") == season,
                    pl.col("target") == target,
                    pl.col("fraction") == fraction,
                )
                for group in GROUPS:
                    g = sub.filter(pl.col("group") == group)
                    for x, y in CONTRASTS:
                        diff, base, n = per_world(long=g, x=x, y=y, sites=all_sites)
                        if len(n) == 0 or n.sum() == 0:
                            continue
                        row = interval(diff=diff, n=n, rng=rng)
                        row["base_per_half_hour"] = float(base.sum() / n.sum())
                        row["relative_percent"] = (
                            100.0 * row["est"] / row["base_per_half_hour"]
                            if row["base_per_half_hour"] > 0
                            else float("nan")
                        )
                        row["n_half_hours"] = float(n.sum())
                        key = f"{season}|{target}|f{fraction}|{group}|{x}-{y}"
                        result[key] = row
                        if fraction == 0.5 and season == "all":
                            print(
                                f"{target:3s} {group:16s} {x}-{y}: {row['est']:+9.1f} GBP/half-hour "
                                f"[{row['lo']:+9.1f}, {row['hi']:+9.1f}] ({row['relative_percent']:+5.1f}% of {y}) "
                                f"mde {row['mde']:.1f} n {int(row['n_half_hours'])}"
                            )
    print("\nLeave-one-Primary-out, deciding contrast A-C, DA, obs, f0.5")
    g = long.filter(
        pl.col("season") == "all",
        pl.col("target") == "obs",
        pl.col("fraction") == 0.5,
        pl.col("group") == "DA_during_after",
    )
    for drop in [None, *all_sites]:
        sites = [s for s in all_sites if s != drop]
        diff, _, n = per_world(long=g, x="A", y="C", sites=sites)
        row = interval(diff=diff, n=n, rng=rng)
        result[f"loo|{drop}"] = row
        print(f"  without {drop}: {row['est']:+9.1f} [{row['lo']:+9.1f}, {row['hi']:+9.1f}]")
    print("\nPer Primary, A-C, DA, obs, f0.5")
    for s in all_sites:
        diff, base, n = per_world(long=g, x="A", y="C", sites=[s])
        row = interval(diff=diff, n=n, rng=rng)
        result[f"site|{s}"] = row
        print(
            f"  {s}: {row['est']:+9.1f} [{row['lo']:+9.1f}, {row['hi']:+9.1f}] ({100 * row['est'] / (base.sum() / n.sum()):+.1f}%)"
        )
    (OUT / "inj_contrasts.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
