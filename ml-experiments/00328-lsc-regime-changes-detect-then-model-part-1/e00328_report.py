"""Stage 2 report: quantile score, error of the median, and tail calibration by arm and row group.

The quantile score is twice the mean pinball loss over 11 levels (0.01, 0.05, 0.10, 0.25, 0.40,
0.50, 0.60, 0.75, 0.90, 0.95, 0.99), in points of effective capacity. Intervals resample whole
injected worlds (one Primary, one onset, one step size) or whole calendar months of the real
records, never half-hours. Four contrasts against arm B are pre-registered (C1, C2, D1, D2), so
every interval is 98.75% (the Bonferroni level for four contrasts at 5%).

    PYTHONPATH=ml-experiments/00328-* uv run --no-sync python ml-experiments/00328-*/e00328_report.py
"""

from __future__ import annotations

import glob

import e00328_common as c
import numpy as np
import polars as pl

ARMS = ("A", "Ar", "B", "C1", "C2", "C2o", "D1", "D2", "D2o", "C1t", "C2t", "C2ot", "D1t", "D2t", "D2ot")
CONTRASTS = ("C1", "C2", "D1", "D2", "C2t", "D2t")
GROUPS_90 = ("d0_30", "d30_90")
LOW, HIGH = 0.625, 99.375
N_BOOT = 4000


def load(*, pattern: str) -> pl.DataFrame:
    """Read every parquet file matching the pattern under the model folder."""
    return pl.concat([pl.read_parquet(p) for p in sorted(glob.glob(str(c.WORK_DIR / "model" / pattern)))])


def score(*, frame: pl.DataFrame, arm: str, groups: tuple[str, ...]) -> float:
    """Return the pooled quantile score of one arm over the given row groups."""
    f = frame.filter((pl.col("arm") == arm) & pl.col("group").is_in(groups))
    return 2.0 * f["pinball_sum"].sum() / max(f["n"].sum(), 1)


def clustered_difference(
    *, frame: pl.DataFrame, cluster: str, arm_a: str, arm_b: str, groups: tuple[str, ...], stat: str = "pinball_sum"
) -> tuple[float, float, float]:
    """Return arm_a minus arm_b (pooled over rows) and its interval, resampling clusters."""
    f = frame.filter(pl.col("group").is_in(groups))
    a = f.filter(pl.col("arm") == arm_a).group_by(cluster).agg(s=pl.col(stat).sum(), n=pl.col("n").sum()).sort(cluster)
    b = f.filter(pl.col("arm") == arm_b).group_by(cluster).agg(s=pl.col(stat).sum(), n=pl.col("n").sum()).sort(cluster)
    j = a.join(b, on=cluster, suffix="_b")
    scale = 2.0 if stat == "pinball_sum" else 1.0
    sa, na, sb, nb = (j[k].to_numpy().astype(float) for k in ("s", "n", "s_b", "n_b"))
    point = scale * (sa.sum() / na.sum() - sb.sum() / nb.sum())
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(j), len(j))
        draws.append(scale * (sa[i].sum() / na[i].sum() - sb[i].sum() / nb[i].sum()))
    lo, hi = np.percentile(draws, [LOW, HIGH])
    return float(point), float(lo), float(hi)


def main() -> None:
    """Print the report."""
    real = load(pattern="real_*.parquet")
    inj = load(pattern="inj_*.parquet").with_columns(cluster=pl.col("site") + "|" + pl.col("world"))
    inj = inj.with_columns(size=pl.col("world").str.extract(r"step_([0-9.]+)_").cast(pl.Float64))
    print("== Injected permanent steps (even-month onsets, 14 Primaries), quantile score by arm and row group")
    for size in c.STEP_SIZES:
        sub = inj.filter(pl.col("size") == size)
        print(f"\n-- step of {size:.0%} of load; {sub['cluster'].n_unique()} worlds")
        for groups, label in ((("d0_30",), "days 0-30"), (GROUPS_90, "first 90 days"), (("d90_180",), "days 90-180"), (("d180p",), "after day 180"), (("d0_30", "d30_90", "d90_180", "d180p"), "all post-onset rows")):
            row = {arm: score(frame=sub, arm=arm, groups=groups) for arm in ARMS}
            print(f"{label:22s} " + "  ".join(f"{a}={v:.3f}" for a, v in row.items()))
        for groups, label in ((GROUPS_90, "first 90 days"), (("d90_180", "d180p"), "after day 90")):
            for arm in CONTRASTS:
                m, lo, hi = clustered_difference(frame=sub, cluster="cluster", arm_a="B", arm_b=arm, groups=groups)
                m99, l99, h99 = clustered_difference(frame=sub, cluster="cluster", arm_a="B", arm_b=arm, groups=groups, stat="p99_sum")
                cov = {k: sub.filter((pl.col("arm") == k) & pl.col("group").is_in(groups))["above99"].sum() / sub.filter((pl.col("arm") == k) & pl.col("group").is_in(groups))["n"].sum() for k in ("B", arm)}
                print(f"  B minus {arm:3s} {label:14s}: quantile score {m:+.3f} ({lo:+.3f} to {hi:+.3f}); q99 pinball*2 {m99:+.4f} ({l99:+.4f} to {h99:+.4f}); q99 exceedance B {cov['B']:.4f} vs {arm} {cov[arm]:.4f} (nominal 0.0100)")
        mae = {arm: sub.filter((pl.col("arm") == arm) & pl.col("group").is_in(GROUPS_90))["abs_sum"].sum() / sub.filter((pl.col("arm") == arm) & pl.col("group").is_in(GROUPS_90))["n"].sum() for arm in ARMS}
        print("  mean absolute error of the median, first 90 days: " + "  ".join(f"{a}={v:.3f}" for a, v in mae.items()))
    print("\n== D8 real record (break 2023-08-15)")
    d8 = real.filter(pl.col("site") == "D8")
    for g in ("pre", "d0_30", "d30_90", "d90_180", "d180p"):
        row = {arm: score(frame=d8, arm=arm, groups=(g,)) for arm in ARMS}
        n = d8.filter((pl.col("arm") == "A") & (pl.col("group") == g))["n"].sum()
        print(f"{g:8s} n={n:6d} " + "  ".join(f"{a}={v:.3f}" for a, v in row.items()))
    for arm in CONTRASTS:
        for groups, label in ((("d0_30", "d30_90", "d90_180", "d180p"), "all rows after the break"), (GROUPS_90, "first 90 days")):
            m, lo, hi = clustered_difference(frame=d8, cluster="month", arm_a="B", arm_b=arm, groups=groups)
            print(f"  D8: B minus {arm:3s} {label}: {m:+.3f} ({lo:+.3f} to {hi:+.3f})")
    after = d8.filter(pl.col("group") != "pre")
    print("  D8 after the break, q01 and q99 exceedance by arm (nominal 0.01):")
    for arm in ARMS:
        f = after.filter(pl.col("arm") == arm)
        print(f"    {arm:3s} below q01 {f['below01'].sum() / f['n'].sum():.4f}  above q99 {f['above99'].sum() / f['n'].sum():.4f}")
    print("\n== Real records of the other 14 Primaries (no injection): rows scored since 2021-09; false alarms act")
    others = real.filter(pl.col("site") != "D8")
    base = {arm: score(frame=others, arm=arm, groups=("all",)) for arm in ARMS}
    print("  " + "  ".join(f"{a}={v:.3f}" for a, v in base.items()))
    for arm in ("B", *CONTRASTS, "C2o"):
        m, lo, hi = clustered_difference(frame=others, cluster="month", arm_a=arm, arm_b="A", groups=("all",))
        print(f"  {arm:3s} minus A: {m:+.3f} ({lo:+.3f} to {hi:+.3f})  [margin 0.01]")
    print("  per Primary, C2 minus A and D2 minus A:")
    for site in sorted(others["site"].unique()):
        s = others.filter(pl.col("site") == site)
        print(f"    {site}: C2 {score(frame=s, arm='C2', groups=('all',)) - score(frame=s, arm='A', groups=('all',)):+.3f}, D2 {score(frame=s, arm='D2', groups=('all',)) - score(frame=s, arm='A', groups=('all',)):+.3f}, B {score(frame=s, arm='B', groups=('all',)) - score(frame=s, arm='A', groups=('all',)):+.3f}")


if __name__ == "__main__":
    main()
