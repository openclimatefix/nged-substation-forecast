# ruff: noqa: E501
"""Pooled scores, gains and recovery on the real record, from `e00309_real.py`'s unit sums.

Thresholds are chosen on odd-month rows (lowest pooled pinball loss at the 95% quantile in the rows
near a shift) and every contrast is read on even-month rows. Intervals resample whole shifts (near
groups) or whole calendar months (ordinary rows), 20,000 draws, 95%.

    uv run --no-sync python e00309_real_report.py
"""

from __future__ import annotations

import json
from typing import Final

import e00309_data as dat
import e00309_state as st
import numpy as np
import polars as pl

N_DRAWS: Final[int] = 20_000
GROUPS: Final[tuple[str, ...]] = ("near", "during", "after", "ordinary")
METRICS: Final[tuple[str, ...]] = ("pin95", "qs", "abs_median")


def unit_arrays(
    *, table: pl.DataFrame, arm: str, group: str, odd: bool, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Units, row counts and metric sums of one arm, summed over series, per unit."""
    t = table.filter((pl.col("arm") == arm) & (pl.col("group") == group) & (pl.col("odd") == odd))
    t = t.group_by("unit").agg(pl.col("n").sum(), pl.col(metric).sum()).sort("unit")
    return t["unit"].to_numpy(), t["n"].to_numpy(), t[metric].to_numpy()


def pooled(*, table: pl.DataFrame, arm: str, group: str, odd: bool, metric: str) -> float:
    """Row-weighted mean of a metric."""
    _, n, s = unit_arrays(table=table, arm=arm, group=group, odd=odd, metric=metric)
    return float(s.sum() / n.sum()) if n.sum() else float("nan")


def bootstrap_ratio(
    *,
    table: pl.DataFrame,
    num: tuple[str, str],
    den: tuple[str, str] | None,
    group: str,
    metric: str,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Mean gain (first arm minus second) and, given `den`, the recovery ratio, over whole units."""
    units, n, a = unit_arrays(table=table, arm=num[0], group=group, odd=False, metric=metric)
    _, _, b = unit_arrays(table=table, arm=num[1], group=group, odd=False, metric=metric)
    gain = a - b
    idx = rng.integers(0, len(units), size=(N_DRAWS, len(units)))
    mean_draws = gain[idx].sum(axis=1) / n[idx].sum(axis=1)
    out = {
        "n": float(n.sum()),
        "units": len(units),
        "gain": float(gain.sum() / n.sum()),
        "lo": float(np.quantile(mean_draws, 0.025)),
        "hi": float(np.quantile(mean_draws, 0.975)),
        "base": float(a.sum() / n.sum()),
    }
    if den is not None:
        _, _, c = unit_arrays(table=table, arm=den[0], group=group, odd=False, metric=metric)
        _, _, d = unit_arrays(table=table, arm=den[1], group=group, odd=False, metric=metric)
        denom = c - d
        r = gain[idx].sum(axis=1) / denom[idx].sum(axis=1)
        out |= {
            "ratio": float(gain.sum() / denom.sum()),
            "ratio_lo": float(np.quantile(r, 0.025)),
            "ratio_hi": float(np.quantile(r, 0.975)),
        }
    return out


def main() -> None:
    """Print and store every real-record number the page quotes."""
    table = pl.read_parquet(dat.WORK_DIR / "results" / "real_units.parquet")
    rng = np.random.default_rng(309)
    result: dict = {}
    for score, ks in st.THRESHOLDS.items():
        sel = {
            k: pooled(table=table, arm=f"RULE_{score}_{k}", group="near", odd=True, metric="pin95")
            for k in ks
        }
        best = min(sel, key=lambda x: sel[x])
        print(f"{score}: odd-month near pin95 by k {sel}; chosen {best}")
        result[score] = {"selection": sel, "best_k": best}
    print("=== pooled means, even months")
    for group in GROUPS:
        for metric in METRICS:
            row = {
                arm: round(pooled(table=table, arm=arm, group=group, odd=False, metric=metric), 4)
                for arm in (
                    "GAM_ONLY",
                    "K0",
                    *[f"RULE_{s}_{result[s]['best_k']}" for s in st.THRESHOLDS],
                    "ORACLE_ROW",
                    "ORACLE_STATE",
                )
            }
            print(group, metric, row)
    print(
        "=== gains over the better fixed arm and recovery of ORACLE_STATE / ORACLE_ROW, even months"
    )
    rows = []
    for group in GROUPS:
        for metric in METRICS:
            fixed = min(
                ("GAM_ONLY", "K0"),
                key=lambda a: pooled(table=table, arm=a, group=group, odd=False, metric=metric),
            )
            for score in st.THRESHOLDS:
                rule = f"RULE_{score}_{result[score]['best_k']}"
                for oracle in ("ORACLE_STATE", "ORACLE_ROW"):
                    res = bootstrap_ratio(
                        table=table,
                        num=(fixed, rule),
                        den=(fixed, oracle),
                        group=group,
                        metric=metric,
                        rng=rng,
                    )
                    rows.append(
                        {
                            "group": group,
                            "metric": metric,
                            "fixed": fixed,
                            "rule": rule,
                            "oracle": oracle,
                            **res,
                        }
                    )
                    print(
                        f"{group:8s} {metric:10s} fixed={fixed:8s} {rule:14s} vs {oracle:12s}: gain {res['gain']:+.4f} ({res['lo']:+.4f}, {res['hi']:+.4f}) of base {res['base']:.3f}; recovery {res['ratio']:+.3f} ({res['ratio_lo']:+.3f}, {res['ratio_hi']:+.3f}) units={res['units']}"
                    )
    result["rows"] = rows
    print("=== K0 minus GAM_ONLY (positive = GAM_ONLY better), even months")
    for group in GROUPS:
        for metric in ("pin95", "qs"):
            res = bootstrap_ratio(
                table=table, num=("K0", "GAM_ONLY"), den=None, group=group, metric=metric, rng=rng
            )
            result[f"k0_minus_gam_{group}_{metric}"] = res
            print(group, metric, {k: round(v, 4) for k, v in res.items()})
    (dat.WORK_DIR / "results" / "real.json").write_text(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
