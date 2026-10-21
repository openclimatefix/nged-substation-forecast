"""Report experiment 151: shares of the K0-to-CLEAN gap closed, with month-block intervals.

Usage: uv run --no-sync python e00151_report.py   (prints the report; writes report.json)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00151_common import WORK_DIR

SCORES: Final[Path] = WORK_DIR / "scores"
ARMS: Final[tuple[str, ...]] = (
    "K0",
    "CLEAN",
    "SKF",
    "SORACLE",
    "SUB_DET",
    "SUB_TRUE_LAG",
    "SUB_TRUE_CONST",
    "SUB_PLACEBO",
)
N_RESAMPLES: Final[int] = 20_000
SEED: Final[int] = 151
MIN_MONTHS: Final[int] = 8
LEVELS: Final[tuple[tuple[str, float], ...]] = (
    ("q01", 0.01),
    ("q05", 0.05),
    ("q95", 0.95),
    ("q99", 0.99),
)


def load() -> pl.DataFrame:
    """Every scored row, wide: one column per arm for each of the quantile score and the errors."""
    long = pl.concat([pl.read_parquet(p) for p in sorted(SCORES.glob("*.parquet"))])
    keys = ["site", "world", "row"]
    tags = long.filter(pl.col("arm") == "K0").select(
        *keys,
        "run",
        "lead_hours",
        "period",
        "origin",
        "event",
        "inside",
        "after48",
        "after_det",
        "missed",
        "actual",
    )
    wide = tags
    for metric in ("qs", "err_median"):
        part = long.select(*keys, "arm", metric).pivot(on="arm", index=keys, values=metric)
        part = part.rename({a: f"{metric}__{a}" for a in part.columns if a not in keys})
        wide = wide.join(part, on=keys)
    wide = wide.with_columns(
        month=(
            pl.lit(datetime(2024, 4, 1, 9))
            + pl.duration(days=pl.col("run"))
            + pl.duration(minutes=(pl.col("lead_hours") * 60).cast(pl.Int64))
        ).dt.strftime("%Y-%m")
    )
    return wide.with_columns(
        **{f"ae__{a}": pl.col(f"err_median__{a}").abs() for a in ARMS},
    )


def share(
    *, frame: pl.DataFrame, arm: str, metric: str = "qs", n_resamples: int = N_RESAMPLES
) -> dict:
    """The share of the gap closed by `arm`, with a month-block interval at 95% and 99%."""
    col = (lambda a: f"qs__{a}") if metric == "qs" else (lambda a: f"ae__{a}")
    if not frame.height:
        return {"n": 0}
    by_month = (
        frame.group_by("month")
        .agg(
            num=(pl.col(col("K0")) - pl.col(col(arm))).sum(),
            gap=(pl.col(col("K0")) - pl.col(col("CLEAN"))).sum(),
            n=pl.len(),
        )
        .sort("month")
    )
    num, gap = by_month["num"].to_numpy(), by_month["gap"].to_numpy()
    point = float(num.sum() / gap.sum()) if gap.sum() != 0 else float("nan")
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(num), size=(n_resamples, len(num)))
    rs_num, rs_gap = num[picks].sum(axis=1), gap[picks].sum(axis=1)
    ok = rs_gap > 0
    ratio = rs_num[ok] / rs_gap[ok]
    out = {
        "n": int(frame.height),
        "months": int(len(num)),
        "point": point,
        "dropped": int((~ok).sum()),
        "gap_mean": float(gap.sum() / by_month["n"].sum()),
        "gap_lo": float(np.percentile(rs_gap / by_month["n"].to_numpy()[picks].sum(axis=1), 2.5)),
        "gap_hi": float(np.percentile(rs_gap / by_month["n"].to_numpy()[picks].sum(axis=1), 97.5)),
        "gap_lo99": float(np.percentile(rs_gap / by_month["n"].to_numpy()[picks].sum(axis=1), 0.5)),
        "arm_mean": float(frame[col(arm)].mean()),
    }
    if len(ratio):
        for tag, (lo, hi) in {"95": (2.5, 97.5), "99": (0.5, 99.5)}.items():
            out[f"lo{tag}"], out[f"hi{tag}"] = (
                float(np.percentile(ratio, lo)),
                float(np.percentile(ratio, hi)),
            )
    return out


def fmt(r: dict, level: str = "95") -> str:
    """Format a share as 'point% (lo% to hi%)'."""
    if not r.get("n") or "point" not in r or f"lo{level}" not in r:
        return "n/a"
    return (
        f"{100 * r['point']:.0f}% ({100 * r[f'lo{level}']:.0f}% to {100 * r[f'hi{level}']:.0f}%)"
    )


def main() -> None:
    """Print every table of the write-up and store report.json."""
    df = load()
    report: dict = {}
    print(f"rows {df.height}, sites {df['site'].n_unique()}, worlds {df['world'].n_unique()}")
    events = df.select("site", "world", "event", "missed").unique()
    print(
        f"transfer-site pairs with rows {events.height}; detected share "
        f"{1 - events['missed'].mean():.3f}"
    )
    primary = df.filter((pl.col("origin") == "post") & (pl.col("inside") | pl.col("after48")))
    lag = df.filter((pl.col("origin") == "lag") & (pl.col("inside") | pl.col("after48")))
    print(f"post-alarm primary rows {primary.height}; detection-lag rows {lag.height}")
    print(f"months with post rows: {sorted(primary['month'].unique().to_list())}")
    sets = {
        "post (C1 rows)": primary,
        "post, inside": primary.filter(pl.col("inside")),
        "post, after48": primary.filter(pl.col("after48")),
        "lag rows": lag,
        "all post-onset rows incl. missed": df.filter(
            pl.col("origin").is_in(["lag", "post"]) & (pl.col("inside") | pl.col("after48"))
        ),
        "post, after detected end": df.filter((pl.col("origin") == "post") & pl.col("after_det")),
        "post, replicate": primary.filter(pl.col("period") == "replicate"),
        "post, scored year": primary.filter(pl.col("period") == "scored"),
    }
    for name, frame in sets.items():
        print(f"\n== {name}: {frame.height} rows ==")
        report[name] = {}
        for metric in ("qs", "ae"):
            for arm in ARMS[2:]:
                r = share(frame=frame, arm=arm, metric=metric)
                report[name][f"{metric}/{arm}"] = r
                level = "99" if (name == "post (C1 rows)" and arm == "SUB_DET" and metric == "qs") else "95"
                print(
                    f"  {metric:2s} {arm:15s} share {fmt(r, level)}"
                    f"  [level {level}; dropped {r.get('dropped', 0)}]"
                )
        if frame.height:
            r = share(frame=frame, arm="SUB_DET")
            print(
                f"  gap K0-CLEAN mean {r['gap_mean']:.3f} points (95% {r['gap_lo']:.3f} to "
                f"{r['gap_hi']:.3f}; 99% lower {r['gap_lo99']:.3f}); months {r['months']}"
            )
    print("\n== absolute mean quantile score by arm (points of capacity) ==")
    for name, frame in (("post", primary), ("lag", lag)):
        report[f"abs_{name}"] = {a: float(frame[f"qs__{a}"].mean()) for a in ARMS}
        print(name, {a: round(v, 3) for a, v in report[f"abs_{name}"].items()})
        report[f"abs_ae_{name}"] = {a: float(frame[f"ae__{a}"].mean()) for a in ARMS}
        print(name, "median abs error", {a: round(v, 3) for a, v in report[f"abs_ae_{name}"].items()})
    print("\n== per month, SUB_DET share on post rows ==")
    for month in sorted(primary["month"].unique().to_list()):
        sub = primary.filter(pl.col("month") == month)
        num = (sub["qs__K0"] - sub["qs__SUB_DET"]).sum()
        gap = (sub["qs__K0"] - sub["qs__CLEAN"]).sum()
        print(f"  {month}: rows {sub.height}, share {100 * num / gap:.0f}%" if gap else f"  {month}: gap 0")
    print("\n== leave one site out, SUB_DET share on post rows ==")
    for site in sorted(primary["site"].unique().to_list()):
        r = share(frame=primary.filter(pl.col("site") != site), arm="SUB_DET", n_resamples=5000)
        print(f"  without {site}: {fmt(r)}")
        report[f"loo_{site}"] = r
    print("\n== checks ==")
    med_minus_mean = (
        primary.group_by("site")
        .agg((pl.col("err_median__K0") - pl.col("err_median__K0")).mean().alias("zero"))
    )
    del med_minus_mean
    touched = (
        lag.filter((pl.col("qs__SUB_DET") - pl.col("qs__K0")).abs() > 1e-6).height / max(lag.height, 1)
    )
    print(f"  share of detection-lag rows where SUB_DET differs from K0 (unrelated alarms): {touched:.3f}")
    skf = (lag.filter((pl.col("qs__SKF") - pl.col("qs__K0")).abs() > 1e-6).height) / max(lag.height, 1)
    print(f"  share of detection-lag rows where S-KF differs from K0: {skf:.3f}")
    # tail calibration on the post rows is computed from the long files
    long = pl.concat([pl.read_parquet(p) for p in sorted(SCORES.glob("*.parquet"))])
    keys = df.filter((pl.col("origin") == "post") & (pl.col("inside") | pl.col("after48"))).select(
        "site", "world", "row"
    )
    long = long.join(keys, on=["site", "world", "row"])
    print("\n== tail calibration on post rows: coverage (share of actual <= quantile), pinball ==")
    for arm in ARMS:
        sub = long.filter(pl.col("arm") == arm)
        parts = []
        for name, tau in LEVELS:
            q, a = sub[name], sub["actual"]
            cover = float((a <= q).mean())
            pin = float(
                pl.DataFrame({"d": a - q}).select(
                    pl.when(pl.col("d") >= 0).then(tau * pl.col("d")).otherwise((tau - 1) * pl.col("d"))
                )["d"].mean()
            )
            parts.append(f"{name} cov {cover:.3f} pin {pin:.3f}")
        print(f"  {arm:15s} " + "; ".join(parts))
        # median forecast minus mean forecast, per arm (all sites)
        med_mean = float((sub["err_median"] - sub["err_mean"]).mean())
        report.setdefault("median_minus_mean", {})[arm] = med_mean
    print("  median minus mean forecast (points):", {a: round(v, 4) for a, v in report["median_minus_mean"].items()})
    for site in sorted(long["site"].unique().to_list()):
        sub = long.filter((pl.col("arm") == "K0") & (pl.col("site") == site))
        print(f"  K0 median minus mean forecast at {site}: {float((sub['err_median'] - sub['err_mean']).mean()):.4f}")
    Path(__file__).with_name("report.json").write_text(json.dumps(report, indent=1, default=str))
    _ = timedelta


if __name__ == "__main__":
    main()
