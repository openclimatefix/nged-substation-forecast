# type: ignore
# ruff: noqa: PGH003, DTZ001, PLR0912, PLR0915, E501
"""Report experiment 153: shares of the oracle gap closed by the latent-NRA filter, with intervals.

Usage: uv run --no-sync python e00153_report.py   (prints every number the write-up quotes)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00153_common import WORK_DIR

SCORES: Final[Path] = WORK_DIR / "scores_v2"
PLACEBO: Final[Path] = WORK_DIR / "placebo_v2"
RATIOS: Final[tuple[int, ...]] = (10, 100, 1000, 10000)
N_RESAMPLES: Final[int] = 20_000
SEED: Final[int] = 153
LEVELS: Final[tuple[tuple[str, float], ...]] = (
    ("q01", 0.01),
    ("q05", 0.05),
    ("q95", 0.95),
    ("q99", 0.99),
)


def _month(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        month=(
            pl.lit(datetime(2024, 4, 1, 9))
            + pl.duration(days=pl.col("run"))
            + pl.duration(minutes=(pl.col("lead_hours") * 60).cast(pl.Int64))
        ).dt.strftime("%Y-%m")
    )


def load(directory: Path) -> pl.DataFrame:
    """Every scored row, wide: one column per arm for the quantile score and the median error."""
    long = pl.concat([pl.read_parquet(p) for p in sorted(directory.glob("*.parquet"))])
    keys = ["site", "world", "row"]
    tag_columns = [
        c
        for c in (
            "period",
            "origin",
            "event",
            "inside",
            "after48",
            "after_det",
            "missed",
            "duration_days",
            "in_alarm",
        )
        if c in long.columns
    ]
    wide = long.filter(pl.col("arm") == "K0").select(
        *keys, "run", "lead_hours", "actual", *tag_columns
    )
    for metric in ("qs", "err_median", "err_mean", "q01", "q05", "q95", "q99"):
        part = long.select(*keys, "arm", metric).pivot(on="arm", index=keys, values=metric)
        part = part.rename({a: f"{metric}__{a}" for a in part.columns if a not in keys})
        wide = wide.join(part, on=keys)
    return _month(wide)


def share(*, frame: pl.DataFrame, arm: str, reference: str, metric: str = "qs") -> dict[str, float]:
    """Share of the gap K0 minus `reference` closed by `arm`, with month-block intervals."""
    if not frame.height:
        return {"n": 0}
    col = (lambda a: f"qs__{a}") if metric == "qs" else (lambda a: f"ae__{a}")
    frame = frame.with_columns(
        **{f"ae__{a}": pl.col(f"err_median__{a}").abs() for a in ARMS_IN(frame)}
    )
    by_month = (
        frame.group_by("month")
        .agg(
            num=(pl.col(col("K0")) - pl.col(col(arm))).sum(),
            gap=(pl.col(col("K0")) - pl.col(col(reference))).sum(),
            n=pl.len(),
        )
        .sort("month")
    )
    num, gap = by_month["num"].to_numpy(), by_month["gap"].to_numpy()
    picks = np.random.default_rng(SEED).integers(0, len(num), size=(N_RESAMPLES, len(num)))
    rs_num, rs_gap = num[picks].sum(axis=1), gap[picks].sum(axis=1)
    ok = rs_gap > 0
    ratio = rs_num[ok] / rs_gap[ok]
    out = {
        "n": float(frame.height),
        "months": float(len(num)),
        "point": float(num.sum() / gap.sum()) if gap.sum() != 0 else float("nan"),
        "gap_mean": float(gap.sum() / frame.height),
        "dropped": float((~ok).sum()),
    }
    for tag, (lo, hi) in {"95": (2.5, 97.5), "99": (0.5, 99.5)}.items():
        if not len(ratio):
            break
        out[f"lo{tag}"], out[f"hi{tag}"] = (
            float(np.percentile(ratio, lo)),
            float(np.percentile(ratio, hi)),
        )
    return out


def ARMS_IN(frame: pl.DataFrame) -> list[str]:  # noqa: N802
    """Arm names present in a wide frame."""
    return [c.removeprefix("err_median__") for c in frame.columns if c.startswith("err_median__")]


def fmt(r: dict[str, float], level: str = "95") -> str:
    """Format a share as 'point% (lo% to hi%)'."""
    if not r.get("n") or f"lo{level}" not in r or not np.isfinite(r["point"]):
        return "n/a"
    return f"{100 * r['point']:.0f}% ({100 * r[f'lo{level}']:.0f}% to {100 * r[f'hi{level}']:.0f}%)"


def month_mean_difference(*, frame: pl.DataFrame, a: str, b: str, metric: str = "qs") -> dict:
    """Mean of metric(a) - metric(b) with a month-block 95% interval."""
    by_month = (
        frame.group_by("month")
        .agg(s=(pl.col(f"{metric}__{a}") - pl.col(f"{metric}__{b}")).sum(), n=pl.len())
        .sort("month")
    )
    s, n = by_month["s"].to_numpy(), by_month["n"].to_numpy()
    picks = np.random.default_rng(SEED).integers(0, len(s), size=(N_RESAMPLES, len(s)))
    draws = s[picks].sum(axis=1) / n[picks].sum(axis=1)
    return {
        "mean": float(s.sum() / n.sum()),
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
        "n": int(n.sum()),
    }


def pinball(*, frame: pl.DataFrame, arm: str, level: float, name: str) -> tuple[float, float]:
    """Coverage and pinball loss of one quantile of `arm`."""
    q, y = frame[f"{name}__{arm}"], frame["actual"]
    diff = y - q
    loss = pl.select(pl.max_horizontal(level * diff, (level - 1.0) * diff).mean()).item()
    return float((y <= q).mean()), float(loss)


def main() -> None:
    """Print every table of the write-up and store report.json."""
    df = load(SCORES)
    report: dict = {}
    print(f"rows {df.height}, sites {sorted(df['site'].unique())}, worlds {df['world'].n_unique()}")
    # ---- tuning: choose the offset's noise ratio on the tuning months
    tun = df.filter((pl.col("period") == "tuning") & (pl.col("origin") == "post"))
    tun = tun.filter(pl.col("inside") | pl.col("after48"))
    print(f"\n== tuning rows (post-alarm, inside or after48): {tun.height} ==")
    means = {r: float(tun[f"qs__NEW_DET_{r}"].mean()) for r in RATIOS}
    print({r: round(v, 4) for r, v in means.items()}, "K0", round(float(tun["qs__K0"].mean()), 4))
    best = min(means, key=means.get)
    print(f"chosen ratio {best}")
    report["ratio"] = best
    det, tru, orc = f"NEW_DET_{best}", f"NEW_TRUEEND_{best}", f"NEW_ORACLE_{best}"
    scored = df.filter(pl.col("period") != "tuning")
    post = scored.filter(pl.col("origin") == "post")
    sets = {
        "C1 rows (post-alarm, inside or 48 h after true end)": post.filter(
            pl.col("inside") | pl.col("after48")
        ),
        "post-alarm, inside the transfer": post.filter(pl.col("inside")),
        "post-alarm, 48 h after the true end": post.filter(pl.col("after48")),
        "post-alarm, duration 1 to 3 days": post.filter(
            (pl.col("inside") | pl.col("after48")) & (pl.col("duration_days") < 3)
        ),
        "post-alarm, duration 3 to 7 days": post.filter(
            (pl.col("inside") | pl.col("after48"))
            & (pl.col("duration_days") >= 3)
            & (pl.col("duration_days") < 7)
        ),
        "post-alarm, duration over 7 days": post.filter(
            (pl.col("inside") | pl.col("after48")) & (pl.col("duration_days") >= 7)
        ),
        "rest: detection lag (inside or after48)": scored.filter(
            (pl.col("origin") == "lag") & (pl.col("inside") | pl.col("after48"))
        ),
        "rest: 48 h after the detected end, outside both windows": post.filter(
            pl.col("after_det") & ~pl.col("inside") & ~pl.col("after48")
        ),
    }
    arms = {
        "GAM-ONLY": "GAM_ONLY",
        "MASK": "MASK",
        "S-KF": "SKF",
        "NEW-DET": det,
        "NEW-DET96": f"NEW_DET96_{best}",
        "NEW-DETNOCAP": f"NEW_DETNOCAP_{best}",
        "NEW-TRUEEND": tru,
        "NEW-ORACLE": orc,
        "MASK-ORACLE": "MASK_ORACLE",
        "CLEAN": "CLEAN",
    }
    for name, frame in sets.items():
        print(f"\n== {name}: {frame.height} rows ==")
        report[name] = {}
        if not frame.height:
            continue
        print(
            "  mean quantile score:",
            {k: round(float(frame[f"qs__{v}"].mean()), 3) for k, v in {"K0": "K0", **arms}.items()},
        )
        print(
            "  mean |median error|:",
            {
                k: round(float(frame[f"err_median__{v}"].abs().mean()), 3)
                for k, v in {"K0": "K0", **arms}.items()
            },
        )
        for label, arm in arms.items():
            if label in ("NEW-ORACLE", "CLEAN"):
                continue
            for ref_label, ref in (("oracle", orc), ("clean", "CLEAN")):
                level = (
                    "99"
                    if (name.startswith("C1") and label == "NEW-DET" and ref_label == "oracle")
                    else "95"
                )
                r = share(frame=frame, arm=arm, reference=ref)
                report[name][f"{label}/{ref_label}"] = r
                print(
                    f"  {label:12s} share of K0-to-{ref_label} gap {fmt(r, level)} [level {level}]"
                )
        r = share(frame=frame, arm=det, reference=orc)
        print(f"  gap K0 - ORACLE mean {r['gap_mean']:.3f} points")
        d = month_mean_difference(frame=frame, a=det, b="K0")
        print(f"  NEW-DET minus K0 mean qs {d['mean']:+.3f} ({d['lo']:+.3f} to {d['hi']:+.3f})")
        d = month_mean_difference(frame=frame, a="SKF", b="K0")
        print(f"  S-KF minus K0 mean qs {d['mean']:+.3f} ({d['lo']:+.3f} to {d['hi']:+.3f})")
    c1 = sets["C1 rows (post-alarm, inside or 48 h after true end)"]
    print("\n== leave one site out, C1 share against ORACLE ==")
    for site in sorted(c1["site"].unique()):
        print(
            f"  without {site}: {fmt(share(frame=c1.filter(pl.col('site') != site), arm=det, reference=orc))}"
        )
    print("\n== tail calibration on C1 rows ==")
    for arm_label, arm in {
        "K0": "K0",
        "S-KF": "SKF",
        "NEW-DET": det,
        "NEW-ORACLE": orc,
        "CLEAN": "CLEAN",
    }.items():
        cells = []
        for name, level in LEVELS:
            cov, loss = pinball(frame=c1, arm=arm, level=level, name=name)
            cells.append(f"{name} cov {cov:.3f} pin {loss:.3f}")
        print(f"  {arm_label:11s}", "; ".join(cells))
    print("\n== checks ==")
    print("  median minus mean forecast error, per site, C1 rows (NEW-DET):")
    for site in sorted(c1["site"].unique()):
        sub = c1.filter(pl.col("site") == site)
        print(
            f"    {site}: median error {sub[f'err_median__{det}'].mean():+.3f}, mean error "
            f"{sub[f'err_mean__{det}'].mean():+.3f}, K0 median {sub['err_median__K0'].mean():+.3f}, "
            f"K0 mean {sub['err_mean__K0'].mean():+.3f}"
        )
    for s, a in (("median", "err_median"), ("mean", "err_mean")):
        report[f"abs_error_{s}"] = {
            k: float(c1[f"{a}__{v}"].abs().mean())
            for k, v in {"K0": "K0", "NEW-DET": det, "NEW-ORACLE": orc}.items()
        }
        print(
            f"  mean |{s} error| on C1 rows:",
            {k: round(x, 3) for k, x in report[f"abs_error_{s}"].items()},
        )
    # ---- placebo: real alarms on the recorded power
    if PLACEBO.exists() and any(PLACEBO.glob("*.parquet")):
        pl_df = load(PLACEBO)
        print(f"\n== placebo (real alarms, recorded power): {pl_df.height} rows ==")
        for name, frame in (
            ("all rows near a real alarm", pl_df),
            ("rows inside an open real alarm", pl_df.filter(pl.col("in_alarm"))),
        ):
            print(f"  -- {name}: {frame.height} rows")
            for arm in [f"NEW_DET_{r}" for r in RATIOS] + ["MASK", "GAM_ONLY"]:
                d = month_mean_difference(frame=frame, a=arm, b="K0")
                shift = float((frame[f"err_median__{arm}"] - frame["err_median__K0"]).abs().mean())
                print(
                    f"    {arm:14s} minus K0 mean qs {d['mean']:+.4f} "
                    f"({d['lo']:+.4f} to {d['hi']:+.4f}); mean |median shift| {shift:.3f} points"
                )
        d = month_mean_difference(frame=pl_df, a=det, b="K0")
        shift = float((pl_df[f"err_median__{det}"] - pl_df["err_median__K0"]).abs().mean())
        report["placebo"] = {**d, "shift": shift}
    Path(WORK_DIR / "report.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
