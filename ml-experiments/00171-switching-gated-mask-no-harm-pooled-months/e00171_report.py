"""Print the pooled no-harm contrast of the gated mask and the alarm and masked half-hour counts.

Reads the tables `e00171_run.py` stored. Every number quoted in `result.md` is printed here and
saved to `report.json`.

    uv run --no-sync python e00171_report.py | tee printed_report.txt
"""

from __future__ import annotations

import json
from typing import Final

import numpy as np
import polars as pl
from e00171_common import WORK_DIR

SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
N_DRAWS: Final[int] = 20_000
LIMIT_ALL: Final[float] = 0.03
LIMIT_WINTER: Final[float] = 0.05
SCORED_YEAR_START: Final = pl.datetime(2025, 7, 1)
REPORT: dict[str, object] = {}


def boot(*, values: np.ndarray, groups: np.ndarray, seed: int = 171) -> tuple[float, float, float]:
    """Mean of `values` with a 95% interval from resampling whole `groups` (calendar months)."""
    labels, inverse = np.unique(groups, return_inverse=True)
    total = np.bincount(inverse, weights=values, minlength=len(labels))
    count = np.bincount(inverse, minlength=len(labels)).astype(float)
    picks = np.random.default_rng(seed).integers(0, len(labels), size=(N_DRAWS, len(labels)))
    draws = total[picks].sum(axis=1) / count[picks].sum(axis=1)
    return (
        float(total.sum() / count.sum()),
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
    )


def fmt(t: tuple[float, float, float]) -> str:
    """Format a point estimate with its interval."""
    return f"{t[0]:+.3f} ({t[1]:+.3f} to {t[2]:+.3f})"


def build() -> pl.DataFrame:
    """One row per (site, row) outside logged transfers, with each arm's score as a column."""
    rows = pl.concat([pl.read_parquet(WORK_DIR / f"rows_{s}.parquet") for s in SITES])
    flags = pl.concat([pl.read_parquet(WORK_DIR / f"flags_{s}.parquet") for s in SITES])
    keys = ["site", "row", "power_fcst_init_time", "valid_time", "lead_hours", "actual"]
    wide = rows.pivot(on="arm", index=keys, values="qs")
    mean = rows.pivot(on="arm", index=["site", "row"], values="mean_forecast").rename(
        {a: f"mean_{a}" for a in ("K0", "MASK_GATED", "MASK_NEW", "MASK_UNION")}
    )
    q50 = rows.filter(pl.col("arm") == "K0").select("site", "row", "q50")
    return (
        wide.join(mean, on=["site", "row"])
        .join(q50, on=["site", "row"])
        .join(flags, on=["site", "row"])
        .filter(~pl.col("in_logged"))
        .with_columns(
            month=pl.col("valid_time").dt.strftime("%Y-%m"),
            mo=pl.col("valid_time").dt.month(),
        )
        .with_columns(winter=pl.col("mo").is_in([11, 12, 1, 2, 3]))
    )


def contrast(*, frame: pl.DataFrame, arm: str, ref: str = "K0") -> tuple[float, float, float]:
    """Arm minus reference quantile score, resampling calendar months."""
    values = (frame[arm] - frame[ref]).to_numpy()
    return boot(values=values, groups=frame["month"].to_numpy())


def show(*, name: str, frame: pl.DataFrame, arm: str = "MASK_GATED") -> tuple[float, float, float]:
    """Print and store one contrast."""
    t = contrast(frame=frame, arm=arm)
    print(
        f"{name}: {arm} minus K0 {fmt(t)}  rows {frame.height}, months {frame['month'].n_unique()}"
    )
    REPORT[f"{name}|{arm} minus K0"] = list(t)
    return t


def decide(*, pooled: tuple[float, float, float], winter: tuple[float, float, float]) -> None:
    """Apply the pre-registered decision rule."""
    g1 = pooled[1] >= -LIMIT_ALL and pooled[2] <= LIMIT_ALL
    g2 = winter[1] >= -LIMIT_WINTER and winter[2] <= LIMIT_WINTER
    print("\n== DECISION RULE (pre-registered in result.md) ==")
    half_all, half_winter = (pooled[2] - pooled[1]) / 2, (winter[2] - winter[1]) / 2
    print(f"G1 pooled interval inside +/-{LIMIT_ALL}: {g1} (half-width {half_all:.3f})")
    print(f"G2 winter interval inside +/-{LIMIT_WINTER}: {g2} (half-width {half_winter:.3f})")
    if pooled[0] > LIMIT_ALL:
        verdict = "refuted"
    elif g1 and g2:
        verdict = "supported"
    elif g1:
        verdict = "partly supported"
    else:
        verdict = "inconclusive"
    print("verdict:", verdict)
    REPORT["verdict"] = verdict


def alarm_counts() -> None:
    """Gated alarms and masked half-hours per substation and calendar year."""
    masked = pl.concat([pl.read_parquet(WORK_DIR / f"masked_{s}.parquet") for s in SITES])
    first_run = pl.datetime(2024, 4, 10)
    masked = masked.filter(pl.col("start_time") >= first_run).with_columns(
        year=pl.col("start_time").dt.year()
    )
    for arm in ("MASK_GATED", "MASK_NEW", "MASK_UNION"):
        sub = masked.filter(pl.col("arm") == arm)
        table = (
            sub.group_by("site", "year")
            .agg(
                alarms=pl.len(),
                masked_half_hours=pl.col("n_half_hours").sum(),
                outside_logged=pl.col("n_outside_logged").sum(),
            )
            .sort("site", "year")
        )
        print(f"\n== {arm}: alarms starting on or after 2024-04-10, and masked half-hours ==")
        print(table)
        share = int(sub["n_outside_logged"].sum()) / int(sub["n_half_hours"].sum())
        print(
            f"{arm}: {sub.height} alarms, {sub['n_half_hours'].sum()} masked half-hours, "
            f"{sub['n_outside_logged'].sum()} outside every logged transfer (share {share:.3f})"
        )
        REPORT[f"{arm}|alarms"] = sub.height
        REPORT[f"{arm}|masked_half_hours"] = int(sub["n_half_hours"].sum())
        REPORT[f"{arm}|share_outside_logged"] = float(share)
        REPORT[f"{arm}|table"] = table.to_dicts()


def main() -> None:
    """Print every number of the write-up."""
    wide = build()
    print("rows outside logged transfers:", wide.height)
    for site in SITES:
        print(site, wide.filter(pl.col("site") == site).height)
    print("\n== G1 and G2 ==")
    pooled = show(name="pooled", frame=wide)
    winter = show(name="winter", frame=wide.filter(pl.col("winter")))
    show(name="other months", frame=wide.filter(~pl.col("winter")))
    decide(pooled=pooled, winter=winter)
    print("\n== post hoc ==")
    scored = wide.filter(pl.col("valid_time") >= SCORED_YEAR_START)
    earlier = wide.filter(pl.col("valid_time") < SCORED_YEAR_START)
    show(name="valid time from 2025-07 (experiment 166's year)", frame=scored)
    show(name="valid time before 2025-07", frame=earlier)
    for arm in ("MASK_NEW", "MASK_UNION"):
        show(name="pooled", frame=wide, arm=arm)
    for site in SITES:
        show(name=f"without {site}", frame=wide.filter(pl.col("site") != site))
    for site in SITES:
        show(name=f"only {site}", frame=wide.filter(pl.col("site") == site))
    active = wide.filter(pl.col("active_MASK_GATED"))
    quiet = wide.filter(~pl.col("active_MASK_GATED"))
    print("rows with a gated alarm active:", active.height, "share", active.height / wide.height)
    REPORT["gated_active_share"] = active.height / wide.height
    show(name="gated alarm active", frame=active)
    show(name="no gated alarm active", frame=quiet)
    print("\n== absolute scores, pooled ==")
    for arm in ("K0", "MASK_GATED", "MASK_NEW", "MASK_UNION"):
        print(arm, round(float(np.mean(wide[arm].to_numpy())), 4))
        REPORT[f"mean score|{arm}"] = float(np.mean(wide[arm].to_numpy()))
    print("\n== checks ==")
    for site in SITES:
        sub = wide.filter(pl.col("site") == site)
        d = float(np.mean((sub["q50"] - sub["mean_K0"]).to_numpy()))
        print(f"median minus mean forecast, {site}: {d:+.3f}")
        REPORT[f"median minus mean|{site}"] = d
    err = {
        arm: (wide[f"mean_{arm}"] - wide["actual"]).abs().to_numpy() for arm in ("K0", "MASK_GATED")
    }
    t = boot(values=err["MASK_GATED"] - err["K0"], groups=wide["month"].to_numpy())
    print("absolute error of the mean forecast, MASK_GATED minus K0:", fmt(t))
    REPORT["mean abs error|MASK_GATED minus K0"] = list(t)
    alarm_counts()
    (WORK_DIR / "report.json").write_text(json.dumps(REPORT, indent=1, default=str))


if __name__ == "__main__":
    main()
