# ruff: noqa: E501, D103
"""Score the pre-registered prediction and rank residual statistics by leave-one-Primary-out error.

Reads the tables of `e00217_run.py`; prints every number the write-up quotes.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00217-switching-label-free-thresholds-in-noise-units"
EXCEPTIONS: Final = ("D8", "D9")
RATE_FLOOR: Final = 0.25
"""Alarms per Primary-year used in place of a rate of zero when taking logarithms."""
DETECTORS: Final = (
    "RO-NC/current", "RO-NC/R1", "RO-NC/R2", "pair/current", "pair/R1", "pair/R2",
    "RO-NC XL stored", "R re-ranker", "F logistic",
)  # fmt: skip
SHARE_FLOOR: Final = 0.01


def feature_table(*, stats: pl.DataFrame) -> pl.DataFrame:
    """Log-scale candidate predictors, one row per Primary."""
    cols = {
        f"log_{c}": np.log2(stats[c].to_numpy())
        for c in stats.columns
        if c.startswith(("cur_", "mad_all_", "q95_"))
    }
    cols["log_kurt_48"] = np.log2(stats["kurt_48"].to_numpy() + 3.0)
    cols["log_kurt_8"] = np.log2(stats["kurt_8"].to_numpy() + 3.0)
    cols["log_misfit_share"] = np.log2(stats["misfit_share"].to_numpy() + SHARE_FLOOR)
    cols["log_holiday_misfit_share"] = np.log2(
        stats["holiday_misfit_share"].to_numpy() + SHARE_FLOOR
    )
    cols["log_tail_ratio_48"] = np.log2(stats["q95_48"].to_numpy() / stats["cur_48"].to_numpy())
    cols["log_tail_ratio_8"] = np.log2(stats["q95_8"].to_numpy() / stats["cur_8"].to_numpy())
    cols["log_resid_mad"] = np.log2(stats["resid_mad"].to_numpy())
    return pl.DataFrame({"site": stats["site"], **cols})


def lopo(*, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out prediction of `y` from `x` by least squares (intercept and slope)."""
    out = np.empty_like(y)
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        out[i] = slope * x[i] + intercept
    return out


def main() -> None:
    rates = pl.read_parquet(WORK_DIR / "rates.parquet").filter(pl.col("group") == "all")
    stats = pl.read_parquet(WORK_DIR / "stats.parquet")
    pl.Config.set_tbl_rows(100).set_tbl_cols(30).set_tbl_width_chars(200)

    print(
        "== Prediction: share of the 13 other Primaries within a factor 2 of the median; D8 and D9 within 3"
    )
    rows = []
    for det in DETECTORS:
        r = rates.filter(pl.col("detector") == det)
        median = float(np.median(r["rate"].to_numpy()))
        ratio = {s: v / median for s, v in zip(r["site"], r["rate"], strict=True)}
        others = [v for s, v in ratio.items() if s not in EXCEPTIONS]
        within2 = sum(0.5 <= v <= 2.0 for v in others)
        within3 = {s: 1 / 3 <= ratio[s] <= 3.0 for s in EXCEPTIONS}
        logs = np.log2(np.maximum(r["rate"].to_numpy(), RATE_FLOOR))
        rows.append({
            "detector": det, "median_rate": round(median, 2), "others_within_2": f"{within2} of {len(others)}",
            "D8_ratio": round(ratio["D8"], 2), "D9_ratio": round(ratio["D9"], 2),
            "D8_D9_within_3": all(within3.values()), "sd_log2_rate": round(float(logs.std(ddof=1)), 2),
            "min_rate": round(float(r["rate"].to_numpy().min()), 2), "max_rate": round(float(r["rate"].to_numpy().max()), 2),
        })  # fmt: skip
    print(pl.DataFrame(rows))

    print(
        "== Winter against the rest, alarms per Primary-year, pooled over Primaries (exposure-weighted)"
    )
    allg = pl.read_parquet(WORK_DIR / "rates.parquet").filter(pl.col("group") != "all")
    pooled = (
        allg.group_by("detector", "group").agg(alarms=pl.col("alarms").sum(), years=pl.col("years").sum())
        .with_columns(rate=pl.col("alarms") / pl.col("years")).sort("detector", "group")
    )  # fmt: skip
    print(
        pooled.with_columns(
            pl.col("rate").round(2), pl.col("alarms").round(1), pl.col("years").round(2)
        )
    )

    print("== Leave-one-Primary-out prediction of log2 alarm rate from one residual statistic")
    features = feature_table(stats=stats)
    ranking = []
    for det in ("RO-NC/current", "pair/current", "RO-NC XL stored"):
        r = rates.filter(pl.col("detector") == det).join(features, on="site")
        y = np.log2(np.maximum(r["rate"].to_numpy(), RATE_FLOOR))
        baseline = np.array([np.median(np.delete(y, i)) for i in range(len(y))])
        base_mae = float(np.abs(y - baseline).mean())
        for col in features.columns[1:]:
            x = r[col].to_numpy()
            pred = lopo(x=x, y=y)
            mae = float(np.abs(y - pred).mean())
            within2 = int((np.abs(y - pred) <= 1.0).sum())
            ranking.append({"detector": det, "statistic": col, "mae_log2": round(mae, 2),
                            "baseline_mae": round(base_mae, 2), "within_factor_2": f"{within2} of {len(y)}",
                            "corr": round(float(np.corrcoef(x, y)[0, 1]), 2)})  # fmt: skip
    table = pl.DataFrame(ranking)
    for det in table["detector"].unique(maintain_order=True):
        print(det)
        print(table.filter(pl.col("detector") == det).sort("mae_log2").head(8))
    table.write_parquet(WORK_DIR / "lopo.parquet")

    print(
        "== Robustness: drop each Primary in turn (RO-NC/current): share of other Primaries within a factor 2"
    )
    r = rates.filter(pl.col("detector") == "RO-NC/current")
    sites, vals = r["site"].to_list(), r["rate"].to_numpy()
    for drop in sites:
        keep = [i for i, s in enumerate(sites) if s != drop]
        med = float(np.median(vals[keep]))
        others = [vals[i] for i in keep if sites[i] not in EXCEPTIONS]
        n = sum(0.5 * med <= v <= 2 * med for v in others)
        print(f"  without {drop}: median {med:.2f}, {n} of {len(others)}")
    feats = feature_table(stats=stats).join(r.select("site", "rate"), on="site")
    for drop in (None, "D18"):
        f = feats if drop is None else feats.filter(pl.col("site") != drop)
        y = np.log2(np.maximum(f["rate"].to_numpy(), RATE_FLOOR))
        x = f["log_misfit_share"].to_numpy()
        pred = lopo(x=x, y=y)
        print(
            f"  misfit share, without {drop}: corr {np.corrcoef(x, y)[0, 1]:.2f}, LOPO MAE {np.abs(y - pred).mean():.2f},"
            f" baseline {np.mean([abs(y[i] - np.median(np.delete(y, i))) for i in range(len(y))]):.2f}"
        )


if __name__ == "__main__":
    main()
