# ruff: noqa: E501
"""Audit-trigger checks for the carried setting LR_b20 with the plain mask.

Prints (a) the ordinary-day contrast on the mean forecast's absolute error beside the quantile-score
contrast, and (b) the quantile-score harm and the AFTER share with each substation left out.

    OMP_NUM_THREADS=1 uv run --no-sync python e00227_checks.py
"""

from __future__ import annotations

import e00227_report as ra
import numpy as np
import polars as pl
from e00227_common import WORK_DIR

ARMS = ("LR_b20", "LR_b80")


def main() -> None:
    """Print both checks."""
    meta = pl.concat(
        [pl.read_parquet(WORK_DIR / f"meta_{s}.parquet") for s in ra.SITES], how="diagonal_relaxed"
    ).with_columns(month=pl.col("valid_time").dt.strftime("%Y-%m"))
    ordinary = meta.filter(pl.col("ordinary")).select("site", "row", "month", "actual")
    scores = pl.concat(
        [
            pl.read_parquet(WORK_DIR / f"scores_{s}.parquet").with_columns(site=pl.lit(s))
            for s in ra.SITES
        ],
        how="diagonal_relaxed",
    )
    folded = (
        scores.filter(pl.col("arm").is_in(["K0", *ARMS]))
        .join(ordinary, on=["site", "row"])
        .with_columns(abs_err=(pl.col("mean_forecast") - pl.col("actual")).abs())
    )
    agg = folded.group_by("site", "row", "arm", "month").agg(
        qs=pl.col("qs").mean(), err=pl.col("abs_err").mean()
    )
    for arm in ARMS:
        for metric in ("qs", "err"):
            w = agg.filter(pl.col("arm").is_in(["K0", arm])).pivot(
                on="arm", index=["site", "row", "month"], values=metric
            )
            diff = (w[arm] - w["K0"]).to_numpy().astype(float)
            h = ra.boot_ratio(num=diff, den=np.ones(len(diff)), groups=w["month"].to_numpy())
            print(f"{arm} ordinary-day {metric} contrast: {h[0]:+.4f} ({h[1]:+.4f} to {h[2]:+.4f})")
        w = agg.filter(pl.col("arm").is_in(["K0", arm])).pivot(
            on="arm", index=["site", "row", "month"], values="qs"
        )
        for site in ra.SITES:
            f = w.filter(pl.col("site") != site)
            diff = (f[arm] - f["K0"]).to_numpy().astype(float)
            h = ra.boot_ratio(num=diff, den=np.ones(len(diff)), groups=f["month"].to_numpy())
            print(f"  without {site}: {h[0]:+.4f} ({h[1]:+.4f} to {h[2]:+.4f})")
    after = pl.concat(
        [
            pl.read_parquet(WORK_DIR / f"after_{s}.parquet").with_columns(site=pl.lit(s))
            for s in ra.SITES
        ]
    )
    meta_run = meta.select("site", "row", "run_period")
    ev = after.join(meta_run, on=["site", "row"]).filter(
        pl.col("run_period").is_in(["scored", "replicate"])
    )
    arms = ["K0", "ORACLE", "LR_b20"]
    long = []
    for arm in arms:
        s = scores.filter(pl.col("arm") == arm)
        j = (
            ev.join(s.drop("fold"), on=["site", "row"])
            if arm in ("K0", "ORACLE")
            else ev.join(s, on=["site", "row", "fold"])
        )
        long.append(j.select("site", "card", "row", pl.lit(arm).alias("arm"), "qs"))
    longf = pl.concat(long).pivot(on="arm", index=["site", "card", "row"], values="qs")
    for site in ra.SITES:
        f = longf.filter(pl.col("site") != site)
        sh = ra.boot_ratio(
            num=(f["K0"] - f["LR_b20"]).to_numpy().astype(float),
            den=(f["K0"] - f["ORACLE"]).to_numpy().astype(float),
            groups=f["card"].to_numpy(),
        )
        print(
            f"AFTER share LR_b20 without {site}: {sh[0]:.3f} ({sh[1]:.3f} to {sh[2]:.3f}), cards {f['card'].n_unique()}"
        )


if __name__ == "__main__":
    main()
