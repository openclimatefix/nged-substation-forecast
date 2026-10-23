# ruff: noqa: E501
"""Audit checks.

Median minus mean forecast, mean-forecast error contrast, leave-one-site-out harm, and whether L2's loss of matches sits in the load shifts whose onset candidate L2 trained as a negative.

    OMP_NUM_THREADS=1 uv run --no-sync python e00302_checks.py | tee printed_checks.txt
"""

from __future__ import annotations

import numpy as np
import polars as pl
from e00302_common import WORK_DIR, stamp_months

SITES = ("D5", "D9", "D10", "D11", "D14", "D16")
E227 = WORK_DIR.parent / "00227-lsc-fewer-false-alarms-and-softer-use-part-2"


def main() -> None:
    """Print the checks."""
    frames = []
    for site in SITES:
        new = pl.read_parquet(WORK_DIR / f"h_scores_{site}.parquet")
        meta = pl.read_parquet(WORK_DIR / f"h_meta_{site}.parquet").select("row", "valid_time")
        actual = pl.read_parquet(E227 / f"meta_{site}.parquet").select("row", "actual")
        per = (
            new.group_by("arm", "row")
            .agg(
                qs=pl.col("qs").mean(),
                q50=pl.col("q50").mean(),
                mean_forecast=pl.col("mean_forecast").mean(),
            )
            .join(meta, on="row")
            .join(actual, on="row")
            .with_columns(site=pl.lit(site), odd=pl.col("valid_time").dt.month() % 2 == 1)
        )
        frames.append(per)
    d = pl.concat(frames).with_columns(abs_err=(pl.col("mean_forecast") - pl.col("actual")).abs())
    k0 = d.filter(pl.col("arm") == "K0").select(
        "site", "row", k0_qs=pl.col("qs"), k0_err=pl.col("abs_err")
    )
    d = d.filter(pl.col("arm") != "K0").join(k0, on=["site", "row"]).filter(~pl.col("odd"))
    print(
        "== Even months (held-out for L1 to L3): quantile-score harm and mean-forecast absolute-error contrast =="
    )
    for arm in ("L1", "L2", "L3", "L3_5", "L3_2"):
        a = d.filter(pl.col("arm") == arm)
        print(
            f"{arm:5s} harm {float(np.mean((a['qs'] - a['k0_qs']).to_numpy())):+.4f}  mean-forecast MAE minus K0 {float(np.mean((a['abs_err'] - a['k0_err']).to_numpy())):+.4f}"
        )
    print("\n== Median forecast minus mean forecast (MW), per site, for L3 and for K0 ==")
    for site in SITES:
        for arm in ("L3",):
            a = d.filter((pl.col("arm") == arm) & (pl.col("site") == site))
            print(
                f"{site} {arm}: {float(np.mean((a['q50'] - a['mean_forecast']).to_numpy())):+.4f}"
            )
    print("\n== Leave one site out: harm on even months ==")
    for arm in ("L1", "L3"):
        out = []
        for site in SITES:
            a = d.filter((pl.col("arm") == arm) & (pl.col("site") != site))
            out.append(f"without {site} {float(np.mean((a['qs'] - a['k0_qs']).to_numpy())):+.3f}")
        print(arm, "; ".join(out))
    print(
        "\n== Where L2 loses matches (L1 found, L2 not) by whether the shift starts in an odd (training) month =="
    )
    hits = pl.read_parquet(WORK_DIR / "shift_hits.parquet").filter(pl.col("variant") == "all")
    from e00302_score import shifts_table

    months = stamp_months()
    sh = shifts_table().with_columns(
        odd=pl.Series(
            [int(months[min(o, len(months) - 1)]) % 2 == 1 for o in shifts_table()["onset"]]
        )
    )
    wide = hits.pivot(on="arm", index="card_id", values="found").join(
        sh.select("card_id", "odd"), on="card_id"
    )
    for odd in (True, False):
        w = wide.filter(pl.col("odd") == odd)
        print(
            f"start in {'odd (L2 training)' if odd else 'even (not trained on)'} month: {w.height} shifts; L1 found {int(w['L1'].sum())}, L2 found {int(w['L2'].sum())}, L3 found {int(w['L3'].sum())}, L0 found {int(w['L0'].sum())}"
        )


if __name__ == "__main__":
    main()
