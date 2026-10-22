# ruff: noqa: E501, D103
"""Post hoc: RO-NC alarm rate against threshold, per Primary, and the share within a factor 2 of the median.

Uses the current noise estimate. Run after `e00217_run.py`; writes `curves.parquet`.
"""

from __future__ import annotations

import e00217_run as base
import numpy as np
import polars as pl

THRESHOLDS = (2.5, 3.0, 3.5, 4.35, 5.0, 6.0, 8.0)
EXCEPTIONS = ("D8", "D9")


def main() -> None:
    sites = {
        int(r["time_series_id"]): r["site"]
        for r in pl.read_parquet(base.SERIES_PATH).iter_rows(named=True)
    }
    first, second = base.load_residuals()
    zone = base.refit_zone()
    shift = {
        s: np.where(
            zone,
            base.level_shift(values=second[s], window=48),
            base.level_shift(values=first[s], window=48),
        )
        for s in base.SERIES_IDS
    }
    scale = {s: base.mad_scale(x=shift[s], mask=base.ODD) for s in shift}
    years = {s: float(np.isfinite(shift[s]).sum() / base.YEAR_HALF_HOURS) for s in shift}
    rows = []
    for k in THRESHOLDS:
        alarms = base.ronc_alarms(shift=shift, scale=scale, k=k)
        counts = alarms.group_by("series").len()
        for s in shift:
            n = counts.filter(pl.col("series") == s)["len"].sum()
            rows.append(
                {"threshold": k, "site": sites[s], "alarms": int(n), "rate": float(n) / years[s]}
            )
    table = pl.DataFrame(rows)
    table.write_parquet(base.WORK_DIR / "curves.parquet")
    pl.Config.set_tbl_rows(60).set_tbl_cols(30).set_tbl_width_chars(200)
    print(
        table.pivot(on="threshold", index="site", values="rate").with_columns(
            pl.all().exclude("site").round(1)
        )
    )
    for k in THRESHOLDS:
        t = table.filter(pl.col("threshold") == k)
        med = float(np.median(t["rate"].to_numpy()))
        others = t.filter(~pl.col("site").is_in(EXCEPTIONS))
        within2 = int(((others["rate"] >= med / 2) & (others["rate"] <= med * 2)).sum())
        d = {r["site"]: r["rate"] / med for r in t.iter_rows(named=True)}
        print(
            f"z>={k}: median {med:.2f}/yr, others within 2: {within2} of {others.height}, D8 x{d['D8']:.2f}, D9 x{d['D9']:.2f}"
        )


if __name__ == "__main__":
    main()
