# ruff: noqa: E501
"""Leave-one-series-out point estimates of the two deciding contrasts and the stage split.

Real record: A minus C and AF minus C on ordinary rows (mean quantile score and 95% pinball loss),
dropping each Primary in turn. Injected worlds: A minus K0 on the rows during and after a shift
(95% pinball loss), dropping each of the six sites in turn. Point estimates only; positive means the
masked arm is worse.

    uv run --no-sync python e00327_leave_one_out.py
"""

from __future__ import annotations

import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK = ML_EXPERIMENTS_DATA_DIR / "00327-lsc-label-free-detect-mask-train-pipeline"


def real_rows() -> None:
    """Print the real-record leave-one-out table."""
    t = pl.read_parquet(WORK / "results" / "real_units.parquet").filter(
        pl.col("group") == "ordinary"
    )
    print("real record, ordinary rows: drop one Primary, arm minus C (qs, pin95)")
    for s in sorted(t["series"].unique().to_list()):
        sub = (
            t.filter(pl.col("series") != s)
            .group_by("arm")
            .agg(pl.col("n").sum(), pl.col("qs").sum(), pl.col("pin95").sum())
        )
        m = {r["arm"]: (r["qs"] / r["n"], r["pin95"] / r["n"]) for r in sub.iter_rows(named=True)}
        print(
            f"  drop {s:2d}: "
            + "  ".join(
                f"{a} {m[a][0] - m['C'][0]:+.3f}/{m[a][1] - m['C'][1]:+.3f}"
                for a in ("A", "B", "E", "AF")
            )
        )


def injected_rows() -> None:
    """Print the injected-world leave-one-site-out table."""
    t = pl.concat([pl.read_parquet(p) for p in sorted((WORK / "inj").glob("D*_*.parquet"))])
    e = pl.col("actual_nra") - pl.col("q95")
    t = t.with_columns(pin95=pl.max_horizontal(0.95 * e, -0.05 * e)).filter(
        pl.col("phase").is_in(["DURING", "AFTER"])
        & (pl.coalesce(pl.col("onset_month") % 100, 0) % 2 == 0)
    )
    print(
        "injected worlds, during and after a shift, even onset months: drop one site, arm minus K0 (pin95)"
    )
    for site in [None, *sorted(t["site"].unique().to_list())]:
        sub = t if site is None else t.filter(pl.col("site") != site)
        m = {
            r["arm"]: r["pin95"]
            for r in sub.group_by("arm").agg(pl.col("pin95").mean()).iter_rows(named=True)
        }
        print(
            f"  drop {site}: "
            + "  ".join(f"{a} {m[a] - m['K0']:+.3f}" for a in ("A", "B", "PERFECT"))
        )


if __name__ == "__main__":
    real_rows()
    injected_rows()
