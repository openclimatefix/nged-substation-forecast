# ruff: noqa: E501
"""Post hoc checks the audit asked for.

The fit-history contrasts without the cold-start months, and arm 2 minus arm 1 on the rows after the
first as-of fit.

    PYTHONPATH=ml-experiments/00252-gam-stack-at-eight-more-primaries:ml-experiments/00264-dp-gas-engine-asymmetric-bands-at-d8:ml-experiments/00354-d8-own-residual-bands-and-longer-fit-history \
        uv run --no-sync python ml-experiments/00354-*/e00354_post_hoc.py
"""

import json
import os

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")

from datetime import UTC, datetime

import e00354_fit as fit_module
import e00354_lib as lib
import polars as pl

FIRST_FIT_ORIGIN = datetime(2025, 8, 6, 9, tzinfo=UTC)
"""The first origin that has 28 days of resolved rows (found by the audit)."""


def main() -> None:
    """Write `work/post_hoc.json` and print it."""
    fits = pl.read_parquet(fit_module.WORK / "fits_wide.parquet").with_columns(
        oct_jun=pl.col("month") >= "2025-10"
    )
    bands = pl.read_parquet(fit_module.WORK / "bands_wide_D8.parquet").with_columns(
        after_first_fit=pl.col("power_fcst_init_time") >= FIRST_FIT_ORIGIN
    )
    out: dict = {
        "rows_oct_jun": int(fits["oct_jun"].sum()),
        "rows_after_first_fit": int(bands["after_first_fit"].sum()),
    }
    for x, y in (("b", "a"), ("c", "a"), ("b", "c")):
        for stage_band in ("d_arm3", "a_arm3", "d_donated"):
            column = f"p_{x}_{y}_{stage_band}"
            fits = fits.with_columns(
                (pl.col(f"s_{x}_{stage_band}") - pl.col(f"s_{y}_{stage_band}")).alias(column)
            )
            out[column] = lib.contrast_both(wide=fits, column=column, flag="oct_jun", site="D8")
    bands = bands.with_columns((pl.col("s_arm2") - pl.col("s_arm1")).alias("p_arm2_arm1"))
    out["p_arm2_arm1_after_first_fit"] = lib.contrast_both(
        wide=bands, column="p_arm2_arm1", flag="after_first_fit", site="D8"
    )
    out["abs_after_first_fit"] = {
        n: float(bands.filter(pl.col("after_first_fit"))[f"s_{n}"].mean())
        for n in ("arm1", "arm2", "arm3", "bg2")
    }
    (fit_module.WORK / "post_hoc.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
