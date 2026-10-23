"""The night question from experiment 266: does the Economy 7 term explain D10's April-June gap?

Prints, for D10 in April to June of the scored year, the mean quantile score on night and evening
half-hours of arms `a0`, `a1`, `a2`, and BG2 (experiment 87), and the mean signed residual (actual
minus mean) by UTC half-hour of experiment 266's stored best-guess stack, BG2, and our arms. It
also prints the fitted step coefficients of `e2` at D10 at the cut 2026-04-01, to show how large the
step is in a warm month, and the step hinges' coefficients at the cut 2026-01-01 for comparison.

    uv run --no-sync python ml-experiments/00281-*/e00281_d10_night.py
"""

from datetime import UTC, datetime
from typing import Final

import e00281_c155 as c155
import e00281_common as common
import e00281_report as rep
import e00281_tall as tall
import numpy as np
import polars as pl
from studies.dp_pinned_capacity import read_vintages
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SITE: Final[str] = "D10"
MASTER: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00266-summer-gap-diagnostics-solar-term" / "work" / "master.parquet"
)
ARMS: Final[tuple[str, ...]] = ("a0", "a1", "a2", "bg2")
COEFFICIENT_CUTS: Final[tuple[datetime, ...]] = (
    datetime(2026, 1, 1, tzinfo=UTC),
    datetime(2026, 4, 1, tzinfo=UTC),
)


def our_arms() -> dict:
    """Return the scores and residuals of our arms and BG2 at D10 in April to June, at night."""
    cache = c155.load_cache()
    loaded = {
        arm: frame
        for arm in ("k0", *ARMS)
        if (frame := rep.load(arm=arm, period="scored", cache=cache)) is not None
    }
    losses = rep.stack(frames=rep.common_rows(frames=loaded))
    night = (
        losses.filter(pl.col("site") == SITE)
        .filter(pl.col("valid_month").is_in([4, 5, 6]))
        .filter(pl.col("hour_band").is_in(list(rep.NIGHT_BANDS)))
    )
    scores = night.group_by("arm").agg(pl.col("qs").mean(), pl.len().alias("rows")).sort("arm")
    print(scores)
    return {"scores": scores.to_dicts()}


def stack_residuals() -> None:
    """Print the 266 stack's and BG2's night residual by UTC half-hour at D10 in April to June."""
    master = pl.read_parquet(MASTER).filter(
        (pl.col("site") == SITE)
        & pl.col("valid_time").dt.month().is_in([4, 5, 6])
        & ~pl.col("daylight")
    )
    table = (
        master.with_columns(
            utc_start_hh=(pl.col("valid_time") - pl.duration(minutes=30)).dt.hour() * 2
            + (pl.col("valid_time") - pl.duration(minutes=30)).dt.minute() // 30
        )
        .group_by("utc_start_hh")
        .agg(
            (pl.col("actual") - pl.col("med_arm1")).mean().alias("stack_residual"),
            (pl.col("actual") - pl.col("med_bg2")).mean().alias("bg2_residual"),
            pl.len(),
        )
        .sort("utc_start_hh")
    )
    print(table)


def step_coefficients() -> dict:
    """Print the e2 fit's night levels and step hinge coefficients at two cuts."""
    cache = c155.load_cache()
    vintages = read_vintages()
    out = {}
    for cut in COEFFICIENT_CUTS:
        fit = tall.fit_tall(
            cache=cache,
            vintages=vintages,
            site=SITE,
            cut=cut,
            penalty_weight=common.T_ALL_LAMBDA,
            variant="e2",
        )
        theta = fit.fit.theta
        hinges = theta[-4:]
        levels = theta[-24:-4]
        out[cut.isoformat()] = {"hinge_coefficients": hinges.tolist(), "levels": levels.tolist()}
        print(cut.date(), "hinge coefficients (knots 3, 7, 11, 15.5 C):", np.round(hinges, 3))
        print("   night levels, starts 22:00 to 07:30:", np.round(levels, 2))
    return out


def main() -> None:
    """Print the three readouts and write them to `work/d10_night.json`."""
    result = {"our_arms": our_arms(), "coefficients": step_coefficients()}
    stack_residuals()
    c155.write_json_atomic(path=common.WORK_DIR / "d10_night.json", content=result)


if __name__ == "__main__":
    main()
