"""Stage D of experiment 129: the capacity of the unmetered solar at each Primary, with a range.

For each decision Primary and arm, the script fits the convex model to the real reading (data to
`DATA_CUT`, which is 2026-08-01 when `E00129_THROUGH_JULY=1`), and refits under a stratified
month-block bootstrap. A season-year's point estimate is the fitted capacity divided by the
Primary's own median matched-farm recovery in that season-year. Its 80% range is the bootstrap
capacities divided by the recoveries of the known-answer grid at other Primaries and with other
farms, as in the coverage cells of `e00129_stage_c.py`. Season-year 2025 has no known-answer
recoveries, so the 2024 recoveries stand in for it. The table also carries the pinned capacity of
experiment 45's vintage as context, never as a score.

    E00129_THROUGH_JULY=1 uv run --no-sync python ml-experiments/00129-*/e00129_real.py \
        --arms arm2 arm4@0.1 arm3
"""

import argparse
from typing import Final

import e00129_arms as arms
import e00129_common as common
import e00129_stage_c as stage_c
import numpy as np
import polars as pl

N_BOOTSTRAP: Final[int] = 40
LAST_CALIBRATED_YEAR: Final[int] = stage_c.LAST_YEAR


def real_rows(*, site: str, arm_name: str, kat: pl.DataFrame, device: str) -> list[dict]:
    """Return one row per season-year for one Primary and arm."""
    cache = common.load_cache()
    arm, share = stage_c.parse_arm(name=arm_name)
    table = cache.sites[site].table
    fit = arms.fitted_capacity(
        arm=arm, cache=cache, site=site, device=device, increment_share=share
    )
    index = arms.training_index(cache=cache, site=site)
    weights = arms.stratified_month_weights(
        cache=cache, index=index, n_resamples=N_BOOTSTRAP, seed=129
    )
    boot_fit = arms.fit_arm(
        arm=arm, cache=cache, site=site, device=device, increment_share=share, weights=weights
    )
    boot = arms.capacities(fit=boot_fit)
    pinned = common.pinned_points(cache=cache, site=site)
    rows: list[dict] = []
    for year in common.SEASON_YEARS:
        calibration_year = min(year, LAST_CALIBRATED_YEAR)
        own = kat.filter(
            pl.col("site") == site,
            pl.col("plant") == "matched",
            pl.col("fraction") == 1.0,
            ~pl.col("cloud_load"),
            pl.col("season_year") == calibration_year,
        )["recovery"]
        if own.is_empty():
            continue
        divisor = float(np.median(own.to_numpy()))
        ratios = np.concatenate(
            [
                stage_c.calibration_ratios(
                    table=kat,
                    site=site,
                    source=source,
                    arm_name=arm_name,
                    season_year=calibration_year,
                )
                for source in common.FARMS
            ]
        )
        draws = (np.array([c[year] for c in boot])[:, None] / ratios[None, :]).ravel()
        low, high = np.quantile(draws, stage_c.RANGE_LEVELS)
        rows.append(
            {
                "site": site,
                "arm": arm_name,
                "season_year": year,
                "raw_points": fit[year],
                "divisor": divisor,
                "estimate_points": fit[year] / divisor,
                "range_low_points": float(low),
                "range_high_points": float(high),
                "estimate_mw": fit[year] / divisor * table.effective_capacity_mw / 100.0,
                "range_low_mw": float(low) * table.effective_capacity_mw / 100.0,
                "range_high_mw": float(high) * table.effective_capacity_mw / 100.0,
                "pinned_points": pinned,
                "pinned_mw": pinned * table.effective_capacity_mw / 100.0,
            }
        )
    return rows


def main() -> None:
    """Write `work/real_<arm>.parquet` for each requested arm."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=["arm2", "arm4@0.1", "arm3"])
    parser.add_argument("--sites", nargs="+", default=list(common.DECISION_SITES))
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    suffix = "_to_july" if common.THROUGH_JULY else ""
    for arm_name in arguments.arms:
        kat = pl.read_parquet(common.WORK_DIR / f"kat_{arm_name}.parquet")
        rows = [
            row
            for site in arguments.sites
            for row in real_rows(site=site, arm_name=arm_name, kat=kat, device=arguments.device)
        ]
        table = pl.DataFrame(rows)
        table.write_parquet(common.WORK_DIR / f"real_{arm_name}{suffix}.parquet")
        with pl.Config(tbl_rows=60, float_precision=2):
            print(arm_name)
            print(
                table.select(
                    "site",
                    "season_year",
                    "estimate_mw",
                    "range_low_mw",
                    "range_high_mw",
                    "pinned_mw",
                )
            )


if __name__ == "__main__":
    main()
