"""Stage D of experiment 129: what the capacity estimate says on 1 September 2025, and how it ages.

At the origin 2025-09-01 the fit sees only rows before the origin, so the latest season-year with
data is 2024. The as-of estimate for 2024, held forward as the estimate for season-year 2025, is
compared with the hindsight estimate for 2025 (fitted to 2026-07-31) and with the hindsight
estimate for 2024. Estimates and ranges are built as in `e00129_real.py`.

    E00129_THROUGH_JULY=1 uv run --no-sync python ml-experiments/00129-*/e00129_asof.py
"""

import argparse
from datetime import UTC, datetime
from typing import Final

import e00129_arms as arms
import e00129_common as common
import e00129_real as real
import e00129_stage_c as stage_c
import numpy as np
import polars as pl

ORIGIN: Final[datetime] = datetime(2025, 9, 1, tzinfo=UTC)
LATEST_SEASON_YEAR: Final[int] = 2024


def asof_rows(*, site: str, arm_name: str, kat: pl.DataFrame, device: str) -> dict:
    """Return the as-of estimate and range of the latest season-year at one Primary and arm."""
    cache = common.load_cache()
    arm, share = stage_c.parse_arm(name=arm_name)
    scale = cache.sites[site].table.effective_capacity_mw / 100.0
    index = arms.training_index(cache=cache, site=site, cut=ORIGIN)
    weights = arms.stratified_month_weights(
        cache=cache, index=index, n_resamples=real.N_BOOTSTRAP, seed=129
    )
    fit = arms.fitted_capacity(
        arm=arm, cache=cache, site=site, device=device, increment_share=share, cut=ORIGIN
    )
    boot = arms.capacities(
        fit=arms.fit_arm(
            arm=arm,
            cache=cache,
            site=site,
            device=device,
            increment_share=share,
            weights=weights,
            cut=ORIGIN,
        )
    )
    divisor = float(
        kat.filter(
            pl.col("site") == site,
            pl.col("plant") == "matched",
            pl.col("fraction") == 1.0,
            ~pl.col("cloud_load"),
            pl.col("season_year") == LATEST_SEASON_YEAR,
        )["recovery"].median()
    )
    ratios = np.concatenate(
        [
            stage_c.calibration_ratios(
                table=kat,
                site=site,
                source=source,
                arm_name=arm_name,
                season_year=LATEST_SEASON_YEAR,
            )
            for source in common.FARMS
        ]
    )
    draws = (np.array([c[LATEST_SEASON_YEAR] for c in boot])[:, None] / ratios[None, :]).ravel()
    low, high = np.quantile(draws, stage_c.RANGE_LEVELS)
    return {
        "site": site,
        "arm": arm_name,
        "asof_estimate_mw": fit[LATEST_SEASON_YEAR] / divisor * scale,
        "asof_low_mw": float(low) * scale,
        "asof_high_mw": float(high) * scale,
    }


def main() -> None:
    """Write `work/asof_to_july.parquet` and print it beside the hindsight estimates."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=["arm2", "arm4@0.1"])
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    rows = []
    for arm_name in arguments.arms:
        kat = pl.read_parquet(common.WORK_DIR / f"kat_{arm_name}.parquet")
        rows += [
            asof_rows(site=site, arm_name=arm_name, kat=kat, device=arguments.device)
            for site in common.DECISION_SITES
        ]
    asof = pl.DataFrame(rows)
    hindsight = pl.concat(
        [pl.read_parquet(common.WORK_DIR / f"real_{arm}_to_july.parquet") for arm in arguments.arms]
    )
    pivot = hindsight.filter(pl.col("season_year").is_in([LATEST_SEASON_YEAR, 2025])).pivot(
        on="season_year", index=["site", "arm"], values="estimate_mw"
    )
    pivot = pivot.rename({"2024": "hindsight_2024_mw", "2025": "hindsight_2025_mw"})
    table = asof.join(pivot, on=["site", "arm"]).with_columns(
        held_forward_error_mw=pl.col("asof_estimate_mw") - pl.col("hindsight_2025_mw")
    )
    table.write_parquet(common.WORK_DIR / "asof_to_july.parquet")
    with pl.Config(tbl_rows=20, tbl_cols=12, float_precision=2):
        print(table)


if __name__ == "__main__":
    main()
