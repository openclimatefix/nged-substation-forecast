"""Step-up injection of experiment 129: does an arm follow a rise in capacity?

A real farm's output, scaled to `STEP_FRACTION` of the Primary's pinned capacity, is added from
`STEP_DATE` onwards, in the middle of one season-year. The recovered capacity of each complete
season-year (the difference of two fits, divided by the stepped capacity) is compared with the
expected recovery: zero before the step, the matched-plant recovery times the sun-weighted share of
the season-year after the step during it, and the matched-plant recovery after it. All arms are
fitted without the drift term, as in the rest of the experiment.

    uv run --no-sync python ml-experiments/00129-*/e00129_step.py [--device cuda]
"""

import argparse
from datetime import UTC, datetime
from typing import Final

import e00129_arms as arms
import e00129_common as common
import e00129_stage_c as stage_c
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache

STEP_FRACTION: Final[float] = 0.5
STEP_DATE: Final[datetime] = datetime(2024, 3, 1, tzinfo=UTC)
"""The middle of season-year 2023 (July 2023 to June 2024)."""
STEP_START: Final[np.datetime64] = np.datetime64("2024-03-01T00:00", "us")
"""`STEP_DATE` as a naive UTC `datetime64`, for comparison with the cache's grid."""
STEP_SEASON_YEAR: Final[int] = 2023
STEP_ARMS: Final[tuple[str, ...]] = (
    "arm2",
    "arm3",
    "arm4@0.1",
    "arm4@1.0",
)


def sun_weighted_share_after_step(*, cache: FeatureCache, site: str) -> float:
    """Return the share of the step season-year's PV shape that falls after the step."""
    shape = np.nan_to_num(cache.sites[site].pv_unit_shape, nan=0.0)
    starts = cache.end_times - np.timedelta64(30, "m")
    month = starts.astype("datetime64[M]").astype(np.int64)
    season_year = (month + 6) // 12 + 1970 - 1
    in_year = season_year == STEP_SEASON_YEAR
    after = starts >= STEP_START
    return float(shape[in_year & after].sum() / shape[in_year].sum())


def step_rows(*, cache: FeatureCache, site: str, arm_name: str, device: str) -> list[dict]:
    """Return one row per farm and season-year for one arm at one Primary."""
    arm, share = stage_c.parse_arm(name=arm_name)
    pinned = common.pinned_points(cache=cache, site=site)
    stepped = STEP_FRACTION * pinned
    after_step = (cache.end_times - np.timedelta64(30, "m") >= STEP_START).astype(float)
    base = arms.fitted_capacity(
        arm=arm, cache=cache, site=site, device=device, increment_share=share
    )
    rows: list[dict] = []
    for farm in common.FARMS:
        output = stepped * common.farm_norm(cache=cache, farm=farm) * after_step
        injected_cache = common.inject(cache=cache, site=site, output=output)
        capacity = arms.fitted_capacity(
            arm=arm, cache=injected_cache, site=site, device=device, increment_share=share
        )
        rows.extend(
            {
                "site": site,
                "arm": arm_name,
                "farm": farm,
                "season_year": year,
                "recovery": (capacity[year] - base[year]) / stepped,
                "in_step_year": year == STEP_SEASON_YEAR,
                "after_step_year": year > STEP_SEASON_YEAR,
            }
            for year in common.data_years(cache=cache, site=site)
            if capacity[year] > stage_c.ESTIMABLE_SHARE * stepped or year < STEP_SEASON_YEAR
        )
    return rows


def main() -> None:
    """Run the step-up injection and write `work/step.parquet`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sites", nargs="+", default=list(common.DECISION_SITES))
    arguments = parser.parse_args()
    cache = common.load_cache()
    rows = [
        row
        for site in arguments.sites
        for arm_name in STEP_ARMS
        for row in step_rows(cache=cache, site=site, arm_name=arm_name, device=arguments.device)
    ]
    table = pl.DataFrame(rows)
    table.write_parquet(common.WORK_DIR / "step.parquet")
    share = {
        site: sun_weighted_share_after_step(cache=cache, site=site) for site in arguments.sites
    }
    print("sun-weighted share of the step season-year after the step:", share)
    with pl.Config(tbl_rows=60, float_precision=3):
        print(
            table.group_by("arm", "season_year")
            .agg(recovery=pl.col("recovery").median())
            .sort("arm", "season_year")
            .pivot(on="season_year", index="arm", values="recovery")
        )


if __name__ == "__main__":
    main()
