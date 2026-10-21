"""Stage B of experiment 129 on the semi-synthetic target: false capacity, drift, and the placebo.

For each Primary the script fits the sun-blind arm 3 on the real reading, takes its demand terms
and its residual, and builds `N_DRAWS` semi-synthetic targets (demand plus residuals resampled by
calendar month, whose true solar capacity is zero). On each target it fits arms 2 and 3

- with nothing added (the capacity the fit finds from noise alone),
- with each real farm's output added at 100% of the Primary's pinned capacity (recovery, as a
  difference of fits, and drift: the slope of the recovery over season-years, per 5 years), and
- the same two cases again with the placebo sky (the previous year's PV shape).

    uv run --no-sync python ml-experiments/00129-*/e00129_semi_run.py [--device cuda] \
        [--sites D10 D16] [--draws 10]
"""

import argparse
from typing import Final

import e00129_arms as arms
import e00129_common as common
import e00129_pilot as pilot
import e00129_semi as semi
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

N_DRAWS: Final[int] = 10
YEARS_PER_DRIFT: Final[int] = 5


def drift_per_five_years(*, recovery: np.ndarray) -> float:
    """Return the slope of the recovery over season-years, per 5 years, by least squares."""
    if len(recovery) < 3:
        return float("nan")
    slope = np.polyfit(np.arange(len(recovery)), recovery, 1)[0]
    return float(slope * YEARS_PER_DRIFT)


def draw_rows(
    *,
    cache: FeatureCache,
    placebo_cache: FeatureCache,
    site: str,
    arm: arms.ArmType,
    increment_share: float,
    draw: int,
    injected_points: float,
    device: str,
) -> list[dict[str, object]]:
    """Return table rows for one arm and draw: the none and farm cases, real and placebo sky."""
    rows: list[dict[str, object]] = []
    for sky, source in (("real", cache), ("placebo", placebo_cache)):
        none_capacity = arms.fitted_capacity(
            arm=arm, cache=source, site=site, device=device, increment_share=increment_share
        )
        for farm in common.FARMS:
            output = injected_points * common.farm_norm(cache=cache, farm=farm)
            injected_cache = common.inject(cache=source, site=site, output=output)
            capacity = arms.fitted_capacity(
                arm=arm,
                cache=injected_cache,
                site=site,
                device=device,
                increment_share=increment_share,
            )
            years = [
                y
                for y in common.COMPLETE_SEASON_YEARS
                if capacity[y] > pilot.ESTIMABLE_SHARE * injected_points
            ]
            recovery = np.array([(capacity[y] - none_capacity[y]) / injected_points for y in years])
            rows.append(
                {
                    "site": site,
                    "arm": arm,
                    "draw": draw,
                    "sky": sky,
                    "farm": farm,
                    "n_years": len(years),
                    "recovery_mean": float(recovery.mean()) if years else float("nan"),
                    "drift_5y": drift_per_five_years(recovery=recovery),
                    "false_capacity_last": none_capacity[common.COMPLETE_SEASON_YEARS[-1]]
                    / injected_points,
                }
            )
    return rows


def run_site(
    *,
    cache: FeatureCache,
    site: str,
    draws: int,
    device: str,
    arm_names: list[arms.ArmType],
    increment_share: float,
) -> list[dict[str, object]]:
    """Return every table row of one Primary."""
    injected_points = pinned_capacity(
        vintages=read_vintages(), table=cache.sites[site].table, cut=common.DATA_CUT
    ).held_points
    real_fit = common.fit_site(
        cache=cache, site=site, spec=arms.arm_spec(arm="arm3", ridge=common.RIDGE), device=device
    )
    demand, residual = semi.demand_and_residual(cache=cache, site=site, fit=real_fit)
    rows: list[dict[str, object]] = []
    for draw in range(draws):
        resampled = semi.resample_by_month(
            residual=residual, end_times=cache.end_times, seed=1000 * draw + 129
        )
        target = semi.semi_synthetic_cache(
            cache=cache, site=site, demand=demand, resampled=resampled
        )
        placebo = semi.placebo_sky(cache=target, site=site)
        for arm in arm_names:
            rows += draw_rows(
                cache=target,
                placebo_cache=placebo,
                site=site,
                arm=arm,
                increment_share=increment_share,
                draw=draw,
                injected_points=injected_points,
                device=device,
            )
        print(f"{site} draw {draw}: done", flush=True)
    return rows


def main() -> None:
    """Run the semi-synthetic stage and write `work/semi.parquet`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sites", nargs="+", default=list(common.PILOT_SITES))
    parser.add_argument("--draws", type=int, default=N_DRAWS)
    parser.add_argument("--arms", nargs="+", default=["arm2", "arm3"])
    parser.add_argument("--increment-share", type=float, default=0.0)
    parser.add_argument("--out", default="semi.parquet")
    arguments = parser.parse_args()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = common.load_cache()
    rows: list[dict[str, object]] = []
    for site in arguments.sites:
        rows += run_site(
            cache=cache,
            site=site,
            draws=arguments.draws,
            device=arguments.device,
            arm_names=arguments.arms,
            increment_share=arguments.increment_share,
        )
    table = pl.DataFrame(rows)
    table.write_parquet(common.WORK_DIR / arguments.out)
    summary = (
        table.group_by("site", "arm", "sky")
        .agg(
            recovery=pl.col("recovery_mean").mean(),
            recovery_sd=pl.col("recovery_mean").std(),
            drift_5y=pl.col("drift_5y").mean(),
            drift_5y_sd=pl.col("drift_5y").std(),
            false_capacity_last=pl.col("false_capacity_last").mean(),
        )
        .sort("site", "arm", "sky")
    )
    with pl.Config(tbl_rows=50, tbl_cols=12):
        print(summary)


if __name__ == "__main__":
    main()
