"""Stage 2a of experiment 269: the capacity of the unmetered solar at each substation, two shapes.

For each substation, the script fits experiment 129's convex model (capacity increments per
season-year, rows before 2026-07-01) to the real reading twice: with the physical fleet shape and
with the pooled empirical shape (the mean learned response of all six metered farms read at the
substation's satellite irradiance). Each capacity is divided by that shape's own recovery in
experiment 269's injection test (capacity ratio, season-year 2024, median over injected farms),
as experiment 129 divided by the recovery of its known-answer grid. Experiment 129's published
capacity and the pinned capacity of experiment 45 are written beside them as context.

    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00269-*/e00269_real.py
"""

from dataclasses import replace
from typing import Final

import e00269_common as common
import e00269_fit as fit
import numpy as np
import polars as pl
import torch
from studies.dp_gam import GamSpec, penalty_matrix, prepare_fit
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

PUBLISHED_PATH: Final = (
    common.ML_EXPERIMENTS_DATA_DIR
    / "00129-differentiable-physics-pv-capacity-known-answer-test"
    / "work"
    / "real_arm2_to_july.parquet"
)
SHAPES: Final[tuple[str, ...]] = ("physical", "empirical")


def capacities_points(*, cache: fit.FeatureCache, site: str, shape: np.ndarray) -> np.ndarray:
    """Return the fitted capacity in points of effective capacity at each season-year."""
    features = cache.sites[site]
    shaped = replace(cache, sites={**cache.sites, site: replace(features, pv_unit_shape=shape)})
    spec = GamSpec(ridge=common.RIDGE, capacity="fitted")
    layout, index = prepare_fit(
        cache=shaped,
        site=site,
        spec=spec,
        cut=common.CUT,
        window=common.TRAINING_WINDOW,
        season_years=common.SEASON_YEARS,
    )
    zeros = np.zeros(len(features.y))
    stats = fit.monthly_stats(
        cache=shaped, site=site, layout=layout, index=index, output=zeros, norm=zeros
    )
    spec_cost = penalty_matrix(layout=layout, n_rows=len(index))
    theta, _, _, _ = fit.full_fit(
        stats=stats, layout=layout, weights=np.ones((1, len(stats.months))), spec_cost=spec_cost
    )
    return np.cumsum(theta[0][layout.slices["capacity"]])


def main() -> None:
    """Write `work/real_capacities.parquet`."""
    torch.set_num_threads(1)
    cache = common.load_cache()
    injection = pl.concat(
        [pl.read_parquet(common.WORK_DIR / f"result_{s}.parquet") for s in common.ALL_SITES]
    )
    divisors = (
        injection.filter(pl.col("stat") == "capacity_ratio", pl.col("b") == 0)
        .group_by("site", "arm")
        .agg(pl.col("value").median())
    )
    published = pl.read_parquet(PUBLISHED_PATH)
    rows = []
    for site in common.ALL_SITES:
        table = cache.sites[site].table
        physical = cache.sites[site].pv_unit_shape
        empirical = common.learned_unit_shape(cache=cache, site=site, farms=common.ALL_FARMS)
        both = np.isfinite(physical) & np.isfinite(empirical)
        pinned = pinned_capacity(vintages=read_vintages(), table=table, cut=common.CUT).held_points
        for name, shape, arm in (
            ("physical", physical, "cams_same_rows"),
            ("empirical", empirical, "pooled_plant"),
        ):
            capacity = capacities_points(
                cache=cache, site=site, shape=np.where(both, shape, np.nan)
            )
            divisor = float(
                divisors.filter(pl.col("site") == site, pl.col("arm") == arm)["value"].item()
            )
            for k, points in enumerate(capacity):
                year = common.SEASON_YEARS[k]
                old = published.filter(pl.col("site") == site, pl.col("season_year") == year)
                rows.append(
                    {
                        "site": site,
                        "shape": name,
                        "season_year": year,
                        "raw_mw": float(points) * table.effective_capacity_mw / 100.0,
                        "divisor": divisor,
                        "estimate_mw": float(points)
                        / divisor
                        * table.effective_capacity_mw
                        / 100.0,
                        "pinned_mw": pinned * table.effective_capacity_mw / 100.0,
                        "published_mw": float(old["estimate_mw"].item()) if old.height else None,
                        "published_low_mw": float(old["range_low_mw"].item())
                        if old.height
                        else None,
                        "published_high_mw": float(old["range_high_mw"].item())
                        if old.height
                        else None,
                    }
                )
        print(f"{site} done", flush=True)
    pl.DataFrame(rows).write_parquet(common.WORK_DIR / "real_capacities.parquet")


if __name__ == "__main__":
    main()
