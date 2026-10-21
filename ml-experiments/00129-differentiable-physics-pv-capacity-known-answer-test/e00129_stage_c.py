"""Stage C of experiment 129: the known-answer grid and the coverage cells on the real Primaries.

**Known-answer grid (`kat`).** For each decision Primary and arm, the script fits the convex model
without an injection, and then with each source added at 20, 50, and 100% of the Primary's pinned
capacity, with and without the day-long cloud-driven load. The sources are the five real farms, the
synthetic east-west plant, and the synthetic south-west plant. Recovery is the difference of two
fits divided by the injected capacity. The table is `work/kat_<arm>.parquet`.

**Coverage cells (`coverage`).** For each Primary, plant class (matched with a rotating farm,
east-west, south-west), and load, the script injects the source at 100% into a semi-synthetic
target of the Primary (true capacity zero before the injection), fits the arm under a stratified
month-block bootstrap, and builds the 80% range of the last complete season-year as the bootstrap
capacities divided by the recovery ratios of the known-answer grid at other Primaries and with other
farms. The range covers the truth when the injected capacity lies inside it. The table is
`work/coverage_<arm>.parquet`.

An arm is written `arm2`, `arm3`, `arm4@<share>` (arm 4 with that increment-penalty share).

    uv run --no-sync python ml-experiments/00129-*/e00129_stage_c.py --parts kat coverage \
        --arms arm2 arm3 arm4@0.1 [--sites D5 D10 D11 D14 D16] [--device cuda]
"""

import argparse
from typing import Final

import e00129_arms as arms
import e00129_common as common
import e00129_semi as semi
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache

FRACTIONS: Final[tuple[float, ...]] = (0.2, 0.5, 1.0)
SYNTHETIC_SOURCES: Final[tuple[str, ...]] = ("east_west", "south_west")
N_BOOTSTRAP: Final[int] = 40
LAST_YEAR: Final[int] = common.COMPLETE_SEASON_YEARS[-1]
RANGE_LEVELS: Final[tuple[float, float]] = (0.10, 0.90)
"""The 80% range's quantile levels."""
ESTIMABLE_SHARE: Final[float] = 0.01
"""A season-year is estimable when the capacity fitted with the injection exceeds this share of
the injected capacity; a season-year with no usable row fits to zero."""


def parse_arm(*, name: str) -> tuple[arms.ArmType, float]:
    """Return the arm and its increment share from a name such as `arm4@0.1`."""
    arm, _, share = name.partition("@")
    if arm not in arms.ARMS:
        msg = f"unknown arm {arm!r}"
        raise ValueError(msg)
    return arm, float(share) if share else 0.0  # ty: ignore[invalid-return-type]


def plant_class(*, source: str) -> str:
    """Return `matched` for a real farm and the source's own name for a synthetic plant."""
    return source if source in SYNTHETIC_SOURCES else "matched"


def source_shapes(*, cache: FeatureCache, site: str) -> dict[str, np.ndarray]:
    """Return each source's output per unit of capacity on the grid: five farms and two plants."""
    shapes = {farm: common.farm_norm(cache=cache, farm=farm) for farm in common.FARMS}
    shapes["east_west"] = common.synthetic_plant_shape(
        cache=cache, site=site, plants=common.EAST_WEST_PLANTS
    )
    shapes["south_west"] = common.synthetic_plant_shape(
        cache=cache, site=site, plants=common.SOUTH_WEST_PLANTS
    )
    return shapes


def known_answer_rows(
    *, cache: FeatureCache, site: str, arm_name: str, device: str
) -> list[dict[str, object]]:
    """Return one row per injection, load, fraction, and estimable season-year at one Primary."""
    arm, share = parse_arm(name=arm_name)
    pinned = common.pinned_points(cache=cache, site=site)
    shapes = source_shapes(cache=cache, site=site)
    base = arms.fitted_capacity(
        arm=arm, cache=cache, site=site, device=device, increment_share=share
    )
    rows: list[dict[str, object]] = []
    for source, shape in shapes.items():
        for fraction in FRACTIONS:
            injected = fraction * pinned
            for load in (False, True):
                cloud = (
                    common.cloud_load(cache=cache, site=site, capacity_points=injected)
                    if load
                    else None
                )
                injected_cache = common.inject(
                    cache=cache, site=site, output=injected * shape, load=cloud
                )
                capacity = arms.fitted_capacity(
                    arm=arm,
                    cache=injected_cache,
                    site=site,
                    device=device,
                    increment_share=share,
                )
                rows.extend(
                    {
                        "site": site,
                        "arm": arm_name,
                        "source": source,
                        "plant": plant_class(source=source),
                        "fraction": fraction,
                        "cloud_load": load,
                        "season_year": year,
                        "injected_points": injected,
                        "capacity_base": base[year],
                        "capacity_injected": capacity[year],
                        "recovery": (capacity[year] - base[year]) / injected,
                    }
                    for year in common.COMPLETE_SEASON_YEARS
                    if capacity[year] > ESTIMABLE_SHARE * injected
                )
        print(f"{site} {arm_name} {source}: done", flush=True)
    return rows


def calibration_ratios(
    *, table: pl.DataFrame, site: str, source: str, arm_name: str, season_year: int = LAST_YEAR
) -> np.ndarray:
    """Return the recovery ratios at 100% in a season-year, at other Primaries and farms.

    Args:
        table: The known-answer table of every Primary and arm.
        site: The Primary whose range is built; its own injections are left out.
        source: The source being tested; a real farm's own injections are left out.
        arm_name: The arm.
        season_year: The season-year whose recoveries are returned.

    Returns:
        The recovery of every remaining injection, with and without load, over plants.
    """
    held = table.filter(
        pl.col("arm") == arm_name,
        pl.col("fraction") == 1.0,
        pl.col("season_year") == season_year,
        pl.col("site") != site,
    )
    if plant_class(source=source) == "matched":
        held = held.filter(pl.col("source") != source)
    return held["recovery"].to_numpy()


def coverage_row(
    *,
    cache: FeatureCache,
    target: FeatureCache,
    site: str,
    arm_name: str,
    source: str,
    load: bool,
    ratios: np.ndarray,
    device: str,
) -> dict[str, object]:
    """Return one coverage cell: the range of the last complete year, and whether it covers."""
    arm, share = parse_arm(name=arm_name)
    pinned = common.pinned_points(cache=cache, site=site)
    shape = source_shapes(cache=cache, site=site)[source]
    cloud = common.cloud_load(cache=cache, site=site, capacity_points=pinned) if load else None
    injected_cache = common.inject(cache=target, site=site, output=pinned * shape, load=cloud)
    index = arms.training_index(cache=injected_cache, site=site)
    weights = arms.stratified_month_weights(
        cache=injected_cache, index=index, n_resamples=N_BOOTSTRAP, seed=129
    )
    fit = arms.fit_arm(
        arm=arm,
        cache=injected_cache,
        site=site,
        device=device,
        increment_share=share,
        weights=weights,
    )
    boot = np.array([c[LAST_YEAR] for c in arms.capacities(fit=fit)])
    draws = (boot[:, None] / ratios[None, :]).ravel()
    low, high = np.quantile(draws, RANGE_LEVELS)
    return {
        "site": site,
        "arm": arm_name,
        "source": source,
        "plant": plant_class(source=source),
        "cloud_load": load,
        "truth": pinned,
        "estimate": float(np.median(boot)),
        "range_low": float(low),
        "range_high": float(high),
        "covered": bool(low <= pinned <= high),
        "width_share": float((high - low) / pinned),
    }


def coverage_rows(
    *, cache: FeatureCache, sites: list[str], arm_name: str, kat: pl.DataFrame, device: str
) -> list[dict[str, object]]:
    """Return the coverage cells of an arm: three plant classes and two loads at every Primary."""
    rows: list[dict[str, object]] = []
    for position, site in enumerate(sites):
        real_fit = arms.fit_arm(arm="arm3", cache=cache, site=site, device=device)
        demand, residual = semi.demand_and_residual(cache=cache, site=site, fit=real_fit)
        resampled = semi.resample_by_month(
            residual=residual, end_times=cache.end_times, seed=129 + position
        )
        target = semi.semi_synthetic_cache(
            cache=cache, site=site, demand=demand, resampled=resampled
        )
        sources = (common.FARMS[position % len(common.FARMS)], *SYNTHETIC_SOURCES)
        for source in sources:
            ratios = calibration_ratios(table=kat, site=site, source=source, arm_name=arm_name)
            rows.extend(
                coverage_row(
                    cache=cache,
                    target=target,
                    site=site,
                    arm_name=arm_name,
                    source=source,
                    load=load,
                    ratios=ratios,
                    device=device,
                )
                for load in (False, True)
            )
        print(f"{site} {arm_name} coverage: done", flush=True)
    return rows


def main() -> None:
    """Run the requested parts of stage C and write their tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs="+", default=["kat", "coverage"])
    parser.add_argument("--arms", nargs="+", default=["arm2", "arm3", "arm4@0.1"])
    parser.add_argument("--sites", nargs="+", default=list(common.DECISION_SITES))
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = common.load_cache()
    if "kat" in arguments.parts:
        for arm_name in arguments.arms:
            rows = [
                row
                for site in arguments.sites
                for row in known_answer_rows(
                    cache=cache, site=site, arm_name=arm_name, device=arguments.device
                )
            ]
            pl.DataFrame(rows).write_parquet(common.WORK_DIR / f"kat_{arm_name}.parquet")
    if "coverage" in arguments.parts:
        for arm_name in arguments.arms:
            kat = pl.read_parquet(common.WORK_DIR / f"kat_{arm_name}.parquet")
            rows = coverage_rows(
                cache=cache,
                sites=arguments.sites,
                arm_name=arm_name,
                kat=kat,
                device=arguments.device,
            )
            pl.DataFrame(rows).write_parquet(common.WORK_DIR / f"coverage_{arm_name}.parquet")


if __name__ == "__main__":
    main()
