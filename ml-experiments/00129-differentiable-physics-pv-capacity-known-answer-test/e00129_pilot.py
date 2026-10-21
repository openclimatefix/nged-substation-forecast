"""Stage B pilot of experiment 129: arm 2 against arm 3, and the ridge, on two Primaries.

For each pilot Primary the script fits the convex model without an injection, and then with the
output of each real farm and of the synthetic east-west plant added to the Primary's power at 100%
of its pinned capacity, with and without a day-long cloud-driven load. Recovery is the difference
of two fits: the fitted capacity of a season-year with the injection minus the fitted capacity
without it, divided by the injected capacity. It measures bias only, because the fit is nearly
linear in its target.

Arm 2 is the monotone season-year capacity (`k_base` of experiment 110) and arm 3 adds
experiment 110's sun-blind terms (`k_ii`). The ridge is shared by the demand and capacity blocks of
`studies.dp_gam`, so it is varied for arm 2 only, over `RIDGES`.

    uv run --no-sync python ml-experiments/00129-*/e00129_pilot.py [--device cuda] [--sites D10 D16]
"""

import argparse
from dataclasses import dataclass
from typing import Final, Literal

import e00129_arms as arms
import e00129_common as common
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache
from studies.dp_gam import GamSpec
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

RIDGES: Final[tuple[float, ...]] = (0.01, 0.1, 1.0)
"""The ridge values compared for arm 2, whose value of 0.1 is experiment 104's."""
ESTIMABLE_SHARE: Final[float] = 0.01
"""A season-year is estimable when the capacity fitted with the injection exceeds this share of
the injected capacity; a season-year with no usable row fits to zero."""
PlantType = Literal["matched", "east_west"]
"""The matched plant is a real farm's output; the east-west plant is synthetic."""


@dataclass(frozen=True)
class Scenario:
    """One injection.

    Attributes:
        plant: Whether the output is a real farm's (matched) or the synthetic east-west plant's.
        source: The farm's label, or `synthetic` for the east-west plant.
        cloud_load: True to add the day-long cloud-driven load beside the output.
    """

    plant: PlantType
    source: str
    cloud_load: bool


def scenarios() -> list[Scenario]:
    """Return every pilot injection: each farm and the east-west plant, with and without load."""
    sources: list[tuple[PlantType, str]] = [("matched", farm) for farm in common.FARMS]
    sources.append(("east_west", "synthetic"))
    return [Scenario(plant, source, load) for plant, source in sources for load in (False, True)]


def fitted_capacity(
    *, cache: FeatureCache, site: str, spec: GamSpec, device: str
) -> dict[int, float]:
    """Return the fitted capacity of each season-year, in points of effective capacity."""
    return common.fit_site(cache=cache, site=site, spec=spec, device=device).capacity_points()


def injection_for(
    *, cache: FeatureCache, site: str, scenario: Scenario, capacity_points: float
) -> FeatureCache:
    """Return the cache with the scenario's output, and its load if any, added to the site."""
    if scenario.plant == "matched":
        output = capacity_points * common.farm_norm(cache=cache, farm=scenario.source)
    else:
        output = capacity_points * common.synthetic_plant_shape(
            cache=cache, site=site, plants=common.EAST_WEST_PLANTS
        )
    load = (
        common.cloud_load(cache=cache, site=site, capacity_points=capacity_points)
        if scenario.cloud_load
        else None
    )
    return common.inject(cache=cache, site=site, output=output, load=load)


def pilot_rows(*, cache: FeatureCache, site: str, device: str) -> list[dict[str, object]]:
    """Return one table row per arm, ridge, scenario, and complete season-year for one Primary."""
    injected_points = pinned_capacity(
        vintages=read_vintages(), table=cache.sites[site].table, cut=common.DATA_CUT
    ).held_points
    cases = [("arm2", ridge) for ridge in RIDGES] + [("arm3", common.RIDGE)]
    injected = {
        scenario: injection_for(
            cache=cache, site=site, scenario=scenario, capacity_points=injected_points
        )
        for scenario in scenarios()
    }
    rows: list[dict[str, object]] = []
    for arm, ridge in cases:
        spec = arms.arm_spec(arm=arm, ridge=ridge)
        base = fitted_capacity(cache=cache, site=site, spec=spec, device=device)
        for scenario, injected_cache in injected.items():
            with_injection = fitted_capacity(
                cache=injected_cache, site=site, spec=spec, device=device
            )
            rows.extend(
                {
                    "site": site,
                    "arm": arm,
                    "ridge": ridge,
                    "plant": scenario.plant,
                    "source": scenario.source,
                    "cloud_load": scenario.cloud_load,
                    "season_year": year,
                    "injected_points": injected_points,
                    "capacity_base": base[year],
                    "capacity_injected": with_injection[year],
                    "recovery": (with_injection[year] - base[year]) / injected_points,
                }
                for year in common.COMPLETE_SEASON_YEARS
                if with_injection[year] > ESTIMABLE_SHARE * injected_points
            )
        print(f"{site} {arm} ridge {ridge}: done", flush=True)
    return rows


def summarise(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return the median recovery over farms and complete season-years, per site, arm, and case."""
    return (
        table.group_by("site", "arm", "ridge", "plant", "cloud_load")
        .agg(
            median_recovery=pl.col("recovery").median(),
            low=pl.col("recovery").quantile(0.1),
            high=pl.col("recovery").quantile(0.9),
            n=pl.len(),
        )
        .sort("site", "arm", "ridge", "plant", "cloud_load")
    )


def main() -> None:
    """Run the pilot and write `work/pilot.parquet`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sites", nargs="+", default=list(common.PILOT_SITES))
    arguments = parser.parse_args()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = common.load_cache()
    rows = [
        row
        for site in arguments.sites
        for row in pilot_rows(cache=cache, site=site, device=arguments.device)
    ]
    table = pl.DataFrame(rows)
    table.write_parquet(common.WORK_DIR / "pilot.parquet")
    with pl.Config(tbl_rows=100, tbl_cols=12):
        print(summarise(table=table))
    print(np.round(table["recovery"].median(), 3))


if __name__ == "__main__":
    main()
