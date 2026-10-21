"""The shape-mismatch ceiling on recovery: how much of a farm a perfect fit of the shape finds.

If the only difference between an injected farm and the model's solar term were the shape, the
fitted capacity would be the least-squares scale of the farm's output on the derated unit shape.
Its ratio to the injected capacity is the highest recovery any capacity fit of that shape can reach
once the demand terms absorb nothing. The ceiling is read over the rows of season-year 2024 that
train the fit.

    uv run --no-sync python ml-experiments/00157-*/e00157_ceiling.py
"""

from dataclasses import replace

import e00157_common as common
import numpy as np
import polars as pl
from studies.dp_gam import GamSpec, derated_unit_shape, prepare_fit


def main() -> None:
    """Write `work/ceiling.parquet`: one row per Primary and farm."""
    cache = common.load_cache()
    rows = []
    for site in common.ALL_SITES:
        features = cache.sites[site]
        for farm in common.FARMS:
            norm = common.farm_norm(cache=cache, farm=farm)
            masked = replace(
                cache,
                sites={**cache.sites, site: replace(features, y=features.y - norm)},
            )
            _, index = prepare_fit(
                cache=masked,
                site=site,
                spec=GamSpec(ridge=common.RIDGE, capacity="fitted"),
                cut=common.ORIGIN,
                window=common.TRAINING_WINDOW,
                season_years=common.SEASON_YEARS,
            )
            year = cache.calendar.season_year[index] == common.LATEST_SEASON_YEAR
            rows_in_year = index[year]
            shape = derated_unit_shape(features=features, index=rows_in_year)
            rows.append(
                {
                    "site": site,
                    "farm": farm,
                    "ceiling": float(np.sum(norm[rows_in_year] * shape) / np.sum(shape**2)),
                }
            )
    table = pl.DataFrame(rows)
    table.write_parquet(common.WORK_DIR / "ceiling.parquet")
    print(table.group_by("site").agg(pl.col("ceiling").median()).sort("site"))


if __name__ == "__main__":
    main()
