"""Re-apply the spread factor to every stored mixture, so the arms need no refit.

Reads `work/arms/<arm>_mixture.parquet` (keys, `seed`, and the 11 mixture quantiles) and writes
`work/arms/<arm>_spread.parquet` with the spread factor of `e00230_common.apply_spread`.

    uv run --no-sync python ml-experiments/00230-wind-physics-core-part-1/e00230_respread.py [ARM]
"""

import sys

import e00230_common as common
import numpy as np
import polars as pl

ARM_DIR = common.WORK_DIR / "arms"


def main() -> None:
    """Write the spread-adjusted forecasts of each requested arm, or of every stored mixture."""
    sites = common.site_table()
    cases = common.cases_table(sites=sites)
    farm_order = sites["time_series_id"].to_list()
    ordered = pl.concat([cases.filter(pl.col("time_series_id") == s) for s in farm_order])
    names = sys.argv[1:] or sorted(
        p.name.removesuffix("_mixture.parquet") for p in ARM_DIR.glob("*_mixture.parquet")
    )
    for name in names:
        mixture = pl.read_parquet(ARM_DIR / f"{name}_mixture.parquet")
        pieces = []
        for seed in sorted(mixture["seed"].unique().to_list()):
            part = mixture.filter(pl.col("seed") == seed)
            joined = ordered.join(
                part, on=list(common.KEY_COLUMNS), how="left", validate="1:1", maintain_order="left"
            )
            values = joined.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64)
            spread = common.apply_spread(cases=joined, mixture=values)
            frame = joined.select(*common.KEY_COLUMNS).hstack(
                pl.DataFrame(spread.astype(np.float32), schema=list(common.LEVEL_COLUMNS))
            )
            pieces.append(frame.with_columns(seed=pl.lit(seed, dtype=pl.Int32)))
        target = ARM_DIR / f"{name}_spread.parquet"
        pl.concat(pieces).write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
        print(name, "done", flush=True)


if __name__ == "__main__":
    main()
