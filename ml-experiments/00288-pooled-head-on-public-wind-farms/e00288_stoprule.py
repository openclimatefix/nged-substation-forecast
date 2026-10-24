"""Experiment 00288's second stop rule: the base model against climatology on the public farms.

Prints, per farm, the quantile score of R0 and of the climatology at lead days 1 and 2 on the
scored runs, and how many farms R0 loses on. The plan stops when R0 is worse on more than a
quarter of the farms.
"""

import e00288_climatology as climatology
import e00288_common as common
import polars as pl

CLEAN_DIR = common.WORK_DIR / "clean"
BASE_DIR = common.WORK_DIR / "base"
MAX_WORSE_SHARE = 0.25


def main() -> None:
    """Print the per-farm comparison and the stop-rule verdict."""
    cases = pl.read_parquet(BASE_DIR / "cases_scored.parquet")
    base = pl.read_parquet(BASE_DIR / "R0_scored_hall_seed0.parquet")
    actual = pl.read_parquet(CLEAN_DIR / "actual_centred.parquet").rename(
        {"elexon_bmu_id": "farm", "actual_norm": "target"}
    )
    table = climatology.climatology_table(training=actual)
    merged = cases.join(base, on=["farm", "power_fcst_init_time", "valid_time"])
    reference = climatology.climatology_quantiles(cases=merged, table=table)
    target = merged["actual_norm"].to_numpy().astype(float)
    merged = merged.with_columns(
        r0=pl.Series(
            common.quantile_score(
                quantiles=merged.select(common.LEVEL_COLUMNS).to_numpy().astype(float),
                actual=target,
            )
        )
        * common.POINTS_PER_FRACTION,
        climatology=pl.Series(common.quantile_score(quantiles=reference, actual=target))
        * common.POINTS_PER_FRACTION,
    )
    per_farm = (
        merged.filter(pl.col("lead_day").is_between(1, 2))
        .group_by("farm")
        .agg(pl.col("r0").mean(), pl.col("climatology").mean())
        .with_columns(r0_loses=pl.col("r0") > pl.col("climatology"))
        .sort("farm")
    )
    pl.Config.set_tbl_rows(60)
    print(per_farm)
    losing = int(per_farm["r0_loses"].sum())
    print(f"R0 loses to climatology on {losing} of {per_farm.height} farms")
    print(per_farm.select(pl.col("r0").mean(), pl.col("climatology").mean()))
    if losing > MAX_WORSE_SHARE * per_farm.height:
        raise SystemExit("stop rule: R0 is worse than climatology on more than a quarter of farms")


if __name__ == "__main__":
    main()
