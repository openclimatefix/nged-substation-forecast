"""Build the night table of every fitted series and run test 0 (feasibility).

Writes `night_tables.parquet` (all nights of the 14 fitted series and the two generator-linked
series' ids are never read) and prints, per series, the winters with enough present nights.
"""

import e00123_common as common
import polars as pl


def main() -> None:
    """Build and store the night tables, then print the feasibility table."""
    infos = common.series_infos()
    ids = [infos[site].time_series_id for site in common.FITTED_SERIES]
    power = common.load_power(time_series_ids=ids)
    frames = []
    for site in common.FITTED_SERIES:
        info = infos[site]
        own = power.filter(pl.col("time_series_id") == info.time_series_id).select("time", "power")
        frames.append(common.build_night_table(info=info, power=own))
    table = pl.concat(frames)
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    table.write_parquet(common.WORK_DIR / "night_tables.parquet")
    allowed = list(common.allowed_winters(winters=common.WINTERS))
    usable = table.filter(
        (pl.col("season") == "gmt") & ~pl.col("excluded") & pl.col("winter").is_in(allowed)
    )
    feasibility = (
        usable.group_by("site", "winter")
        .agg(
            nights=pl.len(),
            with_d=pl.col("d").is_not_null().sum(),
            with_temperature=pl.col("daily_mean_c").is_not_null().sum(),
        )
        .with_columns(share=pl.col("with_d") / pl.col("nights"))
        .sort("site", "winter")
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(feasibility)
        print(table.group_by("site").agg(first=pl.col("evening").min()).sort("site"))


if __name__ == "__main__":
    main()
