"""Write the two reference arms of experiment 00230: climatology and persistence.

Climatology: the empirical 11 quantiles of the observed fraction of capacity in each cell of
(month of the valid time, half-hour of the day, weekend or not), from the training window of
`e00230_common.observed_training_power` (valid times before 2025-06-17). It has no seed, so its rows
repeat once per seed. Persistence: the last observed fraction at least 3.5 hours before the ENS
run's time, used as a point forecast at all 11 levels, for cases up to 6 hours after the run only.

Output: `work/arms/CLIM_spread.parquet` and `work/arms/PERS_spread.parquet`.

    uv run --no-sync python ml-experiments/00230-wind-physics-core-part-1/e00230_references.py
"""

import e00230_common as common
import numpy as np
import polars as pl
from studies.power import scan_power

PERSISTENCE_MAX_LEAD_HOURS = 6.0
CELL = ("time_series_id", "month", "half_hour", "weekend")


def cell_columns() -> list[pl.Expr]:
    """Return the expressions that give a row's climatology cell from its `valid_time`."""
    return [
        pl.col("valid_time").dt.month().alias("month"),
        (pl.col("valid_time").dt.hour() * 2 + pl.col("valid_time").dt.minute() // 30).alias(
            "half_hour"
        ),
        (pl.col("valid_time").dt.weekday() >= 6).alias("weekend"),
    ]


def climatology(*, sites: pl.DataFrame, cases: pl.DataFrame) -> pl.DataFrame:
    """Return the climatology quantiles of every case."""
    training = common.observed_training_power(sites=sites).with_columns(cell_columns())
    table = training.group_by(*CELL).agg(
        *[
            pl.col("target").quantile(float(level), "linear").alias(name)
            for level, name in zip(common.LEVELS, common.LEVEL_COLUMNS, strict=True)
        ]
    )
    joined = (
        cases.with_columns(cell_columns())
        .join(table, on=list(CELL), how="left", validate="m:1")
        .select(*common.KEY_COLUMNS, *common.LEVEL_COLUMNS)
    )
    if joined[common.LEVEL_COLUMNS[0]].null_count():
        msg = "some cases fall in a climatology cell with no training rows"
        raise ValueError(msg)
    return joined


def persistence(*, sites: pl.DataFrame, cases: pl.DataFrame) -> pl.DataFrame:
    """Return the persistence quantiles of every case up to `PERSISTENCE_MAX_LEAD_HOURS` ahead."""
    power = (
        scan_power()
        .filter(pl.col("time_series_id").is_in(sites["time_series_id"].to_list()))
        .select("time_series_id", "time", "power")
        .collect()
        .join(sites.select("time_series_id", "effective_capacity_mw"), on="time_series_id")
        .select(
            "time_series_id",
            "time",
            observed=pl.col("power") / pl.col("effective_capacity_mw"),
        )
        .drop_nulls()
        .sort("time_series_id", "time")
    )
    near = (
        cases.filter(pl.col("lead_hours") <= PERSISTENCE_MAX_LEAD_HOURS)
        .with_columns(known_at=pl.col("power_fcst_init_time") - pl.duration(minutes=210))
        .sort("known_at")
        .join_asof(
            power.rename({"time": "known_time"}).sort("known_time"),
            left_on="known_at",
            right_on="known_time",
            by="time_series_id",
            strategy="backward",
        )
    )
    return near.select(
        *common.KEY_COLUMNS,
        *[pl.col("observed").cast(pl.Float32).alias(name) for name in common.LEVEL_COLUMNS],
    ).drop_nulls()


def main() -> None:
    """Write both reference arms."""
    sites = common.site_table()
    cases = common.cases_table(sites=sites)
    out = common.WORK_DIR / "arms"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("CLIM", climatology(sites=sites, cases=cases)),
        ("PERS", persistence(sites=sites, cases=cases)),
    ):
        seeded = pl.concat(
            [frame.with_columns(seed=pl.lit(s, dtype=pl.Int32)) for s in common.SEEDS]
        )
        target = out / f"{name}_spread.parquet"
        seeded.write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
        print(name, frame.height, flush=True)
        if name == "PERS":
            print(
                "persistence quantile spread check", np.unique(frame[common.LEVEL_COLUMNS[0]]).size
            )


if __name__ == "__main__":
    main()
