"""Find each metered farm's outages and write them (farms A to F) to the work folder.

An outage is a run of at least `MIN_OUTAGE_DAYS` days, each with at least `BAD_DAY_SHARE` of its
clear-sky half-hours flagged `zero_in_clear_sky` in experiment 179's cleaned series, or missing.

    uv run --no-sync python ml-experiments/00206-*/e00206_outages.py
"""

import e00206_common as common
import numpy as np
import polars as pl


def farm_days(*, grid: np.ndarray, farm: pl.DataFrame, cleaned: pl.DataFrame) -> pl.DataFrame:
    """Return one row per day for one farm: clear-sky count, flagged-zero count, missing count.

    Args:
        grid: The cache's period-end stamps as naive UTC `datetime64[us]`.
        farm: A one-row frame from `farm_table`.
        cleaned: The cleaned farm series, all farms.

    Returns:
        Columns `day`, `n_clear`, `n_zero`, `n_missing`, restricted to commissioned days.
    """
    row = farm.row(0, named=True)
    clear = common.clear_sky_ghi(
        end_times=grid, latitude=row["latitude"], longitude=row["longitude"]
    )
    frame = pl.DataFrame({"time": pl.Series(grid).dt.replace_time_zone("UTC"), "clear": clear})
    own = (
        cleaned.filter(pl.col("time_series_id") == row["time_series_id"])
        .select("time", "drop_reason")
        .with_columns(present=pl.lit(value=True))
    )
    joined = frame.join(own, on="time", how="left", maintain_order="left").with_columns(
        present=pl.col("present").fill_null(value=False)
    )
    start = common.FARM_COMMISSIONED.get(row["site"])
    first = own["time"].min()
    joined = joined.filter(pl.col("time") >= first)
    if start is not None:
        joined = joined.filter(pl.col("time") >= pl.lit(start).str.to_datetime(time_zone="UTC"))
    joined = joined.filter(pl.col("time") <= own["time"].max())
    mid_day = (pl.col("time") - pl.duration(minutes=15)).dt.date().alias("day")
    return (
        joined.filter(pl.col("clear") > common.FARM_CLEAR_SKY_W_M2)
        .group_by(mid_day)
        .agg(
            n_clear=pl.len(),
            n_zero=(pl.col("drop_reason") == "zero_in_clear_sky").sum(),
            n_missing=(~pl.col("present")).sum(),
        )
        .sort("day")
    )


def find_outages(*, days: pl.DataFrame, site: str) -> list[dict]:
    """Group bad days into outage runs.

    Args:
        days: The output of `farm_days`.
        site: The farm's label.

    Returns:
        One dict per outage with its start and end day, length, and `zero` or `missing` class.
    """
    eligible = days.filter(pl.col("n_clear") >= common.MIN_CLEAR_HALF_HOURS_PER_DAY).with_columns(
        bad=(pl.col("n_zero") + pl.col("n_missing")) >= common.BAD_DAY_SHARE * pl.col("n_clear")
    )
    runs, current = [], []
    for row in eligible.iter_rows(named=True):
        if current and (row["day"] - current[-1]["day"]).days > common.MAX_NEUTRAL_GAP_DAYS:
            runs.append(current)
            current = []
        if row["bad"]:
            current.append(row)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    outages = []
    for run in runs:
        if len(run) < common.MIN_OUTAGE_DAYS:
            continue
        n_zero = sum(r["n_zero"] for r in run)
        n_missing = sum(r["n_missing"] for r in run)
        outages.append(
            {
                "farm": site,
                "start": run[0]["day"],
                "end": run[-1]["day"],
                "calendar_days": (run[-1]["day"] - run[0]["day"]).days + 1,
                "eligible_days": len(run),
                "kind": "zero" if n_zero >= n_missing else "missing",
                "n_zero": n_zero,
                "n_missing": n_missing,
            }
        )
    return outages


def main() -> None:
    """Write `work/outages.parquet` and `work/farm_days_<farm>.parquet`."""
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = common.load_cache()
    cleaned = pl.read_parquet(common.CLEANED_FARMS_PATH)
    farms = common.farm_table()
    rows = []
    for farm in farms.iter_rows(named=True):
        one = farms.filter(pl.col("site") == farm["site"])
        days = farm_days(grid=cache.end_times, farm=one, cleaned=cleaned)
        days.write_parquet(common.WORK_DIR / f"farm_days_{farm['site']}.parquet")
        found = find_outages(days=days, site=farm["site"])
        rows += found
        kinds = [o["kind"] for o in found]
        print(
            farm["site"],
            f"outages {len(found)} (zero {kinds.count('zero')}, missing {kinds.count('missing')})",
        )
    table = pl.DataFrame(rows)
    table.write_parquet(common.WORK_DIR / "outages.parquet")
    print(table.sort("farm", "start"))


if __name__ == "__main__":
    main()
