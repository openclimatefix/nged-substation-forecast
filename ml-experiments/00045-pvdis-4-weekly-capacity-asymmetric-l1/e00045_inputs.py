"""Build experiment 00045's per-Primary half-hourly table, week table, and farm-week table.

The frame of experiment 00039 is read and never rebuilt. Half-hours inside a logged switching
closure, padded by 24 hours on each side, are marked `masked` and take no part in any fit. A week
with fewer than 3 usable days is `unobserved`. A farm-week in which the farm's output is missing on
more than 2 days is dropped from that farm's injection units. Farm output is stored divided by the
farm's own 99th percentile, so no farm megawatt value is written.

    POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
        uv run python ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1/e00045_inputs.py
"""

from datetime import timedelta

import e00045_common as common
import numpy as np
import polars as pl

LAST_DAY_EXCLUSIVE = common.LAST_DATA_END.date()


def load_frame() -> pl.DataFrame:
    """Return experiment 00039's frame restricted to the 361 weeks, after the row-count check.

    Returns:
        Every site's half-hours from the first Monday to Friday of the last week, with `week` and
        `day` columns.

    Raises:
        AssertionError: When the frame does not hold the row count experiment 00039 wrote.
    """
    frame = pl.read_parquet(common.FRAME_PATH)
    assert frame.height == common.FRAME_ROWS, f"frame has {frame.height} rows"
    return (
        frame.with_columns(
            week=common.week_expr(time=pl.col("time")), day=common.day_expr(time=pl.col("time"))
        )
        .filter(
            (pl.col("week") >= 0)
            & (pl.col("week") < common.N_WEEKS)
            & (pl.col("day") < LAST_DAY_EXCLUSIVE)
        )
        .sort("site", "time")
    )


def closure_intervals() -> pl.DataFrame:
    """Return each logged closure once per Primary it touches, with the padded mask window.

    Returns:
        Columns `site`, `mask_start`, and `mask_end`: the half-hours `tau` with
        `mask_start < tau <= mask_end` are masked (the closure's affected stamps, padded).
    """
    logged = pl.read_parquet(common.LOGGED_CLOSURES_PATH).select(
        "start_utc", "end_utc", "side_a", "side_b"
    )
    series_to_site = {series: site for site, series in common.PRIMARY_SERIES_IDS.items()}
    pad = timedelta(hours=common.MASK_PAD_HOURS)
    rows = [
        {
            "site": series_to_site[int(side)],
            "mask_start": row["start_utc"] - pad,
            "mask_end": row["end_utc"] + timedelta(minutes=30) + pad,
        }
        for row in logged.iter_rows(named=True)
        for side in (row["side_a"], row["side_b"])
        if side is not None and side.isdigit() and int(side) in series_to_site
    ]
    return pl.DataFrame(
        rows,
        schema={
            "site": pl.String,
            "mask_start": pl.Datetime("us", "UTC"),
            "mask_end": pl.Datetime("us", "UTC"),
        },
    )


def mask_flags(*, times: pl.Series, intervals: pl.DataFrame) -> np.ndarray:
    """Return True for each stamp inside any of `intervals` (mask_start < tau <= mask_end).

    Args:
        times: The stamps, UTC.
        intervals: Rows with `mask_start` and `mask_end`.

    Returns:
        A boolean array aligned with `times`.
    """
    stamp = times.dt.epoch("us").to_numpy()
    starts = intervals["mask_start"].dt.epoch("us").to_numpy()
    ends = intervals["mask_end"].dt.epoch("us").to_numpy()
    delta = np.zeros(stamp.size + 1, dtype=np.int64)
    order = np.argsort(stamp)
    sorted_stamp = stamp[order]
    first = np.searchsorted(sorted_stamp, starts, side="right")
    last = np.searchsorted(sorted_stamp, ends, side="right")
    np.add.at(delta, first, 1)
    np.add.at(delta, last, -1)
    covered_sorted = np.cumsum(delta[:-1]) > 0
    out = np.empty(stamp.size, dtype=bool)
    out[order] = covered_sorted
    return out


def primary_half_hourly(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the Primaries' half-hours with `masked`, `daylight`, and `usable` flags.

    A half-hour is usable when it is daylight (zenith under 75 degrees, CAMS shape above zero), has
    power and temperature, and is not masked.
    """
    intervals = closure_intervals()
    parts = []
    for site in common.PRIMARIES:
        rows = frame.filter(pl.col("site") == site)
        masked = mask_flags(times=rows["time"], intervals=intervals.filter(pl.col("site") == site))
        parts.append(rows.with_columns(masked=pl.Series(masked)))
    return (
        pl.concat(parts)
        .with_columns(
            series_id=pl.col("site").replace_strict(common.PRIMARY_SERIES_IDS),
            daylight=(pl.col("zenith") < common.MAX_ZENITH_DEG) & (pl.col("u") > 0.0),
        )
        .with_columns(
            usable=pl.col("daylight")
            & pl.col("power").is_not_null()
            & pl.col("temperature_2m").is_not_null()
            & ~pl.col("masked")
        )
        .drop("kind")
        .sort("site", "time")
    )


def expected_daylight_by_week() -> pl.DataFrame:
    """Return the daylight half-hours each week should hold, from solar geometry at the area.

    The point is the mean of the Primaries' areas to within a degree, which moves a week's count of
    half-hours under 75 degrees by at most a few.
    """
    from studies import solar

    stamps = pl.datetime_range(
        common.FIRST_WEEK_START + timedelta(minutes=30),
        common.LAST_DATA_END,
        interval="30m",
        time_zone="UTC",
        time_unit="us",
        eager=True,
    )
    zenith = solar.zenith(stamps=stamps, latitude=53.0, longitude=-0.1)
    return (
        pl.DataFrame({"time": stamps, "zenith_expected": zenith})
        .with_columns(week=common.week_expr(time=pl.col("time")))
        .filter(pl.col("zenith_expected") < common.MAX_ZENITH_DEG)
        .group_by("week")
        .agg(expected_daylight=pl.len())
        .sort("week")
    )


def week_table(*, half_hourly: pl.DataFrame) -> pl.DataFrame:
    """Return one row per Primary-week with coverage counts and the `unobserved` flag.

    Columns: `site`, `week`, `n_rows`, `n_daylight`, `n_usable`, `n_masked_days` (days holding a
    masked daylight half-hour), `n_usable_days`, `cams_dropped_share` (one minus present daylight
    half-hours over expected), and `unobserved` (fewer than 3 usable days). `observed_from` is the
    first week with at least 3 usable days, and `before_start` marks the weeks before it.
    """
    per_day = half_hourly.group_by("site", "week", "day").agg(
        usable_day=(pl.col("usable").sum() >= common.MIN_USABLE_HALF_HOURS_PER_DAY),
        masked_day=(pl.col("masked") & pl.col("daylight")).any(),
    )
    day_counts = per_day.group_by("site", "week").agg(
        n_usable_days=pl.col("usable_day").sum().cast(pl.Int32),
        n_masked_days=pl.col("masked_day").sum().cast(pl.Int32),
    )
    counts = half_hourly.group_by("site", "week").agg(
        n_rows=pl.len(),
        n_daylight=pl.col("daylight").sum().cast(pl.Int32),
        n_usable=pl.col("usable").sum().cast(pl.Int32),
    )
    grid = pl.DataFrame({"site": list(common.PRIMARIES)}).join(
        pl.DataFrame({"week": pl.int_range(common.N_WEEKS, eager=True, dtype=pl.Int32)}),
        how="cross",
    )
    table = (
        grid.join(counts, on=["site", "week"], how="left")
        .join(day_counts, on=["site", "week"], how="left")
        .join(expected_daylight_by_week(), on="week", how="left")
        .with_columns(
            pl.col("n_rows", "n_daylight", "n_usable", "n_usable_days", "n_masked_days").fill_null(
                0
            ),
            week_start=pl.lit(common.FIRST_WEEK_START) + pl.duration(weeks=pl.col("week")),
        )
        .with_columns(
            cams_dropped_share=(1.0 - pl.col("n_daylight") / pl.col("expected_daylight")).clip(
                lower_bound=0.0
            ),
            unobserved=pl.col("n_usable_days") < common.MIN_USABLE_DAYS_PER_WEEK,
        )
    )
    first_observed = (
        table.filter(~pl.col("unobserved")).group_by("site").agg(observed_from=pl.col("week").min())
    )
    return (
        table.join(first_observed, on="site", how="left")
        .with_columns(before_start=pl.col("week") < pl.col("observed_from"))
        .sort("site", "week")
    )


def farm_half_hourly(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return each farm's daylight output divided by its own 99th percentile, dropouts removed.

    A dropout is a half-hour at or below zero power while CAMS shows more than 100 W/m2. Remaining
    negative readings are set to zero, as in experiment 00039. Columns: `farm`, `time`, `week`,
    `day`, `farm_norm`.
    """
    out = []
    for farm in common.FARMS:
        rows = frame.filter(
            (pl.col("site") == farm)
            & (pl.col("zenith") < common.MAX_ZENITH_DEG)
            & pl.col("power").is_not_null()
        )
        p99 = float(np.percentile(rows["power"].to_numpy(), 99))
        rows = rows.filter(
            ~((pl.col("power") <= 0.0) & (pl.col("ghi_u") > common.DROPOUT_GHI_W_M2))
        )
        out.append(
            rows.select(
                farm=pl.lit(farm),
                time=pl.col("time"),
                week=pl.col("week"),
                day=pl.col("day"),
                farm_norm=pl.col("power").clip(lower_bound=0.0) / p99,
            )
        )
    return pl.concat(out)


def farm_week_table(*, farm_rows: pl.DataFrame) -> pl.DataFrame:
    """Return one row per farm-week with `missing_days` and whether the farm-week is `dropped`.

    A farm's output is missing on a day with fewer than 10 daylight half-hours of output. The last
    week has 5 days, so its missing days are counted out of 5.
    """
    per_day = farm_rows.group_by("farm", "week", "day").agg(n=pl.len())
    usable = (
        per_day.filter(pl.col("n") >= common.MIN_USABLE_HALF_HOURS_PER_DAY)
        .group_by("farm", "week")
        .agg(usable_days=pl.len().cast(pl.Int32))
    )
    grid = pl.DataFrame({"farm": list(common.FARMS)}).join(
        pl.DataFrame({"week": pl.int_range(common.N_WEEKS, eager=True, dtype=pl.Int32)}),
        how="cross",
    )
    days_in_week = pl.when(pl.col("week") == common.N_WEEKS - 1).then(5).otherwise(7)
    return (
        grid.join(usable, on=["farm", "week"], how="left")
        .with_columns(
            usable_days=pl.col("usable_days").fill_null(0),
            days_in_week=days_in_week.cast(pl.Int32),
        )
        .with_columns(missing_days=pl.col("days_in_week") - pl.col("usable_days"))
        .with_columns(dropped=pl.col("missing_days") > common.MAX_FARM_MISSING_DAYS)
        .sort("farm", "week")
    )


def main() -> None:
    """Write the half-hourly, week, and farm tables, and print the coverage numbers."""
    common.DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    print(f"frame rows inside the 361 weeks: {frame.height}")
    half_hourly = primary_half_hourly(frame=frame)
    weeks = week_table(half_hourly=half_hourly)
    farm_rows = farm_half_hourly(frame=frame)
    farm_weeks = farm_week_table(farm_rows=farm_rows)
    half_hourly.write_parquet(common.HALF_HOURLY_PATH)
    weeks.write_parquet(common.INPUTS_PATH)
    farm_rows.write_parquet(common.FARM_HALF_HOURLY_PATH)
    farm_weeks.write_parquet(common.FARM_WEEKS_PATH)
    print("\nPrimary weeks: observed from, unobserved weeks after the start, masked half-hours")
    for site in common.PRIMARIES:
        rows = weeks.filter(pl.col("site") == site)
        after = rows.filter(~pl.col("before_start"))
        masked = half_hourly.filter((pl.col("site") == site) & pl.col("masked")).height
        print(
            f"{common.PRIMARY_NAMES[site]:20s} observed from week {rows['observed_from'][0]:3d}, "
            f"unobserved {int(after['unobserved'].sum()):3d} of {after.height}, "
            f"masked half-hours {masked}"
        )
    print("\nFarm-weeks dropped (output missing on more than 2 days)")
    for farm in common.FARMS:
        rows = farm_weeks.filter(pl.col("farm") == farm)
        dropped = int(rows["dropped"].sum())
        print(
            f"{common.FARM_NAMES[farm]}: {dropped} of {rows.height} dropped, "
            f"{rows.height - dropped} kept"
        )
    print(f"total farm-weeks dropped: {int(farm_weeks['dropped'].sum())} of {farm_weeks.height}")


if __name__ == "__main__":
    main()
