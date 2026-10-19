"""Build the half-hourly panel and print the data-validation checks for experiment 103.

Writes `panel.parquet` (one row per series and half-hour from 2019-09-01 to 2026-06-30, with the
power, whether the row is present or flagged, ERA5 temperature, CAMS irradiance, the dark flag, and
the hand-filtered switching mask) and `series_table.parquet`. The checks are printed, not plotted.

    POLARS_MAX_THREADS=4 uv run python ml-experiments/00103-*/e00103_build.py
"""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
from e00103_common import (
    CAMS_PATH,
    CLEANED_POWER_PATH,
    ERA5_PATH,
    GENERATOR_SERIES,
    LOCAL_TIME_ZONE,
    PANEL_END,
    PANEL_PATH,
    PERIOD_END,
    PERIOD_START,
    PRIMARIES,
    SERIES_TABLE_PATH,
    WORK_DIR,
    hand_filtered_events,
    load_series,
)

STUCK_RUN_LENGTH = 12
"""A run of this many identical half-hours counts as stuck."""


def half_hour_grid(*, series: pl.DataFrame) -> pl.DataFrame:
    """One row per series and half-hour stamp (end of half-hour) in the period."""
    stamps = pl.datetime_range(
        PERIOD_START, PANEL_END, interval="30m", closed="right", time_zone="UTC", eager=True
    ).alias("time")
    return series.select("time_series_id", "site").join(stamps.to_frame(), how="cross")


def power_columns() -> pl.DataFrame:
    """Power with `present` and `flagged`; the cleaned table keeps flagged rows and a reason."""
    return (
        pl.scan_delta(str(CLEANED_POWER_PATH))
        .filter(pl.col("time") > PERIOD_START, pl.col("time") <= PANEL_END)
        .select("time_series_id", "time", "power", "drop_reason")
        .collect()
    )


def temperature_columns(*, grid: pl.DataFrame) -> pl.DataFrame:
    """ERA5 `temperature_2m`, interpolated linearly in time from the hourly values."""
    hourly = pl.read_parquet(ERA5_PATH, columns=["time_series_id", "valid_time", "temperature_2m"])
    hourly = hourly.rename({"valid_time": "time"}).sort("time_series_id", "time")
    both = pl.concat(
        [
            hourly.with_columns(temperature_c=pl.col("temperature_2m")),
            grid.select("time_series_id", "time")
            .with_columns(temperature_2m=pl.lit(None, dtype=pl.Float32))
            .with_columns(temperature_c=pl.lit(None, dtype=pl.Float32)),
        ],
        how="vertical_relaxed",
    )
    return (
        both.unique(subset=["time_series_id", "time"], keep="first", maintain_order=True)
        .sort("time_series_id", "time")
        .with_columns(
            temperature_c=pl.col("temperature_c").interpolate_by("time").over("time_series_id")
        )
        .join(grid.select("time_series_id", "time"), on=["time_series_id", "time"], how="semi")
        .select("time_series_id", "time", "temperature_c")
    )


def irradiance_columns(*, series: pl.DataFrame) -> pl.DataFrame:
    """CAMS irradiance per power half-hour: the half-hour ending in an hour joins that hour."""
    hourly = pl.read_parquet(CAMS_PATH, columns=["site", "time", "ghi_w_m2", "clear_sky_ghi_w_m2"])
    return (
        hourly.rename({"time": "cams_hour_end"})
        .join(series.select("site", "time_series_id"), on="site")
        .drop("site")
    )


def main() -> None:
    """Write the panel and series table, then print the data-validation checks."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    series = load_series()
    series.write_parquet(SERIES_TABLE_PATH)
    grid = half_hour_grid(series=series)
    power = power_columns()
    panel = (
        grid.join(power, on=["time_series_id", "time"], how="left")
        .with_columns(
            present=pl.col("power").is_not_null(),
            flagged=pl.col("drop_reason").is_not_null(),
            cams_hour_end=pl.col("time").dt.truncate("1h")
            + pl.duration(hours=pl.when(pl.col("time").dt.minute() == 0).then(0).otherwise(1)),
        )
        .join(temperature_columns(grid=grid), on=["time_series_id", "time"], how="left")
        .join(irradiance_columns(series=series), on=["time_series_id", "cams_hour_end"], how="left")
        .with_columns(
            dark=pl.col("ghi_w_m2") < 1.0,
            local_time=pl.col("time").dt.convert_time_zone(LOCAL_TIME_ZONE),
        )
    )
    panel = panel.with_columns(
        local_hour=pl.col("local_time").dt.hour() + pl.col("local_time").dt.minute() / 60.0,
        # `time` marks the end of the half-hour, so a stamp of exactly 00:00 belongs to 23:30-24:00.
        local_hour_start=(
            (pl.col("local_time") - pl.duration(minutes=30)).dt.hour()
            + (pl.col("local_time") - pl.duration(minutes=30)).dt.minute() / 60.0
        ),
        local_date=(pl.col("local_time") - pl.duration(minutes=30)).dt.date(),
    ).drop("local_time", "drop_reason")
    panel = add_switching_mask(panel=panel)
    panel.write_parquet(PANEL_PATH)
    print(f"panel: {panel.height} rows, {PANEL_PATH}")
    print_validation(panel=panel)


def add_switching_mask(*, panel: pl.DataFrame) -> pl.DataFrame:
    """Flag `switching_mask` on every series an event touches, from `onset_utc` to `end_utc`.

    The labelled set's `onset_utc` and `end_utc` are the power stamps (half-hour ends) of the first
    and the last affected half-hours, so both are inside the mask.
    """
    events = (
        hand_filtered_events()
        .explode("series_ids")
        .rename({"series_ids": "time_series_id"})
        .with_columns(pl.col("time_series_id").cast(pl.Int32))
    )
    masked = (
        panel.select("time_series_id", "time")
        .join(events, on="time_series_id")
        .filter(
            pl.col("time") >= pl.col("onset_utc"),
            pl.col("time") <= pl.col("end_utc"),
        )
        .select("time_series_id", "time")
        .unique()
        .with_columns(switching_mask=pl.lit(True))
    )
    return panel.join(masked, on=["time_series_id", "time"], how="left").with_columns(
        switching_mask=pl.col("switching_mask").fill_null(False)
    )


def print_validation(*, panel: pl.DataFrame) -> None:
    """Print duplicates, nulls, stuck runs, zeros, month steps, and negative MVA values.

    Every table names a series by its `series` label: the D-label, or G-a and G-b for the two
    metered generators, so the printed report can be published.
    """
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(220)
    panel = panel.filter(pl.col("time") <= PERIOD_END)
    series = load_series()
    labels = {
        row["time_series_id"]: GENERATOR_SERIES.get(row["site"], row["site"])
        for row in series.iter_rows(named=True)
    }

    def named(frame: pl.DataFrame) -> pl.DataFrame:
        """`frame` with `time_series_id` replaced by the `series` label, sorted by label."""
        return (
            frame.with_columns(
                series=pl.col("time_series_id").replace_strict(labels, return_dtype=pl.String)
            )
            .drop("time_series_id")
            .select("series", pl.all().exclude("series"))
            .sort("series")
        )

    print("\n== duplicate (series, time) keys in the cleaned table ==")
    duplicates = (
        pl.scan_delta(str(CLEANED_POWER_PATH))
        .filter(pl.col("time") > PERIOD_START, pl.col("time") <= PERIOD_END)
        .group_by("time_series_id", "time")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
        .collect()
    )
    print(f"duplicate keys: {duplicates.height}")
    print("\n== present, flagged, NaN, and missing-weather counts per series (to 2026-06-30) ==")
    print(
        named(
            panel.group_by("time_series_id").agg(
                rows=pl.len(),
                present=pl.col("present").sum(),
                flagged=pl.col("flagged").sum(),
                nan_power=pl.col("power").is_nan().sum(),
                ghi_null=pl.col("ghi_w_m2").is_null().sum(),
                temp_null_to_2026_07=pl.col("temperature_c")
                .filter(pl.col("time") <= datetime(2026, 7, 31, tzinfo=UTC))
                .is_null()
                .sum(),
            )
        )
    )
    print("\n== stuck runs of 12 or more identical half-hours (unflagged rows) ==")
    stuck = (
        panel.filter(pl.col("present"), ~pl.col("flagged"))
        .sort("time_series_id", "time")
        .with_columns(
            run=(pl.col("power") != pl.col("power").shift()).cum_sum().over("time_series_id")
        )
        .group_by("time_series_id", "run")
        .agg(length=pl.len(), value=pl.col("power").first(), start=pl.col("time").min())
        .filter(pl.col("length") >= STUCK_RUN_LENGTH)
    )
    print(
        named(
            stuck.group_by("time_series_id").agg(
                runs=pl.len(),
                longest=pl.col("length").max(),
                values=pl.col("value").unique().head(5),
            )
        )
    )
    print("\n== zeros outside the `substation_zero` flag (unflagged exact zeros) ==")
    print(
        named(
            panel.filter(pl.col("present"), ~pl.col("flagged"), pl.col("power") == 0.0)
            .group_by("time_series_id")
            .len()
        )
    )
    print("\n== MVA series with power below zero ==")
    mva = series.filter(pl.col("units") == "MVA")["time_series_id"]
    negative = named(
        panel.filter(pl.col("time_series_id").is_in(mva.implode()), pl.col("power") < 0)
        .group_by("time_series_id")
        .len()
    )
    print(f"MVA rows with power below zero: {negative['len'].sum() if negative.height else 0}")
    print("\n== per-month step check: months whose mean moves > 3 monthly SDs from the last ==")
    monthly = (
        panel.filter(pl.col("present"), ~pl.col("flagged"))
        .with_columns(month=pl.col("time").dt.truncate("1mo"))
        .group_by("time_series_id", "month")
        .agg(mean=pl.col("power").mean(), n=pl.len())
        .filter(pl.col("n") > 24 * 2 * 10)
        .sort("time_series_id", "month")
        .with_columns(step=(pl.col("mean") - pl.col("mean").shift()).over("time_series_id"))
    )
    spread = monthly.group_by("time_series_id").agg(sd=pl.col("step").std())
    steps = named(
        monthly.join(spread, on="time_series_id")
        .filter(pl.col("step").abs() > 3 * pl.col("sd"))
        .select("time_series_id", "month", "mean", "step", "sd")
    ).sort("series", "month")
    generator_labels = list(GENERATOR_SERIES.values())
    print(steps.filter(~pl.col("series").is_in(generator_labels)))
    counts = (
        steps.filter(pl.col("series").is_in(generator_labels))
        .group_by("series")
        .len()
        .sort("series")
    )
    print("metered generators (dates withheld): months with a step over 3 SDs:")
    print(counts)
    print("\n== switching mask half-hours per series ==")
    masked = named(panel.group_by("time_series_id").agg(masked=pl.col("switching_mask").sum()))
    print(masked)
    primaries = masked.filter(pl.col("series").is_in(PRIMARIES))
    print(f"masked half-hours on the 16 Primaries: {primaries['masked'].sum()}")


if __name__ == "__main__":
    main()
