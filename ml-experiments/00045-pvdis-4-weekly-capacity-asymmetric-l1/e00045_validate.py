"""Validate experiment 00045's inputs before any fit, printing every number.

The report is written to `validation_report.txt` in the experiment's data folder. The script exits
with a non-zero status when a check fails: a duplicate key, a week with more than 336 half-hours,
a peak of the power-to-CAMS correlation outside 10:00 to 14:00 UTC, or a timestamp-shift scan that
does not peak at shift zero, for a Primary other than the ones flagged in `PRIMARY_STATUS`.

    POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
        uv run python ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1/e00045_validate.py
"""

import sys
from datetime import timedelta
from typing import Final

import e00045_common as common
import numpy as np
import polars as pl

SHIFTS: Final[range] = range(-4, 5)
"""Half-hour shifts of the CAMS shape tried against the power."""
PEAK_HOURS_UTC: Final[tuple[int, int]] = (10, 14)
"""The hours (inclusive) in which the power-to-CAMS correlation must peak."""
WINTER_MONTHS: Final[tuple[int, ...]] = (11, 12, 1)
"""Months in which the sun is under 75 degrees from the zenith for under 10 half-hours a day."""

report: list[str] = []
failures: list[str] = []


def say(text: str = "") -> None:
    """Print a line and keep it for the report file."""
    print(text, flush=True)
    report.append(text)


def fail(text: str) -> None:
    """Record a failed check."""
    failures.append(text)
    say(f"FAIL: {text}")


def correlation(*, x: np.ndarray, y: np.ndarray) -> float:
    """Return the Pearson correlation, or NaN with under 30 pairs or no variance."""
    if x.size < 30 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def check_rows(*, half_hourly: pl.DataFrame, weeks: pl.DataFrame) -> None:
    """Row counts per Primary-week and duplicate keys."""
    say("== Row counts per Primary-week (a full week holds 336 half-hours; the last holds 240)")
    duplicates = half_hourly.group_by("site", "time").len().filter(pl.col("len") > 1).height
    say(f"duplicate (site, time) keys: {duplicates}")
    if duplicates:
        fail(f"{duplicates} duplicate (site, time) keys")
    for site in common.PRIMARIES:
        rows = weeks.filter((pl.col("site") == site) & ~pl.col("before_start"))
        over = rows.filter(pl.col("n_rows") > common.HALF_HOURS_PER_WEEK).height
        say(
            f"{common.PRIMARY_NAMES[site]:20s} weeks {rows.height:3d}  rows per week: "
            f"min {rows['n_rows'].min():3d}  median {rows['n_rows'].median():5.0f}  "
            f"max {rows['n_rows'].max():3d}  weeks over 336: {over}  "
            f"weeks with no rows: {rows.filter(pl.col('n_rows') == 0).height}"
        )
        if over:
            fail(f"{common.PRIMARY_NAMES[site]} has {over} weeks with more than 336 half-hours")
    total_per_week = (
        weeks.filter(pl.col("week") < common.N_WEEKS - 1).select(pl.col("n_rows").max()).item()
    )
    say(f"largest count in any full Primary-week: {total_per_week}")


def check_zenith(*, half_hourly: pl.DataFrame) -> None:
    """The frame's zenith against solar geometry at the area point."""
    from studies import solar

    sample = half_hourly.filter(pl.col("site") == "D16").head(40000)
    expected = solar.zenith(stamps=sample["time"], latitude=53.0, longitude=-0.1)
    diff = np.abs(sample["zenith"].to_numpy() - expected)
    say(
        f"\n== Frame zenith against solar.zenith at (53.0, -0.1), {sample.height} rows: "
        f"median difference {np.median(diff):.2f} degrees, "
        f"99th percentile {np.percentile(diff, 99):.2f}"
    )
    if np.percentile(diff, 99) > 3.0:
        fail("the frame's zenith differs from solar geometry by over 3 degrees")


def check_hour_profile(*, half_hourly: pl.DataFrame) -> None:
    """Hour-of-day correlation of power with the CAMS shape u, April to September."""
    say("\n== Correlation of power with the CAMS unit shape u, by UTC hour, April to September")
    say("(power is net demand, so a negative correlation is the PV signature)")
    summer = half_hourly.filter(
        pl.col("time").dt.offset_by("-15m").dt.month().is_in(common.SUMMER_MONTHS)
        & pl.col("usable")
    ).with_columns(hour=pl.col("time").dt.offset_by("-15m").dt.hour())
    say("hour " + " ".join(f"{h:5d}" for h in range(4, 21)))
    for site in common.PRIMARIES:
        rows = summer.filter(pl.col("site") == site)
        profile = {}
        for hour in range(4, 21):
            block = rows.filter(pl.col("hour") == hour)
            profile[hour] = correlation(
                x=block["u"].to_numpy(), y=block["power"].cast(pl.Float64).to_numpy()
            )
        say(
            f"{common.PRIMARY_NAMES[site][:4]:4s} "
            + " ".join(f"{profile[h]:5.2f}" for h in range(4, 21))
        )
        valid = {h: abs(c) for h, c in profile.items() if not np.isnan(c)}
        peak = max(valid, key=lambda h: valid[h])
        flagged = common.PRIMARY_STATUS[site] != "clean"
        ok = PEAK_HOURS_UTC[0] <= peak <= PEAK_HOURS_UTC[1]
        say(
            f"  {common.PRIMARY_NAMES[site]}: strongest |correlation| at {peak}:00 UTC "
            f"({valid[peak]:.2f})"
        )
        if not ok:
            message = f"{common.PRIMARY_NAMES[site]}: correlation peaks at {peak}:00 UTC"
            if flagged:
                say(f"  NOTE (flagged Primary, {common.PRIMARY_STATUS[site]}): {message}")
            else:
                fail(message)


def check_dropped(*, weeks: pl.DataFrame, half_hourly: pl.DataFrame) -> None:
    """The share of CAMS daylight half-hours dropped per week, and by hour of day."""
    say("\n== Share of CAMS daylight half-hours dropped (reliability under 0.9), per week")
    for site in common.PRIMARIES:
        rows = weeks.filter((pl.col("site") == site) & ~pl.col("before_start"))
        shares = rows["cams_dropped_share"].drop_nulls()
        say(
            f"{common.PRIMARY_NAMES[site]:20s} mean {shares.mean():.3f}  "
            f"median {shares.median():.3f}  "
            f"90th percentile {shares.quantile(0.9):.3f}  max {shares.max():.3f}  "
            f"weeks over 0.5: {(shares > 0.5).sum()}"
        )
    one = weeks.filter(pl.col("site") == common.PRIMARIES[0])
    say(
        "expected daylight half-hours per week (zenith under 75 degrees): "
        f"min {one['expected_daylight'].min()}  median {one['expected_daylight'].median():.0f}  "
        f"max {one['expected_daylight'].max()}"
    )
    stamps = half_hourly.filter(pl.col("site") == "D16")
    hours = (
        stamps.group_by(hour=pl.col("time").dt.offset_by("-15m").dt.hour())
        .agg(rows=pl.len())
        .sort("hour")
    )
    full = stamps["time"].dt.offset_by("-15m").dt.date().n_unique() * 2
    say("rows present per UTC hour of D16 against the days spanned (2 half-hours per day-hour):")
    say(" ".join(f"{h}:{r / full:.2f}" for h, r in hours.iter_rows()))


def check_shift(*, half_hourly: pl.DataFrame) -> None:
    """Correlation of deseasonalised power with u shifted by -4 to +4 half-hours."""
    say("\n== Timestamp-shift scan: correlation of deseasonalised power with u at t + shift")
    say(
        "(each series has its month x half-hour-of-day mean removed; "
        "the strongest |correlation| must be at shift 0)"
    )
    say("shift " + " ".join(f"{s:6d}" for s in SHIFTS))
    for site in common.PRIMARIES:
        rows = (
            half_hourly.filter(
                (pl.col("site") == site) & pl.col("usable") & (pl.col("zenith") < 70.0)
            )
            .with_columns(
                cell=pl.col("time").dt.offset_by("-15m").dt.month() * 100
                + pl.col("time").dt.offset_by("-15m").dt.hour() * 2
                + pl.col("time").dt.offset_by("-15m").dt.minute() // 30
            )
            .select("time", "cell", power=pl.col("power").cast(pl.Float64), u="u")
        )
        shapes = half_hourly.filter(pl.col("site") == site).select("time", "u")
        values = []
        for shift in SHIFTS:
            shifted = shapes.select(
                time=pl.col("time") - timedelta(minutes=30 * shift), u_shift="u"
            )
            joined = rows.join(shifted, on="time", how="inner").with_columns(
                power_dev=pl.col("power") - pl.col("power").mean().over("cell"),
                u_dev=pl.col("u_shift") - pl.col("u_shift").mean().over("cell"),
            )
            values.append(
                correlation(x=joined["u_dev"].to_numpy(), y=joined["power_dev"].to_numpy())
            )
        say(f"{common.PRIMARY_NAMES[site][:5]:5s} " + " ".join(f"{v:6.2f}" for v in values))
        peak_shift = list(SHIFTS)[int(np.nanargmax(np.abs(values)))]
        say(f"  {common.PRIMARY_NAMES[site]}: strongest |correlation| at shift {peak_shift}")
        if peak_shift != 0:
            message = f"{common.PRIMARY_NAMES[site]}: timestamp scan peaks at shift {peak_shift}"
            if common.PRIMARY_STATUS[site] != "clean":
                say(f"  NOTE (flagged Primary, {common.PRIMARY_STATUS[site]}): {message}")
            else:
                fail(message)


def check_unobserved(*, weeks: pl.DataFrame) -> None:
    """Count unobserved weeks by Primary and by season."""
    say(
        "\n== Unobserved weeks (under 3 usable days; a usable day has 10 or more usable half-hours)"
    )
    for site in common.PRIMARIES:
        rows = weeks.filter((pl.col("site") == site) & ~pl.col("before_start"))
        unobserved = rows.filter(pl.col("unobserved")).with_columns(
            month=(pl.col("week_start") + pl.duration(days=3)).dt.month()
        )
        winter = unobserved.filter(pl.col("month").is_in(WINTER_MONTHS)).height
        summer = unobserved.filter(pl.col("month").is_in(common.SUMMER_MONTHS)).height
        say(
            f"{common.PRIMARY_NAMES[site]:20s} unobserved {unobserved.height:3d} "
            f"of {rows.height:3d} "
            f"({unobserved.height / rows.height:.0%}); in November to January {winter}; "
            f"in April to September {summer}; weeks before the first observed week: "
            f"{weeks.filter((pl.col('site') == site) & pl.col('before_start')).height}"
        )


def check_masks(*, half_hourly: pl.DataFrame) -> None:
    """The switching mask's share of daylight half-hours, and the log's coverage."""
    say("\n== Switching mask (logged closures padded 24 h each side)")
    logged = pl.read_parquet(common.LOGGED_CLOSURES_PATH)
    say(
        f"logged closure intervals: {logged.height}; first start {logged['start_utc'].min()}, "
        f"last end {logged['end_utc'].max()}"
    )
    for site in common.PRIMARIES:
        rows = half_hourly.filter((pl.col("site") == site) & pl.col("daylight"))
        say(
            f"{common.PRIMARY_NAMES[site]:20s} masked share of daylight half-hours "
            f"{rows['masked'].mean():.3f}"
        )
    after = half_hourly.filter(pl.col("time") > pl.datetime(2026, 3, 26, time_zone="UTC"))
    say(
        "half-hours after the log ends (2026-03-25) carry no mask: "
        f"{after.filter(pl.col('site') == 'D16').height} per Primary"
    )


def check_farms(*, farm_weeks: pl.DataFrame) -> None:
    """Each farm's coverage against the 361 weeks."""
    say("\n== Solar-farm coverage against the 361 weeks (output on all but 2 days kept)")
    for farm in common.FARMS:
        rows = farm_weeks.filter(pl.col("farm") == farm)
        kept = rows.filter(~pl.col("dropped"))
        summer_rows = rows.filter(
            (pl.col("week") * 7 + 3 + 0)
            .map_elements(
                lambda d: (common.FIRST_WEEK_START + timedelta(days=d)).month,
                return_dtype=pl.Int64,
            )
            .is_in(common.SUMMER_MONTHS)
        )
        say(
            f"{common.FARM_NAMES[farm]}: kept {kept.height} of {rows.height}; "
            f"April to September kept {summer_rows.filter(~pl.col('dropped')).height} of "
            f"{summer_rows.height}; weeks with all days present "
            f"{rows.filter(pl.col('missing_days') == 0).height}"
        )
    say(f"farm-weeks dropped in total: {int(farm_weeks['dropped'].sum())} of {farm_weeks.height}")


def main() -> None:
    """Run every check, write the report, and exit non-zero on a failure."""
    half_hourly = pl.read_parquet(common.HALF_HOURLY_PATH)
    weeks = pl.read_parquet(common.INPUTS_PATH)
    farm_weeks = pl.read_parquet(common.FARM_WEEKS_PATH)
    say(
        f"Primaries: {len(common.PRIMARIES)}; weeks: {common.N_WEEKS}; "
        f"half-hourly rows: {half_hourly.height}"
    )
    say(f"week-table rows: {weeks.height} (expected {len(common.PRIMARIES) * common.N_WEEKS})")
    if weeks.height != len(common.PRIMARIES) * common.N_WEEKS:
        fail("the week table is not Primaries x 361")
    say(
        "week 0 starts "
        f"{common.week_start(week=0):%Y-%m-%d}, "
        f"week 360 starts {common.week_start(week=360):%Y-%m-%d}"
    )
    check_rows(half_hourly=half_hourly, weeks=weeks)
    check_zenith(half_hourly=half_hourly)
    check_hour_profile(half_hourly=half_hourly)
    check_dropped(weeks=weeks, half_hourly=half_hourly)
    check_shift(half_hourly=half_hourly)
    check_unobserved(weeks=weeks)
    check_masks(half_hourly=half_hourly)
    check_farms(farm_weeks=farm_weeks)
    say(f"\nFAILED CHECKS: {len(failures)}" if failures else "\nAll checks passed.")
    for text in failures:
        say(f"  {text}")
    common.VALIDATION_REPORT_PATH.write_text("\n".join(report) + "\n")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
