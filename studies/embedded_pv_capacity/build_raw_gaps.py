"""Summarise the raw demand of the Primaries on clear and overcast days, with no model.

    POLARS_MAX_THREADS=2 uv run python studies/embedded_pv_capacity/build_raw_gaps.py

Writes `raw_gaps.parquet` (mean power by half-hour, season-year, season and sky class),
`midday_gaps.parquet` (the per-day midday mean, summarised per season-year), and `weeks.parquet`
(the clearest summer week and winter week of one Primary with the fitted solar output added back).
The sky class of a day comes from CAMS irradiance at the Primary's own location.
"""

from typing import Final

import embedded_pv_inputs as inputs
import polars as pl
from scipy import stats

from studies import pv_capacity_misfit as misfit

WEEK_PRIMARY: Final[str] = "D16"
"""Donington: the Primary with the largest clean capacity in the latest season-year."""
WEEK_SEASON_YEAR: Final[int] = 2025
POOL_FIRST_SEASON_YEAR: Final[int] = 2020
WEEK_MIDDAY: Final[tuple[int, int]] = (22, 27)
WEEK_EVENING: Final[tuple[int, int]] = (38, 43)
NOON_HALF_HOUR: Final[int] = 24
NIGHT_END: Final[int] = 10
"""Half-hours before 05:00 local time, when no solar output is possible."""
DAYS_PER_WEEK: Final[int] = 7
WIDE_COLUMNS: Final[tuple[str, ...]] = (
    "site",
    "season_year",
    "n_days_clear",
    "n_days_overcast",
    "mean_mw_clear",
    "mean_mw_overcast",
    "se_mw_clear",
    "se_mw_overcast",
    "gap_mw",
)
SEASONS: Final[dict[str, tuple[int, ...]]] = {
    "summer": inputs.SUMMER_MONTHS,
    "winter": inputs.WINTER_MONTHS,
}


def half_hourly_frame() -> pl.DataFrame:
    """Return the Primaries' daylight-and-night rows with local half-hour, date and season-year."""
    return (
        pl.scan_parquet(inputs.FRAME_PATH)
        .filter(pl.col("site").is_in(list(inputs.PRIMARY_NAMES)) & pl.col("power").is_not_null())
        .with_columns(
            local=pl.col("time").dt.offset_by("-15m").dt.convert_time_zone("Europe/London")
        )
        .with_columns(
            date=pl.col("local").dt.date(),
            month=pl.col("local").dt.month(),
            half_hour=pl.col("local").dt.hour().cast(pl.Int64) * 2
            + pl.col("local").dt.minute().cast(pl.Int64) // 30,
            season_year=misfit.season_year_of(timestamps=pl.col("time").dt.offset_by("-15m")),
        )
        .filter(pl.col("season_year").is_between(inputs.FIRST_SEASON_YEAR, inputs.LAST_SEASON_YEAR))
        .select(
            "site",
            "date",
            "month",
            "half_hour",
            "season_year",
            "power",
            "u",
            "ghi_u",
            "ghi_u_clear",
            "zenith",
            "time",
        )
        .collect()
    )


def sky_class_per_day(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return each day's clear-sky index and its sky class: clear, overcast, or neither."""
    return (
        rows.filter(pl.col("zenith") < inputs.MAX_ZENITH_DEG)
        .group_by("site", "date")
        .agg(
            n=pl.len(),
            index=pl.col("ghi_u").sum() / pl.col("ghi_u_clear").sum(),
        )
        .filter(pl.col("n") >= inputs.MIN_DAYLIGHT_ROWS_PER_DAY)
        .with_columns(
            sky_class=pl.when(pl.col("index") >= inputs.CLEAR_INDEX)
            .then(pl.lit("clear"))
            .when(pl.col("index") <= inputs.OVERCAST_INDEX)
            .then(pl.lit("overcast"))
        )
        .drop_nulls("sky_class")
        .select("site", "date", "index", "sky_class")
    )


def seasonal(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Label rows with their season (summer or winter) and drop the other months."""
    return pl.concat(
        [
            rows.filter(pl.col("month").is_in(months)).with_columns(season=pl.lit(name))
            for name, months in SEASONS.items()
        ]
    )


def profiles(*, classified: pl.DataFrame) -> pl.DataFrame:
    """Return mean power by half-hour for each Primary, season-year, season, and sky class."""
    return (
        classified.group_by("site", "season_year", "season", "sky_class", "half_hour")
        .agg(mean_mw=pl.col("power").mean(), n_days=pl.col("date").n_unique())
        .sort("site", "season_year", "season", "sky_class", "half_hour")
    )


def midday_gaps(*, classified: pl.DataFrame) -> pl.DataFrame:
    """Return the summer midday mean power per sky class, with its standard error over days."""
    start, end = inputs.MIDDAY_HALF_HOURS
    per_day = (
        classified.filter(
            (pl.col("season") == "summer") & pl.col("half_hour").is_between(start, end - 1)
        )
        .group_by("site", "season_year", "sky_class", "date")
        .agg(midday_mw=pl.col("power").mean())
    )
    return (
        per_day.group_by("site", "season_year", "sky_class")
        .agg(
            n_days=pl.len(),
            mean_mw=pl.col("midday_mw").mean(),
            se_mw=pl.col("midday_mw").std() / pl.len().sqrt(),
        )
        .with_columns(
            ci_half_mw=pl.col("se_mw")
            * pl.col("n_days").map_elements(
                lambda n: float(stats.t.ppf(0.975, n - 1)), return_dtype=pl.Float64
            )
        )
        .sort("site", "season_year", "sky_class")
    )


def pooled_gap_profile(*, profile: pl.DataFrame) -> pl.DataFrame:
    """Return overcast minus clear power by half-hour at Kirton and Donington, pooled over years.

    The pool covers season-years `POOL_FIRST_SEASON_YEAR` to `inputs.LAST_SEASON_YEAR`, which both
    Primaries have, and weights each season-year's mean by its number of days.
    """
    return (
        profile.filter(
            pl.col("site").is_in([inputs.KIRTON, inputs.DONINGTON])
            & (pl.col("season_year") >= POOL_FIRST_SEASON_YEAR)
        )
        .group_by("site", "season", "sky_class", "half_hour")
        .agg(mean_mw=(pl.col("mean_mw") * pl.col("n_days")).sum() / pl.col("n_days").sum())
        .pivot(on="sky_class", index=["site", "season", "half_hour"], values="mean_mw")
        .with_columns(gap_mw=pl.col("overcast") - pl.col("clear"))
        .sort("site", "season", "half_hour")
    )


def clearest_week(*, days: pl.DataFrame, months: tuple[int, ...]) -> pl.DataFrame:
    """Return the first date of the 7-day window with the highest mean clear-sky index."""
    window = (
        days.filter(pl.col("month").is_in(months))
        .sort("date")
        .with_columns(
            mean_index=pl.col("index")
            .rolling_mean(window_size=DAYS_PER_WEEK)
            .shift(-(DAYS_PER_WEEK - 1))
        )
        .with_columns(
            span=(pl.col("date").shift(-(DAYS_PER_WEEK - 1)) - pl.col("date")).dt.total_days()
        )
        .filter(pl.col("span") == DAYS_PER_WEEK - 1)
        .drop_nulls("mean_index")
    )
    return window.sort("mean_index", descending=True).head(1).select("date", "mean_index")


def weeks(*, rows: pl.DataFrame, capacity_mw: float) -> pl.DataFrame:
    """Return the clearest summer week and winter week of one Primary, with solar added back.

    The chosen week of each season is the 7-day run with the highest mean clear-sky index in the
    Primary's season-year. `gross_mw` is the metered flow plus `capacity_mw * u`.
    """
    one = rows.filter(
        (pl.col("site") == WEEK_PRIMARY) & (pl.col("season_year") == WEEK_SEASON_YEAR)
    )
    daily = (
        one.filter(pl.col("zenith") < inputs.MAX_ZENITH_DEG)
        .group_by("date", "month")
        .agg(index=pl.col("ghi_u").sum() / pl.col("ghi_u_clear").sum(), n=pl.len())
        .filter(pl.col("n") >= inputs.MIN_DAYLIGHT_ROWS_PER_DAY)
    )
    out = []
    for season, months in SEASONS.items():
        start = clearest_week(days=daily, months=months)["date"][0]
        chosen = one.filter(
            (pl.col("date") >= start) & (pl.col("date") < start + pl.duration(days=DAYS_PER_WEEK))
        ).with_columns(
            season=pl.lit(season),
            week_start=pl.lit(start),
            day=(pl.col("date") - start).dt.total_days() + 1,
            solar_mw=capacity_mw * pl.col("u"),
        )
        out.append(chosen.with_columns(gross_mw=pl.col("power") + pl.col("solar_mw")))
    return pl.concat(out).select(
        "site", "season", "week_start", "day", "half_hour", "power", "solar_mw", "gross_mw"
    )


def main() -> None:
    """Write the three tables and print the numbers the page quotes."""
    rows = half_hourly_frame()
    days = sky_class_per_day(rows=rows)
    classified = seasonal(rows=rows.join(days, on=["site", "date"], how="inner"))
    profile = profiles(classified=classified)
    gaps = midday_gaps(classified=classified)
    capacity = pl.read_parquet(inputs.TABLE_PATH).filter(
        (pl.col("site") == WEEK_PRIMARY) & (pl.col("season_year") == WEEK_SEASON_YEAR)
    )["capacity_mw"][0]
    week = weeks(rows=rows, capacity_mw=capacity)
    pooled = pooled_gap_profile(profile=profile)
    pooled.write_parquet(inputs.STUDY_DIR / "pooled_gaps.parquet")
    peaks = (
        pooled.group_by("site", "season")
        .agg(
            peak_gap_mw=pl.col("gap_mw").max(),
            peak_half_hour=pl.col("half_hour").sort_by("gap_mw").last(),
            night_mean_gap_mw=pl.col("gap_mw").filter(pl.col("half_hour") < NIGHT_END).mean(),
        )
        .sort("site", "season")
    )
    day_counts = (
        profile.filter(
            pl.col("site").is_in([inputs.KIRTON, inputs.DONINGTON])
            & (pl.col("season_year") >= POOL_FIRST_SEASON_YEAR)
            & (pl.col("half_hour") == NOON_HALF_HOUR)
        )
        .group_by("site", "season", "sky_class")
        .agg(n_days=pl.col("n_days").sum())
        .sort("site", "season", "sky_class")
    )
    profile.write_parquet(inputs.RAW_PATH)
    gaps.write_parquet(inputs.STUDY_DIR / "midday_gaps.parquet")
    week.write_parquet(inputs.STUDY_DIR / "weeks.parquet")
    midday = (
        week.filter(pl.col("season") == "summer")
        .with_columns(
            period=pl.when(pl.col("half_hour").is_between(*WEEK_MIDDAY))
            .then(pl.lit("midday 11:00 to 14:00"))
            .when(pl.col("half_hour").is_between(*WEEK_EVENING))
            .then(pl.lit("evening 19:00 to 22:00"))
        )
        .drop_nulls("period")
        .group_by("period")
        .agg(
            metered_mw=pl.col("power").mean(),
            solar_mw=pl.col("solar_mw").mean(),
            gross_mw=pl.col("gross_mw").mean(),
        )
        .sort("period")
    )
    with pl.Config(float_precision=3):
        midday_text = str(midday)
    wide = (
        gaps.pivot(
            on="sky_class", index=["site", "season_year"], values=["mean_mw", "se_mw", "n_days"]
        )
        .with_columns(gap_mw=pl.col("mean_mw_overcast") - pl.col("mean_mw_clear"))
        .sort("site", "season_year")
    )
    with pl.Config(tbl_rows=100, tbl_cols=30, tbl_width_chars=200, float_precision=3):
        wide_text = str(wide.select(list(WIDE_COLUMNS)))
        peaks_text = str(peaks)
        counts_text = str(day_counts)
    lines = [
        "# Raw gaps report",
        "",
        (
            f"Clear day: index of at least {inputs.CLEAR_INDEX}; overcast day: at most "
            f"{inputs.OVERCAST_INDEX}. Index = summed CAMS global irradiance over summed clear-sky "
            f"irradiance on half-hours with the sun more than {90 - inputs.MAX_ZENITH_DEG:g} "
            "degrees above the horizon, for days with at least "
            f"{inputs.MIN_DAYLIGHT_ROWS_PER_DAY} such half-hours of metered power."
        ),
        "",
        "## Summer midday (11:00 to 14:00 local) mean power: overcast minus clear (MW)",
        "",
        "```",
        wide_text,
        "```",
        "",
        "## Donington's clearest summer week: mean power by period of day (MW)",
        "",
        "```",
        midday_text,
        "```",
        "",
        "## Kirton and Donington: classified days pooled (clear and overcast days at 12:00)",
        "",
        "```",
        counts_text,
        "```",
        "",
        "## Kirton and Donington: overcast minus clear power, pooled over 2020/21 to 2025/26",
        "",
        "```",
        peaks_text,
        "```",
        "",
        (
            f"Week figure: Primary {WEEK_PRIMARY}, season-year {WEEK_SEASON_YEAR}, capacity "
            f"{capacity:.3f} MW. Chosen weeks (the 7-day run with the highest mean clear-sky index "
            "in each season): "
            + "; ".join(
                f"{r['season']} from {r['week_start']}"
                for r in week.select("season", "week_start")
                .unique()
                .sort("season")
                .iter_rows(named=True)
            )
            + "."
        ),
        "",
    ]
    text = "\n".join(lines)
    (inputs.STUDY_DIR / "raw_gaps_report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
