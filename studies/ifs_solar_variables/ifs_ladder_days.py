"""Choose the days the IFS ladder study's weather and prediction figures draw, by a stated rule.

The study is planned in <https://github.com/openclimatefix/nged-substation-forecast/pull/1137>.
`ifs_ladder_charts.py` draws the days and `ifs_ladder_report.py` prints their months and years, so
both call the one rule here and cannot disagree about which days were chosen.

**The days are chosen by the CAMS clear-sky index of a farm's daylight hours.** The clearest day has
the highest mean index, the dullest day the lowest, and the most variable day the highest standard
deviation of the hourly index. The weather figure's days are chosen on one farm from the hours that
lead days 1 and 7 both hold. The prediction figure's days are chosen separately for each farm and
are never the weather figure's days.
"""

from collections.abc import Sequence
from datetime import date
from typing import Final

import polars as pl

MIN_DAYLIGHT_HOURS: Final[int] = 5
"""A day counts for the chosen-day figures only if the farm has this many kept hours on it."""

WEATHER_LEAD_DAYS: Final[tuple[int, int]] = (1, 7)
"""The two lead days the weather figure draws."""


def choose_days(*, dataset: pl.DataFrame, site: str, exclude: Sequence[date] = ()) -> pl.DataFrame:
    """Choose the clearest, most variable, and dullest day on one farm by the stated rule.

    Args:
        dataset: The kept rows.
        site: The farm.
        exclude: Dates that may not be chosen.

    Returns:
        One row per chosen day with `day_label`, `site`, and `date`.
    """
    daily = (
        dataset.filter((pl.col("site") == site) & pl.col("cams_clear_sky_index").is_not_null())
        .with_columns(date=pl.col("time").dt.date())
        .filter(~pl.col("date").is_in(list(exclude)))
        .group_by("date")
        .agg(
            mean_index=pl.col("cams_clear_sky_index").mean(),
            spread=pl.col("cams_clear_sky_index").std(),
            hours=pl.len(),
        )
        .filter(pl.col("hours") >= MIN_DAYLIGHT_HOURS)
    )
    chosen = [
        ("Clearest day", daily.sort("mean_index", descending=True).row(0, named=True)),
        ("Most variable day", daily.sort("spread", descending=True).row(0, named=True)),
        ("Dullest day", daily.sort("mean_index").row(0, named=True)),
    ]
    return pl.DataFrame(
        {
            "day_label": [label for label, _ in chosen],
            "site": [site] * len(chosen),
            "date": [row["date"] for _, row in chosen],
        }
    )


def chosen_days(*, near: pl.DataFrame, far: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Choose the weather figure's days and every farm's prediction days.

    Args:
        near: The lead-day-1 frame.
        far: The frame of the far lead day of the weather figure.

    Returns:
        The weather figure's three days at the first farm, and the prediction figure's three days at
        every farm, which exclude the weather figure's dates.
    """
    shared = near.join(far.select("site", "time"), on=["site", "time"])
    weather_days = choose_days(dataset=shared, site=min(near["site"].unique().to_list()))
    farm_days = pl.concat(
        choose_days(dataset=near, site=site, exclude=weather_days["date"].to_list())
        for site in sorted(near["site"].unique().to_list())
    )
    return weather_days, farm_days


def month_lines(*, weather_days: pl.DataFrame, farm_days: pl.DataFrame) -> list[str]:
    """Return markdown lines naming the month and year of each chosen day, never the day.

    The weather figure's lines carry no farm letter, because a page that named the farm beside
    public weather would let a reader place the farm.

    Args:
        weather_days: The weather figure's days.
        farm_days: The prediction figure's days at every farm.

    Returns:
        One line per weather day, then one per farm and day.
    """
    lines = [
        f"- Weather figure, {row['day_label'].lower()}: {row['date'].strftime('%B %Y')}."
        for row in weather_days.iter_rows(named=True)
    ]
    lines += [
        f"- Prediction figure, farm {row['site']}, {row['day_label'].lower()}: "
        f"{row['date'].strftime('%B %Y')}."
        for row in farm_days.iter_rows(named=True)
    ]
    return lines
