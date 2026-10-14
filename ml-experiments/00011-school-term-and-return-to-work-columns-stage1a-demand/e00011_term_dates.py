"""Validate Lincolnshire's school-term dates and build the daily calendar columns for experiment 11.

The columns are computable from the calendar alone and the council publishes each school year's
dates months ahead, so all of them exist at forecast time. `validate` raises on any fault in the
hand-collected dates; `calendar_frame` returns one row per local date.

    cd ml-experiments/00011-school-term-and-return-to-work-columns-stage1a-demand
    uv run python e00011_term_dates.py
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Final

import holidays
import polars as pl
from dateutil.easter import easter

TERM_DATES_PATH: Final[Path] = Path(__file__).parent / "lincolnshire_term_dates.csv"
FIRST_DATE: Final[date] = date(2021, 9, 1)
LAST_DATE: Final[date] = date(2026, 7, 31)
WINDOW_DAYS: Final[int] = 13
"""The school-term day counts are kept inside this many days of a term's first or last day."""
RETURN_WINDOW_BEFORE: Final[int] = 7
RETURN_WINDOW_AFTER: Final[int] = 14
PLACEBO_SHIFT_DAYS: Final[int] = 17
"""The placebo columns read the real calendar from this many days earlier."""
TERMS_PER_YEAR: Final[int] = 6
CROSS_CHECK_2025_26: Final[dict[int, tuple[str, str]]] = {
    1: ("2025-09-04", "2025-10-23"),
    2: ("2025-11-03", "2025-12-19"),
    3: ("2026-01-06", "2026-02-13"),
    4: ("2026-02-23", "2026-04-02"),
    5: ("2026-04-21", "2026-05-22"),
    6: ("2026-06-01", "2026-07-22"),
}
"""The 2025-26 terms as two school calendars (Heckington and Boston High School) print them."""


def read_terms() -> pl.DataFrame:
    """Return the term dates with `first_day` and `last_day` as dates."""
    return pl.read_csv(TERM_DATES_PATH, try_parse_dates=True)


def _holiday_problems(*, ordered: pl.DataFrame) -> list[str]:
    """Return a problem for each Christmas or Easter that falls inside a term."""
    problems: list[str] = []
    first, last = ordered["first_day"].to_list(), ordered["last_day"].to_list()
    for i in range(1, len(first)):
        term = ordered["term"][i]
        if term == 3:
            xmas = date(first[i].year - 1, 12, 25)
            if not last[i - 1] < xmas < first[i]:
                problems.append(f"Christmas {xmas} is not in the holiday before {first[i]}")
        if term == 5:
            e = easter(first[i].year)
            if not (last[i - 1] < e - timedelta(days=2) and e + timedelta(days=1) < first[i]):
                problems.append(f"Easter {e} is not in the holiday before {first[i]}")
    return problems


def _shape_problems(*, ordered: pl.DataFrame) -> list[str]:
    """Return a problem for each term or gap whose length or weekday is implausible."""
    problems: list[str] = []
    first, last = ordered["first_day"].to_list(), ordered["last_day"].to_list()
    for i, (a, b) in enumerate(zip(first, last, strict=True)):
        label = f"{ordered['academic_year'][i]} term {ordered['term'][i]}"
        if a.weekday() >= 5:
            problems.append(f"{label} starts on a weekend")
        if not 4 <= ((b - a).days + 1) / 7 <= 8:
            problems.append(f"{label} lasts {(b - a).days + 1} days")
        if i:
            gap = (a - last[i - 1]).days - 1
            lo, hi = (30, 60) if ordered["term"][i] == 1 else (6, 20)
            if not lo <= gap <= hi:
                problems.append(f"gap before {label} is {gap} days")
    return problems


def validate(*, terms: pl.DataFrame) -> None:
    """Raise `ValueError` if the hand-collected dates break any of the checks below.

    The checks: six terms in each academic year; every term starts on a weekday and lasts 4 to 8
    weeks; terms are in order and never overlap; the gap before term 1 is 30 to 60 days and every
    other gap is 6 to 20 days; the Christmas holiday contains 25 December and the Easter holiday
    contains Good Friday and Easter Monday; the 2025-26 dates match the two other school calendars;
    and the terms cover the experiment's dates.
    """
    problems: list[str] = []
    for year, frame in terms.group_by("academic_year", maintain_order=True):
        if frame["term"].to_list() != list(range(1, TERMS_PER_YEAR + 1)):
            problems.append(f"{year}: expected terms 1 to {TERMS_PER_YEAR}")
    ordered = terms.sort("first_day")
    if ordered["first_day"].to_list() != terms["first_day"].to_list():
        problems.append("terms are not listed in date order")
    problems += _shape_problems(ordered=ordered) + _holiday_problems(ordered=ordered)
    for number, (a, b) in CROSS_CHECK_2025_26.items():
        row = terms.filter(pl.col("academic_year") == "2025-26", pl.col("term") == number)
        if (str(row["first_day"][0]), str(row["last_day"][0])) != (a, b):
            problems.append(f"2025-26 term {number} differs from the school calendars")
    if ordered["first_day"][0] > FIRST_DATE + timedelta(days=6) or ordered["last_day"][
        -1
    ] < LAST_DATE - timedelta(days=15):
        problems.append("terms do not cover the experiment's dates")
    if problems:
        raise ValueError("; ".join(problems))


def first_working_day(*, year: int) -> date:
    """Return the first weekday of a year that is not an England and Wales bank holiday."""
    bank = holidays.country_holidays("GB", subdiv="ENG", years=year)
    day = date(year, 1, 1)
    while day.weekday() >= 5 or day in bank:
        day += timedelta(days=1)
    return day


def _signed_nearest(*, local: pl.Expr, anchors: list[date], keep: tuple[int, int]) -> pl.Expr:
    """Return the signed days from the nearest anchor date, null outside `keep` (before, after)."""
    nearest = pl.min_horizontal([(local - pl.lit(a)).dt.total_days().abs() for a in anchors])
    signed = pl.coalesce(
        [
            pl.when((local - pl.lit(a)).dt.total_days().abs() == nearest).then(
                (local - pl.lit(a)).dt.total_days()
            )
            for a in anchors
        ]
    )
    return pl.when(signed.is_between(-keep[0], keep[1])).then(signed).otherwise(None)


def calendar_frame(*, terms: pl.DataFrame) -> pl.DataFrame:
    """Return one row per local date from `FIRST_DATE` to `LAST_DATE` with the new columns.

    Columns, each also with a `placebo_` copy that reads the real calendar `PLACEBO_SHIFT_DAYS`
    earlier:

    - `school_term_flag`: 1 on every date from a term's first to its last day (weekends inside a
      term included), otherwise 0.
    - `days_from_term_start_w`: signed days to the nearest term's first day, within 13 days either
      way, otherwise missing (negative in the holiday before the term).
    - `days_from_term_end_w`: the same for a term's last day.
    - `days_from_return_w`: signed days from the year's first working day, from 7 days before to
      14 days after, otherwise missing.
    """
    starts, ends = terms["first_day"].to_list(), terms["last_day"].to_list()
    returns = [first_working_day(year=y) for y in range(FIRST_DATE.year, LAST_DATE.year + 2)]
    dates = pl.date_range(FIRST_DATE - timedelta(days=PLACEBO_SHIFT_DAYS), LAST_DATE, eager=True)
    base = pl.DataFrame({"local_date": dates})
    local = pl.col("local_date")
    in_term = pl.any_horizontal([local.is_between(a, b) for a, b in zip(starts, ends, strict=True)])
    real = base.select(
        "local_date",
        school_term_flag=in_term.cast(pl.Float64),
        days_from_term_start_w=_signed_nearest(
            local=local, anchors=starts, keep=(WINDOW_DAYS, WINDOW_DAYS)
        ),
        days_from_term_end_w=_signed_nearest(
            local=local, anchors=ends, keep=(WINDOW_DAYS, WINDOW_DAYS)
        ),
        days_from_return_w=_signed_nearest(
            local=local, anchors=returns, keep=(RETURN_WINDOW_BEFORE, RETURN_WINDOW_AFTER)
        ),
    )
    columns = [c for c in real.columns if c != "local_date"]
    placebo = real.select(
        (pl.col("local_date") + pl.duration(days=PLACEBO_SHIFT_DAYS)).alias("local_date"),
        *[pl.col(c).alias(f"placebo_{c}") for c in columns],
    )
    return real.join(placebo, on="local_date", how="inner", validate="1:1").filter(
        pl.col("local_date") >= FIRST_DATE
    )


def main() -> None:
    """Validate the dates and print the calendar's summary."""
    terms = read_terms()
    validate(terms=terms)
    frame = calendar_frame(terms=terms)
    print(f"{terms.height} terms valid; {frame.height} dates from {FIRST_DATE} to {LAST_DATE}")
    print(frame.null_count())
    print(frame.filter(pl.col("local_date").is_between(date(2025, 12, 15), date(2026, 1, 25))))
    print([first_working_day(year=y) for y in range(2022, 2027)])


if __name__ == "__main__":
    main()
