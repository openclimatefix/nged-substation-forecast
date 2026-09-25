"""Draw the charts for the nearby-weather-station section of the past-wind page.

One-off throwaway script for the charts of the station-wind addition to
<https://openclimatefix.github.io/nged-substation-forecast/studies/past-weather/wind/>,
in `wind_icon_dream_charts.py`'s style. **Every number a chart shares with the page is read from
`intervals.parquet` or `report.md`, both written by `station_wind_arms.py`**, so a chart cannot
disagree with the page. Before any chart is saved, `Source.verify` requires every interval a chart
draws, printed the way the report prints it, to be in `report.md`, and `_check_title_numbers`
requires every decimal number in a chart's title to be a value or interval end in
`intervals.parquet`, rounded half up once from its full precision to the title's precision.
The script prints every number each chart draws.

Wind farms appear only as `W1` to `W3`. No chart names a weather station, gives a station's
position or a farm-to-station distance, or plots a station's wind against dates. The month chart's
axis holds calendar months pooled over the three farms, and no series of a farm's or a station's
values. Every mark is drawn with `aria=False`, so Vega does not write a point's value into the SVG.

**Do not run this script until `station_wind_arms.py` has fitted every arm and written its
report.**

Run it with `uv run python studies/beam_diffuse_split/station_wind_arms_charts.py`, after
`station_wind_arms.py`. Optimise each SVG with `npx svgo@4 --multipass --precision=1
--final-newline` before committing it.
"""

import argparse
import logging
import math
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Final, cast

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from figure_numbers import WIND_FIGURE_NUMBERS, wind_figure_number, wind_figure_title
from station_wind_arms import OUTPUT_DIR, PLANNED_CONTRASTS
from studies.charts import (
    CONDITION_COLOURS,
    CONDITION_SHAPES,
    PLOT_WIDTH_PX,
    figure,
    interval_panel,
    leaderboard_panel,
    wrapped,
)
from weather_product_charts import ASSETS_DIR, CAPACITY, LEADERBOARD_X_TITLE, X_TITLE

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

SITES: Final[tuple[str, ...]] = ("W1", "W2", "W3")
"""The anonymous wind farm labels."""

_MONTH_PANEL_TITLE_CHARACTERS: Final[int] = 44
"""The characters a by-calendar-month panel's title line holds, which is set in a larger font."""

_PANEL_TITLE_CHARACTERS: Final[int] = 56
"""The characters a contrast panel's title line holds before wrapping, inside the plot's width."""

SETTINGS: Final[tuple[str, str]] = ("pooled", "sensitivity")
"""The two hyperparameter settings, as `intervals.parquet` names them."""

SETTING_CONDITIONS: Final[tuple[str, str]] = (
    "Main XGBoost setting",
    "Second XGBoost setting (shallower trees; a check)",
)
"""The key text of each setting, in the order of `SETTINGS`."""

DOTS: Final[str] = (
    "Dot: estimate. Line: 95% interval from resampling whole calendar months and a fitting seed."
)
"""The subtitle line saying what a dot and a line mean."""

LEADERBOARD_ARMS: Final[Mapping[str, str]] = {
    "station_wind": "Nearest station",
    "era5_10m_wind": "ERA5 10\u00a0m",
    "era5_wind": "ERA5",
    "ukv_wind": "UKV",
    "icon_d2_wind": "ICON-D2",
    "icon_eu_wind": "ICON-EU",
    "icon_global_wind": "ICON global",
    "ukv_station_wind": "UKV + nearest station",
    "ukv_padded_wind": "UKV + its own 80\u00a0m wind",
    "station_k3_wind": "Mean of 3 nearest stations",
    "ukv_icon_d2_wind": "UKV + ICON-D2",
}
"""Each arm's row label on the leaderboard; the chart appends the arm's wind-column count."""

CONTRAST_NAMES: Final[Mapping[str, str]] = {
    "station_wind": "Nearest station",
    "era5_10m_wind": "ERA5 10\u00a0m",
    "era5_wind": "ERA5",
    "ukv_wind": "UKV",
    "station_k3_wind": "Mean of 3 stations",
    "ukv_station_wind": "UKV + station",
    "ukv_padded_wind": "UKV + its own 80\u00a0m wind",
    "ukv_icon_d2_wind": "UKV + ICON-D2",
    "station_speed_only": "Station speed alone",
    "era5_10m_speed_only": "ERA5 10\u00a0m speed alone",
}
"""How a contrast row names each arm."""

EXPLORATORY: Final[tuple[tuple[str, str, str, tuple[str, ...]], ...]] = (
    ("post_review", "station_speed_only", "era5_10m_speed_only", SETTINGS),
    ("post_review", "ukv_station_wind", "ukv_icon_d2_wind", SETTINGS),
    ("exploratory", "ukv_padded_wind", "ukv_wind", SETTINGS),
    ("exploratory", "station_k3_wind", "station_wind", SETTINGS[:1]),
    ("exploratory", "station_k3_wind", "era5_10m_wind", SETTINGS[:1]),
    ("height", "era5_10m_wind", "era5_wind", SETTINGS[:1]),
)
"""The exploratory contrasts drawn: `intervals.parquet` section, first arm, second arm, settings.

The mean-of-three-stations arm and the ERA5 height contrast are fitted at the main setting only.
"""

DOMAIN_MARGIN: Final[float] = 0.15
"""How far past the lowest and highest value a difference chart's x domain extends."""

LEADERBOARD_MARGIN: Final[float] = 0.3
"""How far past the lowest and highest interval end the leaderboard's x domain extends."""

MONTH_NAMES: Final[tuple[str, ...]] = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
"""The x axis of the calendar-month panels."""

MONTH_TABLE_HEADING: Final[str] = "S1 and S2 by calendar month, pooled over the three farms"
"""The heading of the per-calendar-month table in `report.md`."""

MONTH_PANEL_HEIGHT_PX: Final[int] = 130
"""The height of a calendar-month panel's plot area."""


@dataclass
class Source:
    """The study's saved results, and every line a chart drew from them.

    Attributes:
        intervals: `intervals.parquet`, one row per interval `report.md` prints.
        report: `report.md`'s text.
        printed: The report line each interval a chart drew would have been printed as.
    """

    intervals: pl.DataFrame
    report: str
    printed: list[str] = field(default_factory=list)

    def row(
        self,
        *,
        section: str,
        scope: str,
        treatment: str,
        reference: str | None,
        setting: str = "pooled",
    ) -> dict[str, Any]:
        """Return the one interval that matches, and record the report line it must match.

        Args:
            section: The `intervals.parquet` section, such as `planned`.
            scope: The scope cell, such as `all`, `W1` or `Aug-Dec`.
            treatment: The first arm, such as `station_wind`.
            reference: The second arm of a contrast, or None for an arm's own error.
            setting: `pooled` or `sensitivity`.

        Returns:
            The row, as a dictionary.

        Raises:
            ValueError: If the key matches no row, or more than one.
        """
        matches = self.intervals.filter(
            pl.col("section") == section,
            pl.col("setting") == setting,
            pl.col("scope") == scope,
            pl.col("treatment") == treatment,
            (pl.col("reference") == reference)
            if reference is not None
            else pl.col("reference").is_null(),
        )
        if matches.height != 1:
            msg = f"{(section, setting, scope, treatment, reference)} matches {matches.height} rows"
            raise ValueError(msg)
        row = matches.row(index=0, named=True)
        self.printed.append(_printed_line(row=row, scope=scope))
        return row

    def verify(self) -> None:
        """Stop unless every recorded line appears in `report.md` exactly as printed.

        Raises:
            ValueError: Naming the lines the report does not hold.
        """
        missing = [line for line in self.printed if line not in self.report]
        if missing:
            msg = f"{len(missing)} intervals are not in report.md as printed, such as {missing[:3]}"
            raise ValueError(msg)


def _printed_line(*, row: dict[str, Any], scope: str) -> str:
    """Return the start of a report table row for one interval, as the report prints it.

    Args:
        row: An `intervals.parquet` row.
        scope: The scope cell of the row.

    Returns:
        A leaderboard row's arm, error, interval, rows and months; a calendar-balanced row's
        contrast, difference and interval; or another contrast row's scope, contrast, difference
        and interval.
    """
    value, lower, upper = row["value"], row["lower"], row["upper"]
    if row["reference"] is None:
        return (
            f"| `{row['treatment']}` | {row['n_wind_columns']} | {value:.3f} "
            f"| [{lower:.3f}, {upper:.3f}] | {row['n_rows']:,} | {row['n_months']} |"
        )
    if row["section"] == "calendar_balanced":
        name = "S1" if row["treatment"] == "station_wind" else "S2"
        return (
            f"| {name}: {row['treatment']} − {row['reference']} | {value:+.3f} "
            f"| [{lower:+.3f}, {upper:+.3f}] |"
        )
    return (
        f"| {scope} | {row['treatment']} − {row['reference']} | {value:+.3f} "
        f"| [{lower:+.3f}, {upper:+.3f}] |"
    )


def _check_title_numbers(*, title: str, intervals: pl.DataFrame) -> None:
    """Stop unless every decimal number in a title is an interval number rounded half up once.

    Args:
        title: A chart's title.
        intervals: `intervals.parquet`, whose `value`, `lower` and `upper` are at full precision.

    Raises:
        ValueError: If a decimal number in the title matches no interval number.
    """
    numbers = {
        abs(number)
        for column in ("value", "lower", "upper")
        for number in intervals[column].drop_nulls().to_list()
    }
    for printed in re.findall(r"\d+\.\d+", title):
        places = len(printed.split(".")[1])
        step = Decimal(1).scaleb(-places)
        rounded = {
            Decimal(repr(number)).quantize(step, rounding=ROUND_HALF_UP) for number in numbers
        }
        if Decimal(printed) not in rounded:
            msg = f"the title number {printed} is not an interval number rounded to {places} places"
            raise ValueError(msg)


def _round_half_up(*, value: float, places: int = 2) -> str:
    """Return `abs(value)` rounded half up once from its full precision, for a title."""
    step = Decimal(1).scaleb(-places)
    return str(Decimal(repr(abs(value))).quantize(step, rounding=ROUND_HALF_UP))


def _signed_round_half_up(*, value: float, places: int = 2) -> str:
    """Return `value` with its sign, rounded half up once from its full precision."""
    step = Decimal(1).scaleb(-places)
    rounded = Decimal(repr(abs(value))).quantize(step, rounding=ROUND_HALF_UP)
    return f"{'-' if value < 0 else '+'}{rounded}"


def _load() -> Source:
    """Read the saved results.

    Returns:
        `intervals.parquet` and `report.md`, with each leaderboard row's wind-column count parsed
        from the report's arm-column list.
    """
    report = (OUTPUT_DIR / "report.md").read_text()
    counts = {
        arm: int(wind) for arm, wind in re.findall(r"- `(\w+)` \(\d+ columns, (\d+) wind\)", report)
    }
    intervals = pl.read_parquet(OUTPUT_DIR / "intervals.parquet").with_columns(
        n_wind_columns=pl.col("treatment").replace_strict(counts, default=None)
    )
    return Source(intervals=intervals, report=report)


def _without_key(*, panel: alt.VConcatChart | alt.LayerChart) -> alt.LayerChart:
    """Return the plot of an `interval_panel`, without the keys it draws above it."""
    return cast(
        "alt.LayerChart", panel.vconcat[-1] if isinstance(panel, alt.VConcatChart) else panel
    )


def _difference_domain(*, rows: pl.DataFrame) -> tuple[float, float]:
    """Return an x range holding every interval and zero, with a margin.

    Args:
        rows: Rows with `lower_95` and `upper_95`.

    Returns:
        The lowest and highest x value, rounded outwards to a tick step of 0.1.
    """
    low = min(0.0, *rows["lower_95"].to_list()) - DOMAIN_MARGIN
    high = max(0.0, *rows["upper_95"].to_list()) + DOMAIN_MARGIN
    return (round(low * 10) / 10, round(high * 10) / 10)


def _mark(*, label: str, row: dict[str, Any], condition: str, planned: bool) -> dict[str, Any]:
    """Return one interval-panel row from an `intervals.parquet` row."""
    return {
        "label": label,
        "family": "weather model",
        "condition": condition,
        "difference": row["value"],
        "lower_95": row["lower"],
        "upper_95": row["upper"],
        "planned": planned,
    }


def _bracketed(*, name: str) -> str:
    """Return an arm's name in brackets when the name holds a plus, so a difference reads clearly.

    Args:
        name: An arm's name as `CONTRAST_NAMES` gives it.

    Returns:
        The name, in brackets where it names two sources added together.
    """
    return f"({name})" if " + " in name else name


def _contrast_label(*, first: str, second: str, prefix: str = "") -> str:
    """Return a contrast row's label, such as `S1: Nearest station − ERA5 10 m`."""
    text = f"{_bracketed(name=CONTRAST_NAMES[first])} − {_bracketed(name=CONTRAST_NAMES[second])}"
    return f"{prefix}: {text}" if prefix else text


def _scope_line(*, source: Source) -> tuple[str, int, int]:
    """Return the scope every chart states, and the pooled row and month counts.

    Args:
        source: The saved results.

    Returns:
        A sentence read from the report's heading, and the rows and months every arm is scored on.

    Raises:
        ValueError: If the report has no heading with a row count and dates, or the heading
            disagrees with `intervals.parquet`.
    """
    match = re.search(
        r"on ([\d,]+) farm-hours \((\d{4})-(\d{2})-\d{2} to (\d{4})-(\d{2})-\d{2}\)", source.report
    )
    if match is None:
        msg = "report.md has no 'on N farm-hours (start to end)' heading"
        raise ValueError(msg)
    rows = int(match[1].replace(",", ""))
    board = source.intervals.filter(
        pl.col("section") == "leaderboard", pl.col("setting") == "pooled", pl.col("scope") == "all"
    )
    counts = board.select(rows=pl.col("n_rows").unique(), months=pl.col("n_months").unique())
    if counts["rows"].to_list() != [rows] or counts["months"].n_unique() != 1:
        msg = f"the report heading's {rows} rows disagree with intervals.parquet: {counts}"
        raise ValueError(msg)
    months = counts["months"][0]
    first = datetime(int(match[2]), int(match[3]), 1, tzinfo=UTC)
    last = datetime(int(match[4]), int(match[5]), 1, tzinfo=UTC)
    text = (
        f"Three wind farms in Lincolnshire, {first:%B %Y} to {last:%B %Y}: {rows:,} farm-hours in "
        f"{months} calendar months. Three farms are few independent sites."
    )
    return text, rows, months


def _headline(
    *, source: Source, scope: str, rows: int, months: int
) -> tuple[alt.VConcatChart, str]:
    """Draw the leaderboard, the two planned contrasts, and the exploratory contrasts.

    Args:
        source: The saved results.
        scope: The scope sentence every chart states.
        rows: Rows every arm is scored on.
        months: Calendar months those rows cover.

    Returns:
        The `contrasts` figure of `figure_numbers.WIND_FIGURE_NUMBERS`, and its title.
    """
    records = []
    for arm, label in LEADERBOARD_ARMS.items():
        row = source.row(section="leaderboard", scope="all", treatment=arm, reference=None)
        records.append(
            {
                "label": f"{label} ({row['n_wind_columns']})",
                "family": "weather model",
                "condition": SETTING_CONDITIONS[0],
                "value": row["value"],
                "lower_95": row["lower"],
                "upper_95": row["upper"],
            }
        )
    board = pl.DataFrame(records).sort("value")
    board_domain = (
        min(board["lower_95"].to_list()) - LEADERBOARD_MARGIN,
        max(board["upper_95"].to_list()) + LEADERBOARD_MARGIN,
    )
    leaderboard = leaderboard_panel(
        rows=board,
        x_domain=board_domain,
        x_title=LEADERBOARD_X_TITLE,
        panel_title="Each XGBoost model's own error, main settings (wind columns in brackets)",
        conditions=SETTING_CONDITIONS,
        solid=True,
        keys=False,
        row_step_px=30,
    )
    planned_marks = []
    for name, first, second in PLANNED_CONTRASTS:
        for setting, condition in zip(SETTINGS, SETTING_CONDITIONS, strict=True):
            row = source.row(
                section="planned",
                scope="all",
                treatment=first,
                reference=second,
                setting=setting,
            )
            planned_marks.append(
                _mark(
                    label=_contrast_label(first=first, second=second, prefix=name),
                    row=row,
                    condition=condition,
                    planned=True,
                )
            )
    exploratory_marks = []
    for section, first, second, settings in EXPLORATORY:
        for setting in settings:
            row = source.row(
                section=section,
                scope="all",
                treatment=first,
                reference=second,
                setting=setting,
            )
            exploratory_marks.append(
                _mark(
                    label=_contrast_label(first=first, second=second),
                    row=row,
                    condition=SETTING_CONDITIONS[SETTINGS.index(setting)],
                    planned=False,
                )
            )
    planned = pl.DataFrame(planned_marks)
    exploratory = pl.DataFrame(exploratory_marks)
    panels = [leaderboard]
    for frame, panel_title, first in (
        (planned, "Paired differences: the two planned contrasts", True),
        (exploratory, "Paired differences: six exploratory contrasts", False),
    ):
        panel = interval_panel(
            rows=frame,
            x_domain=_difference_domain(rows=frame),
            x_title=X_TITLE,
            zero_label="no difference",
            better_label="first-named arm better",
            conditions=SETTING_CONDITIONS,
            condition_title="XGBoost settings",
            panel_title=panel_title,
            figure_planning="mixed",
        )
        panels.append(panel if first else _without_key(panel=panel))
    s1 = planned.filter(pl.col("label").str.starts_with("S1"))["difference"][0]
    s2 = planned.filter(pl.col("label").str.starts_with("S2"))["difference"][0]
    if not (s1 > 0.0 and s2 < 0.0):
        msg = f"the title says S1 is positive and S2 negative, but S1 is {s1} and S2 is {s2}"
        raise ValueError(msg)
    title = (
        f"The nearest weather station trails ERA5's 10\u00a0m wind by {_round_half_up(value=s1)} "
        f"points, and adding it to UKV lowers UKV's error by {_round_half_up(value=s2)} points "
        "against a control with as many columns"
    )
    return (
        figure(
            panels=panels,
            number=WIND_FIGURE_NUMBERS["contrasts"],
            figure_planning="mixed",
            title=title,
            subtitle=[
                (
                    f"Top: each XGBoost model's own mean absolute error (% of capacity; smaller is "
                    f"better) on the same {rows:,} farm-hours in {months} calendar months, best "
                    "first. Brackets: wind columns given to the model. Below: first-named arm's "
                    "error minus the second's."
                ),
                (
                    "S1 and S2 were named in this section's plan before any station arm was "
                    "fitted, after the five gridded products had been scored. Each planned pair "
                    "carries the same number of wind columns. S1 compares one 10\u00a0m weather "
                    "station with ERA5's 10\u00a0m wind. S2 compares UKV plus the station with "
                    "UKV plus three of its own extra columns."
                ),
                f"{DOTS} {CAPACITY}",
                scope,
            ],
        ),
        title,
    )


def _season_rows(*, source: Source, first: str, second: str) -> pl.DataFrame:
    """Return one contrast's season rows: January to July, August to December, balanced, all.

    Args:
        source: The saved results.
        first: The first arm of the contrast.
        second: The second arm of the contrast.

    Returns:
        One row per scope and setting, in the order the panel draws them. The all-months row is
        the planned contrast, and the other rows are exploratory.
    """
    marks = []
    for section, scope, label in (
        ("january_to_july", "Jan-Jul", "January to July"),
        ("august_to_december", "Aug-Dec", "August to December"),
        ("calendar_balanced", "all", "Every calendar month weighted equally"),
        ("planned", "all", "All months, every row weighted equally"),
    ):
        for setting, condition in zip(SETTINGS, SETTING_CONDITIONS, strict=True):
            row = source.row(
                section=section, scope=scope, treatment=first, reference=second, setting=setting
            )
            marks.append(
                _mark(
                    label=(
                        label
                        if section == "calendar_balanced"
                        else f"{label} ({row['n_months']} months, {row['n_rows']:,} farm-hours)"
                    ),
                    row=row,
                    condition=condition,
                    planned=section == "planned",
                )
            )
    return pl.DataFrame(marks)


def _month_table(*, report: str) -> pl.DataFrame:
    """Read the per-calendar-month table from `report.md`.

    Args:
        report: `report.md`'s text.

    Returns:
        One row per calendar month, with `month` (1 to 12), `years`, `s1`, `s2`, `s1_second` and
        `s2_second` in percentage points.

    Raises:
        ValueError: If the table does not hold exactly twelve months.
    """
    _, _, after = report.partition(f"#### {MONTH_TABLE_HEADING}")
    records = []
    for line in after.splitlines()[1:]:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 7 and cells[0].isdigit():
            values = [float(cell) for cell in cells[3:]]
            records.append(
                {
                    "month": int(cells[0]),
                    "years": int(cells[1]),
                    "s1": values[0],
                    "s2": values[1],
                    "s1_second": values[2],
                    "s2_second": values[3],
                }
            )
        elif records and not line.startswith("|"):
            break
    if [record["month"] for record in records] != list(range(1, 13)):
        msg = f"the calendar-month table holds months {[r['month'] for r in records]}"
        raise ValueError(msg)
    return pl.DataFrame(records)


def _month_panel(*, months: pl.DataFrame, contrast: str, title: str) -> alt.LayerChart:
    """Draw one contrast's mean difference in each calendar month, at both settings.

    A month has no interval, because each calendar month rests on one or two `YYYY-MM` months.
    Colour and shape mark the setting, as in the interval panels above.

    Args:
        months: The output of `_month_table`.
        contrast: `s1` or `s2`.
        title: The panel's title.

    Returns:
        The panel.
    """
    long = pl.concat(
        [
            months.select(
                month=pl.col("month").replace_strict(dict(enumerate(MONTH_NAMES, start=1))),
                difference=pl.col(column),
                setting=pl.lit(condition),
                years=pl.col("years"),
            )
            for column, condition in (
                (contrast, SETTING_CONDITIONS[0]),
                (f"{contrast}_second", SETTING_CONDITIONS[1]),
            )
        ]
    ).with_columns(pl.col("difference").round(decimals=3))
    values = long["difference"].to_list()
    low = min(0.0, *values) - 0.2
    high = max(0.0, *values) + 0.2
    y_scale = alt.Scale(domain=[round(low * 10) / 10, round(high * 10) / 10], nice=False)
    colour = alt.Color(
        "setting:N",
        scale=alt.Scale(domain=list(SETTING_CONDITIONS), range=list(CONDITION_COLOURS)),
        legend=None,
    )
    x = alt.X(
        "month:N",
        sort=list(MONTH_NAMES),
        title=wrapped(
            text=(
                "Calendar month, pooled over the three farms (January to July occur once in the "
                "window, August to December twice)"
            ),
            width=78,
        ),
        axis=alt.Axis(labelAngle=0, grid=False),
    )
    whole_numbers = [
        float(tick)
        for tick in range(math.ceil(y_scale.domain[0]), math.floor(y_scale.domain[1]) + 1)
    ]
    y = alt.Y(
        "difference:Q",
        scale=y_scale,
        title=wrapped(
            text="Mean difference (points of capacity; below zero: first-named arm better)",
            width=60,
        ),
        axis=alt.Axis(values=whole_numbers, format=".0f"),
    )
    rule = (
        alt.Chart(pl.DataFrame({"zero": [0.0]}))
        .mark_rule(color=ocf.BLACK_1, strokeWidth=1, aria=False)
        .encode(y=alt.Y("zero:Q", scale=y_scale))  # ty: ignore[unresolved-attribute]
    )
    filled = (
        alt.Chart(long.filter(pl.col("setting") == SETTING_CONDITIONS[0]))
        .mark_point(filled=True, size=70, opacity=1, shape=CONDITION_SHAPES[0], aria=False)
        .encode(x=x, y=y, color=colour)  # ty: ignore[unresolved-attribute]
    )
    hollow = (
        alt.Chart(long.filter(pl.col("setting") == SETTING_CONDITIONS[1]))
        .mark_point(
            filled=False, size=70, opacity=1, strokeWidth=2, shape=CONDITION_SHAPES[1], aria=False
        )
        .encode(x=x, y=y, color=colour)  # ty: ignore[unresolved-attribute]
    )
    return alt.LayerChart(
        layer=[rule, filled, hollow],
        width=PLOT_WIDTH_PX,
        height=MONTH_PANEL_HEIGHT_PX,
        title=alt.TitleParams(
            text=wrapped(text=title, width=_MONTH_PANEL_TITLE_CHARACTERS),
            anchor="start",
            frame="group",
        ),
    )


def _season(*, source: Source, scope: str) -> tuple[alt.VConcatChart, str]:
    """Draw S1 and S2 by season, and by calendar month.

    Args:
        source: The saved results.
        scope: The scope sentence every chart states.

    Returns:
        The `station_season` figure of `figure_numbers.WIND_FIGURE_NUMBERS`, and its title.

    Raises:
        ValueError: If S1 is not positive in both halves of the year, so the title's "trails" is
            wrong.
    """
    panels: list[alt.VConcatChart | alt.LayerChart] = []
    month_table = _month_table(report=source.report)
    january = august = 0.0
    for index, (name, first, second) in enumerate(PLANNED_CONTRASTS):
        contrast = name.lower()
        frame = _season_rows(source=source, first=first, second=second)
        if name == "S1":
            summer = frame.filter(pl.col("condition") == SETTING_CONDITIONS[0])
            january = summer["difference"][0]
            august = summer["difference"][1]
        panel = interval_panel(
            rows=frame,
            x_domain=_difference_domain(rows=frame),
            x_title=X_TITLE,
            zero_label="no difference",
            better_label="first-named arm better",
            conditions=SETTING_CONDITIONS,
            condition_title="XGBoost settings",
            panel_title=wrapped(
                text=(
                    f"{name}, by season: {_bracketed(name=CONTRAST_NAMES[first])} minus "
                    f"{_bracketed(name=CONTRAST_NAMES[second])}"
                ),
                width=_PANEL_TITLE_CHARACTERS,
            ),
            figure_planning="mixed",
        )
        panels += [
            panel if index == 0 else _without_key(panel=panel),
            _month_panel(
                months=month_table,
                contrast=contrast,
                title=f"{name}, by calendar month: mean difference over the month's rows",
            ),
        ]
    second_january = source.row(
        section="january_to_july",
        scope="Jan-Jul",
        treatment=PLANNED_CONTRASTS[0][1],
        reference=PLANNED_CONTRASTS[0][2],
        setting=SETTINGS[1],
    )
    main_january = source.row(
        section="january_to_july",
        scope="Jan-Jul",
        treatment=PLANNED_CONTRASTS[0][1],
        reference=PLANNED_CONTRASTS[0][2],
    )
    if not (
        january > 0.0
        and august > 0.0
        and august > january
        and main_january["lower"] > 0.0
        and second_january["lower"] < 0.0
    ):
        msg = (
            "the title says S1 is larger in August to December than in January to July, "
            "significant in January to July at the main setting only: "
            f"{january}, {august}, {main_january['lower']}, {second_january['lower']}"
        )
        raise ValueError(msg)
    title = (
        f"In exploratory splits, the nearest station trails ERA5's 10\u00a0m wind by "
        f"{_round_half_up(value=august)} points in August to December, and by less in January to "
        f"July ({_round_half_up(value=january)} points at the main XGBoost setting, and not "
        "statistically significant at the 5% level at the second)"
    )
    return (
        figure(
            panels=panels,
            number=WIND_FIGURE_NUMBERS["station_season"],
            figure_planning="mixed",
            title=title,
            subtitle=[
                (
                    "Every row except the all-months row is exploratory, and the season split is "
                    "confounded with which calendar months the models trained on. "
                    "First-named arm's mean absolute error minus the second's. The season rows "
                    "re-score the models fitted on every month."
                ),
                (
                    "Every calendar month weighted equally: the mean, over the 12 calendar "
                    "months, of each month's mean difference."
                ),
                f"{DOTS} {CAPACITY}",
                scope,
            ],
        ),
        title,
    )


def _by_farm(*, source: Source, scope: str) -> tuple[alt.VConcatChart, str]:
    """Draw S1 and S2 at each farm, at both settings.

    Args:
        source: The saved results.
        scope: The scope sentence every chart states.

    Returns:
        The `per_generator` figure of `figure_numbers.WIND_FIGURE_NUMBERS`, and its title.

    Raises:
        ValueError: If the count of farms at which a contrast is statistically significant at the
            5% level, in the direction the title states, differs between the two settings, so the
            title cannot state one count.
    """
    panels = []
    counts: dict[str, set[int]] = {}
    for index, (name, first, second) in enumerate(PLANNED_CONTRASTS):
        marks = []
        significant = dict.fromkeys(SETTINGS, 0)
        title_side = "lower" if name == "S1" else "upper"
        for site in SITES:
            for setting, condition in zip(SETTINGS, SETTING_CONDITIONS, strict=True):
                row = source.row(
                    section="planned_by_farm",
                    scope=site,
                    treatment=first,
                    reference=second,
                    setting=setting,
                )
                marks.append(_mark(label=site, row=row, condition=condition, planned=False))
                if (title_side == "lower" and row["lower"] > 0.0) or (
                    title_side == "upper" and row["upper"] < 0.0
                ):
                    significant[setting] += 1
        counts[name] = set(significant.values())
        frame = pl.DataFrame(marks)
        panel = interval_panel(
            rows=frame,
            x_domain=_difference_domain(rows=frame),
            x_title=X_TITLE,
            zero_label="no difference",
            better_label="first-named arm better",
            conditions=SETTING_CONDITIONS,
            condition_title="XGBoost settings",
            panel_title=wrapped(
                text=(
                    f"{name}, at each farm: {_bracketed(name=CONTRAST_NAMES[first])} minus "
                    f"{_bracketed(name=CONTRAST_NAMES[second])}"
                ),
                width=_PANEL_TITLE_CHARACTERS,
            ),
            figure_planning="exploratory",
        )
        panels.append(panel if index == 0 else _without_key(panel=panel))
    if len(counts["S2"]) != 1 or next(iter(counts["S2"])) != len(SITES):
        msg = f"the title says S2 is significant at every farm at both settings: {counts['S2']}"
        raise ValueError(msg)
    gaps = {
        setting: source.row(
            section="between_farm",
            scope="W1 - W3",
            treatment=PLANNED_CONTRASTS[0][1],
            reference=PLANNED_CONTRASTS[0][2],
            setting=setting,
        )
        for setting in SETTINGS
    }
    if any(row["lower"] <= 0.0 for row in gaps.values()):
        msg = f"the title says S1 is larger at W1 than at W3 at both settings: {gaps}"
        raise ValueError(msg)
    title = (
        "The nearest station's deficit against ERA5's 10\u00a0m wind is larger at W1 than at W3, "
        "and adding it to UKV lowers UKV's error at all three farms"
    )
    return (
        figure(
            panels=panels,
            number=wind_figure_number(key="per_generator", row_set="station"),
            figure_planning="exploratory",
            title=wind_figure_title(row_set="station", title=title),
            subtitle=[
                (
                    "First-named arm's mean absolute error minus the second's, at each farm. "
                    "Statistically significant at the 5% level means the 95% interval lies "
                    "wholly on one side of zero."
                ),
                (
                    "S1 at W1 minus S1 at W3, exploratory: "
                    f"{_signed_round_half_up(value=gaps[SETTINGS[0]]['value'])} points "
                    f"[{_signed_round_half_up(value=gaps[SETTINGS[0]]['lower'])}, "
                    f"{_signed_round_half_up(value=gaps[SETTINGS[0]]['upper'])}] at the main "
                    "setting. Where one farm's interval excludes zero and another's does not, the "
                    "farms are not shown to differ; the direct comparison is this one."
                ),
                "Each farm's interval resamples whole calendar months and a fitting seed.",
                f"{DOTS} {CAPACITY}",
                scope,
            ],
        ),
        title,
    )


def main() -> int:
    """Read the report, check every number the charts draw, and write the three SVGs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    source = _load()
    scope, rows, months = _scope_line(source=source)
    charts: dict[str, tuple[alt.VConcatChart, str]] = {
        "station_wind_headline": _headline(source=source, scope=scope, rows=rows, months=months),
        "station_wind_season": _season(source=source, scope=scope),
        "station_wind_by_farm": _by_farm(source=source, scope=scope),
    }
    source.verify()
    _LOG.info("all %d intervals drawn are in report.md as printed", len(source.printed))
    for name, (chart, title) in charts.items():
        _check_title_numbers(title=title, intervals=source.intervals)
        path = ASSETS_DIR / f"{name}.svg"
        chart.save(path)
        _LOG.info("wrote %s: %s", path, title)
    for line in source.printed:
        _LOG.info("drew %s", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
