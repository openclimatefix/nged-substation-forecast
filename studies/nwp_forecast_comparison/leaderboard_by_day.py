"""Draw the mean-absolute-error leaderboards as one panel per lead day, stacked top to bottom.

Each technology (solar, wind) gets one figure. Its panels run from day 0 at the top to day 14 at the
bottom, and only lead days at which at least one product has a mark get a panel. Every panel has
one row per forecast product fitted at that day. A dot is an XGBoost model's mean absolute error
(% of capacity) given that product's forecast, and its line is the 95% interval from resampling
whole calendar months and a fitting seed. The numbers are the ones `nwp_forecast_charts.py`'s
leaderboard draws, read with the same loading code (`nwp_forecast_charts.load`,
`lead_board_rows`, `row_set_board_rows`), so nothing is refitted or recomputed.

All panels share one x range, one set of tick marks, and one plot width, so a dot at day 7 can be
compared by eye with a dot at day 0. Every dot has one colour. In each panel:

- the rows fitted on the published months are sorted best first (smallest error at the top);
- below a labelled rule sit the rows fitted on fewer months (AIFS Single, the AIFS ENS mean, and
  WeatherNext 3), in a fixed order and not ranked against the rows above, with hollow dots and the
  number of months in their names. A grey tick beside each is the ENS mean fitted on the same
  rows;
- two dashed vertical lines mark the no-weather baselines: Climatology, which is the same in every
  panel, and Smart persistence, drawn only in the panels of lead days at which it was scored
  (the saved losses hold `smart_persistence_day<N>` for days 0 to 3).

Optional sources: with `--optional-sources`, the script also reads whichever of the folders in
`OPTIONAL_SOURCES` exist under the data directory (`optional_sources`). These hold the day-4 cells
of the full-window products and the day-5 cells of AIFS and WeatherNext 3, fitted after the
published leaderboards. A folder missing any file it needs is left out, with a log line.

Every limiting caveat the old leaderboard's caption carried is kept in `caveat_notes`, which the
script prints after each figure's title so the page can reuse the list; the figure's own subtitle
carries only what a reader needs to decode the chart.

Run it with `uv run python studies/nwp_forecast_comparison/leaderboard_by_day.py --input-dir DIR
--extra-dir DIR ... --leaderboard-blends-dir DIR --wn3-dir DIR --leaderboard-blends-extra-dir DIR
--wn3-extra-dir DIR --first-figure-number 1`, the same folders `nwp_forecast_charts.py` takes. The
script writes `marks.parquet`, `report.md` and each SVG once, and refuses to overwrite a file
unless `--replace-svgs` is given for the SVGs. Generators appear nowhere: every error is pooled over
the technology's generators, and the loading code refuses an unanonymised site label.
"""

import argparse
import json
import logging
import math
import os
import subprocess
import sys
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Final, NamedTuple

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from nwp_forecast_charts import (
    CAPACITY_NOTE,
    DOMAINS,
    FIGURE_NUMBERS,
    SHARED_ROWS_NOTE,
    TITLES,
    DayFolder,
    Loaded,
    RowSetMarks,
    by_setting,
    check_single_device,
    lead_board_rows,
    load,
    load_row_set_marks,
    minor_grid_values,
    parsed_lead_arms,
    row_set_board_rows,
    scope_text,
)
from nwp_forecast_comparison import PERCENTAGE_POINTS, DomainType, arms_present, leaderboard
from studies.charts import CONTENT_WIDTH_PX, figure, wrapped

_LOG: Final[logging.Logger] = logging.getLogger("leaderboard_by_day")

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

MAE_TITLE: Final[str] = "Mean absolute error (% of capacity; smaller is better)"
"""The x axis's title on the bottom panel: the quantity, its unit, and which way is better."""

MARK_COLOUR: Final[str] = ocf.DATA_BLUE
"""The colour of every dot and interval line. Colour no longer encodes the lead day, because each
lead day has a panel of its own."""

TICK_COLOUR: Final[str] = ocf.BLACK_1
"""The colour of the grey tick beside a fewer-months row, drawn at `TICK_OPACITY`."""

TICK_OPACITY: Final[float] = 0.45
TICK_HEIGHT_PX: Final[int] = 9
TICK_THICKNESS_PX: Final[int] = 2

ROW_STEP_PX: Final[int] = 16
"""The height of one row in pixels. Nine panels of up to 20 rows stack to a tall figure, so the rows
sit closer than a single panel's."""

POINT_SIZE: Final[int] = 45
"""The area of one dot, in square pixels."""

BASELINE_DASH: Final[tuple[int, int]] = (5, 3)
BASELINE_WIDTH_PX: Final[float] = 1.5
LABEL_FONT_PX: Final[int] = 10
PANEL_TITLE_PX: Final[int] = ocf.font_size(style="Body Large", body_px=11)

SHORT_HEADER: Final[str] = "Fewer months:"
"""The label of the row that opens the block of rows fitted on fewer months."""

BASELINE_NAMES: Final[dict[str, str]] = {
    "climatology": "Climatology",
    "smart_persistence": "Smart persistence",
}
"""Each no-weather baseline's name on its dashed line."""

GRID_WIDTH_PX: Final[float] = 1.5
"""The stroke width of every vertical grid line, labelled or not. It is the width the old
leaderboard gave its half-point (minor) lines, so a labelled line is no heavier than the others."""

MAJOR_GRID_COLOUR: Final[str] = "#C8C8C8"
MINOR_GRID_COLOUR: Final[str] = "#DDDDDD"
"""The grid's colours: the lines at whole numbers, which carry a tick label, are the darker."""

LABEL_PX_PER_CHARACTER: Final[float] = 6.2
"""The width of one character of a row label, in pixels, at `LABEL_FONT_PX` in the theme's
monospaced label font (6 px at 10 px, plus a little slack)."""

LABEL_PAD_PX: Final[int] = 12
"""The space between the longest row label and the plot, in pixels."""

RIGHT_MARGIN_PX: Final[int] = 10
"""The space kept to the right of the plot so the last tick label is not cut off."""

LABEL_SHORTENINGS: Final[dict[str, str]] = {
    "IFS HRES (9 km, Open-Meteo)": "IFS HRES 9 km",
    "ENS control member": "ENS control",
    "WeatherNext 3 mean": "WeatherNext 3",
}
"""Row labels shortened to widen the plot. Each short form names one product: there is one IFS HRES
9 km row, one ENS control row, and one WeatherNext 3 row in a panel."""

OPTIONAL_DAY4_SHARED: Final[str] = "nwp_forecast_comparison_day4_shared"
OPTIONAL_DAY5_AIFS_WN3: Final[str] = "nwp_forecast_comparison_day5_aifs_wn3"
OPTIONAL_DAY4_DAYS: Final[tuple[int, ...]] = (4,)
OPTIONAL_DAY5_DAYS: Final[tuple[int, ...]] = (5,)


class OptionalSources(NamedTuple):
    """The optional fit folders that exist, sorted into how the loading code reads each."""

    extra_dirs: list[Path]
    blends_folders: list[DayFolder]
    wn3_folders: list[DayFolder]


OPTIONAL_SOURCES: Final[tuple[str, ...]] = (OPTIONAL_DAY4_SHARED, OPTIONAL_DAY5_AIFS_WN3)
"""The optional folders, by name under the data directory, switched on by `--optional-sources`:
`OPTIONAL_DAY4_SHARED` holds `<domain>_losses.parquet` (an extra-lead fit, read like
`--extra-dir`), and `OPTIONAL_DAY5_AIFS_WN3` holds `<domain>_single_day5_losses.parquet`,
`<domain>_ens_day5_losses.parquet` and `<domain>_wn3_day5_losses.parquet`."""


def optional_sources(*, data_dir: Path, domain: DomainType) -> OptionalSources:
    """Return whichever optional fit folders exist and hold every file this technology needs.

    Args:
        data_dir: The directory holding the study folders (`data/studies`).
        domain: `solar` or `wind`.

    Returns:
        The day-4 folder as an extra-lead directory if it holds `<domain>_losses.parquet`; the
        day-5 folder as an AIFS folder if it holds the `single` and `ens` files, and as a
        WeatherNext 3 folder if it holds the `wn3` file. Each is left out, with a log line, if
        the folder or a file is missing.
    """
    extra_dirs: list[Path] = []
    blends: list[DayFolder] = []
    wn3: list[DayFolder] = []
    day4 = data_dir / OPTIONAL_DAY4_SHARED
    if (day4 / f"{domain}_losses.parquet").exists():
        extra_dirs.append(day4)
    else:
        _LOG.info("optional source %s has no %s losses; left out", day4, domain)
    day5 = data_dir / OPTIONAL_DAY5_AIFS_WN3
    aifs_files = [
        day5 / f"{domain}_{name}_day{day}_losses.parquet"
        for name in ("single", "ens")
        for day in OPTIONAL_DAY5_DAYS
    ]
    wn3_files = [day5 / f"{domain}_wn3_day{day}_losses.parquet" for day in OPTIONAL_DAY5_DAYS]
    if all(path.exists() for path in aifs_files):
        blends.append(DayFolder(folder=day5, days=OPTIONAL_DAY5_DAYS))
    else:
        _LOG.info("optional source %s lacks the %s AIFS day-5 files; left out", day5, domain)
    if all(path.exists() for path in wn3_files):
        wn3.append(DayFolder(folder=day5, days=OPTIONAL_DAY5_DAYS))
    else:
        _LOG.info(
            "optional source %s lacks the %s WeatherNext 3 day-5 file; left out", day5, domain
        )
    return OptionalSources(extra_dirs=extra_dirs, blends_folders=blends, wn3_folders=wn3)


# --- The caveats the page can reuse -------------------------------------------------------------

WN3_DAYS_NOTES: Final[dict[DomainType, str]] = {
    "solar": (
        "At days 7 and 14 WeatherNext 3's solar error is not the lowest plotted: the ENS mean "
        "(13.7%) and GFS native (14.7%) are lower, on more months. Against the ENS mean on the "
        "same rows (14.7% and 16.0%) the difference is not resolved."
    ),
    "wind": (
        "At days 7 and 14, against the ENS mean on the same rows (16.7% and 18.7%), no "
        "difference from WeatherNext 3 is resolved. At day 14 WeatherNext 3's 17.9% is not "
        "detectably below the shuffled-weather arms (18.3% and 18.6%), so the chart does not "
        "show that WeatherNext 3 beats the 18.5% climatology. The grey tick is the ENS mean "
        "of wind speed, not the mean-vector reference matched to WeatherNext 3 (see the "
        "matched-reference table)."
    ),
}
"""The sentence each technology adds about WeatherNext 3's row at days 7 and 14."""

DAY_ZERO_NOTES: Final[dict[DomainType, str]] = {
    "solar": (
        "Solar day 0 omits the hours ending 01:00 to 06:00 UTC for every product, because they "
        "precede the first 6-hourly step of the AIFS arms."
    ),
    "wind": (
        "The WeatherNext 3 row drops the hour ending 00:00 UTC at day 0, for which the WN3 store "
        "holds no lead."
    ),
}
"""What day 0 drops, by technology."""


def caveat_notes(*, domain: DomainType, figure_numbers: dict[str, int]) -> list[str]:
    """Return every limiting caveat of the leaderboard, for the page to reuse.

    Args:
        domain: `solar` or `wind`.
        figure_numbers: The page's figure numbers, by the keys `wn3_groups` and `headline`.

    Returns:
        One sentence or short group of sentences per caveat.
    """
    return [
        (
            f"{SHARED_ROWS_NOTE} IFS HRES (9 km, Open-Meteo) is scored on slightly fewer hours: "
            "the shared hours minus the target days its archive lacks."
        ),
        (
            "Day 0 is a hindcast, not a day-ahead forecast a service could read, because each "
            "product reads a run that started before the hour it describes. "
            f"{DAY_ZERO_NOTES[domain]}"
        ),
        (
            "Leads are not equal: a Previous Runs product reads the freshest run at least a day "
            "old, a shorter lead than ENS's on most hours, which favours that product."
        ),
        (
            "Rows with fewer months were fitted on fewer months, shown in their names, so they "
            "are not ranked against each other or against the full-window rows. A grey tick "
            "beside a mark is the ENS mean fitted on the same rows."
        ),
        (
            "WeatherNext 3's row holds every row from February to September 2026, and Google has "
            "not documented which WeatherNext 3 model version made that archive, so the February "
            "to June months may overlap WeatherNext 3's training data (Figure "
            f"{figure_numbers['wn3_groups']} splits the rows). {WN3_DAYS_NOTES[domain]}"
        ),
        (
            "A month counts whole in the resampling even where the row set holds part of it "
            "(September 2026 holds 10 days)."
        ),
        (
            "A lead day with no mark was not fitted, because it is beyond the product's forecast "
            "range or not in the archive we hold; nothing is filled in."
        ),
        (
            "Overlapping intervals can still hide a significant paired difference (Figure "
            f"{figure_numbers['headline']})."
        ),
        "Smart persistence is drawn only at the lead days at which it was scored.",
        CAPACITY_NOTE,
    ]


# --- Rows ---------------------------------------------------------------------------------------


def board_rows(*, full: pl.DataFrame, short: pl.DataFrame | None) -> pl.DataFrame:
    """Stack the full-window rows and the fewer-months rows into one frame.

    Args:
        full: `lead_board_rows`'s result without `n_months`: `product`, `day`, `value`,
            `lower_95`, `upper_95`.
        short: `row_set_board_rows`'s result (those columns plus `kind`), or None.

    Returns:
        Those columns plus `kind` (`mark` or `ens_same_rows`) and `window` (`full` or `short`).
    """
    frames = [full.with_columns(kind=pl.lit("mark"), window=pl.lit("full"))]
    if short is not None and short.height:
        frames.append(short.with_columns(window=pl.lit("short")))
    return (
        pl.concat(frames, how="diagonal_relaxed")
        .with_columns(product=shorten(expr=pl.col("product")))
        .select("product", "day", "value", "lower_95", "upper_95", "kind", "window")
    )


def panel_days(*, board: pl.DataFrame) -> list[int]:
    """Return the lead days that get a panel: those at which some product has a mark.

    Args:
        board: `board_rows`'s result.

    Returns:
        The days, earliest first.
    """
    return sorted(board.filter(pl.col("kind") == "mark")["day"].unique().to_list())


def day_rows(*, board: pl.DataFrame, day: int) -> pl.DataFrame:
    """Return one panel's rows, top to bottom.

    Args:
        board: `board_rows`'s result.
        day: The lead day.

    Returns:
        `label`, `value`, `lower_95`, `upper_95`, `window` (`full`, `header` or `short`) and `tick`
        (the ENS mean fitted on the row's own rows, null for a full-window row). Full-window rows
        come first, smallest error first, then, if any, a header row and the fewer-months rows in
        the order the board holds them.
    """
    at_day = board.filter(pl.col("day") == day)
    marks = at_day.filter(pl.col("kind") == "mark")
    full = marks.filter(pl.col("window") == "full").sort("value", "product")
    order = board.filter(pl.col("window") == "short")["product"].unique(maintain_order=True)
    short = (
        marks.filter(pl.col("window") == "short")
        .with_columns(rank=pl.col("product").replace_strict(order, pl.int_range(order.len())))
        .sort("rank")
        .drop("rank")
    )
    ticks_at_day = at_day.filter(pl.col("kind") == "ens_same_rows").select(
        "product", tick=pl.col("value")
    )
    columns = ["label", "value", "lower_95", "upper_95", "window", "tick"]
    parts = [
        full.with_columns(label=pl.col("product"), tick=pl.lit(None, dtype=pl.Float64)).select(
            columns
        )
    ]
    if short.height:
        header = pl.DataFrame(
            {
                "label": [SHORT_HEADER],
                "value": [None],
                "lower_95": [None],
                "upper_95": [None],
                "window": ["header"],
                "tick": [None],
            },
            schema={
                "label": pl.String,
                "value": pl.Float64,
                "lower_95": pl.Float64,
                "upper_95": pl.Float64,
                "window": pl.String,
                "tick": pl.Float64,
            },
        )
        parts.append(header)
        parts.append(
            short.join(ticks_at_day, on="product", how="left")
            .with_columns(label=pl.col("product"))
            .select(columns)
        )
    return pl.concat(parts)


class Baseline(NamedTuple):
    """One no-weather baseline's error at one lead day."""

    name: str
    value: float


def baselines_at_day(*, losses: pl.DataFrame, day: int) -> list[Baseline]:
    """Return the no-weather baselines scored at one lead day, largest error first.

    Args:
        losses: Saved per-row losses.
        day: The lead day.

    Returns:
        Climatology (lead-independent, so every day has it if it was scored) and the day's own
        smart persistence if the losses hold `smart_persistence_day<day>`, in % of capacity.
    """
    primary = by_setting(losses=losses)["primary"]
    arms = {
        "climatology": "climatology",
        "smart_persistence": f"smart_persistence_day{day}",
    }
    held = {key: arm for key, arm in arms.items() if arms_present(losses=primary, arms=(arm,))}
    if not held:
        return []
    board = leaderboard(losses=primary, arms=list(held.values()))
    by_arm = dict(zip(board["arm"].to_list(), board["value"].to_list(), strict=True))
    found = [
        Baseline(name=BASELINE_NAMES[key], value=float(by_arm[arm]) * PERCENTAGE_POINTS)
        for key, arm in held.items()
    ]
    return sorted(found, key=lambda baseline: -baseline.value)


def shared_x_domain(
    *, panels: Sequence[pl.DataFrame], baselines: Sequence[Sequence[Baseline]]
) -> tuple[float, float]:
    """Return the smallest range of whole points holding every value the panels draw.

    Args:
        panels: Each panel's `day_rows`.
        baselines: Each panel's baselines.

    Returns:
        The range, holding every interval end, every grey tick, and every baseline.
    """
    values: list[float] = [
        value
        for frame in panels
        for column in ("lower_95", "upper_95", "tick")
        for value in frame[column].drop_nulls().to_list()
    ]
    values += [item.value for group in baselines for item in group]
    return float(math.floor(min(values))), float(math.ceil(max(values)))


class AxisPlan(NamedTuple):
    """Each panel's x range, and how far the shifted panels' range sits above the others'."""

    domains: dict[int, tuple[float, float]]
    shift: float


def axis_plan(
    *,
    panels: Mapping[int, pl.DataFrame],
    baselines: Mapping[int, Sequence[Baseline]],
    shifted_days: Collection[int] = (),
) -> AxisPlan:
    """Choose every panel's x range.

    With no `shifted_days`, every panel shares one range. Otherwise the panels of `shifted_days`
    share a second range with the same span in points, so one point is the same number of pixels in
    every panel, and it sits higher by `AxisPlan.shift` points, a whole number.

    Args:
        panels: Each lead day's `day_rows`.
        baselines: Each lead day's baselines.
        shifted_days: The lead days drawn on the shifted range.

    Returns:
        Each lead day's range, and the shift (0 if no day is shifted).

    Raises:
        ValueError: If a shifted day's values do not fit in a range of the other panels' span.
    """
    plain = [day for day in panels if day not in shifted_days]
    moved = [day for day in panels if day in shifted_days]
    base = shared_x_domain(
        panels=[panels[day] for day in plain], baselines=[baselines[day] for day in plain]
    )
    domains = dict.fromkeys(plain, base)
    if not moved:
        return AxisPlan(domains=domains, shift=0.0)
    low, high = shared_x_domain(
        panels=[panels[day] for day in moved], baselines=[baselines[day] for day in moved]
    )
    span = base[1] - base[0]
    if high - low > span:
        msg = f"days {moved} span {high - low} points, more than the other panels' {span}"
        raise ValueError(msg)
    shifted = (low, low + span)
    domains |= dict.fromkeys(moved, shifted)
    return AxisPlan(domains=domains, shift=low - base[0])


def whole_ticks(*, x_domain: tuple[float, float]) -> list[float]:
    """Return the labelled ticks: every whole number in the range.

    Args:
        x_domain: The x range, whose ends are whole numbers.

    Returns:
        The whole numbers from one end to the other.
    """
    return [float(value) for value in range(math.ceil(x_domain[0]), math.floor(x_domain[1]) + 1)]


# --- Drawing ------------------------------------------------------------------------------------


def blank_labels(*, count: int) -> list[str]:
    """Return labels for the blank rows that hold baseline names above a panel's first product.

    Args:
        count: How many blank rows.

    Returns:
        Distinct all-space labels, so each is its own row on the nominal axis.
    """
    return [" " * (index + 1) for index in range(count)]


def label_width_px(*, labels: Sequence[str]) -> int:
    """Return the width of the row-label column: room for the longest label, and no more.

    Args:
        labels: Every row label of every panel.

    Returns:
        The width in pixels.
    """
    return math.ceil(max(len(label) for label in labels) * LABEL_PX_PER_CHARACTER) + LABEL_PAD_PX


def plot_width_px(*, label_px: int) -> int:
    """Return the plot's width: whatever the text column holds beside the label column.

    Args:
        label_px: The label column's width.

    Returns:
        The width in pixels.
    """
    return CONTENT_WIDTH_PX - label_px - RIGHT_MARGIN_PX


def day_title(*, day: int, shift: float = 0.0) -> str:
    """Return a panel's title.

    Args:
        day: The lead day.
        shift: How many points the panel's x axis is shifted up from the first panel's, or 0.

    Returns:
        `Day N`, with `(hindcast)` at day 0 and a note of the shift where there is one.
    """
    title = f"Day {day} (hindcast)" if day == 0 else f"Day {day}"
    return f"{title}: x axis shifted up by {shift:g} points" if shift else title


def grid_layers(*, x_domain: tuple[float, float], x_encoding: alt.X) -> list[alt.Chart]:
    """Draw the vertical grid behind a panel: a line at every whole number and every half point.

    Every line has the same width, so a labelled line is not heavier than an unlabelled one; the
    labelled lines are darker.

    Args:
        x_domain: The panel's x range.
        x_encoding: The x encoding every layer of the panel shares, axis and title included: a
            layer that turned its axis off would switch off the merged axis.

    Returns:
        The two layers, the half-point lines first so the whole-number lines sit on top.
    """
    major = whole_ticks(x_domain=x_domain)
    minor = minor_grid_values(x_ticks=major, x_domain=x_domain)
    return [
        alt.Chart(pl.DataFrame({"x": values}, schema={"x": pl.Float64}))
        .mark_rule(color=colour, strokeWidth=GRID_WIDTH_PX, aria=False)
        .encode(x=x_encoding)  # ty: ignore[unresolved-attribute]
        for values, colour in ((minor, MINOR_GRID_COLOUR), (major, MAJOR_GRID_COLOUR))
    ]


def day_panel(
    *,
    rows: pl.DataFrame,
    baselines: Sequence[Baseline],
    day: int,
    x_domain: tuple[float, float],
    x_title: str | None,
    label_px: int,
    shift: float = 0.0,
) -> alt.LayerChart:
    """Draw one lead day's panel.

    Args:
        rows: `day_rows`'s result.
        baselines: The baselines drawn in this panel, largest error first.
        day: The lead day.
        x_domain: This panel's x range.
        x_title: The x axis's title, or None to leave it off (every panel but the bottom one).
        label_px: The row-label column's width, the same for every panel.
        shift: How many points this panel's x axis is shifted up from the first panel's, or 0.

    Returns:
        The panel: the vertical grid; dashed baseline lines across every row, each named in a blank
        row above the first product; an interval line per row; a filled dot for a full-window row
        and a hollow dot for a fewer-months row; and a grey tick for the ENS mean on each
        fewer-months row's rows.
    """
    blanks = blank_labels(count=len(baselines))
    labels = [*blanks, *rows["label"].to_list()]
    x_scale = alt.Scale(domain=list(x_domain), nice=False, zero=False)
    x_axis = alt.Axis(values=whole_ticks(x_domain=x_domain), format=".0f", grid=False)
    x_title_lines = wrapped(text=x_title) if x_title else None

    def x_shared(field: str) -> alt.X:
        return alt.X(f"{field}:Q", scale=x_scale, title=x_title_lines, axis=x_axis)

    y = alt.Y(
        "label:N",
        sort=labels,
        title=None,
        axis=alt.Axis(
            labelLimit=label_px,
            minExtent=label_px,
            maxExtent=label_px,
            labelPadding=6,
            ticks=False,
            domain=False,
            labelFontSize=LABEL_FONT_PX,
            labelFontWeight=alt.ExprRef(
                expr=f"datum.value == {json.dumps(SHORT_HEADER)} ? 'bold' : 'normal'"
            ),
        ),
    )
    data = pl.concat(
        [
            pl.DataFrame({"label": blanks}, schema={"label": pl.String}).with_columns(
                value=pl.lit(None, dtype=pl.Float64),
                lower_95=pl.lit(None, dtype=pl.Float64),
                upper_95=pl.lit(None, dtype=pl.Float64),
                window=pl.lit("blank"),
                tick=pl.lit(None, dtype=pl.Float64),
            ),
            rows,
        ]
    ).with_columns(pl.col("value", "lower_95", "upper_95", "tick").round(3))
    intervals = (
        alt.Chart(data)
        .mark_rule(strokeWidth=2, clip=True, aria=False, color=MARK_COLOUR)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x_shared("lower_95"),
            x2="upper_95:Q",
            y=y,
        )
    )
    dot_x = x_shared("value")
    tooltip = [
        alt.Tooltip("label:N", title="Product"),
        alt.Tooltip("value:Q", title="Mean absolute error"),
        alt.Tooltip("lower_95:Q", title="Lower 95%"),
        alt.Tooltip("upper_95:Q", title="Upper 95%"),
    ]
    filled = (
        alt.Chart(data.filter(pl.col("window") == "full"))
        .mark_point(
            filled=True, size=POINT_SIZE, opacity=1, clip=True, aria=False, color=MARK_COLOUR
        )
        .encode(x=dot_x, y=y, tooltip=tooltip)  # ty: ignore[unresolved-attribute]
    )
    layers: list[alt.Chart] = [
        *grid_layers(x_domain=x_domain, x_encoding=x_shared("x")),
        intervals,
        filled,
    ]
    hollow_rows = data.filter(pl.col("window") == "short")
    if hollow_rows.height:
        layers.append(
            alt.Chart(hollow_rows)
            .mark_point(
                filled=False,
                size=POINT_SIZE,
                strokeWidth=2,
                opacity=1,
                clip=True,
                aria=False,
                color=MARK_COLOUR,
            )
            .encode(x=dot_x, y=y, tooltip=tooltip)  # ty: ignore[unresolved-attribute]
        )
        same_rows = hollow_rows.filter(pl.col("tick").is_not_null())
        if same_rows.height:
            layers.append(
                alt.Chart(same_rows)
                .mark_tick(
                    color=TICK_COLOUR,
                    opacity=TICK_OPACITY,
                    thickness=TICK_THICKNESS_PX,
                    size=TICK_HEIGHT_PX,
                    aria=False,
                )
                .encode(x=x_shared("tick"), y=y)  # ty: ignore[unresolved-attribute]
            )
    if baselines:
        lines = pl.DataFrame(
            {
                "x": [baseline.value for baseline in baselines],
                "name": [baseline.name for baseline in baselines],
                "label": blanks,
            }
        ).with_columns(pl.col("x").round(3))
        layers.append(
            alt.Chart(lines)
            .mark_rule(
                strokeDash=list(BASELINE_DASH),
                strokeWidth=BASELINE_WIDTH_PX,
                color=ocf.BLACK_1,
                aria=False,
            )
            .encode(x=x_shared("x"))  # ty: ignore[unresolved-attribute]
        )
        layers.append(
            alt.Chart(lines)
            .mark_text(
                align="right",
                dx=-4,
                baseline="middle",
                fontSize=LABEL_FONT_PX,
                color=ocf.BLACK_1,
                aria=False,
            )
            .encode(x=x_shared("x"), y=y, text="name:N")  # ty: ignore[unresolved-attribute]
        )
    return alt.LayerChart(
        layer=layers,
        width=plot_width_px(label_px=label_px),
        height=alt.Step(ROW_STEP_PX),
        title=alt.TitleParams(
            day_title(day=day, shift=shift),
            anchor="start",
            frame="group",
            fontSize=PANEL_TITLE_PX,
        ),
    )


def subtitle_lines(
    *, scope: str, smart_days: Sequence[int], shift: float = 0.0, shifted_days: Sequence[int] = ()
) -> list[str]:
    """Return the figure's subtitle: only what a reader needs to decode the chart.

    Args:
        scope: The sentence naming the technology and period, and the capacity definition.
        smart_days: The lead days whose panel draws Smart persistence.
        shift: How many points the shifted panels' x axis sits above the others', or 0.
        shifted_days: The lead days drawn on the shifted axis.

    Returns:
        The lines.
    """
    smart = ", ".join(str(day) for day in smart_days)
    axis = (
        "All panels share one x axis."
        if not shifted_days
        else (
            "Panels share one x axis, except that days "
            f"{', '.join(str(day) for day in shifted_days)} use an axis {shift:g} points higher "
            "with the same width in points, so their tick labels differ from the other panels'."
        )
    )
    return [
        (
            f"One panel per lead day, day 0 at the top. {axis} Each row is one forecast product. "
            "A dot is the mean absolute error of an XGBoost model given that product's forecast, "
            "as a percentage of capacity; smaller is better. The line is the 95% interval from "
            "resampling whole months and a fitting seed. Rows are sorted best first."
        ),
        (
            'Hollow dots below the bold heading "Fewer months" were fitted on fewer months '
            "(in their names) and are not ranked against the rows above; the grey tick is the "
            "ENS mean on the same rows. Dashed lines need no weather forecast: Climatology "
            f"(every panel) and Smart persistence (days {smart}). A day with no dot was not "
            "fitted."
        ),
        (
            "Day 0 is a hindcast: each product reads a run that started before the hour it "
            f"describes. {scope}"
        ),
    ]


def shorten(*, expr: pl.Expr) -> pl.Expr:
    """Shorten a product's name for the row-label column wherever the short form is unambiguous.

    Args:
        expr: An expression of product names.

    Returns:
        The expression with each `LABEL_SHORTENINGS` substring replaced.
    """
    for long, short in LABEL_SHORTENINGS.items():
        expr = expr.str.replace_all(long, short, literal=True)
    return expr


def by_day_figure(
    *,
    loaded: Loaded,
    domain: DomainType,
    title: str,
    number: int,
    row_set_marks: Sequence[RowSetMarks] = (),
    shifted_days: Sequence[int] = (),
) -> tuple[alt.VConcatChart, pl.DataFrame]:
    """Draw each product's mean absolute error as one panel per lead day.

    Args:
        loaded: `load`'s result; the marks come from `loaded.leaderboard_losses`.
        domain: `solar` or `wind`.
        title: The figure's title.
        number: The figure's number on its page.
        row_set_marks: Products fitted on fewer months than the published ones.
        shifted_days: Lead days drawn on an x axis shifted up from the others' (see `axis_plan`);
            none by default, so every panel shares one axis.

    Returns:
        The figure, and the marks it plots (`board_rows`'s frame).

    Raises:
        ValueError: If the marks were fitted on more than one device.
    """
    losses = loaded.leaderboard_losses
    check_single_device(
        arms=list(parsed_lead_arms(losses=by_setting(losses=losses)["primary"])),
        extra_devices=loaded.extra_devices,
        published_arms=loaded.published_arms,
    )
    board = board_rows(
        full=lead_board_rows(losses=losses).drop("n_months"),
        short=row_set_board_rows(marks=row_set_marks) if row_set_marks else None,
    )
    days = panel_days(board=board)
    rows = {day: day_rows(board=board, day=day) for day in days}
    baselines = {day: baselines_at_day(losses=losses, day=day) for day in days}
    plan = axis_plan(panels=rows, baselines=baselines, shifted_days=shifted_days)
    label_px = label_width_px(labels=[label for frame in rows.values() for label in frame["label"]])
    smart_days = [
        day
        for day in days
        if any(item.name == BASELINE_NAMES["smart_persistence"] for item in baselines[day])
    ]
    panels = [
        day_panel(
            rows=rows[day],
            baselines=baselines[day],
            day=day,
            x_domain=plan.domains[day],
            x_title=MAE_TITLE if day == days[-1] else None,
            label_px=label_px,
            shift=plan.shift if day in shifted_days else 0.0,
        )
        for day in days
    ]
    subtitle = subtitle_lines(
        scope=f"{scope_text(losses=losses, domain=domain)} {CAPACITY_NOTE}",
        smart_days=smart_days,
        shift=plan.shift,
        shifted_days=[day for day in days if day in shifted_days],
    )
    return (
        figure(
            panels=panels,
            number=number,
            title=title,
            subtitle=subtitle,
            figure_planning=None,
        ),
        board,
    )


# --- Output -------------------------------------------------------------------------------------


def report_text(
    *, boards: dict[DomainType, pl.DataFrame], notes: dict[DomainType, list[str]]
) -> str:
    """Return the markdown report printing every plotted mark and every caveat.

    Args:
        boards: Each technology's plotted marks.
        notes: Each technology's caveats.

    Returns:
        The report.
    """
    lines = [
        "# Leaderboards by lead day",
        "",
        (
            "Written once by `studies/nwp_forecast_comparison/leaderboard_by_day.py` from saved "
            "per-row losses. Every dot is printed below in % of capacity."
        ),
        "",
    ]
    for domain, board in boards.items():
        lines += [
            f"## {domain}",
            "",
            "| Day | Product | Window | Kind | Error | Lower 95 | Upper 95 |",
        ]
        lines.append("|---|---|---|---|---|---|---|")
        for row in board.sort("day", "window", "value", "product").iter_rows(named=True):
            lines.append(
                f"| {row['day']} | {row['product']} | {row['window']} | {row['kind']} | "
                f"{row['value']:.3f} | "
                + (
                    f"{row['lower_95']:.3f} | {row['upper_95']:.3f} |"
                    if row["lower_95"] is not None
                    else " | |"
                )
            )
        lines += ["", f"### Caveats ({domain})", "", *(f"- {note}" for note in notes[domain]), ""]
    return "\n".join(lines)


def optimise(*, path: Path) -> None:
    """Optimise one SVG in place with `svgo`.

    Args:
        path: The SVG.

    Raises:
        subprocess.CalledProcessError: If `svgo` fails.
    """
    subprocess.run(
        ["npx", "svgo@4", "--multipass", "--precision=1", "--final-newline", str(path)],
        check=True,
        capture_output=True,
    )


def repo_data_dir() -> Path:
    """Return the shared `data/` directory, resolving a linked worktree to the main checkout.

    Returns:
        The directory holding `studies/`.
    """
    root = PROJECT_ROOT
    marker = root / ".git"
    if marker.is_file():
        git_dir = Path(marker.read_text().removeprefix("gitdir:").strip())
        if git_dir.parent.name == "worktrees":
            root = git_dir.parent.parent.parent
    return Path(os.environ.get("DATA_PATH_INTERNAL") or root / "data")


def main() -> int:
    """Draw both technologies' leaderboards by day and write them once.

    Returns:
        The exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    studies_dir = repo_data_dir() / "studies"
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--extra-dir", type=Path, action="append", default=[])
    parser.add_argument("--leaderboard-blends-dir", type=Path, default=None)
    parser.add_argument("--wn3-dir", type=Path, default=None)
    parser.add_argument("--leaderboard-blends-extra-dir", type=Path, default=None)
    parser.add_argument("--wn3-extra-dir", type=Path, default=None)
    parser.add_argument(
        "--optional-sources",
        action="store_true",
        help=f"Also read the folders of OPTIONAL_SOURCES ({', '.join(OPTIONAL_SOURCES)}) under "
        "--data-dir where they exist.",
    )
    parser.add_argument("--data-dir", type=Path, default=studies_dir)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=studies_dir / "nwp_forecast_comparison_leaderboard_by_day",
    )
    parser.add_argument(
        "--svg-dir", type=Path, default=PROJECT_ROOT / "docs" / "studies" / "assets"
    )
    parser.add_argument(
        "--first-figure-number",
        type=int,
        default=FIGURE_NUMBERS[("solar", "leaderboard")],
        help="The solar figure's number on the page; the wind figure follows it.",
    )
    parser.add_argument(
        "--wn3-groups-figure-number", type=int, default=FIGURE_NUMBERS[("solar", "wn3_groups")]
    )
    parser.add_argument(
        "--headline-figure-number", type=int, default=FIGURE_NUMBERS[("solar", "headline")]
    )
    parser.add_argument(
        "--shifted-days",
        type=int,
        nargs="*",
        default=[],
        help="Lead days drawn on an x axis shifted up from the others' (same span), so that "
        "the days of the largest errors do not stretch the shared axis.",
    )
    parser.add_argument(
        "--replace-svgs", action="store_true", help="Replace SVGs that already exist."
    )
    parser.add_argument("--no-svgo", action="store_true", help="Skip the svgo optimisation.")
    args = parser.parse_args()
    svgs = {domain: args.svg_dir / f"nwp_forecast_{domain}_leaderboard.svg" for domain in DOMAINS}
    taken = [
        path
        for path in (
            args.output_dir,
            *(() if args.replace_svgs else svgs.values()),
        )
        if path.exists()
    ]
    if taken:
        msg = f"{taken} exist; this script writes each output once"
        raise FileExistsError(msg)
    charts: dict[DomainType, alt.VConcatChart] = {}
    boards: dict[DomainType, pl.DataFrame] = {}
    notes: dict[DomainType, list[str]] = {}
    for index, domain in enumerate(DOMAINS):
        extras = (
            optional_sources(data_dir=args.data_dir, domain=domain)
            if args.optional_sources
            else OptionalSources([], [], [])
        )
        loaded = load(
            input_dir=args.input_dir,
            domain=domain,
            extra_dirs=[*args.extra_dir, *extras.extra_dirs],
        )
        marks = load_row_set_marks(
            blends_dir=args.leaderboard_blends_dir,
            wn3_dir=args.wn3_dir,
            domain=domain,
            blends_extra_dir=args.leaderboard_blends_extra_dir,
            wn3_extra_dir=args.wn3_extra_dir,
            more_blends_folders=extras.blends_folders,
            more_wn3_folders=extras.wn3_folders,
        )
        charts[domain], boards[domain] = by_day_figure(
            loaded=loaded,
            domain=domain,
            title=TITLES[(domain, "leaderboard")],
            number=args.first_figure_number + index,
            row_set_marks=marks,
            shifted_days=args.shifted_days,
        )
        notes[domain] = caveat_notes(
            domain=domain,
            figure_numbers={
                "wn3_groups": args.wn3_groups_figure_number + index,
                "headline": args.headline_figure_number + index,
            },
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.md").write_text(report_text(boards=boards, notes=notes))
    pl.concat(
        [board.with_columns(domain=pl.lit(domain)) for domain, board in boards.items()]
    ).write_parquet(args.output_dir / "marks.parquet")
    for domain, path in svgs.items():
        charts[domain].save(path)
        if not args.no_svgo:
            optimise(path=path)
        _LOG.info("wrote %s", path)
        for note in notes[domain]:
            sys.stdout.write(f"{domain} note: {note}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
