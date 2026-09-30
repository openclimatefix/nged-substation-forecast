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

Sources: `default_sources` names every fit folder the leaderboard reads, under the data directory:
the published fit, the extra-lead fits (including `nwp_forecast_comparison_day4_shared`, the day-4
cells of the full-window products), the AIFS and WeatherNext 3 fits (including
`nwp_forecast_comparison_day5_aifs_wn3`, their day-5 cells), and it raises if a folder or file is
missing, so a missing day cannot silently leave a blank cell.

`caveat_notes` holds every limiting caveat of the chart, and is the single source for the figure
captions on the page. The script prints the list and writes it to `report.md`; the figure's own
subtitle carries only what a reader needs to decode the chart.

Run it with `uv run python studies/nwp_forecast_comparison/leaderboard_by_day.py
--first-figure-number 3`. The script writes `marks.parquet`, `report.md` and each SVG once, and
refuses to overwrite a file unless `--replace-svgs` is given for the SVGs. Generators appear
nowhere: every error is pooled over the technology's generators, and the loading code refuses an
unanonymised site label.
"""

import argparse
import json
import logging
import math
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final, NamedTuple

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from fit_aifs import BLEND_DAYS, LEAN_DAYS, ROW_SETS, WN3_DAYS, WN3_EXTRA_DAYS
from nwp_forecast_charts import (
    CAPACITY_NOTE,
    DOMAINS,
    FIGURE_NUMBERS,
    SHARED_ROWS_EXCEPT_IFS_NOTE,
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
"""The colour of every dot and interval line. Colour does not encode the lead day, because each
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
"""The stroke width of every vertical grid line, labelled or not, so a labelled line is no heavier
than an unlabelled one."""

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

PUBLISHED_FOLDER: Final[str] = "nwp_forecast_comparison"
EXTRA_LEAD_FOLDERS: Final[tuple[str, ...]] = (
    "nwp_forecast_comparison_leads_day10",
    "nwp_forecast_comparison_leads_day10b",
    "nwp_forecast_comparison_leads_day10c",
    "nwp_forecast_comparison_leads_day10d",
    "nwp_forecast_comparison_day4_shared",
)
"""The extra-lead fits (`<domain>_losses.parquet` each), read like `nwp_forecast_charts.py`'s
`--extra-dir`. The last holds day 4 of the full-window products."""

AIFS_BLENDS_FOLDER: Final[str] = "nwp_forecast_comparison_aifs_blends"
AIFS_EXTRA_FOLDER: Final[str] = "nwp_forecast_comparison_aifs_extra_days"
WN3_FOLDER: Final[str] = "nwp_forecast_comparison_wn3"
WN3_EXTRA_FOLDER: Final[str] = "nwp_forecast_comparison_wn3_extra_days"
DAY5_FOLDER: Final[str] = "nwp_forecast_comparison_day5_aifs_wn3"
DAY5: Final[tuple[int, ...]] = (5,)
"""The folders of AIFS Single, the AIFS ENS mean, and WeatherNext 3 (one losses file per row set
and lead day, `<domain>_<row set>_day<N>_losses.parquet`), and the one lead day the last holds."""


class Sources(NamedTuple):
    """Every fit folder the leaderboard of one technology reads."""

    published: Path
    extra_dirs: list[Path]
    blends: Path
    blends_extra: Path
    wn3: Path
    wn3_extra: Path
    day5: DayFolder


def default_sources(*, data_dir: Path, domain: DomainType) -> Sources:
    """Return the leaderboard's fit folders, after checking every file they must hold exists.

    Args:
        data_dir: The directory holding the study folders (`data/studies`).
        domain: `solar` or `wind`.

    Returns:
        The folders.

    Raises:
        FileNotFoundError: If any expected file is missing.
    """
    sources = Sources(
        published=data_dir / PUBLISHED_FOLDER,
        extra_dirs=[data_dir / name for name in EXTRA_LEAD_FOLDERS],
        blends=data_dir / AIFS_BLENDS_FOLDER,
        blends_extra=data_dir / AIFS_EXTRA_FOLDER,
        wn3=data_dir / WN3_FOLDER,
        wn3_extra=data_dir / WN3_EXTRA_FOLDER,
        day5=DayFolder(folder=data_dir / DAY5_FOLDER, days=DAY5),
    )
    expected = [
        sources.published / f"{domain}_losses.parquet",
        sources.published / f"{domain}_predictions.parquet",
        *(folder / f"{domain}_losses.parquet" for folder in sources.extra_dirs),
        *(
            folder / f"{domain}_{row_set}_day{day}_losses.parquet"
            for folder, days in (
                (sources.blends, BLEND_DAYS),
                (sources.blends_extra, LEAN_DAYS),
                (sources.day5.folder, DAY5),
            )
            for row_set in ROW_SETS
            for day in days
        ),
        *(
            folder / f"{domain}_wn3_day{day}_losses.parquet"
            for folder, days in (
                (sources.wn3, WN3_DAYS),
                (sources.wn3_extra, WN3_EXTRA_DAYS),
                (sources.day5.folder, DAY5),
            )
            for day in days
        ),
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        msg = f"{domain}: missing fit files: {missing}"
        raise FileNotFoundError(msg)
    return sources


# --- The caveats the page can reuse -------------------------------------------------------------

WN3_DAYS_NOTES: Final[dict[DomainType, str]] = {
    "solar": (
        "At days 7 and 14 WeatherNext 3's solar error is not the lowest plotted: the ENS mean "
        "(13.7% at day 7) and GFS native (14.7% at day 14) are lower, on more months. Against "
        "the ENS mean on the same rows (14.7% and 16.0%) the difference is not resolved."
    ),
    "wind": (
        "At days 7 and 14, against the ENS mean on the same rows (16.7% and 18.7%), no "
        "difference from WeatherNext 3 is resolved, and the plotted ranks compare different row "
        "sets. At day 14 WeatherNext 3's 17.9% is not detectably below the shuffled-weather "
        "arms (18.3% and 18.6%), so the chart does not show that WeatherNext 3 beats the 18.5% "
        "climatology. The grey tick is the ENS mean of wind speed, not the mean-vector "
        "reference matched to WeatherNext 3, which the WeatherNext 3 results section compares."
    ),
}
"""The sentence each technology adds about WeatherNext 3's row at days 7 and 14."""

SHARED_FULL_WINDOW_NOTE: Final[str] = SHARED_ROWS_EXCEPT_IFS_NOTE.replace(
    "Every product", "Among the full-window rows, every product", 1
)
"""The shared-rows claim, scoped to the full-window rows: the fewer-months rows, IFS HRES 9 km, and
solar day 0 each score different hours."""

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
            f"{SHARED_FULL_WINDOW_NOTE} The fewer-months rows are scored on their own, smaller "
            "row sets."
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


def day_title(*, day: int) -> str:
    """Return a panel's title.

    Args:
        day: The lead day.

    Returns:
        `Day N`, with `(hindcast)` at day 0.
    """
    return f"Day {day} (hindcast)" if day == 0 else f"Day {day}"


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
) -> alt.LayerChart:
    """Draw one lead day's panel.

    Args:
        rows: `day_rows`'s result.
        baselines: The baselines drawn in this panel, largest error first.
        day: The lead day.
        x_domain: This panel's x range.
        x_title: The x axis's title, or None to leave it off (every panel but the bottom one).
        label_px: The row-label column's width, the same for every panel.

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
        total_px = ROW_STEP_PX * len(labels)
        for rank, baseline in enumerate(baselines):
            # Each rule starts at the bottom of its own label row, so no rule crosses a name.
            layers.append(
                alt.Chart(pl.DataFrame({"x": [round(baseline.value, 3)]}))
                .mark_rule(
                    strokeDash=list(BASELINE_DASH),
                    strokeWidth=BASELINE_WIDTH_PX,
                    color=ocf.BLACK_1,
                    aria=False,
                )
                .encode(  # ty: ignore[unresolved-attribute]
                    x=x_shared("x"),
                    y=alt.value(ROW_STEP_PX * (rank + 1)),
                    y2=alt.value(total_px),
                )
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
            day_title(day=day),
            anchor="start",
            frame="group",
            fontSize=PANEL_TITLE_PX,
        ),
    )


def _join_with_serial_comma(items: Sequence[str]) -> str:
    """Join items as prose: `a`, `a and b`, or `a, b, and c`."""
    if len(items) <= 2:
        return " and ".join(items)
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def subtitle_lines(*, scope: str, smart_days: Sequence[int]) -> list[str]:
    """Return the figure's subtitle: only what a reader needs to decode the chart.

    Args:
        scope: The sentence naming the technology and period, and the capacity definition.
        smart_days: The lead days whose panel draws Smart persistence.

    Returns:
        The lines.
    """
    smart = _join_with_serial_comma([str(day) for day in smart_days])
    return [
        (
            "One panel per lead day, day 0 at the top, all on the same x axis. Each row is one "
            "forecast product. "
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
) -> tuple[alt.VConcatChart, pl.DataFrame]:
    """Draw each product's mean absolute error as one panel per lead day.

    Args:
        loaded: `load`'s result; the marks come from `loaded.leaderboard_losses`.
        domain: `solar` or `wind`.
        title: The figure's title.
        number: The figure's number on its page.
        row_set_marks: Products fitted on fewer months than the published ones.

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
    x_domain = shared_x_domain(panels=list(rows.values()), baselines=list(baselines.values()))
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
            x_domain=x_domain,
            x_title=MAE_TITLE if day == days[-1] else None,
            label_px=label_px,
        )
        for day in days
    ]
    subtitle = subtitle_lines(
        scope=f"{scope_text(losses=losses, domain=domain)} {CAPACITY_NOTE}",
        smart_days=smart_days,
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

    Duplicated from the other study scripts, because study scripts cannot import one another's
    private helpers.

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
        sources = default_sources(data_dir=args.data_dir, domain=domain)
        loaded = load(input_dir=sources.published, domain=domain, extra_dirs=sources.extra_dirs)
        marks = load_row_set_marks(
            blends_dir=sources.blends,
            wn3_dir=sources.wn3,
            domain=domain,
            blends_extra_dir=sources.blends_extra,
            wn3_extra_dir=sources.wn3_extra,
            more_blends_folders=[sources.day5],
            more_wn3_folders=[sources.day5],
        )
        charts[domain], boards[domain] = by_day_figure(
            loaded=loaded,
            domain=domain,
            title=TITLES[(domain, "leaderboard")],
            number=args.first_figure_number + index,
            row_set_marks=marks,
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
