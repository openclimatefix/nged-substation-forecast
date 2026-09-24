"""The dot-and-interval chart the study pages draw, and the report parser that feeds it.

**Every number a chart shares with its page is read from the report the study wrote**, so the
chart cannot disagree with the page. `report_contrasts` parses the report's contrast tables,
`select_contrasts` picks the rows a chart wants and raises on any it cannot find exactly once, and
`interval_panel` draws them: a dot at each estimate and a line across its 95% interval, beside a
labelled zero rule. `figure` sets one or more panels under the caption OCF's slide template uses.

**Colour marks what kind of product a row is, not which product it is.** The product's name is
always on the axis beside its marks, so identity never rests on colour. One colour per product
fails a colour-vision check with OCF's main data colours (Data Purple and Data Blue sit 2.0 ΔE
apart under deuteranopia), while one colour per family passes with a worst pair of 19.0 ΔE.
"""

import json
import math
import re
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, NamedTuple

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl

ProductFamily = Literal["satellite", "reanalysis", "weather model"]
"""The three kinds of weather product the study pages compare."""

FAMILY_COLOURS: Final[dict[ProductFamily, str]] = {
    "satellite": ocf.BRAND_ORANGE,
    "reanalysis": ocf.DATA_SKY,
    "weather model": ocf.DATA_BLUE,
}
"""Each family's colour, from the main data colours OCF's brand guidelines allow in published work.

Data Sky has only 2.0:1 contrast with the background, which is acceptable because every mark is
labelled on the axis.
"""

FAMILY_COLOURS_LIGHT: Final[dict[ProductFamily, str]] = {
    "satellite": ocf.BRAND_ORANGE_LIGHT,
    "reanalysis": ocf.DATA_SKY_LIGHT,
    "weather model": ocf.DATA_BLUE_LIGHT,
}
"""The light shade of each family's colour, for a second condition of the same row.

A light shade sits too close to its parent to carry a distinction alone, so `interval_panel`
always draws it with a hollow point of a second shape.
"""

CONDITION_COLOURS: Final[tuple[str, ...]] = (ocf.BRAND_ORANGE, ocf.DATA_BLUE)
"""The colour of each condition, in the order the conditions are given, on a one-family panel.

Brand Orange and Data Blue pass the `dataviz` skill's `validate_palette.js` against the page
background: 32.5 ΔE apart under protanopia, the worst case, and both above 3:1 contrast. A panel
whose rows are all one family has no use for the family colour, so it colours the conditions
instead, which the season charts use for April to September and October to March.
"""

Panel = alt.LayerChart | alt.HConcatChart | alt.VConcatChart
"""A chart `figure` can set under its caption."""

BetterDirectionType = Literal["negative", "positive"]
"""Which sign of difference is the better one."""

NAMED_SUFFIX: Final[str] = " (planned)"
"""Ends the label of a planned row in a figure that also holds exploratory rows, set bold."""

PlanningType = Literal["planned", "exploratory", "mixed"]
"""Whether a figure's rows are all planned, all exploratory, or a mix of the two."""

PLANNING_NOTES: Final[dict[PlanningType, str]] = {
    "planned": "All rows are planned: written into the study plan before any result existed.",
    "exploratory": "All rows are exploratory.",
    "mixed": (
        "Planned: one of the comparisons written into the study plan before any result existed. "
        "Every other row is exploratory."
    ),
}
"""The subtitle line `figure` adds, saying once which rows are planned.

Only a figure that mixes the two kinds labels each planned row with `NAMED_SUFFIX`: where every
row shares one kind, a label on each would repeat what one subtitle line says.
"""

CONDITION_SHAPES: Final[tuple[str, ...]] = ("circle", "diamond", "square")
"""The point shape of each condition, in the order the conditions are given."""

CONTRAST_COLUMNS: Final[tuple[str, ...]] = (
    "Scope",
    "Contrast",
    "ΔMAE (pp of capacity)",
    "95% interval",
    "Excludes zero?",
    "Folds agreeing",
    "Rows",
)
"""The header of every contrast table a study report writes, cell by cell."""

CONTENT_WIDTH_PX: Final[int] = 680
"""The width of a published docs page's text column, which every figure is drawn to fill.

MkDocs Material sets its grid to 61rem, at 20 px a rem on a screen at least 76.25em wide, and
takes 12.1rem for each sidebar and 1.2rem either side of the text: 1,220 - 484 - 48 = 688 px. A
figure drawn to 680 px therefore shows at its drawn size, and its text at the size it was set.
"""

LABEL_WIDTH_PX: Final[int] = 220
"""The width of the row-label column to the left of every interval panel."""

PLOT_WIDTH_PX: Final[int] = CONTENT_WIDTH_PX - LABEL_WIDTH_PX - 10
"""The width of an interval panel's plot area, so the labels and the plot fill the text column."""

_LABEL_CHARACTERS: Final[int] = 30
"""The characters a row-label line holds before wrapping, in the monospaced label font."""

_TITLE_CHARACTERS: Final[int] = 72
"""The characters a figure title's line holds before wrapping."""

_TEXT_CHARACTERS: Final[int] = 115
"""The characters a subtitle or axis-title line holds before wrapping."""

_AXIS_TITLE_CHARACTERS: Final[int] = 78
"""The characters an interval panel's axis-title line holds before wrapping, at `PLOT_WIDTH_PX`."""

_ZERO_LABEL_ROOM: Final[float] = 0.25
"""The share of the axis the zero label needs on its side of the rule to stay inside the plot."""

_INTERVAL: Final[re.Pattern[str]] = re.compile(r"^\[(\S+), (\S+)\]$")
_FOLDS: Final[re.Pattern[str]] = re.compile(r"^(\d+) of (\d+)$")
_ROW_STEP_PX: Final[int] = 40
"""Each y-axis label's height, fixed rather than scaled by any row's wrapped label, which
multiplying by the panel's longest label inflated every row to the tallest one's height.

Where a label carries more than one condition, offset side by side, Vega-Lite's step size is the
height of one (label, condition) position, not one label, so `interval_panel` divides this by the
number of conditions sharing a label, and the conditions' bands add back up to this height."""
_POINT_SIZE: Final[int] = 70
_PANEL_TITLE_PX: Final[int] = ocf.font_size(style="Body Large", body_px=11)
_KEY_TITLE_PX: Final[int] = ocf.font_size(style="Body", body_px=11)


class ContrastKey(NamedTuple):
    """The four fields that pick one row out of a report's contrast tables."""

    section: str
    scope: str
    treatment: str
    reference: str


def _cells(line: str) -> tuple[str, ...]:
    """Return a markdown table row's cells, stripped of padding."""
    return tuple(cell.strip() for cell in line.strip().strip("|").split("|"))


def _number(text: str) -> float:
    """Read a report number, taking the minus sign `−` as well as a hyphen as negative."""
    return float(text.replace("−", "-"))


def _contrast_row(*, cells: tuple[str, ...], section: str, line_number: int) -> dict[str, object]:
    """Parse one row of a contrast table.

    Args:
        cells: The row's cells.
        section: The heading above the row's table.
        line_number: The row's line in the report, for the error message.

    Returns:
        The row's fields, named as `report_contrasts` returns them.

    Raises:
        ValueError: If any cell does not read as the contrast table's format.
    """
    try:
        scope, contrast, difference, interval, excludes, folds, n_rows = cells
        treatment, reference = contrast.split(" − ")
        bounds = _INTERVAL.match(interval)
        agreeing = _FOLDS.match(folds)
        if bounds is None or agreeing is None or excludes not in ("**yes**", "no"):
            raise ValueError(cells)  # noqa: TRY301 - reported with the line number below
        return {
            "section": section,
            "scope": scope,
            "treatment": treatment,
            "reference": reference,
            "difference": _number(difference),
            "lower_95": _number(bounds.group(1)),
            "upper_95": _number(bounds.group(2)),
            "excludes_zero": excludes == "**yes**",
            "folds_agreeing": int(agreeing.group(1)),
            "n_folds": int(agreeing.group(2)),
            "n_rows": int(n_rows.replace(",", "")),
        }
    except ValueError as error:
        msg = f"line {line_number} is not a contrast row: {'|'.join(cells)}"
        raise ValueError(msg) from error


def report_contrasts(*, report_path: Path) -> pl.DataFrame:
    """Read every contrast table in a study report, and nothing else.

    A table counts as a contrast table only when its header is `CONTRAST_COLUMNS`, so the error
    table and the implied-capacity table are skipped.

    Args:
        report_path: The `report.md` a study script wrote.

    Returns:
        One row per contrast-table row, with `section` (the text of the nearest heading above the
        table), `scope`, `treatment`, `reference`, `difference`, `lower_95` and `upper_95` in
        percentage points of capacity, `excludes_zero`, `folds_agreeing`, `n_folds`, and
        `n_rows`.
    """
    section = ""
    in_contrast_table = False
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(report_path.read_text().splitlines(), start=1):
        if line.startswith("#"):
            section = line.lstrip("#").strip()
        if not line.startswith("|"):
            in_contrast_table = False
            continue
        cells = _cells(line)
        if cells == CONTRAST_COLUMNS:
            in_contrast_table = True
        elif in_contrast_table and not set(line) <= set("|-"):
            rows.append(_contrast_row(cells=cells, section=section, line_number=line_number))
    return pl.DataFrame(rows)


def report_errors(*, report_path: Path, column: str) -> dict[str, float]:
    """Read one column of a study report's first table, each product's mean absolute error.

    Args:
        report_path: The `report.md` a study script wrote.
        column: The header of the column to read, such as `Global only`.

    Returns:
        Each product's value in that column, in percentage points of capacity.

    Raises:
        ValueError: If the first table has no such column.
    """
    table: list[tuple[str, ...]] = []
    for line in report_path.read_text().splitlines():
        if line.startswith("|"):
            table.append(_cells(line))
        elif table:
            break
    header = table[0]
    if column not in header:
        msg = f"the first table in {report_path} has no column {column!r}: {header}"
        raise ValueError(msg)
    index = header.index(column)
    return {cells[0]: _number(cells[index]) for cells in table[2:]}


def select_contrasts(*, contrasts: pl.DataFrame, wanted: Sequence[ContrastKey]) -> pl.DataFrame:
    """Pick report rows by section, scope and contrast, in the order asked for.

    Args:
        contrasts: The output of `report_contrasts`.
        wanted: The rows to pick.

    Returns:
        One row per key, in the order of `wanted`.

    Raises:
        ValueError: Naming every key that matches no row or more than one.
    """
    selected: list[pl.DataFrame] = []
    problems: list[str] = []
    for key in wanted:
        matches = contrasts.filter(
            pl.col("section") == key.section,
            pl.col("scope") == key.scope,
            pl.col("treatment") == key.treatment,
            pl.col("reference") == key.reference,
        )
        if matches.height != 1:
            problems.append(f"{key} matches {matches.height} rows")
        selected.append(matches)
    if problems:
        raise ValueError("; ".join(problems))
    return pl.concat(selected)


def flip_contrast(*, contrasts: pl.DataFrame) -> pl.DataFrame:
    """Turn each treatment − reference row into reference − treatment.

    Args:
        contrasts: Rows carrying `treatment`, `reference`, `difference`, `lower_95`, and
            `upper_95`.

    Returns:
        The same rows with the two arms swapped, the estimate negated, and the bounds swapped and
        negated.
    """
    return contrasts.with_columns(
        treatment=pl.col("reference"),
        reference=pl.col("treatment"),
        difference=-pl.col("difference"),
        lower_95=-pl.col("upper_95"),
        upper_95=-pl.col("lower_95"),
    )


def planning(*, rows: Sequence[pl.DataFrame]) -> PlanningType:
    """Work out whether a figure's rows are all planned, all exploratory, or a mix.

    Args:
        rows: The rows of every interval panel in the figure. A row is planned where its
            Boolean `planned` column is true; a frame without the column is all exploratory.

    Returns:
        `exploratory` where no row is planned, which includes a figure with no rows, `planned`
        where every row is, and `mixed` otherwise.
    """
    flags = [
        flag
        for frame in rows
        for flag in (
            frame["planned"].fill_null(value=False).to_list()
            if "planned" in frame.columns
            else [False] * frame.height
        )
    ]
    if not any(flags):
        return "exploratory"
    return "planned" if all(flags) else "mixed"


def _labelled(*, rows: pl.DataFrame, figure_planning: PlanningType) -> pl.DataFrame:
    """Append `NAMED_SUFFIX` to each planned row's label, only in a figure of mixed rows.

    Args:
        rows: A panel's rows, with a Boolean `planned` column or without one.
        figure_planning: What `planning` returns for the figure the panel belongs to.

    Returns:
        The rows, with `NAMED_SUFFIX` on each planned row's label where `figure_planning` is
        `mixed`.

    Raises:
        ValueError: If the panel's own rows contradict `figure_planning`, such as a planned row in
            a figure said to be all exploratory.
    """
    panel_planning = planning(rows=[rows])
    if figure_planning not in ("mixed", panel_planning):
        msg = f"the figure is said to be all {figure_planning}, but this panel is {panel_planning}"
        raise ValueError(msg)
    if figure_planning != "mixed" or "planned" not in rows.columns:
        return rows
    return rows.with_columns(
        label=pl.when(pl.col("planned"))
        .then(pl.col("label") + NAMED_SUFFIX)
        .otherwise(pl.col("label"))
    )


def _shade_scale() -> alt.Scale:
    """Return the colour scale every panel shares: each family, then each family's light shade."""
    return alt.Scale(
        domain=[*FAMILY_COLOURS, *(f"{family}, light" for family in FAMILY_COLOURS_LIGHT)],
        range=[*FAMILY_COLOURS.values(), *FAMILY_COLOURS_LIGHT.values()],
    )


def _reference_layers(
    *,
    x_domain: tuple[float, float],
    zero_label: str,
    better_label: str,
    better_direction: BetterDirectionType,
    labelled: bool,
) -> list[alt.Chart]:
    """Return the zero rule, its label, and the label saying which direction is better.

    Args:
        x_domain: The x axis's range.
        zero_label: What a difference of zero means, such as `same as ERA5`.
        better_label: What the better direction means, such as `better than ERA5`.
        better_direction: Which sign of difference is the better one.
        labelled: Whether to draw the two labels, or the rule alone.

    Returns:
        The rule, then the two labels if `labelled`.
    """
    anchor = pl.DataFrame({"x": [0.0]})
    rule = (
        alt.Chart(anchor)
        .mark_rule(color=ocf.BLACK_1, strokeWidth=1)
        .encode(x=alt.X("x:Q", scale=alt.Scale(domain=list(x_domain), nice=False, zero=False)))  # ty: ignore[unresolved-attribute]
    )
    # The zero label sits on the side of the rule away from the better-direction label, so the
    # two collide only if the better-direction label crosses zero, unless that side holds less
    # than a quarter of the axis, where the label would run off the plot.
    low, high = x_domain
    to_the_right = better_direction == "negative"
    if to_the_right and high / (high - low) < _ZERO_LABEL_ROOM:
        to_the_right = False
    if not to_the_right and -low / (high - low) < _ZERO_LABEL_ROOM:
        to_the_right = True
    zero_text = (
        alt.Chart(anchor)
        .mark_text(
            align="left" if to_the_right else "right",
            dx=4 if to_the_right else -4,
            baseline="bottom",
            dy=-4,
            color=ocf.BLACK_1,
        )
        .encode(x="x:Q", y=alt.value(0), text=alt.value(zero_label))  # ty: ignore[unresolved-attribute]
    )
    edge = x_domain[0] if better_direction == "negative" else x_domain[1]
    text = f"← {better_label}" if better_direction == "negative" else f"{better_label} →"
    align = "left" if better_direction == "negative" else "right"
    # The better-direction label is anchored at the axis edge and reads towards the zero rule, so
    # it collides with the zero label wherever the edge sits too close to zero for the label's own
    # text to fit beside it, such as every horizon against day 0, where the difference cannot go
    # below zero and the edge sits right next to the rule. Stack the better-direction label above
    # the zero label instead of relying on the horizontal room that keeps them apart elsewhere.
    room = abs(edge) / (high - low)
    crowded = room < _ZERO_LABEL_ROOM
    better = (
        alt.Chart(pl.DataFrame({"x": [edge]}))
        .mark_text(
            align=align,
            baseline="bottom",
            dy=-16 if crowded else -4,
            color=ocf.BLACK_1,
            fontWeight="bold",
        )
        .encode(x="x:Q", y=alt.value(0), text=alt.value(text))  # ty: ignore[unresolved-attribute]
    )
    return [rule, zero_text, better] if labelled else [rule]


def axis_title_with_direction(
    *, x_title: str, better_label: str, better_direction: BetterDirectionType
) -> str:
    """Add to an axis title which direction is better, in words.

    An arrow beside the zero rule is easy to miss, so the axis title says it too. The words go
    inside the title's closing parenthesis where it has one, such as the unit in `(points of
    capacity)`, and in a new parenthesis otherwise.

    Args:
        x_title: The axis title, naming the quantity and its unit; empty for a panel stacked
            under another that carries the title.
        better_label: What the better direction means, such as `better than ERA5`.
        better_direction: Which sign of difference is the better one.

    Returns:
        The title with, for example, `more negative means better than ERA5` added, or the empty
        string unchanged.
    """
    if not x_title:
        return x_title
    direction = f"more {better_direction} means {better_label}"
    if x_title.endswith(")"):
        return f"{x_title[:-1]}; {direction})"
    return f"{x_title} ({direction})"


def wrapped(*, text: str, width: int = _TEXT_CHARACTERS) -> list[str]:
    """Split text into lines of at most `width` characters, for a Vega-Lite title or label.

    Args:
        text: The text.
        width: The most characters a line holds.

    Returns:
        The lines, one for text that already fits.
    """
    return textwrap.wrap(text, width=width, break_long_words=False) or [text]


def interval_panel(
    *,
    rows: pl.DataFrame,
    x_domain: tuple[float, float],
    x_title: str,
    zero_label: str,
    better_label: str,
    better_direction: BetterDirectionType = "negative",
    conditions: Sequence[str] = (),
    condition_colours: Sequence[str] | None = None,
    condition_title: str = "",
    panel_title: str | Sequence[str] = "",
    reference_labels: bool = True,
    family_key: bool = True,
    width: int = PLOT_WIDTH_PX,
    figure_planning: PlanningType = "mixed",
) -> alt.LayerChart | alt.VConcatChart:
    """Draw one panel of dots and 95% interval lines beside a labelled zero rule.

    The row labels take a column `LABEL_WIDTH_PX` wide, wrapped onto more lines where they are
    longer, so the labels and a plot of the default width fill the docs page's text column. Each
    row's colour is its family's. A row with a `condition` other than the first in `conditions`
    is drawn in the light shade of that colour, with a hollow point of a second shape, and where
    a label carries more than one condition the rows are offset vertically. In a figure of mixed
    rows, each planned row's label ends in `NAMED_SUFFIX` and is set bold. The colour scale's
    domain is every family, so a family keeps its colour whichever families a panel holds, and
    the legend lists only the families present. The keys sit in a row above the plot, inside the
    text column: the family key only where the panel holds more than one family, and the
    condition key where `conditions` is given. A panel of one family, with no more conditions
    than `CONDITION_COLOURS` holds, colours its conditions with those colours instead of a light
    shade, still with a hollow point of a second shape.

    Passing `condition_colours` overrides all of that with the colour-first encoding this
    project's charts default to: every condition gets its own solid colour, drawn filled with one
    shared marker shape and no light-shade or hollow second style, so identity rests on colour
    alone. Use it where `conditions` holds more values than `CONDITION_COLOURS` can tell apart, or
    where a hollow/filled distinction would only add noise.

    Args:
        rows: One row per mark, with `label`, `family` (a `ProductFamily`), `difference`,
            `lower_95` and `upper_95`, `condition` if `conditions` is given, and a Boolean
            `planned` if any row is planned. Rows sharing a label share `planned`. Rows are drawn
            top to bottom in the order given.
        x_domain: The x axis's range, set explicitly so two panels can share it.
        x_title: The x axis's title, naming the quantity and its unit. The better direction is
            added to it in words by `axis_title_with_direction`.
        zero_label: What a difference of zero means, such as `same as ERA5`.
        better_label: What the better direction means, such as `better than ERA5`.
        better_direction: Which sign of difference is the better one.
        conditions: The values of `condition`, in legend order; the first is drawn filled, unless
            `condition_colours` is given.
        condition_colours: Each condition's own colour, in the same order as `conditions`. Leave
            unset for the default family-and-shade encoding described above.
        condition_title: The legend title for `condition`.
        panel_title: A title above this panel alone, or lines a caller has already wrapped
            (`wrapped`) where the title does not fit on one line at this panel's width.
        reference_labels: Whether to label the zero rule and the better direction, which a panel
            stacked under another that already carries them can leave out.
        family_key: Whether to draw the family key, which a panel stacked under another that
            already carries it can leave out.
        width: The plot's width in pixels, `PLOT_WIDTH_PX` unless the panel shares a row.
        figure_planning: What `planning` returns for every panel in the figure, which decides
            whether a planned row's label carries `NAMED_SUFFIX`.

    Returns:
        The panel, under its keys where it has any.
    """
    rows = _labelled(rows=rows, figure_planning=figure_planning)
    families = [family for family in FAMILY_COLOURS if family in set(rows["family"].to_list())]
    explicit_colours = condition_colours is not None
    colour_conditions = 0 < len(conditions) <= len(CONDITION_COLOURS) and len(families) == 1
    shade = pl.col("family")
    shade_scale = _shade_scale()
    if explicit_colours:
        shade = pl.col("condition")
        shade_scale = alt.Scale(domain=list(conditions), range=list(condition_colours))
    elif colour_conditions:
        shade = pl.col("condition")
        shade_scale = alt.Scale(
            domain=list(conditions), range=list(CONDITION_COLOURS[: len(conditions)])
        )
    elif conditions:
        shade = (
            pl.when(pl.col("condition") == conditions[0])
            .then(pl.col("family"))
            .otherwise(pl.col("family") + ", light")
        )
    data = rows.with_columns(pl.col("difference", "lower_95", "upper_95").round(3), shade=shade)
    labels = list(dict.fromkeys(data["label"].to_list()))
    lines = {label: wrapped(text=label, width=_LABEL_CHARACTERS) for label in labels}
    encodings: dict[str, object] = {
        "y": alt.Y(
            "label:N",
            sort=labels,
            title=None,
            axis=alt.Axis(
                labelExpr=f"{json.dumps(lines)}[datum.value]",
                labelLimit=LABEL_WIDTH_PX,
                minExtent=LABEL_WIDTH_PX,
                maxExtent=LABEL_WIDTH_PX,
                labelPadding=6,
                ticks=False,
                domain=False,
                labelFontWeight=alt.ExprRef(
                    expr=f"indexof(datum.value, '{NAMED_SUFFIX.strip()}') >= 0 ? 'bold' : 'normal'"
                ),
            ),
        ),
        "color": alt.Color(
            "shade:N",
            scale=shade_scale,
            legend=None,
        ),
    }
    if conditions and data["label"].is_duplicated().any():
        encodings["yOffset"] = alt.YOffset("condition:N", sort=list(conditions))
    if conditions and not explicit_colours:
        encodings["shape"] = alt.Shape(
            "condition:N",
            scale=alt.Scale(
                domain=list(conditions), range=list(CONDITION_SHAPES[: len(conditions)])
            ),
            legend=None,
        )
    x_title_lines = wrapped(
        text=axis_title_with_direction(
            x_title=x_title, better_label=better_label, better_direction=better_direction
        ),
        width=_AXIS_TITLE_CHARACTERS,
    )
    x_scale = alt.Scale(domain=list(x_domain), nice=False, zero=False)
    x_axis = alt.Axis(values=ticks(x_domain=x_domain), format=".2~f")
    interval = (
        alt.Chart(data)
        .mark_rule(strokeWidth=2, clip=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lower_95:Q", scale=x_scale, title=x_title_lines, axis=x_axis),
            x2="upper_95:Q",
            **{key: value for key, value in encodings.items() if key != "shape"},
        )
    )
    first = pl.col("condition") == conditions[0] if conditions else pl.lit(value=True)
    x = alt.X("difference:Q", scale=x_scale, title=x_title_lines, axis=x_axis)
    tooltip = [
        alt.Tooltip("label:N", title="Row"),
        alt.Tooltip("difference:Q", title="Estimate"),
        alt.Tooltip("lower_95:Q", title="Lower 95%"),
        alt.Tooltip("upper_95:Q", title="Upper 95%"),
    ]
    points = (
        [
            alt.Chart(data)
            .mark_point(filled=True, size=_POINT_SIZE, opacity=1, clip=True, aria=False)
            .encode(x=x, tooltip=tooltip, **encodings)  # ty: ignore[unresolved-attribute]
        ]
        if explicit_colours
        else [
            alt.Chart(data.filter(first))
            .mark_point(filled=True, size=_POINT_SIZE, opacity=1, clip=True, aria=False)
            .encode(x=x, tooltip=tooltip, **encodings),  # ty: ignore[unresolved-attribute]
            alt.Chart(data.filter(~first))
            .mark_point(
                filled=False, size=_POINT_SIZE, strokeWidth=2, opacity=1, clip=True, aria=False
            )
            .encode(x=x, tooltip=tooltip, **encodings),  # ty: ignore[unresolved-attribute]
        ]
    )
    reference = _reference_layers(
        x_domain=x_domain,
        zero_label=zero_label,
        better_label=better_label,
        better_direction=better_direction,
        labelled=reference_labels,
    )
    offset_positions = len(conditions) if "yOffset" in encodings else 1
    panel = alt.LayerChart(
        layer=[*reference, interval, *points],
        width=width,
        height=alt.Step(_ROW_STEP_PX / offset_positions),
        title=alt.TitleParams(panel_title, anchor="start", frame="group", fontSize=_PANEL_TITLE_PX),
    )
    keys = []
    if family_key and len(families) > 1:
        keys.append(
            _key(
                title="Product type",
                labels=families,
                shapes=["circle"] * len(families),
                filled=[True] * len(families),
                colours=[FAMILY_COLOURS[family] for family in families],
            )
        )
    if conditions:
        keys.append(
            _key(
                title=condition_title,
                labels=conditions,
                shapes=(
                    ["circle"] * len(conditions)
                    if explicit_colours
                    else CONDITION_SHAPES[: len(conditions)]
                ),
                filled=(
                    [True] * len(conditions)
                    if explicit_colours
                    else [index == 0 for index in range(len(conditions))]
                ),
                colours=(
                    list(condition_colours)
                    if explicit_colours
                    else CONDITION_COLOURS[: len(conditions)]
                    if colour_conditions
                    else [ocf.BLACK_1] * len(conditions)
                ),
            )
        )
    return alt.vconcat(*keys, panel, spacing=8) if keys else panel


def leaderboard_panel(
    *,
    rows: pl.DataFrame,
    x_domain: tuple[float, float],
    x_title: str,
    width: int = PLOT_WIDTH_PX,
    conditions: Sequence[str] = (),
    condition_title: str = "",
    kinds: Sequence[str] = (),
    kind_title: str = "",
    panel_title: str = "",
    keys: bool = True,
    solid: bool = False,
    row_step_px: int | None = None,
) -> alt.LayerChart | alt.VConcatChart:
    """Draw one product per row, best first, as a dot at its own error with a 95% interval.

    Unlike `interval_panel`, a leaderboard row is an absolute quantity, not a difference from a
    reference arm, so it draws no zero rule and no "better than" direction label: the axis title
    states the direction in words instead, as `x_title` already must.

    Each row is coloured by its family, unless `conditions` is given: then each row is coloured by
    its `condition` from `CONDITION_COLOURS`, with the first condition's point filled and every
    other condition's hollow, so the distinction survives without colour, unless `solid` is set.
    Where `kinds` is given, each row's point takes its `kind`'s shape from `CONDITION_SHAPES`. The
    keys sit in a row above the plot: the family key only where the panel holds more than one
    family and no `conditions`, the condition key where `conditions` is given, and the kind key
    where `kinds` is given.

    Args:
        rows: One row per product, with `label`, `family` (a `ProductFamily`), `value`,
            `lower_95` and `upper_95`, `condition` if `conditions` is given, and `kind` if `kinds`
            is given, in the order to draw them top to bottom (best first).
        x_domain: The x axis's range, set explicitly so the panel and any panel sharing its scale
            agree.
        x_title: The x axis's title, naming the quantity, its unit, and which direction is
            better, such as "Mean absolute error (% of capacity; smaller is better)".
        width: The plot's width in pixels.
        conditions: The values of `condition`, in key order, at most as many as
            `CONDITION_COLOURS` holds; the first is drawn filled, unless `solid` is set.
        condition_title: The key title for `condition`.
        kinds: The values of `kind`, in key order, at most as many as `CONDITION_SHAPES` holds.
        kind_title: The key title for `kind`.
        panel_title: A title above this panel alone.
        keys: Whether to draw the keys, which a panel stacked under another that already carries
            them can leave out.
        solid: Whether every condition's point is drawn filled, with no hollow second style — the
            colour-first default this project's charts favour when colour alone can carry
            `conditions`. False keeps the first condition filled and the rest hollow.
        row_step_px: The height of every row in pixels. None sizes every row to the panel's
            longest wrapped label, which leaves a row with a one-line label mostly empty when one
            label wraps over several lines; a fixed step lets that label sit close to its
            neighbours instead.

    Returns:
        The panel, under its keys where it has any.

    Raises:
        ValueError: If `conditions` holds more values than `CONDITION_COLOURS`, or `kinds` more
            than `CONDITION_SHAPES`.
    """
    if len(conditions) > len(CONDITION_COLOURS) or len(kinds) > len(CONDITION_SHAPES):
        msg = f"too many conditions ({len(conditions)}) or kinds ({len(kinds)}) to draw apart"
        raise ValueError(msg)
    families = [family for family in FAMILY_COLOURS if family in set(rows["family"].to_list())]
    data = rows.with_columns(pl.col("value", "lower_95", "upper_95").round(3))
    labels = data["label"].to_list()
    lines = {label: wrapped(text=label, width=_LABEL_CHARACTERS) for label in labels}
    y = alt.Y(
        "label:N",
        sort=labels,
        title=None,
        axis=alt.Axis(
            labelExpr=f"{json.dumps(lines)}[datum.value]",
            labelLimit=LABEL_WIDTH_PX,
            minExtent=LABEL_WIDTH_PX,
            maxExtent=LABEL_WIDTH_PX,
            labelPadding=6,
            ticks=False,
            domain=False,
        ),
    )
    colour = (
        alt.Color(
            "condition:N",
            scale=alt.Scale(
                domain=list(conditions), range=list(CONDITION_COLOURS[: len(conditions)])
            ),
            legend=None,
        )
        if conditions
        else alt.Color(
            "family:N",
            scale=alt.Scale(domain=list(FAMILY_COLOURS), range=list(FAMILY_COLOURS.values())),
            legend=None,
        )
    )
    shape: dict[str, alt.Shape] = (
        {
            "shape": alt.Shape(
                "kind:N",
                scale=alt.Scale(domain=list(kinds), range=list(CONDITION_SHAPES[: len(kinds)])),
                legend=None,
            )
        }
        if kinds
        else {}
    )
    x_title_lines = wrapped(text=x_title, width=_AXIS_TITLE_CHARACTERS)
    x_scale = alt.Scale(domain=list(x_domain), nice=False, zero=False)
    x_axis = alt.Axis(values=ticks(x_domain=x_domain), format=".2~f")
    tooltip = [
        alt.Tooltip("label:N", title="Product"),
        alt.Tooltip("value:Q", title="Mean absolute error"),
        alt.Tooltip("lower_95:Q", title="Lower 95%"),
        alt.Tooltip("upper_95:Q", title="Upper 95%"),
    ]
    interval = (
        alt.Chart(data)
        .mark_rule(strokeWidth=2, clip=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lower_95:Q", scale=x_scale, title=x_title_lines, axis=x_axis),
            x2="upper_95:Q",
            y=y,
            color=colour,
        )
    )
    # The first condition's points are filled and every other condition's hollow, drawn as two
    # layers because a mark's fill is not an encoding channel. `solid` skips the split so every
    # condition is filled and colour alone carries the distinction.
    groups = [(data, True)]
    if conditions and not solid:
        first = pl.col("condition") == conditions[0]
        groups = [(data.filter(first), True), (data.filter(~first), False)]
    points = [
        alt.Chart(frame)
        .mark_point(
            filled=filled,
            size=_POINT_SIZE,
            opacity=1,
            clip=True,
            aria=False,
            strokeWidth=alt.Undefined if filled else 2,
        )
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("value:Q", scale=x_scale, title=x_title_lines, axis=x_axis),
            y=y,
            color=colour,
            tooltip=tooltip,
            **shape,
        )
        for frame, filled in groups
    ]
    panel = alt.LayerChart(
        layer=[interval, *points],
        width=width,
        height=alt.Step(
            row_step_px
            if row_step_px is not None
            else _ROW_STEP_PX * max(len(line) for line in lines.values())
        ),
        title=alt.TitleParams(panel_title, anchor="start", frame="group", fontSize=_PANEL_TITLE_PX),
    )
    drawn_keys = []
    if keys and not conditions and len(families) > 1:
        drawn_keys.append(
            _key(
                title="Product type",
                labels=families,
                shapes=["circle"] * len(families),
                filled=[True] * len(families),
                colours=[FAMILY_COLOURS[family] for family in families],
            )
        )
    if keys and conditions:
        drawn_keys.append(
            _key(
                title=condition_title,
                labels=conditions,
                shapes=["circle"] * len(conditions),
                filled=(
                    [True] * len(conditions)
                    if solid
                    else [index == 0 for index in range(len(conditions))]
                ),
                colours=CONDITION_COLOURS[: len(conditions)],
            )
        )
    if keys and kinds:
        drawn_keys.append(
            _key(
                title=kind_title,
                labels=kinds,
                shapes=CONDITION_SHAPES[: len(kinds)],
                filled=[True] * len(kinds),
                colours=[ocf.BLACK_1] * len(kinds),
            )
        )
    return alt.vconcat(*drawn_keys, panel, spacing=8) if drawn_keys else panel


def ticks(*, x_domain: tuple[float, float]) -> list[float]:
    """Return round tick values inside an x range, 4 to 10 of them.

    Args:
        x_domain: The x axis's range.

    Returns:
        Every multiple of the smallest round step giving at most 10 ticks, inside the range.
    """
    low, high = x_domain
    for step in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0):
        first = math.ceil(round(low / step, 9))
        last = math.floor(round(high / step, 9))
        if last - first < 10:
            return [round(index * step, 2) for index in range(first, last + 1)]
    msg = f"no round step gives at most 10 ticks on {x_domain}"
    raise ValueError(msg)


def _key(
    *,
    title: str,
    labels: Sequence[str],
    shapes: Sequence[str],
    filled: Sequence[bool],
    colours: Sequence[str],
) -> alt.LayerChart:
    """Draw a key in one row above a panel: one symbol and label per entry, in equal slots.

    A Vega-Lite legend sits outside the panel and draws every shape filled alike, so it can
    neither stay inside the text column's width nor show which condition is hollow.

    Args:
        title: The key's title.
        labels: Each entry's label.
        shapes: Each entry's point shape.
        filled: Whether each entry's point is filled.
        colours: Each entry's colour.

    Returns:
        A one-row chart `PLOT_WIDTH_PX` wide, aligned with the plot area beneath it, taller where
        a label needs a second line.
    """
    slot = PLOT_WIDTH_PX // len(labels)
    # About 7 px a character at this text mark's font size (the same estimate `_LABEL_CHARACTERS`
    # rests on for the row-label column): a label that does not fit `slot` in one line wraps onto
    # a second rather than being cut off with an ellipsis mid-word.
    chars_per_line = max(10, (slot - 24) // 7)
    wrapped_labels = [wrapped(text=label, width=chars_per_line) for label in labels]
    lines = max(len(label_lines) for label_lines in wrapped_labels)
    data = pl.DataFrame(
        {
            "label": ["\n".join(label_lines) for label_lines in wrapped_labels],
            "shape": list(shapes),
            "filled": list(filled),
            "colour": list(colours),
            "x": [6 + index * slot for index in range(len(labels))],
        }
    )
    points = [
        alt.Chart(data.filter(pl.col("filled") == is_filled))
        .mark_point(size=_POINT_SIZE, strokeWidth=2, filled=is_filled, opacity=1)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("x:Q", scale=None),
            y=alt.value(8),
            shape=alt.Shape("shape:N", scale=None),
            color=alt.Color("colour:N", scale=None),
        )
        for is_filled in (True, False)
    ]
    text = (
        alt.Chart(data)
        .mark_text(
            align="left",
            baseline="middle",
            dx=12,
            dy=8,
            color=ocf.BLACK_1,
            lineHeight=13,
            lineBreak="\n",
        )
        .encode(x=alt.X("x:Q", scale=None), y=alt.value(8), text="label:N")  # ty: ignore[unresolved-attribute]
    )
    return alt.LayerChart(
        layer=[*points, text],
        width=PLOT_WIDTH_PX,
        height=16 if lines == 1 else 16 + 13 * (lines - 1),
        title=alt.TitleParams(title, anchor="start", fontSize=_KEY_TITLE_PX),
    )


def figure(
    *,
    panels: Sequence[Panel],
    number: int | str,
    title: str,
    subtitle: Sequence[str],
    figure_planning: PlanningType | None,
) -> alt.VConcatChart:
    """Stack panels, one above the other, under a "Figure N:" caption.

    The figure is drawn `CONTENT_WIDTH_PX` wide, so a docs page shows it at the size it was set
    rather than scaling its text down. The title and each subtitle line wrap to fit that width.
    Any Vega-Lite legend a panel draws itself sits below the panels, in one row.

    Args:
        panels: The panels, drawn one above the other.
        number: The figure's number on its page, or a lettered sub-figure such as `"6a"`.
        title: The finding the figure shows.
        subtitle: Short lines naming the quantity, its scope, and what a dot and a line mean.
        figure_planning: What `planning` returns for the figure's rows, which adds the matching
            `PLANNING_NOTES` line to the subtitle, or `None` for a figure with no planned or
            exploratory rows to describe.

    Returns:
        The figure.
    """
    caption = alt.TitleParams(
        wrapped(text=f"Figure {number}: {title}", width=_TITLE_CHARACTERS),
        subtitle=[
            line
            for text in (
                *subtitle,
                *(() if figure_planning is None else (PLANNING_NOTES[figure_planning],)),
            )
            for line in wrapped(text=text)
        ],
        anchor="start",
        offset=14,
        subtitleColor=ocf.BLACK_1,
        subtitlePadding=6,
    )
    return (
        alt.vconcat(*panels, spacing=24)
        .properties(title=caption)
        .resolve_scale(color="shared", shape="shared")
        .configure_view(stroke=None)
        .configure_legend(orient="bottom", direction="horizontal")
    )
