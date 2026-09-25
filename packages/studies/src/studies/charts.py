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

from studies.bootstrap import bootstrap_absolute, bootstrap_difference

ProductFamily = Literal["satellite", "reanalysis", "weather model", "station observations"]
"""The kinds of weather input the study pages compare: three kinds of gridded product, and the
weather-station observations, alone or blended with a gridded product."""

FAMILY_COLOURS: Final[dict[ProductFamily, str]] = {
    "satellite": ocf.BRAND_ORANGE,
    "reanalysis": ocf.DATA_SKY,
    "weather model": ocf.DATA_BLUE,
    "station observations": ocf.DATA_PURPLE,
}
"""Each family's colour, from the main data colours OCF's brand guidelines allow in published work.

Data Sky has only 2.0:1 contrast with the background, which is acceptable because every mark is
labelled on the axis.
"""

FAMILY_COLOURS_LIGHT: Final[dict[ProductFamily, str]] = {
    "satellite": ocf.BRAND_ORANGE_LIGHT,
    "reanalysis": ocf.DATA_SKY_LIGHT,
    "weather model": ocf.DATA_BLUE_LIGHT,
    "station observations": ocf.DATA_PURPLE_LIGHT,
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

POST_HOC_SUFFIX: Final[str] = " (post hoc)"
"""Ends the label of a row added after the first run, in a figure that also holds other rows."""

POST_HOC_PLANNING_NOTE: Final[str] = (
    "Planned: one of the comparisons written into the study plan before any result existed. "
    "Every other row is exploratory or, where marked, post hoc."
)
"""The `mixed` line of `PLANNING_NOTES` for a figure that holds rows ending in `POST_HOC_SUFFIX`."""

CONDITION_SHAPES: Final[tuple[str, ...]] = ("circle", "diamond", "square")
"""The point shape of each condition, in the order the conditions are given."""

SECOND_SETTING_SHAPE: Final[str] = "triangle-up"
"""The hollow shape that marks a contrast at the second hyperparameter setting.

It is none of `CONDITION_SHAPES`, so a marker never reads as a reference row's diamond.
"""

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

CONTRAST_COLUMNS_WITH_MONTHS: Final[tuple[str, ...]] = (*CONTRAST_COLUMNS, "Months")
"""The header of a contrast table that also counts the calendar months it rests on.

`report_contrasts` reads a table with this header only where the caller passes it as an extra
header, so a report that prints such tables and a caller that does not expect them are unaffected.
"""

TOO_FEW_MONTHS: Final[str] = "too few months"
"""What a report's `Excludes zero?` cell says where the rows are too few months to say."""

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
_BETTER_LABEL_CHARACTER_PX: Final[int] = 8
"""The width of one character of the bold better-direction label, generously rounded up."""

_VALUE_LABEL_PX: Final[int] = 125
"""The width of a printed estimate and interval such as `+0.10 [+0.05, +0.15]`."""
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
        scope, contrast, difference, interval, excludes, folds, n_rows, *months = cells
        if len(months) > 1 or (months and not months[0].isdigit()):
            raise ValueError(cells)  # noqa: TRY301 - reported with the line number below
        treatment, reference = contrast.split(" − ")
        bounds = _INTERVAL.match(interval)
        agreeing = _FOLDS.match(folds)
        if bounds is None or agreeing is None or excludes not in ("**yes**", "no", TOO_FEW_MONTHS):
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


def report_contrasts(
    *, report_path: Path, extra_headers: Sequence[tuple[str, ...]] = ()
) -> pl.DataFrame:
    """Read every contrast table in a study report, and nothing else.

    A table counts as a contrast table only when its header is `CONTRAST_COLUMNS` or one of
    `extra_headers`, so the error table and the implied-capacity table are skipped. A row of a
    table with the `CONTRAST_COLUMNS_WITH_MONTHS` header is read the same way, and its month count
    is dropped. A row whose `Excludes zero?` cell says `TOO_FEW_MONTHS` reads as not excluding
    zero.

    Args:
        report_path: The `report.md` a study script wrote.
        extra_headers: Further headers to read as contrast tables, such as
            `CONTRAST_COLUMNS_WITH_MONTHS`.

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
        if cells == CONTRAST_COLUMNS or cells in extra_headers:
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
    width: int,
) -> list[alt.Chart]:
    """Return the zero rule, its label, and the label saying which direction is better.

    Args:
        x_domain: The x axis's range.
        zero_label: What a difference of zero means, such as `same as ERA5`.
        better_label: What the better direction means, such as `better than ERA5`.
        better_direction: Which sign of difference is the better one.
        labelled: Whether to draw the two labels, or the rule alone.
        width: The plot's width in pixels, which decides whether the better-direction label's
            text fits between the axis edge and the rule.

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
    crowded = room < _ZERO_LABEL_ROOM or room * width < _BETTER_LABEL_CHARACTER_PX * len(text)
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
    key_families: Sequence[ProductFamily] | None = None,
    condition_key: bool = True,
    width: int = PLOT_WIDTH_PX,
    figure_planning: PlanningType = "mixed",
    value_labels: bool = False,
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
            `lower_95` and `upper_95`, `condition` if `conditions` is given, a Boolean
            `planned` if any row is planned, and `second_difference` if any row has a second
            hyperparameter setting (drawn as a hollow `SECOND_SETTING_SHAPE`, null for a row with
            none). Rows sharing a label share `planned`. Rows are drawn top to bottom in the
            order given.
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
        key_families: The families the family key lists, where a stacked figure's key must cover
            families the panel carrying it does not hold; `None` lists the panel's own.
        condition_key: Whether to draw the condition key, which a panel stacked under another
            that already carries it can leave out.
        width: The plot's width in pixels, `PLOT_WIDTH_PX` unless the panel shares a row.
        figure_planning: What `planning` returns for every panel in the figure, which decides
            whether a planned row's label carries `NAMED_SUFFIX`.
        value_labels: Whether to print each row's estimate and interval, signed and to two
            decimal places, beside the row, so an interval narrower than its own marker is still
            readable.

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
    rounded = [
        name
        for name in ("difference", "lower_95", "upper_95", "second_difference")
        if name in rows.columns
    ]
    data = rows.with_columns(pl.col(rounded).round(3), shade=shade)
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
    if "second_difference" in data.columns and data["second_difference"].is_not_null().any():
        points.append(
            alt.Chart(data.filter(pl.col("second_difference").is_not_null()))
            .mark_point(
                shape=SECOND_SETTING_SHAPE,
                filled=False,
                size=_POINT_SIZE,
                strokeWidth=2,
                opacity=1,
                clip=True,
                aria=False,
            )
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("second_difference:Q", scale=x_scale, title=x_title_lines, axis=x_axis),
                tooltip=[
                    alt.Tooltip("label:N", title="Row"),
                    alt.Tooltip("second_difference:Q", title="Second setting"),
                ],
                **{key: value for key, value in encodings.items() if key != "shape"},
            )
        )
    reference = _reference_layers(
        x_domain=x_domain,
        zero_label=zero_label,
        better_label=better_label,
        better_direction=better_direction,
        labelled=reference_labels,
        width=width,
    )
    if value_labels:
        points.extend(
            _value_label_layers(data=data, x_domain=x_domain, x_scale=x_scale, width=width)
        )
    offset_positions = len(conditions) if "yOffset" in encodings else 1
    panel = alt.LayerChart(
        layer=[*reference, interval, *points],
        width=width,
        height=alt.Step(_ROW_STEP_PX / offset_positions),
        title=alt.TitleParams(panel_title, anchor="start", frame="group", fontSize=_PANEL_TITLE_PX),
    )
    keys = []
    listed = list(key_families) if key_families is not None else families
    if family_key and len(listed) > 1:
        keys.append(
            _key(
                title="Product type",
                labels=listed,
                shapes=["circle"] * len(listed),
                filled=[True] * len(listed),
                colours=[FAMILY_COLOURS[family] for family in listed],
            )
        )
    if conditions and condition_key:
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


def _value_label_layers(
    *,
    data: pl.DataFrame,
    x_domain: tuple[float, float],
    x_scale: alt.Scale,
    width: int,
) -> list[alt.Chart]:
    """Return the text layers printing each row's estimate and interval beside the row.

    A row's text sits to the right of its interval's upper bound, or to the left of its lower
    bound where the right side has less than `_VALUE_LABEL_PX` of room and the left side has more.

    Args:
        data: The panel's rows, with `difference`, `lower_95` and `upper_95`.
        x_domain: The panel's x range.
        x_scale: The scale the panel's marks share.
        width: The plot's width in pixels.

    Returns:
        One text layer per side that holds a row.
    """
    low, high = x_domain
    pixels_per_point = width / (high - low)
    text = data.with_columns(
        text=pl.format(
            "{} [{}, {}]",
            *(
                pl.col(name).map_elements(lambda value: f"{value:+.2f}", return_dtype=pl.String)
                for name in ("difference", "lower_95", "upper_95")
            ),
        ),
        right_room=(high - pl.col("upper_95")) * pixels_per_point,
        left_room=(pl.col("lower_95") - low) * pixels_per_point,
    )
    on_left = (pl.col("right_room") < _VALUE_LABEL_PX) & (
        pl.col("left_room") > pl.col("right_room")
    )
    layers = []
    for side_is_left in (False, True):
        side = text.filter(on_left if side_is_left else ~on_left)
        if side.height == 0:
            continue
        layers.append(
            alt.Chart(side)
            .mark_text(
                align="right" if side_is_left else "left",
                dx=-7 if side_is_left else 7,
                baseline="middle",
                fontSize=11,
                aria=False,
            )
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("lower_95:Q" if side_is_left else "upper_95:Q", scale=x_scale, title=None),
                y=alt.Y("label:N", sort=list(dict.fromkeys(data["label"].to_list())), title=None),
                text="text:N",
                color=alt.value(ocf.BLACK_1),
            )
        )
    return layers


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
    key_families: Sequence[ProductFamily] | None = None,
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
    Where `kinds` is given, each row's point takes its `kind`'s shape from `CONDITION_SHAPES`. Where
    the rows carry a Boolean `reference` column and `conditions` is not given, each reference row
    is drawn hollow, in the light shade of its family's colour, so a product repeated on every
    panel of a stacked figure reads as a yardstick and not as a competitor. The keys sit in a row
    above the plot: the family key only where the panel holds more than one
    family and no `conditions`, the condition key where `conditions` is given, and the kind key
    where `kinds` is given.

    Args:
        rows: One row per product, with `label`, `family` (a `ProductFamily`), `value`,
            `lower_95` and `upper_95`, `condition` if `conditions` is given, `kind` if `kinds` is
            given, and `reference` if some rows are reference rows, in the order to draw them top
            to bottom (best first).
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
        key_families: The families the family key lists, where a stacked figure's key must cover
            families the panel carrying it does not hold; `None` lists the panel's own.
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
    hollow_references = "reference" in rows.columns and not conditions
    data = rows.with_columns(pl.col("value", "lower_95", "upper_95").round(3))
    if hollow_references:
        data = data.with_columns(
            shade=pl.when(pl.col("reference"))
            .then(pl.col("family") + ", light")
            .otherwise(pl.col("family"))
        )
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
        else alt.Color("shade:N", scale=_shade_scale(), legend=None)
        if hollow_references
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
    elif hollow_references:
        reference = pl.col("reference")
        groups = [(data.filter(~reference), True), (data.filter(reference), False)]
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
    listed = list(key_families) if key_families is not None else families
    if keys and not conditions and len(listed) > 1:
        drawn_keys.append(
            _key(
                title="Product type",
                labels=listed,
                shapes=["circle"] * len(listed),
                filled=[True] * len(listed),
                colours=[FAMILY_COLOURS[family] for family in listed],
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
    post_hoc: bool = False,
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
        post_hoc: Whether any row ends in `POST_HOC_SUFFIX`, which swaps a `mixed` figure's line
            for `POST_HOC_PLANNING_NOTE`.

    Returns:
        The figure.
    """
    note = (
        POST_HOC_PLANNING_NOTE
        if post_hoc and figure_planning == "mixed"
        else None
        if figure_planning is None
        else PLANNING_NOTES[figure_planning]
    )
    caption = alt.TitleParams(
        wrapped(text=f"Figure {number}: {title}", width=_TITLE_CHARACTERS),
        subtitle=[
            line
            for text in (
                *subtitle,
                *(() if note is None else (note,)),
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


PERCENTAGE_POINTS: Final[float] = 100.0
"""Converts a loss expressed as a fraction of capacity into percentage points of capacity."""

ABSOLUTE_ERROR_X_TITLE: Final[str] = "Mean absolute error (% of capacity; smaller is better)"
"""The x axis title of a stacked leaderboard, which says which direction is better."""

REFERENCE_ROW_NOTE: Final[str] = (
    "Lighter, hollow rows are CAMS and ERA5, repeated in every block as a yardstick and scored "
    "on that block's own rows."
)
"""The subtitle line a stacked leaderboard adds to say what its hollow reference rows are."""

CONTRAST_REFERENCE_ROW_NOTE: Final[str] = (
    "The lighter, hollow row is CAMS, repeated in every block as a yardstick and contrasted with "
    "that block's own ERA5."
)
"""The subtitle line a stacked contrast chart adds to say what its hollow reference row is."""

_BLOCK_DOMAIN_STEP: Final[float] = 0.5
"""The multiple of percentage points a stacked figure's shared x range is rounded out to."""

_PLANNED_DOMAIN_STEP: Final[float] = 0.25
"""The multiple of percentage points a block's own planned-contrast x range is rounded out to."""

_BLOCK_ROW_STEP_PX: Final[int] = 26
"""The height of each row of a stacked figure's blocks, which hold one-line labels."""


CONTRAST_X_TITLE: Final[str] = "Mean absolute error minus ERA5's (points of capacity)"
"""The x axis title of a stacked contrast figure."""

REPORT_PRINT_DECIMALS: Final[int] = 3
"""The decimal places every study report prints its numbers at."""


class BlockArm(NamedTuple):
    """One arm of a row-set block, and how the block draws it."""

    arm: str
    label: str
    family: ProductFamily
    reference: bool = False
    planned: bool = False


def _contrast_x_title(*, reference_name: str) -> str:
    """Return a contrast panel's x axis title, naming the arm the contrasts are against."""
    if reference_name == "ERA5":
        return CONTRAST_X_TITLE
    return f"Mean absolute error minus {reference_name} (points of capacity)"


class RowSetBlock(NamedTuple):
    """One block of a stacked figure: the arms scored on one row set.

    `rows` holds one row per arm, with `arm`, `label`, `family`, `reference`, `planned`, and either
    `value` (a leaderboard block) or `difference` (a contrast block), each with `lower_95` and
    `upper_95`, all in percentage points of capacity. A contrast block's `rows` may also carry
    `second_difference`, the same contrast at the second hyperparameter setting, null where none
    was computed. `planned_rows` is `planned_contrast_rows`'s output for the row set's planned
    contrasts, which a contrast block draws in a lower panel; `None` draws no lower panel.
    `hours_unit` is what a row is called in the title, and `reference_name` is what a contrast
    block's zero rule and axis call the arm its contrasts are against, such as `ERA5` or
    `ERA5's 10 m wind`.
    """

    label: str
    dates: str
    site_hours: int
    rows: pl.DataFrame
    planned_rows: pl.DataFrame | None = None
    hours_unit: str = "site-hours"
    reference_name: str = "ERA5"

    @property
    def title(self) -> str:
        """The block's panel title, naming its row set, its dates, and its row count."""
        return f"{self.label}: {self.dates}, {self.site_hours:,} {self.hours_unit}"


def assert_matches_printed(
    *, name: str, recomputed: float, printed: float, decimals: int = REPORT_PRINT_DECIMALS
) -> None:
    """Stop unless a recomputed value rounds to the number a report printed.

    A chart draws numbers it recomputes from `losses.parquet`, the report prints the page's
    numbers, and the two must be the same number: a difference means the chart was drawn from
    other losses than the report, and the page and the chart would disagree.

    Args:
        name: The product or arm the value belongs to, for the error message.
        recomputed: The value recomputed from the saved losses.
        printed: The value the report prints.
        decimals: The decimal places the report prints it at.

    Raises:
        ValueError: If the recomputed value, rounded to `decimals` places, differs from
            `printed`.
    """
    if round(recomputed, decimals) != printed:
        msg = f"{name}: bootstrapped {recomputed:.{decimals}f} but the report says {printed}"
        raise ValueError(msg)


def _rows_at_setting(*, losses: pl.DataFrame, setting: str) -> pl.DataFrame:
    """Restrict saved losses to one hyperparameter setting.

    The saved `losses.parquet` files reuse an arm's name across their `pooled` and `sensitivity`
    settings, and the bootstraps join on `(site, time, seed)` alone, so an unfiltered frame is
    silently cross-joined between the two settings.

    Args:
        losses: A `losses.parquet`, with a `setting` column.
        setting: The setting to keep, such as `pooled`.

    Returns:
        The rows of that setting.

    Raises:
        ValueError: If no row has that setting.
    """
    kept = losses.filter(pl.col("setting") == setting)
    if kept.is_empty():
        held = losses["setting"].unique().to_list()
        msg = f"no row has setting {setting!r}; the losses hold {held}"
        raise ValueError(msg)
    return kept


def _check_rows(*, arm: str, n_rows: int, site_hours: int) -> None:
    """Raise unless one arm's bootstrap rests on exactly its block's site-hours."""
    if n_rows != site_hours:
        msg = f"arm {arm!r} has {n_rows:,} rows per seed but the block holds {site_hours:,}"
        raise ValueError(msg)


def block_leaderboard_rows(
    *,
    losses: pl.DataFrame,
    arms: Sequence[BlockArm],
    setting: str,
    site_hours: int,
    metric: str,
    printed: dict[str, float] | None = None,
    decimals: int = REPORT_PRINT_DECIMALS,
) -> pl.DataFrame:
    """Compute each arm's own mean absolute error and 95% interval on one row set.

    Filters to `setting` before any bootstrap, and checks that each arm's bootstrap rests on
    exactly `site_hours` rows. The rows come back best first.

    Args:
        losses: A `losses.parquet`.
        arms: The arms to score.
        setting: The hyperparameter setting to score, such as `pooled`.
        site_hours: The row set's number of site-hours.
        metric: The loss column to average.
        printed: Each arm's mean absolute error as the row set's report prints it, to check the
            recomputed values against; `None` skips the check.
        decimals: The decimal places `printed` is given at.

    Returns:
        One row per arm with `arm`, `label`, `family`, `reference`, `planned`, `value`, `lower_95`
        and `upper_95`, in percentage points of capacity.

    Raises:
        ValueError: If the setting is absent, an arm's bootstrap does not rest on `site_hours`
            rows, or a recomputed value differs from its printed value.
    """
    at_setting = _rows_at_setting(losses=losses, setting=setting)
    records = []
    for block_arm in arms:
        interval = bootstrap_absolute(losses=at_setting, arm=block_arm.arm, metric=metric)
        _check_rows(arm=block_arm.arm, n_rows=interval["n_rows"], site_hours=site_hours)
        value = interval["value"] * PERCENTAGE_POINTS
        if printed is not None:
            assert_matches_printed(
                name=block_arm.arm,
                recomputed=value,
                printed=printed[block_arm.arm],
                decimals=decimals,
            )
        records.append(
            {
                "arm": block_arm.arm,
                "label": block_arm.label,
                "family": block_arm.family,
                "reference": block_arm.reference,
                "planned": block_arm.planned,
                "value": value,
                "lower_95": interval["lower_95"] * PERCENTAGE_POINTS,
                "upper_95": interval["upper_95"] * PERCENTAGE_POINTS,
            }
        )
    return pl.DataFrame(records).sort("value")


def block_contrast_rows(
    *,
    losses: pl.DataFrame,
    arms: Sequence[BlockArm],
    reference_arm: str,
    setting: str,
    site_hours: int,
    metric: str,
) -> pl.DataFrame:
    """Compute each arm's mean absolute error minus a reference arm's, with its 95% interval.

    Filters to `setting` before any bootstrap, and checks that each contrast rests on exactly
    `site_hours` rows. The rows keep the order of `arms`.

    Args:
        losses: A `losses.parquet`.
        arms: The arms to contrast with the reference arm. The reference arm itself is not
            listed: the zero rule stands for it.
        reference_arm: The arm every contrast is taken against, such as `era5_global`.
        setting: The hyperparameter setting to score, such as `pooled`.
        site_hours: The row set's number of site-hours.
        metric: The loss column to difference.

    Returns:
        One row per arm with `arm`, `label`, `family`, `reference`, `planned`, `difference`,
        `lower_95` and `upper_95`, in percentage points of capacity.

    Raises:
        ValueError: If the setting is absent or a contrast does not rest on `site_hours` rows.
    """
    at_setting = _rows_at_setting(losses=losses, setting=setting)
    records = []
    for block_arm in arms:
        interval = bootstrap_difference(
            losses=at_setting, treatment=block_arm.arm, reference=reference_arm, metric=metric
        )
        _check_rows(arm=block_arm.arm, n_rows=interval["n_rows"], site_hours=site_hours)
        records.append(
            {
                "arm": block_arm.arm,
                "label": block_arm.label,
                "family": block_arm.family,
                "reference": block_arm.reference,
                "planned": block_arm.planned,
                "difference": interval["difference"] * PERCENTAGE_POINTS,
                "lower_95": interval["lower_95"] * PERCENTAGE_POINTS,
                "upper_95": interval["upper_95"] * PERCENTAGE_POINTS,
            }
        )
    return pl.DataFrame(records)


class PlannedContrast(NamedTuple):
    """One planned contrast: the treatment arm's error minus the reference arm's."""

    treatment: BlockArm
    reference: BlockArm

    @property
    def label(self) -> str:
        """The row's label, naming the treatment first."""
        return f"{self.treatment.label} against {self.reference.label}"


def planned_contrast_rows(
    *,
    losses: pl.DataFrame,
    contrasts: Sequence[PlannedContrast],
    setting: str,
    site_hours: int,
    metric: str,
) -> pl.DataFrame:
    """Compute each planned contrast, treatment minus reference, with its 95% interval.

    Filters to `setting` before any bootstrap, and checks that each contrast rests on exactly
    `site_hours` rows. Every row is planned, and its colour is its treatment's family. The rows
    keep the order of `contrasts`.

    Args:
        losses: A `losses.parquet`.
        contrasts: The planned contrasts. Unlike `block_contrast_rows`, each names its own
            reference arm.
        setting: The hyperparameter setting to score, such as `pooled`.
        site_hours: The row set's number of site-hours.
        metric: The loss column to difference.

    Returns:
        One row per contrast with `arm` (the treatment), `reference_arm`, `label`, `family`,
        `reference` (always false), `planned` (always true), `difference`, `lower_95` and
        `upper_95`, in percentage points of capacity.

    Raises:
        ValueError: If the setting is absent or a contrast does not rest on `site_hours` rows.
    """
    at_setting = _rows_at_setting(losses=losses, setting=setting)
    records = []
    for contrast in contrasts:
        interval = bootstrap_difference(
            losses=at_setting,
            treatment=contrast.treatment.arm,
            reference=contrast.reference.arm,
            metric=metric,
        )
        _check_rows(arm=contrast.label, n_rows=interval["n_rows"], site_hours=site_hours)
        records.append(
            {
                "arm": contrast.treatment.arm,
                "reference_arm": contrast.reference.arm,
                "label": contrast.label,
                "family": contrast.treatment.family,
                "reference": False,
                "planned": True,
                "difference": interval["difference"] * PERCENTAGE_POINTS,
                "lower_95": interval["lower_95"] * PERCENTAGE_POINTS,
                "upper_95": interval["upper_95"] * PERCENTAGE_POINTS,
            }
        )
    return pl.DataFrame(
        records,
        schema={
            "arm": pl.String,
            "reference_arm": pl.String,
            "label": pl.String,
            "family": pl.String,
            "reference": pl.Boolean,
            "planned": pl.Boolean,
            "difference": pl.Float64,
            "lower_95": pl.Float64,
            "upper_95": pl.Float64,
        },
    )


def shared_domain(
    *,
    blocks: Sequence[RowSetBlock],
    include_zero: bool,
    step: float = _BLOCK_DOMAIN_STEP,
) -> tuple[float, float]:
    """Return one x range covering every block's intervals, rounded out to a step.

    Args:
        blocks: The blocks of a stacked figure. A block's `planned_rows` count as well as its
            `rows`.
        include_zero: Whether the range must contain zero, as a contrast chart's does.
        step: The multiple of percentage points the range is rounded outwards to.

    Returns:
        The lowest and highest of every lower bound, upper bound, estimate, and second-setting
        estimate over every block, each rounded outwards to a multiple of `step`. Estimates count
        because a mark is clipped at the plot's edge, so an estimate outside its own interval would
        otherwise vanish.
    """
    frames = [
        frame for block in blocks for frame in (block.rows, block.planned_rows) if frame is not None
    ]
    lows = [low for frame in frames for low in frame["lower_95"].to_list()]
    highs = [high for frame in frames for high in frame["upper_95"].to_list()]
    seconds = [
        value
        for frame in frames
        if "second_difference" in frame.columns
        for value in frame["second_difference"].drop_nulls().to_list()
    ]
    estimates = [
        value
        for frame in frames
        if "difference" in frame.columns
        for value in frame["difference"].drop_nulls().to_list()
    ]
    lows += seconds + estimates
    highs += seconds + estimates
    if include_zero:
        lows.append(0.0)
        highs.append(0.0)
    return (math.floor(min(lows) / step) * step, math.ceil(max(highs) / step) * step)


def planned_domain(*, block: RowSetBlock) -> tuple[float, float]:
    """Return the x range of a block's own planned-contrast panel, from its planned rows alone.

    The planned contrasts of one row set can be far narrower than the range every block shares,
    which would draw a 0.1-point interval under its own marker. The range holds zero, each
    estimate, and each second-setting marker, and rounds outwards to `_PLANNED_DOMAIN_STEP`. A
    range this narrow still cannot show a 0.1-point interval beside a 2.7-point one, so
    `interval_panel(value_labels=True)` prints each row's estimate and interval as text.

    Args:
        block: A block with `planned_rows`.

    Returns:
        The range of the block's planned rows, in percentage points.

    Raises:
        ValueError: If the block has no planned rows.
    """
    if block.planned_rows is None:
        msg = f"{block.label} has no planned rows"
        raise ValueError(msg)
    return shared_domain(
        blocks=[block._replace(rows=block.planned_rows, planned_rows=None)],
        include_zero=True,
        step=_PLANNED_DOMAIN_STEP,
    )


def _block_families(*, blocks: Sequence[RowSetBlock]) -> list[ProductFamily]:
    """List every family any block draws, in `FAMILY_COLOURS` order, for the figure's one key."""
    held = {
        family
        for block in blocks
        for frame in (block.rows, block.planned_rows)
        if frame is not None
        for family in frame["family"].to_list()
    }
    return [family for family in FAMILY_COLOURS if family in held]


def stacked_leaderboard(
    *,
    blocks: Sequence[RowSetBlock],
    number: int | str,
    title: str,
    subtitle: Sequence[str],
    reference_note: str = REFERENCE_ROW_NOTE,
) -> alt.VConcatChart:
    """Stack one leaderboard panel per row set, on one x range, under one caption.

    Each block is a `leaderboard_panel` titled with its row set, dates and site-hours. A block's
    reference rows are drawn hollow. Blocks are scored on different rows, so the panels are for
    reading each block's own ranking; the subtitle should say so.

    Args:
        blocks: The row-set blocks from top to bottom, each holding `block_leaderboard_rows`'s
            output.
        number: The figure's number on its page.
        title: The finding the figure shows.
        subtitle: Short lines for the caption; `reference_note` is added.
        reference_note: The caption line saying what the hollow reference rows are.

    Returns:
        The figure.
    """
    domain = shared_domain(blocks=blocks, include_zero=False)
    panels = [
        leaderboard_panel(
            rows=block.rows,
            x_domain=domain,
            x_title=ABSOLUTE_ERROR_X_TITLE if index == len(blocks) - 1 else "",
            panel_title=block.title,
            keys=index == 0,
            key_families=_block_families(blocks=blocks),
            row_step_px=_BLOCK_ROW_STEP_PX,
        )
        for index, block in enumerate(blocks)
    ]
    return figure(
        panels=panels,
        number=number,
        title=title,
        subtitle=[*subtitle, reference_note],
        figure_planning=None,
    )


PLANNED_CONTRAST_X_TITLE: Final[str] = (
    "Mean absolute error of the first product minus the second's (points of capacity)"
)
"""The x axis title of a block's lower panel of planned contrasts."""

SECOND_SETTING_NOTE: Final[str] = (
    "Hollow triangle: the same contrast at the second hyperparameter setting, shown for planned "
    "contrasts and for contrasts near the 5% line."
)
"""The subtitle line a stacked contrast chart adds to explain its second-setting markers."""


def stacked_contrasts(
    *,
    blocks: Sequence[RowSetBlock],
    number: int | str,
    title: str,
    subtitle: Sequence[str],
    reference_note: str = CONTRAST_REFERENCE_ROW_NOTE,
) -> alt.VConcatChart:
    """Stack, per row set, a panel of contrasts against ERA5 and a panel of planned contrasts.

    Each block is an `interval_panel` titled with its row set, dates and site-hours. A block's
    reference rows (CAMS) are drawn hollow, in the light shade of their family's colour. A block
    with `planned_rows` gets a second panel under the first, holding that row set's planned
    contrasts, each the first product's error minus the second's: a chart of differences from
    ERA5 cannot show whether two other products differ. A block with planned
    contrasts titles the x axis of both its panels, because the two measure different
    differences; a block without them titles its axis only if it is the last block. The contrast
    panels share one x range; each planned-contrast panel has its own, from `planned_domain`, so
    a narrow interval is not drawn under its marker.

    Args:
        blocks: The row-set blocks from top to bottom, each holding `block_contrast_rows`'s
            output, and optionally `planned_contrast_rows`'s.
        number: The figure's number on its page.
        title: The finding the figure shows.
        subtitle: Short lines for the caption; `reference_note` is added, and
            `SECOND_SETTING_NOTE` where any row has a second setting.
        reference_note: The caption line saying what the hollow reference row is.

    Returns:
        The figure.
    """
    domain = shared_domain(blocks=blocks, include_zero=True)
    frames = [
        frame for block in blocks for frame in (block.rows, block.planned_rows) if frame is not None
    ]
    figure_planning = planning(rows=frames)
    post_hoc = any(label.endswith(POST_HOC_SUFFIX) for frame in frames for label in frame["label"])
    conditions = ("Product", "Reference row")
    panels = []
    for index, block in enumerate(blocks):
        panels.append(
            interval_panel(
                rows=block.rows.with_columns(
                    condition=pl.when(pl.col("reference"))
                    .then(pl.lit(conditions[1]))
                    .otherwise(pl.lit(conditions[0]))
                ),
                x_domain=domain,
                x_title=(
                    _contrast_x_title(reference_name=block.reference_name)
                    if block.planned_rows is not None or index == len(blocks) - 1
                    else ""
                ),
                zero_label=f"same as {block.reference_name}",
                better_label=f"better than {block.reference_name}",
                conditions=conditions,
                panel_title=block.title,
                family_key=index == 0,
                key_families=_block_families(blocks=blocks),
                condition_key=False,
                figure_planning=figure_planning,
            )
        )
        if block.planned_rows is not None:
            panels.append(
                interval_panel(
                    rows=block.planned_rows,
                    x_domain=planned_domain(block=block),
                    x_title=PLANNED_CONTRAST_X_TITLE,
                    zero_label="same as the second product",
                    better_label="first product better",
                    value_labels=True,
                    panel_title=f"{block.label}: planned contrasts",
                    family_key=False,
                    condition_key=False,
                    figure_planning=figure_planning,
                )
            )
    has_second_setting = any(
        "second_difference" in frame.columns and frame["second_difference"].is_not_null().any()
        for frame in frames
    )
    notes = [reference_note, *([SECOND_SETTING_NOTE] if has_second_setting else [])]
    return figure(
        panels=panels,
        number=number,
        title=title,
        subtitle=[*subtitle, *notes],
        figure_planning=figure_planning,
        post_hoc=post_hoc,
    )
