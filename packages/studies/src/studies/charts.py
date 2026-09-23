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

import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, NamedTuple

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl

ProductFamily = Literal["satellite", "reanalysis", "weather model"]
"""The three kinds of weather product the study pages compare."""

FAMILY_COLOURS: Final[dict[ProductFamily, str]] = {
    "satellite": ocf.ORANGE_RED,
    "reanalysis": ocf.SKY_BLUE,
    "weather model": ocf.BLUE,
}
"""Each family's colour, from the main data colours OCF's brand guidelines allow in published work.

Data Sky has only 2.0:1 contrast with the background, which is acceptable because every mark is
labelled on the axis.
"""

FAMILY_COLOURS_LIGHT: Final[dict[ProductFamily, str]] = {
    "satellite": ocf.ORANGE_RED_LIGHT,
    "reanalysis": ocf.SKY_BLUE_LIGHT,
    "weather model": ocf.BLUE_LIGHT,
}
"""The light shade of each family's colour, for a second condition of the same row.

A light shade sits too close to its parent to carry a distinction alone, so `interval_panel`
always draws it with a hollow point of a second shape.
"""

Panel = alt.LayerChart | alt.HConcatChart | alt.VConcatChart
"""A chart `figure` can set under its caption."""

BetterDirectionType = Literal["negative", "positive"]
"""Which sign of difference is the better one."""

NAMED_SUFFIX: Final[str] = " (named before the run)"
"""Ends the label of a row whose contrast was named before the run, which `interval_panel` bolds."""

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

_INTERVAL: Final[re.Pattern[str]] = re.compile(r"^\[(\S+), (\S+)\]$")
_FOLDS: Final[re.Pattern[str]] = re.compile(r"^(\d+) of (\d+)$")
_ROW_STEP_PX: Final[int] = 22
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
    # two collide only if the better-direction label crosses zero.
    zero_text = (
        alt.Chart(anchor)
        .mark_text(
            align="left" if better_direction == "negative" else "right",
            dx=4 if better_direction == "negative" else -4,
            baseline="bottom",
            dy=-4,
            color=ocf.BLACK_1,
        )
        .encode(x="x:Q", y=alt.value(0), text=alt.value(zero_label))  # ty: ignore[unresolved-attribute]
    )
    edge = x_domain[0] if better_direction == "negative" else x_domain[1]
    text = f"← {better_label}" if better_direction == "negative" else f"{better_label} →"
    align = "left" if better_direction == "negative" else "right"
    better = (
        alt.Chart(pl.DataFrame({"x": [edge]}))
        .mark_text(align=align, baseline="bottom", dy=-4, color=ocf.BLACK_1, fontWeight="bold")
        .encode(x="x:Q", y=alt.value(0), text=alt.value(text))  # ty: ignore[unresolved-attribute]
    )
    return [rule, zero_text, better] if labelled else [rule]


def interval_panel(
    *,
    rows: pl.DataFrame,
    x_domain: tuple[float, float],
    x_title: str,
    zero_label: str,
    better_label: str,
    better_direction: BetterDirectionType = "negative",
    conditions: Sequence[str] = (),
    condition_title: str = "",
    panel_title: str = "",
    reference_labels: bool = True,
    width: int = 340,
) -> alt.LayerChart | alt.HConcatChart:
    """Draw one panel of dots and 95% interval lines beside a labelled zero rule.

    Each row's colour is its family's. A row with a `condition` other than the first in
    `conditions` is drawn in the light shade of that colour, with a hollow point of a second
    shape, and where a label carries more than one condition the rows are offset vertically. A
    label ending in `NAMED_SUFFIX` is set bold. The colour scale's domain is every family, so a
    family keeps its colour whichever families a panel holds, and the legend lists only the
    families present.

    Args:
        rows: One row per mark, with `label`, `family` (a `ProductFamily`), `difference`,
            `lower_95` and `upper_95`, and `condition` if `conditions` is given. Rows are drawn
            top to bottom in the order given.
        x_domain: The x axis's range, set explicitly so two panels can share it.
        x_title: The x axis's title, naming the quantity and its unit.
        zero_label: What a difference of zero means, such as `same as ERA5`.
        better_label: What the better direction means, such as `better than ERA5`.
        better_direction: Which sign of difference is the better one.
        conditions: The values of `condition`, in legend order; the first is drawn filled.
        condition_title: The legend title for `condition`.
        panel_title: A title above this panel alone.
        reference_labels: Whether to label the zero rule and the better direction, which a panel
            stacked under another that already carries them can leave out.
        width: The plot's width in pixels.

    Returns:
        The panel, with the key to the conditions beside it if `conditions` is given.
    """
    shade = pl.col("family")
    if conditions:
        shade = (
            pl.when(pl.col("condition") == conditions[0])
            .then(pl.col("family"))
            .otherwise(pl.col("family") + ", light")
        )
    data = rows.with_columns(pl.col("difference", "lower_95", "upper_95").round(3), shade=shade)
    labels = list(dict.fromkeys(data["label"].to_list()))
    families = [family for family in FAMILY_COLOURS if family in set(data["family"].to_list())]
    encodings: dict[str, object] = {
        "y": alt.Y(
            "label:N",
            sort=labels,
            title=None,
            axis=alt.Axis(
                labelLimit=420,
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
            scale=_shade_scale(),
            legend=alt.Legend(title="Product type", values=families),
        ),
    }
    if conditions and data["label"].is_duplicated().any():
        encodings["yOffset"] = alt.YOffset("condition:N", sort=list(conditions))
    if conditions:
        encodings["shape"] = alt.Shape(
            "condition:N",
            scale=alt.Scale(
                domain=list(conditions), range=list(CONDITION_SHAPES[: len(conditions)])
            ),
            legend=None,
        )
    x_scale = alt.Scale(domain=list(x_domain), nice=False, zero=False)
    x_axis = alt.Axis(values=ticks(x_domain=x_domain))
    interval = (
        alt.Chart(data)
        .mark_rule(strokeWidth=2, clip=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lower_95:Q", scale=x_scale, title=x_title, axis=x_axis),
            x2="upper_95:Q",
            **{key: value for key, value in encodings.items() if key != "shape"},
        )
    )
    first = pl.col("condition") == conditions[0] if conditions else pl.lit(value=True)
    x = alt.X("difference:Q", scale=x_scale, title=x_title, axis=x_axis)
    tooltip = [
        alt.Tooltip("label:N", title="Row"),
        alt.Tooltip("difference:Q", title="Estimate"),
        alt.Tooltip("lower_95:Q", title="Lower 95%"),
        alt.Tooltip("upper_95:Q", title="Upper 95%"),
    ]
    points = [
        alt.Chart(data.filter(first))
        .mark_point(filled=True, size=_POINT_SIZE, opacity=1, clip=True)
        .encode(x=x, tooltip=tooltip, **encodings),  # ty: ignore[unresolved-attribute]
        alt.Chart(data.filter(~first))
        .mark_point(filled=False, size=_POINT_SIZE, strokeWidth=2, opacity=1, clip=True)
        .encode(x=x, tooltip=tooltip, **encodings),  # ty: ignore[unresolved-attribute]
    ]
    reference = _reference_layers(
        x_domain=x_domain,
        zero_label=zero_label,
        better_label=better_label,
        better_direction=better_direction,
        labelled=reference_labels,
    )
    panel = alt.LayerChart(
        layer=[*reference, interval, *points],
        width=width,
        height=alt.Step(_ROW_STEP_PX),
        title=alt.TitleParams(panel_title, anchor="start", frame="group", fontSize=_PANEL_TITLE_PX),
    )
    if not conditions:
        return panel
    return alt.hconcat(panel, _condition_key(conditions=conditions, title=condition_title))


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


def _condition_key(*, conditions: Sequence[str], title: str) -> alt.LayerChart:
    """Draw the key to the conditions: the first filled, the others hollow, each its own shape.

    A Vega-Lite shape legend draws every symbol alike, so it cannot show which condition is hollow.

    Args:
        conditions: The conditions, in the order `interval_panel` was given them.
        title: The key's title.

    Returns:
        A small chart of one symbol and label per condition.
    """
    data = pl.DataFrame(
        {
            "condition": list(conditions),
            "shape": list(CONDITION_SHAPES[: len(conditions)]),
            "filled": [index == 0 for index in range(len(conditions))],
        }
    )
    y = alt.Y("condition:N", sort=list(conditions), axis=None)
    layers = [
        alt.Chart(data.filter(pl.col("filled") == filled))
        .mark_point(size=_POINT_SIZE, strokeWidth=2, color=ocf.BLACK_1, filled=filled, opacity=1)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.value(6), y=y, shape=alt.Shape("shape:N", scale=None)
        )
        for filled in (True, False)
    ]
    label = (
        alt.Chart(data)
        .mark_text(align="left", dx=16, color=ocf.BLACK_1)
        .encode(x=alt.value(0), y=y, text="condition:N")  # ty: ignore[unresolved-attribute]
    )
    return alt.LayerChart(
        layer=[*layers, label],
        width=10,
        height=alt.Step(_ROW_STEP_PX),
        title=alt.TitleParams(title, anchor="start", fontSize=_KEY_TITLE_PX),
    )


def figure(
    *,
    panels: Sequence[Panel],
    number: int,
    title: str,
    subtitle: Sequence[str],
    width: int,
    direction: Literal["horizontal", "vertical"] = "horizontal",
) -> alt.VConcatChart:
    """Set panels under the figure caption OCF's slide template uses.

    The template sets "Figure N:" and the title above the chart, under a thin rule in the brand's
    warm grey.

    Args:
        panels: The panels, drawn side by side or one above the other.
        number: The figure's number on its page.
        title: The finding the figure shows.
        subtitle: Lines naming the quantity, its unit, its scope, and what a dot and a line mean.
        width: The figure's width in pixels, which the rule above the caption spans.
        direction: Whether the panels sit side by side or one above the other.

    Returns:
        The figure.
    """
    body = (
        alt.hconcat(*panels, spacing=40)
        if direction == "horizontal"
        else alt.vconcat(*panels, spacing=24)
    )
    caption = alt.TitleParams(
        f"Figure {number}: {title}",
        subtitle=list(subtitle),
        anchor="start",
        offset=14,
        subtitleColor=ocf.BLACK_1,
        subtitlePadding=6,
    )
    rule = (
        alt.Chart(alt.Data(values=[{}]))
        .mark_rule(color=ocf.GREY_3, strokeWidth=1)
        .encode(x=alt.value(0), x2=alt.value(width), y=alt.value(0))  # ty: ignore[unresolved-attribute]
        .properties(width=width, height=1)
    )
    return (
        alt.vconcat(rule, body.properties(title=caption), spacing=10)
        .resolve_scale(color="shared", shape="shared")
        .configure_view(stroke=None)
    )
