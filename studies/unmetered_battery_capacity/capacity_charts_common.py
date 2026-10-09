"""Shared pieces of the unmetered-battery-capacity figure scripts.

Every figure reads a saved result: a table that `capacity_report.py` printed into `report.md`, or a
parquet file a rung script saved. Nothing is refitted. A table is parsed from the report by the
bold line that introduces it, so a figure and the page quote the same numbers.
"""

import re
from collections.abc import Sequence
from io import StringIO
from pathlib import Path
from typing import Any, Final, cast

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from capacity_inputs import OUTPUT_DIR
from studies.charts import CONTENT_WIDTH_PX, figure

ASSETS_DIR: Final[Path] = Path("docs/studies/assets")
ASSET_PREFIX: Final[str] = "unmetered_battery_capacity_"
REPORT_PATH: Final[Path] = OUTPUT_DIR / "report.md"
SECOND_SETTING_HEADING: Final[str] = "# The second setting"
PLOT_WIDTH_PX: Final[int] = CONTENT_WIDTH_PX - 110
PANEL_HEIGHT_PX: Final[int] = 130
PERCENT: Final[float] = 100.0
NOMINAL_FALSE_ALARM: Final[float] = 0.05
IN_FAMILY_COLOUR: Final[str] = ocf.DATA_BLUE
RANK_RULE_COLOUR: Final[str] = ocf.BRAND_ORANGE
NOISY_PRICE_COLOUR: Final[str] = ocf.DATA_PURPLE
REAL_COLOUR: Final[str] = ocf.DATA_GREEN
REFERENCE_COLOUR: Final[str] = ocf.BLACK_1
SHAPES: Final[tuple[str, ...]] = ("circle", "square", "diamond", "triangle-up")


def report_table(*, marker: str, occurrence: int = 0, second_setting: bool = False) -> pl.DataFrame:
    """Return the pipe-separated table that follows a marker line in `report.md`.

    Args:
        marker: The start of the line that introduces the table (a bold heading or a sentence).
        occurrence: Which line starting with the marker to use, counting from 0.
        second_setting: Read the part of the report after the second setting's heading instead of
            the part before it.

    Returns:
        The table, with numbers parsed.

    Raises:
        ValueError: If the marker or the table after it is missing.
    """
    text = REPORT_PATH.read_text()
    standard, _, second = text.partition(SECOND_SETTING_HEADING)
    lines = (second if second_setting else standard).splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(marker)]
    if len(starts) <= occurrence:
        raise ValueError(f"Marker {marker!r} occurs {len(starts)} times in the report")
    rows: list[str] = []
    for line in lines[starts[occurrence] + 1 :]:
        if "|" in line:
            rows.append(line)
        elif rows:
            break
    if not rows:
        raise ValueError(f"No table follows {marker!r}")
    return pl.read_csv(StringIO("\n".join(rows)), separator="|", infer_schema_length=None)


def save(*, chart: alt.TopLevelMixin, name: str) -> Path:
    """Write a figure as an SVG under `docs/studies/assets/` and a PNG preview in the results.

    Args:
        chart: The figure.
        name: The figure's file stem, without the study prefix.

    Returns:
        The SVG's path.
    """
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    previews = OUTPUT_DIR / "previews"
    previews.mkdir(exist_ok=True)
    target = ASSETS_DIR / f"{ASSET_PREFIX}{name}.svg"
    chart.save(target)
    chart.save(previews / f"{name}.png", scale_factor=1.3)
    return target


def line_panel(
    *,
    frame: pl.DataFrame,
    x: str,
    y: str,
    group: str,
    domain: list[str],
    colours: Sequence[str],
    x_title: str,
    y_title: str,
    title: str,
    log_x: bool = False,
    x_values: list[float] | None = None,
    y_domain: tuple[float, float] | None = None,
    interval: tuple[str, str] | None = None,
    rule: float | None = None,
    log_y: bool = False,
) -> alt.LayerChart:
    """Draw one panel of lines, one per group, with a mark on every point.

    Args:
        frame: One row per group and x value.
        x: The x column (quantitative).
        y: The y column.
        group: The column that names the line.
        domain: The group names, in legend order.
        colours: One colour per group.
        x_title: The x axis title.
        y_title: The y axis title.
        title: The panel title.
        log_x: Whether x is on a base-2 log scale.
        x_values: The x tick values, if fixed.
        y_domain: The y axis limits, if fixed.
        interval: The columns holding the lower and upper end of an interval, drawn as a band.
        rule: A horizontal reference line's y value, if any.
        log_y: Whether y is on a log scale.

    Returns:
        The panel.
    """
    x_scale = alt.Scale(type="log", base=2) if log_x else alt.Scale(zero=False)
    y_scale = alt.Scale(domain=list(y_domain)) if y_domain else alt.Scale(zero=True)
    if log_y:
        y_scale = alt.Scale(type="log")
    colour = alt.Color(
        f"{group}:N",
        scale=alt.Scale(domain=domain, range=colours),
        legend=alt.Legend(title=None, columns=2, symbolLimit=0, labelLimit=330),
    )
    shape = alt.Shape(
        f"{group}:N",
        scale=alt.Scale(domain=domain, range=list(SHAPES[: len(domain)])),
        legend=None,
    )
    x_axis = alt.X(
        f"{x}:Q",
        scale=x_scale,
        title=x_title,
        axis=alt.Axis(values=x_values) if x_values else alt.Axis(),
    )
    base = alt.Chart(frame)
    layers: list[alt.Chart] = []
    if interval is not None:
        low, high = interval
        layers.append(
            base.mark_area(opacity=0.12, aria=False).encode(  # ty: ignore[unresolved-attribute]
                x=x_axis,
                y=alt.Y(f"{low}:Q", scale=y_scale, title=y_title),
                y2=f"{high}:Q",
                color=colour,
                detail=f"{group}:N",
            )
        )
    layers.append(
        base.mark_line(strokeWidth=1.8, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=x_axis, y=alt.Y(f"{y}:Q", scale=y_scale, title=y_title), color=colour
        )
    )
    layers.append(
        base.mark_point(filled=True, size=55, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=x_axis,
            y=alt.Y(f"{y}:Q", scale=y_scale, title=y_title),
            color=colour,
            shape=shape,
        )
    )
    if rule is not None:
        layers.append(
            alt.Chart(pl.DataFrame({"rule": [rule]}))
            .mark_rule(color=REFERENCE_COLOUR, strokeDash=[4, 3], aria=False)
            .encode(y="rule:Q")  # ty: ignore[unresolved-attribute]
        )
    return cast(
        alt.LayerChart,
        alt.layer(*layers).properties(
            width=PLOT_WIDTH_PX,
            height=PANEL_HEIGHT_PX,
            title=alt.TitleParams(title, anchor="start"),
        ),
    )


def report_number(*, pattern: str) -> float:
    """Return the first number a regular expression captures from `report.md`.

    Args:
        pattern: A regular expression with one group that captures the number.

    Returns:
        The number.

    Raises:
        ValueError: If the pattern does not match.
    """
    match = re.search(pattern, REPORT_PATH.read_text())
    if match is None:
        raise ValueError(f"Pattern {pattern!r} does not match the report")
    return float(match.group(1))


def write_notes(*, name: str, notes: list[str]) -> Path:
    """Write the numbers a figure script derived, so the page quotes them from a saved report.

    Args:
        name: The file stem, written as `report_figures_<name>.md` beside the report.
        notes: One line per derived number.

    Returns:
        The path written.
    """
    target = OUTPUT_DIR / f"report_figures_{name}.md"
    target.write_text(
        "# Numbers derived by a figure script\n\n" + "\n".join(f"- {n}" for n in notes) + "\n"
    )
    return target


def draw_figure(*, panels: Sequence[alt.TopLevelMixin], **kwargs: Any) -> alt.VConcatChart:
    """Stack panels under a caption with `studies.charts.figure`.

    Altair types a faceted panel more widely than `figure` accepts, so the panels are passed
    through untyped.

    Args:
        panels: The panels, one above the other.
        **kwargs: The caption's arguments, as `studies.charts.figure` takes them.

    Returns:
        The figure.
    """
    return figure(panels=cast(Any, panels), **kwargs)
