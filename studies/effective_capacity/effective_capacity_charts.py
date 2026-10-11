"""Draw the effective-capacity page's figures from the study's `report.md` and saved tables.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1143>.
**Every number a chart shares with the page is parsed from `report.md`**, so a chart cannot
disagree with the page, and a parse that finds a line zero times or twice raises. Two figures draw
traces that the report script saves beside the report (`tables/example_event.parquet` and
`tables/kf_seasonal_shape.parquet`).

**No NGED metered generator's identifier, name, coordinates, or calendar date appears.** Plants are
unlabelled, a planted step is drawn in days from its first day, and no figure names the plant that
is the subject of the low-output or active-network-management comparisons.

The figures, in the order of the page:

1. The headline: each estimator's change in day-ahead error against the baseline, with intervals.
2. One planted step, and what each estimator reports across it.
3. How closely each estimator recovers the planted losses.
4. How often each estimator moves on months in which a plant's capacity did not change.
5. The raw Kalman filter level through the calendar year.
6. The Kalman filter and the hold-at-static detector in December to February and in other months.
7. The same two arms on the three low-output plant-months and on every other row.
8. The positive control: an oracle and the Kalman filter on a planted 30% step.
9. How often the hold-at-static detector left the static capacity, by month.
10. The explainability rubric.
11. The hindsight reference against the causal estimate.

**Run it with `--text-only` first, review the figure text, then render.** A render writes one SVG
per figure under the study folder's `figures/` and publishes an optimised copy (svgo) as
`docs/studies/assets/effective_capacity_figure_<n>.svg`, then writes a 2x PNG beside the raw SVG
for a look before commit.

Run it with `uv run python studies/effective_capacity/effective_capacity_charts.py`.
"""

import argparse
import logging
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from capacity_paths import claim_root
from contracts.settings import PROJECT_ROOT
from studies.charts import CONTENT_WIDTH_PX, ticks, wrapped
from studies.charts import figure as captioned_figure
from studies.report_text import (
    BOOT_PATTERN,
    INTERVAL_PATTERN,
    NUMBER_PATTERN,
    T_PATTERN,
    Estimate,
    ReportText,
    estimate_in,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("effective_capacity_charts")

DOCS_ASSETS_DIR: Final[Path] = PROJECT_ROOT / "docs" / "studies" / "assets"
"""Where a render publishes each figure, optimised."""

ASSET_PREFIX: Final[str] = "effective_capacity_figure_"
"""The start of each published file's name."""

SVGO_COMMAND: Final[tuple[str, ...]] = (
    "npx",
    "svgo@4",
    "--multipass",
    "--precision=1",
    "--final-newline",
)
"""The optimisation `CLAUDE.md` asks for before a chart image is committed."""

STAGE_B_HEADING: Final[str] = (
    "Stage B: absolute error of every arm (points of the static 99th-percentile power)"
)
"""The report heading above the table of every arm's absolute error."""

POINTS: Final[float] = 100.0
"""Shares of capacity are drawn as percentage points."""

LABEL_WIDTH_PX: Final[int] = 250
"""The room the row labels of a dot-and-interval chart take to the left of the plot."""

PLOT_WIDTH_PX: Final[int] = CONTENT_WIDTH_PX - LABEL_WIDTH_PX
"""The width of a dot-and-interval chart's plot area."""

ROW_STEP_PX: Final[int] = 26
"""The height of each row of a dot-and-interval chart."""

FAMILY_NAMES: Final[dict[str, str]] = {
    "reference": "reference",
    "cheap": "simple formula",
    "wacky": "hidden Markov",
    "convex": "convex programme",
    "pytorch": "differentiable physics",
    "post hoc": "post hoc",
}
"""How a figure names each estimator family."""

FAMILY_COLOURS: Final[dict[str, str]] = {
    "reference": ocf.BLACK_1,
    "simple formula": ocf.DATA_BLUE,
    "hidden Markov": ocf.DATA_PURPLE,
    "convex programme": ocf.DATA_GREEN,
    "differentiable physics": ocf.DATA_SKY,
    "post hoc": ocf.BRAND_ORANGE,
}
"""The colour of each family, from the main data colours OCF's brand guidelines allow."""

POST_HOC_ESTIMATORS: Final[tuple[str, ...]] = ("DSKF", "HYST")
"""The estimators designed after the main results."""

ESTIMATOR_TITLES: Final[dict[str, str]] = {
    "S0": "S0: the static capacity",
    "EM": "EM: running maximum",
    "R30": "R30: 30-day 99th percentile",
    "CSQ": "CSQ: clear-sky quantile",
    "NR": "NR: neighbour ratio",
    "PR": "PR: performance ratio",
    "KF": "KF: Kalman filter",
    "HM": "HM: hidden Markov",
    "CX1": "CX1: convex, DC trace",
    "CX2": "CX2: convex, DC and AC",
    "DP1": "DP1: differentiable physics",
    "HYST": "HYST: hold at static",
    "DSKF": "DSKF: deseasonalised KF",
}
"""The name drawn above each estimator's panel."""


class FigureText:
    """Collects the text of every figure for the text reviews."""

    def __init__(self) -> None:
        """Start with no figures."""
        self.entries: dict[int, list[str]] = {}

    def add(self, *, number: int, plots: str, lines: Sequence[str]) -> None:
        """Record one figure's text.

        Args:
            number: The figure's number on the page.
            plots: What the figure plots, for the reviewer.
            lines: The title, subtitle lines, axis titles, legend entries, and annotations.
        """
        self.entries[number] = [f"[plots] {plots}", *lines]

    def write(self, *, path: Path) -> None:
        """Write every figure's text to one file.

        Args:
            path: The file to write.
        """
        blocks = [
            f"Figure {number}\n" + "\n".join(lines)
            for number, lines in sorted(self.entries.items())
        ]
        path.write_text("\n\n".join(blocks) + "\n")


def claim(*, holds: bool, what: str) -> None:
    """Raise if a claim a figure's title makes does not hold in the report it draws.

    Args:
        holds: Whether the claim holds.
        what: The claim, for the message.

    Raises:
        ValueError: If the claim does not hold.
    """
    if not holds:
        msg = f"the figure's title claims that {what}, and the report says otherwise"
        raise ValueError(msg)


def signed(value: float, *, places: int = 2) -> str:
    """Format a number with an explicit sign, using a true minus."""
    text = f"{value:+.{places}f}"
    return text.replace("-", "−")


def estimate_text(estimate: Estimate) -> str:
    """Format an estimate and its interval, such as `+0.255 [+0.004, +0.563]`."""
    point, low, high = (signed(value, places=3) for value in estimate)
    return f"{point} [{low}, {high}]"


def key_row(*, entries: Sequence[tuple[str, str, str, bool]]) -> alt.LayerChart:
    """Draw a key in one row above a panel: one symbol and label per entry, in equal slots.

    Args:
        entries: Each entry's label, colour, point shape, and whether the point is filled.

    Returns:
        A one-row chart as wide as the plot beneath it.
    """
    width = PLOT_WIDTH_PX
    slot = width // len(entries)
    chars_per_line = max(10, (slot - 24) // 7)
    wrapped_labels = [wrapped(text=label, width=chars_per_line) for label, *_ in entries]
    lines = max(len(label_lines) for label_lines in wrapped_labels)
    data = pl.DataFrame(
        {
            "label": ["\n".join(label_lines) for label_lines in wrapped_labels],
            "colour": [colour for _, colour, _, _ in entries],
            "shape": [shape for _, _, shape, _ in entries],
            "filled": [filled for _, _, _, filled in entries],
            "x": [6 + index * slot for index in range(len(entries))],
        }
    )
    points = [
        alt.Chart(data.filter(pl.col("filled") == is_filled))
        .mark_point(size=70, strokeWidth=2, filled=is_filled, opacity=1, aria=False)
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
            aria=False,
        )
        .encode(x=alt.X("x:Q", scale=None), y=alt.value(8), text="label:N")  # ty: ignore[unresolved-attribute]
    )
    return alt.LayerChart(
        layer=[*points, text],
        width=width,
        height=16 if lines == 1 else 16 + 13 * (lines - 1),
    )


def dot_interval_panel(
    *,
    rows: pl.DataFrame,
    x_domain: tuple[float, float],
    text_room: float,
    x_title: str,
    zero_label: str | None,
    colour_scale: tuple[Sequence[str], Sequence[str]],
    colour_title: str | None,
    references: Sequence[tuple[float, str | None]] = (),
    tick_values: Sequence[float] | None = None,
) -> alt.LayerChart:
    """Draw rows of dots with interval lines against a labelled zero rule.

    Args:
        rows: One row per mark, with `label`, `point`, nullable `low` and `high`, `colour_key`,
            a Boolean `hollow`, `text` (printed at the right of the plot), and `shape`.
        x_domain: The range of the data on the x axis.
        text_room: How much further right than the domain the printed text may reach, in axis
            units.
        x_title: The axis title.
        zero_label: What zero means, drawn on the rule, or `None` for no rule.
        colour_scale: The colour keys and their colours.
        colour_title: The legend title, or `None` for no Vega legend.
        references: Extra vertical dashed rules and their labels.
        tick_values: The x axis's tick values, by default round values inside `x_domain`.

    Returns:
        The layered chart.
    """
    low, high = x_domain
    full = (low, high + text_room)
    order = rows["label"].to_list()
    y = alt.Y(
        "label:N",
        sort=order,
        title=None,
        axis=alt.Axis(labelLimit=LABEL_WIDTH_PX - 10, ticks=False, domain=False),
    )
    x_axis = alt.Axis(
        domain=False,
        values=list(tick_values) if tick_values is not None else ticks(x_domain=x_domain),
        grid=True,
        title=x_title,
        titleLimit=0,
    )
    scale = alt.Scale(domain=list(full), nice=False)
    colour = alt.Color(
        "colour_key:N",
        scale=alt.Scale(domain=list(colour_scale[0]), range=list(colour_scale[1])),
        legend=None
        if colour_title is None
        else alt.Legend(title=colour_title, titleLimit=0, labelLimit=0, columns=3),
    )
    layers: list[alt.Chart] = []
    rules = [(0.0, zero_label, ocf.BLACK_1, [])] if zero_label is not None else []
    rules += [(position, label, ocf.BLACK_1, [4, 3]) for position, label in references]
    for position, label, rule_colour, dash in rules:
        data = pl.DataFrame({"x": [position], "label": [label]})
        layers.append(
            alt.Chart(data)
            .mark_rule(color=rule_colour, strokeDash=dash, aria=False)
            .encode(x=alt.X("x:Q", scale=scale, axis=x_axis))  # ty: ignore[unresolved-attribute]
        )
        if label is None:
            continue
        layers.append(
            alt.Chart(data)
            .mark_text(align="left", baseline="bottom", dx=3, dy=-2, fontSize=10, color=ocf.BLACK_1)
            .encode(x=alt.X("x:Q", scale=scale), y=alt.value(0), text="label:N")  # ty: ignore[unresolved-attribute]
        )
    intervals = rows.filter(pl.col("low").is_not_null())
    layers.append(
        alt.Chart(intervals)
        .mark_rule(strokeWidth=2.5, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=y, x=alt.X("low:Q", scale=scale, axis=x_axis), x2="high:Q", color=colour
        )
    )
    for hollow in (False, True):
        part = rows.filter(pl.col("hollow") == hollow)
        if part.height:
            layers.append(
                alt.Chart(part)
                .mark_point(size=90, strokeWidth=2, filled=not hollow, opacity=1, aria=False)
                .encode(  # ty: ignore[unresolved-attribute]
                    y=y,
                    x=alt.X("point:Q", scale=scale, axis=x_axis),
                    color=colour,
                    shape=alt.Shape("shape:N", scale=None),
                )
            )
    layers.append(
        alt.Chart(rows.with_columns(text_x=pl.lit(high)))
        .mark_text(align="left", dx=6, fontSize=11, color=ocf.BLACK_1, aria=False)
        .encode(y=y, x=alt.X("text_x:Q", scale=scale, axis=x_axis), text="text:N")  # ty: ignore[unresolved-attribute]
    )
    return alt.LayerChart(layer=layers, width=PLOT_WIDTH_PX, height=ROW_STEP_PX * rows.height)


def make_rows(
    *,
    labels: Sequence[str],
    estimates: Sequence[Estimate | None],
    colour_keys: Sequence[str],
    hollow: Sequence[bool] | None = None,
    texts: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Build the rows `dot_interval_panel` draws.

    Args:
        labels: Each row's label.
        estimates: Each row's estimate, or `None` for a row of a point without an interval.
        colour_keys: Each row's colour key.
        hollow: Whether each row's dot is hollow (drawn as a triangle).
        texts: Each row's printed text, by default the estimate and its interval.

    Returns:
        The rows.
    """
    flags = list(hollow) if hollow is not None else [False] * len(labels)
    return pl.DataFrame(
        {
            "label": list(labels),
            "point": [e.point for e in estimates],  # ty: ignore[unresolved-attribute]
            "low": [e.low for e in estimates],  # ty: ignore[unresolved-attribute]
            "high": [e.high for e in estimates],  # ty: ignore[unresolved-attribute]
            "colour_key": list(colour_keys),
            "hollow": flags,
            "shape": ["triangle-up" if flag else "circle" for flag in flags],
            "text": list(texts) if texts is not None else [estimate_text(e) for e in estimates],  # ty: ignore[invalid-argument-type]
        }
    )


def stage_a_table(*, report: ReportText) -> pl.DataFrame:
    """Return Stage A's per-estimator results, with the post hoc estimators added.

    Args:
        report: The report.

    Returns:
        One row per estimator with `family`, `error_points`, `false_movement`, and `delay`.
    """
    main = report.table(heading="Stage A: recovery of planted losses").select(
        "estimator",
        "family",
        error=pl.col("mean_selection_error") * POINTS,
        false_movement=pl.col("false_movement") * POINTS,
        delay=pl.col("mean_step_detection_delay_days"),
    )
    posthoc = report.table(heading="POST HOC: Stage A recovery and false movement").filter(
        pl.col("estimator").is_in(POST_HOC_ESTIMATORS)
    )
    added = posthoc.select(
        "estimator",
        family=pl.lit("post hoc"),
        error=pl.col("mean_selection_error") * POINTS,
        false_movement=pl.col("false_movement_real_windows") * POINTS,
        delay=pl.col("mean_step_detection_delay_days"),
    )
    combined = pl.concat([main, added])
    return combined.with_columns(
        family_name=pl.col("family").replace_strict(FAMILY_NAMES, return_dtype=pl.String)
    )


def family_scale(*, frame: pl.DataFrame) -> tuple[list[str], list[str]]:
    """Return the family names that appear in a frame, in legend order, with their colours."""
    present = [name for name in FAMILY_COLOURS if name in set(frame["family_name"].to_list())]
    return present, [FAMILY_COLOURS[name] for name in present]


def render_caption(
    *,
    panels: Sequence[alt.TopLevelMixin],
    number: int,
    title: str,
    subtitle: Sequence[str],
) -> alt.VConcatChart:
    """Stack panels under the shared "Figure N:" caption."""
    return captioned_figure(
        panels=list(panels),  # ty: ignore[invalid-argument-type]
        number=number,
        title=title,
        subtitle=subtitle,
        figure_planning=None,
    )


def headline_figure(*, report: ReportText, text: FigureText) -> alt.VConcatChart:
    """Figure 1: every arm's change in day-ahead error against the baseline.

    Args:
        report: The report.
        text: The text collector.

    Returns:
        The figure.
    """
    p1_heading = "P1 (planned): the selected causal arm minus B0, pooled over the plants"
    p1_pattern = (
        rf"difference {NUMBER_PATTERN} points; "
        rf"\*\*headline: month t-interval {INTERVAL_PATTERN}\*\*"
    )
    kf = {
        setting: estimate_in(
            line=report.line(heading=p1_heading, prefix=f"- {setting}:"), pattern=p1_pattern
        )
        for setting in ("primary", "sensitivity")
    }
    bound_heading = (
        "POST HOC: the capacity-only neighbour-ratio bound against the same decision rule"
    )
    bound = {
        setting: estimate_in(
            line=report.line(heading=bound_heading, prefix=f"- {setting}:"), pattern=T_PATTERN
        )
        for setting in ("primary", "sensitivity")
    }
    hyst_heading = "POST HOC: HYST, designed after the main results"
    hyst = {
        setting: estimate_in(
            line=report.line(heading=hyst_heading, prefix=f"- {setting}, HYST minus B0:"),
            pattern=T_PATTERN,
        )
        for setting in ("primary", "sensitivity")
    }
    arms = report.table(heading="POST HOC: other estimators as Stage B arms (primary setting)")
    others = {
        str(row["estimator"]): Estimate(
            float(row["fit_minus_B0_points"]),
            float(row["fit_t_lower"]),
            float(row["fit_t_upper"]),
        )
        for row in arms.iter_rows(named=True)
    }
    threshold = float(
        re.search(
            r"threshold ([\d.]+) points",
            report.line(heading=p1_heading, prefix="- primary:"),
        ).group(1)  # ty: ignore[unresolved-attribute]
    )
    b0_error = float(
        report.table(heading=STAGE_B_HEADING).filter(
            (pl.col("arm") == "B0") & (pl.col("setting") == "primary")
        )["mean_error_points"][0]
    )
    off_scale = others.pop("CX1")
    off_scale_text = (
        f"CX1, a post hoc convex estimator, is off the scale at {estimate_text(off_scale)}."
    )
    ordered = ["HM", "PR", "CX2", "NR", "DSKF", "DP1"]
    labels = [
        "KF pick, primary setting",
        "KF pick, second setting",
        "No-fit bound, primary setting",
        "No-fit bound, second setting",
        "HYST, primary setting",
        "HYST, second setting",
        *[f"{name}, primary setting" for name in ordered],
    ]
    estimates = [
        kf["primary"],
        kf["sensitivity"],
        bound["primary"],
        bound["sensitivity"],
        hyst["primary"],
        hyst["sensitivity"],
        *[others[name] for name in ordered],
    ]
    keys = ["Planned"] * 2 + ["Exploratory"] * 2 + ["Post hoc"] * (2 + len(ordered))
    hollow = [False, True, False, True, False, True, *([False] * len(ordered))]
    rows = make_rows(labels=labels, estimates=estimates, colour_keys=keys, hollow=hollow)
    colours = {
        "Planned": ocf.BRAND_ORANGE,
        "Exploratory": ocf.DATA_BLUE,
        "Post hoc": ocf.DATA_PURPLE,
    }
    x_title = "Error minus B0's error (points of capacity; more negative is better)"
    panel = dot_interval_panel(
        rows=rows,
        x_domain=(-0.4, 0.7),
        text_room=0.75,
        x_title=x_title,
        zero_label="same as B0",
        colour_scale=(list(colours), list(colours.values())),
        colour_title=None,
        references=[(-threshold, None), (threshold, None)],
    )
    key = key_row(
        entries=[
            ("Planned", ocf.BRAND_ORANGE, "circle", True),
            ("Exploratory", ocf.DATA_BLUE, "circle", True),
            ("Post hoc", ocf.DATA_PURPLE, "circle", True),
            ("Second setting", ocf.BLACK_1, "triangle-up", False),
        ]
    )
    hold_days = int(
        re.search(
            r"lasts (\d+) days",
            report.line(heading="Constants the page quotes", prefix="- HYST"),
        ).group(1)  # ty: ignore[unresolved-attribute]
    )
    title = (
        "A capacity estimate that updates daily raised day-ahead solar error at 6 plants, "
        "and one that holds the static value until a drop lasts did not"
    )
    subtitle = [
        (
            "Change in an XGBoost forecast's mean absolute error when its training power is "
            f"divided by an estimate of each plant's capacity, against B0, which divides by the "
            f"plant's static capacity and has an error of {b0_error:.2f} points of capacity."
        ),
        (
            "Dot: estimate. Line: 95% interval over the 18 scored months. Dashed lines: 2% of "
            f"B0's error ({threshold:.3f} points), the smallest change the plan counted as worth "
            "acting on. Planned: written into the plan before any result existed. Post hoc: "
            "designed after the results. Hollow triangle: the second XGBoost setting."
        ),
        (
            "KF pick: the Kalman filter, which the plan's selection rule chose. No-fit bound: B0's "
            "forecast multiplied by each plant-month's output relative to its neighbours. HYST "
            f"holds the static capacity until a drop lasts {hold_days} days. Estimators below "
            f"HYST ran at the primary setting only. {off_scale_text}"
        ),
    ]
    text.add(
        number=1,
        plots="Change in day-ahead error against B0, 12 rows, KF pick and no-fit bound and HYST "
        "at two settings",
        lines=[
            title,
            *subtitle,
            x_title,
            *labels,
            "same as B0",
            "Key: Planned / Exploratory / Post hoc / Second setting",
        ],
    )
    return render_caption(panels=[key, panel], number=1, title=title, subtitle=subtitle)


def step_figure(*, root: Path, report: ReportText, text: FigureText) -> alt.VConcatChart:
    """Figure 2: what each estimator reports across one planted step.

    Args:
        root: The output root.
        report: The report.
        text: The text collector.

    Returns:
        The figure.
    """
    traces = pl.read_parquet(root / "tables" / "example_event.parquet")
    rule = report.line(
        heading="Example planted step (the figure of one planted step)", prefix="- rule"
    )
    found = re.search(r"this one is a ([\d.]+)% loss lasting (\d+) days", rule)
    if found is None:
        msg = f"no size and length in {rule!r}"
        raise ValueError(msg)
    size, length = float(found.group(1)), int(found.group(2))
    order = list(ESTIMATOR_TITLES)
    order = [
        name for name in order if name in set(traces["estimator"].to_list()) and name != "DSKF"
    ]
    long = pl.concat(
        [
            traces.select(
                "estimator",
                "day",
                series=pl.lit("Share of power left after the planted step"),
                value=pl.col("planted_factor"),
            ),
            traces.select(
                "estimator",
                "day",
                series=pl.lit("Estimator's capacity after the step over its capacity without it"),
                value=pl.col("ratio"),
            ),
        ]
    ).with_columns(
        title=pl.col("estimator").replace_strict(ESTIMATOR_TITLES, return_dtype=pl.String)
    )
    truth, estimated = (
        "Share of power left after the planted step",
        "Estimator's capacity after the step over its capacity without it",
    )
    chart = (
        alt.Chart(long)
        .mark_line(strokeWidth=2, clip=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "day:Q",
                title="Days since the step began",
                scale=alt.Scale(domain=[-30, 90], nice=False),
                axis=alt.Axis(values=[0, 30, 60, 90]),
            ),
            y=alt.Y(
                "value:Q",
                title="Share of capacity",
                scale=alt.Scale(domain=[0.4, 1.3], nice=False),
                axis=alt.Axis(values=[0.5, 0.7, 1.0], format=".1f"),
            ),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=[truth, estimated], range=[ocf.BLACK_1, ocf.DATA_BLUE]),
                legend=alt.Legend(title=None, labelLimit=0, symbolType="stroke", columns=1),
            ),
            order=alt.Order("series:N"),
        )
        .properties(width=170, height=105)
        .facet(
            facet=alt.Facet(
                "title:N",
                sort=[ESTIMATOR_TITLES[name] for name in order],
                header=alt.Header(title=None, labelFontSize=11, labelLimit=0, labelAnchor="start"),
            ),
            columns=3,
        )
    )
    title = (
        f"On one planted {size:.0f}% step, most estimators followed the loss within days, "
        "and S0 and EM never did"
    )
    subtitle = [
        (
            "One planted loss, the same in every panel: a step that cuts one plant's power by "
            f"{size:.0f}% for {length} days, starting on day 0. Each panel draws the capacity an "
            "estimator reports for the plant with the loss, divided by what it reports without "
            "the loss."
        ),
        (
            "A forecast for day T reads whole days up to day T−2. Before day 0 both runs read "
            "the same power, so the ratio is exactly 1. The convex programmes are solved only "
            "from the step on, so their lines start there."
        ),
    ]
    text.add(
        number=2,
        plots="One planted step, 12 small panels of capacity ratio against days",
        lines=[
            title,
            *subtitle,
            "Days since the step began",
            "Share of capacity",
            truth,
            estimated,
            *ESTIMATOR_TITLES.values(),
        ],
    )
    return render_caption(panels=[chart], number=2, title=title, subtitle=subtitle)


def recovery_figure(*, report: ReportText, text: FigureText) -> alt.VConcatChart:
    """Figure 3: how closely each estimator recovers planted losses.

    Args:
        report: The report.
        text: The text collector.

    Returns:
        The figure.
    """
    table = stage_a_table(report=report).sort("error")
    reference_error = float(table.filter(pl.col("estimator") == "S0")["error"][0])
    better = table.filter(pl.col("error") < reference_error - 1e-9)
    claim(
        holds=better.height > table.height / 2 and table["estimator"][0] == "KF",
        what="most estimators beat doing nothing and KF is first",
    )
    labels = [
        f"{row['estimator']} ({ESTIMATOR_TITLES[row['estimator']].split(': ')[1]})"
        for row in table.iter_rows(named=True)
    ]
    texts = [
        f"{row['error']:.2f}" + ("" if row["delay"] is None else f", delay {row['delay']:.1f} days")
        for row in table.iter_rows(named=True)
    ]
    rows = pl.DataFrame(
        {
            "label": labels,
            "point": table["error"],
            "low": [None] * table.height,
            "high": [None] * table.height,
            "colour_key": table["family_name"],
            "hollow": [False] * table.height,
            "shape": ["circle"] * table.height,
            "text": texts,
        },
        schema_overrides={"low": pl.Float64, "high": pl.Float64},
    )
    names, colours = family_scale(frame=table)
    x_title = "Mean error of the recovered share of capacity (points; smaller is better)"
    panel = dot_interval_panel(
        rows=rows,
        x_domain=(0.0, 7.0),
        text_room=4.0,
        x_title=x_title,
        zero_label=None,
        colour_scale=(names, colours),
        colour_title="Estimator family",
        references=[(reference_error, "S0: do nothing")],
    )
    title = (
        "Most estimators recovered a planted loss better than doing nothing, "
        "and the Kalman filter did so best"
    )
    subtitle = [
        (
            "Each dot: an estimator's mean gap, over planted steps, ramps, and intermittent "
            "losses at every plant in 10 random draws, between the capacity share it recovered "
            "and the share left after the loss. Smaller is better."
        ),
        (
            "The text at the right gives the mean days an estimator took to cross halfway to a "
            "planted step. EM and S0 never detect a step. The dashed line is S0, which always "
            "reports the static capacity. Post hoc: designed after the results."
        ),
    ]
    text.add(
        number=3,
        plots="Mean recovery error by estimator, 13 rows, family colours, S0 reference line",
        lines=[title, *subtitle, x_title, "Estimator family: " + ", ".join(names), *labels],
    )
    return render_caption(panels=[panel], number=3, title=title, subtitle=subtitle)


def false_movement_figure(*, report: ReportText, text: FigureText) -> alt.VConcatChart:
    """Figure 4: how often each estimator moves when nothing changed.

    Args:
        report: The report.
        text: The text collector.

    Returns:
        The figure.
    """
    table = stage_a_table(report=report).sort("false_movement")
    claim(
        holds=set(table["estimator"][:3].to_list()) == {"S0", "HYST", "EM"}
        and float(table["false_movement"][2]) < 10.0 <= float(table["false_movement"][3]),
        what="only S0, HYST, and EM stay under the 10% gate",
    )
    labels = [
        f"{row['estimator']} ({ESTIMATOR_TITLES[row['estimator']].split(': ')[1]})"
        for row in table.iter_rows(named=True)
    ]
    rows = pl.DataFrame(
        {
            "label": labels,
            "point": table["false_movement"],
            "low": [None] * table.height,
            "high": [None] * table.height,
            "colour_key": table["family_name"],
            "hollow": [False] * table.height,
            "shape": ["circle"] * table.height,
            "text": [f"{value:.1f}%" for value in table["false_movement"]],
        },
        schema_overrides={"low": pl.Float64, "high": pl.Float64},
    )
    names, colours = family_scale(frame=table)
    x_title = "Share of days on which the estimate moved more than 5% (smaller is better)"
    panel = dot_interval_panel(
        rows=rows,
        x_domain=(-5.0, 100.0),
        text_room=18.0,
        x_title=x_title,
        zero_label=None,
        colour_scale=(names, colours),
        colour_title="Estimator family",
        references=[(10.0, "10% gate")],
        tick_values=[0, 20, 40, 60, 80, 100],
    )
    title = (
        "On plant-months whose output matched the neighbouring plants', "
        "only HYST and the references stayed still"
    )
    subtitle = [
        (
            "Each dot: the share of days, in 23 plant-months whose output was within 5% of the "
            "neighbouring plants', on which an estimator's capacity moved more than 5% from its "
            "median over the previous 90 days. A day on which an estimator fell back to the "
            "static capacity counts as a move."
        ),
        (
            "Dashed line: the plan's gate. Smaller is better, but S0 and EM stay still because "
            "they cannot follow a loss (Figure 3). Post hoc: designed after the results."
        ),
    ]
    text.add(
        number=4,
        plots="Share of days moved more than 5% on 23 quiet plant-months, 13 rows, 10% gate line",
        lines=[title, *subtitle, x_title, "Estimator family: " + ", ".join(names), *labels],
    )
    return render_caption(panels=[panel], number=4, title=title, subtitle=subtitle)


def seasonal_figure(*, root: Path, text: FigureText) -> alt.VConcatChart:
    """Figure 5: the raw Kalman filter level through the calendar year.

    Args:
        root: The output root.
        text: The text collector.

    Returns:
        The figure.
    """
    table = pl.read_parquet(root / "tables" / "kf_seasonal_shape.parquet")
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    table = table.with_columns(
        month=pl.col("calendar_month").replace_strict(
            dict(enumerate(names, start=1)), return_dtype=pl.String
        )
    )
    lowest_month = table.sort("level_over_median").group_by("plant", maintain_order=True).first()
    claim(
        holds=bool((lowest_month["calendar_month"] == 12).all()),
        what="December is every plant's lowest month",
    )
    median = (
        table.group_by("month", "calendar_month")
        .agg(level=pl.col("level_over_median").median())
        .sort("calendar_month")
    )
    x = alt.X(
        "month:N",
        sort=names,
        title="Calendar month",
        axis=alt.Axis(labelAngle=0, grid=False),
    )
    one_plant = (
        alt.Chart(table.with_columns(plant=pl.col("plant").cast(pl.String)))
        .mark_line(strokeWidth=1.5, opacity=0.6, color=ocf.DATA_BLUE_LIGHT, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y(
                "level_over_median:Q",
                title="Level over the plant's median level",
                scale=alt.Scale(domain=[0.6, 1.2], nice=False),
                axis=alt.Axis(values=[0.6, 0.8, 1.0, 1.2]),
            ),
            detail="plant:N",
        )
    )
    across = (
        alt.Chart(median)
        .mark_line(
            strokeWidth=3,
            color=ocf.DATA_BLUE,
            point=alt.OverlayMarkDef(filled=True, size=60),
            aria=False,
        )
        .encode(x=x, y="level:Q")  # ty: ignore[unresolved-attribute]
    )
    one = (
        alt.Chart(pl.DataFrame({"y": [1.0]}))
        .mark_rule(color=ocf.BLACK_1, strokeDash=[4, 3], aria=False)
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    panel = (one_plant + across + one).properties(width=CONTENT_WIDTH_PX - 90, height=210)
    title = "The Kalman filter's raw capacity level is lowest in December at every plant"
    subtitle = [
        (
            "Each thin line: one plant's mean Kalman filter level in each calendar month, over "
            "that plant's median level, on days the filter did not fall back to the static "
            "capacity. Thick line: the median across the 6 plants. Dashed line: the plant's "
            "median level."
        ),
        (
            "A plant whose capacity did not change would draw a flat line at 1. Plants are not "
            "labelled."
        ),
    ]
    text.add(
        number=5,
        plots="KF raw level over plant median by calendar month, 6 plants and their median",
        lines=[title, *subtitle, "Calendar month", "Level over the plant's median level"],
    )
    return render_caption(panels=[panel], number=5, title=title, subtitle=subtitle)


def winter_figure(*, report: ReportText, text: FigureText) -> alt.VConcatChart:
    """Figure 6: December to February against the other months, for KF and HYST.

    Args:
        report: The report.
        text: The text collector.

    Returns:
        The figure.
    """
    kf_heading = "POST HOC: winter against the other months (selected minus B0)"
    kf_winter = estimate_in(
        line=report.line(heading=kf_heading, prefix="- December to February"), pattern=T_PATTERN
    )
    kf_other = estimate_in(
        line=report.line(heading=kf_heading, prefix="- the other months"), pattern=T_PATTERN
    )
    hyst_heading = "POST HOC: HYST, designed after the main results"
    hyst_winter = estimate_in(
        line=report.line(heading=hyst_heading, prefix="- primary, December to February"),
        pattern=T_PATTERN,
    )
    hyst_other = estimate_in(
        line=report.line(heading=hyst_heading, prefix="- primary, the other months"),
        pattern=T_PATTERN,
    )
    labels = [
        "KF, December to February",
        "KF, the other 13 months",
        "HYST, December to February",
        "HYST, the other 13 months",
    ]
    rows = make_rows(
        labels=labels,
        estimates=[kf_winter, kf_other, hyst_winter, hyst_other],
        colour_keys=["KF", "KF", "HYST", "HYST"],
    )
    x_title = "Error minus B0's error (points of capacity; more negative is better)"
    panel = dot_interval_panel(
        rows=rows,
        x_domain=(-0.5, 1.7),
        text_room=1.6,
        x_title=x_title,
        zero_label="same as B0",
        colour_scale=(["KF", "HYST"], [ocf.BRAND_ORANGE, ocf.DATA_PURPLE]),
        colour_title=None,
    )
    title = "The Kalman filter raised error in December to February, and HYST did not"
    subtitle = [
        (
            "Change in an XGBoost forecast's mean absolute error against B0, for the 5 scored "
            "months from December to February and the 13 others. Dot: estimate. Line: 95% "
            "interval from a t-distribution over those months."
        ),
        (
            "KF: the Kalman filter (orange). HYST: hold at static (purple). Post hoc, at the "
            "primary setting."
        ),
    ]
    text.add(
        number=6,
        plots="KF and HYST minus B0 in Dec-Feb and in the other months, 4 rows",
        lines=[title, *subtitle, x_title, *labels, "same as B0"],
    )
    return render_caption(panels=[panel], number=6, title=title, subtitle=subtitle)


def low_months_figure(*, report: ReportText, text: FigureText) -> alt.VConcatChart:
    """Figure 7: the three low-output plant-months against every other row.

    Args:
        report: The report.
        text: The text collector.

    Returns:
        The figure.
    """
    pattern = rf"{NUMBER_PATTERN} points on the .*?, {NUMBER_PATTERN} points on every other row"
    kf_line = report.line(
        heading="POST HOC: break-even share of low-output rows", prefix="- row-weighted"
    )
    hyst_line = report.line(
        heading="POST HOC: HYST, designed after the main results", prefix="- primary, row weighted"
    )
    kf_low, kf_rest = (float(g) for g in re.search(pattern, kf_line).groups())  # ty: ignore[unresolved-attribute]
    hyst_low, hyst_rest = (float(g) for g in re.search(pattern, hyst_line).groups())  # ty: ignore[unresolved-attribute]
    labels = [
        "KF, the 3 low plant-months",
        "KF, every other row",
        "HYST, the 3 low plant-months",
        "HYST, every other row",
    ]
    points = [kf_low, kf_rest, hyst_low, hyst_rest]
    rows = pl.DataFrame(
        {
            "label": labels,
            "point": points,
            "low": [None] * 4,
            "high": [None] * 4,
            "colour_key": ["KF", "KF", "HYST", "HYST"],
            "hollow": [False] * 4,
            "shape": ["circle"] * 4,
            "text": [signed(value) for value in points],
        },
        schema_overrides={"low": pl.Float64, "high": pl.Float64},
    )
    x_title = "Error minus B0's error (points of capacity; more negative is better)"
    panel = dot_interval_panel(
        rows=rows,
        x_domain=(-6.0, 1.0),
        text_room=2.0,
        x_title=x_title,
        zero_label="same as B0",
        colour_scale=(["KF", "HYST"], [ocf.BRAND_ORANGE, ocf.DATA_PURPLE]),
        colour_title=None,
    )
    title = (
        "The Kalman filter cut error by about 5 points in 3 low plant-months "
        "and added 0.4 points elsewhere"
    )
    subtitle = [
        (
            "Mean change in an XGBoost forecast's absolute error against B0, weighted by rows, in "
            "the three plant-months in which one plant produced about two thirds to three "
            "quarters of its neighbours' output, and in every other row. Post hoc."
        ),
        (
            "No interval: three plant-months cannot be resampled. KF: the Kalman filter "
            "(orange). HYST: hold at static (purple)."
        ),
    ]
    text.add(
        number=7,
        plots="KF and HYST minus B0 on the 3 low plant-months and on all other rows, 4 rows",
        lines=[title, *subtitle, x_title, *labels, "same as B0"],
    )
    return render_caption(panels=[panel], number=7, title=title, subtitle=subtitle)


def control_figure(*, report: ReportText, text: FigureText) -> alt.VConcatChart:
    """Figure 8: the positive control.

    Args:
        report: The report.
        text: The text collector.

    Returns:
        The figure.
    """
    heading = "Positive control: a 30% step in 3 scored months at every plant"
    oracle = estimate_in(
        line=report.line(heading=heading, prefix="- oracle minus B0 on planted data"),
        pattern=BOOT_PATTERN.replace(" points,", " points,"),
    )
    selected = estimate_in(
        line=report.line(heading=heading, prefix="- selected minus B0 on planted data"),
        pattern=BOOT_PATTERN,
    )
    real = estimate_in(
        line=report.line(
            heading="Exploratory: other arms against B0, and hindsight against causal",
            prefix="- causal__KF minus B0",
        ),
        pattern=BOOT_PATTERN,
    )
    share = float(
        re.search(
            r": ([\d.]+)$",
            report.line(heading=heading, prefix="- recovery share"),
        ).group(1)  # ty: ignore[unresolved-attribute]
    )
    steps = report.line(heading=heading, prefix="- control steps per plant")
    plant_months = sum(int(count) for count in re.findall(r"\d+", steps.split(":", 1)[1]))
    labels = [
        "Oracle, planted losses",
        "KF pick, planted losses",
        "KF pick, real data",
    ]
    rows = make_rows(
        labels=labels,
        estimates=[oracle, selected, real],
        colour_keys=["Oracle", "KF", "KF"],
        hollow=[False, False, True],
    )
    x_title = "Error minus B0's error (points of capacity; more negative is better)"
    panel = dot_interval_panel(
        rows=rows,
        x_domain=(-2.5, 1.0),
        text_room=2.2,
        tick_values=[-2, -1, 0, 1],
        x_title=x_title,
        zero_label="same as B0",
        colour_scale=(["Oracle", "KF"], [ocf.DATA_GREEN, ocf.BRAND_ORANGE]),
        colour_title=None,
    )
    title = (
        f"When 30% steps were planted in {plant_months} plant-months, the Kalman filter recovered "
        "most of the oracle's gain"
    )
    subtitle = [
        (
            "Change in an XGBoost forecast's mean absolute error against B0, when a 30% loss is "
            "planted in power in up to 3 months at every plant. The oracle divides by the true "
            f"loss. The Kalman filter recovered {share:.2f} of the oracle's gain, net of its "
            "real-data change."
        ),
        (
            "Dot: estimate. Line: 95% interval from resampling whole months. Hollow triangle: "
            "the Kalman filter on the real, unplanted data. The control shows the instrument can "
            "detect a loss."
        ),
    ]
    text.add(
        number=8,
        plots="Positive control: oracle, KF pick on planted data, KF pick on real data, vs B0",
        lines=[title, *subtitle, x_title, *labels, "same as B0"],
    )
    return render_caption(panels=[panel], number=8, title=title, subtitle=subtitle)


def hyst_activity_figure(*, report: ReportText, text: FigureText) -> alt.VConcatChart:
    """Figure 9: how often the hold-at-static detector left the static capacity, by month.

    Args:
        report: The report.
        text: The text collector.

    Returns:
        The figure.
    """
    table = report.table(heading="POST HOC: HYST, designed after the main results").select(
        "month",
        share=pl.col("share_of_plant_days_off_static") * POINTS,
        plants=pl.col("plants_off_static_on_some_day"),
    )
    month_names = [
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
    ]
    table = table.with_columns(
        label=pl.col("month").map_elements(
            lambda month: f"{month_names[int(month[5:7]) - 1]} {month[:4]}", return_dtype=pl.String
        )
    )
    overall = report.line(
        heading="POST HOC: HYST, designed after the main results",
        prefix="- over all scored days",
    )
    share_all = float(re.search(r"on ([\d.]+)% of plant-days", overall).group(1))  # ty: ignore[unresolved-attribute]
    order = table["label"].to_list()
    bars = (
        alt.Chart(table)
        .mark_bar(color=ocf.DATA_PURPLE, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("label:N", sort=order, title=None, axis=alt.Axis(labelAngle=-60, labelLimit=0)),
            y=alt.Y(
                "share:Q",
                title="Plant-days off the static capacity (%)",
                scale=alt.Scale(domain=[0, 30], nice=False),
                axis=alt.Axis(values=[0, 10, 20, 30]),
            ),
        )
    )
    panel = bars.properties(width=CONTENT_WIDTH_PX - 90, height=190)
    quiet_months = int((table["share"] == 0).sum())
    title = (
        f"HYST left the static capacity on {share_all:.0f}% of plant-days, and on none in "
        f"{quiet_months} of the {table.height} months"
    )
    subtitle = [
        (
            "Each bar: the share of plant-days, across the 6 plants, in a scored month on which "
            "HYST reported a capacity other than the plant's static capacity. Over all 18 "
            f"months the share is {share_all:.1f}%."
        ),
        "The 18 months are December 2024 to June 2026 without January 2026.",
    ]
    text.add(
        number=9,
        plots="HYST share of plant-days off the static capacity, by scored month, 18 bars",
        lines=[title, *subtitle, "Plant-days off the static capacity (%)", *order],
    )
    return render_caption(panels=[panel], number=9, title=title, subtitle=subtitle)


def explainability_figure(*, report: ReportText, text: FigureText) -> alt.VConcatChart:
    """Figure 10: the explainability rubric as a matrix.

    Args:
        report: The report.
        text: The text collector.

    Returns:
        The figure.
    """
    table = report.table(heading="Explainability")
    criteria = {
        "one_sentence": "One-sentence reason",
        "physical_units": "Physical units",
        "attributable": "Cause named",
        "spread": "Spread returned",
        "differentiable": "Differentiable",
    }
    order = table.sort("score", descending=True, maintain_order=True)["estimator"].to_list()
    long = table.unpivot(
        on=list(criteria),
        index=["estimator", "score"],
        variable_name="criterion",
        value_name="grade",
    ).with_columns(
        criterion=pl.col("criterion").replace_strict(criteria, return_dtype=pl.String),
    )
    lowest = table.filter(pl.col("score") == table["score"].min())["estimator"].to_list()
    claim(
        holds=order[:2] == ["DP1", "KF"] and set(lowest) == {"EM", "R30", "CSQ"},
        what="DP1 and KF score highest and EM, R30, and CSQ lowest",
    )
    grades = ["yes", "partly", "no"]
    colours = [ocf.DATA_BLUE, ocf.DATA_BLUE_LIGHT, ocf.GREY_3]
    cells = (
        alt.Chart(long)
        .mark_rect(stroke=ocf.GREY_1, strokeWidth=2, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "criterion:N",
                sort=list(criteria.values()),
                title=None,
                axis=alt.Axis(
                    orient="top",
                    labelAngle=0,
                    labelLimit=0,
                    labelExpr="split(datum.label, ' ')",
                    labelPadding=22,
                    ticks=False,
                    domain=False,
                ),
            ),
            y=alt.Y("estimator:N", sort=order, title=None),
            color=alt.Color(
                "grade:N",
                scale=alt.Scale(domain=grades, range=colours),
                legend=alt.Legend(title="Grade", orient="bottom"),
            ),
        )
    )
    words = (
        alt.Chart(long)
        .mark_text(fontSize=10, color=ocf.BLACK_1, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("criterion:N", sort=list(criteria.values())),
            y=alt.Y("estimator:N", sort=order),
            text="grade:N",
        )
    )
    scores = (
        alt.Chart(
            table.with_columns(
                score_text=pl.col("score").map_elements(
                    lambda v: f"{v:.1f}", return_dtype=pl.String
                )
            )
        )
        .mark_text(align="left", fontSize=12, fontWeight="bold", color=ocf.BLACK_1, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("estimator:N", sort=order), text="score_text:N", x=alt.value(PLOT_WIDTH_PX + 8)
        )
    )
    header = (
        alt.Chart(pl.DataFrame({"t": ["Score"]}))
        .mark_text(
            align="left",
            baseline="bottom",
            fontSize=12,
            fontWeight="bold",
            color=ocf.BLACK_1,
            aria=False,
        )
        .encode(text="t:N", x=alt.value(PLOT_WIDTH_PX + 8), y=alt.value(-8))  # ty: ignore[unresolved-attribute]
    )
    panel = (cells + words + scores + header).properties(
        width=PLOT_WIDTH_PX, height=ROW_STEP_PX * len(order)
    )
    title = (
        "DP1 and KF score highest for explainability, and the running maximum "
        "and the quantile formulas lowest"
    )
    subtitle = [
        (
            "Each estimator is graded yes, partly, or no on five criteria by reading its code and "
            "outputs, before any result existed. The score is the number of yes grades plus half "
            "the number of partly grades."
        ),
        (
            "Criteria: an operator can be told in one sentence why the capacity fell; the fitted "
            "quantities have physical units; a change can be attributed to a cause; the "
            "estimator returns a spread; the estimator can be differentiated through."
        ),
    ]
    text.add(
        number=10,
        plots="Explainability rubric: 11 estimators by 5 criteria, yes/partly/no, score column",
        lines=[title, *subtitle, "Grade: yes, partly, no", *criteria.values(), *order],
    )
    return render_caption(panels=[panel], number=10, title=title, subtitle=subtitle)


def hindsight_figure(*, report: ReportText, text: FigureText) -> alt.VConcatChart:
    """Figure 11: the hindsight reference against the causal estimate.

    Args:
        report: The report.
        text: The text collector.

    Returns:
        The figure.
    """
    exploratory = "Exploratory: other arms against B0, and hindsight against causal"
    causal = estimate_in(
        line=report.line(heading=exploratory, prefix="- causal__KF minus B0"), pattern=BOOT_PATTERN
    )
    hindsight = estimate_in(
        line=report.line(heading=exploratory, prefix="- hindsight__KF minus B0"),
        pattern=BOOT_PATTERN,
    )
    why = "POST HOC: why hindsight is worse than causal"
    boot = (
        rf"{NUMBER_PATTERN} points; month t-interval {INTERVAL_PATTERN}; "
        rf"bootstrap 95% {INTERVAL_PATTERN}"
    )

    def bootstrap_estimate(prefix: str) -> Estimate:
        found = re.search(boot, report.line(heading=why, prefix=prefix))
        if found is None:
            msg = f"no bootstrap interval on the line starting {prefix!r}"
            raise ValueError(msg)
        return Estimate(float(found.group(1)), float(found.group(4)), float(found.group(5)))

    all_months = bootstrap_estimate("- hindsight__KF minus causal__KF, all plant-months")
    without = bootstrap_estimate("- hindsight__KF minus causal__KF, without those two plant-months")
    labels = [
        "Causal KF minus B0",
        "Hindsight KF minus B0",
        "Hindsight minus causal",
        "... without 2 plant-months",
    ]
    rows = make_rows(
        labels=labels,
        estimates=[causal, hindsight, all_months, without],
        colour_keys=[
            "Against B0",
            "Against B0",
            "Hindsight minus causal",
            "Hindsight minus causal",
        ],
    )
    x_title = "Error difference (points of capacity; more negative is better)"
    panel = dot_interval_panel(
        rows=rows,
        x_domain=(-0.5, 1.5),
        text_room=1.6,
        x_title=x_title,
        zero_label="no difference",
        colour_scale=(["Against B0", "Hindsight minus causal"], [ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
        colour_title=None,
    )
    claim(
        holds=all_months.low < 0 < all_months.high and without.low < 0 < without.high,
        what="the hindsight-minus-causal intervals include zero",
    )
    title = (
        "The hindsight Kalman filter did not beat the causal one, and the gap shrank from "
        f"{all_months.point:.2f} to {without.point:.2f} points without two plant-months"
    )
    subtitle = [
        (
            "Change in an XGBoost forecast's mean absolute error. The hindsight Kalman filter "
            "also reads the weeks after the day it estimates; the causal one reads only the "
            "days before. Dot: estimate. Line: 95% interval from resampling whole months."
        ),
        (
            "Rows 1 and 2 are against B0 (orange). Rows 3 and 4 are hindsight minus causal "
            "(blue), the second without the two plant-months with the largest gap. Exploratory "
            "and post hoc."
        ),
    ]
    text.add(
        number=11,
        plots="Causal KF and hindsight KF against B0, and hindsight minus causal with and "
        "without two plant-months",
        lines=[title, *subtitle, x_title, *labels, "no difference"],
    )
    return render_caption(panels=[panel], number=11, title=title, subtitle=subtitle)


def build_figures(*, root: Path, text: FigureText) -> dict[int, alt.VConcatChart]:
    """Build every figure.

    Args:
        root: The output root.
        text: The text collector.

    Returns:
        The figures by number.
    """
    report = ReportText(path=root / "report.md")
    return {
        1: headline_figure(report=report, text=text),
        2: step_figure(root=root, report=report, text=text),
        3: recovery_figure(report=report, text=text),
        4: false_movement_figure(report=report, text=text),
        5: seasonal_figure(root=root, text=text),
        6: winter_figure(report=report, text=text),
        7: low_months_figure(report=report, text=text),
        8: control_figure(report=report, text=text),
        9: hyst_activity_figure(report=report, text=text),
        10: explainability_figure(report=report, text=text),
        11: hindsight_figure(report=report, text=text),
    }


def render_figures(*, figures: dict[int, alt.VConcatChart], output: Path) -> None:
    """Save every figure as SVG and PNG, and publish an optimised SVG of each.

    Args:
        figures: The figures by number.
        output: The folder for the raw SVGs and PNGs.

    Raises:
        subprocess.CalledProcessError: If svgo fails.
        ValueError: If a figure is wider than the text column.
    """
    output.mkdir(exist_ok=True)
    published = {n: DOCS_ASSETS_DIR / f"{ASSET_PREFIX}{n}.svg" for n in figures}
    DOCS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for number, chart in figures.items():
        raw = output / f"figure_{number}.svg"
        chart.save(str(raw))
        width = re.search(r'<svg[^>]* width="(\d+(?:\.\d+)?)"', raw.read_text())
        if width is None or float(width.group(1)) > CONTENT_WIDTH_PX + 4:
            msg = f"figure {number} is {width and width.group(1)} px wide, past {CONTENT_WIDTH_PX}"
            raise ValueError(msg)
        chart.save(str(output / f"figure_{number}.png"), scale_factor=2)
        subprocess.run(
            [*SVGO_COMMAND, "--input", str(raw), "--output", str(published[number])], check=True
        )
        _LOG.info("published %s", published[number])


def main() -> int:
    """Write the figure text, and render the figures unless `--text-only` is given."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-only", action="store_true", help="Write the figure text and stop.")
    arguments = parser.parse_args()
    root = claim_root(smoke=False)
    text = FigureText()
    figures = build_figures(root=root, text=text)
    text_path = root / "figure_text.txt"
    text.write(path=text_path)
    _LOG.info("wrote %s", text_path)
    if arguments.text_only:
        return 0
    render_figures(figures=figures, output=root / "figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
