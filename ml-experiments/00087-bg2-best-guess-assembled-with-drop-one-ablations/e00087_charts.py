"""Draw experiment 00087's figures (the BG2 best guess and its drop-one ablations) as SVG.

The figures are for readers who are not forecasting specialists, so every title, axis title, and
legend entry is in plain words, says which direction is better, and says what an interval is.
Generators appear only as `Solar farm A` to `F` and `Wind farm A` to `C`, and only as scores,
never as time series.

Figures (the numbers follow `bg2_design.md`, section 3):

1. `figure_1_headline`: mean absolute error against forecast day, five panels, five lines each.
2. `figure_2_<panel>_stages`: the same panels with every stage and ablation as a small multiple.
3. `figure_3_staircase`: how the best guess was reached, per class, plus the curtailment cost.
4. `figure_4_ablations`: each ablation minus the best guess, per class.
5. `figure_5_<class>_breakdown`: error by lead band, hour of day, month, season, and sunshine.
6. `figure_6_by_series`: one dot per substation or farm, best guess against the manual heuristic.
7. `figure_7a_reliability`, `figure_7b_tail_gaps`, `figure_7c_pinball`, `figure_7d_tail_scores`.
8. `figure_8_curtailment`: the curtailment cost against NGED's method, with its sweeps.
9. `figure_9_side_panels`: experiment 81's control-only contrasts and experiment 54's UKV ceiling,
   each drawn only if its result page exists.

The tables are read from `work/results/*.parquet` (written by `e00087_score.py` and
`e00087_curtailment.py`). `--fixture` draws every figure from synthetic tables with the same
schema, writes only under `work/charts_fixture/`, and needs no data. A normal run writes the SVGs
to `docs/ml-experiments/00087-*/`, optimises them with `npx svgo@4 --multipass --precision=1
--final-newline`, and writes PNG previews under `work/charts/`.

    uv run python ml-experiments/00087-*/e00087_charts.py [--fixture] [--no-svgo]
"""

import argparse
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00087_arms as arms
import e00087_common as common
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import wrapped

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DOCS_DIR: Final[Path] = (
    REPO_ROOT / "docs" / "ml-experiments" / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
CHARTS_DIR: Final[Path] = common.WORK_DIR / "charts"
FIXTURE_DIR: Final[Path] = common.WORK_DIR / "charts_fixture"
EXPERIMENT_81_PAGE: Final[Path] = REPO_ROOT / "docs" / "ml-experiments"
EXPERIMENT_54_PAGE: Final[Path] = REPO_ROOT / "docs" / "ml-experiments"

Tables = dict[str, pl.DataFrame]

PLOT_WIDTH: Final[int] = 540
"""The plot area's width where the axis labels sit to the left, so the figure fills the column."""
SCOPE: Final[str] = (
    "292 daily forecast runs from July 2025 to June 2026, each forecasting up to 14 days ahead."
)
INTERVAL_NOTE: Final[str] = "Shaded band or line: 95% interval from resampling whole months."
ERROR_TITLE: Final[str] = "Mean absolute error of the median (% of capacity; smaller is better)"
SCORE_TITLE: Final[str] = "Quantile score (points of capacity; smaller is better)"

CLASS_TITLES: Final[dict[str, str]] = {
    "Primary": "Primary substations",
    "BSP/GSP": "Bulk and grid supply points",
    "Solar farms": "Solar farms",
    "Wind farms": "Wind farms",
}
"""Plain names for the four target classes."""
CLASS_NOTES: Final[dict[str, str]] = {
    "Primary": "16 Primary substations",
    "BSP/GSP": "2 bulk supply points (BSP) and 2 grid supply points (GSP)",
    "Solar farms": "6 solar farms",
    "Wind farms": "3 wind farms",
}
PANELS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("primary", "Primary substations, all hours", "Primary", "lead_day"),
    ("primary_midday", "Primary substations, 11:00 to 15:00 UK time", "Primary", "lead_day_midday"),
    ("supply_points", "Bulk and grid supply points", "BSP/GSP", "lead_day"),
    ("solar", "Solar farms", "Solar farms", "lead_day"),
    ("wind", "Wind farms", "Wind farms", "lead_day"),
)
"""Slug, title, class, and scope dimension of each of the five panels of figures 1 and 2."""

PRODUCTION: Final[str] = "Production-style model (a weak baseline)"
CLIMATOLOGY: Final[str] = "Climatology (typical value for the time of year)"
MANUAL: Final[str] = "NGED's manual heuristic"
PREVIOUS: Final[str] = "Previous best forecast"
BEST_GUESS: Final[str] = "Best guess (BG2)"
LINE_ORDER: Final[tuple[str, ...]] = (PRODUCTION, CLIMATOLOGY, MANUAL, PREVIOUS, BEST_GUESS)
LINE_COLOURS: Final[dict[str, str]] = {
    PRODUCTION: ocf.DATA_PURPLE,
    CLIMATOLOGY: ocf.ENSEMBLE_LINE,
    MANUAL: ocf.DATA_GREEN,
    PREVIOUS: ocf.DATA_BLUE,
    BEST_GUESS: ocf.BRAND_ORANGE,
}
LINE_DASHES: Final[dict[str, list[int]]] = {
    PRODUCTION: [2, 2],
    CLIMATOLOGY: [6, 3],
    MANUAL: [1, 0],
    PREVIOUS: [1, 0],
    BEST_GUESS: [1, 0],
}
LINE_WIDTHS: Final[dict[str, float]] = {
    PRODUCTION: 1.5,
    CLIMATOLOGY: 1.5,
    MANUAL: 2.0,
    PREVIOUS: 2.0,
    BEST_GUESS: 3.0,
}
HEADLINE_ARMS: Final[dict[str, str]] = {
    "ref_xgboost_baseline": PRODUCTION,
    "ref_climatology": CLIMATOLOGY,
    "ref_manual_heuristic": MANUAL,
}
PREVIOUS_NOTE: Final[str] = (
    "Previous best forecast: the frozen model with a 14-day correction for substations, and the "
    "weather-trained model for farms."
)
SEASON_ORDER: Final[tuple[str, ...]] = ("Winter", "Spring", "Summer", "Autumn")
SUN_LABELS: Final[dict[str, str]] = {
    "cloudy": "Cloudiest third of days",
    "middle": "Middle third of days",
    "sunny": "Sunniest third of days",
}
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
HEADLINE_LIMIT: Final[str] = "scaled f=0.9"
HEADLINE_RATIO: Final[int] = 20
LIMIT_NAMES: Final[dict[str, str]] = {
    "scaled f=0.9": "Limit at 0.9 x past maximum (headline)",
    "scaled f=0.75": "Limit at 0.75 x past maximum",
    "scaled f=0.5": "Limit at 0.5 x past maximum",
    "real": "Published thermal limit (reference only)",
}
CURTAILMENT_LABEL: Final[str] = "calibration to history, not a saving"
CURTAILMENT_ARM_ORDER: Final[tuple[str, ...]] = (
    "bg2",
    "previous_best",
    "monthly",
    "frozen",
    "nged_nodes_bg2_generators",
    "bg2_nodes_capacity_generators",
    "xgboost_baseline",
    "climatology",
)
STAIRCASE_CURTAILMENT_ARMS: Final[tuple[str, ...]] = ("previous_best", "frozen", "monthly", "bg2")
TAIL_LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.95, 0.99)
TAIL_SCORES: Final[tuple[tuple[str, str, tuple[str, ...]], ...]] = (
    (
        "tw_p99",
        "Above the 99th percentile of the series' history",
        ("Primary", "BSP/GSP", "Solar farms", "Wind farms"),
    ),
    (
        "tw_p95",
        "Above the 95th percentile of the series' history",
        ("Primary", "BSP/GSP", "Solar farms", "Wind farms"),
    ),
    ("tw_p05", "Below the 5th percentile of the series' history", ("Primary", "BSP/GSP")),
)
"""Threshold-weighted score columns, their plain names, and the classes that report each."""

SIDE_81: Final[tuple[tuple[str, float, float, float], ...]] = (
    ("Refit weekly, no correction, minus the stack", -0.240, -0.360, -0.145),
    ("Refit monthly, no correction, minus the stack", -0.201, -0.324, -0.105),
    ("Refit weekly minus refit monthly", -0.039, -0.064, -0.012),
    ("Blend of four models minus refit weekly", -0.038, -0.070, -0.007),
)
"""Experiment 81, fold B, quantile score: label, difference, lower, and upper bound as quoted on
the experiment's page. The interval is the 99.17% interval of the block bootstrap."""
SIDE_54: Final[tuple[tuple[str, float, float, float], ...]] = (
    ("ENS plus UKV minus ENS alone", -0.198, -0.316, -0.069),
    ("UKV alone minus ENS alone", 0.881, 0.460, 1.308),
)
"""Experiment 54, lead hours 24 to 54, quantile score: label, difference, lower, upper (95%)."""


# --- text and layout helpers -------------------------------------------------------------------


def _title(*, text: str, subtitle: str, width: int = 78) -> alt.TitleParams:
    """Return a figure title with a subtitle wrapped to the plot area's width."""
    return alt.TitleParams(
        text=wrapped(text=text, width=width),
        subtitle=wrapped(text=subtitle, width=width + 30),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _panel_title(*, text: str, subtitle: str | None = None) -> alt.TitleParams:
    return alt.TitleParams(
        text=text,
        subtitle=wrapped(text=subtitle, width=100) if subtitle else "",
        anchor="start",
        fontSize=12,
        subtitleFontSize=10,
    )


def _wrap_label(*, text: str, width: int = 34) -> str:
    """Return a label with `|` where it wraps, for a label expression that splits on `|`."""
    return "|".join(wrapped(text=text, width=width))


SPLIT_LABEL: Final[str] = "split(datum.label, '|')"


def _line_colour(*, order: Sequence[str] = LINE_ORDER, legend: bool = True) -> alt.Color:
    return alt.Color(
        "line:N",
        scale=alt.Scale(domain=list(order), range=[LINE_COLOURS[n] for n in order]),
        legend=alt.Legend(title=None, orient="top", columns=2, symbolStrokeWidth=3, labelLimit=420)
        if legend
        else None,
    )


def _line_dash(*, order: Sequence[str] = LINE_ORDER) -> alt.StrokeDash:
    return alt.StrokeDash(
        "line:N",
        scale=alt.Scale(domain=list(order), range=[LINE_DASHES[n] for n in order]),
        legend=None,
    )


def _series_label(*, site: str) -> str:
    """Return a plain anonymous name for a series label such as `D4`, `A`, or `W2`."""
    if site.startswith("W") and site[1:].isdigit():
        return f"Wind farm {chr(ord('A') + int(site[1:]) - 1)}"
    if len(site) == 1 and site.isalpha():
        return f"Solar farm {site}"
    return f"Substation {site}"


# --- reading tables ----------------------------------------------------------------------------


def read_tables(*, results_dir: Path) -> Tables:
    """Read every results table that exists under `results_dir`."""
    names = (
        "absolute",
        "scopes",
        "staircase",
        "contrasts",
        "reliability",
        "tails",
        "noise_floor",
        "curtailment_costs",
        "curtailment_census",
        "primary_census",
    )
    return {
        name: pl.read_parquet(results_dir / f"{name}.parquet")
        for name in names
        if (results_dir / f"{name}.parquet").exists()
    }


def scope_rows(
    *,
    tables: Tables,
    target_class: str,
    dimension: str,
    metric: str,
    lines: Mapping[str, str],
) -> pl.DataFrame:
    """Return scope rows of some arms, one `line` label each (an alias reads its target's rows).

    Args:
        tables: The results tables.
        target_class: A class name.
        dimension: A scope dimension of `e00087_score.scope_table`.
        metric: `mae` or `qs`.
        lines: Arm name mapped to the label to draw it under.

    Returns:
        Rows with `line`, `scope`, `value`, `lower`, `upper`, and `cases`.
    """
    scopes = tables["scopes"].filter(
        (pl.col("class") == target_class)
        & (pl.col("dimension") == dimension)
        & (pl.col("metric") == metric)
    )
    parts = [
        scopes.filter(pl.col("arm") == arms.resolve(name=arm)).with_columns(line=pl.lit(label))
        for arm, label in lines.items()
    ]
    return pl.concat(parts) if parts else pl.DataFrame()


def headline_lines(*, target_class: str) -> dict[str, str]:
    """Return the five headline arms of a class mapped to their labels."""
    return {
        **HEADLINE_ARMS,
        arms.PREVIOUS_BEST_OF_CLASS[target_class]: PREVIOUS,
        arms.BG2_OF_CLASS[target_class]: BEST_GUESS,
    }


def _has(*, tables: Tables, names: Sequence[str]) -> bool:
    return all(name in tables and not tables[name].is_empty() for name in names)


# --- figure 1 and 2: error against forecast day ------------------------------------------------


def _lead_panel(*, frame: pl.DataFrame, title: str, height: int, last: bool) -> Any:
    data = frame.with_columns(day=pl.col("scope").cast(pl.Int32))
    base = alt.Chart(data)
    x = alt.X(
        "day:Q",
        title="Days ahead (0 is the first 24 hours; 14 is 14 days or more)" if last else None,
        scale=alt.Scale(domain=[0, 14], nice=False),
        axis=alt.Axis(values=list(range(0, 15, 2))),
    )
    band = base.mark_area(opacity=0.14, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=x, y=alt.Y("lower:Q"), y2="upper:Q", color=_line_colour()
    )
    line = base.mark_line(aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=x,
        y=alt.Y(
            "value:Q",
            title=["Mean absolute error of", "the median (% of capacity;", "smaller is better)"],
            scale=alt.Scale(zero=True),
        ),
        color=_line_colour(),
        strokeDash=_line_dash(),
        strokeWidth=alt.StrokeWidth(
            "line:N",
            scale=alt.Scale(domain=list(LINE_ORDER), range=[LINE_WIDTHS[n] for n in LINE_ORDER]),
            legend=None,
        ),
    )
    return alt.layer(band, line).properties(
        width=PLOT_WIDTH, height=height, title=_panel_title(text=title)
    )


def figure_1_headline(*, tables: Tables) -> Any:
    """Draw the headline: error against forecast day for five lines in five panels."""
    panels = []
    for index, (_, title, target_class, dimension) in enumerate(PANELS):
        frame = scope_rows(
            tables=tables,
            target_class=target_class,
            dimension=dimension,
            metric="mae",
            lines=headline_lines(target_class=target_class),
        )
        panels.append(
            _lead_panel(frame=frame, title=title, height=150, last=index == len(PANELS) - 1)
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="How far off each forecast is, from the first day to two weeks ahead",
            subtitle=(
                "Mean absolute error of the median forecast, as a percentage of each series' "
                f"capacity; smaller is better. {SCOPE} {INTERVAL_NOTE} {PREVIOUS_NOTE}"
            ),
        )
    )


def _stage_frame(*, tables: Tables, target_class: str, dimension: str) -> pl.DataFrame:
    """Return every stage and ablation's rows, with BG2 and the heuristic repeated per facet."""
    specs = [
        spec
        for spec in (*arms.STAIRCASE[target_class], *arms.ABLATIONS[target_class])
        if spec != arms.BG2_OF_CLASS[target_class]
    ]
    specs = list(dict.fromkeys(specs))
    context = scope_rows(
        tables=tables,
        target_class=target_class,
        dimension=dimension,
        metric="mae",
        lines={
            arms.BG2_OF_CLASS[target_class]: BEST_GUESS,
            "ref_manual_heuristic": MANUAL,
        },
    )
    parts = []
    for spec in specs:
        own = scope_rows(
            tables=tables,
            target_class=target_class,
            dimension=dimension,
            metric="mae",
            lines={spec: "This stage"},
        )
        if own.is_empty():
            continue
        facet = _wrap_label(text=arms.arm_label(name=spec))
        parts.append(pl.concat([own, context]).with_columns(facet=pl.lit(facet)))
    return pl.concat(parts)


def figure_2_stages(*, tables: Tables, panel: tuple[str, str, str, str]) -> Any:
    """Draw one panel of figure 2: a small multiple for every stage and ablation."""
    _, title, target_class, dimension = panel
    data = _stage_frame(tables=tables, target_class=target_class, dimension=dimension).with_columns(
        day=pl.col("scope").cast(pl.Int32)
    )
    order = ["This stage", MANUAL, BEST_GUESS]
    colours = {"This stage": ocf.DATA_BLUE, MANUAL: ocf.DATA_GREEN, BEST_GUESS: ocf.BRAND_ORANGE}
    dashes = {"This stage": [1, 0], MANUAL: [4, 3], BEST_GUESS: [1, 0]}
    facets = data["facet"].unique(maintain_order=True).to_list()
    base = alt.Chart(data).encode(
        x=alt.X(
            "day:Q",
            title="Days ahead",
            scale=alt.Scale(domain=[0, 14], nice=False),
            axis=alt.Axis(values=[0, 7, 14]),
        ),
        y=alt.Y("value:Q", title="Error (% of capacity)", scale=alt.Scale(zero=True)),
        color=alt.Color(
            "line:N",
            scale=alt.Scale(domain=order, range=[colours[n] for n in order]),
            legend=alt.Legend(title=None, orient="top", symbolStrokeWidth=3),
        ),
        strokeDash=alt.StrokeDash(
            "line:N", scale=alt.Scale(domain=order, range=[dashes[n] for n in order]), legend=None
        ),
    )
    chart = base.mark_line(aria=False).properties(width=140, height=100)  # ty: ignore[unresolved-attribute]
    return (
        chart.facet(
            facet=alt.Facet(
                "facet:N",
                sort=facets,
                title=None,
                header=alt.Header(
                    labelExpr=SPLIT_LABEL, labelFontSize=10, labelLimit=170, labelAnchor="start"
                ),
            ),
            columns=3,
        )
        .properties(
            title=_title(
                text=f"{title}: every stage and ablation",
                subtitle=(
                    "Blue line: the stage or ablation named above each small chart. Orange: the "
                    "best guess (BG2). Green dashed: NGED's manual heuristic. Mean absolute error "
                    f"of the median as a percentage of capacity; smaller is better. {SCOPE}"
                ),
            )
        )
        .resolve_scale(y="shared")
    )


# --- figure 3: the staircase -------------------------------------------------------------------


def _staircase_class_chart(*, frame: pl.DataFrame, target_class: str, last: bool) -> Any:
    ordered = frame.sort("position").with_columns(
        row=pl.col("label").map_elements(
            lambda t: _wrap_label(text=t, width=32), return_dtype=pl.String
        ),
        previous=pl.col("qs") - pl.col("step"),
        step_low=pl.col("qs") - pl.col("step") + pl.col("step_lower"),
        step_high=pl.col("qs") - pl.col("step") + pl.col("step_upper"),
        text=pl.when(pl.col("step").is_null())
        .then(pl.lit("first rung"))
        .otherwise(
            pl.format(
                "{} ({} to {})",
                pl.col("step").map_elements(lambda v: f"{v:+.2f}", return_dtype=pl.String),
                pl.col("step_lower").map_elements(lambda v: f"{v:+.2f}", return_dtype=pl.String),
                pl.col("step_upper").map_elements(lambda v: f"{v:+.2f}", return_dtype=pl.String),
            )
        ),
        kind=pl.when(pl.col("arm") == arms.BG2_OF_CLASS[target_class])
        .then(pl.lit(BEST_GUESS))
        .when(pl.col("arm") == "g_a_wind_margin")
        .then(pl.lit("Added, not in the best guess"))
        .otherwise(pl.lit("Earlier stage")),
    )
    rows = ordered["row"].to_list()
    y = alt.Y(
        "row:N",
        sort=rows,
        title=None,
        axis=alt.Axis(labelExpr=SPLIT_LABEL, labelLimit=250, labelFontSize=10),
    )
    x = alt.X(
        "qs:Q",
        title=SCORE_TITLE if last else None,
        scale=alt.Scale(zero=False, padding=12),
    )
    base = alt.Chart(ordered)
    kinds = [BEST_GUESS, "Earlier stage", "Added, not in the best guess"]
    colour = alt.Color(
        "kind:N",
        scale=alt.Scale(domain=kinds, range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE]),
        legend=alt.Legend(title=None, orient="top", columns=3, labelLimit=420),
    )
    first = (
        base.transform_filter("datum.step === null")
        .mark_rule(color=ocf.ENSEMBLE_LINE, strokeWidth=1)
        .encode(  # ty: ignore[unresolved-attribute]
            x="qs_lower:Q", x2="qs_upper:Q", y=y
        )
    )
    joiner = base.mark_rule(strokeDash=[2, 2], color=ocf.ENSEMBLE_LINE).encode(  # ty: ignore[unresolved-attribute]
        x="previous:Q", x2="qs:Q", y=y
    )
    step = base.mark_rule(strokeWidth=3, opacity=0.6).encode(  # ty: ignore[unresolved-attribute]
        x="step_low:Q", x2="step_high:Q", y=y, color=colour
    )
    dots = base.mark_point(filled=True, size=90, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=x, y=y, color=colour
    )
    left = alt.layer(first, joiner, step, dots).properties(
        width=270, height=34 * len(rows), title=_panel_title(text=CLASS_TITLES[target_class])
    )
    right = (
        base.mark_text(align="left", fontSize=10, aria=False)  # ty: ignore[unresolved-attribute]
        .encode(y=alt.Y("row:N", sort=rows, axis=None), text="text:N", x=alt.value(0))
        .properties(
            width=130,
            height=34 * len(rows),
            title=_panel_title(text="Step from rung above"),
        )
    )
    return alt.hconcat(left, right, spacing=4)


def _curtailment_stage_panel(*, tables: Tables) -> Any:
    if not _has(tables=tables, names=["curtailment_costs"]):
        return None
    costs = tables["curtailment_costs"].filter(
        (pl.col("limit_kind") == HEADLINE_LIMIT)
        & (pl.col("breach_ratio") == HEADLINE_RATIO)
        & pl.col("arm").is_in(list(STAIRCASE_CURTAILMENT_ARMS))
    )
    if costs.is_empty():
        return None
    events = int(costs["scored_events"][0])
    half_hours = int(costs["scored_half_hours"][0])
    labels = costs["label"].to_list()
    base = alt.Chart(costs.sort("vs_nged_gbp_per_mw_day"))
    y = alt.Y("label:N", sort=labels, title=None, axis=alt.Axis(labelLimit=300, labelFontSize=10))
    zero = (
        alt.Chart(pl.DataFrame({"zero": [0.0]}))
        .mark_rule(color=ocf.BLACK_1)
        .encode(  # ty: ignore[unresolved-attribute]
            x="zero:Q"
        )
    )
    interval = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x="vs_nged_gbp_per_mw_day_lower:Q",
        x2="vs_nged_gbp_per_mw_day_upper:Q",
        y=y,
        color=alt.value(ocf.DATA_BLUE),
    )
    dots = base.mark_point(filled=True, size=90, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "vs_nged_gbp_per_mw_day:Q",
            title=(
                "Cost minus NGED's method (GBP per MW per day; "
                "below zero means a lower cost than NGED's method)"
            ),
        ),
        y=y,
        color=alt.value(ocf.BRAND_ORANGE),
    )
    return alt.layer(zero, interval, dots).properties(
        width=PLOT_WIDTH - 60,
        height=34 * len(labels),
        title=_panel_title(
            text=f"Curtailment cost compared with NGED's current method ({CURTAILMENT_LABEL})",
            subtitle=(
                f"Bulk and grid supply points with their farms, limit at 0.9 x the past maximum "
                f"export, July 2025 to June 2026. Events: {events} of {half_hours:,} scored "
                "half-hours were beyond the limit; too few events to rank the stages."
            ),
        ),
    )


def figure_3_staircase(*, tables: Tables) -> Any:
    """Draw the staircase of each class, then the curtailment cost against NGED's method."""
    charts: list[Any] = []
    present = [c for c in arms.CLASSES if (tables["staircase"]["class"] == c).any()]
    for index, target_class in enumerate(present):
        frame = tables["staircase"].filter(pl.col("class") == target_class)
        charts.append(
            _staircase_class_chart(
                frame=frame, target_class=target_class, last=index == len(present) - 1
            )
        )
    curtailment = _curtailment_stage_panel(tables=tables)
    if curtailment is not None:
        charts.append(curtailment)
    notes = "; ".join(f"{name} ({why})" for name, why in arms.NOT_RUN_STEPS)
    return (
        alt.vconcat(*charts)
        .resolve_scale(x="independent", color="shared")
        .properties(
            title=_title(
                text="How we got to the best guess: the score after each added idea",
                subtitle=(
                    "Dot: quantile score of a stage (a score of the whole forecast range, in "
                    "points of capacity; smaller is better). Thick line: 95% interval of the "
                    "step from the stage above, drawn from that stage's dot, not the uncertainty "
                    "of the dot's own position. Dotted line joins the two dots. Thin grey line: "
                    "the first rung's own interval. The order of the ideas shapes each step's "
                    f"size. Ideas tested and not adopted: {notes}."
                ),
            )
        )
    )


# --- figure 4: ablations -----------------------------------------------------------------------


def figure_4_ablations(*, tables: Tables) -> Any:
    """Draw each ablation's quantile score minus the best guess' score, per class."""
    contrasts = tables["contrasts"].filter(
        (pl.col("family") == "ablation") & (pl.col("metric") == "qs") & (pl.col("runs") == "gap")
    )
    kinds = {
        "worse": "Scores worse than the best guess (the idea helps)",
        "better": "Scores better than the best guess",
        "unclear": "Not statistically significant at the 5% level",
    }
    colours = [ocf.BRAND_ORANGE, ocf.DATA_PURPLE, ocf.DATA_SKY]
    panels = []
    present = [c for c in arms.CLASSES if (contrasts["class"] == c).any()]
    for target_class in present:
        rows = contrasts.filter(pl.col("class") == target_class).with_columns(
            row=pl.col("treatment").map_elements(
                lambda t: _wrap_label(text=arms.arm_label(name=t), width=36), return_dtype=pl.String
            ),
            kind=pl.when(pl.col("lower") > 0)
            .then(pl.lit(kinds["worse"]))
            .when(pl.col("upper") < 0)
            .then(pl.lit(kinds["better"]))
            .otherwise(pl.lit(kinds["unclear"])),
        )
        order = rows["row"].to_list()
        y = alt.Y(
            "row:N",
            sort=order,
            title=None,
            axis=alt.Axis(labelExpr=SPLIT_LABEL, labelLimit=260, labelFontSize=10),
        )
        colour = alt.Color(
            "kind:N",
            scale=alt.Scale(domain=list(kinds.values()), range=colours),
            legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=420),
        )
        base = alt.Chart(rows)
        layers = []
        if _has(tables=tables, names=["noise_floor"]) and target_class == "Primary":
            floor = tables["noise_floor"].filter(pl.col("metric") == "qs")
            if not floor.is_empty():
                width = abs(float(floor["difference"][0]))
                layers.append(
                    alt.Chart(pl.DataFrame({"low": [-width], "high": [width]}))
                    .mark_rect(color="#DDDDDD", opacity=0.6)
                    .encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
                )
        layers += [
            alt.Chart(pl.DataFrame({"zero": [0.0]}))
            .mark_rule(color=ocf.BLACK_1)
            .encode(  # ty: ignore[unresolved-attribute]
                x="zero:Q"
            ),
            base.mark_bar(opacity=0.8, size=14).encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "difference:Q",
                    title=ABLATION_X_TITLE,
                    axis=alt.Axis(tickCount=6),
                ),
                y=y,
                color=colour,
            ),
            base.mark_rule(strokeWidth=2, color=ocf.BLACK_1).encode(  # ty: ignore[unresolved-attribute]
                x="lower:Q", x2="upper:Q", y=y
            ),
        ]
        panels.append(
            alt.layer(*layers).properties(
                width=PLOT_WIDTH - 60,
                height=28 * len(order),
                title=_panel_title(text=CLASS_TITLES[target_class]),
            )
        )
    return (
        alt.vconcat(*panels)
        .resolve_scale(x="independent", color="shared")
        .properties(
            title=_title(
                text="Taking out one idea at a time: which ideas the best guess needs",
                subtitle=(
                    "Each bar is the quantile score with one idea removed minus the best guess' "
                    "score, in points of capacity. A bar to the right means the forecast got "
                    "worse without the idea. Line: 95% interval from resampling whole months. "
                    "Grey band (Primary substations): the size of the difference between two "
                    f"fits of the same model. {SCOPE}"
                ),
            )
        )
    )


ABLATION_X_TITLE: Final[list[str]] = [
    "Quantile score without the idea minus the best guess' (points of capacity;",
    "right of zero means the idea helps)",
]


# --- figure 5: breakdowns ----------------------------------------------------------------------


BREAKDOWN_LINES: Final[dict[str, str]] = {
    "ref_xgboost_baseline": PRODUCTION,
    "ref_manual_heuristic": MANUAL,
    "BG2": BEST_GUESS,
}
BREAKDOWN_ORDER: Final[tuple[str, ...]] = (PRODUCTION, MANUAL, BEST_GUESS)


def _category_panel(
    *, frame: pl.DataFrame, order: Sequence[str], title: str, x_title: str | None
) -> Any:
    base = alt.Chart(frame)
    x = alt.X("category:N", sort=list(order), title=x_title, axis=alt.Axis(labelAngle=0))
    offset = alt.XOffset("line:N", sort=list(BREAKDOWN_ORDER))
    colour = _line_colour(order=BREAKDOWN_ORDER)
    rules = base.mark_rule(strokeWidth=1.5).encode(  # ty: ignore[unresolved-attribute]
        x=x, xOffset=offset, y="lower:Q", y2="upper:Q", color=colour
    )
    dots = base.mark_point(filled=True, size=45, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=x,
        xOffset=offset,
        y=alt.Y("value:Q", title="Error (% of capacity)", scale=alt.Scale(zero=True)),
        color=_line_colour(order=BREAKDOWN_ORDER),
    )
    return alt.layer(rules, dots).properties(
        width=PLOT_WIDTH, height=110, title=_panel_title(text=title)
    )


def _numeric_line_panel(
    *,
    frame: pl.DataFrame,
    title: str,
    x_title: str,
    domain: tuple[int, int],
    ticks: list[int],
    bands: bool,
    label_names: Sequence[str] | None = None,
) -> Any:
    base = alt.Chart(frame)
    x = alt.X(
        "number:Q",
        title=x_title,
        scale=alt.Scale(domain=list(domain), nice=False),
        axis=alt.Axis(values=ticks)
        if label_names is None
        else alt.Axis(values=ticks, labelExpr=f"{list(label_names)}[datum.value]"),
    )
    colour = _line_colour(order=BREAKDOWN_ORDER)
    layers = []
    if bands:
        layers.append(
            base.mark_area(opacity=0.14, aria=False, clip=True).encode(  # ty: ignore[unresolved-attribute]
                x=x, y="lower:Q", y2="upper:Q", color=colour
            )
        )
    layers.append(
        base.mark_line(aria=False, clip=True).encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y("value:Q", title="Error (% of capacity)", scale=alt.Scale(zero=True)),
            color=colour,
            strokeDash=_line_dash(order=BREAKDOWN_ORDER),
        )
    )
    return alt.layer(*layers).properties(
        width=PLOT_WIDTH, height=110, title=_panel_title(text=title)
    )


def figure_5_breakdown(*, tables: Tables, target_class: str) -> Any:
    """Draw one class's error by lead band, hour of day, month, season, and sunshine."""
    lines = {
        "ref_xgboost_baseline": PRODUCTION,
        "ref_manual_heuristic": MANUAL,
        arms.BG2_OF_CLASS[target_class]: BEST_GUESS,
    }

    def get(dimension: str) -> pl.DataFrame:
        return scope_rows(
            tables=tables,
            target_class=target_class,
            dimension=dimension,
            metric="mae",
            lines=lines,
        )

    band_names = [name for name, _, _ in common.LEAD_BANDS]
    band = get("lead_band").with_columns(category=pl.col("scope"))
    season = get("season").with_columns(category=pl.col("scope"))
    sun = get("sun").with_columns(category=pl.col("scope").replace_strict(SUN_LABELS))
    hour = get("hour_uk").with_columns(number=pl.col("scope").cast(pl.Int32))
    months = sorted(get("origin_month")["scope"].unique().to_list())
    month_number = {m: i for i, m in enumerate(months)}
    month = get("origin_month").with_columns(
        number=pl.col("scope").replace_strict(month_number, return_dtype=pl.Int32)
    )
    solar = target_class == "Solar farms"
    hour_panel = _numeric_line_panel(
        frame=hour,
        title="By hour of day (UK time)",
        x_title="Hour of the day, UK time (hour starting)",
        domain=(4, 20) if solar else (0, 23),
        ticks=[4, 8, 12, 16, 20] if solar else [0, 6, 12, 18, 24],
        bands=True,
    )
    month_panel = _numeric_line_panel(
        frame=month,
        title="By month the forecast was made (no interval: one month each)",
        x_title="Month the forecast run started",
        domain=(0, max(len(months) - 1, 1)),
        ticks=list(range(len(months))),
        label_names=[MONTH_NAMES[int(m[5:7]) - 1] for m in months],
        bands=False,
    )
    panels = [
        _category_panel(frame=band, order=band_names, title="By how far ahead", x_title=None),
        hour_panel,
        month_panel,
        _category_panel(frame=season, order=SEASON_ORDER, title="By season", x_title=None),
        _category_panel(
            frame=sun,
            order=list(SUN_LABELS.values()),
            title="Sunny and cloudy days (CAMS daily clear-sky index)",
            x_title=None,
        ),
    ]
    return alt.vconcat(*panels).properties(
        title=_title(
            text=f"{CLASS_TITLES[target_class]}: where the error is largest",
            subtitle=(
                "Mean absolute error of the median forecast as a percentage of capacity; "
                f"smaller is better. {CLASS_NOTES[target_class]}. {SCOPE} Dot or line: "
                "estimate. Vertical line or band: 95% interval from resampling whole months. "
                "Sunny and cloudy days are the top and bottom thirds of each month's "
                "daily clear-sky index."
            ),
        )
    )


# --- figure 6: by series -----------------------------------------------------------------------


def figure_6_by_series(*, tables: Tables) -> Any:
    """Draw one dot per series for the best guess and the manual heuristic."""
    panels = []
    present = [
        c
        for c in arms.CLASSES
        if (tables["scopes"].filter(pl.col("dimension") == "series")["class"] == c).any()
    ]
    for index, target_class in enumerate(present):
        frame = scope_rows(
            tables=tables,
            target_class=target_class,
            dimension="series",
            metric="mae",
            lines={"ref_manual_heuristic": MANUAL, arms.BG2_OF_CLASS[target_class]: BEST_GUESS},
        ).with_columns(
            series=pl.col("scope").map_elements(
                lambda s: _series_label(site=s), return_dtype=pl.String
            )
        )
        ordering = (
            frame.filter(pl.col("line") == MANUAL)
            .sort("value", descending=True)["series"]
            .to_list()
        )
        order = ordering + [s for s in frame["series"].unique().to_list() if s not in ordering]
        y = alt.Y("series:N", sort=order, title=None, axis=alt.Axis(labelFontSize=10))
        base = alt.Chart(frame)
        order_lines = [MANUAL, BEST_GUESS]
        colour = alt.Color(
            "line:N",
            scale=alt.Scale(domain=order_lines, range=[LINE_COLOURS[n] for n in order_lines]),
            legend=alt.Legend(title=None, orient="top", columns=2),
        )
        last = index == len(present) - 1
        rule = (
            alt.Chart(frame)
            .transform_aggregate(low="min(value)", high="max(value)", groupby=["series"])
            .mark_rule(color=ocf.ENSEMBLE_LINE)
            .encode(  # ty: ignore[unresolved-attribute]
                x="low:Q", x2="high:Q", y=y
            )
        )
        interval = (
            base.transform_filter(alt.datum.line == BEST_GUESS)
            .mark_rule(strokeWidth=1.5, color=ocf.BRAND_ORANGE, opacity=0.6)
            .encode(  # ty: ignore[unresolved-attribute]
                x="lower:Q", x2="upper:Q", y=y
            )
        )
        dots = base.mark_point(filled=True, size=60, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "value:Q",
                title=ERROR_TITLE if last else None,
                scale=alt.Scale(zero=True),
            ),
            y=y,
            color=colour,
            shape=alt.Shape(
                "line:N",
                scale=alt.Scale(domain=order_lines, range=["circle", "diamond"]),
                legend=None,
            ),
        )
        panels.append(
            alt.layer(rule, interval, dots).properties(
                width=PLOT_WIDTH - 20,
                height=max(18 * len(order), 40),
                title=_panel_title(text=CLASS_TITLES[target_class]),
            )
        )
    return (
        alt.vconcat(*panels)
        .resolve_scale(x="shared")
        .properties(
            title=_title(
                text="Error by substation and by farm: best guess against NGED's heuristic",
                subtitle=(
                    "Mean absolute error of the median forecast as a percentage of that series' "
                    f"capacity; smaller is better. {SCOPE} Orange diamond: the best "
                    "guess (BG2), with "
                    "its 95% interval from resampling whole months. Green dot: NGED's manual "
                    "heuristic. Generators are anonymised."
                ),
            )
        )
    )


# --- figure 7: tails ---------------------------------------------------------------------------


TAIL_ARMS_ORDER: Final[tuple[str, ...]] = (PRODUCTION, CLIMATOLOGY, MANUAL, PREVIOUS, BEST_GUESS)


def _arm_rows(*, frame: pl.DataFrame, target_class: str) -> pl.DataFrame:
    lines = headline_lines(target_class=target_class)
    parts = [
        frame.filter((pl.col("class") == target_class) & (pl.col("arm") == arm)).with_columns(
            line=pl.lit(label)
        )
        for arm, label in lines.items()
    ]
    return pl.concat(parts)


def figure_7a_reliability(*, tables: Tables) -> Any:
    """Draw observed against nominal coverage at the 11 levels, one panel per class."""
    panels = []
    for target_class in arms.CLASSES:
        rows = _arm_rows(frame=tables["reliability"], target_class=target_class)
        if rows.is_empty():
            continue
        base = alt.Chart(rows)
        x = alt.X(
            "nominal:Q",
            title="Share of outcomes the forecast promises to fall at or below the quantile",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(values=[0, 0.25, 0.5, 0.75, 1], format=".0%"),
        )
        diagonal = (
            alt.Chart(pl.DataFrame({"a": [0.0, 1.0]}))
            .mark_line(color=ocf.BLACK_1, strokeDash=[4, 3])
            .encode(  # ty: ignore[unresolved-attribute]
                x="a:Q", y="a:Q"
            )
        )
        band = base.mark_area(opacity=0.12, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=x, y="lower:Q", y2="upper:Q", color=_line_colour(order=TAIL_ARMS_ORDER)
        )
        line = base.mark_line(point=True, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y(
                "observed:Q",
                title="Share that actually did",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(values=[0, 0.25, 0.5, 0.75, 1], format=".0%"),
            ),
            color=_line_colour(order=TAIL_ARMS_ORDER),
            strokeDash=_line_dash(order=TAIL_ARMS_ORDER),
        )
        panels.append(
            alt.layer(diagonal, band, line).properties(
                width=PLOT_WIDTH - 140,
                height=PLOT_WIDTH - 140,
                title=_panel_title(text=CLASS_TITLES[target_class]),
            )
        )
    grid = (
        alt.vconcat(alt.hconcat(*panels[:2]), alt.hconcat(*panels[2:]))
        if len(panels) > 2
        else alt.hconcat(*panels)
    )
    return grid.resolve_legend(color="shared").properties(
        title=_title(
            text="Do the forecast ranges contain the outcomes as often as they promise?",
            subtitle=(
                "Each point is one of the 11 quantiles each forecast gives. On the dashed line "
                "the forecast is perfectly calibrated: a quantile promised to be exceeded 5% of "
                "the time was exceeded 5% of the time. Above the line, the quantile sits too "
                f"high; below it, too low. {SCOPE} Band: 95% interval from resampling whole months."
            ),
        )
    )


def figure_7b_tail_gaps(*, tables: Tables) -> Any:
    """Draw how far the observed share is from the promised share at 0.01, 0.05, 0.95, 0.99."""
    panels = []
    present = [c for c in arms.CLASSES if (tables["tails"]["class"] == c).any()]
    for index, target_class in enumerate(present):
        rows = (
            _arm_rows(frame=tables["tails"], target_class=target_class)
            .filter(pl.col("n_rows") > 0)
            .with_columns(
                level_label=pl.col("level").map_elements(
                    lambda v: f"{v:.0%}", return_dtype=pl.String
                ),
                gap_points=pl.col("gap") * 100,
                low=(pl.col("coverage_lower_95") - pl.col("level")) * 100,
                high=(pl.col("coverage_upper_95") - pl.col("level")) * 100,
            )
        )
        x = alt.X(
            "level_label:N",
            sort=[f"{v:.0%}" for v in TAIL_LEVELS],
            title="Quantile promised (share of outcomes at or below it)"
            if index == len(present) - 1
            else None,
            axis=alt.Axis(labelAngle=0),
        )
        offset = alt.XOffset("line:N", sort=list(TAIL_ARMS_ORDER))
        base = alt.Chart(rows)
        zero = (
            alt.Chart(pl.DataFrame({"zero": [0.0]}))
            .mark_rule(color=ocf.BLACK_1)
            .encode(  # ty: ignore[unresolved-attribute]
                y="zero:Q"
            )
        )
        rules = base.mark_rule(strokeWidth=1.5).encode(  # ty: ignore[unresolved-attribute]
            x=x,
            xOffset=offset,
            y="low:Q",
            y2="high:Q",
            color=_line_colour(order=TAIL_ARMS_ORDER),
        )
        dots = base.mark_point(filled=True, size=45, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=x,
            xOffset=offset,
            y=alt.Y("gap_points:Q", title="Observed minus promised (percentage points)"),
            color=_line_colour(order=TAIL_ARMS_ORDER),
        )
        panels.append(
            alt.layer(zero, rules, dots).properties(
                width=PLOT_WIDTH, height=110, title=_panel_title(text=CLASS_TITLES[target_class])
            )
        )
    return (
        alt.vconcat(*panels)
        .resolve_legend(color="shared")
        .properties(
            title=_title(
                text="Extreme quantiles: how far the forecast ranges miss their promise",
                subtitle=(
                    "Zero is a perfectly calibrated quantile. Above zero, outcomes fell at or "
                    "below the quantile more often than promised, so the quantile sits too "
                    "high; below zero it "
                    f"sits too low. {SCOPE} Line: 95% interval from resampling whole months."
                ),
            )
        )
    )


def figure_7c_pinball(*, tables: Tables) -> Any:
    """Draw the pinball-loss table at 0.01, 0.05, 0.95, and 0.99 as text, one block per class."""
    parts = []
    for target_class in arms.CLASSES:
        rows = _arm_rows(frame=tables["tails"], target_class=target_class).filter(
            pl.col("n_rows") > 0
        )
        parts.append(rows.with_columns(group=pl.lit(CLASS_TITLES[target_class])))
    data = pl.concat(parts).with_columns(
        level_label=pl.col("level").map_elements(lambda v: f"{v:.0%}", return_dtype=pl.String),
        text=pl.col("pinball").map_elements(lambda v: f"{v:.3f}", return_dtype=pl.String),
    )
    data = data.with_columns(
        best=(pl.col("pinball") == pl.col("pinball").min().over("group", "level"))
    )
    base = alt.Chart(data).encode(
        x=alt.X(
            "level_label:N",
            sort=[f"{v:.0%}" for v in TAIL_LEVELS],
            title="Quantile (share of outcomes at or below it)",
            axis=alt.Axis(orient="top", labelAngle=0),
        ),
        y=alt.Y(
            "line:N",
            sort=list(reversed(TAIL_ARMS_ORDER)),
            title=None,
            axis=alt.Axis(labelLimit=300),
        ),
    )
    text = base.mark_text(fontSize=11, aria=False).encode(  # ty: ignore[unresolved-attribute]
        text="text:N",
        color=alt.condition(alt.datum.best, alt.value(ocf.BRAND_ORANGE), alt.value(ocf.BLACK_1)),
    )
    return (
        text.properties(width=300, height=90)
        .facet(
            row=alt.Row(
                "group:N",
                sort=[CLASS_TITLES[c] for c in arms.CLASSES],
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=11),
            )
        )
        .properties(
            title=_title(
                text="Pinball loss in the tails",
                subtitle=(
                    "Pinball loss of the 1st, 5th, 95th, and 99th percentile forecasts, in points "
                    "of "
                    "capacity; smaller is better, and the smallest in each block is orange. "
                    f"{SCOPE}"
                ),
            )
        )
    )


def figure_7d_tail_scores(*, tables: Tables) -> Any:
    """Draw the threshold-weighted score above p99 and p95 and below p05, per class and arm."""
    parts = []
    for column, name, classes in TAIL_SCORES:
        for target_class in classes:
            rows = _arm_rows(
                frame=tables["absolute"].filter(pl.col("runs") == "gap"), target_class=target_class
            )
            if rows.is_empty():
                continue
            parts.append(
                rows.select(
                    "line",
                    value=pl.col(column),
                    lower=pl.col(f"{column}_lower"),
                    upper=pl.col(f"{column}_upper"),
                ).with_columns(group=pl.lit(CLASS_TITLES[target_class]), tail=pl.lit(name))
            )
    data = pl.concat(parts)
    tails = [name for _, name, _ in TAIL_SCORES]
    y = alt.Y(
        "group:N",
        sort=[CLASS_TITLES[c] for c in arms.CLASSES],
        title=None,
        axis=alt.Axis(labelFontSize=10),
    )
    offset = alt.YOffset("line:N", sort=list(TAIL_ARMS_ORDER))
    base = alt.Chart(data)
    rules = base.mark_rule(strokeWidth=1.5).encode(  # ty: ignore[unresolved-attribute]
        y=y,
        yOffset=offset,
        x="lower:Q",
        x2="upper:Q",
        color=_line_colour(order=TAIL_ARMS_ORDER),
    )
    dots = base.mark_point(filled=True, size=40, aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=y,
        yOffset=offset,
        x=alt.X(
            "value:Q",
            title="Threshold-weighted quantile score (points of capacity; smaller is better)",
            scale=alt.Scale(zero=True),
        ),
        color=_line_colour(order=TAIL_ARMS_ORDER),
    )
    return (
        alt.layer(rules, dots)
        .properties(width=PLOT_WIDTH - 60, height=120)
        .facet(
            row=alt.Row(
                "tail:N",
                sort=tails,
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=11),
            )
        )
        .resolve_scale(x="independent", y="independent")
        .properties(
            title=_title(
                text="Extreme-value accuracy: errors counted only beyond a series' usual range",
                subtitle=(
                    "A threshold-weighted score of the whole forecast range, counting only what "
                    "happens above the 99th or 95th percentile of the series' history (farm output "
                    "peaks and demand peaks) or below its 5th percentile (demand lows, which drive "
                    f"reverse flow). {SCOPE} Line: 95% interval from resampling whole months."
                ),
            )
        )
    )


# --- figure 8: curtailment ---------------------------------------------------------------------


def _curtailment_rows(*, tables: Tables, limit_kind: str, ratio: int) -> pl.DataFrame:
    return tables["curtailment_costs"].filter(
        (pl.col("limit_kind") == limit_kind) & (pl.col("breach_ratio") == ratio)
    )


def _cost_x(*, title: str | None) -> alt.X:
    return alt.X("vs_nged_gbp_per_mw_day:Q", title=title)


def _cost_layers(
    *,
    data: pl.DataFrame,
    y: alt.Y,
    x_title: str | None,
    colour: Any = alt.Undefined,
    shape: Any = alt.Undefined,
    y_offset: Any = alt.Undefined,
) -> Any:
    base = alt.Chart(data)
    zero = (
        alt.Chart(pl.DataFrame({"zero": [0.0]}))
        .mark_rule(color=ocf.BLACK_1)
        .encode(  # ty: ignore[unresolved-attribute]
            x="zero:Q"
        )
    )
    rules = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x="vs_nged_gbp_per_mw_day_lower:Q",
        x2="vs_nged_gbp_per_mw_day_upper:Q",
        y=y,
        yOffset=y_offset,
        color=colour,
    )
    dots = base.mark_point(filled=True, size=70, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=_cost_x(title=x_title), y=y, yOffset=y_offset, color=colour, shape=shape
    )
    return alt.layer(zero, rules, dots)


COST_X_TITLE: Final[str] = (
    "Cost minus NGED's method (GBP per MW of farm capacity per day; below zero is cheaper)"
)


def figure_8_curtailment(*, tables: Tables) -> Any:
    """Draw the curtailment cost against NGED's method: arms, annual totals, and two sweeps."""
    headline = _curtailment_rows(tables=tables, limit_kind=HEADLINE_LIMIT, ratio=HEADLINE_RATIO)
    headline = headline.filter(pl.col("arm").is_in(list(CURTAILMENT_ARM_ORDER)))
    order_labels = [
        headline.filter(pl.col("arm") == a)["label"][0]
        for a in CURTAILMENT_ARM_ORDER
        if (headline["arm"] == a).any()
    ]
    events = int(headline["scored_events"][0])
    half_hours = int(headline["scored_half_hours"][0])
    annual = headline.with_columns(
        text=pl.format(
            "GBP {} per MW a year ({} to {})",
            pl.col("vs_nged_gbp_per_mw_year").round(0).cast(pl.Int64),
            pl.col("vs_nged_gbp_per_mw_year_lower").round(0).cast(pl.Int64),
            pl.col("vs_nged_gbp_per_mw_year_upper").round(0).cast(pl.Int64),
        )
    )
    y = alt.Y(
        "label:N", sort=order_labels, title=None, axis=alt.Axis(labelLimit=380, labelFontSize=10)
    )
    main = _cost_layers(
        data=annual, y=y, x_title=COST_X_TITLE, colour=alt.value(ocf.BRAND_ORANGE)
    ).properties(
        width=PLOT_WIDTH - 220,
        height=26 * len(order_labels),
        title=_panel_title(
            text=f"Headline: limit at 0.9 x the past maximum, risk ratio {HEADLINE_RATIO}",
            subtitle=f"Events: {events} of {half_hours:,} scored half-hours were beyond the limit.",
        ),
    )
    totals = (
        alt.Chart(annual)
        .mark_text(align="left", fontSize=10, aria=False)
        .encode(y=alt.Y("label:N", sort=order_labels, axis=None), text="text:N", x=alt.value(0))  # ty: ignore[unresolved-attribute]
        .properties(
            width=170, height=26 * len(order_labels), title=_panel_title(text="Annual total")
        )
    )
    panel_a = alt.hconcat(main, totals, spacing=4)

    sweep_arms = ["bg2", "previous_best", "xgboost_baseline"]
    arm_colours = [ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE]
    # Panel b: the limit fraction.
    limit_parts = []
    for kind, name in LIMIT_NAMES.items():
        rows = _curtailment_rows(tables=tables, limit_kind=kind, ratio=HEADLINE_RATIO)
        if rows.is_empty():
            continue
        count = int(rows["scored_events"][0])
        limit_parts.append(
            rows.filter(pl.col("arm").is_in(sweep_arms)).with_columns(
                row=pl.lit(f"{name}: {count} events")
            )
        )
    limit_data = pl.concat(limit_parts).with_columns(label=pl.col("label"))
    limit_rows = limit_data["row"].unique(maintain_order=True).to_list()
    colour_b = alt.Color(
        "label:N",
        scale=alt.Scale(
            domain=[limit_data.filter(pl.col("arm") == a)["label"][0] for a in sweep_arms],
            range=arm_colours,
        ),
        legend=alt.Legend(title=None, orient="top", columns=3, labelLimit=420),
    )
    y_b = alt.Y(
        "row:N", sort=limit_rows, title=None, axis=alt.Axis(labelLimit=380, labelFontSize=10)
    )
    panel_b = _cost_layers(
        data=limit_data,
        y=y_b,
        x_title=COST_X_TITLE,
        colour=colour_b,
        y_offset=alt.YOffset("label:N"),
    ).properties(
        width=PLOT_WIDTH - 100,
        height=40 * len(limit_rows),
        title=_panel_title(
            text="Sweep of the limit",
            subtitle=(
                "The published thermal limit is a reference only; NGED does not operate to "
                "it. Each row prints its event count."
            ),
        ),
    )
    # Panel c: the breach ratio.
    ratio_parts = []
    for ratio in sorted(tables["curtailment_costs"]["breach_ratio"].unique().to_list()):
        rows = _curtailment_rows(tables=tables, limit_kind=HEADLINE_LIMIT, ratio=int(ratio))
        if rows.is_empty():
            continue
        count = int(rows["scored_events"][0])
        ratio_parts.append(
            rows.filter(pl.col("arm").is_in(sweep_arms)).with_columns(
                row=pl.lit(f"Risk ratio {int(ratio)} (tau {1 - 1 / ratio:.2f}): {count} events"),
                order=pl.lit(int(ratio)),
            )
        )
    ratio_data = pl.concat(ratio_parts)
    ratio_rows = ratio_data.sort("order")["row"].unique(maintain_order=True).to_list()
    y_c = alt.Y(
        "row:N", sort=ratio_rows, title=None, axis=alt.Axis(labelLimit=380, labelFontSize=10)
    )
    panel_c = _cost_layers(
        data=ratio_data,
        y=y_c,
        x_title=COST_X_TITLE,
        colour=colour_b,
        y_offset=alt.YOffset("label:N"),
    ).properties(
        width=PLOT_WIDTH - 100,
        height=40 * len(ratio_rows),
        title=_panel_title(
            text="Sweep of the assumed risk",
            subtitle=(
                "Risk ratio r: one MWh over the limit counts as r MWh of curtailed output. "
                f"The headline uses r = {HEADLINE_RATIO}."
            ),
        ),
    )
    panels: list[Any] = [panel_a, panel_b, panel_c]
    if _has(tables=tables, names=["primary_census"]):
        panels.append(_primary_census_panel(census=tables["primary_census"]))
    return (
        alt.vconcat(*panels)
        .resolve_scale(color="shared")
        .properties(
            title=_title(
                text=f"Curtailment cost compared with NGED's current method ({CURTAILMENT_LABEL})",
                subtitle=(
                    "Cost is GBP 2,000 per MWh of farm output cut plus the risk ratio times the "
                    "MWh still over the limit, per MW of farm capacity beneath the bulk and grid "
                    "supply points, per day, July 2025 to June 2026. Dot: estimate. Line: 95% "
                    "interval from resampling whole months. Below zero is a lower cost than NGED's "
                    "method as we understand it (unconfirmed). This is a fit to history, not a "
                    "saving."
                ),
            )
        )
    )


def _primary_census_panel(*, census: pl.DataFrame) -> Any:
    totals = (
        census.group_by("fraction")
        .agg(events=pl.col("events").sum(), half_hours=pl.col("half_hours").sum())
        .sort("fraction", descending=True)
        .with_columns(
            row=pl.format("Limit at {} x past maximum", pl.col("fraction")),
            text=pl.format(
                "{} events in {} half-hours",
                pl.col("events"),
                pl.col("half_hours").map_elements(lambda v: f"{v:,}", return_dtype=pl.String),
            ),
        )
    )
    rows = totals["row"].to_list()
    return (
        alt.Chart(totals)
        .mark_text(align="left", fontSize=11, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(
                "row:N", sort=rows, title=None, axis=alt.Axis(labelFontSize=10, labelLimit=250)
            ),
            text="text:N",
            x=alt.value(8),
        )
        .properties(
            width=PLOT_WIDTH - 60,
            height=22 * len(rows),
            title=_panel_title(
                text="Primary substations: event counts only, summed over all Primaries",
                subtitle="No farm sits beneath a Primary, so no cost is computed.",
            ),
        )
    )


# --- figure 9: side panels ---------------------------------------------------------------------


def _side_panel(
    *, rows: Sequence[tuple[str, float, float, float]], title: str, subtitle: str, x_title: str
) -> Any:
    data = pl.DataFrame(
        {
            "row": [_wrap_label(text=r[0], width=40) for r in rows],
            "difference": [r[1] for r in rows],
            "lower": [r[2] for r in rows],
            "upper": [r[3] for r in rows],
        }
    )
    order = data["row"].to_list()
    y = alt.Y(
        "row:N",
        sort=order,
        title=None,
        axis=alt.Axis(labelExpr=SPLIT_LABEL, labelLimit=250, labelFontSize=10),
    )
    base = alt.Chart(data)
    zero = (
        alt.Chart(pl.DataFrame({"zero": [0.0]}))
        .mark_rule(color=ocf.BLACK_1)
        .encode(  # ty: ignore[unresolved-attribute]
            x="zero:Q"
        )
    )
    rules = base.mark_rule(strokeWidth=2, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x="lower:Q", x2="upper:Q", y=y
    )
    dots = base.mark_point(filled=True, size=80, color=ocf.DATA_BLUE, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("difference:Q", title=x_title), y=y
    )
    return alt.layer(zero, rules, dots).properties(
        width=PLOT_WIDTH - 60,
        height=32 * len(order),
        title=_panel_title(text=title, subtitle=subtitle),
    )


def side_pages_exist() -> tuple[bool, bool]:
    """Return whether experiment 81's and experiment 54's result pages exist."""
    return (
        any(EXPERIMENT_81_PAGE.glob("00081-*/result.md")),
        any(EXPERIMENT_54_PAGE.glob("00054-*/result.md")),
    )


def figure_9_side_panels(*, use_81: bool, use_54: bool) -> Any:
    """Draw experiment 81's control-only contrasts and experiment 54's UKV ceiling."""
    panels = []
    if use_81:
        panels.append(
            _side_panel(
                rows=SIDE_81,
                title="Experiment 81: control-member runs only, 13 Primary substations",
                subtitle=(
                    "Scored on a different set of runs (one ensemble member, July 2023 to March "
                    "2024), so never compare these numbers with the 51-member scores elsewhere."
                    " Line: 99.17% interval, as quoted on the experiment's page."
                ),
                x_title="Quantile score difference (points of capacity; below zero is better)",
            )
        )
    if use_54:
        panels.append(
            _side_panel(
                rows=SIDE_54,
                title="Experiment 54: UKV for 3 wind farms - a ceiling, not available live",
                subtitle=(
                    "Lead times 24 to 54 hours. UKV comes from a research archive (CEDA), not the "
                    "live feed, so this is an upper limit. Line: 95% interval."
                ),
                x_title="Quantile score difference (points of capacity; below zero is better)",
            )
        )
    return (
        alt.vconcat(*panels)
        .resolve_scale(x="independent")
        .properties(
            title=_title(
                text="Side results from earlier experiments, on their own rows",
                subtitle=(
                    "These are not scores of the best guess. Dot: estimate of the first named "
                    "forecast minus the second; smaller is better."
                ),
            )
        )
    )


# --- fixture -----------------------------------------------------------------------------------


CLASS_LEVEL: Final[dict[str, float]] = {
    "Primary": 4.0,
    "BSP/GSP": 6.0,
    "Solar farms": 7.0,
    "Wind farms": 9.0,
}
FIXTURE_SERIES: Final[dict[str, list[str]]] = {
    "Primary": [f"D{i}" for i in range(1, 17)],
    "BSP/GSP": [f"D{i}" for i in range(17, 21)],
    "Solar farms": list("ABCDEF"),
    "Wind farms": ["W1", "W2", "W3"],
}


def _arm_factor(*, target_class: str, arm: str) -> float:
    """Return a made-up error multiplier that puts the best guess lowest and references highest."""
    if arm == "ref_xgboost_baseline":
        return 1.45
    if arm == "ref_climatology":
        return 1.6
    if arm == "ref_manual_heuristic":
        return 1.3
    if arm == arms.BG2_OF_CLASS[target_class]:
        return 1.0
    if arm in arms.STAIRCASE[target_class]:
        return 1.0 + 0.05 * (
            len(arms.STAIRCASE[target_class]) - arms.STAIRCASE[target_class].index(arm)
        )
    names = arms.ABLATIONS[target_class]
    index = names.index(arm) if arm in names else 0
    return 1.02 + 0.03 * ((index * 5) % 7 - 2)


def fixture_tables() -> Tables:  # noqa: PLR0912 (one flat loop over classes and arms)
    """Return synthetic tables with the schema of the real results tables."""
    rng = np.random.default_rng(87)
    levels = [float(v) for v in common.LEVELS]
    band_names = [name for name, _, _ in common.LEAD_BANDS]
    scope_rows_: list[dict[str, object]] = []
    absolute: list[dict[str, object]] = []
    staircase: list[dict[str, object]] = []
    contrasts: list[dict[str, object]] = []
    reliability: list[dict[str, object]] = []
    tails: list[dict[str, object]] = []

    def interval(value: float, width: float) -> tuple[float, float]:
        return value * (1 - width), value * (1 + width)

    for target_class in arms.CLASSES:
        names = {
            arms.resolve(name=a)
            for a in (
                *arms.STAIRCASE[target_class],
                *arms.ABLATIONS[target_class],
                *arms.REFERENCES,
            )
        }
        for arm in sorted(names):
            factor = _arm_factor(target_class=target_class, arm=arm)
            level = CLASS_LEVEL[target_class] * factor
            for metric, scale in (("mae", 1.0), ("qs", 1.25)):
                for dimension, labels, shape in _fixture_dimensions(
                    target_class=target_class, band_names=band_names
                ):
                    if dimension == "lead_day_midday" and target_class != "Primary":
                        continue
                    for index, label in enumerate(labels):
                        value = level * scale * shape(index) * float(rng.uniform(0.97, 1.03))
                        if dimension == "origin_month":
                            low = high = float("nan")
                        else:
                            low, high = interval(value, 0.05 + 0.002 * index)
                        scope_rows_.append(
                            {
                                "class": target_class,
                                "arm": arm,
                                "dimension": dimension,
                                "scope": label,
                                "metric": metric,
                                "value": value,
                                "lower": low,
                                "upper": high,
                                "cases": 1000,
                            }
                        )
            row: dict[str, object] = {
                "class": target_class,
                "arm": arm,
                "runs": "gap",
                "cases": 10_000,
                "qs": level * 1.25,
                "qs_lower": level * 1.25 * 0.95,
                "qs_upper": level * 1.25 * 1.05,
                "mae": level,
                "mae_lower": level * 0.95,
                "mae_upper": level * 1.05,
            }
            for column, share in (("tw_p99", 0.05), ("tw_p95", 0.12), ("tw_p05", 0.1)):
                value = level * share * float(rng.uniform(0.9, 1.1))
                row[column] = value
                row[f"{column}_lower"] = value * 0.85
                row[f"{column}_upper"] = value * 1.15
            absolute.append(row)
            for nominal in levels:
                bias = float(rng.normal(0, 0.01)) + (0.015 if nominal > 0.9 else -0.01) * factor
                observed = min(max(nominal + bias, 0.001), 0.999)
                reliability.append(
                    {
                        "class": target_class,
                        "arm": arm,
                        "nominal": nominal,
                        "observed": observed,
                        "lower": observed - 0.02,
                        "upper": observed + 0.02,
                        "cases": 10_000,
                    }
                )
            for tail in (0.01, 0.05, 0.95, 0.99):
                coverage = tail + float(rng.normal(0, 0.01)) * factor
                tails.append(
                    {
                        "class": target_class,
                        "arm": arm,
                        "level": tail,
                        "n_rows": 10_000,
                        "coverage": coverage,
                        "coverage_lower_95": coverage - 0.012,
                        "coverage_upper_95": coverage + 0.012,
                        "gap": coverage - tail,
                        "pinball": 0.2 * level * factor * (1 + abs(tail - 0.5)) / 4,
                        "pinball_lower_95": 0.19 * level * factor / 4,
                        "pinball_upper_95": 0.21 * level * factor * 1.5 / 4,
                    }
                )
        qs = {row["arm"]: float(row["qs"]) for row in absolute if row["class"] == target_class}  # ty: ignore[invalid-argument-type]
        previous = None
        for position, arm in enumerate(arms.STAIRCASE[target_class]):
            value = qs[arms.resolve(name=arm)]
            record: dict[str, object] = {
                "class": target_class,
                "position": position,
                "arm": arm,
                "label": arms.arm_label(name=arm),
                "qs": value,
                "qs_lower": value * 0.95,
                "qs_upper": value * 1.05,
                "mae": value / 1.25,
                "mae_lower": value / 1.25 * 0.95,
                "mae_upper": value / 1.25 * 1.05,
                "cases": 10_000,
            }
            if previous is not None:
                step = value - qs[arms.resolve(name=previous)]
                record |= {
                    "step_from": previous,
                    "step": step,
                    "step_lower": step - 0.15,
                    "step_upper": step + 0.15,
                }
            staircase.append(record)
            if arm != "g_a_wind_margin":
                previous = arm
        bg2 = qs[arms.resolve(name=arms.BG2_OF_CLASS[target_class])]
        for arm in arms.ABLATIONS[target_class]:
            difference = qs[arms.resolve(name=arm)] - bg2
            contrasts.append(
                {
                    "family": "ablation",
                    "id": f"{arm} minus BG2",
                    "class": target_class,
                    "treatment": arm,
                    "reference": arms.BG2_OF_CLASS[target_class],
                    "metric": "qs",
                    "level": 95.0,
                    "runs": "gap",
                    "difference": difference,
                    "lower": difference - 0.12,
                    "upper": difference + 0.12,
                    "cases": 10_000,
                    "months": 12,
                }
            )
    return {
        "absolute": pl.DataFrame(absolute, infer_schema_length=None),
        "scopes": pl.DataFrame(scope_rows_, infer_schema_length=None),
        "staircase": pl.DataFrame(staircase, infer_schema_length=None),
        "contrasts": pl.DataFrame(contrasts, infer_schema_length=None),
        "reliability": pl.DataFrame(reliability),
        "tails": pl.DataFrame(tails),
        "noise_floor": pl.DataFrame(
            [
                {
                    "metric": "qs",
                    "series": 4,
                    "cases": 4000,
                    "first_fit": 5.0,
                    "repeat_fit": 5.02,
                    "difference": 0.02,
                    "lower": 0.0,
                    "upper": 0.04,
                }
            ]
        ),
        **_fixture_curtailment(rng=rng),
    }


def _fixture_dimensions(
    *, target_class: str, band_names: list[str]
) -> list[tuple[str, list[str], Callable[[int], float]]]:
    months = [f"2025-{m:02d}" for m in range(7, 13)] + [f"2026-{m:02d}" for m in range(1, 7)]
    return [
        ("lead_day", [str(d) for d in range(15)], lambda i: 0.8 + 0.035 * i),
        ("lead_day_midday", [str(d) for d in range(15)], lambda i: 1.0 + 0.04 * i),
        ("lead_band", band_names, lambda i: 0.8 + 0.1 * i),
        (
            "hour_uk",
            [f"{h:02d}" for h in range(24)],
            lambda i: (
                (0.05 if target_class == "Solar farms" and (i < 5 or i > 19) else 1.0)
                * (0.8 + 0.4 * abs(np.sin(i / 24 * np.pi)))
            ),
        ),
        ("origin_month", months, lambda i: 1.0 + 0.2 * np.cos(i / 2)),
        ("season", list(SEASON_ORDER), lambda i: 1.0 + 0.1 * i),
        ("sun", ["cloudy", "middle", "sunny"], lambda i: 1.2 - 0.2 * i),
        (
            "series",
            FIXTURE_SERIES[target_class],
            lambda i: 0.7 + 0.6 * ((i * 7) % 11) / 11,
        ),
    ]


def _fixture_curtailment(*, rng: np.random.Generator) -> Tables:
    labels = {
        "nged_method": "NGED's method (unconfirmed)",
        "previous_best": "Previous best (stack and weather-trained generators)",
        "frozen": "Frozen model",
        "monthly": "Refitted monthly",
        "bg2": "Best guess (refitted weekly)",
        "bg2_nodes_capacity_generators": "Best-guess nodes, generators at capacity",
        "nged_nodes_bg2_generators": "NGED nodes, best-guess generators",
        "climatology": "Climatology",
        "xgboost_baseline": "Production-style XGBoost model",
        "perfect": "Perfect forecast",
    }
    rows = []
    scenarios = [("real", 20), *[(HEADLINE_LIMIT, r) for r in (4, 10, 20, 100)]]
    scenarios += [("scaled f=0.75", 20), ("scaled f=0.5", 20)]
    for kind, ratio in scenarios:
        events = {"real": 0, HEADLINE_LIMIT: 19, "scaled f=0.75": 61, "scaled f=0.5": 340}[kind]
        for index, (arm, label) in enumerate(labels.items()):
            centre = 0.0 if arm == "nged_method" else (-3.0 + 0.7 * index) * ratio / 20
            row: dict[str, object] = {
                "limit_kind": kind,
                "breach_ratio": ratio,
                "arm": arm,
                "label": label,
                "scored_events": events,
                "scored_half_hours": 5000,
                "forecast_capacity_mw": 100.0,
                "days": 292,
                "note": CURTAILMENT_LABEL,
            }
            spread = 0.0 if arm == "nged_method" else 2.0 + float(rng.uniform(0, 1))
            for window, scale in (("day", 1.0), ("year", 365.0)):
                row[f"vs_nged_gbp_per_mw_{window}"] = centre * scale
                row[f"vs_nged_gbp_per_mw_{window}_lower"] = (centre - spread) * scale
                row[f"vs_nged_gbp_per_mw_{window}_upper"] = (centre + spread) * scale
            rows.append(row)
    census = [
        {
            "fraction": f,
            "site": f"D{i}",
            "half_hours": 17_000,
            "events": int(100 / (1 + 4 * i) / f),
            "limit_mw": 5.0,
        }
        for f in (0.9, 0.75, 0.5)
        for i in range(1, 4)
    ]
    return {
        "curtailment_costs": pl.DataFrame(rows),
        "primary_census": pl.DataFrame(census),
    }


# --- writing -----------------------------------------------------------------------------------


def figures(*, tables: Tables) -> dict[str, Callable[[], alt.TopLevelMixin]]:
    """Return a drawing function per output file, for the figures the tables can support."""
    out: dict[str, Callable[[], alt.TopLevelMixin]] = {}
    if _has(tables=tables, names=["scopes"]):
        out["figure_1_headline"] = lambda: figure_1_headline(tables=tables)
        for panel in PANELS:
            out[f"figure_2_{panel[0]}_stages"] = lambda panel=panel: figure_2_stages(
                tables=tables, panel=panel
            )
        for target_class in arms.CLASSES:
            slug = target_class.replace("/", "_").replace(" ", "_").lower()
            out[f"figure_5_{slug}_breakdown"] = lambda c=target_class: figure_5_breakdown(
                tables=tables, target_class=c
            )
        out["figure_6_by_series"] = lambda: figure_6_by_series(tables=tables)
    if _has(tables=tables, names=["staircase"]):
        out["figure_3_staircase"] = lambda: figure_3_staircase(tables=tables)
    if _has(tables=tables, names=["contrasts"]):
        out["figure_4_ablations"] = lambda: figure_4_ablations(tables=tables)
    if _has(tables=tables, names=["reliability"]):
        out["figure_7a_reliability"] = lambda: figure_7a_reliability(tables=tables)
    if _has(tables=tables, names=["tails"]):
        out["figure_7b_tail_gaps"] = lambda: figure_7b_tail_gaps(tables=tables)
        out["figure_7c_pinball"] = lambda: figure_7c_pinball(tables=tables)
    if _has(tables=tables, names=["absolute", "tails"]):
        out["figure_7d_tail_scores"] = lambda: figure_7d_tail_scores(tables=tables)
    if _has(tables=tables, names=["curtailment_costs"]):
        out["figure_8_curtailment"] = lambda: figure_8_curtailment(tables=tables)
    use_81, use_54 = side_pages_exist()
    if use_81 or use_54:
        out["figure_9_side_panels"] = lambda: figure_9_side_panels(use_81=use_81, use_54=use_54)
    return out


def write_all(*, tables: Tables, svg_dir: Path, png_dir: Path, optimise: bool) -> None:
    """Write every figure as SVG and PNG, then optimise the SVGs if asked."""
    svg_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    for name, draw in figures(tables=tables).items():
        chart = draw().configure_view(stroke=None)
        chart = chart.properties(padding=8)
        chart.save(svg_dir / f"{name}.svg")
        chart.save(png_dir / f"{name}.png", scale_factor=2)
        if optimise:
            subprocess.run(
                [
                    "npx",
                    "svgo@4",
                    "--multipass",
                    "--precision=1",
                    "--final-newline",
                    str(svg_dir / f"{name}.svg"),
                ],
                check=True,
                capture_output=True,
            )
        print("wrote", name, flush=True)


def main() -> None:
    """Draw the figures from the real tables, or from synthetic ones with `--fixture`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="store_true", help="draw from synthetic tables")
    parser.add_argument("--no-svgo", action="store_true", help="skip the SVG optimiser")
    parser.add_argument("--results-dir", type=Path, default=common.RESULTS_DIR)
    args = parser.parse_args()
    if args.fixture:
        write_all(
            tables=fixture_tables(),
            svg_dir=FIXTURE_DIR,
            png_dir=FIXTURE_DIR,
            optimise=False,
        )
        return
    tables = read_tables(results_dir=args.results_dir)
    if not tables:
        msg = f"no results tables under {args.results_dir}; run e00087_score.py first"
        raise FileNotFoundError(msg)
    write_all(tables=tables, svg_dir=DOCS_DIR, png_dir=CHARTS_DIR, optimise=not args.no_svgo)


if __name__ == "__main__":
    main()
