"""Draw experiment 00092's figures from the saved results, as SVG.

1. Drift slope by hour band and season, pooled over series, with intervals (headline).
2. Monthly level of the residual per series, with the persistent steps found by the step census.
3. The step census by tag.
4. Loss from the extra years of history against each series' level change (D4).
5. Pooled-minus-own bias against the series' level (D5).
6. Weekly-minus-monthly refit gap in flagged and unflagged series-months (D6).

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    cd ml-experiments/00092-lv0-drift-diagnostics-on-stored-predictions-demand
    uv run python e00092_charts.py
"""

import json
import textwrap
from datetime import timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00092_diagnostics as diag
import plotting.ocf_theme as ocf
import polars as pl

WIDTH: Final[int] = 540
CHART_DIR: Final[Path] = Path(__file__).resolve().parents[2] / (
    "docs/ml-experiments/00092-lv0-drift-diagnostics-on-stored-predictions-demand"
)
RESULTS: Final[Path] = diag.RESULTS
SCOPE: Final[str] = (
    "13 Primary substations with at least 24 months of data, January 2022 to June 2026. "
    "Residual: observed power minus experiment 17's model frozen at its first origin, ERA5 weather."
)


def figure_1() -> alt.TopLevelMixin:
    """Slope of the residual by band and season."""
    data = pl.read_parquet(RESULTS / "d1_band_slopes.parquet").with_columns(
        season=pl.when(pl.col("group").str.contains("Apr-Sep"))
        .then(pl.lit("Apr-Sep"))
        .otherwise(pl.lit("Oct-Mar")),
    )
    data = data.filter(pl.col("group") != "all hours")
    order = data["group"].to_list()
    base = alt.Chart(data).encode(
        y=alt.Y("group:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
        color=alt.Color(
            "season:N",
            scale=alt.Scale(domain=["Apr-Sep", "Oct-Mar"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
            legend=alt.Legend(orient="top", title=None),
        ),
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#555555").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "median_slope:Q",
            title="Drift of the residual (points of capacity per year; 0 is no drift)",
        )
    )
    bars = base.mark_rule().encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=WIDTH,
        height=230,
        title=alt.TitleParams(
            "Summer midday and evening load fell, and night load rose, against a frozen model",
            subtitle=[
                SCOPE,
                "Dot: median over series of each series' straight-line slope of monthly residual.",
                "Line: 95% interval from re-drawing whole months.",
                "Positive: demand above the frozen forecast.",
            ],
            anchor="start",
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """Heat map of monthly residual per series with steps marked."""
    residual = diag.load_residual()
    monthly = residual.group_by("time_series_id", "month_idx").agg(level=pl.col("r").mean())
    monthly = monthly.with_columns(
        level=pl.col("level") - pl.col("level").mean().over("time_series_id")
    )
    steps = pl.read_parquet(RESULTS / "d2_steps.parquet").filter(~pl.col("night_only"))
    steps = steps.with_columns(
        month_idx=pl.Series(
            [
                (d := diag.GRID_START + timedelta(weeks=w)).year * 12 + d.month - 1 - 2022 * 12
                for w in steps["week"].to_list()
            ],
            dtype=pl.Int64,
        )
    )
    order = sorted(monthly["time_series_id"].unique().to_list())
    heat = (
        alt.Chart(monthly)
        .mark_rect()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "month_idx:O",
                title="Months since January 2022",
                axis=alt.Axis(values=[0, 12, 24, 36, 48]),
            ),
            y=alt.Y("time_series_id:O", sort=order, title="Primary substation (anonymous index)"),
            color=alt.Color(
                "level:Q",
                scale=alt.Scale(
                    domain=[-15, 0, 15],
                    range=[ocf.DATA_BLUE, "#FFFFFF", ocf.BRAND_ORANGE],
                    clamp=True,
                ),
                legend=alt.Legend(
                    orient="top",
                    title="Monthly residual minus own mean (points)",
                ),
            ),
        )
    )
    marks = (
        alt.Chart(steps)
        .mark_point(shape="diamond", color="#111111", filled=True, size=40)
        .encode(x="month_idx:O", y=alt.Y("time_series_id:O", sort=order))  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(heat, marks).properties(
        width=WIDTH,
        height=260,
        title=alt.TitleParams(
            "Monthly residuals swing with the weather, and a few series also shift level for years",
            subtitle=[
                SCOPE,
                "Black diamond: a persistent step of 2 points or more found in the weekly mean.",
            ],
            anchor="start",
        ),
    )


def figure_3() -> alt.TopLevelMixin:
    """Step census by tag."""
    steps = pl.read_parquet(RESULTS / "d2_steps.parquet")
    data = (
        steps.group_by("night_only", "tag")
        .len()
        .with_columns(
            hours=pl.when(pl.col("night_only"))
            .then(pl.lit("Night only (00-06)"))
            .otherwise(pl.lit("All hours"))
        )
    )
    tags = [
        "none",
        "matches an RO-NC alarm",
        "opposite step at a neighbour",
        "common to most series",
    ]
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("tag:N", sort=tags, title=None, axis=alt.Axis(labelLimit=300)),
            x=alt.X("len:Q", title="Persistent steps of 2 points or more (count)"),
            yOffset="hours:N",
            color=alt.Color(
                "hours:N",
                scale=alt.Scale(
                    domain=["All hours", "Night only (00-06)"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=alt.TitleParams(
                "Half of the steps match no switching alarm and no neighbour's opposite step",
                subtitle=[
                    SCOPE,
                    "Steps found in the weekly mean residual.",
                    "RO-NC is the refit-offset neighbour-count detector.",
                    "The neighbour tag is loose: any opposite step elsewhere within 4 weeks.",
                ],
                anchor="start",
            ),
        )
    )


def _scatter(
    *,
    data: pl.DataFrame,
    x: str,
    y: str,
    x_title: str,
    y_title: str,
    title: str,
    subtitle: list[str],
) -> alt.TopLevelMixin:
    points = (
        alt.Chart(data)
        .mark_point(filled=True, size=60, color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(f"{x}:Q", title=x_title), y=alt.Y(f"{y}:Q", title=y_title)
        )
    )
    zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color="#555555").encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, points).properties(
        width=WIDTH, height=200, title=alt.TitleParams(title, subtitle=subtitle, anchor="start")
    )


def figure_4() -> alt.TopLevelMixin:
    """Extra-years loss against level change."""
    a = pl.read_parquet(RESULTS / "d4_exp17.parquet").with_columns(
        source=pl.lit("Experiment 17: all history minus 12-month window")
    )
    b = (
        pl.read_parquet(RESULTS / "d4_exp89.parquet")
        .select("time_series_id", "level_change", "loss")
        .with_columns(source=pl.lit("Experiment 89: longest history minus 12 months"))
    )
    data = pl.concat(
        [
            a.select("time_series_id", "level_change", "loss", "source"),
            b.select("time_series_id", "level_change", "loss", "source"),
        ]
    )
    zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color="#555555").encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    panels = []
    for source in data["source"].unique(maintain_order=True):
        points = (
            alt.Chart(data.filter(pl.col("source") == source))
            .mark_point(filled=True, size=60, color=ocf.DATA_BLUE)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "level_change:Q",
                    title="Level change, first to last 12 months (points of capacity)",
                ),
                y=alt.Y("loss:Q", title="Extra history minus 12 months (points; above 0: worse)"),
            )
        )
        panels.append(
            alt.layer(zero, points).properties(
                width=WIDTH, height=130, title=alt.TitleParams(source, anchor="start", fontSize=12)
            )
        )
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Loss from extra history does not follow a series' drift, outside one series",
            subtitle=[
                SCOPE,
                "One dot per series. The outlier on the right has a 30-point level change.",
            ],
            anchor="start",
        )
    )


def figure_5() -> alt.TopLevelMixin:
    """Pooled-minus-own bias against level."""
    data = pl.read_parquet(RESULTS / "d5_bias_vs_level.parquet")
    points = (
        alt.Chart(data)
        .mark_point(filled=True, size=20, opacity=0.5, color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "level:Q",
                title="Monthly residual minus the series' own mean (points)",
            ),
            y=alt.Y("bias_diff:Q", title="Pooled model's bias minus own model's bias (points)"),
        )
    )
    return points.properties(
        width=WIDTH,
        height=220,
        title=alt.TitleParams(
            "A pooled model's extra bias does not follow each series' level",
            subtitle=[
                SCOPE,
                "One dot per series and month, runs of July 2025 to June 2026 (experiment 24).",
            ],
            anchor="start",
        ),
    )


def figure_6() -> alt.TopLevelMixin:
    """Weekly-minus-monthly refit gap by flag."""
    summary = json.loads((RESULTS / "summary.json").read_text())["d6"]
    labels = {
        "step": "A persistent step in the month",
        "steep": "Series in the steepest quarter of drift",
        "event": "A switching alarm in the month",
        "any_flag": "Any of the four",
    }
    rows = [
        {
            "flag": labels[k],
            "diff": summary[k]["mean_gap_flagged_minus_unflagged"],
            "lo": summary[k]["mean_gap_diff_ci"][0],
            "hi": summary[k]["mean_gap_diff_ci"][1],
        }
        for k in labels
    ]
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("flag:N", sort=list(labels.values()), title=None, axis=alt.Axis(labelLimit=330))
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#555555").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    points = base.mark_point(filled=True, size=60, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "diff:Q",
            title="Weekly refit's gain, flagged minus unflagged (points of capacity)",
        )
    )
    bars = base.mark_rule(color=ocf.DATA_BLUE).encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=WIDTH,
        height=170,
        title=alt.TitleParams(
            "Weekly refits gain more in the steepest-drifting series, by a small amount",
            subtitle=[
                "13 Primaries, 156 series-months, July 2025 to June 2026 (experiment 48)",
                "Dot: mean gap in flagged minus unflagged months.",
                "Line: 95% interval from re-drawing whole months.",
                "Positive: weekly refits gain more where the flag is true. Zero: no difference.",
                "Intervals re-draw months only, with the series fixed.",
            ],
            anchor="start",
        ),
    )


def wrapped(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """Wrap the title and subtitle lines to the plot area's width."""
    params = chart.title  # ty: ignore[unresolved-attribute]
    text = params.text if isinstance(params.text, str) else params.text[0]
    sub = [w for line in params.subtitle for w in textwrap.wrap(line, 95)]
    chart.title = alt.TitleParams(textwrap.wrap(text, 70), subtitle=sub, anchor="start")  # ty: ignore[unresolved-attribute]
    return chart


def main() -> None:
    """Write the figures."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_1", figure_1()),
        ("figure_2", figure_2()),
        ("figure_3", figure_3()),
        ("figure_4", figure_4()),
        ("figure_5", figure_5()),
        ("figure_6", figure_6()),
    ):
        wrapped(chart).save(CHART_DIR / f"{name}.svg")
        print(f"wrote {name}.svg", flush=True)


if __name__ == "__main__":
    main()
