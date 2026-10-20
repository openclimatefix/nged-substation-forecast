"""Draw the figures of experiment 122 as SVG into the docs folder.

Only Primary substations appear, under their D-labels. Every number comes from `numbers.json`,
which `e00122_report.py` writes, so a figure and the page quote the same numbers. Afterwards
optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00122-*/e00122_report.py
    uv run python ml-experiments/00122-*/e00122_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00122_report as report
import polars as pl

DOCS_DIR: Final[Path] = report.REPORT_PATH.parent
WIDTH: Final[int] = 500
"""Plot-area width: the axis labels bring a chart to the 680 px text column."""
WRAP: Final[int] = 95
WINTER_COLOURS: Final[dict[str, str]] = {
    "2023-24": "#306BFF",
    "2024-25": "#FF4901",
    "2025-26": "#17E58F",
}
"""Data Blue, Brand Orange, and Data Green: one colour per winter."""
ARM_COLOURS: Final[dict[str, str]] = {"GAM": "#306BFF", "GAM-K": "#FF4901", "XGBoost": "#B701FF"}
"""Data Blue, Brand Orange, and Data Purple: one colour per forecasting model."""
GREY: Final[str] = "#808080"


def _numbers() -> dict[str, Any]:
    return json.loads((report.DATA_DIR / "numbers.json").read_text())


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 70),
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")


def figure_1_headline() -> alt.TopLevelMixin:
    """Return the six planned contrasts, one row per winter and plain model, with 95% intervals."""
    numbers = _numbers()["contrast"]
    rows = [
        {
            "winter": winter,
            "other": other,
            "label": f"{winter} vs {other}",
            "estimate": values["all"][0],
            "low": values["all"][1],
            "high": values["all"][2],
        }
        for winter in report.WINTERS
        for other in report.OTHER_ARMS
        for values in [numbers[winter][other]]
    ]
    frame = pl.DataFrame(rows)
    order = [row["label"] for row in rows][::-1]
    base = alt.Chart(frame).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260, labelPadding=6))
    )
    bar = (
        alt.Chart(pl.DataFrame({"x": [report.BAR_POINTS]}))
        .mark_rule(strokeDash=[5, 4], color="#333333")
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    intervals = base.mark_rule(strokeWidth=3).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title=None),
        x2="high:Q",
        color=alt.Color(
            "winter:N",
            scale=alt.Scale(domain=list(WINTER_COLOURS), range=list(WINTER_COLOURS.values())),
            legend=alt.Legend(title="Winter", orient="top"),
        ),
    )
    dots = base.mark_point(filled=True, size=90).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "estimate:Q",
            title=(
                "GAM-K minus the other model, mean absolute error in points of capacity "
                "(more negative = GAM-K better)"
            ),
            scale=alt.Scale(domain=[-4.5, 0.5]),
        ),
        color=alt.Color("winter:N", legend=None),
    )
    return (zero + bar + intervals + dots).properties(
        width=WIDTH,
        padding={"left": 30, "right": 10, "top": 5, "bottom": 5},
        height=200,
        title=_title(
            text="GAM-K beat both plain models in all three winters by more than 1 point",
            subtitle=(
                "All six planned contrasts, five Primary substations, observed past weather, "
                "day-ahead. Dot: estimate. Line: 95% interval from resampled 7-day blocks of "
                "days. Dashed line: the 1-point bar. Grey line: no difference."
            ),
        ),
    )


def figure_2_absolute() -> alt.TopLevelMixin:
    """Return each model's mean absolute error by winter."""
    numbers = _numbers()["mae"]
    rows = [
        {"winter": winter, "model": arm.label, "mae": numbers[winter][arm.label]["all"]}
        for winter in report.WINTERS
        for arm in report.ARMS
    ]
    gam_k = [row["mae"] for row in rows if row["model"] == "GAM-K"]
    plain = [row["mae"] for row in rows if row["model"] != "GAM-K"]
    headline = (
        f"GAM-K's error is {min(gam_k):.1f} to {max(gam_k):.1f} points in each winter; "
        f"the plain models' is {min(plain):.1f} to {max(plain):.1f}"
    )
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_point(filled=True, size=110)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("winter:N", title=None),
            x=alt.X(
                "mae:Q",
                title="Mean absolute error, points of capacity (smaller is better)",
                scale=alt.Scale(domain=[0, 8]),
            ),
            color=alt.Color(
                "model:N",
                scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                legend=alt.Legend(title="Model", orient="top"),
            ),
        )
        .properties(
            width=WIDTH,
            height=150,
            title=_title(
                text=headline,
                subtitle=(
                    "Mean absolute error of the median forecast over all November to March days, "
                    "five Primary substations, observed past weather, day-ahead."
                ),
            ),
        )
    )


def figure_3_bias() -> alt.TopLevelMixin:
    """Return the unfiltered models' mean signed error by substation and winter."""
    numbers = _numbers()["bias"]
    rows = [
        {"winter": winter, "model": arm.label, "site": site, "bias": value}
        for winter in report.WINTERS
        for arm in report.ARMS
        for site, value in numbers[winter][arm.label].items()
    ]
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_point(filled=True, size=80)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("site:N", sort=list(report.DECISION_SITES), title=None),
            x=alt.X(
                "bias:Q",
                title="Observed minus forecast, points of capacity (0 = unbiased)",
                scale=alt.Scale(domain=[-13, 8]),
            ),
            color=alt.Color(
                "model:N",
                scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                legend=alt.Legend(title="Model", orient="top"),
            ),
            row=alt.Row("winter:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
        )
        .properties(
            width=WIDTH,
            height=110,
            title=_title(
                text="The plain models carry level biases that the filter removes",
                subtitle=(
                    "Mean signed error of the median forecast over all winter days. Positive: the "
                    "model under-forecast demand. In 2023-24 one substation, D10, was "
                    "over-forecast by 12 points by both plain models."
                ),
            ),
        )
    )


def main() -> None:
    """Write the three figures."""
    _save(figure_1_headline(), name="figure_1_headline")
    _save(figure_2_absolute(), name="figure_2_absolute")
    _save(figure_3_bias(), name="figure_3_bias")


if __name__ == "__main__":
    main()
