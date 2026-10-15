"""Draw experiment 25's figures from the results tables, as SVG.

Reads the tables `e00025_report.py` wrote and writes the figures to
`docs/ml-experiments/00025-price-change-features-stage1-demand/`. Only Primary substations are
drawn, because BSP and GSP have two series each and are given in the tables.

    uv run python ml-experiments/00025-price-change-features-stage1-demand/e00025_charts.py
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00025_inputs as inputs
import polars as pl
from contracts.settings import PROJECT_ROOT

OUT_DIR: Final[Path] = (
    PROJECT_ROOT / "docs" / "ml-experiments" / "00025-price-change-features-stage1-demand"
)
RESULTS: Final[Path] = inputs.EXPERIMENT_DIR / "results"
WIDTH: Final[int] = 600
BLUE: Final[str] = "#306BFF"
ORANGE: Final[str] = "#FF4901"
GREY: Final[str] = "#9AA0A6"
FOLD_LABELS: Final[dict[str, str]] = {
    "select_2024": "Selection year (2024-07 to 2025-06)",
    "confirm_2025": "Confirmation year (2025-07 to 2026-06)",
}
CONTRAST_LABELS: Final[dict[str, str]] = {
    "PRICE_CHANGE - PRICE_LEVEL": "C1 changes vs level only",
    "PRICE_CHANGE - SHUFFLED_CHANGE": "C2 changes vs other-week changes",
    "PRICE_CHANGE - NATIONAL": "C3 changes vs national demand, wind",
    "PRICE_LEVEL - BASE": "C5 level vs no price",
    "SYSTEM - PRICE_LEVEL": "C6 system price vs level only",
}
ARM_ORDER: Final[list[str]] = [
    "BASE",
    "PRICE_LEVEL",
    "PRICE_CHANGE",
    "PRICE_SHAPE",
    "SHUFFLED_CHANGE",
    "SHUFFLED_SHAPE",
    "NATIONAL",
    "SYSTEM",
]


def _zero_rule() -> alt.Chart:
    return alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]


def contrast_chart() -> alt.VConcatChart:
    """Return stacked interval charts of the Primary contrasts, one panel per year."""
    panels = []
    for fold, label in FOLD_LABELS.items():
        raw = pl.read_parquet(RESULTS / f"{fold}_contrasts.parquet").filter(
            pl.col("scope") == "Primary", pl.col("contrast").is_in(list(CONTRAST_LABELS))
        )
        data = raw.with_columns(label=pl.col("contrast").replace(CONTRAST_LABELS))
        y = alt.Y(
            "label:N",
            sort=list(CONTRAST_LABELS.values()),
            title=None,
            axis=alt.Axis(labelLimit=250),
        )
        bars = (
            alt.Chart(data)
            .mark_rule(strokeWidth=3, color=BLUE)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("lower:Q", title="Change in error (points of capacity; below 0 is better)"),
                x2="upper:Q",
                y=y,
            )
        )
        points = (
            alt.Chart(data).mark_circle(size=70, color=ORANGE).encode(x="difference:Q", y=y)  # ty: ignore[unresolved-attribute]
        )
        panels.append(
            alt.layer(_zero_rule(), bars, points).properties(
                width=WIDTH - 250, height=150, title=label
            )
        )
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Price changes lowered Primary error by under 0.1 points.",
                "Dot: estimate. Line: 97.5% interval, resampling whole months.",
            ],
            anchor="start",
            fontSize=13,
        )
    )


def hour_chart() -> alt.VConcatChart:
    """Return stacked charts of C1 and C2 by 4-hour block of local time, one panel per year."""
    panels = []
    for fold, label in FOLD_LABELS.items():
        raw = pl.read_parquet(RESULTS / f"{fold}_contrasts.parquet").filter(
            pl.col("scope").str.starts_with("C7 hours")
        )
        data = raw.with_columns(
            block=pl.col("scope").str.replace("C7 hours ", ""),
            against=pl.when(pl.col("contrast").str.ends_with("PRICE_LEVEL"))
            .then(pl.lit("Changes vs level only (C1)"))
            .otherwise(pl.lit("Changes vs other-week changes (C2)")),
        )
        chart = (
            alt.Chart(data)
            .mark_line(point=True)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("block:N", title="Local hours of day", sort=None),
                y=alt.Y("difference:Q", title="Change in error (points)"),
                color=alt.Color(
                    "against:N",
                    scale=alt.Scale(range=[ORANGE, BLUE]),
                    legend=alt.Legend(orient="top", title=None),
                ),
            )
        )
        panels.append(
            alt.layer(
                alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color=GREY).encode(y="y:Q"),  # ty: ignore[unresolved-attribute]
                chart,
            ).properties(width=WIDTH - 100, height=150, title=label)
        )
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Primary error change by time of day. Below 0 is better.",
            anchor="start",
            fontSize=13,
        )
    )


def level_chart() -> alt.VConcatChart:
    """Return stacked bar charts of each arm's absolute Primary error, one panel per year."""
    panels = []
    for fold, label in FOLD_LABELS.items():
        data = pl.read_parquet(RESULTS / f"{fold}_levels.parquet").filter(
            pl.col("scope") == "Primary", pl.col("arm").is_in(ARM_ORDER)
        )
        panels.append(
            alt.Chart(data)
            .mark_bar(color=BLUE)
            .encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("arm:N", sort=ARM_ORDER, title=None),
                x=alt.X("value:Q", title="Mean absolute error (points of capacity)"),
            )
            .properties(width=WIDTH - 200, height=170, title=label)
        )
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Absolute error of each arm, Primary substations.", anchor="start", fontSize=13
        )
    )


def main() -> None:
    """Write the three figures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_1.svg", contrast_chart()),
        ("figure_2.svg", hour_chart()),
        ("figure_3.svg", level_chart()),
    ):
        chart.save(str(OUT_DIR / name))
        print(f"wrote {OUT_DIR / name}")


if __name__ == "__main__":
    main()
