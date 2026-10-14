"""Draw experiment 15's figures from the results tables, as SVG.

Reads the tables `e00015_report.py` wrote and writes the figures to
`docs/ml-experiments/00015-same-time-price-stage1-demand/`. Series carry the anonymous labels
experiment 00006 gave them (P1 to P16, B1, B2, G1, G2).

    uv run python ml-experiments/00015-same-time-price-stage1-demand/e00015_charts.py
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00015_inputs as inputs
import polars as pl
from contracts.settings import PROJECT_ROOT

OUT_DIR: Final[Path] = (
    PROJECT_ROOT / "docs" / "ml-experiments" / "00015-same-time-price-stage1-demand"
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
AGAINST_LABELS: Final[dict[str, str]] = {
    "PRICE - BASE": "Against no price",
    "PRICE - SHUFFLED": "Against other week",
    "PRICE - CONTROL": "Against demand, wind",
}
GAIN_DOMAIN: Final[list[str]] = ["Price vs no price", "Price vs other-week price"]


def _zero_rule() -> alt.Chart:
    return alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]


def contrast_chart() -> alt.VConcatChart:
    """Return stacked interval charts of the pooled contrasts, one panel per year."""
    panels = []
    for fold, label in FOLD_LABELS.items():
        raw = pl.read_parquet(RESULTS / f"{fold}_contrasts.parquet").filter(
            pl.col("scope") == "pooled"
        )
        data = raw.with_columns(label=pl.col("contrast").replace(AGAINST_LABELS))
        order = list(AGAINST_LABELS.values())
        y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=190))
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
                width=WIDTH - 200, height=110, title=label
            )
        )
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "The price beat no price in the selection year, but a price from",
                "another week did almost as well. Pooled over series. Below 0 is better.",
                "Dot: estimate. Line: 97.5% interval, resampling whole months.",
            ],
            anchor="start",
            fontSize=13,
        )
    )


def hour_chart() -> alt.VConcatChart:
    """Return stacked line charts of the pooled gain by local hour, one panel per year."""
    panels = []
    for fold, label in FOLD_LABELS.items():
        both = pl.concat(
            [
                pl.read_parquet(RESULTS / f"{fold}_by_hour.parquet").with_columns(
                    against=pl.lit("Price vs no price")
                ),
                pl.read_parquet(RESULTS / f"{fold}_by_hour_noise_floor.parquet").with_columns(
                    against=pl.lit("Price vs other-week price")
                ),
            ]
        )
        chart = (
            alt.Chart(both)
            .mark_line(point=True)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("hour:Q", title="Local hour of day", axis=alt.Axis(values=[0, 6, 12, 18])),
                y=alt.Y("gain:Q", title="Change in error (points)"),
                color=alt.Color(
                    "against:N",
                    scale=alt.Scale(domain=GAIN_DOMAIN, range=[ORANGE, BLUE]),
                    legend=alt.Legend(orient="top", title=None),
                ),
            )
        )
        panels.append(
            alt.layer(
                chart,
                alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color=GREY).encode(y="y:Q"),  # ty: ignore[unresolved-attribute]
            ).properties(width=WIDTH, height=170, title=label)
        )
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "The price helped most in the afternoon, pooled over the fitted series.",
                "Below 0 is better; the gain is in points of capacity.",
            ],
            anchor="start",
            fontSize=13,
        )
    )


def series_chart() -> alt.VConcatChart:
    """Return stacked dot charts of each series' change in error, one panel per year."""
    panels = []
    for fold, label in FOLD_LABELS.items():
        data = pl.concat(
            [
                pl.read_parquet(RESULTS / f"{fold}_by_series.parquet").with_columns(
                    against=pl.lit("Price vs no price")
                ),
                pl.read_parquet(RESULTS / f"{fold}_by_series_noise_floor.parquet").with_columns(
                    against=pl.lit("Price vs other-week price")
                ),
            ]
        )
        order = data.filter(pl.col("against") == GAIN_DOMAIN[0]).sort("gain")["site"].to_list()
        dots = (
            alt.Chart(data)
            .mark_circle(size=70)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("gain:Q", title="Change in error (points of capacity; below 0 is better)"),
                y=alt.Y("site:N", sort=order, title=None),
                color=alt.Color(
                    "against:N",
                    scale=alt.Scale(domain=GAIN_DOMAIN, range=[ORANGE, BLUE]),
                    legend=alt.Legend(orient="top", title=None),
                ),
            )
        )
        panels.append(
            alt.layer(_zero_rule(), dots).properties(width=WIDTH, height=280, title=label)
        )
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "The price changed the error mainly at the series with the largest",
                "errors. Below 0 is better.",
            ],
            anchor="start",
            fontSize=13,
        )
    )


def main() -> None:
    """Write the three figures as SVG."""
    alt.theme.enable("ocf") if "ocf" in alt.theme.names() else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    contrast_chart().save(OUT_DIR / "figure_1.svg")
    hour_chart().save(OUT_DIR / "figure_2.svg")
    series_chart().save(OUT_DIR / "figure_3.svg")
    print(f"wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
