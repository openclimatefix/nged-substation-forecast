"""Draw experiment 17's figures from the saved results, as SVG.

1. The nine planned contrasts against arm A, with 99.44% intervals.
2. The mean signed error of the median in each calendar year, for four arms.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00017-*/e00017_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00017_report as report
import plotting.ocf_theme as ocf
import polars as pl

CHART_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments"
    / ("00017-training-window-and-retraining-rule-drift-demand")
)
PLOT_WIDTH: Final[int] = 520
WRAP: Final[int] = 92
LABELS: Final[dict[str, str]] = {
    "W6": "C1: 6-month window",
    "W12": "C2: 12-month window",
    "W18": "C3: 18-month window",
    "H0.5": "C4: half-life 0.5 years",
    "H1": "C5: half-life 1 year",
    "H2": "C6: half-life 2 years",
    "F": "C7: never retrained",
    "Y": "C8: retrained yearly",
    "K": "C9: 14-day error correction",
}
KIND: Final[dict[str, str]] = {
    "W6": "Sliding window",
    "W12": "Sliding window",
    "W18": "Sliding window",
    "H0.5": "Recency weights",
    "H1": "Recency weights",
    "H2": "Recency weights",
    "F": "Retraining rule",
    "Y": "Retraining rule",
    "K": "Error correction",
}
KIND_COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.DATA_PURPLE, ocf.BRAND_ORANGE, ocf.DATA_GREEN]
SCOPE: Final[str] = (
    "16 Primary substations; monthly forecasts January 2022 to June 2026, ERA5 weather."
)


def _title(*, title: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [w for line in lines for w in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(title, subtitle=wrapped, anchor="start", subtitleFontSize=11)


def figure_1() -> alt.Chart:
    """The planned contrasts with their intervals."""
    frame = pl.read_parquet(report.RESULTS_DIR / "contrasts.parquet").with_columns(
        label=pl.col("arm").replace_strict(LABELS), kind=pl.col("arm").replace_strict(KIND)
    )
    order = [LABELS[a] for a in report.CONTRASTS]
    base = alt.Chart(frame).encode(
        y=alt.Y("label:N", sort=order, title=None),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(domain=list(dict.fromkeys(KIND.values())), range=KIND_COLOURS),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    rules = base.mark_rule(strokeWidth=2, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q",
            title="Mean absolute error minus A's (points of capacity; more negative is better)",
        ),
        x2="upper:Q",
    )
    dots = base.mark_point(filled=True, size=70, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x="difference:Q"
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    return (zero + rules + dots).properties(
        width=PLOT_WIDTH,
        height=230,
        title=_title(
            title="Recent training rows and an error correction lower the demand error",
            lines=[
                (
                    "Each arm minus A, which trains on all history and retrains monthly. A's error is "
                    "7.62 points of capacity. Dot: estimate. Line: 99.44% interval from resampling "
                    "whole months. "
                    "Rule at zero: same as A."
                ),
                SCOPE,
            ],
        ),
    )


def figure_2() -> alt.Chart:
    """Mean signed error by calendar year."""
    year = pl.read_parquet(report.RESULTS_DIR / "by_year.parquet")
    arms = {
        "A": "A: all history, monthly",
        "H2": "H2: half-life 2 years",
        "H0.5": "H0.5: half-life 0.5 years",
        "F": "F: never retrained",
        "K": "K: A + 14-day correction",
    }
    long = pl.concat(
        [
            year.select("year", bias=pl.col(f"bias_{a}").cast(pl.Float64)).with_columns(
                arm=pl.lit(label)
            )
            for a, label in arms.items()
        ]
    ).with_columns(pl.col("bias").cast(pl.Float64))
    return (
        alt.Chart(long)
        .mark_line(point=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "year:O", title="Calendar year (2026: January to June)", axis=alt.Axis(labelAngle=0)
            ),
            y=alt.Y("bias:Q", title="Observed minus forecast (points of capacity)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=list(arms.values()),
                    range=[
                        ocf.GREY_3 if hasattr(ocf, "GREY_3") else "#999999",
                        ocf.DATA_PURPLE,
                        ocf.DATA_SKY,
                        ocf.BRAND_ORANGE,
                        ocf.DATA_GREEN,
                    ],
                ),
                legend=alt.Legend(title=None, orient="top", columns=2),
            ),
        )
        .properties(
            width=PLOT_WIDTH,
            height=240,
            title=_title(
                title="Forecast bias swung from too high to too low",
                lines=[
                    (
                        "Mean of observed minus forecast median over each calendar year. Positive: the "
                        "forecast is too low."
                    ),
                    SCOPE,
                ],
            ),
        )
    )


def main() -> None:
    """Write the figures as SVG."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for number, chart in enumerate((figure_1(), figure_2()), start=1):
        chart.configure_axis(labelFontSize=11, titleFontSize=11).save(
            CHART_DIR / f"figure_{number}.svg"
        )


if __name__ == "__main__":
    main()
