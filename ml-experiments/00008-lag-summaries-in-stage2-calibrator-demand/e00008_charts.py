"""Draw experiment 00008's figures from the saved results, as SVG.

1. The Primary substations' quantile score by lead band, one dot per arm with its 95% interval.
2. The four planned contrasts and the exploratory S1 minus S2, for the Primary substations, by lead
   scope, with their intervals.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00008-lag-summaries-in-stage2-calibrator-demand/e00008_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00008_common as common
import plotting.ocf_theme as ocf
import polars as pl

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 540
WRAP_CHARACTERS: Final[int] = 92
HEADLINE: Final[str] = "Primary"
BANDS: Final[tuple[str, ...]] = tuple(name for name, _, _ in common.LEAD_BANDS)
ARM_LABELS: Final[dict[str, str]] = {
    "r2": "Climatology (month, time of day, weekday or weekend)",
    "r0c": "No recent power (R0c)",
    "s4": "Summaries only, no ERA5 residual (S4)",
    "l3": "Raw lags, lag anomalies, residuals (L3)",
    "s1": "L3 plus the 6 summaries (S1)",
    "s2": "Summaries replace the raw power lags (S2)",
    "s3": "Summaries replace all lags (S3)",
}
ARM_COLOURS: Final[list[str]] = [
    "#999999",
    ocf.DATA_SKY,
    ocf.DATA_GREEN,
    ocf.DATA_BLUE,
    ocf.BRAND_ORANGE,
    ocf.DATA_PURPLE,
    "#5a5a5a",
]
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1 (primary): S1 minus L3": "C1 (primary): summaries added to L3",
    "C2: S2 minus L3": "C2: summaries replace the raw power lags",
    "C3: S3 minus L3": "C3: summaries replace all lags",
    "C4: S4 minus R0c": "C4: summaries only, against no recent power",
    "E1: S1 minus S2 (the raw power lags)": "Exploratory: S1 minus S2 (the raw power lags)",
}
SCOPE: Final[str] = (
    "16 Primary substations; 73 forecast runs, one every 5 days, July 2025 to June 2026. Arms "
    "with ERA5 residuals use ERA5 weather at past times, so they are an upper bound on production."
)


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1() -> alt.TopLevelMixin:
    """The Primary substations' quantile score by lead band."""
    data = (
        pl.read_parquet(RESULTS_DIR / "absolute.parquet")
        .filter(
            (pl.col("group") == HEADLINE)
            & pl.col("scope").is_in(list(BANDS))
            & pl.col("arm").is_in(list(ARM_LABELS))
        )
        .with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
    )
    labels = list(ARM_LABELS.values())
    base = alt.Chart(data).encode(
        y=alt.Y("scope:N", sort=list(BANDS), title="Forecast lead", axis=alt.Axis(labelLimit=200)),
        yOffset=alt.YOffset("label:N", sort=labels),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(domain=labels, range=ARM_COLOURS),
            legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=500),
        ),
    )
    points = base.mark_point(filled=True, size=50).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("value:Q", title="Quantile score (points of capacity; smaller is better)")
    )
    bars = base.mark_rule().encode(x="lower_95:Q", x2="upper_95:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(bars, points).properties(
        width=PLOT_WIDTH,
        height=340,
        title=_title(
            text="Summaries of recent power add little to the calibrator that already has raw lags",
            lines=[
                (
                    "Quantile score of the 9-quantile demand forecast, in points of the "
                    "substation's capacity (smaller is better), by days ahead."
                ),
                SCOPE,
                "Dot: estimate. Line: 95% interval from re-drawing whole months and seeds.",
            ],
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """The planned contrasts by lead scope, with their intervals."""
    scopes = ["up to 48 h", "all leads", *BANDS]
    data = (
        pl.read_parquet(RESULTS_DIR / "contrasts.parquet")
        .filter(
            (pl.col("group") == HEADLINE)
            & pl.col("label").is_in(list(CONTRAST_LABELS))
            & pl.col("scope").is_in(scopes)
        )
        .with_columns(contrast=pl.col("label").replace_strict(CONTRAST_LABELS))
    )
    labels = list(CONTRAST_LABELS.values())
    colours = [ocf.BRAND_ORANGE, ocf.DATA_PURPLE, "#5a5a5a", ocf.DATA_GREEN, ocf.DATA_BLUE]
    base = alt.Chart(data).encode(
        y=alt.Y("scope:N", sort=scopes, title="Forecast lead", axis=alt.Axis(labelLimit=200)),
        yOffset=alt.YOffset("contrast:N", sort=labels),
        color=alt.Color(
            "contrast:N",
            scale=alt.Scale(domain=labels, range=colours),
            legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=500),
        ),
    )
    points = base.mark_point(filled=True, size=50).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q",
            title="Quantile score of the first arm minus the second (points of capacity; "
            "more negative is better)",
        )
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="black", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(zero, bars, points).properties(
        width=PLOT_WIDTH,
        height=380,
        title=_title(
            text="Summaries alone, without ERA5 residuals, clearly beat having no recent power",
            lines=[
                (
                    "Difference in quantile score between two arms, in points of capacity. The "
                    "dashed line at zero means the two arms score the same."
                ),
                SCOPE,
                (
                    "Dot: estimate. Line: 98.75% interval (planned contrasts) or 95% "
                    "(exploratory) from re-drawing whole months and seeds."
                ),
            ],
        ),
    )


def main() -> None:
    """Write both figures as SVG."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (("figure_1", figure_1()), ("figure_2", figure_2())):
        chart.save(str(CHART_DIR / f"{name}.svg"))


if __name__ == "__main__":
    main()
