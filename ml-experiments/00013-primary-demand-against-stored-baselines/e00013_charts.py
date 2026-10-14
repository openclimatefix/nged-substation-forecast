"""Draw experiment 00013's figures from the saved results, as SVG.

1. The error of the median by lead band for the Primary substations, one dot per arm with its 95%
   interval (the absolute scores).
2. The five planned contrasts, in three lead scopes, with their 99% intervals.
3. The mean signed error of the median by lead band, per arm (negative: the forecast is too low).

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00013-primary-demand-against-stored-baselines/e00013_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00013_common as common
import plotting.ocf_theme as ocf
import polars as pl

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 540
WRAP_CHARACTERS: Final[int] = 92
HEADLINE: Final[str] = "Primary"
BANDS: Final[tuple[str, ...]] = tuple(name for name, _, _ in common.LEAD_BANDS)
ARM_LABELS: Final[dict[str, str]] = {
    "climatology": "Climatology (stored)",
    "production": "Production recipe (stored)",
    "manual_heuristic": "manual_heuristic (stored)",
    "s1": "Stage 1: ERA5-trained, no lagged power (S1)",
    "n1_nolag": "N1 without lagged power",
    "n1": "N1: ENS-trained quantile model with lagged power",
}
ARM_COLOURS: Final[list[str]] = [
    "#999999",
    ocf.DATA_SKY,
    ocf.BRAND_ORANGE,
    ocf.DATA_PURPLE,
    ocf.DATA_GREEN,
    ocf.DATA_BLUE,
]
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1 (primary): N1 minus manual_heuristic": "C1 (primary): N1 minus manual_heuristic",
    "C2: N1 minus production": "C2: N1 minus the production recipe",
    "C3: production minus manual_heuristic": "C3: production recipe minus manual_heuristic",
    "C4: N1 minus N1-nolag": "C4: N1 minus N1 without lagged power",
    "C5: N1-nolag minus S1": "C5: N1 without lagged power minus S1",
}
CONTRAST_SCOPES: Final[tuple[str, ...]] = ("all leads", "up to 48 h", "3 d or more")
DEFINITIONS: Final[str] = (
    "N1 is an XGBoost quantile model trained on the ENS control member from April 2024, with "
    "lagged power measured from the forecast origin. S1 is an XGBoost model trained on ERA5 "
    "weather from September 2023, without lagged power."
)
SCOPE: Final[str] = (
    "16 Primary substations; 73 forecast runs, one every 5 days, July 2025 to June 2026."
)


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _legend() -> alt.Legend:
    return alt.Legend(title=None, orient="top", columns=1, labelLimit=500)


def figure_1() -> alt.TopLevelMixin:
    """The Primary substations' error of the median by lead band."""
    data = (
        pl.read_parquet(RESULTS_DIR / "absolute.parquet")
        .filter(
            (pl.col("group") == HEADLINE)
            & (pl.col("metric") == "median_error")
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
            "label:N", scale=alt.Scale(domain=labels, range=ARM_COLOURS), legend=_legend()
        ),
    )
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "value:Q",
            title="Mean absolute error of the median (points of capacity; smaller is better)",
            scale=alt.Scale(zero=False),
        )
    )
    bars = base.mark_rule().encode(x="lower_95:Q", x2="upper_95:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(bars, points).properties(
        width=PLOT_WIDTH,
        height=330,
        title=_title(
            text="The production recipe trails manual_heuristic at every lead; N1 is level with it",
            lines=[
                (
                    "Mean absolute error of the median forecast, in points of the substation's "
                    "capacity (smaller is better), by how many days ahead the forecast is."
                ),
                DEFINITIONS,
                SCOPE,
                "Dot: estimate. Line: 95% interval from re-drawing whole months and seeds.",
            ],
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """The planned contrasts in three lead scopes, with their 99% intervals."""
    data = (
        pl.read_parquet(RESULTS_DIR / "contrasts.parquet")
        .filter(
            pl.col("planned")
            & (pl.col("metric") == "median_error")
            & pl.col("scope").is_in(list(CONTRAST_SCOPES))
        )
        .with_columns(label=pl.col("label").replace_strict(CONTRAST_LABELS))
    )
    labels = list(CONTRAST_LABELS.values())
    base = alt.Chart(data).encode(
        y=alt.Y(
            "scope:N",
            sort=list(CONTRAST_SCOPES),
            title="Forecast lead",
            axis=alt.Axis(labelLimit=200),
        ),
        yOffset=alt.YOffset("label:N", sort=labels),
        color=alt.Color(
            "label:N",
            scale=alt.Scale(
                domain=labels,
                range=[
                    ocf.BRAND_ORANGE,
                    ocf.DATA_BLUE,
                    ocf.DATA_PURPLE,
                    ocf.DATA_GREEN,
                    ocf.DATA_SKY,
                ],
            ),
            legend=_legend(),
        ),
    )
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q",
            title="Change in error of the median (points of capacity; more negative is better)",
        )
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=PLOT_WIDTH,
        height=300,
        title=_title(
            text="N1 may beat manual_heuristic within two days, but the interval includes zero",
            lines=[
                (
                    "Change in the mean absolute error of the median between two arms (points of "
                    "the substation's capacity; more negative means the first arm is better)."
                ),
                DEFINITIONS,
                SCOPE,
                (
                    "Dot: estimate. Line: 99% interval, widened for the five planned contrasts. "
                    "Solid rule: no difference."
                ),
            ],
        ),
    )


def figure_3() -> alt.TopLevelMixin:
    """The mean signed error of the median by lead band, per arm."""
    data = (
        pl.read_parquet(RESULTS_DIR / "bias.parquet")
        .filter(
            (pl.col("group") == HEADLINE)
            & pl.col("arm").is_in(list(ARM_LABELS))
            & pl.col("scope").is_in(list(BANDS))
        )
        .with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
    )
    labels = list(ARM_LABELS.values())
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    base = alt.Chart(data).encode(
        y=alt.Y("scope:N", sort=list(BANDS), title="Forecast lead", axis=alt.Axis(labelLimit=200)),
        yOffset=alt.YOffset("label:N", sort=labels),
        color=alt.Color(
            "label:N", scale=alt.Scale(domain=labels, range=ARM_COLOURS), legend=_legend()
        ),
    )
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "bias:Q",
            title="Mean signed error of the median (points of capacity; zero is unbiased)",
            scale=alt.Scale(domain=[-3.5, 0.5]),
        )
    )
    return alt.layer(zero, points).properties(
        width=PLOT_WIDTH,
        height=330,
        title=_title(
            text="Every arm forecasts demand too low in the validation year",
            lines=[
                (
                    "Mean of forecast minus observed power, in points of the substation's "
                    "capacity; negative means the median forecast is too low."
                ),
                DEFINITIONS,
                SCOPE,
                "Dot: mean over all cases of the lead band. Solid rule: no bias.",
            ],
        ),
    )


def main() -> None:
    """Write the figures as SVG."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for number, chart in enumerate((figure_1(), figure_2(), figure_3()), start=1):
        chart.configure_axis(labelFontSize=11, titleFontSize=11).save(
            CHART_DIR / f"figure_{number}.svg"
        )


if __name__ == "__main__":
    main()
