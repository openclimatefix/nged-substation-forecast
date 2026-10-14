"""Draw experiment 00018's figures from the saved results, as SVG.

1. The quantile score by lead band for the Primary substations, one dot per arm with its 95%
   interval (the absolute scores).
2. The six planned contrasts, in three lead scopes, with their 99.17% intervals.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python \
        ml-experiments/00018-staged-chain-vs-n1-serving-latency-lag-anomalies-demand/e00018_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00018_common as common
import plotting.ocf_theme as ocf
import polars as pl

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 540
WRAP_CHARACTERS: Final[int] = 92
HEADLINE: Final[str] = "Primary"
BANDS: Final[tuple[str, ...]] = tuple(name for name, _, _ in common.LEAD_BANDS)
ARM_LABELS: Final[dict[str, str]] = {
    "r0c": "Stage 1 and calibrator, no lagged power (R0c)",
    "manual_heuristic": "manual_heuristic (stored)",
    "n1": "N1: ENS-trained quantile model with lagged power",
    "ch_none": "Chain with raw lagged power only",
    "ch_s": "Chain with serving-latency lag anomalies",
    "ch_e": "Chain with final ERA5 lag anomalies (ceiling)",
}
ARM_COLOURS: Final[list[str]] = [
    "#999999",
    ocf.DATA_SKY,
    ocf.DATA_BLUE,
    ocf.DATA_PURPLE,
    ocf.BRAND_ORANGE,
    ocf.DATA_GREEN,
]
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1 (primary): CH-S minus N1": "C1 (primary): chain minus N1",
    "C2: CH-S minus R0c": "C2: chain minus the no-lag calibrated stage 1",
    "C3: CH-S minus manual_heuristic": "C3: chain minus manual_heuristic",
    "C4: CH-S minus CH-E": "C4: chain minus the final-ERA5 ceiling",
    "C5: CH-S minus CH-none": "C5: chain minus raw lagged power only",
    "C6: CH-S-spread minus CH-S": "C6: chain with spread factor minus chain",
}
CONTRAST_SCOPES: Final[tuple[str, ...]] = ("all leads", "up to 48 h", "3 d or more")
DEFINITIONS: Final[str] = (
    "The chain is an ERA5-trained stage 1 model plus a stage 2 calibrator; its lag anomalies use "
    "ERA5T, CAMS, and the ENS control member's early leads, with a 7-hour telemetry delay. N1 is "
    "an XGBoost quantile model trained on the ENS control member."
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
            & (pl.col("metric") == "quantile_score")
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
            title="Quantile score (points of capacity; smaller is better)",
            scale=alt.Scale(zero=False),
        )
    )
    bars = base.mark_rule().encode(x="lower_95:Q", x2="upper_95:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(bars, points).properties(
        width=PLOT_WIDTH,
        height=330,
        title=_title(
            text="The chain is level with N1 and behind manual_heuristic beyond two days",
            lines=[
                (
                    "Quantile score of the probabilistic forecast, in points of the substation's "
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
            & (pl.col("metric") == "quantile_score")
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
                    "#999999",
                ],
            ),
            legend=_legend(),
        ),
    )
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q",
            title="Change in quantile score (points of capacity; more negative is better)",
        )
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=PLOT_WIDTH,
        height=300,
        title=_title(
            text="Against N1 and manual_heuristic, the chain's intervals include zero",
            lines=[
                (
                    "Change in the quantile score between two arms (points of "
                    "the substation's capacity; more negative means the first arm is better)."
                ),
                DEFINITIONS,
                SCOPE,
                (
                    "Dot: estimate. Line: 99.17% interval, widened for the six planned contrasts. "
                    "Solid rule: no difference."
                ),
            ],
        ),
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
