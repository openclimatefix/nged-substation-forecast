"""Draw experiment 00021's figures from the saved results, as SVG.

1. The eight planned contrasts on the quantile score over all leads, with 99.375% intervals.
2. The quantile score of the main arms by substation type, with 95% intervals.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00021-error-correction-variants-bsp-gsp-demand/e00021_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00021_common as common
import plotting.ocf_theme as ocf
import polars as pl

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 540
WRAP_CHARACTERS: Final[int] = 92
ARM_LABELS: Final[dict[str, str]] = {
    "climatology": "Climatology",
    "manual_heuristic": "manual_heuristic",
    "n1": "N1, no correction",
    "a14": "N1 plus 14-day correction",
}
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1": "C1 (primary): correction minus N1, BSP and GSP",
    "C2": "C2: correction minus manual_heuristic, BSP and GSP",
    "C3": "C3: correction minus N1, Primary",
    "C4": "C4: 7-day window minus 14-day, Primary",
    "C5": "C5: 28-day window minus 14-day, Primary",
    "C6": "C6: multiplicative minus additive, Primary",
    "C7": "C7: per-half-hour minus single shift, Primary",
    "C8": "C8: recency-weighted minus equal weights, Primary",
}
SCOPE: Final[str] = "73 forecast runs, one every 5 days, July 2025 to June 2026, ENS weather."


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1() -> alt.TopLevelMixin:
    """The eight planned contrasts on the quantile score over all leads."""
    data = pl.read_parquet(RESULTS_DIR / "contrasts.parquet").filter(
        pl.col("planned")
        & (pl.col("metric") == "quantile_score")
        & (pl.col("scope") == "all leads")
    )
    data = data.with_columns(label=pl.col("label").str.slice(0, 2).replace_strict(CONTRAST_LABELS))
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=data["label"].to_list(), title=None, axis=alt.Axis(labelLimit=330))
    )
    points = base.mark_point(filled=True, size=70, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("difference:Q", title="Difference in quantile score (points of capacity)")
    )
    bars = base.mark_rule(color=ocf.DATA_BLUE).encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#999999").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=PLOT_WIDTH,
        height=280,
        title=_title(
            text="The correction helps Primary substations but not BSP and GSP",
            lines=[
                (
                    "Each row is the first arm minus the second on the quantile score over all "
                    "leads; below zero favours the first arm. " + SCOPE
                ),
                (
                    "Dot: estimate. Line: 99.375% interval (Bonferroni across the eight planned "
                    "contrasts) from re-drawing whole months and seeds."
                ),
            ],
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """The quantile score of the main arms, by substation type."""
    data = (
        pl.read_parquet(RESULTS_DIR / "absolute.parquet")
        .filter(
            (pl.col("metric") == "quantile_score")
            & (pl.col("scope") == "all leads")
            & pl.col("arm").is_in(list(ARM_LABELS))
            & pl.col("group").is_in(["Primary", "BSP", "GSP"])
        )
        .with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
    )
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=list(ARM_LABELS.values()), title=None),
        color=alt.Color(
            "arm:N",
            legend=None,
            scale=alt.Scale(
                domain=list(ARM_LABELS),
                range=["#999999", "#999999", ocf.DATA_BLUE, ocf.BRAND_ORANGE],
            ),
        ),
    )
    points = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "value:Q",
            title="Quantile score (points of capacity)",
            scale=alt.Scale(zero=False),
        )
    )
    bars = base.mark_rule().encode(x="lower_95:Q", x2="upper_95:Q")  # ty: ignore[unresolved-attribute]
    return (
        alt.layer(bars, points)
        .properties(width=220, height=130)
        .facet(
            row=alt.Row(
                "group:N",
                sort=["Primary", "BSP", "GSP"],
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left"),
            )
        )
        .resolve_scale(x="independent")
        .properties(
            title=_title(
                text="BSP and GSP score level with N1 whether corrected or not",
                lines=[
                    (
                        "Quantile score over all leads (smaller is better); 16 Primary "
                        "substations, 2 BSP, 2 GSP. Dot: estimate. Line: 95% interval from "
                        "re-drawing whole months and seeds. " + SCOPE
                    )
                ],
            )
        )
    )


def main() -> None:
    """Write both figures."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (("figure_1", figure_1()), ("figure_2", figure_2())):
        chart.save(CHART_DIR / f"{name}.svg")


if __name__ == "__main__":
    main()
