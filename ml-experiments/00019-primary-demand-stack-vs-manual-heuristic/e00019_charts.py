"""Draw experiment 00019's figures from the saved results, as SVG.

1. The quantile score of each arm over all leads, with its 95% interval.
2. The nine planned contrasts on the quantile score over all leads, with 99.44% intervals.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00019-primary-demand-stack-vs-manual-heuristic/e00019_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00019_common as common
import plotting.ocf_theme as ocf
import polars as pl

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 540
WRAP_CHARACTERS: Final[int] = 92
ARM_LABELS: Final[dict[str, str]] = {
    "climatology": "Climatology",
    "production": "Production recipe",
    "n1": "N1",
    "n1_s": "N1 plus spread factor",
    "n1h": "N1 with half-life 0.5 years",
    "n1_c": "N1 plus 14-day correction",
    "stack": "Full stack",
    "manual_heuristic": "manual_heuristic",
}
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1": "C1 (primary): stack minus manual_heuristic",
    "C2": "C2: stack minus N1",
    "C3": "C3: stack minus production",
    "C4": "C4: stack without correction minus stack",
    "C5": "C5: stack without half-life minus stack",
    "C6": "C6: stack without spread factor minus stack",
    "C7": "C7: N1 plus correction minus N1",
    "C8": "C8: N1 with half-life 0.5 years minus N1",
    "C9": "C9: N1 plus spread factor minus N1",
}
SCOPE: Final[str] = (
    "16 Primary substations; 73 forecast runs, one every 5 days, July 2025 to June 2026."
)


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1() -> alt.TopLevelMixin:
    """Each arm's quantile score with its 95% interval."""
    data = (
        pl.read_parquet(RESULTS_DIR / "absolute.parquet")
        .filter(
            (pl.col("metric") == "quantile_score")
            & (pl.col("scope") == "all leads")
            & pl.col("arm").is_in(list(ARM_LABELS))
        )
        .with_columns(
            label=pl.col("arm").replace_strict(ARM_LABELS),
            colour=pl.when(pl.col("arm") == "stack")
            .then(pl.lit("stack"))
            .when(pl.col("arm") == "manual_heuristic")
            .then(pl.lit("benchmark"))
            .otherwise(pl.lit("other")),
        )
    )
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=list(ARM_LABELS.values()), title=None),
        color=alt.Color(
            "colour:N",
            scale=alt.Scale(
                domain=["other", "benchmark", "stack"],
                range=[ocf.DATA_BLUE, "#999999", ocf.BRAND_ORANGE],
            ),
            legend=None,
        ),
    )
    points = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "value:Q",
            title="Quantile score (points of capacity; smaller is better)",
            scale=alt.Scale(zero=False),
        )
    )
    bars = base.mark_rule().encode(x="lower_95:Q", x2="upper_95:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(bars, points).properties(
        width=PLOT_WIDTH,
        height=260,
        title=_title(
            text="The full stack scores below manual_heuristic; N1 alone does not",
            lines=[
                (
                    "Quantile score of the probabilistic forecast over all leads, in points of "
                    "the substation's capacity. " + SCOPE
                ),
                (
                    "Dot: estimate. Line: 95% interval from re-drawing whole months and seeds. "
                    "The intervals are wide because every arm's error rises and falls together "
                    "by month; the paired contrasts in figure 2 are narrower."
                ),
            ],
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """The nine planned contrasts on the quantile score over all leads."""
    data = pl.read_parquet(RESULTS_DIR / "contrasts.parquet").filter(
        pl.col("planned")
        & (pl.col("metric") == "quantile_score")
        & (pl.col("scope") == "all leads")
    )
    data = data.with_columns(label=pl.col("label").str.slice(0, 2).replace_strict(CONTRAST_LABELS))
    order = data["label"].to_list()
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=330))
    )
    points = base.mark_point(filled=True, size=70, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("difference:Q", title="Difference in quantile score (points of capacity)")
    )
    bars = base.mark_rule(color=ocf.DATA_BLUE).encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#999999").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=PLOT_WIDTH,
        height=300,
        title=_title(
            text="The 14-day correction supplies the whole gain",
            lines=[
                (
                    "Each row is the first arm minus the second, on the quantile score over all "
                    "leads; below zero favours the first arm. " + SCOPE
                ),
                (
                    "Dot: estimate. Line: 99.44% interval (Bonferroni across the nine planned "
                    "contrasts) from re-drawing whole months and seeds."
                ),
            ],
        ),
    )


def main() -> None:
    """Write both figures."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (("figure_1", figure_1()), ("figure_2", figure_2())):
        chart.save(CHART_DIR / f"{name}.svg")


if __name__ == "__main__":
    main()
