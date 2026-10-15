"""Draw experiment 00030's figures from the saved results, as SVG.

1. The quantile score of each arm over all leads, without and with the 14-day correction.
2. The contrasts of the lag arms against N1 (A0), without and with the correction.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00030-lagged-power-feature-sets-primary-demand/e00030_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00030_common as common
import plotting.ocf_theme as ocf
import polars as pl

RESULTS_DIR: Final[Path] = common.WORK_DIR / "final"
CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 540
WRAP_CHARACTERS: Final[int] = 92
ARM_LABELS: Final[dict[str, str]] = {
    "climatology": "Climatology",
    "manual_heuristic": "manual_heuristic",
    "a0": "A0: N1, 3 lags",
    "a1": "A1: 6 weekly lags",
    "a3": "A3: 3 summaries of 13 lags",
    "a7": "A7: weekly mean and median",
    "stack": "00019 stack",
}
CONTRASTS: Final[dict[str, str]] = {
    "C1: weekly analogues, A1 minus A0": "A1 minus A0, no correction",
    "E14 (post hoc): A7 minus A0": "A7 minus A0, no correction",
    "E12: A3 with the correction minus A0 with the correction": "A3 minus A0, with correction",
    "E15 (post hoc): A7 with the correction minus A0 with the correction": (
        "A7 minus A0, with correction"
    ),
    "E16 (post hoc): A7 with the correction minus the 00019 stack": (
        "A7 with correction minus 00019 stack"
    ),
}
SCOPE: Final[str] = (
    "16 Primary substations; 73 forecast runs, one every 5 days, July 2025 to June 2026; 3 seeds."
)


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1() -> alt.TopLevelMixin:
    """Each arm's quantile score with its 95% interval, without and with the correction."""
    frame = pl.read_parquet(RESULTS_DIR / "absolute.parquet").filter(
        (pl.col("metric") == "quantile_score") & (pl.col("scope") == "all leads")
    )
    rows = []
    for arm, label in ARM_LABELS.items():
        for suffix, corrected in (("", "no correction"), ("_c", "14-day correction")):
            name = arm + suffix
            if name in frame["arm"].to_list():
                row = frame.filter(pl.col("arm") == name).row(0, named=True)
                rows.append({"label": label, "correction": corrected, **row})
    data = pl.DataFrame(rows).filter(
        ~((pl.col("label") == "00019 stack") & (pl.col("correction") == "no correction"))
    )
    base = alt.Chart(data).encode(
        y=alt.Y(
            "label:N", sort=list(ARM_LABELS.values()), title=None, axis=alt.Axis(labelLimit=330)
        ),
        yOffset="correction:N",
        color=alt.Color(
            "correction:N",
            scale=alt.Scale(
                domain=["no correction", "14-day correction"],
                range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
            ),
            legend=alt.Legend(orient="top", title=None),
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
        height=300,
        title=_title(
            text="Weekly lags help alone; with the correction all arms score near 5.3",
            lines=[
                "Quantile score over all leads, in points of capacity. " + SCOPE,
                "Dot: estimate. Line: 95% interval from re-drawing whole months and seeds.",
            ],
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """The contrasts of the lag arms against N1."""
    data = (
        pl.read_parquet(RESULTS_DIR / "contrasts.parquet")
        .filter(
            (pl.col("metric") == "quantile_score")
            & (pl.col("scope") == "all leads")
            & pl.col("label").is_in(list(CONTRASTS))
        )
        .with_columns(row=pl.col("label").replace_strict(CONTRASTS))
    )
    base = alt.Chart(data).encode(
        y=alt.Y("row:N", sort=list(CONTRASTS.values()), title=None, axis=alt.Axis(labelLimit=330))
    )
    points = base.mark_point(filled=True, size=70, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("difference:Q", title="Difference in quantile score (points of capacity)")
    )
    bars = base.mark_rule(color=ocf.DATA_BLUE).encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#999999").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=PLOT_WIDTH,
        height=200,
        title=_title(
            text="Weekly lags beat N1 before the correction, by little after it",
            lines=[
                (
                    "Each row is the first arm minus the second; below zero favours the first "
                    "arm. " + SCOPE
                ),
                (
                    "Dot: estimate. Line: interval from re-drawing whole months and seeds; "
                    "99.375% for the first row, 95% for the others."
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
