"""Draw experiment 00027's figures from the saved results, as SVG.

1. The three planned contrasts on the GSP quantile score, with 98.33% intervals.
2. The stage 1 variant S1: the change in stage 1a's point error from adding PV_Live, by GSP.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00027-pvlive-lags-in-stage2-gsp-demand/e00027_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00027_common as common
import plotting.ocf_theme as ocf
import polars as pl

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 540
WRAP_CHARACTERS: Final[int] = 92
LABELS: Final[dict[str, str]] = {
    "C1 PV_Live level": "PV_Live level",
    "C2 PV_Live share": "PV_Live share of its capacity",
    "C3 PV_Live-adjusted delta": "Delta plus PV_Live",
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1() -> alt.TopLevelMixin:
    """The three planned contrasts, over all leads and up to 48 hours."""
    data = (
        pl.read_parquet(RESULTS_DIR / "contrasts.parquet")
        .filter(pl.col("planned") & pl.col("scope").is_in(["all leads", "up to 48 h"]))
        .with_columns(arm=pl.col("label").replace_strict(LABELS))
    )
    base = alt.Chart(data).encode(
        y=alt.Y("arm:N", sort=list(LABELS.values()), title=None, axis=alt.Axis(labelLimit=300)),
        color=alt.Color(
            "scope:N",
            title=None,
            scale=alt.Scale(
                domain=["all leads", "up to 48 h"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
            ),
            legend=alt.Legend(orient="top"),
        ),
        yOffset="scope:N",
    )
    points = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q", title="Quantile score minus the no-PV_Live arm (points of capacity)"
        )
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#999999").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=PLOT_WIDTH,
        height=150,
        title=_title(
            text="PV_Live at the lag times does not lower the GSP demand error",
            lines=[
                (
                    "Each row is a stage 2 arm with PV_Live minus the same arm without it; "
                    "below zero favours PV_Live (smaller score is better). The no-PV_Live arm "
                    "scores 19.04 over all leads. Walpole and Bicker Fen, 73 forecast runs, July "
                    "2025 to June 2026, ENS weather."
                ),
                "Dot: estimate. Line: 98.33% interval from re-drawing whole months and seeds.",
            ],
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """The change in stage 1a's point error from adding PV_Live at the valid time."""
    data = pl.read_parquet(RESULTS_DIR / "s1.parquet").with_columns(
        label=pl.col("series").replace({"D1": "Bicker Fen", "D6": "Walpole"}),
        hours=pl.col("scope").replace({"all": "all valid times", "daytime": "daytime only"}),
    )
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", title=None, sort=["Bicker Fen", "Walpole"]),
        yOffset="hours:N",
        color=alt.Color(
            "hours:N",
            title=None,
            scale=alt.Scale(
                domain=["all valid times", "daytime only"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
            ),
            legend=alt.Legend(orient="top"),
        ),
    )
    points = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q",
            title="Mean absolute error with PV_Live minus without (points of capacity)",
        )
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#999999").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=PLOT_WIDTH,
        height=180,
        title=_title(
            text="PV_Live at the valid time helps stage 1a at Bicker Fen, not at Walpole",
            lines=[
                (
                    "Point prediction of stage 1a at observed ERA5 weather, July 2025 to June "
                    "2026; below zero favours adding PV_Live (smaller error is better). "
                    "Daytime means PV_Live above 1% of the GSP's capacity. Dot: estimate. Line: "
                    "95% interval from re-drawing whole months."
                )
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
