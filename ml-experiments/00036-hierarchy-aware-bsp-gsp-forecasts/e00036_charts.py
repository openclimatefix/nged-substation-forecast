"""Draw experiment 00036's figures from the saved results, as SVG.

1. The planned contrasts and the B3 contrast at the BSPs on the quantile score, with intervals.
2. The quantile score of every arm for each of the four series, with 95% intervals.
3. B3 minus B0 by lead band for each series.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00036-*/e00036_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00036_common as common
import plotting.ocf_theme as ocf
import polars as pl

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 400
WRAP_CHARACTERS: Final[int] = 92
SERIES: Final[dict[str, str]] = {
    "D17": "Boston (BSP)",
    "D13": "Skegness (BSP)",
    "D6": "Walpole (GSP)",
    "D1": "Bicker Fen (GSP)",
}
ARM_LABELS: Final[dict[str, str]] = {
    "climatology": "Climatology",
    "b0": "B0 N1",
    "b1": "B1 child columns",
    "b2": "B2 bottom-up sum",
    "b3": "B3 ERA5-trained N1",
    "b4": "B4 wind, solar columns",
    "o": "O oracle (observed children)",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "climatology": "#999999",
    "b0": "#555555",
    "b1": ocf.DATA_BLUE,
    "b2": ocf.DATA_SKY,
    "b3": ocf.BRAND_ORANGE,
    "b4": ocf.DATA_PURPLE,
    "o": ocf.DATA_GREEN,
}
SCOPE: Final[str] = "73 forecast runs, one every 5 days, July 2025 to June 2026, ENS weather."


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1() -> alt.TopLevelMixin:
    """The planned contrasts, and B3 minus B0 at the BSPs."""
    data = pl.read_parquet(RESULTS_DIR / "contrasts.parquet").filter(
        (pl.col("metric") == "quantile_score")
        & (pl.col("scope") == "all leads")
        & (
            pl.col("planned")
            | ((pl.col("treatment") == "b3") & pl.col("site").is_in(["D17", "D13"]))
        )
    )
    data = data.with_columns(
        label=pl.format(
            "{} minus {}, {}",
            pl.col("treatment").str.to_uppercase(),
            pl.col("reference").str.to_uppercase(),
            pl.col("site").replace_strict(SERIES),
        ),
        kind=pl.when(pl.col("planned")).then(pl.lit("Planned")).otherwise(pl.lit("Exploratory")),
    )
    rank = {"b1": 0, "b3": 1, "b4": 2}
    order = [
        row["label"]
        for row in sorted(
            data.iter_rows(named=True),
            key=lambda r: (not r["planned"], rank[r["treatment"]], list(SERIES).index(r["site"])),
        )
    ]
    base = alt.Chart(data).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=290)),
        color=alt.Color(
            "kind:N",
            title=None,
            scale=alt.Scale(domain=["Planned", "Exploratory"], range=[ocf.DATA_BLUE, "#999999"]),
            legend=alt.Legend(orient="top"),
        ),
    )
    points = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("difference:Q", title="Difference in quantile score (points of capacity)")
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#999999").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=PLOT_WIDTH,
        height=230,
        title=_title(
            text="Child forecasts and wind and solar columns do not lower the BSP or GSP score",
            lines=[
                (
                    "Each row is the first arm minus the second on the quantile score over all "
                    "leads; below zero favours the first arm. Dot: estimate. Line: 99.17% "
                    "interval for planned rows (Bonferroni across six) and 95% for exploratory "
                    "rows, from re-drawing whole months and seeds. " + SCOPE
                )
            ],
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """The quantile score of every arm for each series."""
    data = (
        pl.read_parquet(RESULTS_DIR / "absolute.parquet")
        .filter((pl.col("metric") == "quantile_score") & (pl.col("scope") == "all leads"))
        .with_columns(
            arm_label=pl.col("arm").replace_strict(ARM_LABELS, default=None),
            series=pl.col("site").replace_strict(SERIES),
        )
        .drop_nulls("arm_label")
    )
    base = alt.Chart(data).encode(
        y=alt.Y("arm_label:N", sort=list(ARM_LABELS.values()), title=None),
        color=alt.Color(
            "arm_label:N",
            legend=None,
            scale=alt.Scale(domain=list(ARM_LABELS.values()), range=list(ARM_COLOURS.values())),
        ),
    )
    points = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("value:Q", title="Quantile score (points of capacity)", scale=alt.Scale(zero=False))
    )
    bars = base.mark_rule().encode(x="lower_95:Q", x2="upper_95:Q")  # ty: ignore[unresolved-attribute]
    return (
        alt.layer(bars, points)
        .properties(width=PLOT_WIDTH, height=130)
        .facet(
            row=alt.Row(
                "series:N",
                sort=list(SERIES.values()),
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left"),
            )
        )
        .resolve_scale(x="independent")
        .properties(
            title=_title(
                text="Only the oracle, which sees the observed children, scores clearly lower",
                lines=[
                    (
                        "Quantile score over all leads (smaller is better), per series. Dot: "
                        "estimate. Line: 95% interval from re-drawing whole months and seeds. "
                        + SCOPE
                    )
                ],
            )
        )
    )


def figure_3() -> alt.TopLevelMixin:
    """B3 minus B0 by lead band for each series."""
    data = (
        pl.read_parquet(RESULTS_DIR / "contrasts.parquet")
        .filter(
            (pl.col("metric") == "quantile_score")
            & (pl.col("scope") != "all leads")
            & (pl.col("treatment") == "b3")
            & (pl.col("reference") == "b0")
        )
        .with_columns(series=pl.col("site").replace_strict(SERIES))
    )
    bands = [name for name, _, _ in common.LEAD_BANDS]
    base = alt.Chart(data).encode(
        y=alt.Y("scope:N", sort=bands, title=None),
        color=alt.Color(
            "series:N",
            legend=None,
            scale=alt.Scale(
                domain=list(SERIES.values()),
                range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_SKY, ocf.DATA_PURPLE],
            ),
        ),
    )
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("difference:Q", title="B3 minus B0, quantile score (points of capacity)")
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = base.mark_rule(strokeWidth=1).encode(  # ty: ignore[unresolved-attribute]
        x=alt.datum(0), y=alt.value(0), y2=alt.value(110), color=alt.value("#999999")
    )
    return (
        alt.layer(zero, bars, points)
        .properties(width=PLOT_WIDTH, height=110)
        .facet(
            row=alt.Row(
                "series:N",
                sort=list(SERIES.values()),
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left"),
            )
        )
        .properties(
            title=_title(
                text="The ERA5-trained N1 helps Boston up to day 6 and is unresolved elsewhere",
                lines=[
                    (
                        "B3 minus B0 by lead band; below zero favours the ERA5-trained N1. Dot: "
                        "estimate. Line: 95% interval from re-drawing whole months and seeds "
                        "(exploratory). " + SCOPE
                    )
                ],
            )
        )
    )


def main() -> None:
    """Write the three figures."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_1", figure_1()),
        ("figure_2", figure_2()),
        ("figure_3", figure_3()),
    ):
        chart.save(CHART_DIR / f"{name}.svg")


if __name__ == "__main__":
    main()
