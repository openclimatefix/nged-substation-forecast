"""Draw experiment 00031's figures from the saved results, as SVG.

1. The six planned contrasts on the quantile score over all leads, with 99.17% intervals.
2. Each arm's gain over B0 by sky condition, season, and lead band, with 95% intervals.
3. Each Primary's gain from A3 and A5 over B0, ordered by its register capacity.

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00031-embedded-pv-features-primary-demand/e00031_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00031_common as common
import plotting.ocf_theme as ocf
import polars as pl

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
PLOT_WIDTH: Final[int] = 540
WRAP_CHARACTERS: Final[int] = 92
SCOPE: Final[str] = (
    "16 Primary substations; 73 forecast runs, one every 5 days, July 2025 to June 2026."
)
ARM_COLOURS: Final[dict[str, str]] = {
    "A1 capacity statistics": ocf.DATA_BLUE,
    "A2 PV products, data only": ocf.DATA_SKY,
    "A3 PV products + register": ocf.DATA_PURPLE,
    "A4 linear offset": ocf.DATA_GREEN,
    "A5 shaped correction": ocf.BRAND_ORANGE,
}
ARM_NAMES: Final[dict[str, str]] = {
    "a1": "A1 capacity statistics",
    "a2": "A2 PV products, data only",
    "a3": "A3 PV products + register",
    "a4": "A4 linear offset",
    "a5": "A5 shaped correction",
}
SUBSET_ORDER: Final[tuple[str, ...]] = (
    "night",
    "daylight, index under 0.3",
    "daylight, index 0.3 to 0.7",
    "daylight, index 0.7 or more",
    "April to September, index 0.7 or more",
    "April to September",
    "October to March",
    "0-48 h",
    "3-6 d",
    "7-15 d",
)


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1() -> alt.TopLevelMixin:
    """The six planned contrasts on the quantile score over all leads."""
    data = pl.read_parquet(RESULTS_DIR / "contrasts.parquet").filter(
        pl.col("planned")
        & (pl.col("metric") == "quantile_score")
        & (pl.col("subset") == "all leads")
    )
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
        height=200,
        title=_title(
            text="No arm beats the 00019 stack, and the capacity statistics make it worse",
            lines=[
                (
                    "Each row is the first arm minus the second, on the quantile score over all "
                    "leads; below zero favours the first arm. B0 is the 00019 stack. " + SCOPE
                ),
                (
                    "Dot: estimate. Line: 99.17% interval (Bonferroni across the six planned "
                    "contrasts) from re-drawing whole months and seeds."
                ),
            ],
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """Each arm's gain over B0 by sky condition, season, and lead band."""
    data = (
        pl.read_parquet(RESULTS_DIR / "contrasts.parquet")
        .filter(
            pl.col("label").str.starts_with("breakdown")
            & pl.col("subset").is_in(list(SUBSET_ORDER))
            & pl.col("treatment").is_in(list(ARM_NAMES))
        )
        .with_columns(
            arm=pl.col("treatment").replace_strict(ARM_NAMES),
        )
    )
    base = alt.Chart(data).encode(
        y=alt.Y("subset:N", sort=list(SUBSET_ORDER), title=None, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("arm:N", sort=list(ARM_NAMES.values())),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
            legend=alt.Legend(orient="top", columns=2, title=None, labelLimit=300),
        ),
    )
    points = base.mark_point(filled=True, size=40).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("difference:Q", title="Arm minus B0, quantile score (points of capacity)")
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#999999").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=PLOT_WIDTH,
        height=420,
        title=_title(
            text="The shaped correction gains in sunny hours and loses in cloudy ones",
            lines=[
                (
                    "Each arm minus B0 on the quantile score, by the CAMS clear-sky index at the "
                    "valid time, by season, and by lead band; below zero favours the arm. " + SCOPE
                ),
                "Dot: estimate. Line: 95% interval from re-drawing whole months and seeds.",
            ],
        ),
    )


def figure_3() -> alt.TopLevelMixin:
    """Each Primary's quantile-score gain from A3 and A5 over B0, ordered by register capacity."""
    scores = pl.read_parquet(RESULTS_DIR / "per_site.parquet").filter(pl.col("group") == "Primary")
    b0 = scores.filter(pl.col("arm") == "b0").select("site", b0="score")
    data = (
        scores.filter(pl.col("arm").is_in(["a3", "a5"]))
        .join(b0, on="site", validate="m:1")
        .with_columns(
            difference=pl.col("score") - pl.col("b0"),
            arm=pl.col("arm").replace_strict(ARM_NAMES),
            meter=pl.when(pl.col("directional"))
            .then(pl.lit("signed flow (MW)"))
            .otherwise(pl.lit("unsigned flow (MVA)")),
        )
    )
    order = data.filter(pl.col("arm") == ARM_NAMES["a3"]).sort("ecr_share")["site"].to_list()
    base = alt.Chart(data).encode(
        y=alt.Y(
            "site:N", sort=order, title="Primary substation, by register capacity (low to high)"
        ),
        yOffset="arm:N",
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(
                domain=[ARM_NAMES["a3"], ARM_NAMES["a5"]],
                range=[ocf.DATA_PURPLE, ocf.BRAND_ORANGE],
            ),
            legend=alt.Legend(orient="top", title=None, labelLimit=300),
        ),
        shape=alt.Shape("meter:N", legend=alt.Legend(orient="top", title=None)),
    )
    points = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("difference:Q", title="Arm minus B0, quantile score (points of capacity)")
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#999999").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, points).properties(
        width=PLOT_WIDTH,
        height=380,
        title=_title(
            text="The gains from the register and the shaped correction differ by substation",
            lines=[
                (
                    "Each dot is one Primary's mean quantile score for the arm minus B0's, over "
                    "all leads and seeds; below zero favours the arm. No interval is drawn. "
                    + SCOPE
                ),
            ],
        ),
    )


def main() -> None:
    """Write the figures."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_1", figure_1()),
        ("figure_2", figure_2()),
        ("figure_3", figure_3()),
    ):
        chart.save(CHART_DIR / f"{name}.svg")


if __name__ == "__main__":
    main()
