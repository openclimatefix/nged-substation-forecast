# ruff: noqa: E501
"""Draw experiment 271's figures from the saved report tables, as SVG.

Run `e00271_report.py` first. Afterwards optimise each SVG with
`npx svgo@4 --multipass --precision=1 --final-newline`.
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00271_inputs as inputs
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{inputs.EXPERIMENT_ID:05d}-{inputs.SLUG}"
)
REPORT: Final[Path] = inputs.RESULTS_DIR / "report"
WIDTH: Final[int] = 600
WRAP: Final[int] = 100


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1_counts() -> alt.Chart:
    """How many logged pairs, interconnections, and real events cross a BSP boundary."""
    frame = pl.DataFrame(
        [
            ("Logged load-shift pairs (31)", "Same BSP", 22),
            ("Logged load-shift pairs (31)", "Crosses the BSP boundary", 4),
            ("Logged load-shift pairs (31)", "Partner unknown", 5),
            ("Primary interconnections (54)", "Same BSP", 33),
            ("Primary interconnections (54)", "Crosses the BSP boundary", 5),
            ("Primary interconnections (54)", "Partner unknown", 16),
            ("Real events over 1 day (35)", "Same BSP", 27),
            ("Real events over 1 day (35)", "Crosses the BSP boundary", 1),
            ("Real events over 1 day (35)", "Partner unknown", 7),
        ],
        schema=["source", "where", "count"],
        orient="row",
    )
    order = ["Same BSP", "Crosses the BSP boundary", "Partner unknown"]
    frame = frame.with_columns(
        rank=pl.col("where").replace_strict({k: i for i, k in enumerate(order)})
    )
    return (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_bar(aria=False)
        .encode(
            y=alt.Y("source:N", sort=None, title=None),
            x=alt.X("count:Q", title="Count"),
            color=alt.Color(
                "where:N",
                sort=order,
                scale=alt.Scale(domain=order, range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.GREY_3]),
                legend=alt.Legend(orient="top", title=None, direction="horizontal"),
            ),
            order=alt.Order("rank:Q", sort="ascending"),
        )
        .properties(
            width=WIDTH - 220,
            height=130,
            title=_title(
                text="Real load shifts almost never cross a bulk supply point boundary",
                lines=[
                    "Where the two Primary substations of each pair sit: under the same bulk supply point (BSP), under different BSPs, or one end outside the 30 listed substations (for events: one Primary only, so no partner). D-labels only."
                ],
            ),
        )
    )


def figure_2_effect() -> alt.LayerChart:
    """Effect of the injected load shifts on K0's avoidable cost."""
    effects = pl.read_parquet(REPORT / "effects.parquet").filter(
        pl.col("arm") == "K0", pl.col("target") == "observed"
    )
    effects = effects.with_columns(
        label=pl.col("kind").replace({"within": "Within one BSP", "cross": "Across the two BSPs"})
        + pl.lit(", ")
        + pl.col("placement").replace(
            {"random": "random placement", "targeted": "placed before event days"}
        )
    )
    order = [
        "Within one BSP, random placement",
        "Within one BSP, placed before event days",
        "Across the two BSPs, random placement",
        "Across the two BSPs, placed before event days",
    ]
    colours = alt.Scale(
        domain=order,
        range=[ocf.DATA_BLUE_LIGHT, ocf.DATA_BLUE, ocf.BRAND_ORANGE_LIGHT, ocf.BRAND_ORANGE],
    )
    base = alt.Chart(effects).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=330))
    )
    rule = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x="low_pct:Q", x2="high_pct:Q", color=alt.Color("label:N", scale=colours, legend=None)
    )
    dot = base.mark_point(filled=True, size=90, opacity=1, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("mean_pct:Q", title="Change in avoidable cost (% of the real record's)"),
        color=alt.Color("label:N", scale=colours, legend=None),
    )
    marks = (
        alt.Chart(pl.DataFrame({"x": [0.0, 1.0, 5.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color=ocf.GREY_3)
        .encode(x="x:Q")
    )
    return (marks + rule + dot).properties(
        width=WIDTH - 340,
        height=150,
        title=_title(
            text=[  # ty: ignore[invalid-argument-type]
                "Load shifts across the boundary move the cost by about 5%,",
                "those inside it by under 1%",
            ],
            lines=[
                "No-knowledge forecast (K0), avoidable cost at 0.5 times each BSP's earlier maximum export, GBP 2,000 per MWh, 2025-09 to 2026-06. Dot: mean over 40 worlds of 8 load shifts. Line: 97.5% interval from resampling whole worlds. Dashed lines: 0, and the thresholds of 1% (within) and 5% (across)."
            ],
        ),
    )


def figure_3_knowledge() -> alt.LayerChart:
    """Perfect knowledge minus K0, by row group, for the load shifts placed before event days."""
    contrasts = pl.read_parquet(REPORT / "arm_contrasts.parquet").filter(
        pl.col("target") == "observed",
        pl.col("placement") == "targeted",
        pl.col("arm") == "PERFECT",
        pl.col("group").is_in(["G1", "G2", "G3", "G4", "G5"]),
    )
    names = {
        "G1": "Ordinary",
        "G2": "Before alarm",
        "G3": "During",
        "G4": "After",
        "G5": "Near (7 days)",
    }
    contrasts = contrasts.with_columns(
        label=pl.col("group").replace(names),
        kind_label=pl.col("kind").replace(
            {"within": "Within one BSP", "cross": "Across the two BSPs"}
        ),
        low=100 * pl.col("low_gbp") / pl.col("k0_gbp"),
        high=100 * pl.col("high_gbp") / pl.col("k0_gbp"),
        mid=pl.col("difference_pct_of_k0"),
    )
    colours = alt.Scale(
        domain=["Within one BSP", "Across the two BSPs"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
    )
    base = alt.Chart(contrasts).encode(
        y=alt.Y("label:N", sort=list(names.values()), title=None),
        yOffset="kind_label:N",
    )
    rule = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x="low:Q",
        x2="high:Q",
        color=alt.Color(
            "kind_label:N",
            scale=colours,
            legend=alt.Legend(orient="top", title=None, direction="horizontal"),
        ),
    )
    dot = base.mark_point(filled=True, size=70, opacity=1, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("mid:Q", title="PERFECT minus K0, avoidable cost (% of K0's; negative is better)"),
        color=alt.Color("kind_label:N", scale=colours, legend=None),
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=ocf.GREY_3).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return (zero + rule + dot).properties(
        width=WIDTH - 140,
        height=230,
        title=_title(
            text=[  # ty: ignore[invalid-argument-type]
                "Knowing the load shifts exactly changes the cost by less",
                "than the data can resolve",
            ],
            lines=[
                "Avoidable cost with the shifts masked from the filter (PERFECT, known at once) minus the no-knowledge filter (K0), for the load shifts placed before event days, observed-power target. Row groups follow the plan. Dot: mean over 40 worlds. Line: 97.5% interval from resampling whole worlds."
            ],
        ),
    )


def main() -> None:
    """Write the three figures."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_1_counts", figure_1_counts()),
        ("figure_2_effect", figure_2_effect()),
        ("figure_3_knowledge", figure_3_knowledge()),
    ):
        chart.save(str(DOCS_DIR / f"{name}.svg"))


if __name__ == "__main__":
    main()
