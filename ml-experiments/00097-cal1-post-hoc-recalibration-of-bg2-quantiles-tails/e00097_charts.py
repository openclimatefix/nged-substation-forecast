"""Draw experiment 97's figures from the saved results, as SVG.

Run `e00097_report.py` first. Afterwards optimise each SVG with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00097-*/e00097_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00097_report as report
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{report.EXPERIMENT_ID:05d}-{report.SLUG}"
)
WIDTH: Final[int] = 480
WRAP: Final[int] = 100
DECISION_BAND: Final[str] = report.BANDS[report.DECISION_BAND]
ARM_LABELS: Final[dict[str, str]] = {
    "a0_as_stored": "0 As stored",
    "a1_global_1.15": "1 Global factor 1.15",
    "a2_scale_per_level": "2 Scale per level",
    "a3_scale_per_level_band_class": "3 Scale per level, band, class",
    "a4_isotonic_remap": "4 Isotonic remapping",
    "a5_conformal_shift": "5 Conformal shift",
    "a6_pareto_tails": "6 Stored body + Pareto tails",
    "a7a_student_t": "7a Student-t",
    "a7b_skew_normal": "7b Skew-normal",
    "a8_post_hoc_scale_recent_90_days": "8 Scale, last 90 days (post hoc)",
    "a9_post_hoc_conformal_recent_90_days": "9 Conformal, last 90 days (post hoc)",
}
ARM_ORDER: Final[list[str]] = list(ARM_LABELS.values())
ARM_COLOURS: Final[list[str]] = [
    ocf.BRAND_ORANGE,
    ocf.DATA_PURPLE_LIGHT,
    ocf.DATA_BLUE,
    ocf.DATA_BLUE,
    ocf.DATA_SKY,
    ocf.DATA_SKY,
    ocf.DATA_GREEN,
    ocf.DATA_PURPLE,
    ocf.DATA_PURPLE,
    ocf.DATA_AMBER,
    ocf.DATA_AMBER,
]
"""One colour per arm family: stored, global, scaled, empirical, fitted tail, parametric."""
FIGURE_4_ARMS: Final[list[str]] = [
    "a0_as_stored",
    "a1_global_1.15",
    "a3_scale_per_level_band_class",
    "a5_conformal_shift",
    "a6_pareto_tails",
    "a9_post_hoc_conformal_recent_90_days",
]

TAIL_LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.95, 0.99)


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _arm_label() -> pl.Expr:
    return pl.col("arm").replace_strict(ARM_LABELS)


def figure_coverage(*, cls: str, name: str) -> alt.Chart:
    """Coverage minus nominal at the four tail levels, one panel per level, one row per arm."""
    cov = (
        pl.read_parquet(report.RESULTS_DIR / "coverage.parquet")
        .filter(
            (pl.col("class") == cls)
            & (pl.col("band") == DECISION_BAND)
            & pl.col("level").is_in(TAIL_LEVELS)
        )
        .with_columns(
            arm_label=_arm_label(),
            panel=pl.col("level").map_elements(
                lambda p: f"q{p * 100:g} (nominal {p})", return_dtype=pl.String
            ),
            gap=pl.col("coverage") - pl.col("level"),
            gap_low=pl.col("coverage_low") - pl.col("level"),
            gap_high=pl.col("coverage_high") - pl.col("level"),
        )
    )
    cov = cov.with_columns(zero=pl.lit(0.0), band_low=pl.lit(-0.01), band_high=pl.lit(0.01))
    base = alt.Chart().encode(
        y=alt.Y("arm_label:N", sort=ARM_ORDER, title=None, axis=alt.Axis(labelLimit=240)),
        color=alt.Color(
            "arm_label:N", scale=alt.Scale(domain=ARM_ORDER, range=ARM_COLOURS), legend=None
        ),
    )
    x = alt.X("gap:Q", title="Coverage minus nominal level (0 is perfect)")
    interval = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("gap_low:Q", title="Coverage minus nominal level (0 is perfect)"), x2="gap_high:Q"
    )
    dots = base.mark_point(filled=True, size=60, aria=False).encode(x=x)  # ty: ignore[unresolved-attribute]
    zero = alt.Chart().mark_rule(color="black").encode(x="zero:Q")  # ty: ignore[unresolved-attribute]
    band = (
        alt.Chart()
        .mark_rect(fill=ocf.DATA_GREEN_LIGHT, opacity=0.5)
        .encode(x="band_low:Q", x2="band_high:Q")  # ty: ignore[unresolved-attribute]
    )
    layered = alt.layer(band, zero, interval, dots).properties(width=WIDTH, height=150)
    return (
        layered.facet(
            row=alt.Row(
                "panel:N", title=None, header=alt.Header(labelAnchor="start", labelFontSize=12)
            ),
            data=cov,
        )
        .properties(
            title=_title(
                text=f"{cls}: tail coverage of each recalibration, 12 to 36 hour lead",
                lines=[
                    (
                        "Dot: share of measurements at or below the quantile, minus the quantile's "
                        "nominal level. Line: 95% interval from resampling whole months. "
                        "Green band: within 0.01 of nominal. Scored months October 2025 to June "
                        "2026; each fitted on earlier months."
                    )
                ],
            )
        )
        .resolve_scale(x="shared")
    )


def figure_score() -> alt.Chart:
    """Change in quantile score against the stored forecast, 12 to 36 hour lead, by class."""
    scores = (
        pl.read_parquet(report.RESULTS_DIR / "scores.parquet")
        .filter(
            (pl.col("band") == DECISION_BAND)
            & (pl.col("score") == "quantile_score")
            & (pl.col("arm") != "a0_as_stored")
        )
        .with_columns(arm_label=_arm_label(), zero=pl.lit(0.0))
    )
    base = alt.Chart().encode(
        y=alt.Y("arm_label:N", sort=ARM_ORDER, title=None, axis=alt.Axis(labelLimit=240)),
        color=alt.Color(
            "arm_label:N", scale=alt.Scale(domain=ARM_ORDER, range=ARM_COLOURS), legend=None
        ),
    )
    title = "Mean pinball loss minus the stored forecast's (points of capacity; below 0 is better)"
    interval = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("diff_low:Q", title=title), x2="diff_high:Q"
    )
    dots = base.mark_point(filled=True, size=60, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("diff_vs_a0:Q", title=title)
    )
    zero = alt.Chart().mark_rule(color="black").encode(x="zero:Q")  # ty: ignore[unresolved-attribute]
    return (
        alt.layer(zero, interval, dots)
        .properties(width=WIDTH, height=170)
        .facet(
            row=alt.Row(
                "class:N", title=None, header=alt.Header(labelAnchor="start", labelFontSize=12)
            ),
            data=scores,
        )
        .resolve_scale(x="independent")
        .properties(
            title=_title(
                text="Quantile score change against the stored forecast",
                lines=[
                    (
                        "Mean pinball loss over the 11 stored levels, 12 to 36 hour lead, minus "
                        "arm 0's. Dot: estimate. Line: 95% interval from resampling whole months. "
                        "Most arms leave the score unchanged; arms 1, 7a, and 7b worsen it."
                    )
                ],
            )
        )
    )


def figure_reliability() -> alt.Chart:
    """Coverage minus nominal at every output level, Primary, 12 to 36 hour lead."""
    cov = (
        pl.read_parquet(report.RESULTS_DIR / "coverage.parquet")
        .filter(
            (pl.col("class") == "Primary")
            & (pl.col("band") == DECISION_BAND)
            & pl.col("arm").is_in(FIGURE_4_ARMS)
        )
        .with_columns(arm_label=_arm_label(), gap=pl.col("coverage") - pl.col("level"))
        .with_columns(level_label=pl.col("level").cast(pl.String))
    )
    levels = [str(p) for p in sorted(cov["level"].unique())]
    return (
        alt.Chart(cov)
        .mark_line(point=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("level_label:O", sort=levels, title="Nominal quantile level"),
            y=alt.Y("gap:Q", title="Coverage minus nominal level"),
            color=alt.Color(
                "arm_label:N",
                scale=alt.Scale(
                    domain=[ARM_LABELS[a] for a in FIGURE_4_ARMS],
                    range=[ARM_COLOURS[list(ARM_LABELS).index(a)] for a in FIGURE_4_ARMS],
                ),
                legend=alt.Legend(orient="top", title=None, columns=2, labelLimit=300),
            ),
        )
        .properties(
            width=WIDTH + 100,
            height=240,
            title=_title(
                text="Primary substations: where each arm over- or under-covers",
                lines=[
                    (
                        "Coverage minus nominal at every output level, 12 to 36 hour lead. "
                        "Above 0: the quantile is too high; below 0: too low. Levels 0.001 to "
                        "0.005 and 0.995 to 0.999 "
                        "are extrapolated or fitted, not stored."
                    )
                ],
            ),
        )
    )


def main() -> None:
    """Write the SVGs next to the page."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "figure_1_primary_tail_coverage": figure_coverage(cls="Primary", name="primary"),
        "figure_2_supply_point_tail_coverage": figure_coverage(cls="BSP/GSP", name="supply"),
        "figure_3_quantile_score_change": figure_score(),
        "figure_4_reliability": figure_reliability(),
    }
    for name, chart in figures.items():
        chart.save(DOCS_DIR / f"{name}.svg")
        print("wrote", name)


if __name__ == "__main__":
    main()
