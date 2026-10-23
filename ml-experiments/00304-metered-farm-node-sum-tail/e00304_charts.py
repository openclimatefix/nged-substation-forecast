# ruff: noqa: E501, ISC004
"""Draw experiment 304's figures from the score tables (farms pooled to nodes; no series drawn)."""

import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00304_data as data
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / f"{data.EXPERIMENT_ID:05d}-{data.SLUG}"
PLOT_WIDTH: Final[int] = CONTENT_WIDTH_PX - 260
FORM_LABELS: Final[dict[str, str]] = {
    "a_sum_of_quantiles": "a: add the farms' quantiles",
    "b_independent": "b: independent bands",
    "c_member_wise": "c: ensemble members, member by member",
    "c_shuffled": "c control: members shuffled across days",
    "d_copula": "d: fitted correlation (Gaussian copula)",
}
ARM_LABELS: Final[dict[str, str]] = {
    "p2": "Plant, binned (P2)",
    "blend": "Blend",
    "xgb_x2": "XGBoost",
}
DOMAINS: Final[dict[float, list[float]]] = {0.95: [0.88, 1.0], 0.99: [0.94, 1.0]}
"""Share axis range of each panel; the dots sit near their nominal level."""
ARM_COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE]


def _title(lines: list[str]) -> alt.TitleParams:
    lines = [part for line in lines for part in textwrap.wrap(line, 100)]
    return alt.TitleParams(lines, anchor="start", fontSize=12, fontWeight="normal")


def coverage_panel(*, scores: pl.DataFrame, level: float) -> alt.Chart:
    """Return one panel of coverage with month-block intervals at a nominal level."""
    name = f"hit_q{round(100 * level):02d}"
    rows = scores.filter(pl.col("metric") == name).with_columns(
        form_label=pl.col("form").replace(FORM_LABELS),
        arm_label=pl.col("arm").replace(ARM_LABELS),
    )
    base: Any = alt.Chart(rows)
    y = alt.Y(
        "form_label:N", sort=list(FORM_LABELS.values()), title=None, axis=alt.Axis(labelLimit=300)
    )
    colour = alt.Color(
        "arm_label:N",
        scale=alt.Scale(domain=list(ARM_LABELS.values()), range=ARM_COLOURS),
        legend=alt.Legend(title=None, orient="top"),
    )
    nominal: Any = alt.Chart(pl.DataFrame({"nominal": [level]}))
    return (
        nominal.mark_rule(strokeDash=[4, 3], color="#444").encode(
            x=alt.X("nominal:Q", scale=alt.Scale(domain=DOMAINS[level], clamp=True))
        )
        + base.mark_rule(size=2).encode(
            x=alt.X(
                "lo:Q",
                title=f"Share of half-hours at or below the {level} quantile of the sum",
                scale=alt.Scale(domain=DOMAINS[level], clamp=True),
            ),
            x2="hi:Q",
            y=y,
            color=colour,
            yOffset="arm_label:N",
        )
        + base.mark_point(filled=True, size=50).encode(
            x="value:Q", y=y, color=colour, yOffset="arm_label:N"
        )
    ).properties(width=PLOT_WIDTH, height=190, title=_title([f"Nominal level {level}"]))


def figure_coverage(*, scores: pl.DataFrame) -> alt.VConcatChart:
    """Return figure 1: coverage at the upper tail levels, 0.95 above and 0.99 below."""
    panels = [coverage_panel(scores=scores, level=level) for level in (0.95, 0.99)]
    return alt.vconcat(*panels).properties(
        title=_title(
            [
                "Figure 1: coverage of the sum of the three farms beneath a bulk supply point. Dot: share of "
                "half-hours (daylight, 12 to 36 hours ahead, both nodes pooled, September 2025 to August 2026) "
                "with the observed sum at or below the forecast quantile. Line: 98.75% interval from "
                "resampling whole months. Dashed rule: nominal level. A dot to the right of the rule "
                "means the band is wider than it needs to be."
            ]
        )
    )


def figure_dependence(*, diagnostic: pl.DataFrame) -> alt.Chart:
    """Return figure 2: correlation between farms implied by the ensemble versus in the residuals."""
    long = diagnostic.unpivot(
        index=["node", "band", "rows"],
        on=["ensemble_rank_correlation", "residual_level_correlation"],
    ).with_columns(
        source=pl.col("variable").replace(
            {
                "ensemble_rank_correlation": "Ensemble members' ranks",
                "residual_level_correlation": "Observed forecast errors",
            }
        )
    )
    base: Any = alt.Chart(long)
    chart = base.mark_bar().encode(
        x=alt.X("band:N", sort=["overcast", "broken", "clear", "all"], title="Clear-sky band"),
        xOffset="source:N",
        y=alt.Y(
            "value:Q",
            title="Mean correlation between the node's farms",
            scale=alt.Scale(domain=[0, 1]),
        ),
        color=alt.Color(
            "source:N",
            scale=alt.Scale(
                domain=[
                    "Ensemble members' ranks",
                    "Observed forecast errors",
                ],
                range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
            ),
            legend=alt.Legend(title=None, orient="top", labelLimit=400),
        ),
        column=alt.Column("node:N", title=None),
    )
    return chart.properties(
        width=(CONTENT_WIDTH_PX - 120) // 2,
        height=200,
        title=_title(
            [
                "Figure 2: the ensemble treats the three farms as nearly one farm. Bars: mean correlation "
                "between pairs of the node's farms of the level (position within the farm's forecast band) "
                "that each ensemble member takes (blue) and that each observation takes (orange, blend "
                "arm). Higher means the farms move more together."
            ]
        ),
    )


def main() -> None:
    """Write both figures as SVG into the experiment's docs folder."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    scores = pl.read_csv(data.DATA_DIR / "scores.csv")
    diagnostic = pl.read_csv(data.DATA_DIR / "dependence_diagnostic.csv")
    figure_coverage(scores=scores).save(DOCS_DIR / "figure_1_coverage.svg")
    figure_dependence(diagnostic=diagnostic).save(DOCS_DIR / "figure_2_dependence.svg")


if __name__ == "__main__":
    main()
