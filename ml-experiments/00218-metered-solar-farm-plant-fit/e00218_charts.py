"""Draw the anonymised figures for experiment 218 from the tables the earlier scripts wrote."""

from pathlib import Path
from typing import Final

import altair as alt
import e00218_common as common
import e00218_fit as fit
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / common.SLUG
OBSERVED_COLOUR: Final[str] = "#222222"
MONTH_NAMES: Final[dict[int, str]] = {4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug"}
ARM_LABELS: Final[dict[str, str]] = {
    "arm1_fleet_shape": "Arm 1: fleet shape (scale only)",
    "arm2_per_farm": "Arm 2: per-farm plant",
    "arm3_learned_response": "Arm 3: learned response",
    "arm4_heat_off": "Arm 4: per-farm plant, no heat loss",
    "ablation_no_clip": "Ablation: per-farm plant, no clip",
}
FARM_FIGURE_ARMS: Final[tuple[str, ...]] = (
    "arm1_fleet_shape",
    "arm2_per_farm",
    "arm3_learned_response",
)
PLOT_WIDTH: Final[int] = CONTENT_WIDTH_PX - 300


def headline() -> None:
    """Figure 1: median fraction fitted per arm, and the planned contrasts, fold 1."""
    intervals = pl.read_csv(common.DATA_DIR / "intervals.csv").filter(pl.col("fold") == 1)
    arms = intervals.filter(pl.col("name").is_in(list(ARM_LABELS))).with_columns(
        label=pl.col("name").replace_strict(ARM_LABELS)
    )
    base = alt.Chart(arms)
    y = alt.Y("label:N", sort=list(ARM_LABELS.values()), title=None, axis=alt.Axis(labelLimit=260))
    rule = (
        alt.Chart(pl.DataFrame({"x": [0.97], "text": ["Predicted: 0.97"]}))
        .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    chart = (
        base.mark_rule(color=ocf.DATA_BLUE, size=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "low:Q",
                title="Fraction of output fitted (higher is better)",
                scale=alt.Scale(domain=[0.9, 0.98]),
            ),
            x2="high:Q",
            y=y,
        )
        + base.mark_point(color=ocf.DATA_BLUE, filled=True, size=80).encode(x="point:Q", y=y)  # ty: ignore[unresolved-attribute]
        + rule
    ).properties(
        width=PLOT_WIDTH,
        height=150,
        title=alt.TitleParams(
            [
                "Fraction of metered output fitted, median over farms A to F.",
                "Tuned before 2025-09, scored 2025-09 to 2026-08, CAMS irradiance.",
                "Dot: estimate. Line: 98.75% interval from resampling months.",
                "Orange dashed line: the pre-registered prediction (0.97).",
            ],
            anchor="start",
            fontSize=12,
            fontWeight="normal",
        ),
    )
    chart.save(DOCS_DIR / "figure_1_headline.svg")

    contrasts = intervals.filter(pl.col("name").str.starts_with("C")).with_columns(
        label=pl.col("name")
    )
    cbase = alt.Chart(contrasts)
    cy = alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=260))
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#888888").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    threshold = (
        alt.Chart(pl.DataFrame({"x": [0.02]}))
        .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    contrast_chart = (
        cbase.mark_rule(color=ocf.DATA_BLUE, size=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "low:Q",
                title="Difference in fraction fitted (positive: first arm better)",
                scale=alt.Scale(domain=[-0.025, 0.03]),
            ),
            x2="high:Q",
            y=cy,
        )
        + cbase.mark_point(color=ocf.DATA_BLUE, filled=True, size=80).encode(x="point:Q", y=cy)  # ty: ignore[unresolved-attribute]
        + zero
        + threshold
    ).properties(
        width=PLOT_WIDTH,
        height=110,
        title=alt.TitleParams(
            [
                "Planned contrasts, median over farms, rows of figure 1.",
                "Grey: no difference. Orange dashed: smallest gain worth porting.",
            ],
            anchor="start",
            fontSize=12,
            fontWeight="normal",
        ),
    )
    contrast_chart.save(DOCS_DIR / "figure_2_contrasts.svg")


def per_farm() -> None:
    """Figure 3: fraction fitted per farm for arms 1, 2, and 3, fold 1."""
    scores = (
        pl.read_csv(common.DATA_DIR / "per_farm_scores.csv")
        .filter((pl.col("fold") == 1) & pl.col("arm").is_in(FARM_FIGURE_ARMS))
        .with_columns(label=pl.col("arm").replace_strict(ARM_LABELS))
    )
    chart = (
        alt.Chart(scores)
        .mark_point(filled=True, size=90)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "fraction:Q",
                title="Fraction of output fitted (higher is better)",
                scale=alt.Scale(domain=[0.85, 0.97]),
            ),
            y=alt.Y("site:N", title="Farm"),
            color=alt.Color(
                "label:N",
                title=None,
                sort=[ARM_LABELS[a] for a in FARM_FIGURE_ARMS],
                scale=alt.Scale(
                    domain=[ARM_LABELS[a] for a in FARM_FIGURE_ARMS],
                    range=[OBSERVED_COLOUR, ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(orient="top", columns=1, labelLimit=320),
            ),
        )
        .properties(
            width=PLOT_WIDTH,
            height=170,
            title=alt.TitleParams(
                [
                    "Fraction of output fitted per farm, 2025-09 to 2026-08.",
                    "Farm E has only 10 tuning months.",
                ],
                anchor="start",
                fontSize=12,
                fontWeight="normal",
            ),
        )
    )
    chart.save(DOCS_DIR / "figure_3_per_farm.svg")


def flat_top() -> None:
    """Figure 4: monthly 90th-percentile flat level, observed and modelled, farms C to F."""
    predictions = pl.read_parquet(fit.PREDICTIONS_PATH).filter(
        (pl.col("fold") == 1)
        & (pl.col("clear_sky") > common.FLAT_TOP_CLEAR_SKY_W_M2)
        & pl.col("month").is_in(common.FLAT_TOP_MONTHS)
        & pl.col("site").is_in(common.FLAT_TOP_FARMS)
    )
    series = {
        "output": "Metered",
        "arm1_fleet_shape": "Arm 1: fleet shape",
        "arm2_per_farm": "Arm 2: per-farm plant",
        "ablation_no_clip": "Per-farm plant, no clip",
    }
    levels = pl.concat(
        [
            predictions.group_by("site", "month")
            .agg(level=pl.col(column).cast(pl.Float64).quantile(0.9), n=pl.len())
            .filter(pl.col("n") >= 30)
            .with_columns(
                series=pl.lit(label), month_name=pl.col("month").replace_strict(MONTH_NAMES)
            )
            for column, label in series.items()
        ]
    )
    chart = (
        alt.Chart(levels)
        .mark_line(point=True, clip=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "month_name:N",
                sort=list(MONTH_NAMES.values()),
                title="Month",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("level:Q", title="Power over capacity", scale=alt.Scale(domain=[0.85, 1.1])),
            color=alt.Color(
                "series:N",
                title=None,
                sort=list(series.values()),
                scale=alt.Scale(
                    domain=list(series.values()),
                    range=[OBSERVED_COLOUR, ocf.DATA_SKY, ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(width=PLOT_WIDTH + 100, height=100)
        .facet(
            row=alt.Row("site:N", title="Farm"),
            title=alt.TitleParams(
                [
                    "90th percentile of power over capacity by month, at clear-sky",
                    "irradiance above 650 W/m², farms C to F; all fold-1 rows.",
                ],
                anchor="start",
                fontSize=12,
                fontWeight="normal",
            ),
        )
    )
    chart.save(DOCS_DIR / "figure_4_flat_top.svg")


if __name__ == "__main__":
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    headline()
    per_farm()
    flat_top()
