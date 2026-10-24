# ruff: noqa: E501
"""Draw experiment 318's figures from the score tables (farms pooled; no farm series is drawn)."""

from pathlib import Path
from typing import Any, Final

import altair as alt
import e00318_common as common
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / common.SLUG
WIDTH: Final[int] = CONTENT_WIDTH_PX - 140
BANDS: Final[list[str]] = ["d1", "d2", "d3", "d4_5", "d6_7", "d8_10"]
BAND_LABELS: Final[dict[str, str]] = {
    "d1": "day 1",
    "d2": "day 2",
    "d3": "day 3",
    "d4_5": "days 4-5",
    "d6_7": "days 6-7",
    "d8_10": "days 8-10",
}
LABELS: Final[dict[str, str]] = {
    "lead": "default + lead day",
    "full": "default + all added inputs",
    "full_pooled": "all added inputs, one head for all leads",
    "full_shuffled": "added inputs shuffled (control)",
    "default": "default",
    "pure_pooled": "no plant, one head for all leads",
    "pure_lead_pooled": "no plant + lead day, one head",
    "ref_pure_mdn313": "no plant, one head per band (experiment 313)",
    "floor_climatology": "climatology",
    "floor_heuristic": "manual_heuristic",
}
COLOURS: Final[list[str]] = [
    ocf.DATA_SKY,
    ocf.BRAND_ORANGE,
    ocf.DATA_BLUE,
    "#888888",
    ocf.DATA_GREEN,
]


def _band_axis() -> Any:
    return alt.X(
        "band_label:N", sort=[BAND_LABELS[b] for b in BANDS], title="Lead band (days ahead)"
    )


def contrast_figure(
    *, contrasts: pl.DataFrame, metric: str, arms: list[str], title: str, ytitle: str, name: str
) -> None:
    """Draw points of change against `default` with intervals, own history."""
    data = (
        contrasts.filter(
            (pl.col("metric") == metric)
            & (pl.col("reference") == "default")
            & (pl.col("setting") == "own_history")
            & (pl.col("half") == "all")
            & pl.col("arm").is_in(arms)
        )
        .with_columns(
            band_label=pl.col("band").replace_strict(BAND_LABELS),
            label=pl.col("arm").replace_strict(LABELS),
        )
        .to_pandas()
    )
    colour = alt.Color(
        "label:N",
        sort=[LABELS[a] for a in arms],
        scale=alt.Scale(domain=[LABELS[a] for a in arms], range=COLOURS[: len(arms)]),
        legend=alt.Legend(title=None, orient="top", columns=2),
    )
    base: Any = alt.Chart(data).encode(x=_band_axis(), color=colour, xOffset="label:N")
    chart_data: Any = alt.Chart(data)
    rule: Any = chart_data.mark_rule(strokeDash=[3, 3], color="#444444").encode(y=alt.datum(0))
    chart = (
        rule
        + base.mark_errorbar(thickness=2).encode(
            y=alt.Y("points_lo:Q", title=ytitle), y2="points_hi:Q"
        )
        + base.mark_point(filled=True, size=50).encode(y="points:Q")
    ).properties(
        width=WIDTH,
        height=260,
        title=alt.TitleParams(title, anchor="start", fontSize=12, fontWeight="normal"),
    )
    chart.save(DOCS_DIR / name)


def absolute_figure(*, scores: pl.DataFrame) -> None:
    """Draw the absolute quantile score by band, own history."""
    arms = [
        "floor_climatology",
        "floor_heuristic",
        "ref_pure_mdn313",
        "pure_pooled",
        "default",
        "full_pooled",
    ]
    data = (
        scores.filter(
            (pl.col("metric") == "score")
            & (pl.col("setting") == "own_history")
            & (pl.col("half") == "all")
            & pl.col("arm").is_in(arms)
        )
        .with_columns(
            band_label=pl.col("band").replace_strict(BAND_LABELS),
            label=pl.col("arm").replace_strict(LABELS),
        )
        .to_pandas()
    )
    palette = [
        "#888888",
        "#444444",
        ocf.DATA_PURPLE,
        ocf.DATA_GREEN,
        ocf.DATA_BLUE,
        ocf.BRAND_ORANGE,
    ]
    chart_data: Any = alt.Chart(data)
    chart = (
        chart_data.mark_line(point=True)
        .encode(
            x=_band_axis(),
            y=alt.Y(
                "value:Q",
                title="Quantile score (points of capacity; lower is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "label:N",
                sort=[LABELS[a] for a in arms],
                scale=alt.Scale(domain=[LABELS[a] for a in arms], range=palette),
                legend=alt.Legend(title=None, orient="top", columns=2),
            ),
        )
        .properties(width=WIDTH, height=280)
    )
    chart.save(DOCS_DIR / "figure_3_absolute_score.svg")


def coverage_figure(*, scores: pl.DataFrame) -> None:
    """Draw the coverage of the 0.01 quantile by band, own history."""
    arms = ["default", "full", "full_pooled", "pure_pooled"]
    data = (
        scores.filter(
            (pl.col("metric") == "cover01")
            & (pl.col("setting") == "own_history")
            & (pl.col("half") == "all")
            & pl.col("arm").is_in(arms)
        )
        .with_columns(
            band_label=pl.col("band").replace_strict(BAND_LABELS),
            label=pl.col("arm").replace_strict(LABELS),
            nominal=pl.lit(0.01),
        )
        .to_pandas()
    )
    base: Any = alt.Chart(data)
    lines = base.mark_line(point=True).encode(
        x=_band_axis(),
        y=alt.Y(
            "value:Q",
            title="Share of actual values below the 0.01 quantile",
            scale=alt.Scale(domain=[0, 0.045]),
        ),
        color=alt.Color(
            "label:N",
            sort=[LABELS[a] for a in arms],
            scale=alt.Scale(domain=[LABELS[a] for a in arms], range=COLOURS[:4]),
            legend=alt.Legend(title=None, orient="top", columns=2),
        ),
    )
    nominal = base.mark_rule(strokeDash=[3, 3], color="#444444").encode(y="nominal:Q")
    (nominal + lines).properties(width=WIDTH, height=240).save(
        DOCS_DIR / "figure_4_q01_coverage.svg"
    )


def main() -> None:
    """Write the four figures."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    contrasts = pl.read_csv(common.DATA_DIR / "contrasts.csv")
    scores = pl.read_csv(common.DATA_DIR / "scores.csv")
    contrast_figure(
        contrasts=contrasts,
        metric="pin99",
        arms=["full", "full_pooled"],
        title="0.99 pinball loss, change against the default head (below zero is better; 99.17% intervals)",
        ytitle="Change in 0.99 pinball loss (points of capacity)",
        name="figure_1_pinball99_change.svg",
    )
    contrast_figure(
        contrasts=contrasts,
        metric="score",
        arms=["lead", "full", "full_pooled"],
        title="Quantile score, change against the default head (below zero is better; 99.17% intervals)",
        ytitle="Change in quantile score (points of capacity)",
        name="figure_2_score_change.svg",
    )
    absolute_figure(scores=scores)
    coverage_figure(scores=scores)


if __name__ == "__main__":
    main()
