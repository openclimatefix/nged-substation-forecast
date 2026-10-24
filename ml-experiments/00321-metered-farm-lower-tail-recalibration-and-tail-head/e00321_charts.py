# ruff: noqa: E501
"""Draw experiment 321's figures from the score tables (farms pooled; no farm series is drawn)."""

from pathlib import Path
from typing import Any, Final

import altair as alt
import e00321_common as common
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
    "rd1": "research default 1 (blend)",
    "recal": "recalibration table",
    "recal_shrunk": "shrunk recalibration table",
    "recal_stale7": "table, one week stale (control)",
    "recal_swapped": "table of another farm (control)",
    "zi_blend": "zero-inflated head in the blend",
    "zi_alone": "zero-inflated head alone",
    "tail": "tail head in the blend",
    "tail_shuffled": "tail head, lead and clear-sky index shuffled (control)",
    "pure_lead_pooled": "no plant, one head for all leads",
    "floor_climatology": "climatology",
}
COLOURS: Final[dict[str, str]] = {
    "rd1": "#444444",
    "recal": ocf.DATA_SKY,
    "recal_shrunk": ocf.DATA_BLUE,
    "zi_blend": ocf.DATA_GREEN,
    "zi_alone": ocf.DATA_PURPLE,
    "tail": ocf.BRAND_ORANGE,
    "tail_shuffled": "#B5895A",
    "pure_lead_pooled": ocf.DATA_PURPLE,
    "recal_swapped": "#999999",
    "recal_stale7": "#9CC3E6",
    "floor_climatology": "#BBBBBB",
}


def band_axis() -> Any:
    """Return the shared lead-band axis."""
    return alt.X(
        "band_label:N", sort=[BAND_LABELS[b] for b in BANDS], title="Lead band (days ahead)"
    )


def colour(*, arms: list[str]) -> Any:
    """Return the colour encoding of the given arms."""
    return alt.Color(
        "label:N",
        sort=[LABELS[a] for a in arms],
        scale=alt.Scale(domain=[LABELS[a] for a in arms], range=[COLOURS[a] for a in arms]),
        legend=alt.Legend(title=None, orient="top", columns=2),
    )


def contrast_figure(
    *,
    contrasts: pl.DataFrame,
    setting: str,
    metric: str,
    arms: list[str],
    ytitle: str,
    title: str,
    name: str,
) -> None:
    """Draw the change against research default 1 with intervals (points, or percent for pinball)."""
    value, lo, hi = (
        ("points", "points_lo", "points_hi") if metric == "score" else ("percent", "lo", "hi")
    )
    data = (
        contrasts.filter(
            (pl.col("metric") == metric)
            & (pl.col("reference") == "rd1")
            & (pl.col("setting") == setting)
            & (pl.col("half") == "all")
            & pl.col("arm").is_in(arms)
        )
        .with_columns(
            band_label=pl.col("band").replace_strict(BAND_LABELS),
            label=pl.col("arm").replace_strict(LABELS),
        )
        .to_pandas()
    )
    chart_data: Any = alt.Chart(data)
    base: Any = chart_data.encode(x=band_axis(), color=colour(arms=arms), xOffset="label:N")
    rule: Any = chart_data.mark_rule(strokeDash=[3, 3], color="#444444").encode(y=alt.datum(0))
    chart = (
        rule
        + base.mark_errorbar(thickness=2).encode(y=alt.Y(f"{lo}:Q", title=ytitle), y2=f"{hi}:Q")
        + base.mark_point(filled=True, size=50).encode(y=f"{value}:Q")
    ).properties(
        width=WIDTH,
        height=260,
        title=alt.TitleParams(title, anchor="start", fontSize=12, fontWeight="normal"),
    )
    chart.save(DOCS_DIR / name)


def coverage_figure(
    *,
    scores: pl.DataFrame,
    setting: str,
    metric: str,
    arms: list[str],
    nominal: float,
    ytitle: str,
    domain: list[float],
    name: str,
) -> None:
    """Draw a coverage share against lead band with the nominal level marked."""
    data = (
        scores.filter(
            (pl.col("metric") == metric)
            & (pl.col("setting") == setting)
            & (pl.col("half") == "all")
            & pl.col("arm").is_in(arms)
        )
        .with_columns(
            band_label=pl.col("band").replace_strict(BAND_LABELS),
            label=pl.col("arm").replace_strict(LABELS),
            nominal=pl.lit(nominal),
        )
        .to_pandas()
    )
    base: Any = alt.Chart(data)
    lines = base.mark_line(point=True).encode(
        x=band_axis(),
        y=alt.Y("value:Q", title=ytitle, scale=alt.Scale(domain=domain)),
        color=colour(arms=arms),
    )
    rule = base.mark_rule(strokeDash=[3, 3], color="#444444").encode(y="nominal:Q")
    (rule + lines).properties(width=WIDTH, height=250).save(DOCS_DIR / name)


def main() -> None:
    """Write the figures."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    contrasts = pl.read_csv(common.DATA_DIR / "contrasts.csv")
    scores = pl.read_csv(common.DATA_DIR / "scores.csv")
    miss_arms = ["rd1", "recal", "zi_blend", "tail", "pure_lead_pooled", "floor_climatology"]
    for setting, number in (("own_history", 1), ("zero_history", 2)):
        coverage_figure(
            scores=scores,
            setting=setting,
            metric="strict01",
            arms=miss_arms,
            nominal=0.01,
            ytitle="Share of actual values strictly below the 0.01 quantile",
            domain=[0, 0.025],
            name=f"figure_{number}_q01_strict_{setting}.svg",
        )
    contrast_figure(
        contrasts=contrasts,
        setting="own_history",
        metric="pin99",
        arms=["tail", "tail_shuffled", "pure_lead_pooled", "zi_blend"],
        ytitle="Change in 0.99 pinball loss (%)",
        title="0.99 pinball loss against research default 1, own history (below zero is better; 99.83% intervals)",
        name="figure_3_pinball99_change.svg",
    )
    contrast_figure(
        contrasts=contrasts,
        setting="own_history",
        metric="score",
        arms=["recal", "recal_shrunk", "zi_blend", "tail", "pure_lead_pooled"],
        ytitle="Change in quantile score (points of capacity)",
        title="Quantile score against research default 1, own history (below zero is better; 99.83% intervals)",
        name="figure_4_score_change_own.svg",
    )
    contrast_figure(
        contrasts=contrasts,
        setting="zero_history",
        metric="score",
        arms=["recal", "recal_shrunk", "recal_swapped", "zi_blend", "tail", "pure_lead_pooled"],
        ytitle="Change in quantile score (points of capacity)",
        title="Quantile score against research default 1, zero history (below zero is better; 99.83% intervals)",
        name="figure_5_score_change_zero.svg",
    )
    both = (
        scores.filter(
            (pl.col("setting") == "own_history")
            & (pl.col("half") == "all")
            & (pl.col("arm") == "rd1")
            & pl.col("metric").is_in(["cover01", "strict01", "zero_actual"])
        )
        .with_columns(
            band_label=pl.col("band").replace_strict(BAND_LABELS),
            label=pl.col("metric").replace_strict(
                {
                    "cover01": "at or below the 0.01 quantile",
                    "strict01": "strictly below the 0.01 quantile",
                    "zero_actual": "actual value is exactly zero",
                }
            ),
        )
        .to_pandas()
    )
    base: Any = alt.Chart(both)
    (
        base.mark_line(point=True)
        .encode(
            x=band_axis(),
            y=alt.Y("value:Q", title="Share of half-hours", scale=alt.Scale(domain=[0, 0.04])),
            color=alt.Color("label:N", legend=alt.Legend(title=None, orient="top", columns=1)),
        )
        .properties(width=WIDTH, height=240)
        .save(DOCS_DIR / "figure_6_why_coverage_looked_too_high.svg")
    )


if __name__ == "__main__":
    main()
