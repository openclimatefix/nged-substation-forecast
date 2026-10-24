# ruff: noqa: E501
"""Draw experiment 313's figures from the score tables (farms pooled; no farm series is drawn)."""

import sys
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00313_common as common
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / common.SLUG
PLOT_WIDTH: Final[int] = CONTENT_WIDTH_PX - 120
BAND_LABELS: Final[dict[str, str]] = {
    "d1": "day 1",
    "d2": "day 2",
    "d3": "day 3",
    "d4_5": "days 4 to 5",
    "d6_7": "days 6 to 7",
    "d8_10": "days 8 to 10",
}
ARM_LABELS: Final[dict[str, str]] = {
    "floor_climatology": "climatology",
    "mdn_ens": "Plant plus MDN head",
    "blend": "Plant-based blend",
    "pure_xgb": "Pure XGBoost",
    "pure_mdn": "Pure MDN",
    "pure_blend": "Pure blend (equal weights)",
}
ARM_COLOURS: Final[list[str]] = [
    "#888888",
    ocf.DATA_BLUE,
    ocf.DATA_SKY,
    ocf.DATA_GREEN,
    ocf.BRAND_ORANGE,
    ocf.DATA_PURPLE,
]
SETTING_LABELS: Final[dict[str, str]] = {
    "own_history": "Own history",
    "zero_history": "Zero history",
}


def _title(lines: list[str]) -> alt.TitleParams:
    lines = [part for line in lines for part in textwrap.wrap(line, 110)]
    return alt.TitleParams(lines, anchor="start", fontSize=12, fontWeight="normal")


def figure_solar_scores(*, scores: pl.DataFrame) -> None:
    """Quantile score by day band for the pure and plant-based arms."""
    panels = []
    for setting, label in SETTING_LABELS.items():
        data = scores.filter(
            (pl.col("metric") == "score")
            & (pl.col("setting") == setting)
            & pl.col("arm").is_in(list(ARM_LABELS))
        ).with_columns(
            label=pl.col("arm").replace_strict(ARM_LABELS),
            band_label=pl.col("band").replace_strict(BAND_LABELS),
        )
        base: Any = alt.Chart(data)
        chart: Any = base.mark_line(point=True).encode(
            x=alt.X(
                "band_label:N",
                sort=list(BAND_LABELS.values()),
                title="Lead (days ahead of the 00 UTC run)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "value:Q",
                title="Quantile score (points of capacity)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=list(ARM_LABELS.values()), range=ARM_COLOURS),
                legend=alt.Legend(title=None, orient="top", columns=3),
            ),
        )
        panels.append(chart.properties(width=PLOT_WIDTH, height=220, title=_title([label])))
    alt.vconcat(*panels).properties(
        title=_title(
            [
                "Figure 1: quantile score of the pure and plant-based solar forecasts by lead, farms A to F pooled. Lower is better."
            ]
        )
    ).save(DOCS_DIR / "figure_1_solar_scores.svg")


def figure_solar_contrasts(*, contrasts: pl.DataFrame) -> None:
    """Percentage change of the pure arms against the plant-based arms, with 98.75% intervals."""
    pairs = {
        ("pure_mdn", "mdn_ens"): "Pure MDN vs plant plus MDN head",
        ("pure_mdn", "blend"): "Pure MDN vs plant-based blend",
        ("pure_blend", "blend"): "Pure blend vs plant-based blend",
    }
    rows = [
        contrasts.filter(
            (pl.col("arm") == a) & (pl.col("reference") == r) & (pl.col("metric") == "score")
        ).with_columns(pair=pl.lit(label))
        for (a, r), label in pairs.items()
    ]
    data = pl.concat(rows).with_columns(
        band_label=pl.col("band").replace_strict(BAND_LABELS),
        setting_label=pl.col("setting").replace_strict(SETTING_LABELS),
    )
    base: Any = alt.Chart(data)
    y = alt.Y("band_label:N", sort=list(BAND_LABELS.values()), title=None)
    colour = alt.Color(
        "pair:N",
        scale=alt.Scale(
            domain=list(pairs.values()), range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE]
        ),
        legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=400),
    )
    rule = base.mark_rule().encode(y=y, x="lo:Q", x2="hi:Q", color=colour, yOffset="pair:N")
    point = base.mark_point(filled=True, size=40).encode(
        y=y,
        x=alt.X("percent_change:Q", title="Score change (%); negative favours pure"),
        color=colour,
        yOffset="pair:N",
    )
    zero_base: Any = alt.Chart(pl.DataFrame({"x": [0.0]}))
    zero = zero_base.mark_rule(strokeDash=[4, 3], color="#444444").encode(x="x:Q")
    (rule + point + zero).properties(width=PLOT_WIDTH // 2 - 20, height=200).facet(
        column=alt.Column("setting_label:N", title=None)
    ).properties(
        title=_title(
            [
                "Figure 2: the plant's added value: change in quantile score of each pure arm against a plant-based arm, with 98.75% intervals."
            ]
        )
    ).save(DOCS_DIR / "figure_2_solar_plant_value.svg")


def figure_wind(*, scores: pl.DataFrame) -> None:
    """Wind quantile scores by arm, history, and weather source (kept rows)."""
    labels = {
        "core": "Physics core",
        "xgb_binned_margin": "XGBoost, binned power curve",
        "neural_adapter": "Core with neural adapter",
        "pure_xgb": "Pure XGBoost",
        "pure_mdn": "Pure MDN",
    }
    data = scores.filter(
        (pl.col("metric") == "qs")
        & (pl.col("row_set") == "kept")
        & pl.col("arm").is_in(list(labels))
    ).with_columns(
        label=pl.col("arm").replace_strict(labels),
        group=pl.col("weather").replace_strict(
            {"era5": "ERA5 weather (a ceiling)", "ens": "Ensemble weather"}
        )
        + pl.lit(", ")
        + pl.col("history").replace_strict(SETTING_LABELS),
    )
    base: Any = alt.Chart(data)
    colour = alt.Color(
        "label:N",
        scale=alt.Scale(
            domain=list(labels.values()),
            range=["#888888", ocf.DATA_BLUE, ocf.DATA_SKY, ocf.DATA_GREEN, ocf.BRAND_ORANGE],
        ),
        legend=alt.Legend(title=None, orient="top", columns=3),
    )
    y = alt.Y("label:N", sort=list(labels.values()), title=None, axis=None)
    rule = base.mark_rule().encode(
        y=y, x=alt.X("lo:Q", scale=alt.Scale(zero=False)), x2="hi:Q", color=colour
    )
    point = base.mark_point(filled=True, size=50).encode(
        y=y, x=alt.X("value:Q", title="Quantile score (points of capacity)"), color=colour
    )
    (rule + point).properties(width=PLOT_WIDTH // 2 - 30, height=130).facet(
        column=alt.Column("group:N", title=None), columns=2
    ).properties(
        title=_title(
            [
                "Figure 3: wind quantile score by arm, W1 to W3 pooled, kept rows, with 98.75% month-resampled intervals. Lower is better."
            ]
        )
    ).save(DOCS_DIR / "figure_3_wind_scores.svg")


def main() -> None:
    """Draw the solar figures, and the wind figure when its scores exist."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figure_solar_scores(scores=pl.read_csv(common.DATA_DIR / "solar_scores.csv"))
    figure_solar_contrasts(contrasts=pl.read_csv(common.DATA_DIR / "solar_contrasts.csv"))
    if "--wind" in sys.argv:
        figure_wind(scores=pl.read_csv(common.DATA_DIR / "wind_scores.csv"))


if __name__ == "__main__":
    main()
