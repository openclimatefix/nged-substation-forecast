# ruff: noqa: E501
"""Draw experiment 345's figures from the score tables (no farm series is drawn)."""

import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00345_common as common
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
    "solar_pooled": "solar-only pooled network",
    "both": "solar-plus-wind pooled network",
    "both_plant": "solar-plus-wind + plant column",
    "wind_pooled": "wind-only pooled network",
    "ref_blend310": "research default 1 (blend)",
    "ref_mdn_plant318": "network on the plant (single family)",
    "ref_pure313": "pure network (experiment 313)",
    "ref_pure_mdn313": "pure network (experiment 313)",
    "ref_adapter285": "physics core + neural adapter (experiment 285)",
    "floor_climatology": "climatology",
    "floor_heuristic": "manual_heuristic",
    "both_once": "solar-plus-wind, trained once",
    "both_monthly": "solar-plus-wind, monthly refit",
}
COLOUR: Final[dict[str, str]] = {
    "solar_pooled": ocf.DATA_BLUE,
    "both": ocf.BRAND_ORANGE,
    "both_plant": ocf.DATA_PURPLE,
    "wind_pooled": ocf.DATA_BLUE,
    "ref_blend310": ocf.DATA_GREEN,
    "ref_mdn_plant318": "#444444",
    "ref_pure313": ocf.DATA_SKY,
    "ref_pure_mdn313": "#444444",
    "ref_adapter285": ocf.DATA_GREEN,
    "floor_climatology": "#999999",
    "floor_heuristic": "#666666",
    "both_once": ocf.DATA_SKY,
    "both_monthly": ocf.DATA_GREEN,
}
SETTING_LABELS: Final[dict[str, str]] = {
    "own_history": "Own history",
    "zero_history": "Zero history (the farm is held out)",
}
LEGEND: Final = alt.Legend(title=None, orient="top", columns=2, labelLimit=400)


def wrap(lines: list[str]) -> list[str]:
    """Return the text wrapped to the plot area's width."""
    return textwrap.wrap(" ".join(lines), 78)


SORT: Final[list[str]] = [BAND_LABELS[b] for b in BANDS]


def prepare(frame: pl.DataFrame, arms: list[str]) -> Any:
    """Return pandas rows with labels for the chosen arms and bands."""
    return (
        frame.filter(pl.col("arm").is_in(arms) & pl.col("band").is_in(BANDS))
        .with_columns(
            band_label=pl.col("band").replace_strict(BAND_LABELS),
            label=pl.col("arm").replace_strict(LABELS),
            setting_label=pl.col("setting").replace_strict(SETTING_LABELS),
        )
        .to_pandas()
    )


def panels(
    *,
    data: Any,
    arms: list[str],
    y: str,
    lo: str,
    hi: str,
    ytitle: str,
    title: list[str],
    zero_rule: bool,
    name: str,
    settings: list[str] | None = None,
) -> None:
    """Draw one panel per history setting: a dot and a 98.75% interval per arm and band."""
    colour = alt.Color(
        "label:N",
        sort=[LABELS[a] for a in arms],
        scale=alt.Scale(domain=[LABELS[a] for a in arms], range=[COLOUR[a] for a in arms]),
        legend=LEGEND,
    )
    charts = []
    for setting in settings or ["own_history", "zero_history"]:
        sub = data[data["setting"] == setting]
        base: Any = alt.Chart(sub).encode(
            x=alt.X(
                "band_label:N",
                sort=SORT,
                title="Lead band (days ahead)",
                axis=alt.Axis(labelAngle=0),
            ),
            color=colour,
            xOffset="label:N",
        )
        layers = base.mark_errorbar(thickness=2).encode(
            y=alt.Y(f"{lo}:Q", title=ytitle), y2=f"{hi}:Q"
        ) + base.mark_point(filled=True, size=45).encode(y=f"{y}:Q")
        if zero_rule:
            zero: Any = alt.Chart(sub)
            layers = (
                zero.mark_rule(strokeDash=[3, 3], color="#444444").encode(y=alt.datum(0)) + layers
            )
        charts.append(
            layers.properties(
                width=WIDTH,
                height=190,
                title=alt.TitleParams(
                    SETTING_LABELS[setting], anchor="start", fontSize=12, fontWeight="normal"
                ),
            )
        )
    alt.vconcat(*charts).properties(
        title=alt.TitleParams(wrap(title), anchor="start", fontSize=12, fontWeight="normal")
    ).resolve_scale(color="shared").save(DOCS_DIR / name)


def main() -> None:
    """Draw every figure."""
    sc = pl.read_csv(common.DATA_DIR / "solar_scores.csv").filter(pl.col("half") == "all")
    cs = pl.read_csv(common.DATA_DIR / "solar_contrasts.csv").filter(pl.col("half") == "all")
    wc = pl.read_csv(common.DATA_DIR / "wind_contrasts.csv").filter(pl.col("half") == "all")

    # Figure 1: solar change in score against the single-family network on the plant.
    arms = ["solar_pooled", "both", "both_plant"]
    d = prepare(
        cs.filter(
            (pl.col("metric") == "score") & (pl.col("reference") == "ref_mdn_plant318")
        ).select("band", "setting", "arm", "points", "points_lo", "points_hi"),
        arms,
    )
    d_pl = cs.filter(
        (pl.col("metric") == "score")
        & (pl.col("arm") == "both_plant")
        & (pl.col("reference") == "both")
    )
    panels(
        data=d,
        arms=["solar_pooled", "both"],
        y="points",
        lo="points_lo",
        hi="points_hi",
        ytitle="Score change (points of capacity)",
        title=[
            "Solar farms A to F: one pooled network scores close to the single-family network on the plant.",
            "Dot: estimate. Line: 98.75% interval from resampling whole months. Zero: same as the network on the plant; below zero is better.",
        ],
        zero_rule=True,
        name="figure_1_solar_change_vs_single_family.svg",
    )
    del d_pl

    # Figure 2: wind change against experiment 313's pure network (ensemble weather).
    d = prepare(
        wc.filter(
            (pl.col("metric") == "score")
            & (pl.col("weather") == "ens")
            & (pl.col("reference") == "ref_pure_mdn313")
            & pl.col("arm").is_in(["wind_pooled", "both"])
        ).select("band", "setting", "arm", "points", "points_lo", "points_hi"),
        ["wind_pooled", "both"],
    )
    panels(
        data=d,
        arms=["wind_pooled", "both"],
        y="points",
        lo="points_lo",
        hi="points_hi",
        ytitle="Score change (points of capacity)",
        title=[
            "Wind farms W1 to W3, ensemble weather, 12 to 252 hours: with their own history the pooled networks match experiment 313's network from day 3 and trail it at days 1 to 2.",
            "Dot: estimate. Line: 98.75% interval from resampling whole months. Zero: same as experiment 313; below zero is better.",
        ],
        zero_rule=True,
        name="figure_2_wind_change_vs_313.svg",
    )

    # Figure 3: absolute scores.
    solar_arms = [
        "floor_climatology",
        "floor_heuristic",
        "ref_blend310",
        "ref_mdn_plant318",
        "solar_pooled",
        "both",
    ]
    d = prepare(
        sc.filter(pl.col("metric") == "score").select(
            "band", "setting", "arm", "value", "lo", "hi"
        ),
        solar_arms,
    )
    panels(
        data=d,
        arms=solar_arms,
        y="value",
        lo="lo",
        hi="hi",
        ytitle="Quantile score (points)",
        title=[
            "Solar farms A to F: absolute quantile score by lead band, equal weight per farm.",
            "Dot: estimate. Line: 98.75% interval from resampling whole months.",
        ],
        zero_rule=False,
        name="figure_3_solar_absolute_score.svg",
    )

    # Figure 4: pooling gain by farm in zero history.
    farm = pl.read_csv(common.DATA_DIR / "solar_farm_contrasts.csv").filter(
        (pl.col("bands") == "all")
        & (pl.col("arm") == "both")
        & (pl.col("reference") == "solar_pooled")
    )
    base: Any = alt.Chart(farm.to_pandas()).encode(
        x=alt.X("farm:N", title="Solar farm"),
        color=alt.Color(
            "setting:N",
            scale=alt.Scale(
                domain=["own_history", "zero_history"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
        xOffset="setting:N",
    )
    farm_rule: Any = alt.Chart(farm.to_pandas())
    (
        farm_rule.mark_rule(strokeDash=[3, 3], color="#444444").encode(y=alt.datum(0))
        + base.mark_errorbar(thickness=2).encode(
            y=alt.Y("lo:Q", title="Score change (%)"), y2="hi:Q"
        )
        + base.mark_point(filled=True, size=45).encode(y="percent:Q")
    ).properties(
        width=WIDTH,
        height=230,
        title=alt.TitleParams(
            wrap(
                [
                    "Adding wind farms to the solar network changes each solar farm's score by 0.7% to 2.6% for the worse.",
                    "Dot: estimate over days 1 to 10. Line: 98.75% interval. Zero: same as the solar-only pooled network.",
                ]
            ),
            anchor="start",
            fontSize=12,
            fontWeight="normal",
        ),
    ).save(DOCS_DIR / "figure_4_pooling_effect_by_farm.svg")

    # Figure 5: coverage at 0.01 and 0.99.
    cover = sc.filter(
        pl.col("metric").is_in(["cover01", "cover99"]) & (pl.col("setting") == "own_history")
    ).filter(pl.col("arm").is_in(["ref_blend310", "ref_mdn_plant318", "solar_pooled", "both"]))
    charts = []
    for metric, expected, label in (
        ("cover01", 0.01, "Share of actual values below the 0.01 quantile (expected 1%)"),
        ("cover99", 0.99, "Share of actual values below the 0.99 quantile (expected 99%)"),
    ):
        arms = ["ref_blend310", "ref_mdn_plant318", "solar_pooled", "both"]
        data = prepare(cover.filter(pl.col("metric") == metric), arms).assign(
            share=lambda f: 100 * f["value"]
        )
        colour = alt.Color(
            "label:N",
            sort=[LABELS[a] for a in arms],
            scale=alt.Scale(domain=[LABELS[a] for a in arms], range=[COLOUR[a] for a in arms]),
            legend=LEGEND,
        )
        source: Any = alt.Chart(data)
        line = source.mark_line(point=True).encode(
            x=alt.X(
                "band_label:N",
                sort=SORT,
                title="Lead band (days ahead)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("share:Q", title=label.split(" (")[0], scale=alt.Scale(zero=False)),
            color=colour,
        )
        rule = source.mark_rule(strokeDash=[3, 3], color="#444444").encode(
            y=alt.datum(100 * expected)
        )
        charts.append((rule + line).properties(width=WIDTH, height=170))
    alt.vconcat(*charts).properties(
        title=alt.TitleParams(
            wrap(
                [
                    "Solar farms, own history: no network brings the lower tail to 1%, and all sit close to 99% at the upper tail.",
                    "Dashed line: the expected share. Ties and zero output count as below the quantile.",
                ]
            ),
            anchor="start",
            fontSize=12,
            fontWeight="normal",
        )
    ).save(DOCS_DIR / "figure_5_tail_coverage.svg")


if __name__ == "__main__":
    main()
