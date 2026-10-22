"""Draw the figures for experiment 224 from the tables `e00224_score.py` wrote (no farm series)."""

from pathlib import Path
from typing import Final

import altair as alt
import e00224_common as common
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / common.SLUG
PLOT_WIDTH: Final[int] = CONTENT_WIDTH_PX - 300
ARMS: Final[dict[str, str]] = {
    "a5_xgboost": "XGBoost",
    "a4_members": "Plant, all 51 members",
    "a3_two_number": "Plant, mean and spread of 51 members",
    "a2_mean": "Plant, ensemble mean",
    "a2b_rescale": "Placebo: members replaced by their mean",
    "p3_strong": "Placebo: spread from another day (two-number)",
    "floor_climatology": "Floor: climatology",
    "floor_heuristic": "Floor: manual heuristic",
}
BAND_LABELS: Final[dict[int, str]] = {0: "12 to 36 h", 1: "60 to 84 h", 2: "108 to 132 h"}


def _title(lines: list[str]) -> alt.TitleParams:
    return alt.TitleParams(lines, anchor="start", fontSize=12, fontWeight="normal")


def headline() -> None:
    """Figure 1: quantile score per arm in the scored year at 12 to 36 h."""
    scores = (
        pl.read_csv(common.DATA_DIR / "scores.csv")
        .filter((pl.col("fold") == "scored_year") & (pl.col("band") == 0))
        .filter(pl.col("arm").is_in(list(ARMS)))
        .with_columns(label=pl.col("arm").replace_strict(ARMS))
    )
    y = alt.Y("label:N", sort=list(ARMS.values()), title=None, axis=alt.Axis(labelLimit=300))
    base = alt.Chart(scores)
    chart = (
        base.mark_rule(color=ocf.DATA_BLUE, size=3).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lo:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(domain=[4.5, 9.5]),
            ),
            x2="hi:Q",
            y=y,
        )
        + base.mark_point(color=ocf.DATA_BLUE, filled=True, size=80).encode(x="score:Q", y=y)  # ty: ignore[unresolved-attribute]
    ).properties(
        width=PLOT_WIDTH,
        height=230,
        title=_title(
            [
                "Quantile score of the day-ahead forecast of six metered solar farms (A to F),",
                "mean over farms, daylight half-hours at 12 to 36 h leads, 2025-09 to 2026-08.",
                "Dot: estimate. Line: 95% interval from resampling whole months.",
            ]
        ),
    )
    chart.save(DOCS_DIR / "figure_1_headline.svg")


def contrasts() -> None:
    """Figure 2: the percentage change of key contrasts, by lead band, scored year."""
    names = {
        "C0 members minus mean": "All 51 members minus ensemble mean",
        "C1 two-number minus mean": "Mean and spread minus ensemble mean",
        "C3 two-number minus XGBoost": "Mean and spread minus XGBoost",
        "C6 two-number minus strong shuffle": "Mean and spread minus placebo (spread from another day)",
    }
    table = (
        pl.read_csv(common.DATA_DIR / "contrasts.csv")
        .filter((pl.col("fold") == "scored_year") & pl.col("contrast").is_in(list(names)))
        .with_columns(
            label=pl.col("contrast").replace_strict(names),
            lead=pl.col("band").replace_strict(BAND_LABELS),
        )
    )
    y = alt.Y("label:N", sort=list(names.values()), title=None, axis=alt.Axis(labelLimit=330))
    base = alt.Chart(table)
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#888888").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    chart = (
        (
            base.mark_rule(size=3).encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("lo:Q", title="Change in quantile score (%; negative: first arm better)"),
                x2="hi:Q",
                y=y,
                color=alt.Color(
                    "lead:N",
                    sort=list(BAND_LABELS.values()),
                    scale=alt.Scale(
                        domain=list(BAND_LABELS.values()),
                        range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN],
                    ),
                    legend=alt.Legend(title="Lead", orient="top"),
                ),
                yOffset="lead:N",
            )
            + base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
                x="percent_change:Q",
                y=y,
                color=alt.Color("lead:N", legend=None),
                yOffset="lead:N",
            )
            + zero
        )
        .properties(
            width=PLOT_WIDTH,
            height=240,
            title=_title(
                [
                    "Contrasts in quantile score, mean over six farms, scored year 2025-09 to 2026-08.",
                    "Dot: estimate. Line: 95% interval from resampling whole months. Grey rule: no change.",
                ]
            ),
        )
    )
    chart.save(DOCS_DIR / "figure_2_contrasts.svg")


def per_farm() -> None:
    """Figure 3: each farm's gain of the member arms over the ensemble-mean arm at 12 to 36 h."""
    farms = pl.read_csv(common.DATA_DIR / "per_farm.csv").filter(
        (pl.col("fold") == "scored_year") & (pl.col("band") == 0)
    )
    score = farms.pivot(on="arm", index="farm", values="score")
    long = pl.concat(
        [
            score.select(
                "farm",
                arm=pl.lit("All 51 members"),
                change=(pl.col("a4_members") / pl.col("a2_mean") - 1) * 100,
            ),
            score.select(
                "farm",
                arm=pl.lit("Mean and spread of 51 members"),
                change=(pl.col("a3_two_number") / pl.col("a2_mean") - 1) * 100,
            ),
        ]
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [-2.0]}))
        .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    chart = (
        alt.Chart(long).mark_point(filled=True, size=90).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("change:Q", title="Change in quantile score (%; negative is better)"),
            y=alt.Y("farm:N", title="Farm", sort=list(common.FARMS)),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=["All 51 members", "Mean and spread of 51 members"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        + rule
    ).properties(
        width=PLOT_WIDTH,
        height=200,
        title=_title(
            [
                "Change in quantile score against the ensemble-mean arm, per farm, 12 to 36 h leads,",
                "2025-09 to 2026-08. Dashed orange line: the 2% gain that the stop rule asks for.",
            ]
        ),
    )
    chart.save(DOCS_DIR / "figure_3_per_farm.svg")


def tails() -> None:
    """Figure 4: pinball loss at the four tail levels for the main arms."""
    arms = {
        "a5_xgboost": "XGBoost",
        "a3_two_number": "Plant, mean and spread of 51 members",
        "a2_mean": "Plant, ensemble mean",
        "floor_climatology": "Floor: climatology",
    }
    table = (
        pl.read_csv(common.DATA_DIR / "tails.csv")
        .filter(
            (pl.col("fold") == "scored_year")
            & (pl.col("band") == 0)
            & pl.col("arm").is_in(list(arms))
        )
        .with_columns(
            label=pl.col("arm").replace_strict(arms), level_text=pl.col("level").cast(pl.String)
        )
    )
    chart = (
        alt.Chart(table)
        .mark_point(filled=True, size=90)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("pinball:Q", title="Pinball loss (points of capacity; smaller is better)"),
            y=alt.Y("level_text:N", title="Quantile level", sort=["0.01", "0.05", "0.95", "0.99"]),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(
                    domain=list(arms.values()),
                    range=[ocf.DATA_PURPLE, ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_SKY],
                ),
                legend=alt.Legend(title=None, orient="top", columns=2),
            ),
        )
        .properties(
            width=PLOT_WIDTH,
            height=220,
            title=_title(
                [
                    "Pinball loss at the four tail quantile levels, pooled over six farms,",
                    "12 to 36 h leads, 2025-09 to 2026-08.",
                ]
            ),
        )
    )
    chart.save(DOCS_DIR / "figure_4_tails.svg")


if __name__ == "__main__":
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    headline()
    contrasts()
    per_farm()
    tails()
