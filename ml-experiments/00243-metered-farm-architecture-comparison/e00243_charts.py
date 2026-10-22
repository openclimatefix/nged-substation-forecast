# ruff: noqa: E501
"""Draw experiment 243's figures from the score tables (no farm series are drawn)."""

from pathlib import Path
from typing import Final

import altair as alt
import e00243_common as common
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / common.SLUG
PLOT_WIDTH: Final[int] = CONTENT_WIDTH_PX - 300
ARMS: Final[dict[str, str]] = {
    "h5_quantile": "H5: plant + network offsets for all 11 quantiles",
    "xgboost": "XGBoost",
    "nn_pure": "Neural network, no physics",
    "h2_residual": "H2: plant + network correction",
    "h1_scale": "H1: plant x network scale",
    "h4_both": "H4: adapter, then scale",
    "h3_adapter": "H3: network adjusts irradiance and temperature",
    "plant": "Plant alone",
    "h6_blind": "H6: blind correction (no plant input)",
    "floor_climatology": "Floor: climatology",
    "floor_heuristic": "Floor: manual heuristic",
}
SETTING_LABELS: Final[dict[str, str]] = {
    "own_history": "Own history",
    "zero_history": "Zero history",
}
SETTING_COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE]


def _title(lines: list[str]) -> alt.TitleParams:
    return alt.TitleParams(lines, anchor="start", fontSize=12, fontWeight="normal")


def _setting_color() -> alt.Color:
    return alt.Color(
        "setting_label:N",
        scale=alt.Scale(domain=list(SETTING_LABELS.values()), range=SETTING_COLOURS),
        legend=alt.Legend(title=None, orient="top"),
    )


def headline() -> None:
    """Figure 1: quantile score per arm and setting."""
    scores = (
        pl.read_csv(common.DATA_DIR / "scores.csv")
        .filter(
            (pl.col("slice") == "all")
            & (pl.col("metric") == "score")
            & pl.col("arm").is_in(list(ARMS))
        )
        .with_columns(
            label=pl.col("arm").replace_strict(ARMS),
            setting_label=pl.col("setting").replace_strict(SETTING_LABELS),
        )
    )
    y = alt.Y("label:N", sort=list(ARMS.values()), title=None, axis=alt.Axis(labelLimit=330))
    base = alt.Chart(scores)
    chart = (
        base.mark_rule(size=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lo:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(domain=[4.8, 9.4]),
            ),
            x2="hi:Q",
            y=y,
            color=_setting_color(),
            yOffset="setting_label:N",
        )
        + base.mark_point(filled=True, size=55).encode(  # ty: ignore[unresolved-attribute]
            x="value:Q", y=y, color=_setting_color(), yOffset="setting_label:N"
        )
    ).properties(
        width=PLOT_WIDTH,
        height=330,
        title=_title(
            [
                "Quantile score of the day-ahead forecast of six metered solar farms (A to F), mean over farms,",
                "12 to 36 h leads, daylight half-hours, 2025-09 to 2026-08. Own history: the plant scale is fitted on the",
                "farm's own earlier months. Zero history: the scale is the median of the other five farms. Dot: estimate (mean",
                "over five seeds for networks). Line: 95% interval from resampling whole months.",
            ]
        ),
    )
    chart.save(DOCS_DIR / "figure_1_headline.svg")


def contrasts() -> None:
    """Figure 2: percentage change against the plant and against XGBoost."""
    c = pl.read_csv(common.DATA_DIR / "contrasts.csv").filter(
        (pl.col("slice") == "all") & (pl.col("metric") == "score") & (pl.col("arm") != "h6_blind")
    )
    charts = []
    for ref, name in (("plant", "against the plant alone"), ("xgboost", "against XGBoost")):
        table = c.filter(pl.col("reference") == ref).with_columns(
            label=pl.col("arm").replace_strict(ARMS),
            setting_label=pl.col("setting").replace_strict(SETTING_LABELS),
        )
        base = alt.Chart(table)
        y = alt.Y(
            "label:N",
            sort=[
                ARMS[a]
                for a in (
                    "h5_quantile",
                    "nn_pure",
                    "h2_residual",
                    "h1_scale",
                    "h4_both",
                    "h3_adapter",
                )
            ],
            title=None,
            axis=alt.Axis(labelLimit=330),
        )
        zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#888888").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        charts.append(
            (
                base.mark_rule(size=2).encode(  # ty: ignore[unresolved-attribute]
                    x=alt.X(
                        "lo:Q", title=f"Change in quantile score {name} (%; negative is better)"
                    ),
                    x2="hi:Q",
                    y=y,
                    color=_setting_color(),
                    yOffset="setting_label:N",
                )
                + base.mark_point(filled=True, size=55).encode(  # ty: ignore[unresolved-attribute]
                    x="percent_change:Q", y=y, color=_setting_color(), yOffset="setting_label:N"
                )
                + zero
            ).properties(width=PLOT_WIDTH, height=190)
        )
    chart = alt.vconcat(*charts).properties(
        title=_title(
            [
                "Change in quantile score, mean over six farms, 12 to 36 h, 2025-09 to 2026-08.",
                "Dot: estimate. Line: 95% interval from resampling whole months. Grey rule: no change.",
            ]
        )
    )
    chart.save(DOCS_DIR / "figure_2_contrasts.svg")


def tails() -> None:
    """Figure 3: pinball loss at the 0.01 and 0.99 quantiles."""
    arms = {k: ARMS[k] for k in ("h5_quantile", "xgboost", "nn_pure", "h2_residual", "plant")}
    frames = []
    for metric, level in (("pin01", "0.01 quantile"), ("pin99", "0.99 quantile")):
        frames.append(
            pl.read_csv(common.DATA_DIR / "scores.csv")
            .filter(
                (pl.col("slice") == "all")
                & (pl.col("metric") == metric)
                & pl.col("arm").is_in(list(arms))
            )
            .with_columns(level=pl.lit(level))
        )
    table = pl.concat(frames).with_columns(
        label=pl.col("arm").replace_strict(arms),
        setting_label=pl.col("setting").replace_strict(SETTING_LABELS),
    )
    chart = (
        alt.Chart(table)
        .mark_point(filled=True, size=70)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "value:Q",
                title="Pinball loss (points of capacity; smaller is better)",
                scale=alt.Scale(domain=[0.3, 0.5]),
                axis=alt.Axis(values=[0.3, 0.35, 0.4, 0.45, 0.5]),
            ),
            y=alt.Y("label:N", sort=list(arms.values()), title=None, axis=alt.Axis(labelLimit=330)),
            color=_setting_color(),
            row=alt.Row("level:N", title=None),
        )
        .properties(
            width=PLOT_WIDTH,
            height=130,
            title=_title(
                [
                    "Pinball loss at the two tails, mean over six farms, 12 to 36 h, 2025-09 to 2026-08."
                ]
            ),
        )
    )
    chart.save(DOCS_DIR / "figure_3_tails.svg")


def cloudy() -> None:
    """Figure 4: change against the plant on cloudy and clear half-hours."""
    c = pl.read_csv(common.DATA_DIR / "contrasts.csv").filter(
        (pl.col("metric") == "score")
        & (pl.col("reference") == "plant")
        & (pl.col("slice") != "all")
        & (pl.col("setting") == "own_history")
        & (pl.col("arm") != "h6_blind")
    )
    table = c.with_columns(
        label=pl.col("arm").replace_strict(ARMS),
        slice_label=pl.col("slice").replace_strict({"cloudy": "Cloudy", "clear": "Clear"}),
    )
    base = alt.Chart(table)
    y = alt.Y(
        "label:N",
        sort=[
            ARMS[a]
            for a in ("h5_quantile", "nn_pure", "h2_residual", "h1_scale", "h4_both", "h3_adapter")
        ],
        title=None,
        axis=alt.Axis(labelLimit=330),
    )
    col = alt.Color(
        "slice_label:N",
        scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
        legend=alt.Legend(title=None, orient="top"),
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#888888").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    chart = (
        base.mark_rule(size=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lo:Q", title="Change in quantile score against the plant (%; negative is better)"
            ),
            x2="hi:Q",
            y=y,
            color=col,
            yOffset="slice_label:N",
        )
        + base.mark_point(filled=True, size=55).encode(  # ty: ignore[unresolved-attribute]
            x="percent_change:Q", y=y, color=col, yOffset="slice_label:N"
        )
        + zero
    ).properties(
        width=PLOT_WIDTH,
        height=230,
        title=_title(
            [
                "Change in quantile score against the plant alone, own-history setting, 12 to 36 h,",
                "cloudy (ensemble-mean clear-sky index under 0.5) or clear (0.5 or more) half-hours. Dot: estimate. Line: 95% interval.",
            ]
        ),
    )
    chart.save(DOCS_DIR / "figure_4_cloudy.svg")


if __name__ == "__main__":
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    headline()
    contrasts()
    tails()
    cloudy()
