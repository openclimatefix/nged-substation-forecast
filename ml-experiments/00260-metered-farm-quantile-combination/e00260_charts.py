# ruff: noqa: E501
"""Draw experiment 260's figures from the score tables (no farm series are drawn)."""

from pathlib import Path
from typing import Final

import altair as alt
import e00260_common as common
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / common.SLUG
PLOT_WIDTH: Final[int] = CONTENT_WIDTH_PX - 300
ARMS: Final[dict[str, str]] = {
    "c2_lomo": "C2-LOMO: leave-one-month-out weights (optimistic)",
    "c2": "C2: per-level weights fitted on earlier months",
    "c1": "C1: equal weights (seen before the plan)",
    "c3": "C3: C2 plus adaptive tail adjustment",
    "h5_quantile": "H5: plant + neural network offsets",
    "xgboost": "XGBoost",
    "nn_pure": "Neural network, no plant",
    "plant": "Plant alone",
    "floor_climatology": "Floor: climatology",
    "floor_heuristic": "Floor: manual heuristic",
}
SETTING_LABELS: Final[dict[str, str]] = {
    "own_history": "Own history",
    "zero_history": "Zero history",
}
SETTING_COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE]
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1": "C1: C2 minus H5 (own history)",
    "C2": "C2: C2 minus neural network, pinball loss at 0.99 (own history)",
    "C3": "C3: C3 minus C2 (own history)",
    "C4": "C4: C2 minus H5 (zero history)",
}


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
                scale=alt.Scale(domain=[4.6, 9.4]),
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
                "12 to 36 h leads, daylight half-hours, 2025-09 to 2026-08. Own history: fitted on the farm's own earlier",
                "months. Zero history: fitted on the other five farms. C3 has no zero-history row. Dot: estimate (mean",
                "over five seeds for H5, XGBoost, and the neural network). Line: 95% interval from resampling whole months.",
            ]
        ),
    )
    chart.save(DOCS_DIR / "figure_1_headline.svg")


def contrasts() -> None:
    """Figure 2: the planned contrasts with 99% intervals."""
    c = (
        pl.read_csv(common.DATA_DIR / "contrasts.csv")
        .filter(
            (pl.col("slice") == "all")
            & pl.col("contrast").is_in(list(CONTRAST_LABELS))
            & (
                ((pl.col("contrast") == "C2") & (pl.col("metric") == "pin99"))
                | ((pl.col("contrast") != "C2") & (pl.col("metric") == "score"))
            )
        )
        .with_columns(name=pl.col("contrast").replace_strict(CONTRAST_LABELS))
    )
    y = alt.Y(
        "name:N", sort=list(CONTRAST_LABELS.values()), title=None, axis=alt.Axis(labelLimit=380)
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#888888").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    base = alt.Chart(c)
    chart = (
        base.mark_rule(size=2, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lo99:Q", title="Change (%; negative means the first arm is better)"),
            x2="hi99:Q",
            y=y,
        )
        + base.mark_point(filled=True, size=60, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x="percent_change:Q", y=y
        )
        + zero
    ).properties(
        width=PLOT_WIDTH,
        height=170,
        title=_title(
            [
                "Planned contrasts in quantile score (C2 in pinball loss at the 0.99 quantile), mean over six farms,",
                "12 to 36 h, 2025-09 to 2026-08. Dot: estimate. Line: 99% interval from resampling whole months",
                "(Bonferroni over four contrasts). Grey rule: no change.",
            ]
        ),
    )
    chart.save(DOCS_DIR / "figure_2_contrasts.svg")


def weights() -> None:
    """Figure 4: the weight each forecaster gets at each quantile level (own history, mean over months)."""
    w = (
        pl.read_csv(common.DATA_DIR / "weights.csv")
        .filter((pl.col("setting") == "own_history") & (pl.col("kind") == "c2_lomo"))
        .group_by("level")
        .agg(pl.col("h5_quantile").mean(), pl.col("nn_pure").mean(), pl.col("xgboost").mean())
        .unpivot(index="level", variable_name="forecaster", value_name="weight")
        .with_columns(
            forecaster=pl.col("forecaster").replace_strict(
                {"h5_quantile": "H5", "nn_pure": "Neural network, no plant", "xgboost": "XGBoost"}
            ),
            level_label=pl.col("level").cast(pl.String),
        )
    )
    chart = (
        alt.Chart(w)
        .mark_bar()
        .encode(
            x=alt.X(
                "level_label:O",
                title="Quantile level",
                sort=[str(v) for v in sorted(w["level"].unique().to_list())],
            ),
            y=alt.Y("weight:Q", title="Weight (mean over 12 months)", stack="zero"),
            color=alt.Color(
                "forecaster:N",
                scale=alt.Scale(
                    domain=["H5", "Neural network, no plant", "XGBoost"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=PLOT_WIDTH,
            height=220,
            title=_title(
                [
                    "Weight of each forecaster at each quantile level in arm C2-LOMO (own history),",
                    "mean over the 12 scored months. The weights of C2 on earlier months lean the same way.",
                ]
            ),
        )
    )
    chart.save(DOCS_DIR / "figure_4_weights.svg")


def tails() -> None:
    """Figure 3: pinball loss at the 0.01 and 0.99 quantiles per arm."""
    s = (
        pl.read_csv(common.DATA_DIR / "scores.csv")
        .filter(
            (pl.col("slice") == "all")
            & pl.col("metric").is_in(["pin01", "pin99"])
            & pl.col("arm").is_in(["h5_quantile", "nn_pure", "xgboost", "c2", "c3"])
        )
        .with_columns(
            label=pl.col("arm").replace_strict(ARMS),
            setting_label=pl.col("setting").replace_strict(SETTING_LABELS),
            quantile=pl.col("metric").replace_strict(
                {"pin01": "0.01 quantile", "pin99": "0.99 quantile"}
            ),
        )
    )
    y = alt.Y("label:N", sort=list(ARMS.values()), title=None, axis=alt.Axis(labelLimit=330))
    base = alt.Chart(s)
    chart = (
        base.mark_rule(size=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lo:Q", title="Pinball loss (points of capacity; smaller is better)"),
            x2="hi:Q",
            y=y,
            color=_setting_color(),
            yOffset="setting_label:N",
        )
        + base.mark_point(filled=True, size=55).encode(  # ty: ignore[unresolved-attribute]
            x="value:Q", y=y, color=_setting_color(), yOffset="setting_label:N"
        )
    ).properties(width=PLOT_WIDTH, height=170)
    chart = chart.facet(row=alt.Row("quantile:N", title=None)).properties(
        title=_title(
            [
                "Pinball loss at the two outer quantiles, mean over six farms, 12 to 36 h, 2025-09 to 2026-08.",
                "Dot: estimate. Line: 95% interval from resampling whole months.",
            ]
        )
    )
    chart.save(DOCS_DIR / "figure_3_tails.svg")


if __name__ == "__main__":
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    headline()
    contrasts()
    tails()
    weights()
