# ruff: noqa: E501
"""Draw experiment 257's figures from the score tables (no farm series are drawn)."""

from pathlib import Path
from typing import Final

import altair as alt
import e00257_common as common
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / common.SLUG
PLOT_WIDTH: Final[int] = CONTENT_WIDTH_PX - 300
ARMS: Final[dict[str, str]] = {
    "h5_quantile": "H5: plant + neural network offsets",
    "xgboost": "XGBoost",
    "p2": "P2: plant + linear quantile regression",
    "p2_no_ratio": "P2 without the availability ratio",
    "p1": "P1: two-number plant",
    "plant": "P0: plant alone",
    "floor_climatology": "Floor: climatology",
    "floor_heuristic": "Floor: manual heuristic",
}
SETTING_LABELS: Final[dict[str, str]] = {
    "own_history": "Own history",
    "zero_history": "Zero history",
}
SETTING_COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE]
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1": "C1: H5 minus P2 (own history)",
    "C3": "C3: P2 minus P1 (own history)",
    "C4": "C4: P2 minus H5 (zero history)",
    "C5": "C5: P2 minus P2 without the ratio (own history)",
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
        height=290,
        title=_title(
            [
                "Quantile score of the day-ahead forecast of six metered solar farms (A to F), mean over farms,",
                "12 to 36 h leads, daylight half-hours, 2025-09 to 2026-08. Own history: fitted on the farm's own earlier",
                "months. Zero history: fitted on the other five farms. P1 and P2 without the ratio have no zero-history row. Dot: estimate",
                "(mean over five seeds for H5 and XGBoost). Line: 95% interval from resampling whole months.",
            ]
        ),
    )
    chart.save(DOCS_DIR / "figure_1_headline.svg")


def contrasts() -> None:
    """Figure 2: the planned contrasts with 99% intervals."""
    c = (
        pl.read_csv(common.DATA_DIR / "contrasts.csv")
        .filter((pl.col("slice") == "all") & pl.col("metric").is_in(["score", "pin99"]))
        .with_columns(
            name=pl.when(pl.col("metric") == "pin99")
            .then(pl.lit("C2: H5 minus P2, pinball loss at 0.99 (own history)"))
            .otherwise(pl.col("contrast").replace_strict(CONTRAST_LABELS, default=None))
        )
        .filter(pl.col("name").is_not_null())
        .filter(~((pl.col("metric") == "pin99") & (pl.col("contrast") != "C1")))
    )
    order = [
        CONTRAST_LABELS["C1"],
        "C2: H5 minus P2, pinball loss at 0.99 (own history)",
        CONTRAST_LABELS["C3"],
        CONTRAST_LABELS["C4"],
        CONTRAST_LABELS["C5"],
    ]
    y = alt.Y("name:N", sort=order, title=None, axis=alt.Axis(labelLimit=330))
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
        height=200,
        title=_title(
            [
                "Planned contrasts in quantile score (C2 in pinball loss at the 0.99 quantile), mean over six farms,",
                "12 to 36 h, 2025-09 to 2026-08. Dot: estimate. Line: 99% interval from resampling whole months",
                "(Bonferroni over five contrasts). Grey rule: no change.",
            ]
        ),
    )
    chart.save(DOCS_DIR / "figure_2_contrasts.svg")


def tails() -> None:
    """Figure 3: pinball loss at the 0.01 and 0.99 quantiles per arm."""
    s = (
        pl.read_csv(common.DATA_DIR / "scores.csv")
        .filter(
            (pl.col("slice") == "all")
            & pl.col("metric").is_in(["pin01", "pin99"])
            & pl.col("arm").is_in(["h5_quantile", "xgboost", "p2", "p1", "plant"])
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
