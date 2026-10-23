# ruff: noqa: E501
"""Draw experiment 292's figures from the score tables (no farm series are drawn)."""

from pathlib import Path
from typing import Final

import altair as alt
import e00292_common as common
import polars as pl
from plotting import ocf_theme as ocf
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = Path("docs/ml-experiments") / common.SLUG
PLOT_WIDTH: Final[int] = CONTENT_WIDTH_PX - 250
ARMS: Final[dict[str, str]] = {
    "n1": "N1: H5 + neighbourhood summaries",
    "n0": "N0: H5, own cell only",
    "placebo": "Placebo: H5 + summaries of another day",
    "x1": "X1: XGBoost + neighbourhood summaries",
    "x0": "X0: XGBoost, own cell only",
    "floor_climatology": "Floor: climatology",
    "floor_heuristic": "Floor: manual heuristic",
}
CONTRASTS: Final[dict[str, str]] = {
    "C1": "C1: N1 minus N0",
    "C2": "C2: N1 minus placebo",
    "C3": "C3: N1 minus X1",
}


def _title(lines: list[str]) -> alt.TitleParams:
    return alt.TitleParams(lines, anchor="start", fontSize=12, fontWeight="normal")


def _rule(*, x: float, color: str, dash: list[int] | None = None, size: int = 1) -> alt.Chart:
    """Return a vertical reference rule at `x`."""
    return (
        alt.Chart(pl.DataFrame({"x": [x]}))
        .mark_rule(color=color, strokeDash=dash or [1, 0], size=size)
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )


def headline() -> None:
    """Figure 1: quantile score per arm."""
    scores = (
        pl.read_csv(common.DATA_DIR / "scores.csv")
        .filter(
            (pl.col("slice") == "all")
            & (pl.col("metric") == "score")
            & pl.col("arm").is_in(list(ARMS))
        )
        .with_columns(label=pl.col("arm").replace_strict(ARMS))
    )
    y = alt.Y("label:N", sort=list(ARMS.values()), title=None, axis=alt.Axis(labelLimit=250))
    base = alt.Chart(scores)
    chart = (
        base.mark_rule(size=2, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lo:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(domain=[4.5, 9.5]),
            ),
            x2="hi:Q",
            y=y,
        )
        + base.mark_point(filled=True, size=55, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x="value:Q", y=y
        )
    ).properties(
        width=PLOT_WIDTH,
        height=210,
        title=_title(
            [
                "Quantile score of the day-ahead forecast of six metered solar farms (A to F), mean over farms, 12 to",
                "36 h leads, daylight half-hours, 2025-09 to 2026-08. Dot: estimate (mean over five seeds for networks).",
                "Line: 95% interval from resampling whole months.",
            ]
        ),
    )
    chart.save(DOCS_DIR / "figure_1_headline.svg")


def contrasts() -> None:
    """Figure 2: the planned contrasts with their 98.75% intervals."""
    table = (
        pl.read_csv(common.DATA_DIR / "contrasts.csv")
        .filter(
            (pl.col("slice") == "all")
            & (pl.col("metric") == "score")
            & pl.col("contrast").is_in(list(CONTRASTS))
        )
        .with_columns(label=pl.col("contrast").replace_strict(CONTRASTS))
    )
    y = alt.Y("label:N", sort=list(CONTRASTS.values()), title=None, axis=alt.Axis(labelLimit=250))
    base = alt.Chart(table)
    rules = _rule(x=0.0, color="#888888") + _rule(
        x=-2.0, color=ocf.BRAND_ORANGE, dash=[5, 4], size=2
    )
    chart = (
        rules
        + base.mark_rule(size=2, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lo:Q",
                title="Change in quantile score (%; more negative is better)",
                scale=alt.Scale(domain=[-4, 2]),
            ),
            x2="hi:Q",
            y=y,
        )
        + base.mark_point(filled=True, size=55, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x="percent_change:Q", y=y
        )
    ).properties(
        width=PLOT_WIDTH,
        height=130,
        title=_title(
            [
                "Planned contrasts in quantile score, equal weight per farm. Dot: estimate. Line: 98.75% interval from",
                "resampling whole months. Grey rule: no change. Orange dashed rule: the 2% gain the plan required.",
            ]
        ),
    )
    chart.save(DOCS_DIR / "figure_2_contrasts.svg")


def per_farm() -> None:
    """Figure 3: the gain of N1 and of the placebo over N0 at each farm."""
    gain = (
        pl.read_csv(common.DATA_DIR / "per_farm_gain.csv")
        .select("time_series_id", n1="n1_vs_n0", placebo="placebo_vs_n0")
        .unpivot(index="time_series_id", variable_name="arm", value_name="change")
        .with_columns(
            label=pl.col("arm").replace_strict(
                {
                    "n1": "N1",
                    "placebo": "Placebo",
                }
            )
        )
    )
    colour = alt.Color(
        "label:N",
        scale=alt.Scale(
            domain=["N1", "Placebo"],
            range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
        ),
        legend=alt.Legend(title=None, orient="top"),
    )
    chart = (
        _rule(x=0.0, color="#888888")
        + _rule(x=-1.0, color="#888888", dash=[5, 4])
        + alt.Chart(gain)
        .mark_point(filled=True, size=70)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "change:Q",
                title="Change in quantile score against N0 (%; more negative is better)",
                scale=alt.Scale(domain=[-2, 1.5]),
            ),
            y=alt.Y("time_series_id:N", title="Farm"),
            color=colour,
            yOffset="label:N",
        )
    ).properties(
        width=PLOT_WIDTH,
        height=230,
        title=_title(
            [
                "Change in quantile score against N0 (H5 on the own cell) at each farm, 12 to 36 h leads, 2025-09 to",
                "2026-08. N1: H5 with the neighbourhood summaries. Placebo: N1 with another day's summaries. Grey rule:",
                "no change. Dashed rule: the 1% gain that the stop rule counts.",
            ]
        ),
    )
    chart.save(DOCS_DIR / "figure_3_per_farm.svg")


def tails() -> None:
    """Figure 4: pinball loss change at the two tails, N1 against N0 and X1 against X0."""
    c = pl.read_csv(common.DATA_DIR / "contrasts.csv").filter(
        (pl.col("slice") == "all")
        & pl.col("metric").is_in(["pin01", "pin99"])
        & (
            ((pl.col("arm") == "n1") & (pl.col("reference") == "n0"))
            | ((pl.col("arm") == "x1") & (pl.col("reference") == "x0"))
        )
    )
    table = (
        c.with_columns(
            label=pl.col("arm").replace_strict(
                {"n1": "N1 against N0 (H5)", "x1": "X1 against X0 (XGBoost)"}
            ),
            tail=pl.col("metric").replace_strict(
                {"pin01": "0.01 quantile", "pin99": "0.99 quantile"}
            ),
        )
        if c.height
        else None
    )
    if table is None:
        return
    colour = alt.Color(
        "label:N",
        scale=alt.Scale(
            domain=["N1 against N0 (H5)", "X1 against X0 (XGBoost)"],
            range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
        ),
        legend=alt.Legend(title=None, orient="top"),
    )
    y = alt.Y("tail:N", title=None)
    base = alt.Chart(table)
    chart = (
        _rule(x=0.0, color="#888888")
        + base.mark_rule(size=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lo:Q", title="Change in pinball loss (%; more negative is better)"),
            x2="hi:Q",
            y=y,
            color=colour,
            yOffset="label:N",
        )
        + base.mark_point(filled=True, size=55).encode(  # ty: ignore[unresolved-attribute]
            x="percent_change:Q", y=y, color=colour, yOffset="label:N"
        )
    ).properties(
        width=PLOT_WIDTH,
        height=150,
        title=_title(
            [
                "Pinball loss at the two tails, with and without the neighbourhood summaries, equal weight per farm.",
                "Dot: estimate. Line: interval from resampling whole months (98.75% for N1 against N0, 95% for X1 against X0).",
                "Grey rule: no change.",
            ]
        ),
    )
    chart.save(DOCS_DIR / "figure_4_tails.svg")


if __name__ == "__main__":
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    headline()
    contrasts()
    per_farm()
    tails()
