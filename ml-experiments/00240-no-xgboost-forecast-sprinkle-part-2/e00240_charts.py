# ruff: noqa: E501
"""Draw the figures of experiment 240 from `scores.json` and `wind_scores.json`.

uv run --no-sync python ml-experiments/00240-no-xgboost-forecast-sprinkle-part-2/e00240_charts.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00240-no-xgboost-forecast-sprinkle-part-2"
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs/ml-experiments/00240-no-xgboost-forecast-sprinkle-part-2"
)
WIDTH: Final[int] = 420
SOLAR_LABELS: Final[dict[str, str]] = {
    "S": "S: XGBoost offsets around the plant mean",
    "SC": "SC: S without hour and month (post hoc)",
    "RSg": "RSg: regularised mean correction + S (post hoc)",
    "Rg": "Rg: regularised mean correction (post hoc)",
    "RS": "RS: mean correction + S",
    "R": "R: XGBoost mean correction",
    "R0": "R0: R without hour and month",
    "PR": "PR: placebo (shuffled residuals)",
    "A5": "A5: full XGBoost (experiment 224)",
    "X5r": "X5r: direct XGBoost, this protocol (post hoc)",
}
SOLAR_CONTRAST: Final[dict[str, str]] = {
    "S": "C2",
    "SC": "H8",
    "RSg": "H3",
    "Rg": "H1",
    "RS": "C3",
    "R": "C1",
    "R0": "D1",
    "PR": "D2",
    "A5": "D4",
    "X5r": "H9",
}
WIND_LABELS: Final[dict[str, str]] = {
    "B5": "B5: XGBoost on the physics core's expected power",
    "B14": "B14: XGBoost with the core's columns added",
    "B9": "B9: neural calibrator on the core",
    "A0r": "A0r: full XGBoost",
}
WIND_CONTRAST: Final[dict[str, str]] = {"B5": "W1", "B14": "W3", "B9": "W2", "A0r": "W4"}


def save(chart: alt.TopLevelMixin, *, name: str) -> None:
    """Write one chart as SVG beside the write-up."""
    chart.save(DOCS_DIR / f"{name}.svg")
    print("wrote", name)


def forest(
    *, rows: list[dict], order: list[str], xtitle: str, title: str, rule_x: float | None
) -> alt.TopLevelMixin:
    """Draw contrasts with intervals, a zero line, and optionally the half-gap line."""
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=420))
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    parts = [zero]
    if rule_x is not None:
        parts.append(
            alt.Chart(pl.DataFrame({"x": [rule_x]}))
            .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 3])
            .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )
    parts += [
        base.mark_rule(strokeWidth=2, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title=xtitle), x2="high:Q"
        ),
        base.mark_point(filled=True, size=50, color=ocf.DATA_BLUE).encode(x="est:Q"),  # ty: ignore[unresolved-attribute]
    ]
    return alt.layer(*parts).properties(width=WIDTH, height=26 * len(order) + 40, title=title)


def main() -> None:
    """Draw the solar and wind contrasts and the solar tail losses."""
    solar = json.loads((DATA / "scores.json").read_text())["scored_year"]
    gap = solar["arms"]["A3"]["qs"] - solar["arms"]["A5"]["qs"]
    rows = [
        {
            "arm": label,
            "est": solar["contrasts"][SOLAR_CONTRAST[k]]["qs"][0],
            "low": solar["contrasts"][SOLAR_CONTRAST[k]]["qs"][3],
            "high": solar["contrasts"][SOLAR_CONTRAST[k]]["qs"][4],
        }
        for k, label in SOLAR_LABELS.items()
    ]
    save(
        forest(
            rows=rows,
            order=list(SOLAR_LABELS.values()),
            xtitle="Quantile score minus the XGBoost-free plant forecast's (points of capacity; below zero is better)",
            title="Solar farms, 12 to 36 hours: change from the XGBoost-free forecast (dashed: half the gap to XGBoost)",
            rule_x=-gap / 2,
        ),
        name="figure_1_solar_contrasts",
    )
    tails = [
        {"arm": label, "level": q, "loss": solar["arms"][arm][f"pinball_{q}"]}
        for arm, label in (
            ("A3", "XGBoost-free plant (A3)"),
            ("A5", "Full XGBoost (A5)"),
            ("S", "Plant + XGBoost offsets (S)"),
        )
        for q in ("q01", "q05", "q95", "q99")
    ]
    tail_chart = (
        alt.Chart(pl.DataFrame(tails))
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("arm:N", title=None, axis=alt.Axis(labels=False)),
            y=alt.Y("loss:Q", title="Pinball loss (points of capacity)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(range=[ocf.DATA_SKY, ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            column=alt.Column("level:N", title=None, sort=["q01", "q05", "q95", "q99"]),
        )
        .properties(width=110, height=170, title="Solar farms, 12 to 36 hours: tail pinball loss")
    )
    save(tail_chart, name="figure_2_solar_tails")
    wind = json.loads((DATA / "wind_scores.json").read_text())["12-36 h"]
    wgap = wind["arms"]["B1"]["qs"] - wind["arms"]["A0r"]["qs"]
    wrows = [
        {
            "arm": label,
            "est": wind["contrasts"][WIND_CONTRAST[k]]["qs"][0],
            "low": wind["contrasts"][WIND_CONTRAST[k]]["qs"][3],
            "high": wind["contrasts"][WIND_CONTRAST[k]]["qs"][4],
        }
        for k, label in WIND_LABELS.items()
    ]
    save(
        forest(
            rows=wrows,
            order=list(WIND_LABELS.values()),
            xtitle="Quantile score minus the XGBoost-free physics core's (points of capacity; below zero is better)",
            title="Wind farms, 12 to 36 hours: change from the XGBoost-free forecast (dashed: half the gap to XGBoost)",
            rule_x=-wgap / 2,
        ),
        name="figure_3_wind_contrasts",
    )


if __name__ == "__main__":
    main()
