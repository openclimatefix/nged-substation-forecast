# ruff: noqa: E501
"""Draw experiment 361's figures from the stored reports.

uv run --no-sync python ml-experiments/00361-*/e00361_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00361-d8-step-catalogue-detector-injection-and-gam-stack"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
DOCS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / SLUG
WIDTH: Final[int] = 480


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    print("wrote", name)


def _title(text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 85), subtitle=textwrap.wrap(subtitle, 110),
        anchor="start", fontSize=14, subtitleFontSize=11,
    )  # fmt: skip


def figure_1(*, injection: dict) -> None:
    """Share of injected spells recovered, by size and number of overlapping components."""
    rows = []
    for k in (1, 2, 3):
        for size in (2.0, 3.0, 4.0):
            r = injection[f"k{k}_size{size}"]
            rows.append(
                {
                    "overlap": f"{k} at once",
                    "size": f"{size:.0f} MW",
                    "share": r["recovered_share"],
                    "low": r["wilson"][0],
                    "high": r["wilson"][1],
                }
            )
    frame = pl.DataFrame(rows)
    base = alt.Chart(frame).encode(y=alt.Y("size:N", title="Injected size"), color=alt.Color("overlap:N", title="Components running", scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_GREEN])), yOffset="overlap:N")  # fmt: skip
    rule = (
        alt.Chart(pl.DataFrame({"x": [0.8]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#333333")
        .encode(x="x:Q")
    )
    chart = (
        rule
        + base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "low:Q",
                title="Share of injected spells recovered",
                scale=alt.Scale(domain=[0.3, 1.0]),
            ),
            x2="high:Q",
        )
        + base.mark_point(filled=True, size=45).encode(x="share:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH,
        height=200,
        title=_title(
            "The detector finds 4 MW spells most of the time and 2 MW spells about half the time",
            "Injected into D2, D3, and D4 over 2024-07 to 2025-06; 95% Wilson intervals; dashed line is the pre-registered 80%",
        ),
    )
    _save(chart, name="figure_1_injection_recovery")


def figure_2(*, scores: dict) -> None:
    """Planned contrasts with 99% intervals, in points of D8's capacity."""
    rows = []
    for label, col, group in (
        ("C1: level added back minus control", "c_t1_arm3__ctl_arm3", "Planned"),
        ("C2: rows masked minus control", "c_t0_arm3__ctl_arm3", "Planned"),
        ("Added back minus control, donated band", "c_t1_donated__ctl_donated", "Other band rules"),
        ("Masked minus control, donated band", "c_t0_donated__ctl_donated", "Other band rules"),
    ):
        p, lo, hi = scores["contrast"][col]["all"]["ci99"]
        rows.append({"label": label, "estimate": p, "low": lo, "high": hi, "group": group})
    frame = pl.DataFrame(rows)
    order = frame["label"].to_list()
    base = alt.Chart(frame).encode(y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=400)), color=alt.Color("group:N", scale=alt.Scale(domain=["Planned", "Other band rules"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]), legend=alt.Legend(title=None, orient="top")))  # fmt: skip
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    chart = (
        zero
        + base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title="Quantile score difference (points; below zero is better)"),
            x2="high:Q",
        )
        + base.mark_point(filled=True, size=45).encode(x="estimate:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH,
        height=160,
        title=_title(
            "Removing detected steps from the training target changes D8's score by under 0.05 points",
            "Paired, month-resampled 99% intervals on 96,986 rows; the pre-registered threshold was -0.2",
        ),
    )
    _save(chart, name="figure_2_contrasts")


def figure_3(*, catalogue: dict) -> None:
    """Detected falls in power at D8 per month."""
    frame = pl.DataFrame(catalogue["per_month"])
    chart = alt.Chart(frame).mark_bar(color=ocf.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("month:O", title="Month", axis=alt.Axis(labelAngle=-90, labelOverlap="parity")),
        y=alt.Y("len:Q", title="Detected falls of at least 6 hours"),
    ).properties(width=WIDTH, height=180, title=_title("The rule finds 1 to 10 falls a month at D8, and 193 in all", "Threshold 1.5 MW on the 6-hour median of the residual; against 171 rises of the same size"))  # fmt: skip
    _save(chart, name="figure_3_falls_per_month")


def main() -> None:
    """Draw every figure."""
    figure_1(injection=json.loads((WORK / "injection_report.json").read_text()))
    figure_2(scores=json.loads((WORK / "work" / "scores.json").read_text()))
    figure_3(catalogue=json.loads((WORK / "catalogue_report.json").read_text()))


if __name__ == "__main__":
    main()
