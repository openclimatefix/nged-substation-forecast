# ruff: noqa: E501
"""Draw experiment 00339's figures as SVG. Reads `score.json` in the experiment's data folder.

Afterwards optimise with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00339-*/e00339_charts.py   (PYTHONPATH as for the run)
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00339_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00339-wind-research-default-1-adapter-plus-mixture-head"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 95
GREY: Final[str] = "#999999"
LEVEL: Final[str] = "0.9967"
SCOPE: Final[str] = "3 wind farms, 3 seeds, scored at 1 to 10 days."
ARMS: Final[dict[str, str]] = {
    "x1": "x1: head on kept rows",
    "x2": "x2: x1 plus idle-floor rule",
    "x3": "x3: head on all half-hours",
    "default_rule": "default plus the rule (descriptive)",
    "x1_placebo": "x1 with shuffled inputs (placebo)",
}
COLOURS: Final[dict[str, str]] = {
    "default": GREY,
    "x1": ocf.BRAND_ORANGE,
    "x2": ocf.DATA_SKY,
    "x3": ocf.DATA_PURPLE,
    "P3": ocf.DATA_BLUE,
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _panel(*, score: dict, period: str, subset: str, title: alt.TitleParams) -> alt.TopLevelMixin:
    rows = []
    for key, label in ARMS.items():
        entry = score["contrasts"][period].get(key)
        if entry is None:
            continue
        found = entry[f"{subset} | qs"]
        low, high = found[LEVEL]
        rows.append(
            {
                "arm": label,
                "estimate": found["mean"],
                "lower": low,
                "upper": high,
                "grey": key.endswith(("placebo", "rule")),
            }
        )
    data = pl.DataFrame(rows)
    base = alt.Chart(data).encode(
        y=alt.Y("arm:N", sort=list(data["arm"]), title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.condition("datum.grey", alt.value(GREY), alt.value(ocf.DATA_BLUE)),
    )
    points = base.mark_point(filled=True, size=55).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "estimate:Q", title="Score minus the default's (points of capacity; lower is better)"
        )
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=WIDTH, height=26 * data.height, title=title
    )


def figure_1(*, score: dict) -> alt.TopLevelMixin:
    """Arms minus the default: kept rows, all half-hours, idle-band rows, replicate."""
    note = (
        "Dot: estimate. Line: 99.67% interval from resampling whole months. Rule at 0: the default."
    )
    panels = [
        _panel(
            score=score,
            period=period,
            subset=subset,
            title=_title(text=text, lines=[f"{where}. {SCOPE} {note}"]),
        )
        for period, subset, text, where in (
            (
                "scored",
                "kept",
                "The head gains 0.27 points on the kept rows; the rule adds nothing",
                "Kept rows, scored year",
            ),
            (
                "scored",
                "all",
                "On all half-hours the gain shrinks to 0.04",
                "All half-hours, scored year",
            ),
            (
                "scored",
                "idle210",
                "On the idle-band rows the head loses 2.1 points unless trained on them",
                "Idle-band rows, scored year",
            ),
            (
                "replicate",
                "kept",
                "The replicate period gains 0.32 points",
                "Kept rows, January to June 2025",
            ),
        )
    ]
    return alt.vconcat(*panels)  # ty: ignore[invalid-argument-type].configure_view(stroke=None)


def figure_2(*, score: dict) -> alt.TopLevelMixin:
    """Kept-row score by lead day for the default, the best single family, and the head arms."""
    rows = []
    for arm, label in (
        ("default", "default (XGBoost)"),
        ("P3", "physics core + adapter + table"),
        ("x1", "x1 (standard architecture)"),
        ("x3", "x3"),
    ):
        for day, value in score["scores"]["scored"][arm]["by_lead_day_kept"].items():
            rows.append({"arm": label, "day": int(day), "score": value, "key": arm})
    data = pl.DataFrame(rows)
    scale = alt.Scale(
        domain=[
            "default (XGBoost)",
            "physics core + adapter + table",
            "x1 (standard architecture)",
            "x3",
        ],
        range=[GREY, ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE],
    )
    return (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("day:Q", title="Lead (days)", axis=alt.Axis(values=list(range(1, 11)))),
            y=alt.Y(
                "score:Q",
                title="Quantile score (points of capacity; lower is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "arm:N",
                scale=scale,
                legend=alt.Legend(orient="top", title=None, columns=2, labelLimit=300),
            ),
        )
        .properties(
            width=WIDTH,
            height=240,
            title=_title(
                text="The head beats the default at every lead day",
                lines=[f"Kept rows, scored year. {SCOPE} Mean over lead days; one point per day."],
            ),
        )
    )


def figure_3(*, score: dict) -> alt.TopLevelMixin:
    """Tail shares on the kept rows against their nominal values."""
    rows = []
    nominal = {"below_q01": 0.01, "below_q05": 0.05, "above_q95": 0.05, "above_q99": 0.01}
    names = {
        "below_q01": "below q01 (nominal 1%)",
        "below_q05": "below q05 (nominal 5%)",
        "above_q95": "above q95 (nominal 5%)",
        "above_q99": "above q99 (nominal 1%)",
    }
    for arm, label in (("default", "default"), ("x1", "x1"), ("x2", "x2"), ("x3", "x3")):
        for metric, share in nominal.items():
            rows.append(
                {
                    "arm": label,
                    "tail": names[metric],
                    "share": 100 * score["scores"]["scored"][arm]["kept"][metric],
                    "nominal": 100 * share,
                }
            )
    data = pl.DataFrame(rows)
    bars = (
        alt.Chart(data)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("arm:N", sort=["default", "x1", "x2", "x3"], title=None),
            x=alt.X("share:Q", title="Share of half-hours (%)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=["default", "x1", "x2", "x3"],
                    range=[GREY, ocf.BRAND_ORANGE, ocf.DATA_SKY, ocf.DATA_PURPLE],
                ),
                legend=None,
            ),
        )
        .properties(width=WIDTH, height=70)
    )
    rule = alt.Chart(data).mark_rule(color="black", strokeDash=[4, 3]).encode(x="nominal:Q")  # ty: ignore[unresolved-attribute]
    return (
        alt.layer(bars, rule)
        .facet(
            row=alt.Row(
                "tail:N",
                sort=list(names.values()),
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left"),
            )
        )
        .properties(
            title=_title(
                text="The head calibrates the lower tail and leaves the upper tail over-confident",
                lines=[
                    f"Kept rows, scored year. {SCOPE} Dashed line: the share a calibrated forecast gives."
                ],
            )
        )
    )


def figure_4(*, score: dict) -> alt.TopLevelMixin:
    """Each model family alone, and the head, minus the default on the kept rows."""
    rows = []
    for label, found in score["family_contrasts_vs_default"].items():
        rows.append(
            {
                "arm": label,
                "estimate": found["mean"],
                "lower": found[LEVEL][0],
                "upper": found[LEVEL][1],
                "head": False,
            }
        )
    for arm, label in (
        ("x1", "standard architecture (x1: core + adapter + head)"),
        ("x3", "x3 (head on all half-hours)"),
    ):
        found = score["contrasts"]["scored"][arm]["kept | qs"]
        rows.append(
            {
                "arm": label,
                "estimate": found["mean"],
                "lower": found[LEVEL][0],
                "upper": found[LEVEL][1],
                "head": True,
            }
        )
    data = pl.DataFrame(rows)
    base = alt.Chart(data).encode(
        y=alt.Y("arm:N", sort=list(data["arm"]), title=None, axis=alt.Axis(labelLimit=330)),
        color=alt.condition("datum.head", alt.value(ocf.BRAND_ORANGE), alt.value(ocf.DATA_BLUE)),
    )
    points = base.mark_point(filled=True, size=55).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "estimate:Q", title="Score minus the default's (points of capacity; lower is better)"
        )
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=WIDTH,
        height=26 * data.height,
        title=_title(
            text="The standard architecture beats every single family on the kept rows",
            lines=[
                f"Kept rows, scored year. {SCOPE} Dot: estimate. Line: 99.67% interval from resampling whole months. Rule at 0: the default (XGBoost plus the offset table)."
            ],
        ),
    )


def main() -> None:
    """Write the figures beside the result page."""
    score = json.loads((common.DATA_DIR / "score.json").read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for number, chart in enumerate(
        (
            figure_1(score=score),
            figure_2(score=score),
            figure_3(score=score),
            figure_4(score=score),
        ),
        1,
    ):
        chart.save(DOCS_DIR / f"figure_{number}.svg")


if __name__ == "__main__":
    main()
