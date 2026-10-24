# ruff: noqa: E501
"""Draw experiment 324's figures as SVG.

Afterwards optimise with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00324-*/e00324_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00324_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / common.SLUG
)
WIDTH: Final[int] = 560
WRAP: Final[int] = 95
GREY: Final[str] = "#999999"
SCOPE: Final[str] = (
    "3 wind farms, 53 Tuesday runs, 3 seeds, July 2025 to June 2026, rows kept by the idle-band mask."
)
LABELS: Final[dict[str, str]] = {
    "W0": "W0 climatology",
    "W2": "W2 XGBoost, weather only",
    "W3": "W3 mixture network, weather only",
    "P2": "Adapter alone",
    "P3": "Adapter plus offset table",
    "W4": "W4 adapter plus mixture head",
    "W5": "W5 W4 plus lead terms",
    "W6": "W6 one pooled head, farm embedding",
    "W7": "W7 one head per lead band",
    "W5_LOFO": "W5 for a farm with no history",
    "W8_W1": "W8 W1 plus recalibration table",
    "W8_W4": "W8 W4 plus recalibration table",
}
COLOURS: Final[dict[str, str]] = {
    "W0": GREY,
    "W2": ocf.DATA_SKY,
    "W3": ocf.DATA_BLUE,
    "P2": ocf.DATA_GREEN,
    "P3": ocf.DATA_GREEN,
    "W4": ocf.BRAND_ORANGE,
    "W5": ocf.BRAND_ORANGE,
    "W6": ocf.BRAND_ORANGE,
    "W7": ocf.BRAND_ORANGE,
    "W5_LOFO": ocf.DATA_PURPLE,
    "W8_W1": ocf.DATA_SKY,
    "W8_W4": ocf.DATA_PURPLE,
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1(*, score: dict) -> alt.TopLevelMixin:
    """Every arm's score minus W1's, kept rows, 95% intervals."""
    rows = []
    for key, label in LABELS.items():
        entry = score["contrasts"].get(f"vsW1: {key} - W1")
        if entry is None:
            continue
        found = entry["kept | qs"]
        low, high = found["0.95000"]
        rows.append(
            {"arm": label, "key": key, "estimate": found["mean"], "lower": low, "upper": high}
        )
    data = pl.DataFrame(rows)
    order = list(data["arm"])
    base = alt.Chart(data).encode(
        y=alt.Y("arm:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
        color=alt.Color(
            "key:N",
            scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
            legend=None,
        ),
    )
    points = base.mark_point(filled=True, size=55).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "estimate:Q",
            title="Arm's score minus W1's (points of capacity; more negative is better)",
        )
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=WIDTH,
        height=26 * len(order),
        title=_title(
            text="A mixture head on the adapter's quantiles beat XGBoost plus the offset table",
            lines=[
                "Quantile score minus W1's (XGBoost on the 51 ensemble members plus the offset table). "
                "Grey line: same as W1. Dot: estimate. Line: 95% interval from re-drawing whole months "
                "and seeds. " + SCOPE
            ],
        ),
    )


def figure_2(*, score: dict) -> alt.TopLevelMixin:
    """Score minus W1's by lead day for the main arms, kept rows."""
    arms = ("P3", "W3", "W4", "W5")
    base = score["scores"]["W1"]["by_lead_day_kept"]
    rows = [
        {"arm": LABELS[a], "key": a, "day": int(d), "gap": v - base[d]}
        for a in arms
        for d, v in score["scores"][a]["by_lead_day_kept"].items()
    ]
    data = pl.DataFrame(rows)
    chart = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "day:Q",
                title="Lead day (calendar days from the run's date to the valid time)",
                axis=alt.Axis(values=list(range(11))),
            ),
            y=alt.Y("gap:Q", title="Score minus W1's (points; more negative is better)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=[LABELS[a] for a in arms], range=[COLOURS[a] for a in arms]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )
    zero = alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color=GREY).encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, chart).properties(
        width=WIDTH,
        height=220,
        title=_title(
            text="The head's gain over W1 appears at every lead day",
            lines=["Quantile score minus W1's by lead day. Grey line: same as W1. " + SCOPE],
        ),
    )


def figure_3(*, tails: dict) -> alt.TopLevelMixin:
    """Share of observed power beyond each tail quantile, against its nominal share."""
    arms = ("P1", "W3", "W4", "W5", "W8_W1")
    names = {
        "P1": "W1 XGBoost plus table",
        "W3": LABELS["W3"],
        "W4": LABELS["W4"],
        "W5": LABELS["W5"],
        "W8_W1": LABELS["W8_W1"],
    }
    colours = {
        "P1": ocf.DATA_BLUE,
        "W3": ocf.DATA_SKY,
        "W4": ocf.BRAND_ORANGE,
        "W5": ocf.DATA_PURPLE,
        "W8_W1": ocf.DATA_GREEN,
    }
    rows = []
    for arm in arms:
        record = tails[arm]["kept"]
        for label, key, nominal in (
            ("below 1%", "share_below_q01", 0.01),
            ("below 5%", "share_below_q05", 0.05),
            ("above 95%", "share_above_q95", 0.05),
            ("above 99%", "share_above_q99", 0.01),
        ):
            rows.append({"arm": names[arm], "level": label, "ratio": record[key] / nominal})
    data = pl.DataFrame(rows)
    chart = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            x=alt.X(
                "ratio:Q",
                title="Observed share beyond the quantile divided by the nominal share (1 is calibrated)",
            ),
            y=alt.Y("level:N", sort=["below 1%", "below 5%", "above 95%", "above 99%"], title=None),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=[names[a] for a in arms], range=[colours[a] for a in arms]),
                legend=alt.Legend(orient="top", title=None, columns=2, labelLimit=400),
            ),
        )
    )
    one = alt.Chart(pl.DataFrame({"x": [1.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(one, chart).properties(
        width=WIDTH,
        height=150,
        title=_title(
            text="The mixture head fixes the lower 5% quantile but not the upper tail",
            lines=[
                "Share of observed power below the 1% and 5% quantiles, and above the 95% and 99% quantiles, divided by the nominal share. Below 1 means the band is too wide, above 1 too narrow. "
                + SCOPE
            ],
        ),
    )


def main() -> None:
    """Write the three figures."""
    score = json.loads((common.DATA_DIR / "score.json").read_text())
    tails = json.loads((common.DATA_DIR / "tails.json").read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for i, chart in enumerate(
        (figure_1(score=score), figure_2(score=score), figure_3(tails=tails)), start=1
    ):
        chart.save(DOCS_DIR / f"figure_{i}.svg")


if __name__ == "__main__":
    main()
