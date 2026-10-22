"""Draw experiment 00242's figures from `work/score.json`, as SVG.

1. The planned contrasts on the tail score, by lead band, on the rows kept by experiment 210's mask.
2. Each arm's absolute tail score by lead band.
3. Coverage of the 5% to 95% band by lead band, before and after the spread factor.

Afterwards optimise with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00242-*/e00242_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00242_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00242-wind-physics-core-part-3-member-tails"
)
PLOT_WIDTH: Final[int] = 380
WRAP: Final[int] = 80
GREY: Final[str] = "#999999"
SCOPE: Final[str] = "3 wind farms, 53 Tuesday runs, 3 seeds, July 2025 to June 2026."
BANDS: Final[tuple[str, ...]] = ("days 1-2", "days 3-6", "days 7+", "all leads")
BAND_COLOURS: Final[list[str]] = [ocf.DATA_SKY, ocf.DATA_GREEN, ocf.DATA_PURPLE, ocf.BRAND_ORANGE]
CONTRAST_LABELS: Final[dict[str, str]] = {
    "C1": "Physics core: members minus mean",
    "C2": "XGBoost: members minus mean",
    "C3": "Core: with minus without residual quantiles",
    "C4": "Members: core minus XGBoost",
}
ARM_LABELS: Final[dict[str, str]] = {
    "B1_members": "Core, 51 members",
    "A0r_members": "XGBoost, 51 members",
    "B1d_members": "Core, members, no residuals",
    "B1_mean": "Core, ensemble mean",
    "A0r_mean": "XGBoost, ensemble mean",
}
ARM_COLOURS: Final[list[str]] = [
    ocf.BRAND_ORANGE,
    ocf.DATA_BLUE,
    ocf.BRAND_ORANGE_LIGHT,
    ocf.DATA_PURPLE,
    ocf.DATA_SKY,
]


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1(*, score: dict) -> alt.TopLevelMixin:
    """Each contrast on the tail score, kept rows, by lead band, with 95% intervals."""
    rows = []
    for label, text in CONTRAST_LABELS.items():
        for band in BANDS:
            entry = score["contrasts"][f"{label}|spread"][f"kept rows | {band} | tail"]
            rows.append(
                {
                    "contrast": text,
                    "band": band,
                    "difference": entry[0],
                    "lower": entry[1],
                    "upper": entry[2],
                }
            )
    colour = alt.Color(
        "band:N",
        scale=alt.Scale(domain=list(BANDS), range=BAND_COLOURS),
        legend=alt.Legend(title=None, orient="top"),
    )
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y(
            "contrast:N",
            sort=list(CONTRAST_LABELS.values()),
            title=None,
            axis=alt.Axis(labelLimit=300),
        ),
        yOffset=alt.YOffset("band:N", sort=list(BANDS)),
        color=colour,
    )
    points = base.mark_point(filled=True, size=50).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "difference:Q", title="First arm minus second (points of capacity; negative is better)"
        )
    )
    bars = base.mark_rule().encode(x="lower:Q", x2="upper:Q")  # ty: ignore[unresolved-attribute]
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color=GREY).encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return alt.layer(zero, bars, points).properties(
        width=PLOT_WIDTH,
        height=260,
        title=_title(
            text="Running all 51 members cuts the tail score, mostly beyond day 2",
            lines=[
                "Tail score (mean of the 0.01 and 0.99 pinball losses) of the first arm minus the "
                "second, after the spread factor, on the rows experiment 210's mask keeps. Grey "
                "line: no difference. Dot: estimate. Line: 95% interval from re-drawing whole "
                "months and seeds. " + SCOPE
            ],
        ),
    )


def figure_2(*, score: dict) -> alt.TopLevelMixin:
    """Absolute tail score of each arm after the spread factor, one panel per lead band."""
    rows = [
        {
            "arm": label,
            "band": band,
            "score": score["scores"][f"{arm}|spread"][f"kept rows | {band}"]["tail"],
        }
        for arm, label in ARM_LABELS.items()
        for band in BANDS
    ]
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            x=alt.X(
                "score:Q",
                title="Tail score (points of capacity; smaller is better)",
                axis=alt.Axis(tickCount=5),
            ),
            y=alt.Y(
                "arm:N", sort=list(ARM_LABELS.values()), title=None, axis=alt.Axis(labelLimit=260)
            ),
            row=alt.Row(
                "band:N",
                sort=list(BANDS),
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left"),
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(ARM_LABELS.values()), range=ARM_COLOURS),
                legend=None,
            ),
        )
        .resolve_scale(x="independent")
        .properties(
            width=PLOT_WIDTH,
            height=100,
            title=_title(
                text="The ensemble mean loses most of its tail skill at long leads",
                lines=[
                    "Absolute tail score after the spread factor, rows kept by experiment 210's "
                    "mask. Each panel has its own horizontal range. " + SCOPE
                ],
            ),
        )
    )


def figure_3(*, score: dict) -> alt.TopLevelMixin:
    """Coverage of the 5% to 95% band, raw and after the spread factor."""
    rows = [
        {
            "arm": label,
            "band": band,
            "form": "After spread factor" if form == "spread" else "Before spread factor",
            "coverage": 100 * score["scores"][f"{arm}|{form}"][f"kept rows | {band}"]["covered"],
        }
        for arm, label in ARM_LABELS.items()
        for band in BANDS
        for form in ("spread", "raw")
    ]
    points = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=60)
        .properties(width=150, height=90)
        .encode(
            x=alt.X(
                "coverage:Q",
                title="% of rows in band",
                scale=alt.Scale(domain=[50, 100]),
                axis=alt.Axis(values=[50, 70, 90, 100], grid=True),
            ),
            y=alt.Y(
                "arm:N", sort=list(ARM_LABELS.values()), title=None, axis=alt.Axis(labelLimit=260)
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(ARM_LABELS.values()), range=ARM_COLOURS),
                legend=None,
            ),
        )
    )
    return points.facet(
        row=alt.Row(
            "band:N",
            sort=list(BANDS),
            title=None,
            header=alt.Header(labelAngle=0, labelAlign="left"),
        ),
        column=alt.Column(
            "form:N", sort=["Before spread factor", "After spread factor"], title=None
        ),
    ).properties(
        title=_title(
            text="The ensemble mean stays under-dispersed even after the spread factor",
            lines=[
                "Share of rows whose observation lies between the 0.05 and 0.95 quantiles, "
                "rows kept by experiment 210's mask. The grid line at 90 is the calibrated value. "
                + SCOPE
            ],
        )
    )


def main() -> None:
    """Write the three figures."""
    score = json.loads((common.WORK_DIR / "score.json").read_text())
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_1", figure_1(score=score)),
        ("figure_2", figure_2(score=score)),
        ("figure_3", figure_3(score=score)),
    ):
        chart.save(DOCS_DIR / f"{name}.svg")


if __name__ == "__main__":
    main()
