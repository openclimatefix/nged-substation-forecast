"""Draw experiment 151's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The series are Primary substations, not NGED generators.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00151-*/e00151_charts.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00151_report as report
import plotting.ocf_theme as ocf
import polars as pl

HERE: Final[Path] = Path(__file__).resolve().parent
OUT_DIR: Final[Path] = (
    HERE.parents[1] / "docs" / "ml-experiments" / "00151-nra-forecast-rescored-against-the-true-nra-series"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 80
LABELS: Final[dict[str, str]] = {
    "SUB_DET": "Subtract detected MW",
    "SUB_TRUE_LAG": "Subtract true MW (shaped), after the detector's delay",
    "SUB_TRUE_CONST": "Subtract true MW (constant), after the detector's delay",
    "SKF": "Two-regime filter, detected intervals",
    "SORACLE": "Two-regime filter, true intervals",
    "SUB_PLACEBO": "Subtract detected MW, shifted by weeks (control)",
}
ORDER: Final[list[str]] = list(LABELS.values())


def _title(*, text: str, line: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 70),
        subtitle=textwrap.wrap(line, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _shares(*, summary: dict, sets: dict[str, str]) -> pl.DataFrame:
    rows = []
    for key, label in sets.items():
        for arm, name in LABELS.items():
            r = summary[key][f"qs/{arm}"]
            if "lo95" not in r:
                continue
            rows.append(
                {
                    "set": label,
                    "arm": name,
                    "share": 100 * r["point"],
                    "low": 100 * r["lo95"],
                    "high": 100 * r["hi95"],
                }
            )
    return pl.DataFrame(rows)


def _dots(*, data: pl.DataFrame, colour: bool, height: int) -> alt.TopLevelMixin:
    colours = [ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE]
    base = alt.Chart(data)
    y = alt.Y("arm:N", title=None, sort=ORDER)
    enc = {"color": alt.Color("set:N", title=None, scale=alt.Scale(range=colours))} if colour else {}
    yoff = {"yOffset": "set:N"} if colour else {}
    rule = base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
        y=y, x=alt.X("low:Q", title="Share of the gap closed (%; larger is better)"), x2="high:Q", **enc, **yoff
    )
    dot = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        y=y, x="share:Q", **({"color": enc["color"]} if colour else {}), **yoff
    )
    if not colour:
        rule = rule.mark_rule(color=ocf.DATA_BLUE)
        dot = dot.mark_point(filled=True, size=60, color=ocf.DATA_BLUE)
    zero = alt.Chart(pl.DataFrame({"x": [0, 100]})).mark_rule(strokeDash=[3, 3], color="grey").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    return (zero + rule + dot).properties(width=WIDTH, height=height)


def figure_headline(*, summary: dict) -> alt.TopLevelMixin:
    """Share of the gap closed by each arm on the post-alarm rows."""
    data = _shares(summary=summary, sets={"post (C1 rows)": "After the alarm"})
    return _dots(data=data, colour=False, height=200).properties(
        title=_title(
            text="Subtracting the detected MW closes most of the gap once the alarm is raised",
            line="Mean quantile score against the true normal-running-arrangement series, "
            "forecasts issued after the alarm, targets inside the transfer or in the 48 hours "
            "after its end. 0% is the filter without correction, 100% the filter fed the true "
            "series. Dot: estimate. Line: 95% interval from resampling whole calendar months.",
        )
    )


def figure_windows(*, summary: dict) -> alt.TopLevelMixin:
    """Shares by target window and by the origin's position relative to the alarm."""
    sets = {
        "post, inside": "After the alarm: inside the transfer",
        "post, after48": "After the alarm: 48 h after the end",
        "lag rows": "Before the alarm is raised",
    }
    data = _shares(summary=summary, sets=sets)
    return _dots(data=data, colour=True, height=380).properties(
        title=_title(
            text="Before the alarm is raised, no deployable arm helps",
            line="Same measure as the first figure, split by the forecast origin (before or after "
            "the alarm) and by the target window. Dot: estimate. Line: 95% interval from "
            "resampling whole calendar months.",
        )
    )


def figure_months(*, frame: pl.DataFrame) -> alt.TopLevelMixin:
    """Per-month share of the gap closed by SUB_DET on the post-alarm rows."""
    post = frame.filter((pl.col("origin") == "post") & (pl.col("inside") | pl.col("after48")))
    data = (
        post.group_by("month")
        .agg(
            num=(pl.col("qs__K0") - pl.col("qs__SUB_DET")).sum(),
            gap=(pl.col("qs__K0") - pl.col("qs__CLEAN")).sum(),
            rows=pl.len(),
        )
        .with_columns(share=100 * pl.col("num") / pl.col("gap"))
        .sort("month")
    )
    return (
        alt.Chart(data)
        .mark_point(filled=True, size=70, color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("month:N", title="Month of target time"),
            y=alt.Y("share:Q", title="Share of the gap closed (%)"),
            size=alt.Size("rows:Q", title="Rows", legend=alt.Legend(orient="top")),
        )
        .properties(
            width=WIDTH,
            height=170,
            title=_title(
                text="The gain holds in every month with rows",
                line="Subtract detected MW, forecasts issued after the alarm. Larger dots have "
                "more rows.",
            ),
        )
    )


def figure_absolute(*, summary: dict) -> alt.TopLevelMixin:
    """Absolute mean quantile score by arm for the two origin classes."""
    names = {"K0": "No correction", "CLEAN": "Filter fed the true series", **LABELS}
    rows = []
    for which, label in (("abs_post", "After the alarm"), ("abs_lag", "Before the alarm")):
        for arm, v in summary[which].items():
            rows.append({"set": label, "arm": names[arm], "score": v})
    data = pl.DataFrame(rows)
    order = list(names.values())
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("arm:N", title=None, sort=order),
            x=alt.X("score:Q", title="Mean quantile score (points of capacity; smaller is better)"),
            color=alt.Color(
                "set:N", title=None, scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE])
            ),
            yOffset="set:N",
        )
        .properties(
            width=WIDTH,
            height=330,
            title=_title(
                text="Absolute scores: the correction matters only after the alarm",
                line="Mean quantile score against the true normal-running-arrangement series, "
                "rows inside the transfer or in the 48 hours after its end.",
            ),
        )
    )


def main() -> None:
    """Write the figures."""
    summary = json.loads((HERE / "report.json").read_text())
    frame = report.load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_headline(summary=summary).save(OUT_DIR / "figure_1_headline.svg")
    figure_windows(summary=summary).save(OUT_DIR / "figure_2_windows.svg")
    figure_absolute(summary=summary).save(OUT_DIR / "figure_3_absolute.svg")
    figure_months(frame=frame).save(OUT_DIR / "figure_4_months.svg")


if __name__ == "__main__":
    main()
