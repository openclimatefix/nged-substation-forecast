# ruff: noqa: E501
"""Draw the figures of experiment 291 from the tables that `e00291_analysis.py` wrote.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00291_common import DATA_DIR, UNIT_MW

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00291-d8-drops-engine-spells-load-shift"
)
WIDTH: Final[int] = 640
WRAP: Final[int] = 95


def wrapped(text: str) -> list[str]:
    """Wrap a title or subtitle to the plot area's width."""
    return textwrap.wrap(text, WRAP)


def figure_overlap() -> alt.Chart:
    """Share of drops with engine-running states, against matched placebo windows and controls."""
    s = json.loads((DATA_DIR / "summary.json").read_text())
    rows = [
        ("D8's 28 drops", s["v1_observed"], None, None, "drops"),
        (
            "D8 matched placebo windows",
            s["v1_placebo"],
            s["v1_placebo_lo"],
            s["v1_placebo_hi"],
            "placebo",
        ),
        ("D5, same dates as the drops", s["D5_own"], None, None, "control"),
        ("D10, same dates as the drops", s["D10_own"], None, None, "control"),
        ("D11, same dates as the drops", s["D11_own"], None, None, "control"),
    ]
    frame = pl.DataFrame(rows, schema=["row", "share", "lo", "hi", "kind"], orient="row")
    colours = alt.Scale(
        domain=["drops", "placebo", "control"],
        range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_SKY],
    )
    y = alt.Y("row:N", sort=[r[0] for r in rows], title=None)
    base = alt.Chart(frame)
    bars = base.mark_bar(aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=y,
        x=alt.X(
            "share:Q",
            title="Share of drops with at least half of their half-hours flagged (higher: more engine-like)",
            scale=alt.Scale(domain=[0, 1]),
        ),
        color=alt.Color("kind:N", scale=colours, legend=None),
    )
    interval = base.mark_rule(color=ocf.BLACK_1).encode(y=y, x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
    labels = base.mark_text(align="left", dx=6, color=ocf.BLACK_1, aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=y, x="share:Q", text=alt.Text("share:Q", format=".2f")
    )
    return (bars + interval + labels).properties(
        width=WIDTH - 230,
        height=170,
        title=alt.TitleParams(
            "Engine-running states cover 27 of D8's 28 drops, against about 1 in 5 matched windows",
            subtitle=wrapped(
                "Bar: share of drops (or of placebo windows) whose half-hours are at least half flagged as "
                "engine running by experiment 196's detector. Black line: 95% of 5,000 placebo draws. "
                "Controls: the same detector on D5, D10, and D11 over the dates of D8's drops."
            ),
            anchor="start",
        ),
    )


def figure_depth() -> alt.Chart:
    """Histogram of drop depths in engine units, with the whole numbers marked."""
    table = pl.read_parquet(DATA_DIR / "drops_overlap.parquet").filter(pl.col("in_scope"))
    hist = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE, aria=False)
        .encode(
            x=alt.X(
                "depth_in_units:Q",
                bin=alt.Bin(step=0.25, extent=[0, 3]),
                title=f"Drop depth in units of {UNIT_MW} MW (the engine's recorded step)",
                axis=alt.Axis(values=[0, 1, 2, 3]),
            ),
            y=alt.Y("count()", title="Drops", axis=alt.Axis(tickMinStep=1)),
        )
    )
    rules = (
        alt.Chart(pl.DataFrame({"unit": [1.0, 2.0, 3.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 3])
        .encode(x="unit:Q")
    )
    return (hist + rules).properties(
        width=WIDTH - 60,
        height=170,
        title=alt.TitleParams(
            "Drop depths spread between whole engine units",
            subtitle=wrapped(
                "Bars: the 28 drops from 2023-08-15. Dashed lines: 1, 2, and 3 units. If the drops were "
                "whole numbers of units, the bars would stand at the lines."
            ),
            anchor="start",
        ),
    )


def figure_fade() -> alt.VConcatChart:
    """D8's monthly detector flag share and left-tail share, with the drops started."""
    m = pl.read_parquet(DATA_DIR / "monthly.parquet").with_columns(
        pl.col("month").str.to_date("%Y-%m")
    )
    long = m.unpivot(
        on=["v1_flag_share", "left_tail_share"],
        index="month",
        variable_name="series",
        value_name="share",
    ).with_columns(
        pl.col("series").replace(
            {
                "v1_flag_share": "Half-hours flagged as engine running",
                "left_tail_share": "Half-hours with power 2.5 MW or more below expected",
            }
        )
    )
    lines = (
        alt.Chart(long)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, aria=False)
        .encode(
            x=alt.X("month:T", title="Month (2025-03 to 2026-06)", axis=alt.Axis(format="%Y-%m")),
            y=alt.Y("share:Q", title="Share of D8's half-hours", scale=alt.Scale(domain=[0, 0.8])),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                legend=alt.Legend(orient="top", title=None, direction="vertical", labelLimit=500),
            ),
        )
    )
    drops = (
        alt.Chart(m)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.GREY_3, opacity=0.8, aria=False, size=12)
        .encode(
            x=alt.X("month:T", title="Month", axis=alt.Axis(format="%Y-%m")),
            y=alt.Y("drops_started:Q", title="Drops started", axis=alt.Axis(tickMinStep=1)),
        )
        .properties(width=WIDTH - 60, height=70)
    )
    top = lines.properties(width=WIDTH - 60, height=170)
    return alt.vconcat(top, drops).properties(
        title=alt.TitleParams(
            "D8's shortfall and its drops stop after February 2026; the detector's flag does not",
            subtitle=wrapped(
                "Lines: D8 by month. Grey bars: drops starting that month (none after January 2026)."
            ),
            anchor="start",
        )
    )


def main() -> None:
    """Write the three figures as SVG."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_1", figure_overlap()),
        ("figure_2", figure_depth()),
        ("figure_3", figure_fade()),
    ):
        chart.save(OUT_DIR / f"{name}.svg")
        print(name, flush=True)


if __name__ == "__main__":
    main()
