"""Draw experiment 101's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The series are Primary substations, not NGED generators.

    uv run python ml-experiments/00101-*/e00130_charts.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00130_common import WORK_DIR
from e00130_report import (
    ARMS,
    DURATION_BANDS,
    INJECTED_RECALL,
    SHARE_BANDS,
    bootstrap_weighted,
    event_table,
    wilson,
)

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00101-ro-nc-detector-on-real-hand-filtered-switching-events"
)
WIDTH: Final[int] = 600
WRAP: Final[int] = 95
COLOURS: Final[dict[str, str]] = {
    "D0": ocf.DATA_SKY,
    "PF14": ocf.DATA_BLUE,
    "X": ocf.DATA_PURPLE,
    "LOGIT_SAME": ocf.DATA_GREEN,
    "RO-NC": ocf.BRAND_ORANGE,
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(OUT_DIR / name)
    print("wrote", name)


def _colour(*, title: str | None = None, arms: tuple[str, ...] = ARMS) -> alt.Color:
    return alt.Color(
        "arm:N",
        scale=alt.Scale(domain=list(arms), range=[COLOURS[a] for a in arms]),
        legend=alt.Legend(title=title, orient="top", direction="horizontal"),
    )


def figure_recall(*, table: pl.DataFrame) -> alt.TopLevelMixin:
    """MW-weighted recall of each arm on three populations, with the injected figure marked."""
    populations = {
        "All 66 labelled real events": table,
        "35 events lasting over 1 day": table.filter(pl.col("hours") > 24),
        "14 events of 25% or more of the receiving load, over 1 day": table.filter(
            (pl.col("hours") > 24) & (pl.col("share") >= 0.25)
        ),
    }
    rows = []
    for name, frame in populations.items():
        for arm in ARMS:
            point, lo, hi = bootstrap_weighted(
                found=frame[f"found|{arm}"].to_numpy().astype(float),
                weight=frame["moved_mw_p50"].to_numpy(),
            )
            rows.append({"population": name, "arm": arm, "recall": point, "lo": lo, "hi": hi})
    data = pl.DataFrame(rows).with_columns(
        inj_point=pl.lit(INJECTED_RECALL[0]),
        inj_lo=pl.lit(INJECTED_RECALL[1]),
        inj_hi=pl.lit(INJECTED_RECALL[2]),
    )
    x = alt.X(
        "recall:Q",
        scale=alt.Scale(domain=[0, 1]),
        axis=alt.Axis(
            values=[0, 0.25, 0.5, 0.75, 1],
            format=".2f",
            title="Share of moved MW found (higher is better)",
        ),
    )
    y = alt.Y("arm:N", sort=list(ARMS), title=None)
    band = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_rect(opacity=0.04, color=ocf.BRAND_ORANGE)
        .encode(x="inj_lo:Q", x2="inj_hi:Q")
    )
    rule = alt.Chart().mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 3]).encode(x="inj_point:Q")  # ty: ignore[unresolved-attribute]
    points = alt.Chart().mark_point(filled=True, size=70).encode(x=x, y=y, color=_colour())  # ty: ignore[unresolved-attribute]
    bars = alt.Chart().mark_rule(strokeWidth=2).encode(x="lo:Q", x2="hi:Q", y=y, color=_colour())  # ty: ignore[unresolved-attribute]
    return (
        alt.layer(band, rule, bars, points, data=data)
        .properties(width=WIDTH - 150, height=110)
        .facet(
            row=alt.Row(
                "population:N",
                sort=list(populations),
                title=None,
                header=alt.Header(
                    labelAngle=0, labelAlign="left", labelLimit=400, labelFontSize=11
                ),
            )
        )
        .properties(
            title=_title(
                text="Real labelled transfers: share of moved MW found, by detector",
                lines=[
                    (
                        "Dot: share of the moved MW (megawatts) whose event the detector found. "
                        "Line: "
                        "95% interval from resampling whole events; 14, 35 and 66 events are few. "
                        "Orange dashed line and band: RO-NC on injected transfers of the same kind "
                        "(0.770; 0.665 to 0.857)."
                    ),
                ],
            )
        )
    )


def figure_bands(*, table: pl.DataFrame) -> alt.TopLevelMixin:
    """Share of events found by PF14 and RO-NC, by share band and by duration band."""
    rows = []
    groups: list[tuple[str, str, pl.DataFrame]] = [
        ("Share of the receiving load", band, table.filter(pl.col("share_band") == band))
        for band in SHARE_BANDS
    ] + [
        (
            "Duration",
            name,
            table.filter((pl.col("hours") > lo) & (pl.col("hours") <= hi)),
        )
        for name, lo, hi in DURATION_BANDS
    ]
    for facet, band, frame in groups:
        for arm in ("PF14", "RO-NC"):
            k, n = int(frame[f"found|{arm}"].sum()), frame.height
            lo, hi = wilson(k=k, n=n)
            rows.append(
                {
                    "facet": facet,
                    "band": f"{band} (n={n})",
                    "arm": arm,
                    "share": k / n if n else 0.0,
                    "lo": lo,
                    "hi": hi,
                }
            )
    data = pl.DataFrame(rows)
    charts = []
    for facet in ("Share of the receiving load", "Duration"):
        sub = data.filter(pl.col("facet") == facet)
        order = list(dict.fromkeys(sub["band"]))
        y = alt.Y("band:N", sort=order, title=None)
        base = alt.Chart(sub)
        dots = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "share:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(
                    values=[0, 0.25, 0.5, 0.75, 1], format=".2f", title="Share of events found"
                ),
            ),
            y=y,
            yOffset=alt.YOffset("arm:N"),
            color=_colour(arms=("PF14", "RO-NC")),
        )
        lines = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x="lo:Q",
            x2="hi:Q",
            y=y,
            yOffset=alt.YOffset("arm:N"),
            color=_colour(arms=("PF14", "RO-NC")),
        )
        charts.append((lines + dots).properties(width=WIDTH - 200, height=100, title=facet))
    return alt.vconcat(*charts).properties(
        title=_title(
            text="Real events found, by size and by duration",
            lines=[
                (
                    "Dot: share of the labelled real events found (higher is better). Line: 95% "
                    "Wilson "
                    "interval. All 66 labelled events. PF14 is the benchmark single-series "
                    "detector; "
                    "RO-NC is the refit-offset neighbour-count detector."
                ),
            ],
        )
    )


def figure_strips(*, table: pl.DataFrame) -> alt.TopLevelMixin:
    """One strip per event over 1 day: the event, and where PF14 and RO-NC flagged it."""
    long = table.filter(pl.col("hours") > 24).sort("hours")
    order = [f"{e} ({h / 24:.1f} d)" for e, h in zip(long["event"], long["hours"], strict=True)]
    label = dict(zip(long["card_id"], order, strict=True))
    span = pl.DataFrame(
        {
            "event": [label[c] for c in long["card_id"]],
            "start": [0.0] * long.height,
            "end": (long["hours"] / 24).to_list(),
        }
    )
    flagged = (
        pl.read_parquet(WORK_DIR / "event_intervals.parquet")
        .filter(
            pl.col("card_id").is_in(long["card_id"].implode()),
            pl.col("arm").is_in(["PF14", "RO-NC"]),
        )
        .with_columns(
            event=pl.col("card_id").replace_strict(label),
            start=(pl.col("start_h").clip(lower_bound=-24) / 24),
            end=(pl.col("end_h") / 24),
        )
    )
    y = alt.Y("event:N", sort=order, title=None, axis=alt.Axis(labelFontSize=9))
    x = alt.X(
        "start:Q",
        scale=alt.Scale(type="sqrt", domain=[-1, 32], nice=False),
        axis=alt.Axis(
            values=[0, 1, 2, 4, 8, 16, 32],
            title="Days since the labelled start (square-root scale)",
        ),
    )
    background = (
        alt.Chart(span).mark_bar(color="#DDDDDD", size=11, clip=True).encode(x=x, x2="end:Q", y=y)  # ty: ignore[unresolved-attribute]
    )
    marks = (
        alt.Chart(flagged)  # ty: ignore[unresolved-attribute]
        .mark_bar(size=3.5, clip=True)
        .encode(
            x=x,
            x2="end:Q",
            y=y,
            yOffset=alt.YOffset("arm:N", sort=["PF14", "RO-NC"]),
            color=_colour(arms=("PF14", "RO-NC")),
        )
    )
    return (background + marks).properties(
        width=WIDTH - 130,
        height=14 * long.height,
        title=_title(
            text="Where the detectors flagged the 35 labelled events lasting over 1 day",
            lines=[
                (
                    "Grey bar: the event as labelled. Blue bar: interval PF14 flagged. Orange bar: "
                    "interval RO-NC flagged. Events are sorted by length; labels give the "
                    "anonymised "
                    "event number and its length in days. Every alarm needs 24 hours after the "
                    "step."
                ),
            ],
        ),
    )


def figure_steps(*, table: pl.DataFrame) -> alt.TopLevelMixin:
    """Peak step against share of receiving load for the 35 events over 1 day."""
    long = table.filter(pl.col("hours") > 24).with_columns(
        outcome=pl.when(pl.col("found|RO-NC"))
        .then(pl.lit("Found by RO-NC"))
        .otherwise(pl.lit("Missed by RO-NC")),
        power_present=pl.col("power_present_share") >= 0.5,
        share_plot=pl.col("share").fill_null(-0.1),
        peak_plot=pl.col("peak_step_z").fill_nan(None).fill_null(-1.0),
    )
    domain = ["Found by RO-NC", "Missed by RO-NC"]
    colour = alt.Color(
        "outcome:N",
        scale=alt.Scale(domain=domain, range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
        legend=alt.Legend(title=None, orient="top"),
    )
    shape = alt.Shape(
        "units:N",
        scale=alt.Scale(
            domain=["MW only", "MVA only", "mixed"], range=["circle", "square", "triangle-up"]
        ),
        legend=alt.Legend(title="Series type", orient="top"),
    )
    points = (
        alt.Chart(long)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=90, opacity=0.85)
        .encode(
            x=alt.X(
                "share_plot:Q",
                scale=alt.Scale(domain=[-0.15, 1.4]),
                axis=alt.Axis(
                    values=[0, 0.25, 0.5, 0.75, 1.0, 1.25],
                    title=(
                        "Moved MW as a share of the receiving load "
                        "(events with no metered receiver sit left of 0)"
                    ),
                ),
            ),
            y=alt.Y(
                "peak_plot:Q",
                scale=alt.Scale(domain=[-2, 10]),
                axis=alt.Axis(
                    values=[0, 2, 4, 6, 8, 10],
                    title="24-hour step in noise units (series without data sit below 0)",
                ),
            ),
            color=colour,
            shape=shape,
        )
    )
    threshold = (
        alt.Chart(pl.DataFrame({"y": [4.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#888888")
        .encode(y="y:Q")
    )
    quarter = (
        alt.Chart(pl.DataFrame({"x": [0.25]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#888888")
        .encode(x="x:Q")
    )
    return (points + threshold + quarter).properties(
        width=WIDTH - 60,
        height=260,
        title=_title(
            text="Why events are missed: a small step, no metered receiver, or no data",
            lines=[
                (
                    "Each mark is one of the 35 labelled events lasting over 1 day. Dashed "
                    "lines: the "
                    "4.0 noise-unit alarm threshold of the benchmark detector and the 25% share "
                    "used "
                    "for the injected transfers. Noise unit: the usual size of the 24-hour step on "
                    "odd-numbered months."
                ),
            ],
        ),
    )


def main() -> None:
    """Draw every figure."""
    table = event_table()
    _save(figure_recall(table=table), name="figure_1_recall_by_detector.svg")
    _save(figure_bands(table=table), name="figure_2_recall_by_band.svg")
    _save(figure_strips(table=table), name="figure_3_event_strips.svg")
    _save(figure_steps(table=table), name="figure_4_step_and_share.svg")
    print(np.__name__)


if __name__ == "__main__":
    main()
