"""Draw experiment 43's figures as SVG: a raw injected event first, then recall by band.

Series are Primary substations (named). The SVGs are afterwards optimised with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00043-sw2-1-detector-scoreboard/e00043_charts.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00043_common import WORK_DIR, load_panel
from e00043_inject import DURATION_BANDS, HEADLINE_BAND, SIZE_BANDS, Event, inject_power
from e00043_report import LABELS, wilson
from e00043_run import SEED, _detect, _scales

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00043-sw2-1-detector-scoreboard"
)
WIDTH: Final[int] = 640
WRAP: Final[int] = 100


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def figure_1() -> alt.TopLevelMixin:
    """Residual of the donor and a metered receiver around one injected headline event."""
    panel = load_panel()
    events = pl.read_parquet(WORK_DIR / "events.parquet")
    results = pl.read_parquet(WORK_DIR / "event_results.parquet")
    pick = (
        events.filter(
            pl.col("size_band") == HEADLINE_BAND[0],
            pl.col("duration_band") == HEADLINE_BAND[1],
            pl.col("metered_receivers") > 0,
            pl.col("hours") < 120,
        )
        .join(
            results.filter(pl.col("setting") == "alarms_D1_48_2.5", pl.col("variant") == "plain"),
            on="event_id",
        )
        .filter(pl.col("found"))
        .sort("event_id")
        .row(0, named=True)
    )
    receivers = tuple(int(r) for r in pick["receivers"].split(","))
    event = Event(
        event_id=0,
        donor=pick["donor"],
        receivers=receivers,
        weights=tuple(1 / len(receivers) for _ in receivers),
        onset=pick["onset"],
        end=pick["end"],
        size_band=pick["size_band"],
        duration_band=pick["duration_band"],
        fraction=pick["fraction"],
    )
    injection = inject_power(power=panel["power"], events=[event], seed=SEED)
    residual = injection.fast_residual(panel["residual"])
    scales = _scales(residual=panel["residual"])
    intervals = _detect(residual=residual, expected=panel["expected"], scales=scales)[
        ("alarms_D1_48_2.5", "plain")
    ][1]
    lo, hi = event.onset - 5 * 48, event.end + 5 * 48
    hours = (np.arange(lo, hi) - event.onset) / 2
    frames = [
        pl.DataFrame({"hours": hours, "residual": residual[s][lo:hi], "series": f"Series {s}"})
        for s in (event.donor, event.receivers[0])
    ]
    data = pl.concat(frames)
    shade = (
        alt.Chart(pl.DataFrame({"x": [0.0, (event.end - event.onset) / 2]}))
        .mark_rule(color="#444444", strokeDash=[5, 3], strokeWidth=1.5)
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    spans = pl.DataFrame(
        [
            {"a": (i.onset - event.onset) / 2, "b": (i.end - event.onset) / 2}
            for i in intervals
            if i.series in event.series and lo <= i.onset < hi
        ],
        schema={"a": pl.Float64, "b": pl.Float64},
    )
    detected = (
        alt.Chart(spans)
        .mark_rect(opacity=0.35, color=ocf.DATA_GREEN, yOffset=0)
        .encode(x="a:Q", x2="b:Q")  # ty: ignore[unresolved-attribute]
    )
    line = (
        alt.Chart(data)
        .mark_line(strokeWidth=1.1, clip=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("hours:Q", title="Hours from the true start of the injected transfer"),
            y=alt.Y("residual:Q", title="Residual (MW)"),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(
                    domain=sorted(data["series"].unique().to_list()),
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                ),
            ),
        )
    )
    return alt.layer(shade, detected, line).properties(
        width=WIDTH - 130,
        height=220,
        title=_title(
            text="An injected transfer shows as opposite steps; the detector's interval is green",
            lines=[
                (
                    "Dashed lines: the true start and end of the transfer, added to the real "
                    "residual (observed power minus the weather model's forecast) of the donor "
                    "and one receiver. Green: the interval "
                    "that the neighbour-balance detector D1 (24-hour window) reports. An end more "
                    "than 6 hours from the dashed end counts as a miss."
                )
            ],
        ),
    )


def figure_2() -> alt.TopLevelMixin:
    """Recall by size and duration band for the four detector settings."""
    events = pl.read_parquet(WORK_DIR / "events.parquet")
    results = (
        pl.read_parquet(WORK_DIR / "event_results.parquet")
        .filter(pl.col("variant") == "plain")
        .join(events, on="event_id")
    )
    rows = []
    for setting, label in LABELS.items():
        for size in SIZE_BANDS:
            for duration in DURATION_BANDS:
                band = results.filter(
                    pl.col("setting") == setting,
                    pl.col("size_band") == size,
                    pl.col("duration_band") == duration,
                )
                k = int(band["found"].sum())
                low, high = wilson(k, band.height)
                rows.append(
                    {
                        "detector": label,
                        "band": f"{size} of receiving load, {duration}",
                        "recall": k / band.height,
                        "low": low,
                        "high": high,
                    }
                )
    data = pl.DataFrame(rows)
    order = [f"{s} of receiving load, {d}" for s in SIZE_BANDS for d in DURATION_BANDS]
    domain = list(LABELS.values())
    colours = [ocf.DATA_BLUE_LIGHT, ocf.DATA_BLUE, ocf.BRAND_ORANGE_LIGHT, ocf.BRAND_ORANGE]
    colour = alt.Color("detector:N", title=None, scale=alt.Scale(domain=domain, range=colours))
    offset = alt.YOffset("detector:N", scale=alt.Scale(domain=domain))
    y = alt.Y("band:N", sort=order, title=None, axis=alt.Axis(labelLimit=300))
    bars = (
        alt.Chart(data)
        .mark_rule(strokeWidth=2)
        .encode(  # ty: ignore[unresolved-attribute]
            y=y,
            yOffset=offset,
            x=alt.X(
                "low:Q",
                title="Share of injected transfers found (larger is better)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            x2="high:Q",
            color=colour,
        )
    )
    dots = (
        alt.Chart(data)
        .mark_point(filled=True, size=55, opacity=1)
        .encode(  # ty: ignore[unresolved-attribute]
            y=y, yOffset=offset, x="recall:Q", color=colour
        )
    )
    return alt.layer(bars, dots, data=data).properties(
        width=WIDTH - 300,
        height=340,
        title=_title(
            text="No detector setting finds more than 0.3 of any band of injected transfers",
            lines=[
                (
                    "Dot: share of 600 injected transfers (55 per band, 160 in the 25% or more, "
                    "over 1 day band) whose start is within 2 hours and end within 6 hours or "
                    "10% of the duration. Line: 95% Wilson interval."
                )
            ],
        ),
    )


def main() -> None:
    """Write both figures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in (
        ("figure_1_injected_event", figure_1()),
        ("figure_2_recall_by_band", figure_2()),
    ):
        chart.save(OUT_DIR / f"{name}.svg")
        print("wrote", name)


if __name__ == "__main__":
    main()
