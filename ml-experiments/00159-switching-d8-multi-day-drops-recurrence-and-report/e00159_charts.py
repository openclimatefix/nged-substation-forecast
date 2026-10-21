"""Draw the figures of experiment 159 from the tables that `e00159_drops.py` wrote.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.
"""

from __future__ import annotations

import pickle
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00159_drops import (
    D8_SERIES,
    FALSE_ALARM_DIR,
    HALF_HOURS_PER_DAY,
    NEIGHBOUR_LABELS,
    NEIGHBOUR_SERIES,
    STEP_FRACTIONS,
    WORK_DIR,
    wilson,
)

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00159-switching-d8-multi-day-drops-recurrence-and-report"
)
WIDTH: Final[int] = 640
WRAP: Final[int] = 95
SHADE: Final[str] = ocf.DATA_SKY_LIGHT
NEIGHBOUR_COLOURS: Final[dict[str, str]] = {
    "D3": ocf.DATA_PURPLE,
    "D14": ocf.DATA_GREEN,
    "D5": ocf.BRAND_ORANGE,
}
WEEKDAYS: Final[list[str]] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
ZOOM_MARGIN_DAYS: Final[int] = 3


def _title(*, text: str, line: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=text,
        subtitle=textwrap.wrap(line, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _time_axis(*, days: np.ndarray, start: np.datetime64) -> np.ndarray:
    return start + (days * 86400).astype("timedelta64[s]")


def figure_history(*, arrays: dict, drops: pl.DataFrame) -> alt.TopLevelMixin:
    """D8's power over all history, daily means, with every drop shaded."""
    power = arrays[D8_SERIES]["power"]
    block = HALF_HOURS_PER_DAY
    n = len(power) // block * block
    mean = np.nanmean(power[:n].reshape(-1, block), axis=1)
    start = np.datetime64("2022-01-01T00:00", "us")
    when = start + (np.arange(len(mean)) * block * 30).astype("timedelta64[m]").astype(
        "timedelta64[us]"
    )
    line_data = pl.DataFrame({"time": when, "power_mw": mean})
    shade = pl.DataFrame(
        {
            "begin": start
            + (drops["first_stamp"].to_numpy() * 30)
            .astype("timedelta64[m]")
            .astype("timedelta64[us]"),
            "finish": start
            + (drops["last_stamp"].to_numpy() * 30)
            .astype("timedelta64[m]")
            .astype("timedelta64[us]"),
        }
    )
    rects = (
        alt.Chart(shade)
        .mark_rect(color=ocf.DATA_SKY, opacity=0.55)
        .encode(x=alt.X("begin:T", title=None), x2="finish:T")  # ty: ignore[unresolved-attribute]
    )
    line = (
        alt.Chart(line_data)
        .mark_line(color=ocf.DATA_BLUE, strokeWidth=0.8)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("time:T", title=None),
            y=alt.Y("power_mw:Q", title="D8 power (MW, daily mean)"),
        )
    )
    return (rects + line).properties(
        width=WIDTH,
        height=200,
        title=_title(
            text="D8's 29 drops (shaded) sit in the second half of its history",
            line="D8 power, January 2022 to June 2026. Shaded: runs of 2 days or longer in which "
            "the 24-hour median of metered power minus the convex model's expected power is "
            "below -1.5 MW.",
        ),
    )


def _zoom_panel(*, arrays: dict, row: dict[str, Any], label: str) -> alt.VConcatChart:
    first, last = int(row["first_stamp"]), int(row["last_stamp"])
    margin = ZOOM_MARGIN_DAYS * HALF_HOURS_PER_DAY
    lo, hi = max(first - margin, 0), min(last + margin, len(arrays[D8_SERIES]["power"]))
    days = (np.arange(lo, hi) - first) / HALF_HOURS_PER_DAY
    d8 = arrays[D8_SERIES]
    frame = pl.DataFrame(
        {
            "day": days,
            "Metered power": d8["power"][lo:hi],
            "Expected power": d8["expected"][lo:hi],
            "Residual": d8["resid"][lo:hi],
        }
    )
    domain = [-ZOOM_MARGIN_DAYS, (last - first) / HALF_HOURS_PER_DAY + ZOOM_MARGIN_DAYS]
    x = alt.X("day:Q", title=None, scale=alt.Scale(domain=domain), axis=alt.Axis(values=[]))
    shade = pl.DataFrame({"a": [0.0], "b": [(last - first) / HALF_HOURS_PER_DAY]})
    rect = alt.Chart(shade).mark_rect(color=SHADE, opacity=0.5).encode(x="a:Q", x2="b:Q")  # ty: ignore[unresolved-attribute]
    long = frame.unpivot(
        index="day", on=["Metered power", "Expected power"], variable_name="series"
    )
    colour = alt.Color(
        "series:N",
        scale=alt.Scale(
            domain=["Metered power", "Expected power"],
            range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
        ),
        legend=None if label != "first" else alt.Legend(title=None, orient="top"),
    )
    top = (
        alt.Chart(long)
        .mark_line(strokeWidth=0.9)
        .encode(x=x, y=alt.Y("value:Q", title="D8 (MW)"), color=colour)  # ty: ignore[unresolved-attribute]
    )
    top = (rect + top).properties(
        width=WIDTH,
        height=110,
        title=alt.TitleParams(
            text=f"Drop {row['drop']}: depth {row['depth_mw']:.1f} MW, "
            f"{row['duration_days']:.1f} days",
            anchor="start",
            fontSize=12,
        ),
    )
    neighbours = pl.DataFrame({"day": days}).with_columns(
        *[pl.Series(NEIGHBOUR_LABELS[s], arrays[s]["power"][lo:hi]) for s in NEIGHBOUR_SERIES]
    )
    nlong = neighbours.unpivot(
        index="day", on=list(NEIGHBOUR_LABELS.values()), variable_name="neighbour"
    )
    bottom = (
        alt.Chart(nlong)
        .mark_line(strokeWidth=0.9)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "day:Q",
                title="Days from the drop's start" if label == "last" else None,
                scale=alt.Scale(domain=domain),
                axis=alt.Axis(labels=label == "last"),
            ),
            y=alt.Y("value:Q", title="Neighbours (MW)"),
            color=alt.Color(
                "neighbour:N",
                scale=alt.Scale(
                    domain=list(NEIGHBOUR_COLOURS), range=list(NEIGHBOUR_COLOURS.values())
                ),
                legend=None if label != "first" else alt.Legend(title=None, orient="top"),
            ),
        )
    )
    bottom = (rect + bottom).properties(width=WIDTH, height=80)
    return alt.vconcat(top, bottom, spacing=2).resolve_scale(color="independent")


def figure_zoom(
    *, arrays: dict, drops: pl.DataFrame, picks: list[int], name: str
) -> alt.TopLevelMixin:
    """Zoomed drops: metered and expected power, and the three connected neighbours' power."""
    rows = drops.filter(pl.col("drop").is_in(picks)).sort("drop").to_dicts()
    panels: list[alt.VConcatChart] = []
    for k, row in enumerate(rows):
        label = "first" if k == 0 else "last" if k == len(rows) - 1 else "mid"
        panels.append(_zoom_panel(arrays=arrays, row=row, label=label))
    return (
        alt.vconcat(*panels, spacing=14)
        .resolve_scale(color="independent")
        .properties(
            title=_title(
                text=name,
                line="Each drop is shaded, with 3 days either side. Top row: metered and expected "
                "power of D8. Bottom row: power of the three connected metered neighbours D3, D14, "
                "and D5.",
            )
        )
    )


def figure_recurrence(*, drops: pl.DataFrame) -> alt.TopLevelMixin:
    """Weekday and hour-of-day histograms of the drops' starts, and the spacing between them."""
    weekday = (
        drops.group_by("start_weekday")
        .len()
        .join(pl.DataFrame({"start_weekday": range(7)}), on="start_weekday", how="right")
        .fill_null(0)
        .with_columns(day=pl.col("start_weekday").replace_strict(dict(enumerate(WEEKDAYS))))
    )
    bars = (
        alt.Chart(weekday)
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "day:N",
                sort=WEEKDAYS,
                title="Weekday of the drop's start",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("len:Q", title="Drops"),
        )
        .properties(width=WIDTH, height=110)
    )
    hours = drops.with_columns(block=(pl.col("start_hour") // 3 * 3).cast(pl.Int64))
    hour_bars = (
        alt.Chart(hours)
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "block:Q",
                bin=alt.Bin(step=3, extent=[0, 24]),
                title="Hour of day of the drop's start (local time)",
                axis=alt.Axis(values=[0, 6, 12, 18, 24]),
            ),
            y=alt.Y("count():Q", title="Drops"),
        )
        .properties(width=WIDTH, height=110)
    )
    gaps = (
        drops.sort("first_stamp")
        .with_columns(gap=pl.col("first_stamp").diff() / HALF_HOURS_PER_DAY)
        .drop_nulls("gap")
    )
    gap_bars = (
        alt.Chart(gaps)
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "gap:Q",
                bin=alt.Bin(step=5, extent=[0, 65]),
                title="Days from one drop's start to the next (gaps over 65 days not shown)",
            ),
            y=alt.Y("count():Q", title="Drops"),
        )
        .properties(width=WIDTH, height=110)
    )
    return alt.vconcat(bars, hour_bars, gap_bars, spacing=18).properties(
        title=_title(
            text="The drops start on no particular weekday, hour, or interval",
            line="The 29 drops. No test of the weekday and hour-of-day counts against a flat "
            "pattern reaches p = 0.05.",
        )
    )


def figure_depth(*, drops: pl.DataFrame) -> alt.TopLevelMixin:
    """Depth and duration of each drop over time."""
    data = drops.select(
        pl.col("start_date").cast(pl.Datetime).alias("start"), "depth_mw", "duration_days"
    )
    return (
        alt.Chart(data)
        .mark_circle(color=ocf.DATA_BLUE, opacity=0.85)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("start:T", title="Start of the drop"),
            y=alt.Y("depth_mw:Q", title="Depth (MW)", scale=alt.Scale(domain=[0, 6])),
            size=alt.Size(
                "duration_days:Q", title="Duration (days)", scale=alt.Scale(range=[30, 500])
            ),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=_title(
                text="Drops are 1 to 5 MW deep, and the deepest come earlier",
                line="Depth is the median of expected minus metered power during the drop. "
                "Spearman correlation of depth with start time is -0.40 (p = 0.03).",
            ),
        )
    )


def figure_neighbours(*, drops: pl.DataFrame, placebo: dict[float, float]) -> alt.TopLevelMixin:
    """Share of drops with a neighbour step-up of a given fraction of the depth, against chance."""
    depth = drops["depth_mw"].to_numpy()
    steps = drops["best_neighbour_step_mw"].to_numpy()
    n = len(depth)
    rows = []
    for fraction in STEP_FRACTIONS:
        k = int((steps >= fraction * depth).sum())
        low, high = wilson(k=k, n=n)
        rows.append(
            {
                "fraction": f"{fraction:.2f} x depth",
                "group": "At a drop's start",
                "share": k / n,
                "low": low,
                "high": high,
            }
        )
        rows.append(
            {
                "fraction": f"{fraction:.2f} x depth",
                "group": "At a random time, no drop",
                "share": placebo[fraction],
                "low": placebo[fraction],
                "high": placebo[fraction],
            }
        )
    data = pl.DataFrame(rows)
    colour = alt.Color(
        "group:N",
        scale=alt.Scale(
            domain=["At a drop's start", "At a random time, no drop"],
            range=[ocf.DATA_BLUE, ocf.DATA_BLUE_LIGHT],
        ),
        legend=alt.Legend(title=None, orient="top"),
    )
    base = alt.Chart(data)
    bars = base.mark_bar().encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "fraction:N",
            title="Step-up of the best neighbour, as a share of the drop's depth",
            axis=alt.Axis(labelAngle=0),
        ),
        xOffset="group:N",
        y=alt.Y("share:Q", title="Share of drops", scale=alt.Scale(domain=[0, 0.7])),
        color=colour,
    )
    errors = (
        base.transform_filter(alt.datum.group == "At a drop's start")
        .mark_rule(color="black")
        .encode(x="fraction:N", xOffset="group:N", y="low:Q", y2="high:Q")  # ty: ignore[unresolved-attribute]
    )
    return (bars + errors).properties(
        width=WIDTH,
        height=200,
        title=_title(
            text="No connected neighbour steps up at a drop more often than by chance",
            line="Best step-up of D3, D14, and D5 (6-hour median after the drop's start minus "
            "6-hour median before). Whiskers: 95% Wilson interval. The comparison is 5,000 "
            "random times at least 2 days from any drop.",
        ),
    )


def figure_alarms(*, drops: pl.DataFrame, alarms: pl.DataFrame) -> alt.TopLevelMixin:
    """Every unexplained RO-NC alarm on D8 against the drops."""
    start = np.datetime64("2022-01-01T00:00", "us")
    shade = pl.DataFrame(
        {
            "begin": start
            + (drops["first_stamp"].to_numpy() * 30)
            .astype("timedelta64[m]")
            .astype("timedelta64[us]"),
            "finish": start
            + (drops["last_stamp"].to_numpy() * 30)
            .astype("timedelta64[m]")
            .astype("timedelta64[us]"),
        }
    )
    ticks = pl.DataFrame(
        {
            "time": start
            + (alarms["stamp"].to_numpy() * 30).astype("timedelta64[m]").astype("timedelta64[us]"),
            "direction": np.where(alarms["step_mw"].to_numpy() < 0, "Step down", "Step up"),
            "step_mw": alarms["step_mw"].to_numpy(),
        }
    )
    rects = (
        alt.Chart(shade)
        .mark_rect(color=ocf.DATA_SKY, opacity=0.55)
        .encode(x=alt.X("begin:T", title=None), x2="finish:T")  # ty: ignore[unresolved-attribute]
    )
    points = (
        alt.Chart(ticks)
        .mark_point(filled=True, size=45, opacity=0.9)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("time:T", title=None),
            y=alt.Y("step_mw:Q", title="Alarm step (MW)"),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Step down", "Step up"], range=[ocf.BRAND_ORANGE, ocf.DATA_PURPLE]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
    )
    return (rects + points).properties(
        width=WIDTH,
        height=170,
        title=_title(
            text="Only 2 of the 28 unexplained alarms fall within 6 hours of a drop's start or end",
            line="Each dot is an alarm that RO-NC raised on D8 and that matches no logged "
            "closure. Shaded: the 29 drops.",
        ),
    )


def pick_typical(*, drops: pl.DataFrame) -> tuple[list[int], list[int]]:
    """Six drops spread across depth and duration: three per figure."""
    ordered = drops.sort("depth_mw")
    n = ordered.height
    chosen = [ordered["drop"][int(q * (n - 1))] for q in (0.1, 0.3, 0.5, 0.7, 0.9)]
    longest = drops.sort("duration_days")["drop"][-1]
    if longest not in chosen:
        chosen.append(longest)
    else:
        chosen.append(ordered["drop"][int(0.99 * (n - 1))])
    chosen = sorted({int(c) for c in chosen})
    return chosen[:3], chosen[3:6]


def main() -> None:
    """Draw every figure."""
    arrays = pickle.loads((FALSE_ALARM_DIR / "arrays.pkl").read_bytes())
    drops = pl.read_parquet(WORK_DIR / "drops.parquet")
    alarms = pl.read_parquet(WORK_DIR / "alarms_vs_drops.parquet")
    placebo = {float(k): v for k, v in pl.read_parquet(WORK_DIR / "placebo.parquet").iter_rows()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    first, second = pick_typical(drops=drops)
    figures = {
        "figure_1_history.svg": figure_history(arrays=arrays, drops=drops),
        "figure_2_typical_drops_a.svg": figure_zoom(
            arrays=arrays, drops=drops, picks=first, name="Three typical drops: the shallower ones"
        ),
        "figure_3_typical_drops_b.svg": figure_zoom(
            arrays=arrays, drops=drops, picks=second, name="Three typical drops: the deeper ones"
        ),
        "figure_4_recurrence.svg": figure_recurrence(drops=drops),
        "figure_5_depth.svg": figure_depth(drops=drops),
        "figure_6_neighbours.svg": figure_neighbours(drops=drops, placebo=placebo),
        "figure_7_alarms.svg": figure_alarms(drops=drops, alarms=alarms),
    }
    for name, chart in figures.items():
        chart.save(OUT_DIR / name)
    print(f"wrote {len(figures)} figures; zoom drops {first} and {second}")


if __name__ == "__main__":
    main()
