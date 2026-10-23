# ruff: noqa: E501
"""Draw the figures of experiment 274 as SVG beside the write-up.

OMP_NUM_THREADS=1 uv run --no-sync python e00274_charts.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import altair as alt
import e00274_fit as fit
import e00274_report as rep
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import CONTENT_WIDTH_PX, wrapped

OUT: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00274-lsc-long-arrangement-audit-gam-refit"
)
WIDTH: Final[int] = CONTENT_WIDTH_PX - 120
DURATIONS: Final[tuple[int, ...]] = (7, 14, 30, 60, 90)
COLOURS: Final[list[str]] = [ocf.DATA_SKY, ocf.DATA_BLUE, ocf.BRAND_ORANGE]
BASE: Final[dict[str, object]] = {"shape": "share", "sign": 1}


def save(chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG."""
    chart.configure_view(stroke=None).save(OUT / f"{name}.svg")


def band(
    *, rows: list[dict], x: str, color: str, color_domain: list[str], y_title: str
) -> alt.Chart:
    """Return points with 95% interval bars from rows holding x, color, mean, low, high."""
    frame = pl.DataFrame(rows).to_pandas()
    encode = {
        "x": alt.X(
            f"{x}:O",
            title=None,
            axis=alt.Axis(labelAngle=0),
            sort=list(dict.fromkeys(str(row[x]) for row in rows)),
        ),
        "color": alt.Color(
            f"{color}:N",
            scale=alt.Scale(domain=color_domain, range=COLOURS[: len(color_domain)]),
            legend=alt.Legend(title=None, orient="top"),
        ),
        "xOffset": alt.XOffset(f"{color}:N"),
    }
    bars = (
        alt.Chart(frame)
        .mark_rule(strokeWidth=2)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("low:Q", title=y_title), y2="high:Q", **encode
        )
    )
    points = (
        alt.Chart(frame).mark_point(filled=True, size=70).encode(y="mean:Q", **encode)  # ty: ignore[unresolved-attribute]
    )
    return bars + points


def fig_after_end(*, units: pl.DataFrame) -> None:
    """After-end bias in points against duration, by transfer size, with the 0.1 tolerance rule."""
    rows = []
    for size in fit.SIZES:
        for days in DURATIONS:
            mean, low, high = rep.cell(
                units=units,
                where={
                    "window": "24m",
                    "size": size,
                    "duration_days": days,
                    "phase_group": "AFTER_48H",
                    **BASE,
                },
            )
            rows.append(
                {
                    "days": str(days),
                    "size": f"{size:.0%} of load",
                    "mean": mean,
                    "low": low,
                    "high": high,
                }
            )
    chart = band(
        rows=rows,
        x="days",
        color="size",
        color_domain=[f"{s:.0%} of load" for s in fit.SIZES],
        y_title="Forecast too high after the transfer (points; 0 is unbiased)",
    )
    rule = (
        alt.Chart(pl.DataFrame({"y": [rep.TOLERANCE_POINTS]}).to_pandas())
        .mark_rule(color="#5F6368", strokeDash=[4, 3])
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    save(
        (chart + rule).properties(
            width=WIDTH,
            height=260,
            title=alt.TitleParams(
                wrapped(
                    text="After a transfer ends, the forecast is too high by more than the 0.1-point tolerance at every duration tested",
                    width=62,
                ),
                subtitle=wrapped(
                    text="Mean forecast minus the forecast trained on the clean series, in the 48 hours after the "
                    "transfer ends, 24-month training window, monthly refits, 10 Primary substations and 4 onset "
                    "dates. Dot: mean. Bar: 95% interval from resampling substations and onset dates. Dashed "
                    "line: the 0.1-point tolerance. Horizontal axis: duration of the transfer in days."
                ),
            ),
        ),
        "after_end_bias",
    )


def fig_window(*, units: pl.DataFrame) -> None:
    """Learned share in the 48 hours after the end against duration, by training window."""
    names = {"24m": "24-month window", "12m": "12-month window", "all": "whole history"}
    rows = []
    for window, label in names.items():
        for days in DURATIONS:
            mean, low, high = rep.cell(
                units=units,
                column="share",
                where={
                    "window": window,
                    "size": 0.25,
                    "duration_days": days,
                    "phase_group": "AFTER_48H",
                    **BASE,
                },
            )
            rows.append(
                {"days": str(days), "window": label, "mean": mean, "low": low, "high": high}
            )
    chart = band(
        rows=rows,
        x="days",
        color="window",
        color_domain=list(names.values()),
        y_title="Share of the transfer learned as normal",
    )
    save(
        chart.properties(
            width=WIDTH,
            height=240,
            title=alt.TitleParams(
                wrapped(
                    text="A shorter training window learns a transfer faster, and the whole history learns it slowest",
                    width=62,
                ),
                subtitle=wrapped(
                    text="Bias in the 48 hours after a transfer of 25% of load ends, divided by the mean transfer. "
                    "Dot: mean. Bar: 95% interval. Horizontal axis: duration in days."
                ),
            ),
        ),
        "learned_share_by_window",
    )


def fig_phases(*, units: pl.DataFrame) -> None:
    """Learned share by phase for 30-day and 90-day transfers."""
    phases = ["DURING", "AFTER_48H", "LATER_3M", "LATER_4_12M"]
    labels = {
        "DURING": "during",
        "AFTER_48H": "first 48 h after",
        "LATER_3M": "months 1 to 3 after",
        "LATER_4_12M": "months 4 to 12 after",
    }
    rows = []
    for days in (30, 90):
        for phase in phases:
            mean, low, high = rep.cell(
                units=units,
                column="share",
                where={
                    "window": "24m",
                    "size": 0.25,
                    "duration_days": days,
                    "phase_group": phase,
                    **BASE,
                },
            )
            rows.append(
                {
                    "phase": labels[phase],
                    "duration": f"{days}-day transfer",
                    "mean": mean,
                    "low": low,
                    "high": high,
                }
            )
    chart = band(
        rows=rows,
        x="phase",
        color="duration",
        color_domain=["30-day transfer", "90-day transfer"],
        y_title="Share of the transfer learned as normal",
    )
    save(
        chart.properties(
            width=WIDTH,
            height=240,
            title=alt.TitleParams(
                wrapped(
                    text="The learned share is largest in the 48 hours after a transfer ends, falls within three months, and rises again later in the year",
                    width=62,
                ),
                subtitle=wrapped(
                    text="Transfer of 25% of load, 24-month window, monthly refits. Dot: mean. Bar: 95% "
                    "interval. During-transfer rows start at the second monthly refit, because the first refit "
                    "has not yet seen the transfer."
                ),
            ),
        ),
        "learned_share_by_phase",
    )


def fig_permanent(*, rows: pl.DataFrame) -> None:
    """Learned share of a permanent step against months since the onset, by training window."""
    names = {"24m": "24-month window", "12m": "12-month window", "all": "whole history"}
    selected = rows.filter(
        pl.col("duration_days") == -1,
        pl.col("size") == 0.25,
        pl.col("sign") == 1,
        pl.col("shape") == "share",
        pl.col("phase") == "DURING",
    ).with_columns(
        months=((pl.col("cut") - pl.col("onset")).dt.total_days() / 30.4375).round().cast(pl.Int32),
        window_name=pl.col("window").replace_strict(names),
        learned=pl.col("mean_bias") / pl.col("mean_transfer"),
    )
    curve = (
        selected.filter(pl.col("months") <= 36)
        .group_by("window_name", "months")
        .agg(learned=pl.col("learned").mean())
        .sort("months")
    )
    chart = (
        alt.Chart(curve.to_pandas())  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "months:Q",
                title="Months since the step began",
                axis=alt.Axis(values=[0, 6, 12, 18, 24, 30, 36]),
            ),
            y=alt.Y("learned:Q", title="Share of the step learned as normal"),
            color=alt.Color(
                "window_name:N",
                scale=alt.Scale(domain=list(names.values()), range=COLOURS),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
    )
    half = (
        alt.Chart(pl.DataFrame({"y": [0.5]}).to_pandas())
        .mark_rule(color="#5F6368", strokeDash=[4, 3])
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    save(
        (chart + half).properties(
            width=WIDTH,
            height=240,
            title=alt.TitleParams(
                wrapped(
                    text="A permanent step is half learned after about 12 months with a 24-month window, and fully learned once the window holds only the new arrangement",
                    width=62,
                ),
                subtitle=wrapped(
                    text="Step of 25% of load, mean over 10 Primary substations and 4 start dates, monthly refits. "
                    "Dashed line: half learned."
                ),
            ),
        ),
        "permanent_step",
    )


def main() -> None:
    """Draw every figure."""
    units = rep.unit_means(rows=rep.load_rows())
    fig_after_end(units=units)
    fig_window(units=units)
    fig_phases(units=units)
    fig_permanent(rows=rep.load_rows())


if __name__ == "__main__":
    main()
