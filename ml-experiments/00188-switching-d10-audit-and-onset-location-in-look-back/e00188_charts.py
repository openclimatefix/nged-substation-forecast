"""Draw experiment 188's figures as SVG from the stored tables and reports.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.
The series are Primary substations and injected transfers, not NGED generators.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00188-*/e00188_charts.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00188_p2_report as rep
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00188_common import WORK_DIR

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00188-switching-d10-audit-and-onset-location-in-look-back"
)
WIDTH: Final[int] = 520
GREY: Final[str] = "#888888"


def _title(*, text: str, line: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 70),
        subtitle=textwrap.wrap(line, 85),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _dots(
    *,
    data: pl.DataFrame,
    title: alt.TitleParams,
    x_title: str,
    domain: list[float],
    height: int,
    zero: bool = True,
    limits: tuple[float, ...] = (),
    colour: str = ocf.DATA_BLUE,
) -> alt.TopLevelMixin:
    layers = [
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=2, color=colour)
        .encode(
            x=alt.X("low:Q", title=x_title, scale=alt.Scale(domain=domain)),
            x2="high:Q",
            y=alt.Y("name:N", title=None, sort=None, axis=alt.Axis(labelLimit=330)),
        ),
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70, color=colour)
        .encode(x="mean:Q", y=alt.Y("name:N", sort=None)),
    ]
    if zero:
        layers.append(
            alt.Chart(pl.DataFrame({"v": [0.0]}))
            .mark_rule(strokeDash=[4, 3], color=GREY)
            .encode(x="v:Q")  # ty: ignore[unresolved-attribute]
        )
    if limits:
        layers.append(
            alt.Chart(pl.DataFrame({"v": list(limits)}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[2, 2], color=ocf.BRAND_ORANGE)
            .encode(x="v:Q")
        )
    return alt.layer(*layers).properties(width=WIDTH, height=height, title=title)


def figure_1(*, report: dict) -> None:
    """Extra error of each mask start over a mask from the true onset, injected transfers."""
    rows = []
    for family, label in (("BOTH", "Both alarms"), ("PAIR", "Pair alarm"), ("RONC", "RO-NC")):
        for arm, arm_label in (
            ("MASK_ALARM", "from the alarm time"),
            ("MASK_STEP", "from the detector's step stamp"),
            ("MASK_LOCATED", "from the located onset"),
        ):
            t = report[f"extra|injected {family}|{arm}"]
            rows.append({"name": f"{label}: {arm_label}", "mean": t[0], "low": t[1], "high": t[2]})
    title = _title(
        text="Masking back to the step stamp removes the extra error; locating adds nothing",
        line=(
            "Mean quantile score of each mask minus the mask that starts at the true onset "
            "(points of "
            "effective capacity; smaller is better). Dot: estimate. Line: 95% interval from "
            "resampling "
            "found transfers. 247 transfers injected into 44 worlds, six Primary substations."
        ),
    )
    _dots(
        data=pl.DataFrame(rows),
        title=title,
        height=250,
        domain=[-0.05, 0.8],
        x_title="Extra error over the true-onset mask (points; smaller is better)",
    ).save(OUT_DIR / "figure_1_headline.svg")


def figure_2(*, audit: dict) -> None:
    """Part 1: the no-harm contrast before and after removing confirmed unlogged events."""
    names = [
        ("Experiment 171's rows, pooled", "171 pooled"),
        ("Confirmed unlogged events removed, pooled", "pooled, confirmed events removed"),
        ("Same, November to March", "winter, confirmed events removed"),
        ("D10 alone, all rows", "only D10, all rows"),
        ("D10 alone, confirmed events removed", "only D10, confirmed events removed"),
        ("Rows of unconfirmed gated alarms alone", "rows of unconfirmed gated alarms alone"),
    ]
    rows = [
        {"name": n, "mean": audit[k][0], "low": audit[k][1], "high": audit[k][2]} for n, k in names
    ]
    title = _title(
        text="Removing the confirmed unlogged events leaves the pooled interval wider than 0.03",
        line=(
            "Gated mask minus no mask (points of effective capacity; lower is better). Dot: "
            "estimate. Line: 95% interval from resampling whole calendar months. Orange rules: "
            "the pre-registered limit of +/-0.03. Real rows outside logged transfers, six Primary "
            "substations, April 2024 to June 2026."
        ),
    )
    _dots(
        data=pl.DataFrame(rows),
        title=title,
        height=210,
        domain=[-0.1, 0.9],
        x_title="Gated mask minus no mask (points; lower is better)",
        limits=(-0.03, 0.03),
        colour=ocf.DATA_GREEN,
    ).save(OUT_DIR / "figure_2_part1_audit.svg")


def figure_3() -> None:
    """Share of found transfers whose onset estimate lies within a given number of half-hours."""
    events = rep.read_all(pattern="events_world_*.parquet", folder=rep.P2)
    grid = np.arange(0, 49)
    rows = []
    for family, label in (("PAIR", "Pair alarm"), ("RONC", "RO-NC")):
        sub = events.filter(pl.col("family") == family)
        for col, ref, name in (
            ("known", "true_onset", "Alarm time"),
            ("onset", "true_onset", "Detector's step stamp"),
            ("located", "true_onset", "Located onset"),
        ):
            err = (sub[col] - sub[ref]).abs().to_numpy()
            rows.extend(
                {
                    "family": label,
                    "estimate": name,
                    "within": int(g),
                    "share": float(np.mean(err <= g)),
                }
                for g in grid
            )
    data = pl.DataFrame(rows)
    chart = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X(
                "within:Q",
                title="Within this many half-hours of the true onset",
                axis=alt.Axis(values=[0, 12, 24, 36, 48]),
            ),
            y=alt.Y("share:Q", title="Share of found transfers", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "estimate:N",
                scale=alt.Scale(
                    domain=["Alarm time", "Detector's step stamp", "Located onset"],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            strokeDash=alt.StrokeDash("estimate:N", legend=None),
        )
        .properties(width=WIDTH, height=150)
        .facet(row=alt.Row("family:N", title=None))
        .properties(
            title=_title(
                text="The detector's step stamp is already close to the true onset",
                line=(
                    "Share of found injected transfers whose estimate of the onset lies within the "
                    "given number of half-hours (12 half-hours is the 6-hour spacing of forecast "
                    "origins). Higher is better. 44 worlds, six Primary substations."
                ),
            )
        )
    )
    chart.save(OUT_DIR / "figure_3_location_error.svg")


def main() -> None:
    """Draw the figures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads((WORK_DIR / "report_p2.json").read_text())
    audit = json.loads((WORK_DIR / "report_audit.json").read_text())
    figure_1(report=report)
    figure_2(audit=audit)
    figure_3()


if __name__ == "__main__":
    main()
