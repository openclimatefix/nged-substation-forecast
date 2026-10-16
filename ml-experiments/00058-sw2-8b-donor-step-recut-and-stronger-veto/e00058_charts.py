"""Draw experiment 58's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The series are Primary substations, not NGED generators.

    uv run python ml-experiments/00058-*/e00058_charts.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00058_common import WORK_DIR
from e00058_detect import VETO_ZS
from e00058_report import population, results
from e00058_run import vz_tag

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00058-sw2-8b-donor-step-recut-and-stronger-veto"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 80
STEP_EDGES: Final[list[float]] = [0.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 1000.0]
STEP_LABELS: Final[list[str]] = ["0-2", "2-3", "3-4", "4-5", "5-7", "7-10", "10+"]
METERED: Final[str] = "Metered receiver"
UNMETERED: Final[str] = "No metered receiver"
THRESHOLD: Final[float] = 4.0


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(OUT_DIR / name)
    print("wrote", name)


def _rows(*, veto_z: float = 1.5) -> pl.DataFrame:
    rows = population(data=results(out=WORK_DIR, veto_z=veto_z), setting="D0M_48")
    step = rows["largest_step_z"].to_numpy()
    return rows.with_columns(
        step_bin=pl.Series(
            [STEP_LABELS[int(np.searchsorted(STEP_EDGES, s, side="right")) - 1] for s in step]
        ),
        group=pl.when(pl.col("metered_receivers") > 0)
        .then(pl.lit(METERED))
        .otherwise(pl.lit(UNMETERED)),
    )


def figure_recall_by_step() -> alt.TopLevelMixin:
    """Share found by step size and group, D0M, 24-hour window."""
    table = (
        _rows()
        .group_by("step_bin", "group")
        .agg(found=pl.col("found").mean(), n=pl.len())
        .with_columns(label=pl.col("n").cast(pl.String))
    )
    colour = alt.Color(
        "group:N",
        title=None,
        scale=alt.Scale(domain=[METERED, UNMETERED], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
        legend=alt.Legend(orient="top"),
    )
    base = alt.Chart(table).encode(
        x=alt.X(
            "step_bin:N",
            sort=STEP_LABELS,
            title="Largest step of the transfer (noise units of the 24-hour statistic)",
            axis=alt.Axis(labelAngle=0),
        ),
        y=alt.Y("found:Q", title="Share of transfers found", scale=alt.Scale(domain=[0, 1])),
        color=colour,
        xOffset="group:N",
    )
    bars = base.mark_bar()
    text = base.mark_text(dy=-6, fontSize=10).encode(text="label:N")  # ty: ignore[unresolved-attribute]
    return (bars + text).properties(
        width=WIDTH,
        height=240,
        title=_title(
            text="The larger the step, the more transfers are found, with or without a receiver",
            lines=[
                (
                    "Detector D0 at its threshold of 4.0 noise units, 24-hour window; 620 injected "
                    "of the receiving load, over 1 day). Larger is better."
                    "transfers. Larger is better."
                )
            ],
        ),
    )


def figure_step_distribution() -> alt.TopLevelMixin:
    """Histogram of the largest step by group, with the alarm threshold."""
    table = _rows().group_by("step_bin", "group").agg(n=pl.len())
    bars = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X(
                "step_bin:N",
                sort=STEP_LABELS,
                title="Largest step (noise units)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("n:Q", title="Injected transfers"),
            color=alt.Color(
                "group:N",
                title=None,
                scale=alt.Scale(
                    domain=[METERED, UNMETERED], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
                ),
                legend=alt.Legend(orient="top"),
            ),
            xOffset="group:N",
        )
    )
    return bars.properties(
        width=WIDTH,
        height=200,
        title=_title(
            text="Most transfers without a metered receiver step below the threshold",
            lines=[
                (
                    "The detector alarms at 4.0 noise units, so the first three groups of bars "
                    "(under 4) are mostly below it. Same 620 transfers as the previous figure."
                )
            ],
        ),
    )


def figure_veto_strength() -> alt.TopLevelMixin:
    """Share found against veto strength, odd and even months."""
    rows = []
    for veto_z in VETO_ZS:
        data = results(out=WORK_DIR, veto_z=veto_z).filter(
            pl.col("headline"), pl.col("setting") == "D2_48", pl.col("set") != "holiday"
        )
        for parity, mask in (
            ("Odd months (tuning)", ~pl.col("even")),
            ("Even months (scoring)", pl.col("even")),
        ):
            rows.append(
                {
                    "veto": float(vz_tag(veto_z)),
                    "months": parity,
                    "found": float(data.filter(mask)["found"].mean()),  # ty: ignore[invalid-argument-type]
                }
            )
    table = pl.DataFrame(rows)
    return (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "veto:Q",
                title="Veto strength (noise units; smaller is stronger)",
                axis=alt.Axis(values=[1.0, 1.25, 1.5], format=".2f"),
            ),
            y=alt.Y("found:Q", title="Share of transfers found", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "months:N",
                title=None,
                scale=alt.Scale(
                    domain=["Odd months (tuning)", "Even months (scoring)"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=_title(
                text="Strengthening the veto from 1.5 to 1.0 noise units changes little",
                lines=[
                    (
                        "D2 at its matched 24-hour threshold, headline-band transfers (25% or more "
                        "of the receiving load, over 1 day). Larger is better."
                    )
                ],
            ),
        )
    )


def main() -> None:
    """Write the three figures."""
    _save(figure_recall_by_step(), name="figure_1_recall_by_step.svg")
    _save(figure_step_distribution(), name="figure_2_step_distribution.svg")
    _save(figure_veto_strength(), name="figure_3_veto_strength.svg")


if __name__ == "__main__":
    main()
