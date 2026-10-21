"""Draw experiment 146's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The series are Primary substations, not NGED generators.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00146-*/e00146_charts.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00146_common import WORK_DIR

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00146-ro-nc-arm-b-on-the-35-real-transfers-over-1-day"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 80
REASON_ORDER: Final[list[str]] = [
    "found",
    "alarm at the start, end too far off",
    "first step of 4 noise units or more, no alarm",
    "first step of 2 to 4 noise units",
    "first step under 2 noise units",
    "no residual at the start",
]


def _title(*, text: str, line: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=text,
        subtitle=textwrap.wrap(line, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def figure_reasons(*, table: pl.DataFrame) -> alt.TopLevelMixin:
    """Moved MW by outcome of the transfer."""
    grouped = table.group_by("reason").agg(mw=pl.col("moved_mw").sum(), n=pl.len())
    return (
        alt.Chart(grouped)
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("reason:N", sort=REASON_ORDER, title=None, axis=alt.Axis(labelLimit=320)),
            x=alt.X("mw:Q", title="Moved MW (labelled median), summed over transfers"),
            tooltip=["reason", "n", "mw"],
        )
        .properties(
            width=WIDTH,
            height=150,
            title=_title(
                text="18 of 35 transfers carry 28 of the 43 moved MW",
                line="Labelled real transfers lasting over 1 day, by what RO-NC did. Each "
                "transfer is counted once, under the first reason that holds.",
            ),
        )
    )


def figure_step(*, table: pl.DataFrame) -> alt.TopLevelMixin:
    """First step in noise units against moved MW, coloured by found."""
    data = table.drop_nulls("first_step_z").with_columns(
        outcome=pl.when(pl.col("found")).then(pl.lit("found")).otherwise(pl.lit("missed"))
    )
    return (
        alt.Chart(data)
        .mark_circle(size=90, opacity=0.85)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "first_step_z:Q",
                scale=alt.Scale(type="log", domain=[1.5, 50]),
                title="First 24-hour step at the start, in noise units (log scale)",
            ),
            y=alt.Y("moved_mw:Q", title="Moved MW (labelled median)"),
            color=alt.Color(
                "outcome:N",
                scale=alt.Scale(
                    domain=["found", "missed"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
                ),
                title=None,
            ),
        )
        .properties(
            width=WIDTH,
            height=240,
            title=_title(
                text="Misses cluster at steps of 2 to 6 noise units",
                line="Each dot is one real transfer lasting over 1 day. The detector alarms "
                "from 4 noise units, but 8 transfers with a first step of 4 or more got no "
                "alarm within 2 hours of the labelled start.",
            ),
        )
    )


def main() -> None:
    """Draw both figures."""
    table = pl.read_parquet(WORK_DIR / "per_transfer.parquet")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_reasons(table=table).save(OUT_DIR / "figure_1_moved_mw_by_outcome.svg")
    figure_step(table=table).save(OUT_DIR / "figure_2_first_step_and_outcome.svg")
    print("wrote 2 figures")


if __name__ == "__main__":
    main()
