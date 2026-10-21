"""Draw experiment 147's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The series are Primary substations, not NGED generators.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00147-*/e00147_charts.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00147_common import WORK_DIR
from e00147_contrasts import populations

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00147-causal-live-end-rule-for-ro-nc"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 80
SHAPES: Final[list[str]] = ["S2", "S3", "S4"]


def _title(*, text: str, line: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=text,
        subtitle=textwrap.wrap(line, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _base_events(*, results: pl.DataFrame) -> pl.DataFrame:
    return results.filter(
        pl.col("population") == "base",
        pl.col("parity") == "even",
        pl.col("shape").is_in(SHAPES),
    )


def figure_end_error(*, results: pl.DataFrame) -> alt.TopLevelMixin:
    """Histogram of the end error in hours for the four rules, onset-found transfers."""
    data = (
        _base_events(results=results)
        .filter(pl.col("onset_found"), pl.col("setting").is_in(["R0", "R1", "R2", "R3"]))
        .with_columns(hours=pl.col("end_error").clip(-48.0, 48.0))
        .drop_nulls("hours")
    )
    return (
        alt.Chart(data)
        .mark_bar(opacity=0.8)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("hours:Q", bin=alt.Bin(step=2), title="End error in hours (late is positive)"),
            y=alt.Y("count():Q", title="Injected transfers whose onset is found"),
            color=alt.Color("setting:N", title="Rule"),
            row=alt.Row("setting:N", title=None),
        )
        .properties(
            width=WIDTH,
            height=70,
            title=_title(
                text="Ending at the first return step calls the end 8 hours early",
                line="Called end minus true end, shapes S2 to S4, even months. Errors are clipped "
                "at 48 hours.",
            ),
        )
    )


def figure_share_found(*, summary: pl.DataFrame) -> alt.TopLevelMixin:
    """MW-weighted share found by rule, with the 95% interval."""
    return (
        alt.Chart(summary)
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("rule:N", title=None, sort=["R0", "R2", "R3", "R1"]),
            x=alt.X("share:Q", title="Share of moved MW found", scale=alt.Scale(domain=[0, 1])),
            tooltip=["rule", "share", "low", "high"],
        )
        .properties(
            width=WIDTH,
            height=110,
            title=_title(
                text="The running-peak rule loses no recall; the two threshold rules lose half",
                line="MW-weighted share found, base transfers, shapes S2 to S4, even months.",
            ),
        )
    )


def figure_merge(*, merges: pl.DataFrame) -> alt.TopLevelMixin:
    """Share of transfers with 2 or more intervals, by merge tolerance."""
    return (
        alt.Chart(merges)
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("tolerance:N", title="Merge tolerance in half-hours", sort=None),
            y=alt.Y("share_2p:Q", title="Share with 2 or more intervals"),
            tooltip=["tolerance", "share_2p", "share_found"],
        )
        .properties(
            width=WIDTH,
            height=170,
            title=_title(
                text="Merging overlapping alarms removes nearly all duplicate intervals",
                line="Running-peak rule R2, base transfers, shapes S2 to S4, even months.",
            ),
        )
    )


def main() -> None:
    """Write the three figures."""
    import json

    run_dir = WORK_DIR / "run"
    events = _base_events(results=populations())
    summary_path = run_dir / "chart_summary.json"
    summary = json.loads(summary_path.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_end_error(results=events).save(OUT_DIR / "end_error.svg")
    figure_share_found(summary=pl.DataFrame(summary["share"])).save(OUT_DIR / "share_found.svg")
    figure_merge(merges=pl.DataFrame(summary["merge"])).save(OUT_DIR / "merge.svg")


if __name__ == "__main__":
    main()
