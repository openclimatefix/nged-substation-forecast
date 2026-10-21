"""Draw experiment 148's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The series are Primary substations, not NGED generators.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00148-*/e00148_charts.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00148_common import WORK_DIR
from e00148_contrasts import populations

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00148-switching-end-rule-lookahead-ro-nc"
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
    """Histogram of the end error in hours for the lookaheads, onset-found transfers."""
    keep = ["L0", "L6", "L24", "L48"]
    data = (
        _base_events(results=results)
        .filter(pl.col("onset_found"), pl.col("setting").is_in(keep))
        .with_columns(hours=pl.col("end_error").clip(-48.0, 48.0))
        .drop_nulls("hours")
    )
    return (
        alt.Chart(data)
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("hours:Q", bin=alt.Bin(step=2), title="End error in hours (late is positive)"),
            y=alt.Y("count():Q", title="Transfers"),
            row=alt.Row("setting:N", title="Lookahead", sort=keep),
        )
        .properties(
            width=WIDTH,
            height=70,
            title=_title(
                text="A lookahead of 6 hours or more puts the end on the true end",
                line="Called end minus true end, shapes S2 to S4, even months, transfers whose "
                "onset is found. Errors are clipped at 48 hours.",
            ),
        )
    )


def figure_share_found(*, summary: pl.DataFrame) -> alt.TopLevelMixin:
    """MW-weighted share found by rule, with the 95% interval."""
    order = ["L0", "L1", "L2", "L3", "L4", "L6", "L24", "L48", "R2", "R0", "H14"]
    data = summary.filter(pl.col("rule").is_in(order))
    bars = (
        alt.Chart(data)
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("rule:N", title=None, sort=order),
            x=alt.X("share:Q", title="Share of moved MW found", scale=alt.Scale(domain=[0, 1])),
        )
    )
    ticks = (
        alt.Chart(data).mark_rule().encode(y=alt.Y("rule:N", sort=order), x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    )
    return (bars + ticks).properties(
        width=WIDTH,
        height=230,
        title=_title(
            text="Waiting 6 hours after the candidate end finds as much as waiting 2 days",
            line="MW-weighted share found with 95% interval, base transfers, shapes S2 to S4, "
            "even months. L<h> is a lookahead of h hours; R0 is the offline rule.",
        ),
    )


def figure_false_end_and_lag(*, stats: pl.DataFrame) -> alt.TopLevelMixin:
    """False-end share and median call lag by lookahead."""
    data = stats.filter(pl.col("rule").str.starts_with("L")).with_columns(
        hours=pl.col("rule").str.slice(1).cast(pl.Float64)
    )
    base = alt.Chart(data).encode(x=alt.X("hours:Q", title="Lookahead in hours"))
    false_end = base.mark_line(point=True, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("false_end:Q", title="Share of false ends")
    )
    lag = base.mark_line(point=True, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("lag_median_h:Q", title="Median call lag in hours")
    )
    return alt.hconcat(
        false_end.properties(width=WIDTH // 2 - 40, height=170),
        lag.properties(width=WIDTH // 2 - 40, height=170),
    ).properties(
        title=_title(
            text="False ends fall below 0.10 at 6 hours, and the call lag is 24 hours plus the "
            "lookahead",
            line="Base transfers, shapes S2 to S4, even months. The 24 hours are the step "
            "statistic's own window.",
        )
    )


def main() -> None:
    """Write the three figures."""
    import json

    run_dir = WORK_DIR / "run"
    events = _base_events(results=populations())
    summary = json.loads((run_dir / "chart_summary.json").read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_end_error(results=events).save(OUT_DIR / "end_error.svg")
    figure_share_found(summary=pl.DataFrame(summary["share"])).save(OUT_DIR / "share_found.svg")
    figure_false_end_and_lag(stats=pl.DataFrame(summary["stats"]).rename({"setting": "rule"})).save(
        OUT_DIR / "false_end_and_lag.svg"
    )


if __name__ == "__main__":
    main()
