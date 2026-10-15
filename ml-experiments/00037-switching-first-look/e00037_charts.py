"""Draw experiment 00037's figures from the saved tables, as SVG.

The raw-data figures come before any detector figure: the residual of the donor and receiver on
one time axis with the logged intervals shaded.

    uv run python ml-experiments/00037-switching-first-look/e00037_charts.py

The SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.
"""

import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00037_common import WORK_DIR
from e00037_grid import GRID_LENGTH, GRID_START
from e00037_oracle import logged_cover

CHART_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / "00037-switching-first-look"
)
WIDTH: Final[int] = 600
PANEL_HEIGHT: Final[int] = 105
WRAP_CHARACTERS: Final[int] = 90
GREY: Final[str] = "#999999"
SERIES_COLOURS: Final[tuple[str, str]] = (ocf.DATA_BLUE, ocf.BRAND_ORANGE)


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP_CHARACTERS)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _stamp_time(*, stamp: int) -> datetime:
    return GRID_START + timedelta(minutes=30 * int(stamp))


def event_panel(
    *, event: dict, panel: pl.DataFrame, label: str | list[str], alarms: pl.DataFrame | None = None
) -> alt.TopLevelMixin:
    """One event: the residual of each side in a window, the logged interval shaded."""
    first, last = event["first_stamp"], event["last_stamp_exclusive"]
    pad = 72 if last - first <= 144 else 144
    lo, hi = max(first - pad, 0), min(max(last, first + 48) + pad, GRID_LENGTH - 1)
    if last - first > 600:
        hi = min(first + 600, GRID_LENGTH - 1)
    sides = [int(s) for s in (event["series_a"], event["series_b"]) if s is not None]
    frame = panel.filter(pl.col("series").is_in(sides), pl.col("stamp").is_between(lo, hi))
    sides = [s for s in sides if s in set(frame["series"].to_list())]
    frame = frame.with_columns(
        series_label=pl.format("series {}", pl.col("series")),
        time=pl.col("valid_time").dt.replace_time_zone(None),
    )
    start, end = (
        _stamp_time(stamp=first).replace(tzinfo=None),
        _stamp_time(stamp=min(last, hi)).replace(tzinfo=None),
    )
    span = (
        alt.Chart(  # ty: ignore[unresolved-attribute]
            pl.DataFrame({"a": [start], "b": [end]}).to_pandas()
            if False
            else alt.InlineData(values=[{"a": start.isoformat(), "b": end.isoformat()}])
        )
        .mark_rect(color=GREY, opacity=0.25)
        .encode(x="a:T", x2="b:T")
    )
    lines = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1)
        .encode(
            x=alt.X("time:T", title=None),
            y=alt.Y("res_mw:Q", title="Residual (MW)"),
            color=alt.Color(
                "series_label:N",
                scale=alt.Scale(
                    domain=[f"series {x}" for x in sides], range=list(SERIES_COLOURS)[: len(sides)]
                ),
                legend=None,
            ),
        )
    )
    layers = [span, lines]
    if alarms is not None:
        marks = alarms.filter(pl.col("series").is_in(sides), pl.col("stamp").is_between(lo, hi))
        if marks.height:
            layers.append(
                alt.Chart(  # ty: ignore[unresolved-attribute]
                    marks.with_columns(
                        time=pl.col("stamp").map_elements(
                            lambda s: _stamp_time(stamp=s).replace(tzinfo=None),
                            return_dtype=pl.Datetime,
                        )
                    )
                )
                .mark_rule(color="black", strokeDash=[3, 3])
                .encode(x="time:T")
            )
    return alt.layer(*layers).properties(
        width=WIDTH,
        height=PANEL_HEIGHT,
        title=alt.TitleParams(
            text=label if isinstance(label, list) else [label],
            fontSize=11,
            anchor="start",
            fontWeight="normal",
        ),
    )


def _label(*, row: dict) -> list[str]:
    step = " and ".join(
        f"{row[f'step_{t}']:+.2f} MW" for t in ("a", "b") if row[f"step_{t}"] is not None
    )
    plotted = [row[f"series_{t}"] for t in ("a", "b") if row[f"step_{t}"] is not None]
    colours = " and ".join(
        f"series {s} ({c})" for s, c in zip(plotted, ("blue", "orange"), strict=False)
    )
    return [
        (
            f"Sheet {row['sheet']}, closed {row['start_utc']:%Y-%m-%d %H:%M} UTC "
            f"for {row['hours']:.1f} h"
        ),
        f"Steps {step}; {colours}",
    ]


def raw_figures(*, real: pl.DataFrame, panel: pl.DataFrame) -> None:
    """Figures 2 and 3: raw residuals around confirmed, missed, invisible, and quiet cases."""
    validation = real.filter(pl.col("split") == "validation")
    best = validation.filter("confirmed", pl.col("kind") == "two_sided").sort(
        "transfer_mw", descending=True
    )
    panels = [
        event_panel(event=r, panel=panel, label=_label(row=r)) for r in best.iter_rows(named=True)
    ]
    _save(
        alt.vconcat(*panels)  # ty: ignore[invalid-argument-type]
        .resolve_scale(color="independent")
        .properties(
            title=_title(
                text="Six validation closures where load visibly moved between the two substations",
                lines=[
                    (
                        "Residual of the demand forecast without power lags (observed minus "
                        "expected). "
                        "Grey band: the logged closure. The two lines step in opposite directions."
                    )
                ],
            )
        ),
        "figure_2_raw_confirmed.svg",
    )
    hits = pl.read_parquet(WORK_DIR / "event_hits.parquet").filter(
        (pl.col("grid") == "pre_registered")
        & (pl.col("detector") == "D1")
        & (pl.col("window") == 8)
    )
    missed = hits.filter(~pl.col("hit")).sort("transfer_mw", descending=True).head(4)
    alarm_table = pl.read_parquet(WORK_DIR / "alarms_D1_8_4.0.parquet")
    cases = [(r, "Missed: ") for r in missed.iter_rows(named=True)]
    invisible = (
        validation.filter(
            ~pl.col("confirmed"), pl.col("kind") == "two_sided", pl.col("hours").is_between(20, 400)
        )
        .sort("hours", descending=True)
        .unique(subset=["operation"], keep="first", maintain_order=True)
        .head(3)
    )
    cases += [(r, "Invisible: ") for r in invisible.iter_rows(named=True)]
    panels = [
        event_panel(
            event=r,
            panel=panel,
            label=[prefix + _label(row=r)[0], _label(row=r)[1]],
            alarms=alarm_table,
        )
        for r, prefix in cases
    ]
    _save(
        alt.vconcat(*panels)  # ty: ignore[invalid-argument-type]
        .resolve_scale(color="independent")
        .properties(
            title=_title(
                text="Missed and invisible closures, with the neighbour-balance detector's alarms",
                lines=[
                    (
                        "Dashed lines: alarms of the neighbour-balance detector (4-hour window, "
                        "threshold 4). The first four panels are confirmed transfers it missed; "
                        "the "
                        "last three are logged closures with no step in either residual."
                    )
                ],
            )
        ),
        "figure_3_raw_missed_and_invisible.svg",
    )


def quiet_figure(*, intervals: pl.DataFrame, panel: pl.DataFrame) -> None:
    """Figure 4: four-day windows of a connected pair with no logged closure at all."""
    rng = np.random.default_rng(7)
    panels = []
    for pair in ((15, 17), (9, 24), (11, 25)):
        free = np.ones(GRID_LENGTH, dtype=bool)
        for sid in pair:
            free &= logged_cover(intervals=intervals, series=sid) == 0
        starts = [s for s in range(5000, GRID_LENGTH - 400) if free[s : s + 192].all()]
        if not starts:
            continue
        first = int(rng.choice(starts))
        event = {
            "first_stamp": first + 48,
            "last_stamp_exclusive": first + 49,
            "series_a": pair[0],
            "series_b": pair[1],
        }
        label = [
            (
                f"Series {pair[0]} (blue) and {pair[1]} (orange), "
                f"from {_stamp_time(stamp=first):%Y-%m-%d}"
            ),
            "No logged closure on any sheet touching either",
        ]
        panels.append(event_panel(event=event, panel=panel, label=label))
    _save(
        alt.vconcat(*panels)  # ty: ignore[invalid-argument-type]
        .resolve_scale(color="independent")
        .properties(
            title=_title(
                text="Quiet periods: the same residual with no logged switching",
                lines=["Four days of connected pairs with no closure on any sheet touching them."],
            )
        ),
        "figure_4_raw_quiet.svg",
    )


def _save(chart: alt.TopLevelMixin, name: str) -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(CHART_DIR / name)


def headline() -> None:
    """Figure 1: validation recall against unexplained alarms, with the budget and intervals."""
    curve = pl.read_parquet(WORK_DIR / "curve.parquet").filter(pl.col("window") == 8)
    curve = curve.filter(pl.col("pre_registered"))
    chosen = pl.read_parquet(WORK_DIR / "chosen.parquet").filter(
        pl.col("window") == 8, pl.col("grid") == "pre_registered"
    )
    names = {"D0": "Single-series detector (D0)", "D1": "Neighbour-balance detector (D1)"}
    colours = alt.Scale(domain=list(names.values()), range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE])
    curve = curve.with_columns(detector_label=pl.col("detector").replace_strict(names))
    chosen = chosen.with_columns(detector_label=pl.col("detector").replace_strict(names))
    line = (
        alt.Chart(curve)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "rate_validation:Q",
                scale=alt.Scale(type="log"),
                title="Unexplained alarms per series-month (validation, log scale)",
            ),
            y=alt.Y(
                "recall_weighted:Q", title="Size-weighted recall", scale=alt.Scale(domain=[0, 0.8])
            ),
            color=alt.Color(
                "detector_label:N",
                scale=colours,
                legend=alt.Legend(orient="top", title=None, labelLimit=400),
            ),
        )
    )
    bars = (
        alt.Chart(chosen)  # ty: ignore[unresolved-attribute]
        .mark_errorbar(thickness=2)
        .encode(
            x="rate_validation:Q",
            y=alt.Y("recall_lo:Q", title=None),
            y2="recall_hi:Q",
            color=alt.Color("detector_label:N", scale=colours, legend=None),
        )
    )
    budget = (
        alt.Chart(pl.DataFrame({"x": [0.5]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 4], color="black")
        .encode(x="x:Q")
    )
    placebo = (
        alt.Chart(chosen)  # ty: ignore[unresolved-attribute]
        .mark_point(shape="cross", size=80, color=GREY)
        .encode(x="rate_validation:Q", y="placebo_weighted_mean:Q")
    )
    _save(
        alt.layer(line, bars, budget, placebo).properties(
            width=WIDTH,
            height=280,
            title=_title(
                text="D1 finds more large transfers than D0 at a given alarm rate",
                lines=[
                    (
                        "18 validation transfers (even months), 4-hour window. Points: "
                        "thresholds on "
                        "the grid. Bars: 95% interval at the threshold chosen on odd months. "
                        "Dashed "
                        "line: the 0.5 alarm budget. Grey crosses: recall on shifted logs."
                    )
                ],
            ),
        ),
        "figure_1_headline.svg",
    )


def conservation(*, real: pl.DataFrame, placebo: pl.DataFrame) -> None:
    """Figure 5: donor against receiver step for two-sided closures, real and placebo."""
    both = real.filter(pl.col("kind") == "two_sided").with_columns(
        source=pl.lit("Logged closures"), conf=pl.col("confirmed")
    )
    shifted = placebo.filter(pl.col("kind") == "two_sided").with_columns(
        source=pl.lit("Shifted logs (five seeds)"), conf=pl.col("confirmed")
    )
    data = (
        pl.concat([both, shifted], how="diagonal")
        .select("step_a", "step_b", "source", "conf", "has_mva")
        .with_columns(
            units=pl.when(pl.col("has_mva"))
            .then(pl.lit("Pair includes an MVA series"))
            .otherwise(pl.lit("MW series only"))
        )
    )
    points = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_circle(size=40, opacity=0.6)
        .encode(
            x=alt.X("step_a:Q", title="Step on the first substation of the sheet (MW)"),
            y=alt.Y("step_b:Q", title="Step on the second substation (MW)"),
            color=alt.Color(
                "units:N",
                scale=alt.Scale(range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
                legend=alt.Legend(orient="top", title=None, labelLimit=400),
            ),
            shape=alt.Shape("conf:N", legend=alt.Legend(orient="top", title="Confirmed")),
        )
        .properties(width=WIDTH // 2 - 20, height=WIDTH // 2 - 20)
        .facet(column=alt.Column("source:N", title=None))
    )
    _save(
        points.properties(
            title=_title(
                text="Where load moves, the two steps are opposite and similar in size",
                lines=[
                    (
                        "Each point is one logged closure with telemetry on both sides. "
                        "Conserved load lies "
                        "on the line where the second step equals minus the first."
                    )
                ],
            )
        ),
        "figure_5_conservation.svg",
    )


def tail_and_bands() -> None:
    """Figures 6 and 7: recall by size band, and the stack's score shares."""
    chosen = pl.read_parquet(WORK_DIR / "chosen.parquet").filter(
        pl.col("window") == 8, pl.col("grid") == "pre_registered"
    )
    rows = []
    for r in chosen.iter_rows(named=True):
        rows.extend(
            {"detector": r["detector"], "band": band, "recall": r[f"recall_band_{band}"]}
            for band in (">=25%", "10-25%")
        )
    bands = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X(
                "band:N",
                title="Transfer size as a share of the receiving substation's load",
                sort=["10-25%", ">=25%"],
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("recall:Q", title="Recall (unweighted)"),
            xOffset="detector:N",
            color=alt.Color(
                "detector:N",
                scale=alt.Scale(domain=["D0", "D1"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=_title(
                text="Recall by transfer size",
                lines=["Validation, 4-hour window, threshold from the 0.5 budget."],
            ),
        )
    )
    _save(bands, "figure_6_recall_by_size.svg")
    tail = (
        pl.read_parquet(WORK_DIR / "tail_shares.parquet")
        .select("class", cases="share_cases", quantile_score="share_qs", tw_p99="share_tw")
        .unpivot(index="class")
    )
    shares = (
        alt.Chart(tail)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("variable:N", title=None, sort=["cases", "quantile_score", "tw_p99"]),
            x=alt.X("value:Q", title="Share of total", stack="normalize"),
            color=alt.Color(
                "class:N",
                scale=alt.Scale(
                    domain=[
                        "confirmed_one_sided",
                        "confirmed_two_sided",
                        "logged_unconfirmed",
                        "other",
                    ],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_PURPLE, ocf.DATA_SKY, GREY],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=WIDTH,
            height=120,
            title=_title(
                text="Confirmed-transfer half-hours hold 40% of the threshold-weighted score",
                lines=[
                    (
                        "Stack forecast, July 2025 to 2026-03-25. Nearly all of the orange share "
                        "is two "
                        "substations in one 370-hour closure."
                    )
                ],
            ),
        )
    )
    _save(shares, "figure_7_tail_shares.svg")


def step_variation() -> None:
    """Figure 8: relative step in the first and second half of long events."""
    data = pl.read_parquet(WORK_DIR / "step_variation.parquet")
    line = (
        alt.Chart(pl.DataFrame({"x": [-1.1, 0.4], "y": [-1.1, 0.4]}))  # ty: ignore[unresolved-attribute]
        .mark_line(color=GREY)
        .encode(x="x:Q", y="y:Q")
    )
    points = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_circle(size=50, color=ocf.DATA_BLUE)
        .encode(
            x=alt.X("rel_first:Q", title="Step in the first half, as a share of expected power"),
            y=alt.Y("rel_second:Q", title="Second half"),
        )
    )
    _save(
        alt.layer(line, points).properties(
            width=WIDTH // 2,
            height=WIDTH // 2,
            title=_title(
                text="The step usually keeps its sign but changes size within an event",
                lines=["Confirmed closures of 6 hours or more, larger-step side. Diagonal: equal."],
            ),
        ),
        "figure_8_step_variation.svg",
    )


def main() -> None:
    """Draw all figures."""
    real = pl.read_parquet(WORK_DIR / "oracle_real.parquet").filter("primary")
    placebo = pl.read_parquet(WORK_DIR / "oracle_placebo.parquet").filter("primary")
    panel = pl.read_parquet(WORK_DIR / "panel.parquet")
    intervals = pl.read_parquet(WORK_DIR / "intervals.parquet")
    headline()
    raw_figures(real=real, panel=panel)
    quiet_figure(intervals=intervals, panel=panel)
    conservation(real=real, placebo=placebo)
    tail_and_bands()
    step_variation()


if __name__ == "__main__":
    main()
