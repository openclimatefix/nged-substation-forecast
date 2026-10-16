"""Draw experiment 00044's figures as SVG: raw power around the events first, then the shares.

The series are Primary substations; no NGED generator is drawn. Afterwards optimise each SVG with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00044-sw2-2-switching-share-of-best-guess-score/e00044_charts.py
"""

import os
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00044_score as score
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00044-sw2-2-switching-share-of-best-guess-score"
)
ARMS_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00033-best-guess-architecture-vs-production"
    / "work"
    / "arms"
    / "n1_cs.parquet"
)
PREVIEW_DIR: Final[Path | None] = (
    Path(os.environ["E44_PREVIEW_DIR"]) if "E44_PREVIEW_DIR" in os.environ else None
)
"""Set `E44_PREVIEW_DIR` to also write PNG previews for checking the render."""
WIDTH: Final[int] = 640
WRAP: Final[int] = 100
LABELS: Final[dict[str, str]] = {
    "a_before": "Target in event; onset after the forecast",
    "a_after": "Target in event; onset before the forecast",
    "b": "Target outside events; lagged power inside one",
    "flagged": "Any of the three",
}
SETS: Final[dict[str, str]] = {
    "trustworthy": "Trustworthy band",
    "all_visible": "All visible events",
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _results() -> pl.DataFrame:
    return pl.read_parquet(score.WORK_DIR / "results.parquet")


def figure_1_raw() -> alt.TopLevelMixin:
    """Actual power and the best guess's band around each trustworthy event in the scored year."""
    cases = score.load_cases()
    events = score.load_events(series=cases.series).filter(pl.col("trustworthy"))
    first, last = cases.frame["valid_time"].min(), cases.frame["valid_time"].max()
    events = (
        events.filter((pl.col("end") >= first) & (pl.col("onset") <= last))
        .sort("event_id", "time_series_id")
        .group_by("event_id", maintain_order=True)
        .first()
    )
    arms = pl.read_parquet(ARMS_PATH).filter(pl.col("seed") == 0)
    panels = []
    for number, row in enumerate(events.iter_rows(named=True), start=1):
        start = row["onset"] - np.timedelta64(2, "D")
        stop = row["end"] + np.timedelta64(2, "D")
        part = (
            arms.filter(
                (pl.col("time_series_id") == row["time_series_id"])
                & (pl.col("valid_time") >= start)
                & (pl.col("valid_time") <= stop)
            )
            .with_columns(lead=pl.col("valid_time") - pl.col("power_fcst_init_time"))
            .sort("lead")
            .group_by("valid_time")
            .first()
            .sort("valid_time")
            .drop("lead")
            .with_columns(
                days=(pl.col("valid_time") - row["onset"]).dt.total_minutes() / 1440.0,
                panel=pl.lit(f"Event {number}: series {row['time_series_id']}"),
                end_days=(row["end"] - row["onset"]).total_seconds() / 86400.0,
            )
        )
        panels.append(part)
    frame = pl.concat(panels)
    base = alt.Chart(frame).properties(width=WIDTH, height=110)
    band = base.mark_area(opacity=0.25, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("days:Q", title="Days from the event's onset"),
        y=alt.Y("q05:Q", title="Power (fraction of capacity)"),
        y2="q95:Q",
    )
    median = base.mark_line(color=ocf.DATA_BLUE, strokeWidth=1).encode(x="days:Q", y="q50:Q")  # ty: ignore[unresolved-attribute]
    actual = base.mark_line(color=ocf.BRAND_ORANGE, strokeWidth=1.2).encode(  # ty: ignore[unresolved-attribute]
        x="days:Q", y="actual:Q"
    )
    onset_rule = base.mark_rule(color="black", strokeDash=[4, 3]).encode(x=alt.datum(0))  # ty: ignore[unresolved-attribute]
    end_rule = base.mark_rule(color="black", strokeDash=[4, 3]).encode(x="end_days:Q")  # ty: ignore[unresolved-attribute]
    return (
        alt.layer(band, median, actual, onset_rule, end_rule, data=frame)
        .facet(
            row=alt.Row("panel:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left"))
        )
        .properties(
            title=_title(
                text="Power and the best guess around each trustworthy event",
                lines=[
                    (
                        "Orange: metered power. Blue line and band: the best guess's median and "
                        "5% to 95% band, from the run with the shortest lead for each half-hour. "
                        "Dashed lines: the labelled onset and end. Primary substations, "
                        "July 2025 to June 2026."
                    )
                ],
            )
        )
    )


def _interval_chart(
    *,
    frame: pl.DataFrame,
    categories: list[str],
    groups: dict[str, str],
    x_title: str,
    facet: str | None,
    zero: bool,
    height: int,
) -> alt.TopLevelMixin:
    """Dots with 95% intervals, one row per category, one colour per group, optional facet rows."""
    y = alt.Y("category:N", title=None, sort=categories, axis=alt.Axis(labelLimit=330))
    colour = alt.Color(
        "group:N",
        scale=alt.Scale(domain=list(groups.values()), range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
        legend=alt.Legend(orient="top", title=None),
    )
    interval = (
        alt.Chart()
        .mark_rule()
        .encode(  # ty: ignore[unresolved-attribute]
            y=y,
            x=alt.X("lower_95:Q", title=x_title),
            x2="upper_95:Q",
            color=colour,
            yOffset="group:N",
        )
    )
    dots = (
        alt.Chart().mark_circle(size=70).encode(y=y, x="point:Q", color=colour, yOffset="group:N")  # ty: ignore[unresolved-attribute]
    )
    layers = [interval, dots]
    if zero:
        layers.append(alt.Chart().mark_rule(color="grey", strokeDash=[3, 3]).encode(x=alt.datum(0)))  # ty: ignore[unresolved-attribute]
    layered = alt.layer(*layers).properties(width=WIDTH, height=height)
    if facet is None:
        return layered.properties(data=frame)
    return layered.facet(
        row=alt.Row(
            f"{facet}:N",
            title=None,
            header=alt.Header(labelAngle=0, labelAlign="left", labelLimit=200),
        ),
        data=frame,
    ).resolve_scale(x="independent")


def figure_2_shares() -> alt.TopLevelMixin:
    """Share of half-hours, of the quantile score, and of the tail score, per category."""
    results = _results()
    rows = []
    for event_set, set_label in SETS.items():
        part = results.filter(pl.col("event_set") == event_set)
        for category, label in LABELS.items():
            for quantity, name in (
                (f"half_hour_share:{category}", "Share of half-hours"),
                (f"score_share:qs:{category}", "Share of quantile score"),
                (f"score_share:tw_p99:{category}", "Share of tail score (twCRPS at P99)"),
            ):
                hit = part.filter(pl.col("statistic") == quantity).to_dicts()[0]
                rows.append({"category": label, "measure": name, "group": set_label, **hit})
    chart = _interval_chart(
        frame=pl.DataFrame(rows),
        categories=list(LABELS.values()),
        groups=SETS,
        x_title=(
            "Percent of all Primary cases "
            "(dot: estimate; line: 95% interval from resampling months)"
        ),
        facet="measure",
        zero=False,
        height=110,
    )
    return chart.properties(
        title=_title(
            text="Flagged half-hours are few, and they carry a larger share of the quantile score",
            lines=[
                (
                    "Trustworthy band: moved load of 25% or more of the receiving load, lasting "
                    "over 1 day. The best guess is N1 with the 14-day correction and the spread "
                    "factor. Primary substations, July 2025 to June 2026."
                )
            ],
        )
    )


def figure_3_excess() -> alt.TopLevelMixin:
    """Mean quantile score inside each category against outside every flag."""
    results = _results()
    rows = []
    for event_set, set_label in SETS.items():
        part = results.filter(pl.col("event_set") == event_set)
        for category, label in LABELS.items():
            hit = part.filter(pl.col("statistic") == f"mean_loss:qs:{category}").to_dicts()[0]
            rows.append({"group": set_label, "category": label, **hit})
        unflagged = part.filter(pl.col("statistic") == "mean_loss:qs:unflagged").to_dicts()[0]
        rows.append({"group": set_label, "category": "Outside every flag", **unflagged})
    chart = _interval_chart(
        frame=pl.DataFrame(rows),
        categories=[*LABELS.values(), "Outside every flag"],
        groups=SETS,
        x_title="Mean quantile score (points of capacity; smaller is better)",
        facet=None,
        zero=False,
        height=200,
    )
    return chart.properties(
        title=_title(
            text="Flagged half-hours score worse than the rest",
            lines=[
                (
                    "Dot: mean quantile score of the cases in the category. Line: 95% interval "
                    "from resampling whole months. Primary substations, July 2025 to June 2026."
                )
            ],
        )
    )


def figure_4_bound() -> alt.TopLevelMixin:
    """The perfect-detector bound on each score, by view and event set."""
    results = _results()
    views = {
        "metered": "Metered target: events after the forecast, and lagged power",
        "nra": "Normal running arrangement: lagged power only",
        "ceiling": "Ceiling: also events no detector can anticipate",
    }
    rows = []
    for event_set, set_label in SETS.items():
        for loss, loss_label in (
            ("qs", "Quantile score"),
            ("tw_p99", "Tail score (twCRPS at P99)"),
        ):
            for view, view_label in views.items():
                hit = results.filter(
                    (pl.col("event_set") == event_set)
                    & (pl.col("statistic") == f"bound:{loss}:{view}:series_mean")
                ).to_dicts()[0]
                rows.append({"group": set_label, "loss": loss_label, "category": view_label, **hit})
    chart = _interval_chart(
        frame=pl.DataFrame(rows),
        categories=list(views.values()),
        groups=SETS,
        x_title="Fall in the summed score, percent (a perfect detector; upper bound)",
        facet="loss",
        zero=True,
        height=110,
    )
    return chart.properties(
        title=_title(
            text=(
                "A perfect detector would lower the quantile score by little "
                "in the trustworthy band"
            ),
            lines=[
                (
                    "Each counted case is given its series' mean score outside every flag. "
                    "Dot: estimate. Line: 95% interval from resampling whole months. "
                    "Dashed line: no change. Positive means a lower score."
                )
            ],
        )
    )


def figure_5_series() -> alt.TopLevelMixin:
    """Each series' share of half-hours and of the quantile score that is flagged."""
    table = pl.read_parquet(score.WORK_DIR / "per_series.parquet").filter(
        pl.col("event_set") == "trustworthy"
    )
    frame = table.unpivot(
        index="time_series_id",
        on=["half_hour_share", "qs_share"],
        variable_name="measure",
        value_name="percent",
    ).with_columns(
        measure=pl.col("measure").replace(
            {"half_hour_share": "Share of half-hours", "qs_share": "Share of quantile score"}
        ),
        series=pl.col("time_series_id").cast(pl.String),
    )
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("series:N", title="Primary substation series", sort=None),
            xOffset="measure:N",
            y=alt.Y("percent:Q", title="Percent of the series' cases"),
            color=alt.Color(
                "measure:N",
                scale=alt.Scale(
                    domain=["Share of half-hours", "Share of quantile score"],
                    range=[ocf.DATA_BLUE_LIGHT, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=_title(
                text="Which series carry the flagged half-hours",
                lines=[
                    (
                        "Flagged means the target or the lagged power is inside a "
                        "trustworthy-band event. Point estimates only; a series with no events "
                        "shows no bar."
                    )
                ],
            ),
        )
    )


def figure_6_lead() -> alt.TopLevelMixin:
    """Mean quantile score by lead, flagged against unflagged, trustworthy and all visible."""
    cases = score.load_cases()
    events = score.load_events(series=cases.series)
    rows = []
    for event_set, label in SETS.items():
        flags = score.flag_cases(cases=cases, events=events.filter(pl.col(event_set)))
        frame = cases.frame.with_columns(
            flagged=pl.Series(flags["flagged"]),
            lead=(pl.col("lead_hours") // 24).cast(pl.Int64),
        )
        rows.append(
            frame.group_by("lead", "flagged")
            .agg(mean_qs=pl.col("qs").mean(), n=pl.len())
            .with_columns(event_set=pl.lit(label))
        )
    frame = pl.concat(rows).with_columns(
        group=pl.when(pl.col("flagged")).then(pl.lit("Flagged")).otherwise(pl.lit("Not flagged"))
    )
    chart = (
        alt.Chart(frame.filter(pl.col("n") >= 30))
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lead:Q",
                title="Lead time (whole days from the forecast origin)",
                axis=alt.Axis(tickMinStep=1),
            ),
            y=alt.Y("mean_qs:Q", title="Mean quantile score"),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(
                    domain=["Flagged", "Not flagged"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(width=WIDTH, height=130)
    )
    return chart.facet(row=alt.Row("event_set:N", title=None)).properties(
        title=_title(
            text="The excess error of flagged half-hours at every lead",
            lines=[
                (
                    "Mean quantile score in points of capacity; smaller is better. "
                    "Days with fewer than 30 cases are not drawn."
                )
            ],
        )
    )


def main() -> None:
    """Write every figure."""
    OUT.mkdir(parents=True, exist_ok=True)
    figures = {
        "figure_1_raw_events": figure_1_raw,
        "figure_2_shares": figure_2_shares,
        "figure_3_mean_score": figure_3_excess,
        "figure_4_bound": figure_4_bound,
        "figure_5_series": figure_5_series,
        "figure_6_lead": figure_6_lead,
    }
    for name, make in figures.items():
        chart = make()
        chart.save(OUT / f"{name}.svg")
        if PREVIEW_DIR is not None:
            chart.save(PREVIEW_DIR / f"{name}.png", scale_factor=1.3)
        print("wrote", name)


if __name__ == "__main__":
    main()
