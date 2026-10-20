"""Draw the figures of experiment 115's write-up from `work/results.json`, `work/gate.json`, and the
stored frames.

Every figure carries its own scope, unit, and direction in its title and subtitle. SVGs go to the
write-up's folder, PNGs to a scratch folder for the render check. Substations appear by D-label with
dates; no series of embedded generation is drawn.

    uv run python ml-experiments/00115-*/e00115_charts.py [--only name ...]
"""

import argparse
import json
import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import altair as alt
import e00115_common as common
import e00115_tune as tune
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 680
PLOT_WIDTH: Final[int] = 560
WRAP: Final[int] = 100
GREY: Final[str] = "#808080"
DARK: Final[str] = "#333333"
ARM_LABELS: Final[dict[str, str]] = {
    "K0": "K0 (GAM-K as experiment 111)",
    "KW": "KW (K0 plus shape within hour bands)",
    "KW-lb": "KW-lb (KW, noise settings per lead band)",
    "K0-lb": "K0-lb",
    "KW-day": "KW-day (no readings from the day of the forecast)",
    "R": "R (14-day profile recipe)",
    "KW-fast": "KW-fast (post hoc: shape step 16 times larger)",
    "GAM-0": "GAM-0 (DP, no correction)",
    "X-nl+K": "X-nl+K (lagless XGBoost plus filter)",
    "Xlag-d": "Xlag-d (lagged XGBoost)",
    "BG2": "BG2 (production recipe)",
    "climatology": "Climatology",
    "manual_heuristic": "Manual heuristic",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "K0": ocf.DATA_BLUE,
    "KW": ocf.BRAND_ORANGE,
    "KW-lb": ocf.BRAND_ORANGE_LIGHT,
    "R": ocf.DATA_PURPLE,
    "KW-day": ocf.DATA_SKY,
    "KW-fast": ocf.DATA_GREEN,
    "GAM-0": ocf.DATA_SKY,
    "X-nl+K": ocf.DATA_BLUE_LIGHT,
    "Xlag-d": ocf.DATA_GREEN,
    "BG2": DARK,
    "climatology": "#C8C8C8",
    "manual_heuristic": GREY,
}
SAMPLE_WEEKS: Final[tuple[datetime, ...]] = tuple(
    datetime(y, m, d, tzinfo=UTC)
    for y, m, d in (
        (2025, 7, 14),
        (2025, 8, 11),
        (2025, 10, 13),
        (2025, 11, 10),
        (2026, 1, 12),
        (2026, 2, 9),
        (2026, 4, 13),
        (2026, 5, 11),
    )
)
"""Eight fixed weeks of the scored year, two per season, fixed by calendar before any score of the
arms was read: the Mondays nearest the middle and the start of each season's second month."""
D1: Final[tuple[float, float]] = (15.5, 39.0)


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 80),
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name, flush=True)


def results() -> dict:
    """Return the stored results."""
    return json.loads((common.WORK_DIR / "results.json").read_text())


def d1_frame(*, arm: str, site: str) -> pl.DataFrame:
    """Return an arm's D+1 rows at a site (all periods), with the readings and no switching rows."""
    frame = pl.read_parquet(tune.frame_path(arm=arm, site=site))
    return frame.filter(
        (pl.col("lead_hours") >= D1[0])
        & (pl.col("lead_hours") <= D1[1])
        & ~pl.col("switching_mask")
    )


def half_hour_of(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the local half-hour of day (0 to 47) of each row's valid time."""
    cache = common.load_cache()
    position = np.searchsorted(
        cache.end_times, frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    )
    return cache.calendar.half_hour[np.minimum(position, len(cache.end_times) - 1)]


# --- figures ---


def figure_residual_heat() -> alt.TopLevelMixin:
    """GAM-K's day-ahead error by local half-hour and month, one panel per decision substation."""
    panels = []
    for site in common.DECISION_SITES:
        frame = d1_frame(arm="K0", site=site)
        frame = frame.filter(pl.col("period").is_in(["tuning", "replicate", "scored"]))
        h = half_hour_of(frame=frame)
        table = (
            frame.with_columns(
                hour=pl.Series(h) / 2.0,
                month=pl.col("valid_time").dt.strftime("%Y-%m"),
                error=pl.col("q50") - pl.col("actual"),
            )
            .group_by("hour", "month")
            .agg(error=pl.col("error").mean())
        )
        panels.append(
            alt.Chart(table)
            .mark_rect()
            .encode(
                x=alt.X(
                    "hour:O",
                    title="Local hour of day" if site == common.DECISION_SITES[-1] else None,
                    axis=alt.Axis(values=[0, 6, 12, 18, 23.5], labelAngle=0),
                ),
                y=alt.Y(
                    "month:O",
                    title=None,
                    axis=alt.Axis(
                        labelFontSize=8, values=["2024-07", "2025-01", "2025-07", "2026-01"]
                    ),
                ),
                color=alt.Color(
                    "error:Q",
                    scale=alt.Scale(
                        domain=[-4, 0, 4], range=[ocf.DATA_BLUE, "#FFFFFF", ocf.BRAND_ORANGE]
                    ),
                    legend=alt.Legend(
                        title="Median forecast minus reading (points)",
                        orient="top",
                        direction="horizontal",
                    )
                    if site == common.DECISION_SITES[0]
                    else None,
                ),
            )
            .properties(width=PLOT_WIDTH, height=95, title=site)
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="GAM-K's day-ahead error depends on the half-hour of the day, and the pattern changes by month",
            subtitle=(
                "Mean of the median forecast minus the reading for day D+1 (leads 15.5 to 39 hours), "
                "by local half-hour and calendar month, rows outside the switching events, five "
                "decision substations, tuning, replicate, and scored months. Orange: forecast too "
                "high. Blue: forecast too low. Unit: points of effective capacity."
            ),
        )
    )


def _week_panel(
    *, site: str, start: datetime, label: str, frames: dict[str, pl.DataFrame]
) -> alt.TopLevelMixin:
    end = start + timedelta(days=7)
    grid = pl.DataFrame(
        {
            "valid_time": pl.datetime_range(
                start + timedelta(minutes=30), end, "30m", time_zone="UTC", eager=True
            )
        }
    )
    pieces = []
    for arm, frame in frames.items():
        week = frame.filter((pl.col("valid_time") >= start) & (pl.col("valid_time") <= end))
        full = grid.join(
            week.select("valid_time", "q05", "q50", "q95", "actual"), on="valid_time", how="left"
        )
        pieces.append(full.with_columns(arm=pl.lit(arm)))
    table = pl.concat(pieces).with_columns(time=pl.col("valid_time").dt.replace_time_zone(None))
    colour = alt.Color(
        "arm:N",
        scale=alt.Scale(domain=["K0", "KW"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
        legend=alt.Legend(title=None, orient="top", direction="horizontal"),
    )
    x = alt.X(
        "time:T", title=None, axis=alt.Axis(format="%a %d %b", labelFontSize=8, tickCount="day")
    )
    band = (
        alt.Chart(table)
        .mark_area(opacity=0.18)
        .encode(x=x, y=alt.Y("q05:Q", title="Points"), y2="q95:Q", color=colour)
    )
    median = alt.Chart(table).mark_line(strokeWidth=1.2).encode(x=x, y="q50:Q", color=colour)
    actual = (
        alt.Chart(table.filter(pl.col("arm") == "K0"))
        .mark_line(color=DARK, strokeWidth=1.2, strokeDash=[2, 2])
        .encode(x=x, y="actual:Q")
    )
    return (band + median + actual).properties(
        width=PLOT_WIDTH, height=95, title=f"{site}: {label}"
    )


def figure_weeks(*, site: str) -> alt.TopLevelMixin:
    """Fixed sample weeks of the scored year at one substation: K0 and KW day-ahead forecasts."""
    frames = {arm: d1_frame(arm=arm, site=site) for arm in ("K0", "KW")}
    kw = frames["KW"].filter(pl.col("period") == "scored")
    daily = (
        kw.with_columns(
            day=(
                pl.col("valid_time") - pl.duration(days=pl.col("valid_time").dt.weekday() - 1)
            ).dt.truncate("1d")
        )
        .group_by("day")
        .agg(mean_shape=pl.col("shape").abs().mean(), rows=pl.len())
        .filter(pl.col("rows") > 250)
        .filter(~pl.col("day").is_in(list(SAMPLE_WEEKS)))
        .sort("mean_shape", descending=True)
    )
    moved = daily["day"][0].replace(tzinfo=UTC)
    panels = [
        _week_panel(site=site, start=start, label=f"week of {start:%d %b %Y}", frames=frames)
        for start in SAMPLE_WEEKS
    ]
    panels.append(
        _week_panel(
            site=site,
            start=moved,
            label=f"week of {moved:%d %b %Y}, where the shape state is largest",
            frames=frames,
        )
    )
    return alt.vconcat(*panels).properties(
        title=_title(
            text=f"{site}: day-ahead forecasts of K0 and KW against the readings, nine sample weeks",
            subtitle=(
                "Each panel holds the seven UTC days after the date shown, each day forecast at 09:00 UTC "
                "the day before (leads 15.5 to 39 hours). Dashed black: reading. Lines: median forecast. "
                "Bands: 5% to 95% quantiles. Eight weeks were fixed by calendar before any score was read; "
                "the ninth is the scored-year week with the largest mean absolute shape, chosen without "
                "looking at scores. Unit: points of effective capacity."
            ),
        )
    )


def _interval_rows(*, block: dict, band: str) -> list[dict]:
    rows = []
    for period, entries in block.items():
        for label, entry in entries.items():
            if band not in entry["bands"]:
                continue
            q = entry["bands"][band]["qs"]
            rows.append(
                {
                    "contrast": label,
                    "period": period,
                    "estimate": q["estimate"],
                    "low": q["lower"],
                    "high": q["upper"],
                    "sites": f"{q['sites_better']} of {q['n_sites']} sites better",
                }
            )
    return rows


def figure_headline(*, data: dict) -> alt.TopLevelMixin:
    """The four planned contrasts at day D+1, both periods, with 98.75% intervals."""
    rows = _interval_rows(block=data["planned"], band="D+1")
    table = pl.DataFrame(rows).with_columns(
        period=pl.col("period").replace(
            {
                "replicate": "Replicate months (2024-25)",
                "scored": "Scored year (2025-26)",
            }
        ),
        label_x=pl.max_horizontal("estimate", "high"),
    )
    order = [r["contrast"] for r in rows if r["period"] == "scored"]
    base = alt.Chart(table).encode(
        y=alt.Y("contrast:N", sort=order, title=None, axis=alt.Axis(labelLimit=280)),
        yOffset="period:N",
        color=alt.Color(
            "period:N",
            scale=alt.Scale(
                domain=table["period"].unique().sort().to_list(),
                range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
            ),
            legend=alt.Legend(title=None, orient="top", direction="vertical"),
        ),
    )
    rule = base.mark_rule(strokeWidth=2).encode(x="low:Q", x2="high:Q")
    dot = base.mark_point(filled=True, size=60).encode(
        x=alt.X(
            "estimate:Q",
            title="Difference in quantile score (points; below zero favours the first-named arm)",
        )
    )
    text = base.mark_text(align="left", dx=6, fontSize=9, color=DARK).encode(
        x="label_x:Q", text="sites:N"
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    return (zero + rule + dot + text).properties(
        width=PLOT_WIDTH - 130,
        height=190,
        title=_title(
            text="The shape state lowers the day-ahead score in both periods; the recipe does about as well",
            subtitle=(
                "Dot: difference in mean quantile score at day D+1 (leads 15.5 to 39 hours) over the five "
                "decision substations, rows outside the switching events. Line: 98.75% interval from "
                "resampling whole calendar months. Dashed line: no difference. All weather is ECMWF "
                "ensemble forecasts. K0 is GAM-K as experiment 111."
            ),
        ),
    )


def figure_by_lead(*, data: dict) -> alt.TopLevelMixin:
    """Absolute quantile score by lead band for the W1 arms and the references (scored year)."""
    absolute = data["references"]["scored"]["absolute"]
    bands = ["0-24 h", "24-48 h", "48-144 h", "144-168 h"]
    rows = [
        {"arm": ARM_LABELS.get(arm, arm), "key": arm, "band": band, "score": value[band]}
        for arm, value in absolute.items()
        for band in bands
    ]
    table = pl.DataFrame(rows)
    domain = [ARM_LABELS.get(a, a) for a in absolute]
    colours = [ARM_COLOURS.get(a, GREY) for a in absolute]
    return (
        alt.Chart(table)
        .mark_line(point=True, strokeWidth=1.8)
        .encode(
            x=alt.X(
                "band:O",
                sort=bands,
                title="Lead band (hours after the 09:00 UTC origin)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "score:Q",
                title="Mean quantile score (points; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=domain, range=colours),
                legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=300),
            ),
        )
        .properties(
            width=PLOT_WIDTH,
            height=260,
            title=_title(
                text="The filter arms track one another across lead bands; XGBoost with lags and BG2 sit above them",
                subtitle=(
                    "Mean quantile score by lead band, scored year (2025-07-02 to 2026-06-29), five decision "
                    "substations, rows outside the switching events, on the rows every arm holds. Unit: points "
                    "of effective capacity."
                ),
            ),
        )
    )


def figure_gate(*, gate: dict) -> alt.TopLevelMixin:
    """Gate test 1: the expected and measured recovery of each injected profile change."""
    rows = []
    for key, label in (
        ("test_1_recovery", "KW (tuned)"),
        ("post_hoc_test_1_kw_fast", "KW-fast (post hoc)"),
    ):
        if key not in gate:
            continue
        for shape_name, entry in gate[key]["shapes"].items():
            pretty = (
                "a: +3 points, evening"
                if shape_name.startswith("a")
                else "b: morning peak moved earlier"
            )
            for who, value in (
                (f"{label}, expected", entry["expected_median"]),
                (f"{label}, measured", entry["kw_median"]),
                *(
                    (("K0, measured", entry["k0_median"]), ("R, measured", entry["r_median"]))
                    if key == "test_1_recovery"
                    else ()
                ),
            ):
                rows.append({"reading": who, "shape": pretty, "recovery": value})
    table = pl.DataFrame(rows).unique(maintain_order=True)
    order = table["reading"].unique(maintain_order=True).to_list()
    bars = (
        alt.Chart()
        .mark_bar()
        .encode(
            y=alt.Y("reading:N", sort=order, title=None, axis=alt.Axis(labelLimit=240)),
            x=alt.X(
                "recovery:Q",
                title="Share of the injected change in the day-ahead correction, days 8 to 14",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "reading:N",
                legend=None,
                scale=alt.Scale(
                    domain=order,
                    range=[
                        ocf.BRAND_ORANGE_LIGHT,
                        ocf.BRAND_ORANGE,
                        ocf.DATA_BLUE,
                        ocf.DATA_PURPLE,
                        ocf.DATA_GREEN,
                        ocf.DATA_SKY,
                    ][: len(order)]
                    + [GREY] * 6,
                ),
            ),
        )
        .properties(width=PLOT_WIDTH - 120, height=110)
    )
    floor = alt.Chart().mark_rule(color=DARK, strokeDash=[4, 3]).encode(x=alt.datum(0.4))
    return (
        alt.layer(bars, floor)
        .facet(
            data=table,
            row=alt.Row(
                "shape:N",
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=11),
            ),
        )
        .properties(
            title=_title(
                text="The tuned shape state recovers a third of an injected change in two weeks, as its own formula predicts",
                subtitle=(
                    "Median over five decision substations and two start dates of the share of an injected change "
                    "(+3 points from 17:00 to 20:00 local, or a morning peak moved half an hour to an hour "
                    "earlier) that the day-ahead correction carries on days 8 to 14. Dashed line: the 0.4 "
                    "floor the plan set. 'Expected' is the formula's answer on a complete stream."
                ),
            )
        )
    )


def figure_tails(*, data: dict) -> alt.TopLevelMixin:
    """Coverage gap at q01, q05, q95, and q99 for each arm (scored year, D+1 and all leads)."""
    rows = []
    for arm, bands in data["tails"]["scored"].items():
        for band, entries in bands.items():
            if band not in {"D+1", "0-24 h", "24-48 h", "48-144 h", "144-168 h"}:
                continue
            for entry in entries:
                rows.append(
                    {
                        "arm": ARM_LABELS.get(arm, arm)
                        if arm != "KW-shift"
                        else "KW-shift (K0 spread, KW mean)",
                        "band": band,
                        "level": f"q{round(entry['level'] * 100):02d}",
                        "gap": 100 * entry["gap"],
                    }
                )
    table = pl.DataFrame(rows)
    return (
        alt.Chart(table)
        .mark_circle(size=60)
        .encode(
            x=alt.X(
                "level:N",
                title="Quantile level",
                sort=["q01", "q05", "q95", "q99"],
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("gap:Q", title="Coverage minus nominal level (percentage points)"),
            color=alt.Color(
                "arm:N", legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=300)
            ),
            xOffset="arm:N",
        )
        .properties(width=PLOT_WIDTH - 40, height=110)
        .facet(
            row=alt.Row(
                "band:N",
                sort=["D+1", "0-24 h", "24-48 h", "48-144 h", "144-168 h"],
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left"),
            )
        )
        .properties(
            title=_title(
                text="The tails stay close to nominal for every arm",
                subtitle=(
                    "Share of readings at or below each forecast quantile, minus the quantile's level, scored "
                    "year, five decision substations. Zero is exact. Positive: the quantile sits too high."
                ),
            )
        )
    )


def figure_gain_by_hour_and_month() -> alt.TopLevelMixin:
    """The shape state's D+1 gain (KW minus K0 quantile score) by hour band and month."""
    from e00115_score import quantile_score

    pieces = []
    for site in common.DECISION_SITES:
        k0 = d1_frame(arm="K0", site=site).filter(pl.col("row_period") == "scored")
        kw = d1_frame(arm="KW", site=site).filter(pl.col("row_period") == "scored")
        keys = ["power_fcst_init_time", "valid_time"]
        joined = (
            k0.select([*keys, "hour_band", pl.col("actual")])
            .with_columns(k0=quantile_score(frame=k0))
            .join(
                kw.select(keys).with_columns(kw=quantile_score(frame=kw)),
                on=keys,
            )
        )
        pieces.append(joined.with_columns(site=pl.lit(site)))
    table = (
        pl.concat(pieces)
        .with_columns(
            month=pl.col("valid_time").dt.strftime("%Y-%m"),
            band=pl.col("hour_band").replace_strict(
                {0: "night", 1: "morning", 2: "midday", 3: "evening"}, return_dtype=pl.String
            ),
            gain=pl.col("kw") - pl.col("k0"),
        )
        .group_by("month", "band")
        .agg(gain=pl.col("gain").mean())
    )
    return (
        alt.Chart(table)
        .mark_rect()
        .encode(
            x=alt.X("month:O", title=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("band:N", sort=["night", "morning", "midday", "evening"], title=None),
            color=alt.Color(
                "gain:Q",
                scale=alt.Scale(
                    domain=[-0.3, 0, 0.3], range=[ocf.BRAND_ORANGE, "#FFFFFF", ocf.DATA_BLUE]
                ),
                legend=alt.Legend(
                    title="KW minus K0 quantile score (points; below zero favours KW)",
                    orient="top",
                    direction="horizontal",
                ),
            ),
        )
        .properties(
            width=PLOT_WIDTH,
            height=130,
            title=_title(
                text="Where the shape state helps: hour band by month",
                subtitle=(
                    "Mean difference in quantile score at day D+1, scored year, five decision substations. "
                    "Orange: KW scores better than K0. Blue: worse."
                ),
            ),
        )
    )


def figure_card(*, site: str) -> alt.TopLevelMixin:
    """One substation's card: the shape state over the scored year, and weekday profiles at D+1."""
    kw = d1_frame(arm="KW", site=site).filter(pl.col("row_period") == "scored")
    k0 = d1_frame(arm="K0", site=site).filter(pl.col("row_period") == "scored")
    h = half_hour_of(frame=kw)
    heat = (
        kw.with_columns(
            hour=pl.Series(h) / 2.0,
            week=pl.col("valid_time").dt.truncate("1w").dt.replace_time_zone(None),
        )
        .group_by("hour", "week")
        .agg(shape=pl.col("shape").mean())
    )
    heat_chart = (
        alt.Chart(heat)
        .mark_rect()
        .encode(
            x=alt.X("week:T", title=None, axis=alt.Axis(format="%b %Y", labelFontSize=8)),
            y=alt.Y(
                "hour:O",
                sort="descending",
                title="Local hour",
                axis=alt.Axis(values=[0, 6, 12, 18, 23.5], labelAngle=0),
            ),
            color=alt.Color(
                "shape:Q",
                scale=alt.Scale(
                    domain=[-2, 0, 2], range=[ocf.DATA_BLUE, "#FFFFFF", ocf.BRAND_ORANGE]
                ),
                legend=alt.Legend(
                    title="Shape state added to the forecast (points)",
                    orient="top",
                    direction="horizontal",
                ),
            ),
        )
        .properties(width=PLOT_WIDTH, height=130, title="Shape state by week and local half-hour")
    )
    wk = kw.with_columns(hour=pl.Series(h) / 2.0)
    k0_h = half_hour_of(frame=k0)
    wk0 = k0.with_columns(hour=pl.Series(k0_h) / 2.0)
    day_kw = half_hour_day_type(frame=kw)
    day_k0 = half_hour_day_type(frame=k0)
    profile = pl.concat(
        [
            wk.filter(pl.Series(day_kw == 0))
            .group_by("hour")
            .agg(value=pl.col("mean").mean())
            .with_columns(series=pl.lit("Forecast with shape (KW)")),
            wk0.filter(pl.Series(day_k0 == 0))
            .group_by("hour")
            .agg(value=pl.col("mean").mean())
            .with_columns(series=pl.lit("Forecast without shape (K0)")),
            wk.filter(pl.Series(day_kw == 0))
            .group_by("hour")
            .agg(value=pl.col("actual").mean())
            .with_columns(series=pl.lit("Reading")),
        ]
    )
    profile_chart = (
        alt.Chart(profile)
        .mark_line(strokeWidth=1.6)
        .encode(
            x=alt.X("hour:Q", title="Local hour of day", axis=alt.Axis(values=[0, 6, 12, 18, 24])),
            y=alt.Y("value:Q", title="Points", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=["Reading", "Forecast without shape (K0)", "Forecast with shape (KW)"],
                    range=[DARK, ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=PLOT_WIDTH, height=130, title="Working-day profile at day D+1, scored year"
        )
    )
    return alt.vconcat(heat_chart, profile_chart, cold_snap_panel(site=site)).properties(
        title=_title(
            text=f"{site}: what the shape state adds",
            subtitle=(
                "Provisional: fitted to past weather, forecast on ENS, screening run. KW arm, scored year "
                "(2025-07-02 to 2026-06-29), day D+1, rows outside the switching events. Unit: points of "
                "effective capacity."
            ),
        )
    )


def cold_snap_panel(*, site: str) -> alt.TopLevelMixin:
    """Mean day-ahead quantile score of K0, KW, and R on the site's cold-snap days (scored year)."""
    from e00115_report import COLD_SNAP_C, MIN_COLD_DAYS, cold_snap_mask
    from e00115_score import quantile_score

    rows = []
    days = 0
    for arm in ("K0", "KW", "R"):
        frame = d1_frame(arm=arm, site=site).filter(pl.col("row_period") == "scored")
        mask = cold_snap_mask(frame=frame, site=site)
        cold = frame.filter(pl.Series(mask))
        if arm == "K0":
            days = cold["valid_time"].dt.date().n_unique()
        rows.append(
            {"arm": arm, "score": float(quantile_score(frame=cold).mean()) if cold.height else None}
        )
    few = days < MIN_COLD_DAYS
    title = [
        f"How it does in a cold snap: {days} days with a daily mean below {COLD_SNAP_C:g} °C",
        "(November to March)" + (", too few days to read" if few else ""),
    ]
    chart = (
        alt.Chart(pl.DataFrame(rows))
        .mark_bar(opacity=0.45 if few else 1.0)
        .encode(
            y=alt.Y("arm:N", sort=["K0", "KW", "R"], title=None),
            x=alt.X("score:Q", title="Mean quantile score at D+1 (points; smaller is better)"),
            color=alt.Color(
                "arm:N",
                legend=None,
                scale=alt.Scale(
                    domain=["K0", "KW", "R"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE],
                ),
            ),
        )
        .properties(width=PLOT_WIDTH, height=70, title=title)
    )
    return chart


def half_hour_day_type(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the day type (0 for a working day) of each row's valid time."""
    cache = common.load_cache()
    position = np.searchsorted(
        cache.end_times, frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    )
    return cache.calendar.day_type[np.minimum(position, len(cache.end_times) - 1)]


FIGURES: Final[tuple[str, ...]] = (
    "residual_heat",
    "weeks",
    "headline",
    "by_lead",
    "gain",
    "gate",
    "tails",
    "cards",
)


def main() -> None:
    """Draw the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=list(FIGURES))
    args = parser.parse_args()
    data = results()
    gate = json.loads((common.WORK_DIR / "gate.json").read_text())
    if "residual_heat" in args.only:
        _save(figure_residual_heat(), name="figure_01_residual_heat")
    if "weeks" in args.only:
        for number, site in enumerate(common.DECISION_SITES):
            _save(figure_weeks(site=site), name=f"figure_02{'abcde'[number]}_weeks_{site}")
    if "headline" in args.only:
        _save(figure_headline(data=data), name="figure_03_headline")
    if "by_lead" in args.only:
        _save(figure_by_lead(data=data), name="figure_04_by_lead")
    if "gain" in args.only:
        _save(figure_gain_by_hour_and_month(), name="figure_05_gain")
    if "gate" in args.only:
        _save(figure_gate(gate=gate), name="figure_06_gate")
    if "tails" in args.only:
        _save(figure_tails(data=data), name="figure_07_tails")
    if "cards" in args.only:
        for number, site in enumerate(common.DECISION_SITES):
            _save(figure_card(site=site), name=f"figure_08{'abcde'[number]}_card_{site}")


if __name__ == "__main__":
    main()
