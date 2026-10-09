"""Draw the embedded-battery forecast page's figures from the saved report tables and fits.

Nothing is refitted here. Every number a figure shares with the page comes from the tables that
`forecast_report.py` and `census_report.py` saved. The testbed batteries are public Balancing
Mechanism Units, so their figures name them and give megawatts on calendar dates. NGED battery A
appears only as a fraction of its own 99th-percentile absolute output, on days numbered 1 to 7.

Run, after the reports: `uv run python studies/embedded_battery_forecast/forecast_charts.py`.
Optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline` before committing.
"""

from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Final, cast

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from forecast_fit import FITS_DIRS
from studies.battery_market import WEEK_START, market_frame
from studies.charts import CONTENT_WIDTH_PX, figure
from studies.sources import EMBEDDED_BATTERY_FORECAST_DIR

ASSETS_DIR: Final[Path] = Path("docs/studies/assets")
TABLES: Final[Path] = EMBEDDED_BATTERY_FORECAST_DIR / "report_tables"
IDLE_TABLES: Final[Path] = EMBEDDED_BATTERY_FORECAST_DIR / "report_tables_idle_dropped"
FITS: Final[Path] = FITS_DIRS["as_written"]
LABEL_PX: Final[int] = 290
PLOT_PX: Final[int] = CONTENT_WIDTH_PX - LABEL_PX - 20
WIDE_PX: Final[int] = CONTENT_WIDTH_PX - 90
ROW_PX: Final[int] = 30
EXAMPLE_BATTERY: Final[str] = "E_ARBRB-1"
"""The public testbed battery whose week is drawn: the first in the sorted testbed."""
NGED_WEEK_START: Final[str] = "2026-03-02"
"""The Monday that starts NGED battery A's drawn week, fixed by a rule and not by eye. The week
is shown as days 1 to 7."""
SETTING_COLOURS: Final[dict[str, str]] = {
    "Primary setting": ocf.BRAND_ORANGE,
    "Second setting": ocf.DATA_BLUE,
}
ARM_LABELS: Final[dict[str, str]] = {
    "clim": "Climatology (56-day trailing)",
    "persistence_conformal": "Persistence plus past errors",
    "rank_conformal__price_actual": "Rank rule, actual price",
    "rank_conformal__price_naive": "Rank rule, price a week earlier",
    "rank_conformal__price_model": "Rank rule, forecast price",
    "rank_conformal__price_shuffled": "Rank rule, another day's price",
    "rank_conformal__no_neighbour": "Rank rule, actual price",
    "xgb_quantile__price_actual": "XGBoost, actual price",
    "xgb_quantile__price_naive": "XGBoost, price a week earlier",
    "xgb_quantile__price_model": "XGBoost, forecast price",
    "xgb_quantile__price_shuffled": "XGBoost, another day's price",
    "xgb_quantile__no_neighbour": "XGBoost, no Physical Notification",
    "xgb_quantile__own_fpn": "XGBoost, own notification",
    "xgb_quantile__neighbour_fpn": "XGBoost, others' notifications",
    "xgb_quantile__neighbour_fpn_shuffled": "XGBoost, others', another day",
}
KIND_COLOURS: Final[dict[str, str]] = {
    "Climatology": ocf.BLACK_1,
    "Simple rules": ocf.DATA_SKY,
    "XGBoost quantile model": ocf.BRAND_ORANGE,
}


def _kind(arm: str) -> str:
    if arm == "clim":
        return "Climatology"
    if arm.startswith("xgb"):
        return "XGBoost quantile model"
    return "Simple rules"


def _interval_panel(
    *,
    rows: pl.DataFrame,
    x_title: str,
    colour_title: str,
    palette: Mapping[str, str],
    zero_label: str | None,
    panel_title: str,
    x_domain: tuple[float, float] | None = None,
    shapes: dict[str, str] | None = None,
) -> alt.LayerChart:
    """Draw dots with 95% interval lines, one labelled row per `label`, coloured by `group`.

    Args:
        rows: Columns `label`, `group`, `value`, `lower`, `upper`, and `order`.
        x_title: The x axis title, naming the quantity, unit, and which direction is better.
        colour_title: The colour key's title.
        palette: Colour by `group`.
        zero_label: Text beside a zero rule, or None for no rule.
        panel_title: The panel's title.
        x_domain: The x axis range, or None to fit the data.
        shapes: Point shape by `group`, as a second cue beside the colour.

    Returns:
        The panel.
    """
    labels = rows.sort("order")["label"].unique(maintain_order=True).to_list()
    height = ROW_PX * len(labels) + 40
    scale = alt.Scale(domain=list(x_domain)) if x_domain else alt.Scale(zero=False)
    base = alt.Chart(rows, title=alt.TitleParams(panel_title, anchor="start", fontSize=12))
    y = alt.Y(
        "label:N",
        sort=labels,
        axis=alt.Axis(title=None, labelLimit=LABEL_PX, labelFontSize=11),
    )
    colour = alt.Color(
        "group:N",
        scale=alt.Scale(domain=list(palette), range=list(palette.values())),
        legend=alt.Legend(
            title=colour_title,
            orient="top",
            direction="vertical" if len(palette) > 3 else "horizontal",
            labelLimit=520,
            titleLimit=520,
        ),
    )
    shape_args = (
        {
            "shape": alt.Shape(
                "group:N",
                scale=alt.Scale(domain=list(shapes), range=list(shapes.values())),
                legend=None,
            )
        }
        if shapes
        else {}
    )
    lines = base.mark_rule(strokeWidth=2, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("lower:Q", scale=scale, title=x_title),
        x2="upper:Q",
        y=y,
        color=colour,
        yOffset=alt.YOffset("group:N") if len(palette) > 1 else alt.value(0),
    )
    dots = base.mark_point(filled=True, size=70, opacity=1, aria=False).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("value:Q", scale=scale),
        y=y,
        color=colour,
        yOffset=alt.YOffset("group:N") if len(palette) > 1 else alt.value(0),
        **shape_args,
    )
    layers = [lines, dots]
    if zero_label is not None:
        zero = pl.DataFrame({"x": [0.0], "text": [zero_label]})
        layers.append(
            alt.Chart(zero)
            .mark_rule(color=ocf.BLACK_1, strokeDash=[4, 3], strokeWidth=1, aria=False)
            .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )
        layers.append(
            alt.Chart(zero)
            .mark_text(align="left", dx=4, dy=-6, fontSize=10, color=ocf.BLACK_1, aria=False)
            .encode(x="x:Q", y=alt.value(0), text="text:N")  # ty: ignore[unresolved-attribute]
        )
    return cast(alt.LayerChart, alt.layer(*layers).properties(width=PLOT_PX, height=height))


def headline_figure() -> alt.VConcatChart:
    """The five planned contrasts at both hyperparameter settings."""
    planned = pl.read_parquet(TABLES / "planned_contrasts.parquet")
    names = {
        "D1": "D1 XGBoost, actual price, vs climatology",
        "D2": "D2 actual price vs another day's",
        "D3": "D3 forecast price vs another day's",
        "D4": "D4 others' notifications vs another day's",
        "D5": "D5 NGED battery A, vs climatology",
    }
    rows = planned.select(
        label=pl.col("label").replace_strict(names),
        group=pl.when(pl.col("setting") == "primary")
        .then(pl.lit("Primary setting"))
        .otherwise(pl.lit("Second setting")),
        value="difference",
        lower="lower_95",
        upper="upper_95",
        order=pl.col("label"),
    )
    panel = _interval_panel(
        rows=rows,
        x_title="CRPS difference (points of p99; negative means the first forecast is better)",
        colour_title="XGBoost hyperparameter setting",
        palette=SETTING_COLOURS,
        zero_label="no difference",
        panel_title="Planned contrasts, 35 batteries (D5: one battery)",
        shapes={"Primary setting": "circle", "Second setting": "triangle-up"},
    )
    return figure(
        panels=[panel],
        number=1,
        title=(
            "Prices and other batteries' notifications lower the error by 0.3 to 1.3% "
            "(D2 to D4), but no XGBoost forecast beats the climatology day ahead (D1, D5)"
        ),
        subtitle=[
            (
                "CRPS: continuous ranked probability score of 13 forecast quantiles, as % of each "
                "battery's 99th-percentile absolute output (p99). Smaller is better."
            ),
            (
                "Dot: estimate. Line: 95% interval from resampling whole months and a seed. "
                "A notification is the planned output a battery files with the system operator."
            ),
            (
                "All five contrasts are planned. 35 embedded battery BMUs, October 2025 to August "
                "2026; D5 is one NGED battery."
            ),
        ],
        figure_planning=None,
    )


def leaderboard_figure() -> alt.VConcatChart:
    """Each arm's CRPS at the three issue times, testbed pooled."""
    boards = pl.read_parquet(TABLES / "leaderboards.parquet").filter(pl.col("label") == "testbed")
    issues = {
        "DA-early": "06:00 the day before (forecast price)",
        "DA-late": "18:00 the day before (actual price)",
        "ID-1h": "1 hour ahead (gate closure)",
    }
    panels = []
    for issue, title in issues.items():
        part = boards.filter(
            (pl.col("issue") == issue)
            & ~pl.col("arm").str.contains("same_party|largest_party|price_naive")
        ).sort("crps")
        rows = part.select(
            label=pl.col("arm").replace_strict(ARM_LABELS, default=pl.col("arm")),
            group=pl.col("arm").map_elements(_kind, return_dtype=pl.String),
            value="crps",
            lower="crps_lower",
            upper="crps_upper",
            order=pl.col("crps"),
        )
        panels.append(
            _interval_panel(
                rows=rows,
                x_title="CRPS (points of p99; smaller is better)",
                colour_title="Kind of forecast",
                palette=KIND_COLOURS,
                zero_label=None,
                panel_title=title,
                x_domain=(10.0, 19.0) if issue != "ID-1h" else (8.0, 19.0),
            )
        )
    return figure(
        panels=panels,
        number=6,
        title="At every issue time the XGBoost quantile models match the climatology or sit "
        "just above it; only the battery's own notification at gate closure beats it",
        subtitle=[
            (
                "Mean CRPS over 35 batteries and 11 months, as % of each battery's p99. Smaller is "
                "better. Dot: estimate. Line: 95% interval from resampling whole months and a seed."
            ),
            (
                "Intervals of absolute error are wide because every arm's error rises and falls "
                "with the weather together; Figure 1 shows the paired differences."
            ),
        ],
        figure_planning=None,
    )


def lead_figure() -> alt.VConcatChart:
    """CRPS skill against the climatology by lead time."""
    skill = pl.read_parquet(TABLES / "skill_by_lead_testbed.parquet").filter(
        pl.col("arm").is_in(
            [
                "xgb_quantile__price_actual",
                "xgb_quantile__price_model",
                "rank_conformal__price_actual",
            ]
        )
    )
    skill = skill.with_columns(
        series=pl.col("issue") + pl.lit(": ") + pl.col("arm").replace_strict(ARM_LABELS),
        lead=pl.col("lead_from_hours").cast(pl.Int64),
    )
    chart = (
        alt.Chart(skill)
        .mark_line(point=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "lead:Q", title="Lead: hours from the issue time to the start of the half-hour"
            ),
            y=alt.Y("skill:Q", title="CRPS skill against climatology (higher is better)"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_SKY, ocf.DATA_PURPLE]
                ),
                legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=420),
            ),
        )
    )
    band = (
        alt.Chart(skill)
        .mark_errorband(opacity=0.15)
        .encode(x="lead:Q", y=alt.Y("lower:Q", title=None), y2="upper:Q", color="series:N")  # ty: ignore[unresolved-attribute]
    )
    zero = (
        alt.Chart(pl.DataFrame({"y": [0.0]}))
        .mark_rule(strokeDash=[4, 3], aria=False)
        .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    panel = (band + chart + zero).properties(width=WIDE_PX, height=220)
    return figure(
        panels=[panel],
        number=7,
        title=(
            "XGBoost skill against the climatology stays within 0.03 of zero at every lead, "
            "and the rank rule is 0.03 to 0.09 below it"
        ),
        subtitle=[
            (
                "Skill: one minus the ratio of mean CRPS values; zero means the same as the "
                "56-day trailing climatology. Band: 95% interval from resampling whole months."
            ),
            (
                "35 batteries pooled; leads are 6-hour bins from the issue time. The bin from "
                "12 to 18 hours, where skill is +0.013, is one of 12 bins, so it is exploratory."
            ),
        ],
        figure_planning=None,
    )


def sensitivity_figure() -> alt.VConcatChart:
    """D1 to D4 under four ways of drawing the interval or choosing the rows."""
    planned = pl.read_parquet(TABLES / "planned_contrasts.parquet").filter(
        (pl.col("setting") == "primary") & (pl.col("part") == "A")
    )
    idle = pl.read_parquet(IDLE_TABLES / "planned_contrasts.parquet").filter(
        pl.col("setting") == "primary"
    )
    left_out = pl.read_parquet(TABLES / "leave_out_idle_lead_in.parquet").filter(
        pl.col("setting") == "primary"
    )
    party = pl.read_parquet(TABLES / "party_resampling.parquet")
    groups = {
        "As planned (months and seed resampled)": planned,
        "Idle lead-in dropped (post hoc)": idle,
        "Two idle-lead-in batteries removed (post hoc)": left_out,
    }
    pieces = [
        frame.filter(pl.col("label").is_in(["D1", "D2", "D3", "D4"])).select(
            "label",
            group=pl.lit(name),
            value="difference",
            lower="lower_95",
            upper="upper_95",
        )
        for name, frame in groups.items()
    ]
    pieces.append(
        party.select(
            "label",
            group=pl.lit("Lead parties and months resampled (post hoc)"),
            value="difference",
            lower="lower_95",
            upper="upper_95",
        )
    )
    rows = pl.concat(pieces).with_columns(order=pl.col("label"))
    palette = {
        "As planned (months and seed resampled)": ocf.BRAND_ORANGE,
        "Idle lead-in dropped (post hoc)": ocf.DATA_BLUE,
        "Two idle-lead-in batteries removed (post hoc)": ocf.DATA_PURPLE,
        "Lead parties and months resampled (post hoc)": ocf.BLACK_1,
    }
    shapes = dict(
        zip(palette, ["circle", "diamond", "square", "triangle-up"], strict=True),
    )
    panel = _interval_panel(
        rows=rows,
        x_title="CRPS difference (points of p99; negative means the first forecast is better)",
        colour_title="How the rows or the interval are chosen",
        palette=palette,
        zero_label="no difference",
        panel_title="D1 to D4, primary setting, 35 batteries",
        shapes=shapes,
    )
    return figure(
        panels=[panel],
        number=9,
        title="D2 and D4 survive every check; D1 stays null, and D3 loses its significance "
        "once lead parties are resampled",
        subtitle=[
            (
                "Contrasts as in Figure 1. One lead party runs 14 of the 35 batteries, so the "
                "lead-party resampling is the check closest to the question of other batteries."
            ),
            "The three post hoc rows were decided after the first science review saw the results.",
        ],
        figure_planning=None,
    )


def census_figure() -> alt.VConcatChart:
    """NGED's connected storage rows by size class, in count and in megawatts."""
    classes = pl.read_parquet(EMBEDDED_BATTERY_FORECAST_DIR / "census_classes.parquet")
    order = ["under 1 MW", "1 to 10 MW", "10 to 50 MW", "50 to 100 MW"]
    summary = (
        classes.group_by("size_class")
        .agg(rows=pl.len(), megawatts=pl.col("export_mw").sum())
        .filter(pl.col("size_class").is_in(order))
    )
    panels = []
    for column, title, axis in (
        ("rows", "Connected storage rows in the Embedded Capacity Register", "Rows"),
        ("megawatts", "Export capacity of those rows", "Export capacity (MW)"),
    ):
        panels.append(
            alt.Chart(summary, title=alt.TitleParams(title, anchor="start", fontSize=12))
            .mark_bar(color=ocf.BRAND_ORANGE, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("size_class:N", sort=order, title="Export size of the row"),
                x=alt.X(f"{column}:Q", title=axis),
            )
            .properties(width=PLOT_PX, height=110)
        )
    return figure(
        panels=panels,
        number=3,
        title="141 of NGED's 176 connected batteries are under 1 MW, and the 35 larger ones hold "
        "96% of the megawatts",
        subtitle=[
            (
                "Connected rows listing storage in NGED's Embedded Capacity Register, August 2026 "
                "release, four licence areas."
            ),
            (
                "Only 8 embedded storage Balancing Mechanism Units sit in NGED's four grid supply "
                "point groups, 4.5% of the rows."
            ),
        ],
        figure_planning=None,
    )


def _week_frame(*, battery: str, arm: str) -> pl.DataFrame:
    fit = pl.read_parquet(FITS / "primary" / "DA-late" / f"{battery}__{arm}.parquet").rename(
        {"q0.1": "p10", "q0.5": "p50", "q0.9": "p90"}
    )
    return fit.filter(pl.col("seed") == fit["seed"].min()).filter(
        (pl.col("time") >= WEEK_START) & (pl.col("time") < WEEK_START + timedelta(days=7))
    )


def week_figure() -> alt.VConcatChart:
    """One week of a public battery's output and the day-ahead price."""
    clim = _week_frame(battery=EXAMPLE_BATTERY, arm="clim")
    price = market_frame().filter(
        (pl.col("time") >= WEEK_START) & (pl.col("time") < WEEK_START + timedelta(days=7))
    )
    output = (
        alt.Chart(clim)
        .mark_line(color=ocf.BRAND_ORANGE, strokeWidth=1.4, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("time:T", axis=alt.Axis(format="%a %-d %b", labels=False, title=None)),
            y=alt.Y("truth_mw:Q", title="Output (MW; positive is export)"),
        )
        .properties(
            width=WIDE_PX,
            height=120,
            title=alt.TitleParams(
                f"Metered output of {EXAMPLE_BATTERY}", anchor="start", fontSize=12
            ),
        )
    )
    prices = (
        alt.Chart(price)
        .mark_line(color=ocf.DATA_BLUE, strokeWidth=1.4, interpolate="step-after", aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "time:T", axis=alt.Axis(format="%a %-d %b", tickCount="day", title="Day (UTC)")
            ),
            y=alt.Y("day_ahead_gbp_per_mwh:Q", title="Day-ahead price (GBP per MWh)"),
        )
        .properties(
            width=WIDE_PX,
            height=110,
            title=alt.TitleParams("N2EX day-ahead price, hourly", anchor="start", fontSize=12),
        )
    )
    return figure(
        panels=[output, prices],
        number=2,
        title="A battery charges and exports in short bursts that follow the day-ahead price only "
        "loosely",
        subtitle=[
            (
                f"{EXAMPLE_BATTERY}, an embedded battery BMU, for the week starting Monday "
                "15 December 2025, UTC. The week is the study's fixed example week."
            ),
        ],
        figure_planning=None,
    )


def fan_figure() -> alt.VConcatChart:
    """The climatology and the XGBoost quantile model as fans over the same week."""
    panels = []
    for arm, title, colour in (
        ("clim", "Climatology: the 56 trailing days at the same time of day", ocf.DATA_SKY),
        (
            "xgb_quantile__price_actual",
            "XGBoost quantile model given the actual price, issued at 18:00 the day before",
            ocf.BRAND_ORANGE,
        ),
    ):
        week = _week_frame(battery=EXAMPLE_BATTERY, arm=arm)
        band = (
            alt.Chart(week)
            .mark_area(color=colour, opacity=0.25, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "time:T", axis=alt.Axis(format="%a %-d %b", tickCount="day", title="Day (UTC)")
                ),
                y=alt.Y("p10:Q", title="Output (MW)"),
                y2="p90:Q",
            )
        )
        median = (
            alt.Chart(week)
            .mark_line(color=colour, strokeWidth=1.4, aria=False)
            .encode(x="time:T", y="p50:Q")  # ty: ignore[unresolved-attribute]
        )
        truth = (
            alt.Chart(week)
            .mark_line(color=ocf.BLACK_1, strokeWidth=1.0, aria=False)
            .encode(x="time:T", y="truth_mw:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append(
            (band + median + truth).properties(
                width=WIDE_PX,
                height=130,
                title=alt.TitleParams(title, anchor="start", fontSize=12),
            )
        )
    return figure(
        panels=panels,
        number=4,
        title="Both forecasts give a wide band, and neither predicts which days the battery cycles",
        subtitle=[
            (
                f"{EXAMPLE_BATTERY}, the same week as Figure 2. Black line: metered output. "
                "Coloured line: median forecast. Shaded band: p10 to p90 of the forecast."
            ),
        ],
        figure_planning=None,
    )


def reliability_figure() -> alt.VConcatChart:
    """Observed share of half-hours below each forecast quantile, at 18:00 the day before."""
    from forecast_report import coverage_summary
    from forecast_runner import testbed_ids

    levels = [0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98, 0.99]
    rows = []
    for arm in ("clim", "xgb_quantile__price_actual", "rank_conformal__price_actual"):
        stats = coverage_summary(
            setting="primary", issue="DA-late", arm=arm, batteries=testbed_ids()
        )
        rows += [
            {"arm": ARM_LABELS[arm], "level": lv, "observed": share}
            for lv, share in zip(levels, stats["reliability"], strict=True)
        ]
    frame = pl.DataFrame(rows)
    diagonal = pl.DataFrame({"level": [0.0, 1.0], "observed": [0.0, 1.0]})
    palette = [ocf.BLACK_1, ocf.BRAND_ORANGE, ocf.DATA_SKY]
    lines = (
        alt.Chart(frame)
        .mark_line(point=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "level:Q",
                title="Forecast quantile level",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".1f", values=[0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            ),
            y=alt.Y("observed:Q", title="Share of half-hours at or below that quantile"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(range=palette),
                legend=alt.Legend(
                    title=None, orient="top", direction="vertical", labelLimit=500, symbolLimit=0
                ),
            ),
        )
    )
    reference = (
        alt.Chart(diagonal)
        .mark_line(color=ocf.BLACK_1, strokeDash=[4, 3], strokeWidth=1, aria=False)
        .encode(x="level:Q", y="observed:Q")  # ty: ignore[unresolved-attribute]
    )
    panel = (reference + lines).properties(width=WIDE_PX, height=260)
    return figure(
        panels=[panel],
        number=5,
        title="The XGBoost forecast is close to calibrated; the climatology's lower tail is too "
        "narrow, with 6% of outcomes below its p1",
        subtitle=[
            (
                "A calibrated forecast lies on the dashed diagonal. 35 batteries, issued 18:00 the "
                "day before. Below the diagonal at a low level, or above it at a high level, means "
                "outcomes fall outside the forecast more often than it says."
            ),
        ],
        figure_planning=None,
    )


def battery_skill_figure() -> alt.VConcatChart:
    """Each battery's CRPS skill against the climatology at 18:00 the day before."""
    skill = pl.read_parquet(TABLES / "per_battery_skill.parquet").select(
        battery="battery",
        skill="DA-late__xgb_quantile__price_actual",
    )
    positive = int((skill["skill"] > 0).sum())
    ordered = skill.sort("skill")
    sort = ordered["battery"].to_list()
    panel = (
        alt.Chart(skill)
        .mark_point(filled=True, size=60, color=ocf.BRAND_ORANGE, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("battery:N", sort=sort, title=None, axis=alt.Axis(labelFontSize=9)),
            x=alt.X("skill:Q", title="CRPS skill against climatology (higher is better)"),
        )
        .properties(width=PLOT_PX, height=12 * len(sort))
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(strokeDash=[4, 3], aria=False)
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    return figure(
        panels=[(panel + zero).properties(width=PLOT_PX, height=12 * len(sort))],
        number=8,
        title=f"{positive} of 35 batteries beat the climatology slightly, and 2 do much worse",
        subtitle=[
            (
                "XGBoost quantile model given the actual price, issued 18:00 the day before, "
                "against the 56-day trailing climatology. One dot per testbed battery."
            ),
            "The two batteries far below zero were switched off until spring 2026.",
        ],
        figure_planning=None,
    )


def nged_week_figure() -> alt.VConcatChart:
    """NGED battery A over seven numbered days: outcome and the two forecasts' medians."""
    start = pl.lit(NGED_WEEK_START).str.to_datetime(time_zone="UTC")
    frames = []
    for arm, name in (
        ("clim", "Climatology"),
        ("xgb_quantile__price_actual", "XGBoost, actual price"),
    ):
        fit = pl.read_parquet(FITS / "primary" / "DA-late" / f"nged_battery_a__{arm}.parquet")
        fit = fit.filter(pl.col("seed") == fit["seed"].min()).rename({"q0.1": "p10", "q0.9": "p90"})
        week = fit.filter(
            (pl.col("time") >= start) & (pl.col("time") < start + pl.duration(days=7))
        ).with_columns(
            hours=((pl.col("time") - start) / pl.duration(hours=1)),
            name=pl.lit(name),
        )
        frames.append(week)
    long = pl.concat(frames)
    truth = long.filter(pl.col("name") == "Climatology")
    base = alt.Chart(long).encode(
        x=alt.X(
            "hours:Q",
            title="Day (1 to 7)",
            scale=alt.Scale(domain=[0, 168]),
            axis=alt.Axis(
                values=[12 + 24 * i for i in range(7)], labelExpr="floor((datum.value)/24)+1"
            ),
        )
    )
    band = base.mark_area(opacity=0.2, aria=False).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("p10:Q", title="Output (fraction of p99)"),
        y2="p90:Q",
        color=alt.Color(
            "name:N",
            scale=alt.Scale(range=[ocf.DATA_SKY, ocf.BRAND_ORANGE]),
            legend=alt.Legend(title="Forecast, p10 to p90 band", orient="top"),
        ),
    )
    actual = (
        alt.Chart(truth)
        .mark_line(color=ocf.BLACK_1, strokeWidth=1.0, aria=False)
        .encode(x="hours:Q", y="truth_mw:Q")  # ty: ignore[unresolved-attribute]
    )
    panel = (band + actual).properties(width=WIDE_PX, height=180)
    return figure(
        panels=[panel],
        number=10,
        title="NGED battery A's output is irregular, and neither forecast tracks its swings",
        subtitle=[
            (
                "Black line: metered output of NGED battery A as a fraction of its own p99. "
                "Bands: p10 to p90 of each forecast issued 18:00 the day before."
            ),
            "Seven days chosen by a fixed rule (a Monday in spring 2026), shown as days 1 to 7.",
        ],
        figure_planning=None,
    )


FIGURES: Final[dict[str, object]] = {
    "headline": headline_figure,
    "leaderboards": leaderboard_figure,
    "skill_by_lead": lead_figure,
    "sensitivity": sensitivity_figure,
    "census": census_figure,
    "example_week": week_figure,
    "fans": fan_figure,
    "reliability": reliability_figure,
    "per_battery": battery_skill_figure,
    "nged_week": nged_week_figure,
}


def main() -> None:
    """Write every figure as an SVG under `docs/studies/assets/` and a PNG preview."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    previews = EMBEDDED_BATTERY_FORECAST_DIR / "previews"
    previews.mkdir(exist_ok=True)
    for name, build in FIGURES.items():
        chart = build()  # ty: ignore[call-non-callable]
        target = ASSETS_DIR / f"embedded_battery_forecast_{name}.svg"
        chart.save(target)
        chart.save(previews / f"{name}.png", scale_factor=1.3)
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
