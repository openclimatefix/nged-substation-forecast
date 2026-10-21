"""Draw the provisional substation cards and the cross-substation figure of experiment 114.

One card per decision substation (D9 marked short-history), from B0 and T-all at the 2025-07-01 cut
and from the filtered forecasts of the scored year. **Provisional: fitted to past weather on a
screening run.** A threshold is printed only for a substation that passes the per-site printing
rule; the heating slopes carry the line that experiment 113 found them weakly identified.

    uv run python ml-experiments/00114-*/e00114_cards.py
"""

import json
import pickle
import textwrap
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00114_charts as charts
import e00114_common as common
import e00114_design as design
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.dp_calendar import DAY_TYPES, HOLIDAY_PERIODS, SMOOTHING_HOURS
from studies.dp_gam import N_BANDS
from studies.dp_pinned_capacity import read_vintages

CARD_WIDTH: Final[int] = 215
PANEL_HEIGHT: Final[int] = 110
BAND_LABELS: Final[tuple[str, ...]] = ("night", "morning", "midday", "evening")
BLOCK_LABELS: Final[tuple[str, ...]] = (
    "night 00-03",
    "night 03-06",
    "morning 06-08:30",
    "morning 08:30-11",
    "midday 11-13:30",
    "midday 13:30-16",
    "evening 16-20",
    "evening 20-24",
)
TENT_LABELS: Final[tuple[str, ...]] = ("Christmas tent", "Christmas tent (signed)")
WEAK_NOTE: Final[str] = (
    "Weakly identified at real noise levels (experiment 113); compare across substations with care."
)


def _decode(*, site: str) -> dict[str, Any]:
    """Return a substation's coefficients at the scored cut, decoded into named arrays."""
    cards = pickle.loads((common.WORK_DIR / "coefficients.pkl").read_bytes())
    cache = common.load_cache()
    base = design.build_base(
        cache=cache, vintages=read_vintages(), site=site, cut=common.SCORED_CUT
    )
    slices = base.layout.slices
    b0 = cards[site]["b0"]["scored"]
    t_all = cards[site]["t_all"]["scored"]
    arm = design.interaction_arm(name="t_all", penalty_weight=0.0)
    replaced = design.replaced_columns(layout=base.layout, arm=arm)
    drop = {column for column, _ in replaced}
    kept = [j for j in range(len(b0)) if j not in drop]
    position = len(kept)
    pieces: dict[int, np.ndarray] = {}
    for column, factors in replaced:
        count = 2 ** len(factors)
        pieces[column] = t_all[position : position + count]
        position += count
    heating = slices["heating"]
    lighting = slices["lighting"]
    blocks = np.zeros((2, 8))
    for k in range(len(SMOOTHING_HOURS)):
        for band in range(N_BANDS):
            column = heating.start + k * N_BANDS + band
            for combo, value in enumerate(pieces[column]):
                day_group, half = divmod(combo, 2)
                blocks[day_group, 2 * band + half] += value
    light = np.array([pieces[j] for j in range(lighting.start, lighting.stop)])
    return {
        "profile": b0[slices["profile"]].reshape(len(DAY_TYPES), 48),
        "holiday": b0[slices["holiday"]].reshape(len(HOLIDAY_PERIODS), N_BANDS),
        "tent": b0[slices["tent"]].reshape(2, N_BANDS),
        "heating": b0[heating].reshape(len(SMOOTHING_HOURS), N_BANDS),
        "lighting_b0": b0[lighting],
        "lighting_pieces": light,
        "blocks": blocks,
        "pinned_points": float(base.pinned.held_points)
        if base.pinned is not None
        else float("nan"),
    }


def _frame_level(*, site: str) -> pl.DataFrame:
    """Return the daily mean persistent and transient correction of B0-K and T-all-K."""
    parts = []
    for arm, label in (("b0_k", "B0-K"), ("t_all_k", "T-all-K")):
        frame = pl.read_parquet(
            common.DATA_DIR / "predictions" / f"scored_{arm}" / "predictions.parquet"
        )
        site_id = {"D9": 7, "D11": 8, "D10": 10, "D14": 16, "D5": 17, "D16": 18}[site]
        daily = (
            frame.filter(pl.col("time_series_id") == site_id)
            .group_by(pl.col("valid_time").dt.date().alias("date"))
            .agg(pl.col("level").mean(), pl.col("anomaly").mean())
            .sort("date")
            .with_columns(arm=pl.lit(label))
        )
        parts.append(daily)
    return pl.concat(parts).unpivot(
        on=["level", "anomaly"], index=["date", "arm"], variable_name="part", value_name="value"
    )


def _panel_title(text: str) -> alt.TitleParams:
    return alt.TitleParams(text=text, anchor="start", fontSize=11)


def card(*, site: str) -> alt.TopLevelMixin:
    """Return one substation's card."""
    data = _decode(site=site)
    records = json.loads((common.WORK_DIR / "profiles.json").read_text())
    scored = next(r for r in records if r["site"] == site and r["label"] == "scored")
    capacity = next(
        r
        for r in json.loads((common.WORK_DIR / "capacity_check.json").read_text())
        if r["site"] == site
    )
    hours = np.arange(48) / 2.0
    profile = pl.DataFrame(
        [
            {"hour": hours[i], "day type": DAY_TYPES[d].replace("_", " "), "points": v}
            for d in range(len(DAY_TYPES))
            for i, v in enumerate(data["profile"][d])
        ]
    )
    profile_chart = (
        alt.Chart(profile)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.2)
        .encode(
            x=alt.X(
                "hour:Q",
                title="Local hour",
                axis=alt.Axis(values=[0, 6, 12, 18, 24]),
                scale=alt.Scale(domain=[0, 24]),
            ),
            y=alt.Y("points:Q", title="Points", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "day type:N",
                scale=alt.Scale(
                    domain=["weekday", "saturday", "sunday", "bank holiday"],
                    range=[DARK_COLOUR, ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_PURPLE],
                ),
                legend=alt.Legend(title=None, orient="top", columns=2, labelFontSize=9),
            ),
        )
        .properties(
            width=CARD_WIDTH, height=PANEL_HEIGHT, title=_panel_title("Daily profile by day type")
        )
    )
    holiday_rows = [
        {"effect": HOLIDAY_PERIODS[p], "band": BAND_LABELS[b], "points": data["holiday"][p, b]}
        for p in range(len(HOLIDAY_PERIODS))
        for b in range(N_BANDS)
    ] + [
        {"effect": TENT_LABELS[t], "band": BAND_LABELS[b], "points": data["tent"][t, b]}
        for t in range(2)
        for b in range(N_BANDS)
    ]
    holiday_chart = (
        alt.Chart(pl.DataFrame(holiday_rows))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=18)
        .encode(
            y=alt.Y(
                "effect:N",
                sort=[r["effect"] for r in holiday_rows[::N_BANDS]],
                title=None,
                axis=alt.Axis(labelFontSize=8, labelLimit=110),
            ),
            x=alt.X("points:Q", title="Points of effective capacity"),
            color=alt.Color(
                "band:N",
                scale=alt.Scale(
                    domain=list(BAND_LABELS),
                    range=[ocf.DATA_PURPLE, ocf.DATA_BLUE, ocf.DATA_SKY, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None, orient="top", columns=4, labelFontSize=9),
            ),
        )
        .properties(
            width=CARD_WIDTH,
            height=PANEL_HEIGHT + 60,
            title=_panel_title("Holiday and Christmas effects"),
        )
    )
    nats = pl.DataFrame({"threshold": common.HINGE_GRID_C, "nats": scored["nats"]})
    printed = (
        f"Threshold {scored['argmin_c']:g} C or lower (80% interval {scored['interval_low_c']:g} "
        f"to {scored['interval_high_c']:g}; the grid starts at 12 C)"
        if scored["printable"]
        else "Threshold not identified at this substation"
    )
    nats_chart = (
        alt.Chart(nats)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, color=ocf.DATA_BLUE)
        .encode(
            x=alt.X(
                "threshold:Q",
                title="Heating threshold (C)",
                axis=alt.Axis(values=[12, 14, 16, 18]),
                scale=alt.Scale(domain=[11.5, 19.5]),
            ),
            y=alt.Y(
                "nats:Q",
                title="Effective nats",
                scale=alt.Scale(type="symlog", constant=2),
                axis=alt.Axis(values=[0, 2, 10, 50, 200]),
            ),
        )
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [15.5]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    two_nats = (
        alt.Chart(pl.DataFrame({"y": [2.0]})).mark_rule(color=GREY_COLOUR).encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    heating_profile = (nats_chart + rule + two_nats).properties(
        width=CARD_WIDTH,
        height=PANEL_HEIGHT,
        title=alt.TitleParams(
            text="Loss against heating threshold",
            subtitle=[printed],
            anchor="start",
            fontSize=11,
            subtitleFontSize=9,
        ),
    )
    slope_rows = pl.DataFrame(
        {
            "band": list(BAND_LABELS),
            "slope": scored["band_slope"],
            "low": scored["band_slope_low"],
            "high": scored["band_slope_high"],
        }
    )
    slope_base = alt.Chart(slope_rows)
    slopes = slope_base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("band:N", sort=list(BAND_LABELS), title=None),
        x=alt.X("low:Q", title="Points per 10 C (B0, all four windows)"),
        x2="high:Q",
    ) + slope_base.mark_point(filled=True, color=DARK_COLOUR).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("band:N", sort=list(BAND_LABELS)), x="slope:Q"
    )
    slope_chart = slopes.properties(
        width=CARD_WIDTH,
        height=PANEL_HEIGHT,
        title=alt.TitleParams(
            text="Heating slope by hour band (weak)",
            subtitle=textwrap.wrap(WEAK_NOTE, 52),
            anchor="start",
            fontSize=11,
            subtitleFontSize=9,
        ),
    )
    block_rows = pl.DataFrame(
        [
            {"block": BLOCK_LABELS[i], "group": group, "slope": data["blocks"][g, i]}
            for g, group in enumerate(("weekday", "Saturday, Sunday, bank holiday"))
            for i in range(8)
        ]
    )
    block_chart = (
        alt.Chart(block_rows)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=30)
        .encode(
            y=alt.Y("block:N", sort=list(BLOCK_LABELS), title=None, axis=alt.Axis(labelFontSize=8)),
            x=alt.X("slope:Q", title="Points per 10 C (T-all)"),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(
                    domain=["weekday", "Saturday, Sunday, bank holiday"],
                    range=[DARK_COLOUR, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(
                    title=None, orient="top", columns=1, labelFontSize=9, labelLimit=260
                ),
            ),
        )
        .properties(
            width=CARD_WIDTH,
            height=PANEL_HEIGHT + 20,
            title=_panel_title("Heating slope by half of band"),
        )
    )
    inertia = data["heating"].sum(axis=1)
    inertia_rows = pl.DataFrame(
        {"window": [f"{h} h" for h in SMOOTHING_HOURS], "share": (inertia / inertia.sum()).tolist()}
    )
    inertia_chart = (
        alt.Chart(inertia_rows)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_SKY)
        .encode(
            x=alt.X(
                "window:N",
                sort=[f"{h} h" for h in SMOOTHING_HOURS],
                title="Temperature averaged over",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("share:Q", title="Share of heating slope"),
        )
        .properties(
            width=CARD_WIDTH, height=PANEL_HEIGHT, title=_panel_title("Thermal-inertia mix (B0)")
        )
    )
    light_rows = pl.DataFrame(
        [
            {"ramp": f"ramp {i + 1}", "series": "B0", "points": data["lighting_b0"][i]}
            for i in range(4)
        ]
        + [
            {
                "ramp": f"ramp {i + 1}",
                "series": "T-all weekday",
                "points": data["lighting_pieces"][i][0],
            }
            for i in range(4)
        ]
        + [
            {
                "ramp": f"ramp {i + 1}",
                "series": "T-all weekend",
                "points": data["lighting_pieces"][i][1],
            }
            for i in range(4)
        ]
    )
    light_chart = (
        alt.Chart(light_rows)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=40)
        .encode(
            x=alt.X(
                "ramp:N",
                title="Darkness ramp (1: sun just below the horizon)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("points:Q", title="Points"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=["B0", "T-all weekday", "T-all weekend"],
                    range=[GREY_COLOUR, DARK_COLOUR, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None, orient="top", columns=3, labelFontSize=9),
            ),
        )
        .properties(
            width=CARD_WIDTH, height=PANEL_HEIGHT, title=_panel_title("Lighting by day type")
        )
    )
    solar_rows = pl.DataFrame(
        {
            "source": ["pinned vintage", "B0, fitted", "T-all, fitted"],
            "points": [
                data["pinned_points"],
                capacity["b0_latest_capacity_points"],
                capacity["t_all_latest_capacity_points"],
            ],
        }
    )
    solar_chart = (
        alt.Chart(solar_rows)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("source:N", sort=["pinned vintage", "B0, fitted", "T-all, fitted"], title=None),
            x=alt.X("points:Q", title="Latest PV capacity (points of effective capacity)"),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(
                    domain=["pinned vintage", "B0, fitted", "T-all, fitted"],
                    range=[ocf.DATA_GREEN, GREY_COLOUR, ocf.BRAND_ORANGE],
                ),
                legend=None,
            ),
        )
        .properties(
            width=CARD_WIDTH,
            height=PANEL_HEIGHT - 20,
            title=alt.TitleParams(
                text="Solar capacity",
                subtitle=["No interval; the fitted values are not measurements."],
                anchor="start",
                fontSize=11,
                subtitleFontSize=9,
            ),
        )
    )
    filter_chart = (
        alt.Chart(_frame_level(site=site))  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1)
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y", labelFontSize=9)),
            y=alt.Y("value:Q", title="Points"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=["B0-K", "T-all-K"], range=[DARK_COLOUR, ocf.BRAND_ORANGE]),
                legend=alt.Legend(title=None, orient="top", labelFontSize=9),
            ),
            row=alt.Row(
                "part:N",
                title=None,
                sort=["level", "anomaly"],
                header=alt.Header(labelFontSize=9, labelAngle=0, labelAlign="left"),
            ),
        )
        .properties(
            width=2 * CARD_WIDTH + 90,
            height=50,
            title=_panel_title(
                "Filter: daily mean persistent level and transient anomaly, scored year"
            ),
        )
    )
    top = alt.hconcat(profile_chart, holiday_chart).resolve_scale(color="independent")
    mid = alt.hconcat(heating_profile, slope_chart)
    mid2 = alt.hconcat(block_chart, inertia_chart)
    low = alt.hconcat(light_chart, solar_chart).resolve_scale(color="independent")
    stack = alt.vconcat(top, mid, mid2, low, filter_chart).resolve_scale(color="independent")
    suffix = " (short history: 9 months at the replicate cut)" if site == "D9" else ""
    return stack.properties(
        title=alt.TitleParams(
            text=f"Substation {site}{suffix}",
            subtitle=["Provisional: fitted to past weather on a screening run."],
            anchor="start",
            fontSize=14,
            subtitleFontSize=11,
        )
    )


SAMPLE_WEEKS: Final[dict[str, date]] = {
    "Summer: 14 to 20 July 2025": date(2025, 7, 14),
    "Autumn: 13 to 19 October 2025": date(2025, 10, 13),
    "Winter: 12 to 18 January 2026": date(2026, 1, 12),
    "Spring: 13 to 19 April 2026": date(2026, 4, 13),
}
"""One week per season of the scored year, fixed before any score was read by one rule: the week
that starts on the second Monday of the season's middle month."""


def forecast_weeks(*, site: str) -> alt.TopLevelMixin:
    """Return a substation's sample weeks: observed power against B0-K and T-all-K day ahead.

    Each panel shows the median forecast, the 80% band (10th to 90th percentile), the 90% band
    (5th to 95th percentile), and the observed power, in points of effective capacity.

    Args:
        site: The decision substation's label.

    Returns:
        The chart.
    """
    site_id = {"D9": 7, "D11": 8, "D10": 10, "D14": 16, "D5": 17, "D16": 18}[site]
    parts = []
    for arm, label in (("b0_k", "B0-K"), ("t_all_k", "T-all-K")):
        frame = pl.read_parquet(
            common.DATA_DIR / "predictions" / f"scored_{arm}" / "predictions.parquet"
        ).filter(pl.col("time_series_id") == site_id)
        for week, start in SAMPLE_WEEKS.items():
            begin = datetime(start.year, start.month, start.day, tzinfo=UTC)
            part = frame.filter(
                pl.col("valid_time") > begin, pl.col("valid_time") <= begin + timedelta(days=7)
            ).with_columns(
                week=pl.lit(week),
                arm=pl.lit(label),
                day=(pl.col("valid_time") - begin).dt.total_hours() / 24.0,
            )
            parts.append(
                part.select("week", "arm", "day", "q05", "q10", "q50", "q90", "q95", "actual")
            )
    table = pl.concat(parts)
    x = alt.X(
        "day:Q",
        title="Days since Monday 00:00 UTC",
        axis=alt.Axis(values=list(range(8))),
        scale=alt.Scale(domain=[0, 7]),
    )
    base = alt.Chart(table).encode(x=x)
    wide = base.mark_area(opacity=0.18, color=ocf.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("q05:Q", title="Points"), y2="q95:Q"
    )
    narrow = base.mark_area(opacity=0.3, color=ocf.BRAND_ORANGE).encode(y="q10:Q", y2="q90:Q")  # ty: ignore[unresolved-attribute]
    median = base.mark_line(color=ocf.BRAND_ORANGE, strokeWidth=1).encode(y="q50:Q")  # ty: ignore[unresolved-attribute]
    observed = base.mark_line(color=DARK_COLOUR, strokeWidth=0.8).encode(y="actual:Q")  # ty: ignore[unresolved-attribute]
    layered = (wide + narrow + median + observed).properties(width=2 * CARD_WIDTH + 60, height=70)
    return layered.facet(
        row=alt.Row(
            "week:N",
            sort=list(SAMPLE_WEEKS),
            title=None,
            header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=9),
        ),
        column=alt.Column("arm:N", sort=["B0-K", "T-all-K"], title=None),
    ).properties(
        title=alt.TitleParams(
            text=f"Substation {site}: one sample week per season, day ahead",
            subtitle=textwrap.wrap(
                "Black line: observed power. Orange line: median forecast. Shades: 80% and 90% "
                "bands. The weeks are the second Monday of the middle month of each season, fixed "
                "before any score was read. Provisional: fitted to past weather on a "
                "screening run.",
                charts.WRAP,
            ),
            anchor="start",
            fontSize=14,
            subtitleFontSize=11,
        )
    )


DARK_COLOUR: Final[str] = "#333333"
GREY_COLOUR: Final[str] = "#808080"


def cross_substation() -> alt.TopLevelMixin:
    """Return the figure that compares the key properties of the substations side by side."""
    records = json.loads((common.WORK_DIR / "profiles.json").read_text())
    capacity = {
        r["site"]: r for r in json.loads((common.WORK_DIR / "capacity_check.json").read_text())
    }
    vintages = read_vintages()
    cache = common.load_cache()
    rows_capacity, rows_ratio, rows_profile = [], [], []
    sites = [*common.DECISION_SITES, "D9"]
    for site in sites:
        data = _decode(site=site)
        pinned = design.build_base(
            cache=cache, vintages=vintages, site=site, cut=common.SCORED_CUT
        ).pinned
        rows_capacity += [
            {
                "site": site,
                "source": "Pinned vintage",
                "points": pinned.held_points if pinned is not None else float("nan"),
            },
            {
                "site": site,
                "source": "B0, fitted",
                "points": capacity[site]["b0_latest_capacity_points"],
            },
            {
                "site": site,
                "source": "T-all, fitted",
                "points": capacity[site]["t_all_latest_capacity_points"],
            },
        ]
        weekday, weekend = data["blocks"][0].sum(), data["blocks"][1].sum()
        rows_ratio.append(
            {"site": site, "ratio": weekend / weekday, "total": data["heating"].sum()}
        )
        scored = next(r for r in records if r["site"] == site and r["label"] == "scored")
        rows_profile.append(
            {
                "site": site,
                "low": scored["within_2_nats_low_c"],
                "high": scored["within_2_nats_high_c"],
                "argmin": scored["argmin_c"],
            }
        )
    order = sites
    capacity_chart = (
        alt.Chart(pl.DataFrame(rows_capacity))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=50)
        .encode(
            y=alt.Y("site:N", sort=order, title=None),
            x=alt.X(
                "points:Q", title="Latest PV capacity (points of effective capacity; no interval)"
            ),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(
                    domain=["Pinned vintage", "B0, fitted", "T-all, fitted"],
                    range=[ocf.DATA_GREEN, GREY_COLOUR, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None, orient="top", columns=3),
            ),
        )
        .properties(width=charts.WIDTH, height=130)
    )
    ratio_chart = (
        alt.Chart(pl.DataFrame(rows_ratio))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=50, color=ocf.BRAND_ORANGE)
        .encode(
            y=alt.Y("site:N", sort=order, title=None),
            x=alt.X(
                "ratio:Q",
                title="T-all: weekend heating slope over weekday heating slope (1: no difference)",
                scale=alt.Scale(domain=[0.6, 1.2]),
            ),
        )
        .properties(width=charts.WIDTH, height=130)
    )
    profile_base = alt.Chart(pl.DataFrame(rows_profile))
    profile_chart = (
        profile_base.mark_rule(strokeWidth=4, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("site:N", sort=order, title=None),
            x=alt.X(
                "low:Q",
                title="Heating thresholds within 2 effective nats of the best (C)",
                scale=alt.Scale(domain=[11.5, 19.5]),
            ),
            x2="high:Q",
        )
        + profile_base.mark_point(filled=True, color=DARK_COLOUR).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("site:N", sort=order), x="argmin:Q"
        )
        + alt.Chart(pl.DataFrame({"x": [15.5]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[4, 3])
        .encode(x="x:Q")
    ).properties(width=charts.WIDTH, height=130)
    return alt.vconcat(capacity_chart, ratio_chart, profile_chart).properties(
        title=alt.TitleParams(
            text="Cross-substation comparison (provisional: past weather, screening run)",
            subtitle=textwrap.wrap(
                "Top: the pinned vintage capacity and the two fitted capacities side by side; the "
                "fitted capacity is not a measurement. Middle: weekend over weekday heating slope. "
                "Bottom: dot, the threshold with the lowest loss; line, thresholds within 2 "
                "effective nats of it; orange dashed rule, B0's 15.5 C. " + WEAK_NOTE,
                charts.WRAP,
            ),
            anchor="start",
            fontSize=14,
            subtitleFontSize=11,
        )
    )


def main() -> None:
    """Draw every card and the cross-substation figure."""
    Path(charts.PNG_DIR).mkdir(parents=True, exist_ok=True)
    for site in [*common.DECISION_SITES, "D9"]:
        charts._save(card(site=site), name=f"card_{site}")
    for site in common.DECISION_SITES:
        charts._save(forecast_weeks(site=site), name=f"weeks_{site}")
    charts._save(cross_substation(), name="figure_cross_substation")


if __name__ == "__main__":
    main()
