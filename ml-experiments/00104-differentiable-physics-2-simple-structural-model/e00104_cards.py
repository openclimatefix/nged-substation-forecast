"""Draw a card per decision substation of what the fitted G-all+D1 model says, and a comparison.

The comparison sets the key fitted parameters of the five substations side by side.

The cards reuse the tables of `e00104_decomposition.build_tables`, fitted at the final cut
2025-07-01 with the tuned penalties. Only Primary substations appear, under their D-labels. No
interval is drawn for capacity because the fit gives a point estimate only. The cards are
provisional: they show a screening run fitted to past weather.

    POLARS_MAX_THREADS=4 uv run python ml-experiments/00104-*/e00104_cards.py
"""

import textwrap
from typing import Final

import altair as alt
import e00104_charts as charts
import e00104_common as common
import e00104_decomposition as decomposition
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.dp_calendar import BAND_NAMES, HOLIDAY_PERIODS

PANEL_WIDTH: Final[int] = 120
PANEL_HEIGHT: Final[int] = 100
PROVISIONAL: Final[str] = "Provisional: fitted to past weather on a screening run."
ORDER: Final[tuple[str, ...]] = common.ELIGIBLE_SITES


def _band_colour(*, legend: bool = False) -> alt.Color:
    return alt.Color(
        "band:N",
        sort=list(BAND_NAMES),
        scale=alt.Scale(domain=list(BAND_NAMES), range=decomposition.BAND_COLOURS),
        legend=alt.Legend(title="Hour band", orient="top", direction="horizontal")
        if legend
        else None,
    )


def card(*, site: str, tables: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """Return the card of one substation: nine small panels in three rows."""

    def only(name: str) -> pl.DataFrame:
        return tables[name].filter(pl.col("site") == site)

    def sized(chart: alt.Chart, title: str) -> alt.Chart:
        return chart.properties(
            width=PANEL_WIDTH,
            height=PANEL_HEIGHT,
            title=alt.TitleParams(text=textwrap.wrap(title, 26), fontSize=10, anchor="start"),
        )

    window_slopes = (
        only("heating_windows").group_by("window (hours)").agg(pl.col("points per 10 °C").mean())
    )
    holiday_means = only("holiday").group_by("period").agg(pl.col("points").mean())
    capacity = sized(
        alt.Chart(only("capacity").filter(pl.col("season-year") >= 2019))  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_SKY)
        .encode(
            x=alt.X("season-year:O", title=None, axis=alt.Axis(labelAngle=-90)),
            y=alt.Y("capacity:Q", title="Points"),
        ),
        "Unmetered PV capacity per season-year",
    )
    profile = sized(
        alt.Chart(only("profile"))  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.2)
        .encode(
            x=alt.X("hour:Q", title="Local hour", axis=alt.Axis(values=[0, 6, 12, 18, 24])),
            y=alt.Y("points:Q", title="Points"),
            color=alt.Color(
                "day type:N",
                scale=alt.Scale(
                    domain=["weekday", "saturday", "sunday", "bank holiday"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE, ocf.DATA_GREEN],
                ),
                legend=None,
            ),
        ),
        "Daily profile by day type",
    )
    heating = sized(
        alt.Chart(only("heating_curve"))  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.2)
        .encode(
            x=alt.X("temperature (°C):Q", title="Steady temperature (°C)"),
            y=alt.Y("points:Q", title="Points"),
            color=_band_colour(),
        ),
        "Heating response, by hour band",
    )
    windows = sized(
        alt.Chart(window_slopes)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(
            x=alt.X("window (hours):O", title="Smoothing window (hours)"),
            y=alt.Y("points per 10 °C:Q", title="Points per 10 °C"),
        ),
        "Heating slope per smoothing window (thermal inertia)",
    )
    holiday = sized(
        alt.Chart(holiday_means)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.BRAND_ORANGE)
        .encode(
            y=alt.Y("period:N", sort=list(HOLIDAY_PERIODS), title=None),
            x=alt.X("points:Q", title="Points (mean of the four bands)"),
        ),
        "Holiday periods",
    ).properties(height=PANEL_HEIGHT + 60)
    tent = sized(
        alt.Chart(only("tent"))  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.2)
        .encode(
            x=alt.X("days from Christmas Day:Q", title="Days from 25 December"),
            y=alt.Y("points:Q", title="Points"),
            color=_band_colour(),
        ),
        "Christmas tent",
    )
    lighting = sized(
        alt.Chart(only("lighting"))  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.2)
        .encode(
            x=alt.X("hour:Q", title="Local hour", axis=alt.Axis(values=[0, 6, 12, 18, 24])),
            y=alt.Y("points:Q", title="Points"),
            color=alt.Color(
                "day:N",
                scale=alt.Scale(
                    domain=list(decomposition.LIGHTING_DAYS),
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
                legend=None,
            ),
        ),
        "Lighting: winter (blue) and summer (orange) day",
    )
    drift = sized(
        alt.Chart(only("drift"))  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1)
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y", tickCount=3)),
            y=alt.Y("points:Q", title="Points"),
            color=_band_colour(),
            strokeDash=alt.StrokeDash(
                "held:N", scale=alt.Scale(domain=[False, True], range=[[1, 0], [4, 3]]), legend=None
            ),
        ),
        "Drift level, held (dashed) through the scored year",
    )
    steps = sized(
        alt.Chart(only("steps"))  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1)
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y", tickCount=3)),
            y=alt.Y("points:Q", title="Points"),
            color=_band_colour(),
        ),
        "Cumulative steps (steps arm)",
    )
    return (
        alt.vconcat(
            alt.hconcat(capacity, profile, drift).resolve_scale(color="independent"),
            alt.hconcat(heating, windows, steps).resolve_scale(color="independent"),
            alt.hconcat(holiday, tent, lighting).resolve_scale(color="independent"),
        )
        .resolve_scale(color="independent")
        .properties(
            title=charts._title(
                text=f"Substation card: {site}",
                subtitle=(
                    f"{PROVISIONAL} G-all+D1 (G-all+D1+steps for the steps panel), final cut "
                    "2025-07-01. All vertical axes are points of effective capacity unless "
                    "labelled; a positive term raises net power. Hour bands: night, morning, "
                    "midday, evening (purple, orange, sky, blue). Capacity has no interval: the "
                    "fit gives a point estimate."
                ),
            )
        )
    )


def comparison(*, tables: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """Compare capacity, heating slope, profile peak, and evening-to-midday ratio across sites."""
    capacities = pl.read_parquet(common.WORK_DIR / "gam_capacities_scored.parquet").filter(
        pl.col("arm") == "g_all_d1", pl.col("season_year") == 2025
    )
    weekday = tables["profile"].filter(pl.col("day type") == "weekday")
    rows = []
    for site in ORDER:
        mine = weekday.filter(pl.col("site") == site)
        hours = mine["hour"].to_numpy()
        points = mine["points"].to_numpy()
        evening = float(np.mean(points[hours >= 16]))
        midday = float(np.mean(points[(hours >= 11) & (hours < 16)]))
        windows = (
            tables["heating_windows"]
            .filter(pl.col("site") == site)
            .group_by("window (hours)")
            .agg(pl.col("points per 10 °C").mean())
        )
        slope = float(np.sum(windows["points per 10 °C"].to_numpy()))
        rows += [
            {
                "site": site,
                "parameter": "Unmetered PV capacity, season-year 2025 (MW)",
                "value": float(capacities.filter(pl.col("site") == site)["capacity_mw"][0]),
            },
            {
                "site": site,
                "parameter": "Heating slope: sum of the four windows (points per 10 °C)",
                "value": slope,
            },
            {
                "site": site,
                "parameter": "Weekday profile peak (points)",
                "value": float(np.max(points)),
            },
            {
                "site": site,
                "parameter": "Evening-to-midday ratio of the weekday profile",
                "value": evening / midday,
            },
        ]
    table = pl.DataFrame(rows)
    order = list(dict.fromkeys(table["parameter"].to_list()))
    return (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70, color=ocf.DATA_BLUE)
        .encode(
            y=alt.Y("site:N", sort=list(ORDER), title=None),
            x=alt.X("value:Q", title=None, scale=alt.Scale(zero=False)),
        )
        .properties(width=PANEL_WIDTH + 40, height=110)
        .facet(facet=alt.Facet("parameter:N", sort=order, title=None), columns=2)
        .resolve_scale(x="independent")
        .properties(
            title=charts._title(
                text="Key fitted parameters side by side",
                subtitle=(
                    f"{PROVISIONAL} G-all+D1 at the five decision substations, final cut "
                    "2025-07-01. The heating slope sums the four smoothing-window slopes, each "
                    "the mean over the four hour bands. The profile is the weekday daily "
                    "profile; evening is local 16:00 to 24:00 and midday 11:00 to 16:00."
                ),
            )
        )
    )


def main() -> None:
    """Fit the five decision substations and draw the cards and the comparison."""
    tables = decomposition.build_tables(sites=ORDER)
    for number, site in enumerate(ORDER, start=13):
        charts._save(card(site=site, tables=tables), name=f"figure_{number}_card_{site.lower()}")
    charts._save(comparison(tables=tables), name="figure_18_card_comparison")


if __name__ == "__main__":
    main()
