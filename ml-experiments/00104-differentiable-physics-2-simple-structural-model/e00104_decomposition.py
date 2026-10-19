"""Draw what the fitted G-all+D1 convex model learns, one figure per group of additive pieces.

The model's mean forecast is a sum of seven pieces, each a block of the design matrix times its
coefficients: the daily profile, the holiday and Christmas-tent terms, the two day-of-year
harmonics, heating, lighting, solar (the subtracted PV), and drift. The script refits G-all+D1 (and
G-all+D1+steps, for the steps panel) at three decision substations with the tuned penalties and the
final cut, splits the mean forecast of the scored year into the seven pieces, checks that the
pieces add up to `predict_gam`, and draws four figures into `draft/`: the calendar pieces, the
weather pieces, the solar and drift pieces, and the sum of the pieces in a summer week and a winter
week. The figures show means only, so the spread plays no part.

    POLARS_MAX_THREADS=4 uv run python ml-experiments/00104-*/e00104_decomposition.py
"""

import json
from datetime import UTC, date, datetime, timedelta
from typing import Final

import altair as alt
import e00104_charts as charts
import e00104_common as common
import e00104_gam as gam
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.dp_calendar import (
    BAND_NAMES,
    CHRISTMAS_WINDOW_DAYS,
    DAY_TYPES,
    HALF_HOURS_PER_DAY,
    HEATING_THRESHOLD_C,
    HOLIDAY_PERIODS,
    SMOOTHING_HOURS,
    lighting_ramps,
)
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    N_BANDS,
    GamFit,
    derated_unit_shape,
    design_matrix,
    fit_gam,
    predict_gam,
)

SITES: Final[tuple[str, ...]] = ("D10", "D11", "D14")
"""Three decision substations, all Primary demand series."""
PIECES: Final[tuple[str, ...]] = (
    "Daily profile",
    "Holidays",
    "Seasonal harmonics",
    "Heating",
    "Lighting",
    "Solar",
    "Drift",
)
PIECE_COLOURS: Final[list[str]] = [
    "#333333",
    ocf.BRAND_ORANGE,
    ocf.DATA_GREEN,
    ocf.DATA_BLUE,
    "#C8A400",
    ocf.DATA_SKY,
    ocf.DATA_PURPLE,
]
BAND_COLOURS: Final[list[str]] = [ocf.DATA_PURPLE, ocf.BRAND_ORANGE, ocf.DATA_SKY, ocf.DATA_BLUE]
SUMMER_WEEK: Final[date] = date(2025, 7, 14)
WINTER_WEEK: Final[date] = date(2026, 1, 12)
LIGHTING_DAYS: Final[dict[str, date]] = {
    "Winter day (17 Dec 2025)": date(2025, 12, 17),
    "Summer day (17 Jun 2026)": date(2026, 6, 17),
}
SMALL: Final[int] = (charts.WIDTH - 60) // 3
"""Width of one panel of a three-site row."""


def _fit(*, site: str, arm: gam.Arm, cache: FeatureCache) -> GamFit:
    """Refit an arm at the final cut with the tuned penalties."""
    tuned = json.loads((common.WORK_DIR / "tuned.json").read_text())
    return fit_gam(
        cache=cache,
        site=site,
        specs=[gam.spec_for(arm=arm, tuned=tuned)],
        cut=common.FINAL_CUT,
        window=arm.window,
        season_years=common.SEASON_YEARS,
    )


def piece_contributions(
    *, cache: FeatureCache, fit: GamFit, index: np.ndarray
) -> dict[str, np.ndarray]:
    """Split a fit's mean forecast at the rows `index` into the seven pieces.

    Args:
        cache: The cache.
        fit: A fit with drift.
        index: The grid rows, never at or after the held-out period.

    Returns:
        Each piece's contribution in points, in `PIECES` order. The pieces add up to
        `predict_gam`; the caller checks it.
    """
    layout = fit.layout
    theta = fit.theta[0]
    features = cache.sites[fit.site]
    x = design_matrix(cache=cache, features=features, layout=layout, index=index)
    capacity = layout.slices["capacity"]
    held_year = np.minimum(cache.calendar.season_year[index], fit.last_season_year)
    held = held_year[:, None] >= np.array(layout.season_years)[None, :]
    x[:, capacity] = -derated_unit_shape(features=features, index=index)[:, None] * held

    def part(*names: str) -> np.ndarray:
        return sum(x[:, layout.slices[n]] @ theta[layout.slices[n]] for n in names)

    return {
        "Daily profile": part("profile"),
        "Holidays": part("holiday", "tent"),
        "Seasonal harmonics": part("harmonics"),
        "Heating": part("heating"),
        "Lighting": part("lighting"),
        "Solar": part("capacity"),
        "Drift": part("drift"),
    }


def build_tables() -> dict[str, pl.DataFrame]:  # noqa: PLR0912, PLR0915 (one pass per site)
    """Fit the arms at the three sites and return every table the figures draw.

    Returns:
        Tables named `profile`, `holiday`, `tent`, `harmonics`, `heating_curve`, `heating_series`,
        `lighting`, `capacity`, `solar_week`, `drift`, `steps`, and `weeks`.
    """
    cache = common.load_cache()
    arm_d1 = next(arm for arm in gam.ARMS if arm.name == "g_all_d1")
    arm_steps = next(arm for arm in gam.ARMS if arm.name == "g_all_d1_steps")
    out: dict[str, list[dict]] = {
        key: []
        for key in (
            "profile",
            "holiday",
            "tent",
            "harmonics",
            "heating_curve",
            "heating_series",
            "lighting",
            "capacity",
            "solar_week",
            "drift",
            "steps",
            "weeks",
        )
    }
    temperatures = np.linspace(-4.0, 20.0, 49)
    for site in SITES:
        fit = _fit(site=site, arm=arm_d1, cache=cache)
        layout = fit.layout
        theta = fit.theta[0]
        index = common.scoring_index(
            cache=cache, site=site, after=common.FINAL_CUT, through=common.SCORING_END
        )
        pieces = piece_contributions(cache=cache, fit=fit, index=index)
        total = np.sum([pieces[name] for name in PIECES], axis=0)
        reference = predict_gam(cache=cache, fit=fit, index=index)
        if not np.allclose(total, reference, atol=1e-6):
            msg = f"{site}: the pieces do not add up to predict_gam"
            raise ValueError(msg)
        end = cache.end_times[index].astype("datetime64[us]")
        times = end.astype("datetime64[s]").astype(np.int64)
        actual = cache.sites[site].y[index]
        # 1. Daily profile.
        profile = theta[layout.slices["profile"]].reshape(len(DAY_TYPES), HALF_HOURS_PER_DAY)
        for day_type, name in enumerate(DAY_TYPES):
            for half_hour in range(HALF_HOURS_PER_DAY):
                out["profile"].append(
                    {
                        "site": site,
                        "day type": name.replace("_", " "),
                        "hour": half_hour / 2.0,
                        "points": float(profile[day_type, half_hour]),
                    }
                )
        # 2. Holidays and the Christmas tent.
        holiday = theta[layout.slices["holiday"]].reshape(len(HOLIDAY_PERIODS), N_BANDS)
        for period, name in enumerate(HOLIDAY_PERIODS):
            for band, band_name in enumerate(BAND_NAMES):
                out["holiday"].append(
                    {
                        "site": site,
                        "period": name,
                        "band": band_name,
                        "points": float(holiday[period, band]),
                    }
                )
        tent_coef = theta[layout.slices["tent"]].reshape(2, N_BANDS)
        for days in range(-CHRISTMAS_WINDOW_DAYS, CHRISTMAS_WINDOW_DAYS + 1):
            tent = 1.0 - abs(days) / (CHRISTMAS_WINDOW_DAYS + 1)
            signed = days / (CHRISTMAS_WINDOW_DAYS + 1) * tent
            for band, band_name in enumerate(BAND_NAMES):
                out["tent"].append(
                    {
                        "site": site,
                        "days from Christmas Day": days,
                        "band": band_name,
                        "points": float(tent_coef[0, band] * tent + tent_coef[1, band] * signed),
                    }
                )
        # 3. Day-of-year harmonics.
        harmonic_coef = theta[layout.slices["harmonics"]].reshape(4, N_BANDS)
        for day in range(1, 366, 4):
            angle = 2.0 * np.pi * day / 365.25
            basis = np.array([np.sin(angle), np.cos(angle), np.sin(2 * angle), np.cos(2 * angle)])
            for band, band_name in enumerate(BAND_NAMES):
                out["harmonics"].append(
                    {
                        "site": site,
                        "day of year": day,
                        "band": band_name,
                        "points": float(basis @ harmonic_coef[:, band]),
                    }
                )
        # 4. Heating: steady temperature, so every smoothed temperature equals it.
        heating_coef = theta[layout.slices["heating"]].reshape(len(SMOOTHING_HOURS), N_BANDS)
        for temperature in temperatures:
            hinge = max(HEATING_THRESHOLD_C - temperature, 0.0) / 10.0
            for band, band_name in enumerate(BAND_NAMES):
                out["heating_curve"].append(
                    {
                        "site": site,
                        "temperature (°C)": float(temperature),
                        "band": band_name,
                        "points": float(hinge * heating_coef[:, band].sum()),
                    }
                )
        daily = (
            pl.DataFrame(
                {
                    "date": (end - np.timedelta64(30, "m")).astype("datetime64[D]"),
                    "points": pieces["Heating"],
                }
            )
            .group_by("date")
            .agg(pl.col("points").mean())
            .sort("date")
        )
        out["heating_series"] += [
            {"site": site, "date": d, "points": p} for d, p in daily.iter_rows()
        ]
        # 5. Lighting over a winter day and a summer day.
        lighting_coef = theta[layout.slices["lighting"]]
        local_hours = cache.calendar.half_hour[index] / 2.0
        for label, day in LIGHTING_DAYS.items():
            on_day = cache.calendar.local_date[index] == np.datetime64(day, "D")
            ramps = lighting_ramps(elevation_deg=cache.sites[site].elevation_deg[index][on_day])
            for hour, value in zip(local_hours[on_day], ramps @ lighting_coef, strict=True):
                out["lighting"].append(
                    {"site": site, "day": label, "hour": float(hour), "points": float(value)}
                )
        # 6. Solar: capacity per season-year, and the PV over the sunniest June week.
        for year, points in fit.capacity_points().items():
            out["capacity"].append({"site": site, "season-year": year, "capacity": points})
        june = (cache.calendar.local_date[index] >= np.datetime64("2026-06-01")) & (
            cache.calendar.local_date[index] < np.datetime64("2026-07-01")
        )
        weeks = (
            pl.DataFrame(
                {
                    "week": (cache.calendar.local_date[index][june]).astype("datetime64[D]"),
                    "pv": -pieces["Solar"][june],
                }
            )
            .with_columns(week=pl.col("week").dt.truncate("1w"))
            .group_by("week")
            .agg(pl.col("pv").sum(), pl.len().alias("rows"))
            .filter(pl.col("rows") >= 7 * HALF_HOURS_PER_DAY)
            .sort("pv", descending=True)
        )
        sunny = weeks["week"][0]
        in_week = (cache.calendar.local_date[index] >= np.datetime64(sunny)) & (
            cache.calendar.local_date[index] < np.datetime64(sunny) + np.timedelta64(7, "D")
        )
        for stamp, value in zip(end[in_week], pieces["Solar"][in_week], strict=True):
            out["solar_week"].append(
                {
                    "site": site,
                    "day": float((stamp - np.datetime64(sunny, "us")) / np.timedelta64(1, "D")),
                    "points": float(value),
                    "week": f"week of {sunny}",
                }
            )
        # 7. Drift of G-all+D1 and the steps of G-all+D1+steps.
        for row in gam.drift_rows(fit=fit, site=site, arm="g_all_d1", cut=common.FINAL_CUT):
            out["drift"].append(
                {
                    "site": site,
                    "band": BAND_NAMES[row["band"]],
                    "date": datetime(1970, 1, 1, tzinfo=UTC) + timedelta(days=row["day"]),
                    "points": row["drift"],
                    "held": False,
                }
            )
        last = [r for r in out["drift"] if r["site"] == site]
        for band_name in BAND_NAMES:
            final = max((r for r in last if r["band"] == band_name), key=lambda r: r["date"])
            out["drift"] += [
                {**final, "held": True},
                {**final, "date": common.SCORING_END, "held": True},
            ]
        steps_fit = _fit(site=site, arm=arm_steps, cache=cache)
        for row in gam.drift_rows(
            fit=steps_fit, site=site, arm="g_all_d1_steps", cut=common.FINAL_CUT
        ):
            out["steps"].append(
                {
                    "site": site,
                    "band": BAND_NAMES[row["band"]],
                    "date": datetime(1970, 1, 1, tzinfo=UTC) + timedelta(days=row["day"]),
                    "points": row["step_level"],
                }
            )
        # The stacked weeks.
        for label, start in (
            ("Summer: 14 to 20 July 2025", SUMMER_WEEK),
            ("Winter: 12 to 18 January 2026", WINTER_WEEK),
        ):
            begin = int(np.datetime64(start, "s").astype(np.int64))
            in_range = (times >= begin) & (times < begin + 7 * 86400)
            day = (times[in_range] - begin) / 86400.0
            for name in PIECES:
                out["weeks"] += [
                    {
                        "site": site,
                        "week": label,
                        "series": name,
                        "day": float(d),
                        "points": float(v),
                    }
                    for d, v in zip(day, pieces[name][in_range], strict=True)
                ]
            out["weeks"] += [
                {
                    "site": site,
                    "week": label,
                    "series": "Sum of the pieces",
                    "day": float(d),
                    "points": float(v),
                }
                for d, v in zip(day, total[in_range], strict=True)
            ]
            out["weeks"] += [
                {
                    "site": site,
                    "week": label,
                    "series": "Observed",
                    "day": float(d),
                    "points": float(v),
                }
                for d, v in zip(day, actual[in_range], strict=True)
            ]
    return {key: pl.DataFrame(rows) for key, rows in out.items()}


def _site_facet(chart: alt.Chart, *, height: int) -> alt.FacetChart:
    """Lay a chart out as one small panel per site, side by side."""
    return chart.properties(width=SMALL, height=height).facet(
        column=alt.Column("site:N", sort=list(SITES), title=None)
    )


def _band_colour() -> alt.Color:
    return alt.Color(
        "band:N",
        sort=list(BAND_NAMES),
        scale=alt.Scale(domain=list(BAND_NAMES), range=BAND_COLOURS),
        legend=alt.Legend(title="Hour band", orient="top", direction="horizontal"),
    )


def figure_calendar(*, tables: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """The daily profile, the holiday terms, the Christmas tent, and the day-of-year harmonics."""
    profile = _site_facet(
        alt.Chart(tables["profile"])  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.2)
        .encode(
            x=alt.X("hour:Q", title="Local hour of day", axis=alt.Axis(values=[0, 6, 12, 18, 24])),
            y=alt.Y("points:Q", title="Points"),
            color=alt.Color(
                "day type:N",
                scale=alt.Scale(
                    domain=["weekday", "saturday", "sunday", "bank holiday"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE, ocf.DATA_GREEN],
                ),
                legend=alt.Legend(title="Day type", orient="top", direction="horizontal"),
            ),
        ),
        height=90,
    ).properties(title="1. Daily profile")
    holiday = _site_facet(
        alt.Chart(tables["holiday"])  # ty: ignore[unresolved-attribute]
        .mark_rect()
        .encode(
            x=alt.X("band:N", sort=list(BAND_NAMES), title=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("period:N", sort=list(HOLIDAY_PERIODS), title=None),
            color=alt.Color(
                "points:Q",
                scale=alt.Scale(scheme="redblue", domainMid=0, reverse=True),
                legend=alt.Legend(title="Points", orient="top", direction="horizontal"),
            ),
        ),
        height=170,
    ).properties(title="2a. Holiday periods: coefficient per hour band")
    tent = _site_facet(
        alt.Chart(tables["tent"])  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.2)
        .encode(
            x=alt.X("days from Christmas Day:Q", title="Days from 25 December"),
            y=alt.Y("points:Q", title="Points"),
            color=_band_colour(),
        ),
        height=80,
    ).properties(title="2b. Christmas tent: contribution within 14 days")
    harmonics = _site_facet(
        alt.Chart(tables["harmonics"])  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.2)
        .encode(
            x=alt.X("day of year:Q", title="Day of year", scale=alt.Scale(domain=[0, 365])),
            y=alt.Y("points:Q", title="Points"),
            color=_band_colour(),
        ),
        height=90,
    ).properties(title="3. Seasonal harmonics: one and two cycles a year")
    return (
        alt.vconcat(profile, holiday, tent, harmonics)
        .resolve_scale(color="independent")
        .properties(
            title=charts._title(
                text="What the model learns: the calendar pieces",
                subtitle=(
                    "Fitted G-all+D1 at three decision substations (D10, D11, D14), final cut "
                    "2025-07-01. Every vertical axis is points of effective capacity; a positive "
                    "term raises net power. Holiday coefficients are added to the daily profile; "
                    "a period with no training day has coefficient zero."
                ),
            )
        )
    )


def figure_weather(*, tables: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """Heating against temperature and over the scored year, and the lighting term."""
    curve = _site_facet(
        alt.Chart(tables["heating_curve"])  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.2)
        .encode(
            x=alt.X("temperature (°C):Q", title="Steady temperature (°C)"),
            y=alt.Y("points:Q", title="Points"),
            color=_band_colour(),
        ),
        height=90,
    ).properties(title="4a. Heating against temperature: the four hinges summed")
    series = _site_facet(
        alt.Chart(tables["heating_series"])  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=0.8, color=ocf.DATA_BLUE)
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%b", tickCount=6)),
            y=alt.Y("points:Q", title="Points"),
        ),
        height=70,
    ).properties(title="4b. Heating over the scored year: daily mean contribution")
    lighting = _site_facet(
        alt.Chart(tables["lighting"])  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.2)
        .encode(
            x=alt.X("hour:Q", title="Local hour of day", axis=alt.Axis(values=[0, 6, 12, 18, 24])),
            y=alt.Y("points:Q", title="Points"),
            color=alt.Color(
                "day:N",
                scale=alt.Scale(
                    domain=list(LIGHTING_DAYS), range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
                ),
                legend=alt.Legend(title="Day", orient="top", direction="horizontal"),
            ),
        ),
        height=80,
    ).properties(title="5. Lighting: contribution across a winter and a summer day")
    return (
        alt.vconcat(curve, series, lighting)
        .resolve_scale(color="independent")
        .properties(
            title=charts._title(
                text="What the model learns: the weather pieces",
                subtitle=(
                    "Fitted G-all+D1, as in the calendar figure. Heating is the sum of four "
                    "hinges of trailing-mean temperature (6, 24, 72, and 168 hours) below 15.5 "
                    "°C, shown for a steady temperature; the time series uses the ERA5 "
                    "temperatures of the scored year. Lighting rises as the sun sinks."
                ),
            )
        )
    )


def figure_solar_and_drift(*, tables: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """Fitted capacity, the subtracted PV in a sunny June week, the drift, and the steps."""
    capacity = _site_facet(
        alt.Chart(tables["capacity"].filter(pl.col("season-year") >= 2019))  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_SKY)
        .encode(
            x=alt.X("season-year:O", title="Season-year (starts in July)"),
            y=alt.Y("capacity:Q", title="Points"),
        ),
        height=80,
    ).properties(title="6a. Fitted solar capacity per season-year")
    week = _site_facet(
        alt.Chart(tables["solar_week"])  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1, color=ocf.DATA_SKY)
        .encode(
            x=alt.X(
                "day:Q",
                title="Days since the start of the week",
                axis=alt.Axis(values=list(range(8))),
            ),
            y=alt.Y("points:Q", title="Points"),
        ),
        height=80,
    ).properties(title="6b. Subtracted PV in the sunniest full June 2026 week")
    drift = _site_facet(
        alt.Chart(tables["drift"])  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1)
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y", tickCount=3)),
            y=alt.Y("points:Q", title="Points"),
            color=_band_colour(),
            strokeDash=alt.StrokeDash(
                "held:N", scale=alt.Scale(domain=[False, True], range=[[1, 0], [4, 3]]), legend=None
            ),
        ),
        height=90,
    ).properties(title="7a. Drift level per hour band; dashed: held through the forecast year")
    steps = _site_facet(
        alt.Chart(tables["steps"])  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1)
        .encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y", tickCount=3)),
            y=alt.Y("points:Q", title="Points"),
            color=_band_colour(),
        ),
        height=60,
    ).properties(title="7b. Cumulative steps of the steps arm (G-all+D1+steps)")
    return (
        alt.vconcat(capacity, week, drift, steps)
        .resolve_scale(color="independent")
        .properties(
            title=charts._title(
                text="What the model learns: the solar, drift, and step pieces",
                subtitle=(
                    "Fitted G-all+D1 (G-all+D1+steps for the steps panel), final cut 2025-07-01. "
                    "Capacity is cumulative over season-years and held at the last trained year. "
                    "The PV term is negative because it is subtracted from demand. The drift "
                    "path is the level the model carries into the forecast year."
                ),
            )
        )
    )


def figure_stacked_weeks(*, tables: dict[str, pl.DataFrame]) -> alt.TopLevelMixin:
    """Every piece, and their sum against observed net power, in a summer and a winter week."""
    weeks = tables["weeks"]
    pieces = weeks.filter(pl.col("series").is_in(list(PIECES)))
    comparison = weeks.filter(pl.col("series").is_in(["Sum of the pieces", "Observed"]))
    x = alt.X(
        "day:Q",
        title="Days since Monday 00:00 UTC",
        axis=alt.Axis(values=list(range(8))),
        scale=alt.Scale(domain=[0, 7]),
    )
    week_names = list(dict.fromkeys(weeks["week"].to_list()))
    pieces_chart = (
        alt.Chart(pieces)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1)
        .encode(
            x=x,
            y=alt.Y("points:Q", title="Points"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=list(PIECES), range=PIECE_COLOURS),
                legend=alt.Legend(title="Piece", orient="top", direction="horizontal", columns=7),
            ),
        )
        .properties(width=SMALL, height=90)
        .facet(
            row=alt.Row(
                "site:N",
                sort=list(SITES),
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left"),
            ),
            column=alt.Column("week:N", sort=week_names, title=None),
        )
    )
    comparison_chart = (
        alt.Chart(comparison)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1)
        .encode(
            x=x,
            y=alt.Y("points:Q", title="Points"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=["Observed", "Sum of the pieces"], range=["#333333", ocf.BRAND_ORANGE]
                ),
                legend=alt.Legend(title="Line", orient="top", direction="horizontal"),
            ),
        )
        .properties(width=SMALL, height=70)
        .facet(
            row=alt.Row(
                "site:N",
                sort=list(SITES),
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left"),
            ),
            column=alt.Column("week:N", sort=week_names, title=None),
        )
    )
    return (
        alt.vconcat(pieces_chart, comparison_chart)
        .resolve_scale(color="independent")
        .properties(
            title=charts._title(
                text="What the model learns: the pieces add up to the forecast",
                subtitle=(
                    "Top: each of the seven pieces of the G-all+D1 mean forecast. Bottom: the "
                    "pieces' sum against observed net power, in points of effective capacity "
                    "(negative is export), for the same weeks as the raw-weeks figure. Gaps are "
                    "half-hours with no reading."
                ),
            )
        )
    )


def main() -> None:
    """Fit, split, and draw the four decomposition figures."""
    tables = build_tables()
    for name, function in (
        ("figure_09_pieces_calendar", figure_calendar),
        ("figure_10_pieces_weather", figure_weather),
        ("figure_11_pieces_solar_drift", figure_solar_and_drift),
        ("figure_12_pieces_stacked_weeks", figure_stacked_weeks),
    ):
        charts._save(function(tables=tables), name=name)


if __name__ == "__main__":
    main()
