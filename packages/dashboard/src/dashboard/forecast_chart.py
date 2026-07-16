"""Single-panel ensemble forecast chart for the ``view_forecasts.py`` dashboard app.

Builds an Altair chart of one forecast run for one ``time_series_id``: every NWP ensemble member
as a thin grey line, observed power as a thick blue line, a vertical rule at
``power_fcst_init_time``, and a faint shaded band behind each weekend (weekday/weekend structure
dominates demand, so weekends should be findable at a glance). The x-axis deliberately overrides
Altair's adaptive datetime ticks: labels sit at local midnight only (day-of-week first — crucial
for demand forecasting), with unlabelled minor ticks every 3 hours, all in Europe/London wall
time.
"""

import calendar
from datetime import datetime, timedelta
from typing import Any, Final, cast

import altair as alt
import polars as pl
from plotting import ocf_theme

PLOT_HISTORY: Final[timedelta] = timedelta(hours=24)
"""How far before ``power_fcst_init_time`` the plotted window starts."""

PLOT_HORIZON: Final[timedelta] = timedelta(days=14)
"""How far past ``power_fcst_init_time`` the plotted window ends — the live forecast horizon."""

DISPLAY_TIME_ZONE: Final[str] = "Europe/London"
"""All plotted times are converted to this zone's wall time (demand shape follows the local
clock, so midnight ticks and day-of-week labels must be local, not UTC)."""

WEEKEND_SHADE_OPACITY: Final[float] = 0.07
"""Opacity of the weekend background bands.

At this level, ``ocf_theme.MUSTARD`` over the cream theme background reads as a slightly warmer
cream rather than an orange stripe, so the bands mark weekends without competing with the
low-opacity grey ensemble lines.
"""

_MIDNIGHT_TEST: Final[str] = "hours(datum.value) == 0"
"""Vega expression: is this tick at (wall-clock) midnight?

Wall times are serialised as naive timestamps, which Vega parses in the browser's zone — so the
browser-local ``hours()`` recovers the Europe/London wall-clock hour whatever zone the viewer
is in.
"""


def _prepare_for_plot(lf: pl.LazyFrame, time_column: str, power_column: str) -> pl.LazyFrame:
    """Convert times to naive Europe/London wall time and compact powers for serialisation.

    Naive (zone-stripped) values render identically in any viewer's browser — Vega would
    otherwise re-localise tz-aware timestamps to the viewer's zone. Powers are rounded to 3
    decimal places *as Float64*: a raw ``Float32`` serialises to JSON with ~17 significant
    digits (e.g. ``10.300000190734863``), which roughly triples the inline-data size and pushes
    the ~34k-row ensemble past marimo's max-output-size guard. 1 W display precision is far
    below forecast error (the stored values are already rounded to a 13-bit significand).
    """
    return lf.with_columns(
        pl.col(time_column).dt.convert_time_zone(DISPLAY_TIME_ZONE).dt.replace_time_zone(None),
        pl.col(power_column).cast(pl.Float64).round(3),
    )


def _weekend_bands(window_start: datetime, window_end: datetime) -> pl.DataFrame:
    """Weekend intervals (Saturday 00:00 to Monday 00:00 wall time) overlapping the window.

    Args:
        window_start: Start of the plotted window (naive Europe/London wall time).
        window_end: End of the plotted window (naive wall time).

    Returns:
        One row per weekend, with naive wall-time ``start``/``end`` columns clipped to the
        window (possibly zero rows). Bands sit at wall-clock midnights, matching the labelled
        midnight gridlines.
    """
    # Scan from two days before the window so a window that opens mid-weekend still picks up
    # the enclosing band's Saturday.
    day = (window_start - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
    bands: list[tuple[datetime, datetime]] = []
    while day < window_end:
        if day.weekday() == calendar.SATURDAY:
            band = (max(day, window_start), min(day + timedelta(days=2), window_end))
            if band[0] < band[1]:
                bands.append(band)
        day += timedelta(days=1)
    return pl.DataFrame(
        {"start": [b[0] for b in bands], "end": [b[1] for b in bands]},
        schema={"start": pl.Datetime("us"), "end": pl.Datetime("us")},
    )


def _x_axis_ticks(window_start: datetime, window_end: datetime) -> list[datetime]:
    """Every 3-hourly wall-clock tick across the (naive, wall-time) plotted window."""
    first_tick = window_start.replace(minute=0, second=0, microsecond=0)
    while first_tick.hour % 3 != 0:
        first_tick += timedelta(hours=1)
    return pl.datetime_range(first_tick, window_end, interval="3h", eager=True).to_list()


def _x_encoding(window_start: datetime, window_end: datetime) -> alt.X:
    """The shared x encoding: explicit 3-hourly ticks, labels at midnight only.

    Altair's adaptive datetime ticks are hard to read; instead every property is pinned. Labelled
    major ticks (with gridlines) sit at each local midnight, formatted ``Mon 06 Jul``; shorter
    minor ticks every 3 hours carry no label and no gridline. The conditional-axis-property
    dicts are Vega-Lite's native encoding for "major vs minor tick" styling.
    """
    tick_size: dict[str, Any] = {"condition": {"test": _MIDNIGHT_TEST, "value": 7}, "value": 3}
    grid_opacity: dict[str, Any] = {"condition": {"test": _MIDNIGHT_TEST, "value": 1}, "value": 0}
    return alt.X(
        "valid_time:T",
        title=f"Time ({DISPLAY_TIME_ZONE})",
        scale=alt.Scale(domain=[window_start, window_end]),
        axis=alt.Axis(
            values=_x_axis_ticks(window_start, window_end),
            labelExpr=f"{_MIDNIGHT_TEST} ? timeFormat(datum.value, '%a %d %b') : ''",
            labelAngle=-45,
            labelOverlap=False,
            tickSize=tick_size,
            gridOpacity=grid_opacity,
        ),
    )


def build_view_forecast_chart(
    forecasts: pl.LazyFrame,
    actuals: pl.LazyFrame,
    *,
    power_fcst_init_time: datetime,
    units: str,
    title: str,
    subtitle: str,
) -> alt.LayerChart:
    """Build the single-panel forecast chart for one series and one forecast run.

    Args:
        forecasts: ``PowerForecast`` rows for one ``(time_series_id, power_fcst_init_time)``
            (UTC ``valid_time``); every distinct ``ensemble_member`` becomes a thin grey line.
        actuals: ``PowerTimeSeries`` observations for the same series (UTC ``time``); plotted
            wherever available across the whole window, including past the init time.
        power_fcst_init_time: The forecast init time (tz-aware UTC). Sets the plotted window —
            ``PLOT_HISTORY`` before it to ``PLOT_HORIZON`` after it — and the vertical rule.
        units: ``"MW"`` or ``"MVA"``, from this series' ``TimeSeriesMetadata``.
        title: Chart title (the series name / type / id line).
        subtitle: Chart subtitle (the init time / experiment line).

    Returns:
        A layered, zoomable Altair chart in Europe/London wall time.
    """
    # 51 members × 14 days × 48 half-hours ≈ 34k rows exceeds Altair's 5000-row default guard.
    alt.data_transformers.disable_max_rows()

    init_wall = (
        pl.Series([power_fcst_init_time])
        .dt.convert_time_zone(DISPLAY_TIME_ZONE)
        .dt.replace_time_zone(None)
        .item()
    )
    window_start = init_wall - PLOT_HISTORY
    window_end = init_wall + PLOT_HORIZON
    x = _x_encoding(window_start, window_end)

    # Drawn first so every other layer sits on top; no y encoding, so each band spans the full
    # chart height. The other layers carry the labelled x-axis, hence axis=None here.
    weekend_layer = (
        alt.Chart(_weekend_bands(window_start, window_end))
        .mark_rect(color=ocf_theme.MUSTARD, opacity=WEEKEND_SHADE_OPACITY)
        .encode(
            x=alt.X(
                "start:T",
                scale=alt.Scale(domain=[window_start, window_end]),
                axis=None,
            ),
            x2="end:T",
        )
    )
    ensemble_layer = (
        alt.Chart(_prepare_for_plot(forecasts, "valid_time", "power_fcst").collect())
        .mark_line(strokeWidth=1, opacity=0.3, color=ocf_theme.ENSEMBLE_LINE)
        .encode(
            x=x,
            y=alt.Y("power_fcst:Q", title=f"Power ({units})"),
            detail="ensemble_member:N",
            tooltip=["ensemble_member", "valid_time", "power_fcst"],
        )
    )
    actuals_layer = (
        alt.Chart(
            _prepare_for_plot(actuals, "time", "power").rename({"time": "valid_time"}).collect()
        )
        .mark_line(strokeWidth=2.5, color=ocf_theme.BLUE)
        .encode(
            x=x,
            y=alt.Y("power:Q"),
            tooltip=["valid_time", "power"],
        )
    )
    init_time_rule = (
        alt.Chart(pl.DataFrame({"valid_time": [init_wall]}))
        .mark_rule(strokeWidth=1.5, strokeDash=[6, 4], color=ocf_theme.ORANGE_RED)
        .encode(x=x, tooltip=[alt.Tooltip("valid_time", title="Forecast init time")])
    )

    chart = alt.layer(weekend_layer, ensemble_layer, actuals_layer, init_time_rule).properties(
        title=alt.TitleParams(
            text=title,
            subtitle=[subtitle, "Grey: ensemble members · Blue: observed power · Shaded: weekends"],
        ),
        width="container",
        height=400,
    )
    # .interactive() on a LayerChart returns a LayerChart; the FacetChart half of the union in
    # its annotation only arises for faceted charts.
    return cast(alt.LayerChart, chart.interactive())
