"""Chart builders for the ``view_forecasts.py`` dashboard app.

``build_view_forecast_chart`` plots one forecast run for one ``time_series_id``: every forecast
ensemble member as a thin grey line, observed power as a thick blue line, a vertical rule at
``power_fcst_init_time``, a faint shaded band behind each weekend (weekday/weekend structure
dominates demand, so weekends should be findable at a glance), and optional lagged-power lines
(observed power shifted forward by e.g. 7 days — the raw material of the models' power-lag
features). A colour legend above the chart always lists every plottable line — its entry set is
the shared colour scale's explicit domain, so it never changes as lines are toggled on and off.

``build_nwp_ensemble_chart`` plots the NWP ensemble that fed the forecast (one weather variable
at the H3 cell containing the series) as a second panel stacked below the power chart.

The two charts are separate Altair specs, so stacking them relies on identical horizontal
geometry: they share the same pinned x encoding, both pin the y-axis region to ``Y_AXIS_EXTENT``
pixels, and the power chart's legend sits *above* the plot (``orient="top"``) so it consumes no
horizontal space that the legend-less NWP panel wouldn't also lose.

The x-axis deliberately overrides Altair's adaptive datetime ticks: labels sit at
local midnight only (day-of-week first — crucial for demand forecasting), with unlabelled minor
ticks every 3 hours, all in Europe/London wall time.
"""

import calendar
from collections.abc import Sequence
from dataclasses import dataclass
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

Y_AXIS_EXTENT: Final[int] = 56
"""Pinned pixel width of the y-axis region (labels + ticks) on every chart.

The power chart and the NWP subplot are separate stacked charts sharing one container width, so
their x-axes only align vertically if the space to the left of each plot area is identical.
Both charts therefore pin the y-axis extent to this constant instead of letting Vega size it to
the tick labels. Wide enough for the widest expected label (surface pressure, ``100,000``).
"""


@dataclass(frozen=True)
class NwpPlotVariable:
    """Display metadata for one NWP variable offered by the NWP-parameter dropdown."""

    label: str
    """Human-readable name, used in the dropdown, the chart title, and the y-axis title."""

    unit: str
    """Display unit shown on the y-axis — after ``scale`` is applied."""

    scale: float = 1.0
    """Display multiplier applied to the stored values (e.g. kg/m²/s → mm/h for precipitation)."""


NWP_PLOT_VARIABLES: Final[dict[str, NwpPlotVariable]] = {
    "temperature_2m": NwpPlotVariable("2 m temperature", "°C"),
    "dew_point_temperature_2m": NwpPlotVariable("2 m dew-point temperature", "°C"),
    "wind_speed_10m": NwpPlotVariable("10 m wind speed", "m/s"),
    "wind_direction_10m": NwpPlotVariable("10 m wind direction", "°"),
    "wind_speed_100m": NwpPlotVariable("100 m wind speed", "m/s"),
    "wind_direction_100m": NwpPlotVariable("100 m wind direction", "°"),
    "pressure_surface": NwpPlotVariable("Surface pressure", "Pa"),
    "pressure_reduced_to_mean_sea_level": NwpPlotVariable("Mean sea-level pressure", "Pa"),
    "geopotential_height_500hpa": NwpPlotVariable("500 hPa geopotential height", "m"),
    "downward_long_wave_radiation_flux_surface": NwpPlotVariable(
        "Downward long-wave radiation", "W/m²"
    ),
    "downward_short_wave_radiation_flux_surface": NwpPlotVariable(
        "Downward short-wave (solar) radiation", "W/m²"
    ),
    # Stored as kg/m²/s (numerically mm/s); ×3600 displays as mm/h, which forecasters expect
    # and which survives _prepare_for_plot's 3-decimal-place display rounding.
    "precipitation_surface": NwpPlotVariable("Precipitation rate", "mm/h", scale=3600.0),
}
"""The NWP variables the dashboard offers, keyed by ``contracts.weather_schemas.Nwp`` column.

Exactly the *continuous* ``Nwp`` variables (a test guards against drift):
``categorical_precipitation_type_surface`` is excluded because its values are category codes,
which a line chart would render as meaningless slopes.
"""


def _lag_label(lag: timedelta) -> str:
    """The display label for a power lag, used in the UI control and the chart legend alike."""
    return f"{lag.days}-day lagged power"


LAG_OPTIONS: Final[dict[str, timedelta]] = {
    _lag_label(lag): lag for lag in (timedelta(days=7), timedelta(days=14))
}
"""The lagged-power lines the dashboard offers, as UI-label → lag.

Hardcoded, illustrative model inputs: the dashboard cannot see which lag features a given
experiment's model actually used (that config lives in MLflow), so we offer the common
defaults rather than pretending to read them from the model.
"""

_LAG_COLORS: Final[tuple[str, ...]] = (ocf_theme.PURPLE, ocf_theme.SPRING_GREEN)
"""Line colours assigned to lags in ascending-lag order; must cover ``len(LAG_OPTIONS)``."""

_FORECAST_LABEL: Final[str] = "Forecast power"
_ACTUALS_LABEL: Final[str] = "Actual power"
_INIT_TIME_LABEL: Final[str] = "Forecast init time"

_LINE_COLORS: Final[dict[str, str]] = {
    _FORECAST_LABEL: ocf_theme.ENSEMBLE_LINE,
    _ACTUALS_LABEL: ocf_theme.BLUE,
    **dict(zip(LAG_OPTIONS, _LAG_COLORS, strict=True)),
    _INIT_TIME_LABEL: ocf_theme.ORANGE_RED,
}
"""Legend label → line colour for everything the chart can draw, in legend order.

Used as the explicit domain/range of the shared colour scale, so the legend always lists every
entry — whichever lines are currently toggled on — and each line keeps a stable colour.
"""

_MIDNIGHT_TEST: Final[str] = "hours(datum.value) == 0"
"""Vega expression: is this tick at (wall-clock) midnight?

Wall times are serialised as naive timestamps, which Vega parses in the browser's zone — so the
browser-local ``hours()`` recovers the Europe/London wall-clock hour whatever zone the viewer
is in.
"""


def _prepare_for_plot(lf: pl.LazyFrame, time_column: str, value_column: str) -> pl.LazyFrame:
    """Convert times to naive Europe/London wall time and compact values for serialisation.

    Naive (zone-stripped) values render identically in any viewer's browser — Vega would
    otherwise re-localise tz-aware timestamps to the viewer's zone. Values are rounded to 3
    decimal places *as Float64*: a raw ``Float32`` serialises to JSON with ~17 significant
    digits (e.g. ``10.300000190734863``), which roughly triples the inline-data size and pushes
    the ~34k-row ensemble past marimo's max-output-size guard. 3 d.p. display precision is far
    below forecast error (the stored values are already rounded to a 13-bit significand).
    """
    return lf.with_columns(
        pl.col(time_column).dt.convert_time_zone(DISPLAY_TIME_ZONE).dt.replace_time_zone(None),
        pl.col(value_column).cast(pl.Float64).round(3),
    )


def _wall_time(t: datetime) -> datetime:
    """A tz-aware UTC datetime as naive Europe/London wall time (see ``_prepare_for_plot``)."""
    return pl.Series([t]).dt.convert_time_zone(DISPLAY_TIME_ZONE).dt.replace_time_zone(None).item()


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


def _lagged_power_frame(
    actuals: pl.LazyFrame, power_fcst_init_time: datetime, lags: Sequence[timedelta]
) -> pl.LazyFrame:
    """Observed power shifted forward by each lag — the raw material of the power-lag features.

    Only observations at or before ``power_fcst_init_time`` are shifted (the model cannot see
    later ones), so each line ends at ``power_fcst_init_time + lag`` — precisely where feature
    engineering nullifies that lag as leaky for longer lead times. The early end of the line is
    the point, not an artefact.

    Args:
        actuals: ``PowerTimeSeries`` observations (UTC ``time``), including history at least
            ``max(lags)`` before the plotted window starts.
        power_fcst_init_time: The forecast init time (tz-aware UTC).
        lags: The lags to shift by; one output line (``lag`` label) per element.

    Returns:
        A frame of (UTC) ``time``, ``power``, and ``lag`` label rows, clipped to start no
        earlier than the plotted window.
    """
    window_start = power_fcst_init_time - PLOT_HISTORY
    return pl.concat(
        actuals.filter(pl.col("time") <= power_fcst_init_time)
        .with_columns(time=pl.col("time") + lag, lag=pl.lit(_lag_label(lag)))
        .filter(pl.col("time") >= window_start)
        for lag in lags
    )


def _weekend_layer(window_start: datetime, window_end: datetime) -> alt.Chart:
    """The faint weekend-band layer, drawn first so every other layer sits on top.

    No y encoding, so each band spans the full chart height. Uses the shared x encoding — see
    ``_x_encoding``'s docstring for why deviating from it would suppress the merged x-axis.
    """
    return (
        alt.Chart(_weekend_bands(window_start, window_end))
        .mark_rect(color=ocf_theme.MUSTARD, opacity=WEEKEND_SHADE_OPACITY)
        .encode(
            x=_x_encoding(window_start, window_end, field="start"),
            x2="end:T",
        )
    )


def _y_axis() -> alt.Axis:
    """A y-axis with its pixel extent pinned so stacked charts align — see ``Y_AXIS_EXTENT``.

    Like the x encoding, the y-axis definition must be identical on every layer that carries a
    y encoding, or Vega-Lite's axis merge can drop the deviating properties.
    """
    return alt.Axis(minExtent=Y_AXIS_EXTENT, maxExtent=Y_AXIS_EXTENT)


def _x_axis_ticks(window_start: datetime, window_end: datetime) -> list[datetime]:
    """Every 3-hourly wall-clock tick across the (naive, wall-time) plotted window."""
    first_tick = window_start.replace(minute=0, second=0, microsecond=0)
    while first_tick.hour % 3 != 0:
        first_tick += timedelta(hours=1)
    return pl.datetime_range(first_tick, window_end, interval="3h", eager=True).to_list()


def _x_encoding(window_start: datetime, window_end: datetime, field: str = "valid_time") -> alt.X:
    """The shared x encoding: explicit 3-hourly ticks, labels at midnight only.

    Altair's adaptive datetime ticks are hard to read; instead every property is pinned. Labelled
    major ticks (with gridlines) sit at each local midnight, formatted ``Mon 06 Jul``; shorter
    minor ticks every 3 hours carry no label and no gridline. The conditional-axis-property
    dicts are Vega-Lite's native encoding for "major vs minor tick" styling.

    Every layer must use this same encoding (via ``field`` when its time column isn't
    ``valid_time``): the x scale is shared across layers, so Vega-Lite merges the layers' axis
    definitions, and one deviating definition (e.g. ``axis=None``) can suppress the merged axis
    — labels, ticks, and gridlines — for the whole chart.
    """
    tick_size: dict[str, Any] = {"condition": {"test": _MIDNIGHT_TEST, "value": 7}, "value": 3}
    grid_opacity: dict[str, Any] = {"condition": {"test": _MIDNIGHT_TEST, "value": 1}, "value": 0}
    return alt.X(
        f"{field}:T",
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
    shade_weekends: bool = True,
    show_forecast: bool = True,
    show_actuals: bool = True,
    lags: Sequence[timedelta] = (),
) -> alt.LayerChart:
    """Build the single-panel forecast chart for one series and one forecast run.

    Args:
        forecasts: ``PowerForecast`` rows for one ``(time_series_id, power_fcst_init_time)``
            (UTC ``valid_time``); every distinct ``ensemble_member`` becomes a thin grey line.
        actuals: ``PowerTimeSeries`` observations for the same series (UTC ``time``). The thick
            blue line plots the window's observations, including past the init time; rows from
            deeper history (which callers must include when requesting ``lags``) feed only the
            lagged-power lines.
        power_fcst_init_time: The forecast init time (tz-aware UTC). Sets the plotted window —
            ``PLOT_HISTORY`` before it to ``PLOT_HORIZON`` after it — and the vertical rule.
        units: ``"MW"`` or ``"MVA"``, from this series' ``TimeSeriesMetadata``.
        title: Chart title (the series name / type / id line).
        subtitle: Chart subtitle (the init time / experiment line).
        shade_weekends: Whether to draw a faint background band behind each weekend.
        show_forecast: Whether to draw the grey forecast-ensemble lines.
        show_actuals: Whether to draw the thick blue observed-power line.
        lags: Power lags to plot as coloured lines (observed power shifted forward by each lag,
            using only pre-init observations — see ``_lagged_power_frame``). Empty for none.

    Returns:
        A layered, zoomable Altair chart in Europe/London wall time.
    """
    # 51 members × 14 days × 48 half-hours ≈ 34k rows exceeds Altair's 5000-row default guard.
    alt.data_transformers.disable_max_rows()

    init_wall = _wall_time(power_fcst_init_time)
    window_start = init_wall - PLOT_HISTORY
    window_end = init_wall + PLOT_HORIZON
    x = _x_encoding(window_start, window_end)

    # All power layers share the y title (and the pinned _y_axis) so Vega-Lite merges them into
    # one clean y-axis whichever subset of lines is toggled on.
    y_title = f"Power ({units})"

    # The shared colour scale's explicit domain drives the legend, so the legend always lists
    # every line — even ones currently toggled off — and colours never reshuffle. The scale and
    # legend definitions ride on the field-encoded layers (lags and the always-present init-time
    # rule); the single-colour layers reference the same scale via datum encodings.
    # orient="top" keeps the legend out of the plot's horizontal space, so the x-axis aligns
    # with the legend-less NWP panel stacked below (see the module docstring).
    # (Swatch opacity is pinned to 1 in the OCF theme's legend config — a per-legend
    # ``symbolOpacity`` here would lose to the opacity Vega-Lite derives from the marks.)
    line_color_scale = alt.Scale(domain=list(_LINE_COLORS), range=list(_LINE_COLORS.values()))
    line_legend = alt.Legend(title=None, orient="top", symbolType="stroke", symbolStrokeWidth=2)

    # Layers are appended in draw order: weekend bands at the back, then the forecast
    # ensemble, then lagged power (model *inputs*, which must not obscure observations), then
    # observed power, with the init-time rule on top of everything.
    layers: list[alt.Chart] = []
    subtitle_notes: list[str] = []
    if shade_weekends:
        layers.append(_weekend_layer(window_start, window_end))
        subtitle_notes.append("Shaded: weekends")
    if show_forecast:
        layers.append(
            alt.Chart(_prepare_for_plot(forecasts, "valid_time", "power_fcst").collect())
            .mark_line(strokeWidth=1, opacity=0.3)
            .encode(
                x=x,
                y=alt.Y("power_fcst:Q", title=y_title, axis=_y_axis()),
                color=alt.ColorDatum(_FORECAST_LABEL),
                detail="ensemble_member:N",
                tooltip=["ensemble_member", "valid_time", "power_fcst"],
            )
        )
    if lags:
        sorted_lags = sorted(lags)
        layers.append(
            alt.Chart(
                _prepare_for_plot(
                    _lagged_power_frame(actuals, power_fcst_init_time, sorted_lags),
                    "time",
                    "power",
                )
                .rename({"time": "valid_time"})
                .collect()
            )
            .mark_line(strokeWidth=1.5, opacity=0.8)
            .encode(
                x=x,
                y=alt.Y("power:Q", title=y_title, axis=_y_axis()),
                color=alt.Color("lag:N", scale=line_color_scale, legend=line_legend),
                tooltip=["lag", "valid_time", "power"],
            )
        )
    if show_actuals:
        layers.append(
            alt.Chart(
                _prepare_for_plot(
                    actuals.filter(pl.col("time") >= power_fcst_init_time - PLOT_HISTORY),
                    "time",
                    "power",
                )
                .rename({"time": "valid_time"})
                .collect()
            )
            .mark_line(strokeWidth=2.5)
            .encode(
                x=x,
                y=alt.Y("power:Q", title=y_title, axis=_y_axis()),
                color=alt.ColorDatum(_ACTUALS_LABEL),
                tooltip=["valid_time", "power"],
            )
        )
    layers.append(
        alt.Chart(pl.DataFrame({"valid_time": [init_wall], "series": [_INIT_TIME_LABEL]}))
        .mark_rule(strokeWidth=1.5, strokeDash=[6, 4])
        .encode(
            x=x,
            color=alt.Color("series:N", scale=line_color_scale, legend=line_legend),
            tooltip=[alt.Tooltip("valid_time", title="Forecast init time")],
        )
    )
    chart = alt.layer(*layers).properties(
        title=alt.TitleParams(
            text=title,
            subtitle=[subtitle, " · ".join(subtitle_notes)] if subtitle_notes else [subtitle],
        ),
        width="container",
        height=400,
    )
    # .interactive() on a LayerChart returns a LayerChart; the FacetChart half of the union in
    # its annotation only arises for faceted charts.
    return cast(alt.LayerChart, chart.interactive())


def build_nwp_ensemble_chart(
    nwp: pl.LazyFrame,
    *,
    variable: str,
    power_fcst_init_time: datetime,
    nwp_init_time: datetime,
    shade_weekends: bool = True,
) -> alt.LayerChart:
    """Build the NWP-ensemble panel stacked below the power chart, sharing its x-axis.

    Args:
        nwp: ``Nwp`` rows for one ``(nwp_model_id, init_time, h3_index)`` — the run recorded in
            the selected forecast's ``nwp_init_time``, at the H3 cell containing this time
            series (``TimeSeriesMetadata.h3_res_5`` equals ``Nwp.h3_index``; both are resolution
            5). Every distinct ``ensemble_member`` becomes a thin grey line. Native NWP time
            steps (3- or 6-hourly) are drawn as-is, not upsampled.
        variable: The ``Nwp`` column to plot; must be a key of ``NWP_PLOT_VARIABLES``.
        power_fcst_init_time: The *power* forecast init time (tz-aware UTC). Sets the plotted
            window and the dashed rule, matching the power chart above exactly.
        nwp_init_time: When the plotted NWP run was initialised (tz-aware UTC); shown in the
            subtitle. The lines start here — the run has no earlier data — so the panel is
            deliberately empty between the window start and the NWP init time.
        shade_weekends: Whether to draw the same faint weekend bands as the power chart.

    Returns:
        A layered, zoomable Altair chart in Europe/London wall time whose horizontal geometry
        matches ``build_view_forecast_chart``'s (same pinned x encoding and y-axis extent), so
        the two charts' x-axes align when stacked in the dashboard.
    """
    var = NWP_PLOT_VARIABLES[variable]
    init_wall = _wall_time(power_fcst_init_time)
    window_start = init_wall - PLOT_HISTORY
    window_end = init_wall + PLOT_HORIZON
    x = _x_encoding(window_start, window_end)

    # The display scale is applied before _prepare_for_plot so its 3-decimal-place rounding
    # happens in display units (raw precipitation ~1e-4 kg/m²/s would round to zero).
    data = _prepare_for_plot(
        nwp.filter(
            pl.col("valid_time").is_between(
                power_fcst_init_time - PLOT_HISTORY, power_fcst_init_time + PLOT_HORIZON
            )
        )
        .select("valid_time", "ensemble_member", variable)
        .with_columns(pl.col(variable) * var.scale),
        "valid_time",
        variable,
    ).collect()

    layers: list[alt.Chart] = []
    if shade_weekends:
        layers.append(_weekend_layer(window_start, window_end))
    layers.append(
        alt.Chart(data)
        .mark_line(strokeWidth=1, opacity=0.3, color=ocf_theme.ENSEMBLE_LINE)
        .encode(
            x=x,
            # zero=False: weather variables have no meaningful zero baseline, and Vega-Lite's
            # zero default would squash e.g. surface pressure (~101,000 Pa) into a flat sliver
            # at the top of a zero-based axis.
            y=alt.Y(
                f"{variable}:Q",
                title=f"{var.label} ({var.unit})",
                axis=_y_axis(),
                scale=alt.Scale(zero=False),
            ),
            detail="ensemble_member:N",
            tooltip=["ensemble_member", "valid_time", variable],
        )
    )
    # The init-time rule matches the power chart's but takes its colour from the mark: this
    # panel has no legend (the power chart's legend explains the rule), and a colour encoding
    # here would conjure one up.
    layers.append(
        alt.Chart(pl.DataFrame({"valid_time": [init_wall]}))
        .mark_rule(strokeWidth=1.5, strokeDash=[6, 4], color=ocf_theme.ORANGE_RED)
        .encode(x=x, tooltip=[alt.Tooltip("valid_time", title="Forecast init time")])
    )
    chart = alt.layer(*layers).properties(
        title=alt.TitleParams(
            text=f"NWP ensemble — {var.label}",
            subtitle=(
                f"NWP init {nwp_init_time:%a %d %b %Y %H:%M} UTC"
                " · at the H3 cell containing this series"
            ),
        ),
        width="container",
        height=220,
    )
    # See build_view_forecast_chart for why this cast is safe.
    return cast(alt.LayerChart, chart.interactive())
