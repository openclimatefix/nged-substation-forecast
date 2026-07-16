"""Tests for the ``view_forecasts.py`` chart builders (``dashboard.forecast_chart``)."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import polars as pl
from altair import LayerChart
from contracts.weather_schemas import Nwp
from dashboard.forecast_chart import (
    LAG_OPTIONS,
    NWP_ANALYSIS_LEAD,
    NWP_PLOT_VARIABLES,
    PLOT_HISTORY,
    PLOT_HORIZON,
    Y_AXIS_EXTENT,
    build_nwp_ensemble_chart,
    build_view_forecast_chart,
)
from plotting import ocf_theme

INIT_TIME = datetime(2026, 7, 4, 6, 0, tzinfo=UTC)
"""During British Summer Time, so wall time is UTC+1 — the axis tests below rely on that."""

NWP_INIT_TIME = INIT_TIME - timedelta(hours=6)
"""The NWP run feeding the forecast, published 6 h before the power init time."""


def _forecasts(members: tuple[int, ...]) -> pl.LazyFrame:
    valid_times = pl.datetime_range(
        INIT_TIME,
        INIT_TIME + PLOT_HORIZON,
        interval="30m",
        time_zone="UTC",
        eager=True,
    )
    return pl.DataFrame(
        {
            "valid_time": pl.concat([valid_times for _ in members]),
            "power_fcst": pl.Series(
                [float(m + i) for m in members for i in range(len(valid_times))],
                dtype=pl.Float32,
            ),
            "ensemble_member": pl.Series(
                [m for m in members for _ in range(len(valid_times))], dtype=pl.Int8
            ),
        }
    ).lazy()


def _actuals(history: timedelta = PLOT_HISTORY) -> pl.LazyFrame:
    times = pl.datetime_range(
        INIT_TIME - history,
        INIT_TIME + timedelta(days=2),
        interval="30m",
        time_zone="UTC",
        eager=True,
    )
    return pl.DataFrame(
        {"time": times, "power": pl.Series(range(len(times)), dtype=pl.Float32)}
    ).lazy()


def _build(
    *,
    shade_weekends: bool = True,
    show_forecast: bool = True,
    show_actuals: bool = True,
    lags: Sequence[timedelta] = (),
    history: timedelta = PLOT_HISTORY,
) -> LayerChart:
    return build_view_forecast_chart(
        _forecasts((0, 1, 2)),
        _actuals(history),
        power_fcst_init_time=INIT_TIME,
        units="MW",
        title="Test series — PV — id 1",
        subtitle="Forecast init Sat 04 Jul 2026 06:00 UTC",
        shade_weekends=shade_weekends,
        show_forecast=show_forecast,
        show_actuals=show_actuals,
        lags=lags,
    )


def test_chart_layers_weekends_ensemble_actuals_and_init_rule() -> None:
    spec = _build().to_dict()
    assert len(spec["layer"]) == 4  # weekend bands, ensemble members, actuals, init-time rule
    weekends, ensemble, actuals, rule = spec["layer"]
    assert weekends["mark"]["type"] == "rect"  # drawn first, so every line sits on top of it
    assert weekends["encoding"]["x2"]["field"] == "end"
    # Layers share the x scale, so Vega-Lite merges their axis definitions — one deviating
    # definition (e.g. axis=None) suppresses labels, ticks, and gridlines for the whole chart.
    # Guard that every layer carries the identical axis definition.
    axes = [layer["encoding"]["x"].get("axis") for layer in spec["layer"]]
    assert all(axis == axes[0] for axis in axes)
    assert axes[0] is not None
    assert ensemble["encoding"]["detail"]["field"] == "ensemble_member"
    assert ensemble["encoding"]["y"]["title"] == "Power (MW)"
    # Single-colour layers take their colour from the shared scale via a datum encoding, so
    # they participate in the always-shown legend without a constant column in their data.
    assert ensemble["encoding"]["color"]["datum"] == "Forecast power"
    assert actuals["encoding"]["y"]["field"] == "power"
    assert actuals["encoding"]["color"]["datum"] == "Actual power"
    assert rule["mark"]["type"] == "rule"


def test_shade_weekends_false_omits_the_band_layer() -> None:
    spec = _build(shade_weekends=False).to_dict()
    assert len(spec["layer"]) == 3  # ensemble members, actuals, init-time rule
    assert all(layer["mark"]["type"] != "rect" for layer in spec["layer"])
    assert "weekends" not in " ".join(spec["title"]["subtitle"])


def test_weekend_bands_sit_at_wall_midnights_clipped_to_window() -> None:
    spec = _build().to_dict()
    bands = spec["datasets"][spec["layer"][0]["data"]["name"]]
    # Window is Fri 03 Jul 07:00 → Sat 18 Jul 07:00 wall time: two full weekends plus the
    # opening hours of a third, clipped at the window's end.
    assert [(b["start"], b["end"]) for b in bands] == [
        ("2026-07-04T00:00:00", "2026-07-06T00:00:00"),
        ("2026-07-11T00:00:00", "2026-07-13T00:00:00"),
        ("2026-07-18T00:00:00", "2026-07-18T07:00:00"),
    ]


def test_times_are_rendered_as_europe_london_wall_time() -> None:
    spec = _build().to_dict()
    # 06:00 UTC during BST is 07:00 wall time; the rule layer's single datum carries it, naive.
    (rule_datum,) = spec["datasets"][spec["layer"][3]["data"]["name"]]
    assert rule_datum["valid_time"].startswith("2026-07-04T07:00:00")
    assert "+" not in rule_datum["valid_time"]  # naive — no zone for Vega to re-localise
    assert spec["layer"][1]["encoding"]["x"]["title"] == "Time (Europe/London)"


def test_x_axis_has_3_hourly_ticks_labelled_at_midnight_only() -> None:
    spec = _build().to_dict()
    axis = spec["layer"][1]["encoding"]["x"]["axis"]
    # Altair omits an all-zero time-of-day, so a midnight tick has no "hours" key at all.
    tick_hours = [tick.get("hours", 0) for tick in axis["values"]]
    assert all(hour % 3 == 0 for hour in tick_hours)
    # Window is [init − 24 h, init + 14 d] wall time = 15 days → 15 midnights, 8 ticks per day.
    assert sum(hour == 0 for hour in tick_hours) == 15
    # Labels only at midnight, day-of-week first; minor ticks and gridlines styled conditionally.
    assert axis["labelExpr"].startswith("hours(datum.value) == 0 ? timeFormat")
    assert "%a %d %b" in axis["labelExpr"]
    assert axis["tickSize"]["condition"]["test"] == "hours(datum.value) == 0"
    assert axis["gridOpacity"]["condition"]["test"] == "hours(datum.value) == 0"


def test_lag_lines_end_at_init_plus_lag_and_use_only_pre_init_power() -> None:
    lag = timedelta(days=7)
    spec = _build(lags=(lag,), history=PLOT_HISTORY + max(LAG_OPTIONS.values())).to_dict()
    # Layer order: weekend bands, ensemble, lags, actuals, init rule — inputs must not obscure
    # the observations.
    assert len(spec["layer"]) == 5
    lag_rows = spec["datasets"][spec["layer"][2]["data"]["name"]]
    times = sorted(row["valid_time"] for row in lag_rows)
    # Wall-time window starts Fri 03 Jul 07:00; only pre-init observations are shifted, so the
    # line ends exactly at init + lag (Sat 04 Jul 07:00 wall + 7 d) — where feature engineering
    # nullifies the lag as leaky.
    assert times[0] == "2026-07-03T07:00:00"
    assert times[-1] == "2026-07-11T07:00:00"
    # Shifted values equal the observation lag earlier: both frames index power by row order.
    source = _actuals(PLOT_HISTORY + max(LAG_OPTIONS.values())).collect()
    first_shifted = next(row for row in lag_rows if row["valid_time"] == times[0])
    source_time = datetime(2026, 7, 3, 6, 0, tzinfo=UTC) - lag  # 07:00 wall = 06:00 UTC
    assert first_shifted["power"] == source.filter(pl.col("time") == source_time)["power"].item()


def test_legend_always_lists_every_line() -> None:
    # The always-drawn init-rule layer carries the shared colour scale, whose explicit domain
    # drives the legend — so the legend is identical whichever lines are toggled on.
    for spec in (
        _build().to_dict(),
        _build(show_forecast=False, show_actuals=False).to_dict(),
        _build(lags=tuple(LAG_OPTIONS.values())).to_dict(),
    ):
        colour = spec["layer"][-1]["encoding"]["color"]
        assert colour["field"] == "series"
        assert colour["scale"]["domain"] == [
            "Forecast power",
            "Actual power",
            "7-day lagged power",
            "14-day lagged power",
            "Forecast init time",
        ]
        assert colour["scale"]["range"] == [
            ocf_theme.ENSEMBLE_LINE,
            ocf_theme.BLUE,
            ocf_theme.PURPLE,
            ocf_theme.SPRING_GREEN,
            ocf_theme.ORANGE_RED,
        ]
        assert colour["legend"]["symbolType"] == "stroke"


def test_lag_lines_share_the_legend_colour_scale() -> None:
    spec = _build(lags=tuple(LAG_OPTIONS.values())).to_dict()
    colour = spec["layer"][2]["encoding"]["color"]
    assert colour["field"] == "lag"
    # Identical scale to the rule layer's, so Vega-Lite merges them without conflict and the
    # lag lines pick up their (ascending-lag-ordered) colours from the shared domain.
    assert colour["scale"] == spec["layer"][-1]["encoding"]["color"]["scale"]
    lag_rows = spec["datasets"][spec["layer"][2]["data"]["name"]]
    assert {row["lag"] for row in lag_rows} == {"7-day lagged power", "14-day lagged power"}


def test_actuals_line_ignores_the_deep_history_loaded_for_lags() -> None:
    spec = _build(history=PLOT_HISTORY + max(LAG_OPTIONS.values())).to_dict()
    actuals_rows = spec["datasets"][spec["layer"][2]["data"]["name"]]
    # No lags requested → 4 layers, and the blue line still starts at the window, not at the
    # start of the deeper history the dashboard now always loads.
    assert len(spec["layer"]) == 4
    assert min(row["valid_time"] for row in actuals_rows) == "2026-07-03T07:00:00"


def test_show_flags_toggle_the_forecast_and_actuals_layers() -> None:
    no_forecast = _build(show_forecast=False).to_dict()
    assert len(no_forecast["layer"]) == 3  # weekend bands, actuals, init rule
    no_actuals = _build(show_actuals=False).to_dict()
    assert len(no_actuals["layer"]) == 3  # weekend bands, ensemble, init rule
    assert no_actuals["layer"][1]["encoding"]["color"]["datum"] == "Forecast power"
    only_lags = _build(
        show_forecast=False,
        show_actuals=False,
        lags=(timedelta(days=7),),
        history=PLOT_HISTORY + max(LAG_OPTIONS.values()),
    ).to_dict()
    # weekend bands, lag line, init rule — the y-axis title comes from the lag layer alone.
    assert len(only_lags["layer"]) == 3
    assert only_lags["layer"][1]["encoding"]["y"]["title"] == "Power (MW)"


def _nwp(members: tuple[int, ...] = (0, 1, 2)) -> pl.LazyFrame:
    valid_times = pl.datetime_range(
        NWP_INIT_TIME,
        NWP_INIT_TIME + timedelta(days=15),
        interval="3h",
        time_zone="UTC",
        eager=True,
    )
    return pl.DataFrame(
        {
            "valid_time": pl.concat([valid_times for _ in members]),
            "ensemble_member": pl.Series(
                [m for m in members for _ in range(len(valid_times))], dtype=pl.UInt8
            ),
            "temperature_2m": pl.Series(
                [float(m + i % 24) for m in members for i in range(len(valid_times))],
                dtype=pl.Float32,
            ),
            "precipitation_surface": pl.Series(
                [m * 1e-4 for m in members for _ in range(len(valid_times))], dtype=pl.Float32
            ),
        }
    ).lazy()


def _nwp_analysis(inits: Sequence[datetime]) -> pl.LazyFrame:
    """The first ``NWP_ANALYSIS_LEAD`` of one run per element of ``inits``, control member only
    — what the dashboard feeds ``build_nwp_ensemble_chart``'s ``analysis`` argument. The
    temperature encodes the run's index so tests can see which run each plotted point came from.
    """
    frames = []
    for index, run_init in enumerate(inits):
        valid_times = pl.datetime_range(
            run_init,
            run_init + NWP_ANALYSIS_LEAD,
            interval="3h",
            closed="left",
            time_zone="UTC",
            eager=True,
        )
        frames.append(
            pl.DataFrame(
                {
                    "init_time": [run_init] * len(valid_times),
                    "valid_time": valid_times,
                    "temperature_2m": pl.Series(
                        [float(index)] * len(valid_times), dtype=pl.Float32
                    ),
                    "precipitation_surface": pl.Series(
                        [1e-4 * (index + 1)] * len(valid_times), dtype=pl.Float32
                    ),
                }
            )
        )
    return pl.concat(frames).lazy()


def _build_nwp(
    variable: str = "temperature_2m",
    analysis: pl.LazyFrame | None = None,
    **kwargs: bool,
) -> LayerChart:
    return build_nwp_ensemble_chart(
        _nwp(),
        variable=variable,
        power_fcst_init_time=INIT_TIME,
        nwp_init_time=NWP_INIT_TIME,
        analysis=analysis,
        **kwargs,
    )


def test_nwp_chart_horizontal_geometry_matches_the_power_chart() -> None:
    power = _build().to_dict()
    nwp = _build_nwp().to_dict()
    assert len(nwp["layer"]) == 3  # weekend bands, ensemble members, init rule
    # The two charts are stacked in the dashboard, so their x-axes must align: identical x
    # encoding (same pinned domain, ticks, and labels) and identical pinned y-axis extents.
    power_x = power["layer"][1]["encoding"]["x"]
    nwp_x = nwp["layer"][1]["encoding"]["x"]
    assert nwp_x["axis"] == power_x["axis"]
    assert nwp_x["scale"] == power_x["scale"]
    for spec in (power, nwp):
        y_axis = spec["layer"][1]["encoding"]["y"]["axis"]
        assert y_axis["minExtent"] == y_axis["maxExtent"] == Y_AXIS_EXTENT
        # Both legends must sit above their plot — a right-side legend would eat horizontal
        # plot space and break the vertical alignment of the two stacked x-axes.
        assert spec["layer"][-1]["encoding"]["color"]["legend"]["orient"] == "top"


def test_nwp_chart_plots_each_member_with_the_units_on_the_y_axis() -> None:
    spec = _build_nwp().to_dict()
    ensemble = spec["layer"][1]
    assert ensemble["encoding"]["y"]["title"] == "2 m temperature (°C)"
    assert ensemble["encoding"]["detail"]["field"] == "ensemble_member"
    # The grey takes its colour from the shared scale via a datum encoding, joining the legend.
    assert ensemble["encoding"]["color"]["datum"] == "NWP ensemble"
    # Weather variables have no meaningful zero baseline — a zero-based axis would squash
    # e.g. surface pressure (~101,000 Pa) into a flat sliver.
    assert ensemble["encoding"]["y"]["scale"]["zero"] is False
    rows = spec["datasets"][ensemble["data"]["name"]]
    assert {row["ensemble_member"] for row in rows} == {0, 1, 2}


def test_nwp_plot_variables_cover_exactly_the_continuous_nwp_vars() -> None:
    # Drift guard: adding a variable to the Nwp contract must be mirrored here (or this set
    # comparison fails), and categorical variables must stay excluded.
    assert set(NWP_PLOT_VARIABLES) == Nwp.continuous_var_names()


def test_precipitation_is_displayed_in_mm_per_hour() -> None:
    spec = _build_nwp(variable="precipitation_surface").to_dict()
    assert spec["layer"][1]["encoding"]["y"]["title"] == "Precipitation rate (mm/h)"
    rows = spec["datasets"][spec["layer"][1]["data"]["name"]]
    # Stored kg/m²/s values of 0, 1e-4, and 2e-4 (per member) become mm/h on display.
    assert {row["precipitation_surface"] for row in rows} == {0.0, 0.36, 0.72}


def test_nwp_analysis_is_a_single_blue_line_over_the_ensemble() -> None:
    spec = _build_nwp(
        analysis=_nwp_analysis([NWP_INIT_TIME + timedelta(days=d) for d in range(3)])
    ).to_dict()
    assert len(spec["layer"]) == 4  # weekend bands, ensemble members, analysis line, init rule
    analysis = spec["layer"][2]
    # Thick, and blue via the shared scale — deliberately matching the power panel's
    # observed-truth line.
    assert analysis["encoding"]["color"]["datum"] == "NWP proxy analysis"
    assert analysis["mark"]["strokeWidth"] == 2.5
    # Identical y encoding to the ensemble layer's, so the axis definitions merge cleanly.
    assert analysis["encoding"]["y"] == spec["layer"][1]["encoding"]["y"]
    assert "detail" not in analysis["encoding"]  # one stitched line, not a line per member
    assert any("Proxy analysis" in line for line in spec["title"]["subtitle"])


def test_nwp_analysis_keeps_the_freshest_run_where_runs_overlap() -> None:
    # Two runs 12 h apart, each contributing NWP_ANALYSIS_LEAD (24 h) of rows, so the second
    # run's first 12 h of valid times appear in both — the freshest (shortest-lead) run must
    # win there, collapsing each overlapped valid_time to a single point.
    spec = _build_nwp(
        analysis=_nwp_analysis([NWP_INIT_TIME, NWP_INIT_TIME + timedelta(hours=12)])
    ).to_dict()
    rows = spec["datasets"][spec["layer"][2]["data"]["name"]]
    by_time = {row["valid_time"]: row["temperature_2m"] for row in rows}
    assert len(by_time) == len(rows) == 12  # 8 + 8 rows, 4 of them at shared valid times
    # NWP_INIT_TIME is 04 Jul 00:00 UTC = 01:00 wall time (BST); the runs overlap from 13:00.
    assert by_time["2026-07-04T04:00:00"] == 0.0  # before the overlap: only the first run
    assert by_time["2026-07-04T16:00:00"] == 1.0  # inside the overlap: the second run wins
    assert by_time["2026-07-05T10:00:00"] == 1.0  # after the first run's 24 h end


def test_nwp_analysis_line_is_omitted_when_absent_or_outside_the_window() -> None:
    assert len(_build_nwp().to_dict()["layer"]) == 3  # no analysis passed
    # A run whose whole first 24 h predate the window contributes no rows, so the layer and
    # the subtitle note both vanish rather than advertising a line that isn't drawn.
    stale = _build_nwp(
        analysis=_nwp_analysis([INIT_TIME - PLOT_HISTORY - timedelta(days=2)])
    ).to_dict()
    assert len(stale["layer"]) == 3
    assert not any("Proxy analysis" in line for line in stale["title"]["subtitle"])


def test_nwp_analysis_uses_the_same_display_units_as_the_ensemble() -> None:
    spec = _build_nwp(
        variable="precipitation_surface", analysis=_nwp_analysis([NWP_INIT_TIME])
    ).to_dict()
    rows = spec["datasets"][spec["layer"][2]["data"]["name"]]
    # The stored 1e-4 kg/m²/s becomes 0.36 mm/h — same ×3600 display scale as the grey lines.
    assert {row["precipitation_surface"] for row in rows} == {0.36}


def test_nwp_legend_always_lists_every_line() -> None:
    # Same construction as the power chart's legend: the always-drawn init-rule layer carries
    # the shared colour scale, whose explicit domain drives the legend — so the legend is
    # identical whether or not the analysis line is currently drawn.
    for spec in (
        _build_nwp().to_dict(),
        _build_nwp(analysis=_nwp_analysis([NWP_INIT_TIME])).to_dict(),
    ):
        colour = spec["layer"][-1]["encoding"]["color"]
        assert colour["field"] == "series"
        assert colour["scale"]["domain"] == [
            "NWP ensemble",
            "NWP proxy analysis",
            "Forecast init time",
        ]
        assert colour["scale"]["range"] == [
            ocf_theme.ENSEMBLE_LINE,
            ocf_theme.BLUE,
            ocf_theme.ORANGE_RED,
        ]
        assert colour["legend"]["symbolType"] == "stroke"


def test_nwp_rows_are_clipped_to_the_plotted_window() -> None:
    spec = _build_nwp().to_dict()
    rows = spec["datasets"][spec["layer"][1]["data"]["name"]]
    times = sorted(row["valid_time"] for row in rows)
    # The run starts 6 h before the power init (01:00 wall time) — the panel is empty before
    # that — and the run's final day (out to +15 d) is clipped at the window end.
    assert times[0] == "2026-07-04T01:00:00"
    assert times[-1] == "2026-07-18T07:00:00"


def test_chart_renders_to_html() -> None:
    html = _build().to_html()
    assert "ensemble_member" in html
