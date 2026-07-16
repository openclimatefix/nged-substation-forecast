"""Tests for the ``view_forecasts.py`` chart builder (``dashboard.forecast_chart``)."""

from datetime import UTC, datetime, timedelta

import polars as pl
from altair import LayerChart
from dashboard.forecast_chart import (
    PLOT_HISTORY,
    PLOT_HORIZON,
    build_view_forecast_chart,
)

INIT_TIME = datetime(2026, 7, 4, 6, 0, tzinfo=UTC)
"""During British Summer Time, so wall time is UTC+1 — the axis tests below rely on that."""


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


def _actuals() -> pl.LazyFrame:
    times = pl.datetime_range(
        INIT_TIME - PLOT_HISTORY,
        INIT_TIME + timedelta(days=2),
        interval="30m",
        time_zone="UTC",
        eager=True,
    )
    return pl.DataFrame(
        {"time": times, "power": pl.Series(range(len(times)), dtype=pl.Float32)}
    ).lazy()


def _build() -> LayerChart:
    return build_view_forecast_chart(
        _forecasts((0, 1, 2)),
        _actuals(),
        power_fcst_init_time=INIT_TIME,
        units="MW",
        title="Test series — PV — id 1",
        subtitle="Forecast init Sat 04 Jul 2026 06:00 UTC",
    )


def test_chart_layers_ensemble_actuals_and_init_rule() -> None:
    spec = _build().to_dict()
    assert len(spec["layer"]) == 3  # ensemble members, actuals, init-time rule
    ensemble, actuals, rule = spec["layer"]
    assert ensemble["encoding"]["detail"]["field"] == "ensemble_member"
    assert ensemble["encoding"]["y"]["title"] == "Power (MW)"
    assert actuals["encoding"]["y"]["field"] == "power"
    assert rule["mark"]["type"] == "rule"


def test_times_are_rendered_as_europe_london_wall_time() -> None:
    spec = _build().to_dict()
    # 06:00 UTC during BST is 07:00 wall time; the rule layer's single datum carries it, naive.
    (rule_datum,) = spec["datasets"][spec["layer"][2]["data"]["name"]]
    assert rule_datum["valid_time"].startswith("2026-07-04T07:00:00")
    assert "+" not in rule_datum["valid_time"]  # naive — no zone for Vega to re-localise
    assert spec["layer"][0]["encoding"]["x"]["title"] == "Time (Europe/London)"


def test_x_axis_has_3_hourly_ticks_labelled_at_midnight_only() -> None:
    spec = _build().to_dict()
    axis = spec["layer"][0]["encoding"]["x"]["axis"]
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


def test_chart_renders_to_html() -> None:
    html = _build().to_html()
    assert "ensemble_member" in html
