"""Tests for the ``view_forecasts.py`` chart builder (``dashboard.forecast_chart``)."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import polars as pl
from altair import LayerChart
from dashboard.forecast_chart import (
    LAG_OPTIONS,
    PLOT_HISTORY,
    PLOT_HORIZON,
    build_view_forecast_chart,
)
from plotting import ocf_theme

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
    assert actuals["encoding"]["y"]["field"] == "power"
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


def test_lag_colours_assigned_in_ascending_lag_order() -> None:
    spec = _build(lags=tuple(LAG_OPTIONS.values())).to_dict()
    colour = spec["layer"][2]["encoding"]["color"]
    assert colour["scale"]["domain"] == ["7-day lag", "14-day lag"]
    assert colour["scale"]["range"] == [ocf_theme.PURPLE, ocf_theme.SPRING_GREEN]


def test_actuals_line_ignores_the_deep_history_loaded_for_lags() -> None:
    spec = _build(history=PLOT_HISTORY + max(LAG_OPTIONS.values())).to_dict()
    actuals_rows = spec["datasets"][spec["layer"][2]["data"]["name"]]
    # No lags requested → 4 layers, and the blue line still starts at the window, not at the
    # start of the deeper history the dashboard now always loads.
    assert len(spec["layer"]) == 4
    assert min(row["valid_time"] for row in actuals_rows) == "2026-07-03T07:00:00"


def test_chart_renders_to_html() -> None:
    html = _build().to_html()
    assert "ensemble_member" in html
