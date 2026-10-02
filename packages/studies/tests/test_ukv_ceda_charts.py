import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

_STUDIES_DIR = Path(__file__).resolve().parents[3] / "studies"
sys.path.insert(0, str(_STUDIES_DIR / "ukv_ceda_blends"))
sys.path.insert(0, str(_STUDIES_DIR / "nwp_forecast_comparison"))
sys.path.insert(0, str(_STUDIES_DIR / "beam_diffuse_split"))
sys.path.insert(0, str(_STUDIES_DIR / "weather_downloads"))

import ukv_ceda_blends_charts as charts  # noqa: E402
from nwp_forecast_charts import SITES  # noqa: E402
from nwp_forecast_comparison import DomainType  # noqa: E402


def _intervals(*, domain: DomainType) -> pl.DataFrame:
    records = []
    for day in (1, 2, 3, 4):
        for setting in ("primary", "sensitivity"):
            for contrast, centre in (("p1", -0.002), ("p2", -0.003), ("p2b", -0.0025)):
                records.append(
                    {
                        "domain": domain,
                        "day": day,
                        "setting": setting,
                        "contrast": contrast,
                        "scope": "all rows",
                        "level": 95.0,
                        "difference": centre * (1 if setting == "primary" else 1.2),
                        "lower": centre - 0.002,
                        "upper": centre + 0.002,
                        "n_rows": 1000,
                        "n_months": 21,
                    }
                )
        records.extend(
            {
                "domain": domain,
                "day": day,
                "setting": "primary",
                "contrast": "p1",
                "scope": f"E5 generator {site}",
                "level": 95.0,
                "difference": -0.001,
                "lower": -0.004,
                "upper": 0.002,
                "n_rows": 200,
                "n_months": 21,
            }
            for site in SITES[domain]
        )
    return pl.DataFrame(records)


def test_the_headline_rows_are_in_points_of_capacity_with_the_sensitivity_as_a_second_mark():
    rows = charts.headline_rows(intervals=_intervals(domain="wind"), domain="wind", day=2)

    assert rows["label"].to_list() == [
        "Blend minus padded ENS",
        "Blend minus shuffled UKV-CEDA",
        "Blend minus shuffled UKV-CEDA, second seed",
    ]
    first = rows.row(0, named=True)
    assert first["difference"] == pytest.approx(-0.2)
    assert first["lower_95"] == pytest.approx(-0.4)
    assert first["upper_95"] == pytest.approx(0.0)
    assert first["second_difference"] == pytest.approx(-0.24)
    assert rows["planned"].all()


def test_the_headline_draws_one_panel_per_lead_day_under_a_self_contained_caption():
    chart = charts.headline(intervals=_intervals(domain="solar"), domain="solar")

    spec = chart.to_dict()
    text = json.dumps(spec)
    caption = " ".join([*spec["title"]["text"], *spec["title"]["subtitle"]])
    assert len(spec["vconcat"]) == 4
    for day in (1, 2, 3, 4):
        assert f"Lead day {day}" in text
    assert "Figure 1:" in caption
    assert "Negative means the blend forecasts better" in caption
    assert "3 hours fresher than ENS's at every hour" in caption
    assert "Dot: primary setting. Hollow triangle: sensitivity setting." in caption
    assert "the six solar farms" in caption


def test_every_lead_day_panel_shares_one_axis_that_includes_zero():
    intervals = _intervals(domain="wind")
    rows = [charts.headline_rows(intervals=intervals, domain="wind", day=d) for d in (1, 2, 3, 4)]

    low, high = charts.x_domain_of(rows=rows)

    assert low < -0.4
    assert high > 0.0
    assert low <= float(min(r["lower_95"].min() for r in rows))
    assert high >= float(max(r["upper_95"].max() for r in rows))


def test_the_generator_rows_carry_only_anonymised_labels_in_label_order():
    rows = charts.generator_rows(intervals=_intervals(domain="wind"), domain="wind", day=3)

    assert rows["label"].to_list() == ["Generator W1", "Generator W2", "Generator W3"]
    assert rows["difference"].to_list() == pytest.approx([-0.1, -0.1, -0.1])


def test_a_generator_that_is_not_an_anonymised_label_is_refused():
    intervals = _intervals(domain="wind").with_columns(
        scope=pl.col("scope").str.replace("W1", "Real name", literal=True)
    )

    with pytest.raises(ValueError, match="not anonymised labels"):
        charts.generator_rows(intervals=intervals, domain="wind", day=1)


def test_the_generator_figure_is_exploratory_and_says_its_interval_is_within_one_generator():
    chart = charts.generators(intervals=_intervals(domain="solar"), domain="solar")

    spec = chart.to_dict()
    caption = " ".join(spec["title"]["subtitle"])

    assert "All rows are exploratory." in caption
    assert "does not cover differences between generators" in caption
    text = json.dumps(spec)
    assert "Generator A" in text
    assert "Generator F" in text


def _series(*, domain: DomainType) -> pl.DataFrame:
    """Hourly measured and forecast fractions for every generator across the three eras."""
    starts = [
        datetime(2025, 3, 3, tzinfo=UTC),
        datetime(2025, 11, 3, tzinfo=UTC),
        datetime(2026, 4, 6, tzinfo=UTC),
    ]
    rows = []
    for start in starts:
        for site in SITES[domain]:
            for hour in range(14 * 24):
                time = start + timedelta(hours=hour)
                value = 0.5 + 0.4 * np.sin(hour / 7.0) * (1 + 0.2 * (hour // 24 % 3))
                rows.append(
                    {"site": site, "time": time, "measured": value, "forecast": value * 0.9}
                )
    return pl.DataFrame(rows)


def test_one_week_is_chosen_in_each_era_from_measured_output_alone():
    series = _series(domain="wind")

    weeks = charts.era_weeks(series=series, domain="wind")

    assert set(weeks) == {0, 1, 2}
    assert weeks[0].month in (3, 4)
    assert weeks[1].month in (11, 12)
    assert weeks[2].year == 2026
    assert all(week.weekday() == 0 for week in weeks.values())


def test_an_era_with_no_fully_covered_week_is_left_out():
    series = _series(domain="solar").filter(pl.col("time") < datetime(2025, 11, 1, tzinfo=UTC))

    assert set(charts.era_weeks(series=series, domain="solar")) == {0}


def test_a_week_figure_counts_days_and_carries_no_calendar_date():
    series = _series(domain="wind")
    week = charts.era_weeks(series=series, domain="wind")[0]

    chart = charts.week_figure(series=series, week=week, domain="wind", letter="5a")

    text = json.dumps(chart.to_dict())
    assert "Day of the chosen week" in text
    assert "'Day ' + (datum.value + 0.5)" in text
    assert "Figure 5a:" in text
    assert not charts.DATE_IN_SVG.search(text)
    for year in ("2025", "2026"):
        assert year not in text
    assert text.count('"aria": false') >= 3


def test_the_week_figures_name_each_weeks_month_and_year_for_the_pages_text(
    monkeypatch: pytest.MonkeyPatch,
):
    series = _series(domain="wind")
    monkeypatch.setattr(charts, "measured_and_forecast", lambda **_: series)
    losses = series.select("site").with_columns(x=pl.lit(1))

    figures = charts.weeks(losses=losses, predictions=losses, domain="wind")

    assert set(figures) == {0, 1, 2}
    months = [month for _, month in figures.values()]
    assert all(len(month.split()) == 2 and month.split()[1].isdigit() for month in months)


def test_a_week_figure_refuses_a_site_label_that_is_not_anonymised():
    losses = pl.DataFrame({"site": ["Real name"], "x": [1]})

    with pytest.raises(ValueError, match="not anonymised labels"):
        charts.weeks(losses=losses, predictions=losses, domain="wind")


def test_an_svg_with_a_calendar_date_is_refused(tmp_path: Path):
    clean = tmp_path / "clean.svg"
    clean.write_text("<svg>Day 1</svg>")
    dated = tmp_path / "dated.svg"
    dated.write_text("<svg><title>2025-03-04</title></svg>")

    charts.check_no_dates(path=clean)
    with pytest.raises(ValueError, match="calendar date"):
        charts.check_no_dates(path=dated)


def test_a_figure_is_written_once_as_svg(tmp_path: Path):
    chart = charts.headline(intervals=_intervals(domain="wind"), domain="wind")
    path = tmp_path / "figures" / "wind_headline.svg"

    charts.write_figure(chart=chart, path=path, svgo=False)

    assert path.read_text().startswith("<svg")
    charts.check_no_dates(path=path)
    with pytest.raises(FileExistsError):
        charts.write_figure(chart=chart, path=path, svgo=False)
