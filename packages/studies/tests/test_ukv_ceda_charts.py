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


def _record(
    *,
    domain: DomainType,
    day: int,
    setting: str,
    contrast: str,
    scope: str,
    difference: float,
    lower: float,
    upper: float,
    level: float = 95.0,
) -> dict[str, object]:
    return {
        "domain": domain,
        "day": day,
        "setting": setting,
        "contrast": contrast,
        "scope": scope,
        "level": level,
        "difference": difference,
        "lower": lower,
        "upper": upper,
        "n_rows": 1000,
        "n_months": 21,
    }


def _intervals(*, domain: DomainType) -> pl.DataFrame:
    """Day 1 and day 3 meet the planned rule, day 2's second-seed control reaches zero, and day 4's
    P1 includes zero. P1 survives the Bonferroni correction at both settings at days 1 and 2."""
    records = []
    for day in (1, 2, 3, 4):
        for setting in ("primary", "sensitivity"):
            scale = 1 if setting == "primary" else 1.2
            p1 = (-0.001, -0.004, 0.002) if day == 4 else (-0.003, -0.005, -0.001)
            p2b_upper = 0.001 if day == 2 else -0.001
            wide_upper = 0.001 if (day == 3 and setting == "sensitivity") or day == 4 else -0.0005
            for contrast, (centre, lower, upper) in (
                ("p1", p1),
                ("p2", (-0.004, -0.006, -0.002)),
                ("p2b", (-0.003, -0.007, p2b_upper)),
            ):
                records.append(
                    _record(
                        domain=domain,
                        day=day,
                        setting=setting,
                        contrast=contrast,
                        scope="all rows",
                        difference=centre * scale,
                        lower=lower,
                        upper=upper,
                    )
                )
            records.append(
                _record(
                    domain=domain,
                    day=day,
                    setting=setting,
                    contrast="p1",
                    scope="Bonferroni",
                    difference=p1[0],
                    lower=-0.007,
                    upper=wide_upper,
                    level=99.375,
                )
            )
            for role, level in (
                ("_pad", 0.080),
                ("", 0.077),
                ("_control", 0.081),
                ("_control_b", 0.079),
            ):
                error = level + 0.01 * (day - 1) + (0.001 if setting == "sensitivity" else 0.0)
                records.append(
                    _record(
                        domain=domain,
                        day=day,
                        setting=setting,
                        contrast="error",
                        scope=f"blend_ukv_ceda_day{day}{role}",
                        difference=error,
                        lower=error - 0.005,
                        upper=error + 0.005,
                    )
                )
        records.extend(
            _record(
                domain=domain,
                day=day,
                setting="primary",
                contrast="generator_error",
                scope=f"generator error {site}: blend_ukv_ceda_day{day}{role}",
                difference=0.08 + 0.01 * index + (0.0 if role == "" else 0.002) + 0.01 * (day - 1),
                lower=float("nan"),
                upper=float("nan"),
            )
            for index, site in enumerate(SITES[domain])
            for role in ("_pad", "", "_control")
        )
        records.extend(
            _record(
                domain=domain,
                day=day,
                setting="primary",
                contrast="p1",
                scope=f"E5 generator {site}",
                difference=-0.001,
                lower=-0.004,
                upper=0.002,
            )
            for site in SITES[domain]
        )
    return pl.DataFrame(records)


def test_the_headline_rows_are_in_points_of_capacity_with_each_setting_as_its_own_interval():
    rows = charts.headline_rows(intervals=_intervals(domain="wind"), domain="wind", day=2)

    assert rows["label"].to_list() == [
        "Blend minus padded ENS",
        "Blend minus padded ENS",
        "Blend minus shuffled UKV-CEDA",
        "Blend minus shuffled UKV-CEDA",
        "Blend minus shuffled UKV-CEDA, second seed",
        "Blend minus shuffled UKV-CEDA, second seed",
    ]
    assert rows["condition"].to_list() == ["Primary setting", "Sensitivity setting"] * 3
    primary, sensitivity = rows.row(0, named=True), rows.row(1, named=True)
    assert primary["difference"] == pytest.approx(-0.3)
    assert primary["lower_95"] == pytest.approx(-0.5)
    assert primary["upper_95"] == pytest.approx(-0.1)
    assert sensitivity["difference"] == pytest.approx(-0.36)
    assert rows["planned"].all()
    assert "Bonferroni" not in rows["label"].to_list()


def test_a_day_reads_the_planned_rule_from_the_saved_intervals():
    intervals = _intervals(domain="solar")

    readings = [
        charts.day_reading(intervals=intervals, domain="solar", day=d) for d in (1, 2, 3, 4)
    ]

    assert readings == [
        "lowers the error at day 1",
        charts.fit.UNRESOLVED_LOWER,
        "lowers the error at day 3",
        "no detectable difference",
    ]


def test_days_are_named_in_words_with_the_serial_comma():
    assert charts.days_text(days=[4]) == "day 4"
    assert charts.days_text(days=[1, 2]) == "days 1 and 2"
    assert charts.days_text(days=[1, 2, 3]) == "days 1, 2, and 3"


def test_the_headline_draws_one_panel_per_lead_day_under_a_title_that_states_the_finding():
    chart = charts.headline(intervals=_intervals(domain="solar"), domain="solar")

    spec = chart.to_dict()
    text = json.dumps(spec)
    caption = " ".join([*spec["title"]["text"], *spec["title"]["subtitle"]])
    assert len(spec["vconcat"]) == 4
    assert "Figure 1: " + charts.TITLES["solar"] in " ".join(spec["title"]["text"])
    assert "planned rule met" not in " ".join(spec["title"]["text"])
    assert "no detectable difference" not in caption
    assert "Largest gain P1's interval does not exclude" in caption
    assert "day 4: primary " in caption
    assert "Lead day 1: planned rule met" in text
    assert "Lead day 2: unresolved, control test not passed" in text
    assert "Lead day 4: inconclusive" in text
    assert "Negative means the blend forecasts better" in caption
    assert "3 hours fresher than ENS's at every hour" in caption
    assert "Filled dot: primary setting. Lighter hollow mark: sensitivity setting." in caption
    assert "P1 stays below zero at both settings at days 1 and 2." in caption
    assert "The six solar farms, up to " in caption


def test_every_lead_day_panel_shares_one_axis_that_includes_zero():
    intervals = _intervals(domain="wind")
    rows = [charts.headline_rows(intervals=intervals, domain="wind", day=d) for d in (1, 2, 3, 4)]

    low, high = charts.x_domain_of(rows=rows)

    assert low < -0.7
    assert high > 0.0
    assert low <= min(float(np.min(r["lower_95"].to_numpy())) for r in rows)
    assert high >= max(float(np.max(r["upper_95"].to_numpy())) for r in rows)


def test_a_title_says_no_lead_day_survives_when_the_bonferroni_interval_reaches_zero_everywhere():
    intervals = _intervals(domain="wind").with_columns(
        upper=pl.when(pl.col("scope") == "Bonferroni").then(0.001).otherwise(pl.col("upper"))
    )

    assert "at no lead day" in charts.bonferroni_note(intervals=intervals, domain="wind")


def test_the_error_rows_hold_each_arms_own_primary_error_in_a_fixed_arm_order():
    rows = charts.arm_error_rows(intervals=_intervals(domain="wind"), domain="wind", day=2)

    assert rows["role"].to_list() == ["_pad", "", "_control", "_control_b"]
    assert rows["label"].to_list() == list(charts.ARM_LABELS.values())
    assert rows["value"].to_list() == pytest.approx([9.0, 8.7, 9.1, 8.9])


def test_the_error_figure_is_one_panel_of_lead_day_rows_under_a_title_naming_the_errors():
    chart = charts.errors(intervals=_intervals(domain="wind"), domain="wind")

    spec = chart.to_dict()
    title = " ".join(spec["title"]["text"])
    assert "Figure 6:" in title
    assert "ENS's mean alone has a mean absolute error of 8.0% of capacity at lead day 1" in title
    assert "11.0% at lead day 4" in title
    assert "differ by at most 0.40 points at any lead day" in title
    text = json.dumps(spec)
    for day in (1, 2, 3, 4):
        assert f"Lead day {day}" in text
    assert "smaller is better" in text
    assert "lowerBound" not in text.lower()
    assert '"sort": ["Lead day 1", "Lead day 2", "Lead day 3", "Lead day 4"]' in text


def test_the_generator_error_rows_hold_the_padded_arm_and_the_blend_at_each_generator():
    rows = charts.generator_error_rows(intervals=_intervals(domain="wind"), domain="wind", day=2)

    assert (
        rows["row"].to_list() == ["Generator W1"] * 2 + ["Generator W2"] * 2 + ["Generator W3"] * 2
    )
    assert rows["role"].to_list() == ["_pad", ""] * 3
    assert rows["value"].to_list()[:2] == pytest.approx([9.2, 9.0])


def test_the_generator_error_figure_states_the_range_of_the_blends_error_in_its_title():
    chart = charts.generator_errors(intervals=_intervals(domain="wind"), domain="wind")

    spec = chart.to_dict()
    title = " ".join(spec["title"]["text"])
    assert len(spec["vconcat"]) == 4
    assert "Figure 8:" in title
    assert "runs from 8.0% to 10.0% of capacity at lead day 1" in title
    assert "from 11.0% to 13.0% at lead day 4" in title
    assert "Generator W3" in json.dumps(spec)


def test_a_generator_error_row_with_a_label_that_is_not_anonymised_is_refused():
    intervals = _intervals(domain="wind").with_columns(
        scope=pl.col("scope").str.replace("error W1", "error Real name", literal=True)
    )

    with pytest.raises(ValueError, match="not anonymised labels"):
        charts.generator_error_rows(intervals=intervals, domain="wind", day=1)


def test_the_generator_title_names_the_days_every_estimate_is_below_zero_and_their_range():
    intervals = _intervals(domain="solar")
    assert "below padded ENS's at days 1, 2, 3, and 4, by 0.10 to 0.10 points" in (
        charts.generators_title(intervals=intervals, domain="solar")
    )
    mixed = intervals.with_columns(
        difference=pl.when((pl.col("day") == 3) & (pl.col("scope") == "E5 generator A"))
        .then(0.002)
        .otherwise(pl.col("difference"))
    )

    title = charts.generators_title(intervals=mixed, domain="solar")

    assert "at days 1 and 2," in title
    assert "day 3" not in title
    zero = intervals.with_columns(
        difference=pl.when((pl.col("day") == 2) & (pl.col("scope") == "E5 generator A"))
        .then(0.0)
        .otherwise(pl.col("difference"))
    )
    assert "at day 1," in charts.generators_title(intervals=zero, domain="solar")


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


def test_the_generator_figure_says_once_that_it_is_exploratory_and_its_interval_is_within_one():
    chart = charts.generators(intervals=_intervals(domain="solar"), domain="solar")

    spec = chart.to_dict()
    caption = " ".join(spec["title"]["subtitle"])

    assert "Figure 10: At each of the six solar farms, the blend's point estimate" in " ".join(
        spec["title"]["text"]
    )
    assert caption.count("All rows are exploratory.") == 1
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
    captions = [" ".join(chart.to_dict()["title"]["text"]) for chart, _ in figures.values()]
    assert [c.split(":")[0] for c in captions] == ["Figure 4a", "Figure 4b", "Figure 4c"]
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


def test_the_open_gain_note_gives_the_machine_printed_bound_of_each_inconclusive_day():
    intervals = _intervals(domain="wind")
    p1 = {
        setting: charts.contrast_interval(
            intervals=intervals, domain="wind", day=4, setting=setting, contrast="p1"
        )
        for setting in ("primary", "sensitivity")
    }

    note = charts.open_gain_note(intervals=intervals, domain="wind")

    assert note.endswith(f"day 4: {charts.fit.left_open_text(p1=p1)}.")
    assert note.count("day ") == 1
    clear = intervals.with_columns(
        upper=pl.when((pl.col("day") == 4) & (pl.col("contrast") == "p1"))
        .then(-0.001)
        .otherwise(pl.col("upper"))
    )
    assert charts.open_gain_note(intervals=clear, domain="wind") == ""


def test_the_hand_written_titles_scope_their_claims_and_call_day_four_inconclusive():
    for title in charts.TITLES.values():
        assert "every check" not in title
        assert "no detectable difference" not in title
        assert "day 4 is inconclusive" in title
    assert "planned rule is met at day 3 only" in charts.TITLES["solar"]
    for check in (
        "both hyperparameter settings",
        "both shuffled controls",
        "Bonferroni correction",
        "any one month dropped",
    ):
        assert check in charts.TITLES["wind"]


def test_each_headline_carries_its_own_technologys_title():
    for domain in ("solar", "wind"):
        chart = charts.headline(intervals=_intervals(domain=domain), domain=domain)

        assert charts.TITLES[domain] in " ".join(chart.to_dict()["title"]["text"])


# --- the post hoc figures -------------------------------------------------------------------------


def _post_hoc_intervals() -> pl.DataFrame:
    records = []
    for day, p1, rank, p_value in ((1, -0.0016, 1, 1 / 18), (2, -0.0008, 4, 4 / 18)):
        draws = {
            0: 0.0004,
            1000: -0.0007,
            **{2010 + 10 * k: 0.0001 * k - 0.0005 for k in range(15)},
        }
        for seed, value in draws.items():
            records.append(
                {
                    **_record(
                        domain="solar",
                        day=day,
                        setting="primary",
                        contrast="permutation_draw",
                        scope=f"post hoc permutation: seed {seed}",
                        difference=value,
                        lower=float("nan"),
                        upper=float("nan"),
                    )
                }
            )
        for contrast, value in (
            ("permutation_p1", p1),
            ("permutation_rank", float(rank)),
            ("permutation_p", p_value),
        ):
            records.append(
                _record(
                    domain="solar",
                    day=day,
                    setting="primary",
                    contrast=contrast,
                    scope="post hoc permutation",
                    difference=value,
                    lower=float("nan"),
                    upper=float("nan"),
                )
            )
    for domain in ("solar", "wind"):
        for day in (1, 2, 3):
            for setting in ("primary", "sensitivity"):
                for code, centre in (
                    ("fresh_p1_same_rows", -0.003),
                    ("older_p1", -0.001),
                    ("older_p2", -0.002),
                    ("older_vs_fresh", 0.002),
                ):
                    records.append(
                        _record(
                            domain=domain,
                            day=day,
                            setting=setting,
                            contrast=code,
                            scope="post hoc older run",
                            difference=centre,
                            lower=centre - 0.001,
                            upper=centre + 0.001,
                        )
                    )
                # The stale and the day-1 split sections hold rows under the same codes, which no
                # older-run figure may read as the older run's own.
                for scope in (
                    "post hoc stale",
                    "post hoc older run, lead 48 hours or less",
                    "post hoc older run, lead beyond 48 hours",
                ):
                    records.extend(
                        _record(
                            domain=domain,
                            day=day,
                            setting=setting,
                            contrast=code,
                            scope=scope,
                            difference=0.5,
                            lower=0.4,
                            upper=0.6,
                        )
                        for code in ("fresh_p1_same_rows", "older_p1", "older_vs_fresh")
                    )
    return pl.DataFrame(records)


def test_the_permutation_values_are_in_points_with_the_rank_and_p_value_of_p1():
    draws, single = charts.permutation_values(intervals=_post_hoc_intervals(), day=2)

    assert len(draws) == 17
    assert single["p1"] == pytest.approx(-0.08)
    assert single["rank"] == 4.0
    assert single["p_value"] == pytest.approx(4 / 18)
    assert min(draws) == pytest.approx(-0.07)  # seed 1000: -0.0007 of capacity
    assert max(draws) == pytest.approx(0.09)  # the last extra seed: +0.0009


def test_the_permutation_figure_titles_each_panel_with_its_rank_and_the_figure_with_the_days():
    chart = charts.permutation_figure(intervals=_post_hoc_intervals())

    text = json.dumps(chart.to_dict()).replace('", "', " ")
    assert "rank 1 of 18, p-value 0.056" in text
    assert "rank 4 of 18, p-value 0.222" in text
    assert "larger than all 17 shuffled controls' at day 1" in text
    assert "cannot go below 0.056" in text
    assert f'"width": {charts.CONTENT_WIDTH_PX - 40}' in text
    assert "Figure 11:" in text


def test_the_older_rows_hold_four_contrasts_at_both_settings_in_points_from_the_whole_older_scope():
    rows = charts.older_rows(intervals=_post_hoc_intervals(), domain="wind", day=2)

    assert rows["label"].to_list() == [
        label for label in charts.OLDER_LABELS.values() for _ in range(2)
    ]
    assert rows["label"].to_list()[-1] == "Older-run blend minus planned blend"
    assert rows.height == 8
    assert set(rows["condition"]) == {"Primary setting", "Sensitivity setting"}
    assert rows["difference"].to_list()[:2] == [pytest.approx(-0.3)] * 2
    assert rows["lower_95"][0] == pytest.approx(-0.4)
    # Rows of the stale section and of the day-1 split carry the same codes under other scopes.
    assert max(abs(value) for value in rows["difference"].to_list()) < 0.5


def test_the_older_figure_has_a_panel_per_lead_day_and_says_what_it_cannot_separate():
    chart = charts.older_figure(intervals=_post_hoc_intervals(), domain="solar")

    text = json.dumps(chart.to_dict()).replace('", "', " ")
    assert text.count("Lead day ") >= 3
    assert "cannot separate the longer lead from the earlier start" in text
    assert "Figure 12:" in text
    assert "Post hoc." in text


def test_the_older_title_states_a_finding_only_when_every_point_estimate_supports_it():
    intervals = _post_hoc_intervals()

    plain = charts.older_title(intervals=intervals, domain="wind")

    # In the fixture the older P1 (-0.001) is above the planned P1 (-0.003) everywhere.
    assert "gains less over padded ENS than the planned blend" in plain
    flipped = intervals.with_columns(
        difference=pl.when(
            (pl.col("contrast") == "older_p1") & (pl.col("day") == 2) & (pl.col("domain") == "wind")
        )
        .then(-0.01)
        .otherwise(pl.col("difference"))
    )
    assert "against the planned blend" in charts.older_title(intervals=flipped, domain="wind")
    assert "gains less" not in charts.older_title(intervals=flipped, domain="wind")


def test_the_week_figures_are_lettered_in_order_even_when_an_era_has_no_week(
    monkeypatch: pytest.MonkeyPatch,
):
    series = _series(domain="solar").filter(pl.col("time") < datetime(2025, 11, 1, tzinfo=UTC))
    later = _series(domain="solar").filter(pl.col("time") >= datetime(2026, 1, 1, tzinfo=UTC))
    series = pl.concat([series, later])
    monkeypatch.setattr(charts, "measured_and_forecast", lambda **_: series)
    losses = series.select("site").with_columns(x=pl.lit(1))

    figures = charts.weeks(losses=losses, predictions=losses, domain="solar")

    assert set(figures) == {0, 2}
    captions = [" ".join(chart.to_dict()["title"]["text"]) for chart, _ in figures.values()]
    assert [c.split(":")[0] for c in captions] == ["Figure 3a", "Figure 3b"]
