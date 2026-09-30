"""Tests for the leaderboards drawn as one panel per lead day.

Each test is written to fail on the defect it names: a wrong sort order, a wrong panel order or
day assignment, an x range that differs between panels, or a baseline drawn where it was not
scored. The figure tests run on synthetic marks, so no fit is needed.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

_STUDY_DIR = Path(__file__).resolve().parents[3] / "studies" / "nwp_forecast_comparison"
sys.path.insert(0, str(_STUDY_DIR))
sys.path.insert(0, str(_STUDY_DIR.parent / "beam_diffuse_split"))

import leaderboard_by_day as mod  # noqa: E402
import nwp_forecast_charts as charts  # noqa: E402

from studies import charts as study_charts  # noqa: E402

FULL_PRODUCTS = ("Product A", "Product B", "Product C")


def _full(*, rows: list[tuple[str, int, float]]) -> pl.DataFrame:
    """Return `lead_board_rows`-shaped rows (without `n_months`) from (product, day, value)."""
    return pl.DataFrame(
        {
            "product": [product for product, _, _ in rows],
            "day": [day for _, day, _ in rows],
            "value": [value for _, _, value in rows],
            "lower_95": [value - 0.5 for _, _, value in rows],
            "upper_95": [value + 0.5 for _, _, value in rows],
        }
    )


def _short(*, rows: list[tuple[str, int, float, str]]) -> pl.DataFrame:
    """Return `row_set_board_rows`-shaped rows from (product, day, value, kind)."""
    return pl.DataFrame(
        {
            "product": [row[0] for row in rows],
            "day": [row[1] for row in rows],
            "value": [row[3 - 1] for row in rows],
            "lower_95": [row[2] - 0.5 for row in rows],
            "upper_95": [row[2] + 0.5 for row in rows],
            "kind": [row[3] for row in rows],
        }
    )


def _board() -> pl.DataFrame:
    full = _full(
        rows=[
            ("Product A", 0, 9.0),
            ("Product B", 0, 7.0),
            ("Product C", 0, 8.0),
            ("Product A", 1, 10.0),
            ("Product B", 1, 12.0),
            ("Product A", 7, 14.0),
            ("Product B", 7, 13.0),
        ]
    )
    short = _short(
        rows=[
            ("Short X (7 months)", 0, 6.0, "mark"),
            ("Short X (7 months)", 0, 6.5, "ens_same_rows"),
            ("Short Y (11 months)", 0, 5.0, "mark"),
            ("Short Y (11 months)", 0, 5.5, "ens_same_rows"),
            ("Short X (7 months)", 7, 15.0, "mark"),
            ("Short X (7 months)", 7, 15.5, "ens_same_rows"),
            ("Short Z (5 months)", 10, 11.0, "mark"),
            ("Short Z (5 months)", 10, 11.5, "ens_same_rows"),
        ]
    )
    return mod.board_rows(full=full, short=short)


def test_panels_run_from_the_earliest_lead_day_and_skip_days_with_no_mark() -> None:
    assert mod.panel_days(board=_board()) == [0, 1, 7, 10]


def test_a_day_with_only_a_grey_tick_and_no_mark_gets_no_panel() -> None:
    full = _full(rows=[("Product A", 1, 10.0)])
    short = _short(rows=[("Short X (7 months)", 3, 9.0, "ens_same_rows")])

    board = mod.board_rows(full=full, short=short)

    assert mod.panel_days(board=board) == [1]


def test_each_panel_holds_only_its_own_days_rows() -> None:
    rows = mod.day_rows(board=_board(), day=1)

    assert rows["label"].to_list() == ["Product A", "Product B"]
    assert rows["value"].to_list() == [10.0, 12.0]


def test_full_window_rows_are_sorted_with_the_smallest_error_first() -> None:
    rows = mod.day_rows(board=_board(), day=0)

    full = rows.filter(pl.col("window") == "full")
    assert full["label"].to_list() == ["Product B", "Product C", "Product A"]
    assert full["value"].to_list() == [7.0, 8.0, 9.0]


def test_a_short_window_row_with_a_smaller_error_still_sits_below_every_full_window_row() -> None:
    rows = mod.day_rows(board=_board(), day=0)

    assert rows["window"].to_list() == ["full", "full", "full", "header", "short", "short"]
    assert rows["label"].to_list()[3] == mod.SHORT_HEADER


def test_short_window_rows_keep_the_fixed_order_of_the_board_not_their_error_order() -> None:
    rows = mod.day_rows(board=_board(), day=0)

    short = rows.filter(pl.col("window") == "short")
    assert short["label"].to_list() == ["Short X (7 months)", "Short Y (11 months)"]
    assert short["value"].to_list() == [6.0, 5.0]


def test_each_short_window_row_carries_the_ens_mean_fitted_on_its_own_rows() -> None:
    rows = mod.day_rows(board=_board(), day=0)

    short = rows.filter(pl.col("window") == "short")
    assert short["tick"].to_list() == [6.5, 5.5]
    assert rows.filter(pl.col("window") == "full")["tick"].null_count() == 3


def test_a_panel_with_no_short_window_row_has_no_header() -> None:
    rows = mod.day_rows(board=_board(), day=1)

    assert "header" not in rows["window"].to_list()


def test_the_tick_belongs_to_the_days_ens_mean_not_another_days() -> None:
    rows = mod.day_rows(board=_board(), day=7)

    assert rows.filter(pl.col("window") == "short")["tick"].to_list() == [15.5]


def test_the_shared_x_range_holds_every_interval_tick_and_baseline_of_every_panel() -> None:
    board = _board()
    days = mod.panel_days(board=board)
    panels = [mod.day_rows(board=board, day=day) for day in days]
    baselines = [[mod.Baseline(name="Climatology", value=19.2)] for _ in days]

    low, high = mod.shared_x_domain(panels=panels, baselines=baselines)

    assert low == 4.0  # the lowest interval end is 5.0 - 0.5
    assert high == 20.0  # the largest value is the 19.2 baseline
    assert low == int(low)
    assert high == int(high)


def test_the_shared_x_range_is_widened_by_a_grey_tick_beyond_every_interval() -> None:
    full = _full(rows=[("Product A", 1, 10.0)])
    short = _short(
        rows=[
            ("Short X (7 months)", 1, 10.0, "mark"),
            ("Short X (7 months)", 1, 3.2, "ens_same_rows"),
        ]
    )
    board = mod.board_rows(full=full, short=short)

    low, _ = mod.shared_x_domain(panels=[mod.day_rows(board=board, day=1)], baselines=[[]])

    assert low == 3.0


def _losses(*, arms: dict[str, float]) -> pl.DataFrame:
    return pl.DataFrame({"arm": list(arms), "setting": "primary", "value": list(arms.values())})


def _fake_leaderboard(monkeypatch: pytest.MonkeyPatch, *, values: dict[str, float]) -> None:
    def board(*, losses: pl.DataFrame, arms: list[str]) -> pl.DataFrame:
        return pl.DataFrame({"arm": arms, "value": [values[arm] for arm in arms]})

    monkeypatch.setattr(mod, "leaderboard", board)


def test_climatology_is_in_every_panel_and_smart_persistence_only_where_it_was_scored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    losses = _losses(arms={"climatology": 0.14, "smart_persistence_day1": 0.12, "a_day1": 0.1})
    _fake_leaderboard(monkeypatch, values={"climatology": 0.14, "smart_persistence_day1": 0.12})

    day0 = mod.baselines_at_day(losses=losses, day=0)
    day1 = mod.baselines_at_day(losses=losses, day=1)

    assert [baseline.name for baseline in day0] == ["Climatology"]
    assert [baseline.name for baseline in day1] == ["Climatology", "Smart persistence"]


def test_a_baselines_error_is_in_percent_of_capacity_and_the_largest_comes_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    losses = _losses(arms={"climatology": 0.12, "smart_persistence_day1": 0.14})
    _fake_leaderboard(monkeypatch, values={"climatology": 0.12, "smart_persistence_day1": 0.14})

    found = mod.baselines_at_day(losses=losses, day=1)

    assert [(item.name, round(item.value, 6)) for item in found] == [
        ("Smart persistence", 14.0),
        ("Climatology", 12.0),
    ]


def test_a_day_with_no_baseline_arm_draws_no_baseline() -> None:
    assert mod.baselines_at_day(losses=_losses(arms={"a_day1": 0.1}), day=1) == []


def _loaded(*, board_arms: list[str]) -> charts.Loaded:
    losses = pl.DataFrame(
        {
            "arm": board_arms,
            "setting": "primary",
            "time": datetime(2026, 1, 1, tzinfo=UTC),
        }
    )
    return charts.Loaded(
        losses=losses,
        predictions=pl.DataFrame(),
        leaderboard_losses=losses,
        extra_devices={},
        published_arms=frozenset(board_arms),
    )


def _figure(
    monkeypatch: pytest.MonkeyPatch, *, number: int = 4, shifted_days: tuple[int, ...] = ()
) -> dict[str, Any]:
    board = _board()
    full = (
        board.filter(pl.col("window") == "full")
        .select("product", "day", "value", "lower_95", "upper_95")
        .with_columns(n_months=pl.lit(10))
    )
    monkeypatch.setattr(mod, "lead_board_rows", lambda *, losses: full)
    monkeypatch.setattr(mod, "parsed_lead_arms", lambda *, losses: {})
    monkeypatch.setattr(mod, "check_single_device", lambda **_: None)
    monkeypatch.setattr(
        mod,
        "row_set_board_rows",
        lambda *, marks: board.filter(pl.col("window") == "short").drop("window"),
    )
    monkeypatch.setattr(
        mod,
        "baselines_at_day",
        lambda *, losses, day: [mod.Baseline(name="Climatology", value=8.2)],
    )
    monkeypatch.setattr(mod, "scope_text", lambda *, losses, domain: "Scope.")
    chart, _ = mod.by_day_figure(
        loaded=_loaded(board_arms=["a_day1"]),
        domain="solar",
        title="A finding",
        number=number,
        row_set_marks=[charts.RowSetMarks(slug="wn3_mean", losses=pl.DataFrame())],
        shifted_days=shifted_days,
    )
    return chart.to_dict()


def _y_encoding(panel: dict[str, Any]) -> dict[str, Any]:
    """Return a panel's y encoding: the first layer that has one (the grid has none)."""
    return next(layer["encoding"]["y"] for layer in panel["layer"] if "y" in layer["encoding"])


def _panel_titles(spec: dict[str, Any]) -> list[str]:
    return [panel["title"]["text"] for panel in spec["vconcat"]]


def _x_domains(spec: dict[str, Any]) -> list[list[float]]:
    found = []
    for panel in spec["vconcat"]:
        for layer in panel["layer"]:
            x = layer["encoding"].get("x")
            if x is not None and "scale" in x:
                found.append(x["scale"]["domain"])
    return found


def test_panels_are_stacked_day_zero_first_and_titled_by_their_own_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _figure(monkeypatch)

    assert _panel_titles(spec) == ["Day 0 (hindcast)", "Day 1", "Day 7", "Day 10"]


def test_every_panel_uses_the_same_x_range_and_plot_width(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _figure(monkeypatch)

    domains = _x_domains(spec)
    assert domains
    assert all(domain == [4.0, 16.0] for domain in domains)
    assert len({panel["width"] for panel in spec["vconcat"]}) == 1


def test_every_panel_draws_the_same_tick_marks(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _figure(monkeypatch)

    tick_values = {
        tuple(layer["encoding"]["x"]["axis"]["values"])
        for panel in spec["vconcat"]
        for layer in panel["layer"]
        if layer["encoding"].get("x", {}).get("axis") not in (None, {})
        and "values" in layer["encoding"]["x"]["axis"]
    }
    assert len(tick_values) == 1


def test_only_the_bottom_panel_carries_the_axis_title(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _figure(monkeypatch)

    with_title = [
        index
        for index, panel in enumerate(spec["vconcat"])
        if any(layer["encoding"].get("x", {}).get("title") for layer in panel["layer"])
    ]
    assert with_title == [len(spec["vconcat"]) - 1]


def test_the_figure_number_in_the_title_is_the_one_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _figure(monkeypatch, number=7)

    assert spec["title"]["text"][0].startswith("Figure 7: A finding")


def test_each_panels_rows_are_in_the_order_day_rows_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _figure(monkeypatch)

    first = _y_encoding(spec["vconcat"][0])["sort"]
    assert first[1:] == [
        "Product B",
        "Product C",
        "Product A",
        mod.SHORT_HEADER,
        "Short X (7 months)",
        "Short Y (11 months)",
    ]
    assert first[0].strip() == ""


def test_no_data_mark_writes_its_values_into_the_accessibility_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _figure(monkeypatch)

    for panel in spec["vconcat"]:
        for layer in panel["layer"]:
            assert layer["mark"]["aria"] is False


def test_every_dot_has_one_colour_and_short_window_dots_are_hollow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _figure(monkeypatch)

    points = [
        layer["mark"]
        for panel in spec["vconcat"]
        for layer in panel["layer"]
        if layer["mark"]["type"] == "point"
    ]
    assert {mark["color"] for mark in points} == {mod.MARK_COLOUR}
    assert {mark["filled"] for mark in points} == {True, False}


def test_the_baselines_are_dashed(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _figure(monkeypatch)

    dashed = [
        layer["mark"]
        for layer in spec["vconcat"][0]["layer"]
        if layer["mark"]["type"] == "rule" and "strokeDash" in layer["mark"]
    ]
    assert len(dashed) == 1


def test_a_day_with_two_baselines_has_two_blank_rows_above_its_first_product() -> None:
    panel = mod.day_panel(
        rows=mod.day_rows(board=_board(), day=1),
        baselines=[
            mod.Baseline(name="Smart persistence", value=15.0),
            mod.Baseline(name="Climatology", value=14.0),
        ],
        day=1,
        x_domain=(4.0, 20.0),
        x_title=None,
        label_px=150,
    )

    sort = _y_encoding(panel.to_dict())["sort"]
    assert [label.strip() for label in sort[:2]] == ["", ""]
    assert len(set(sort[:2])) == 2
    assert sort[2:] == ["Product A", "Product B"]


def test_the_subtitle_names_the_days_smart_persistence_was_scored() -> None:
    text = " ".join(mod.subtitle_lines(scope="Scope.", smart_days=[0, 1, 2, 3]))

    assert "days 0, 1, 2, 3" in text
    assert "Scope." in text


def test_every_caveat_of_the_old_caption_is_kept_for_the_page() -> None:
    notes = " ".join(
        mod.caveat_notes(domain="solar", figure_numbers={"wn3_groups": 17, "headline": 3})
    )

    for phrase in (
        "slightly fewer hours",
        "hindcast",
        "Solar day 0 omits",
        "Leads are not equal",
        "not ranked against each other",
        "training data",
        "Figure 17",
        "September 2026 holds 10 days",
        "nothing is filled in",
        "Figure 3",
        "99th percentile",
    ):
        assert phrase in notes
    wind = " ".join(
        mod.caveat_notes(domain="wind", figure_numbers={"wn3_groups": 18, "headline": 4})
    )
    assert "drops the hour ending 00:00 UTC" in wind
    assert "mean-vector reference" in wind


# --- Optional sources ---------------------------------------------------------------------------


def _touch(folder: Path, *names: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        pl.DataFrame({"arm": ["x"]}).write_parquet(folder / name)


def test_no_optional_folder_means_no_optional_source(tmp_path: Path) -> None:
    found = mod.optional_sources(data_dir=tmp_path, domain="solar")

    assert (found.extra_dirs, found.blends_folders, found.wn3_folders) == ([], [], [])


def test_the_day_4_folder_is_an_extra_lead_directory_and_the_day_5_folder_a_day_folder(
    tmp_path: Path,
) -> None:
    _touch(tmp_path / mod.OPTIONAL_DAY4_SHARED, "solar_losses.parquet")
    _touch(
        tmp_path / mod.OPTIONAL_DAY5_AIFS_WN3,
        "solar_single_day5_losses.parquet",
        "solar_ens_day5_losses.parquet",
        "solar_wn3_day5_losses.parquet",
    )

    found = mod.optional_sources(data_dir=tmp_path, domain="solar")

    assert found.extra_dirs == [tmp_path / mod.OPTIONAL_DAY4_SHARED]
    assert found.blends_folders == [
        charts.DayFolder(folder=tmp_path / mod.OPTIONAL_DAY5_AIFS_WN3, days=(5,))
    ]
    assert found.wn3_folders == found.blends_folders


def test_a_folder_missing_this_technologys_file_is_left_out(tmp_path: Path) -> None:
    _touch(tmp_path / mod.OPTIONAL_DAY4_SHARED, "solar_losses.parquet")
    _touch(tmp_path / mod.OPTIONAL_DAY5_AIFS_WN3, "solar_wn3_day5_losses.parquet")

    wind = mod.optional_sources(data_dir=tmp_path, domain="wind")
    solar = mod.optional_sources(data_dir=tmp_path, domain="solar")

    assert wind.extra_dirs == []
    assert wind.wn3_folders == []
    assert solar.blends_folders == []
    assert len(solar.wn3_folders) == 1


def test_aifs_needs_both_its_row_sets_files_at_day_5(tmp_path: Path) -> None:
    _touch(tmp_path / mod.OPTIONAL_DAY5_AIFS_WN3, "solar_single_day5_losses.parquet")

    assert mod.optional_sources(data_dir=tmp_path, domain="solar").blends_folders == []


def _per_day_losses(folder: Path, *, domain: str, row_set: str, day: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"arm": [f"{row_set}_day{day}"], "site": ["A"]}).write_parquet(
        folder / f"{domain}_{row_set}_day{day}_losses.parquet"
    )


def test_extra_day_folders_are_stacked_after_the_blends_days(tmp_path: Path) -> None:
    blends, more = tmp_path / "blends", tmp_path / "more"
    for row_set in charts.ROW_SETS:
        for day in charts.BLEND_DAYS:
            _per_day_losses(blends, domain="solar", row_set=row_set, day=day)
        _per_day_losses(more, domain="solar", row_set=row_set, day=5)

    stacked = charts.load_aifs_leads(
        blends_dir=blends,
        domain="solar",
        more_folders=[charts.DayFolder(folder=more, days=(5,))],
    )

    for row_set in charts.ROW_SETS:
        arms = stacked[row_set]["arm"].to_list()
        assert arms == [f"{row_set}_day{day}" for day in (*charts.BLEND_DAYS, 5)]


def test_a_day_folder_is_read_only_for_its_named_days(tmp_path: Path) -> None:
    blends, more = tmp_path / "blends", tmp_path / "more"
    for row_set in charts.ROW_SETS:
        for day in charts.BLEND_DAYS:
            _per_day_losses(blends, domain="solar", row_set=row_set, day=day)
        _per_day_losses(more, domain="solar", row_set=row_set, day=5)
        _per_day_losses(more, domain="solar", row_set=row_set, day=6)

    stacked = charts.load_aifs_leads(
        blends_dir=blends,
        domain="solar",
        more_folders=[charts.DayFolder(folder=more, days=(5,))],
    )

    assert all("day6" not in arm for frame in stacked.values() for arm in frame["arm"])


# --- Grid, axis spans and the label gutter ------------------------------------------------------


def _grid_layers(panel: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a panel's vertical grid layers: the solid rules with no second end."""
    return [
        layer
        for layer in panel["layer"]
        if layer["mark"]["type"] == "rule"
        and "strokeDash" not in layer["mark"]
        and "x2" not in layer["encoding"]
    ]


def _grid_values(spec: dict[str, Any], layer: dict[str, Any]) -> list[float]:
    rows = spec["datasets"][layer["data"]["name"]]
    return [row["x"] for row in rows]


def test_every_grid_line_has_the_width_the_old_leaderboard_gave_its_minor_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _figure(monkeypatch)

    widths = {
        layer["mark"]["strokeWidth"] for panel in spec["vconcat"] for layer in _grid_layers(panel)
    }
    assert widths == {charts.MINOR_GRID_WIDTH_PX}
    assert mod.GRID_WIDTH_PX == charts.MINOR_GRID_WIDTH_PX


def test_each_panel_has_a_grid_line_at_every_whole_number_and_every_half_point_between(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _figure(monkeypatch)

    for panel in spec["vconcat"]:
        layers = _grid_layers(panel)
        assert len(layers) == 2
        minor, major = (_grid_values(spec, layer) for layer in layers)
        assert major == [float(value) for value in range(4, 17)]
        assert minor == [value + 0.5 for value in range(4, 16)]
        assert len({layer["mark"]["color"] for layer in layers}) == 2


def test_a_layer_that_turns_off_the_x_axis_would_hide_every_panels_tick_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _figure(monkeypatch)

    for panel in spec["vconcat"]:
        assert all(
            layer["encoding"]["x"].get("axis")
            for layer in panel["layer"]
            if "x" in layer["encoding"]
        )


def test_the_ticks_are_the_whole_numbers_of_the_range() -> None:
    assert mod.whole_ticks(x_domain=(4.0, 7.0)) == [4.0, 5.0, 6.0, 7.0]


def _panels_for_axes() -> dict[int, pl.DataFrame]:
    low = _full(rows=[("Product A", 0, 6.0), ("Product B", 0, 9.0)])
    high = _full(rows=[("Product A", 10, 14.0), ("Product B", 10, 17.0)])
    board = mod.board_rows(full=pl.concat([low, high]), short=None)
    return {day: mod.day_rows(board=board, day=day) for day in (0, 10)}


def test_with_no_shifted_day_every_panel_shares_one_range() -> None:
    plan = mod.axis_plan(panels=_panels_for_axes(), baselines={0: [], 10: []})

    assert plan.domains == {0: (5.0, 18.0), 10: (5.0, 18.0)}
    assert plan.shift == 0.0


def test_a_shifted_panel_has_the_same_span_as_the_others_and_sits_higher_by_a_whole_shift() -> None:
    plan = mod.axis_plan(panels=_panels_for_axes(), baselines={0: [], 10: []}, shifted_days=[10])

    (low_a, high_a), (low_b, high_b) = plan.domains[0], plan.domains[10]
    assert (low_a, high_a) == (5.0, 10.0)
    assert high_b - low_b == high_a - low_a
    assert plan.shift == low_b - low_a == 8.0
    assert plan.shift == int(plan.shift)


def test_a_shifted_day_wider_than_the_other_panels_span_is_refused() -> None:
    panels = _panels_for_axes()
    wide = _full(rows=[("Product A", 10, 12.0), ("Product B", 10, 30.0)])
    panels[10] = mod.day_rows(board=mod.board_rows(full=wide, short=None), day=10)

    with pytest.raises(ValueError, match="more than the other panels"):
        mod.axis_plan(panels=panels, baselines={0: [], 10: []}, shifted_days=[10])


def test_the_shifted_panels_are_the_only_ones_titled_with_the_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _figure(monkeypatch, shifted_days=(10,))

    titles = _panel_titles(spec)
    assert titles[:3] == ["Day 0 (hindcast)", "Day 1", "Day 7"]
    assert titles[3].startswith("Day 10: x axis shifted up by ")
    assert titles[3].endswith(" points")


def test_the_shifted_panels_share_the_span_in_the_figure_and_keep_whole_number_ticks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _figure(monkeypatch, shifted_days=(10,))

    domains = {
        panel["title"]["text"]: panel["layer"][0]["encoding"]["x"]["scale"]["domain"]
        for panel in spec["vconcat"]
    }
    spans = {title: domain[1] - domain[0] for title, domain in domains.items()}
    assert len(set(spans.values())) == 1
    shifted = spec["vconcat"][-1]["layer"][0]["encoding"]["x"]["scale"]["domain"]
    assert shifted[0] == int(shifted[0])


def test_the_subtitle_says_which_days_use_the_shifted_axis() -> None:
    text = " ".join(
        mod.subtitle_lines(scope="Scope.", smart_days=[1], shift=8.0, shifted_days=[10, 14])
    )

    assert "days 10, 14 use an axis 8 points higher" in text
    assert "All panels share one x axis" not in text


def test_product_names_are_shortened_only_where_the_short_form_is_unambiguous() -> None:
    names = pl.DataFrame(
        {
            "name": [
                "IFS HRES (9 km, Open-Meteo)",
                "ENS control member",
                "WeatherNext 3 mean (7 months)",
                "ENS mean",
                "IFS 0.25°",
            ]
        }
    )

    short = names.select(mod.shorten(expr=pl.col("name")))["name"].to_list()

    assert short == [
        "IFS HRES 9 km",
        "ENS control",
        "WeatherNext 3 (7 months)",
        "ENS mean",
        "IFS 0.25°",
    ]


def test_the_label_column_fits_the_longest_label_and_the_plot_takes_the_rest_of_the_column() -> (
    None
):
    narrow = mod.label_width_px(labels=["ENS mean", "UKV"])
    wide = mod.label_width_px(labels=["ENS mean", "AIFS ENS mean (11 months)"])

    assert narrow < wide
    assert wide >= len("AIFS ENS mean (11 months)") * 6
    width = mod.plot_width_px(label_px=wide)
    assert wide + width + mod.RIGHT_MARGIN_PX == study_charts.CONTENT_WIDTH_PX
    assert mod.plot_width_px(label_px=narrow) > width


def test_the_figure_gives_the_plot_more_width_than_the_old_fixed_label_column_did(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _figure(monkeypatch)

    width = spec["vconcat"][0]["width"]
    assert width > study_charts.PLOT_WIDTH_PX
    extent = _y_encoding(spec["vconcat"][0])["axis"]["minExtent"]
    assert extent + width + mod.RIGHT_MARGIN_PX == study_charts.CONTENT_WIDTH_PX
