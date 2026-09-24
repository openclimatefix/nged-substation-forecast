import re
from pathlib import Path

import plotting.ocf_theme as ocf
import polars as pl
import pytest
from studies.charts import (
    CONDITION_COLOURS,
    CONTENT_WIDTH_PX,
    FAMILY_COLOURS,
    FAMILY_COLOURS_LIGHT,
    LABEL_WIDTH_PX,
    NAMED_SUFFIX,
    PLANNING_NOTES,
    PLOT_WIDTH_PX,
    ContrastKey,
    Panel,
    PlanningType,
    figure,
    flip_contrast,
    interval_panel,
    leaderboard_panel,
    planning,
    report_contrasts,
    report_errors,
    select_contrasts,
    ticks,
)

HEADER = (
    "| Scope | Contrast | ΔMAE (pp of capacity) | 95% interval | Excludes zero? | Folds agreeing "
    "| Rows |\n|---|---|---|---|---|---|---|\n"
)

# Verbatim lines from the two study reports, chosen for the formats each one exercises: a minus
# sign and a hyphen, a `+` sign, asymmetric bounds, `**yes**` and `no`, `4 of 4`, a thousands comma,
# and a scope holding a comma and an en dash.
REPORT = (
    "### Six weather products on 79,384 common site-hours (2022-12-01 to 2026-09-10)\n\n"
    "| Product | Served lead | Global only | Own split |\n"
    "|---|---|---|---|\n"
    "| cams | no forecast step (satellite retrieval) | 5.054 | 4.935 |\n"
    "| era5 | 1 to 12 hours (its own forecasts from 06 and 18 UTC) | 8.975 | 8.898 |\n\n"
    "#### Deciding contrasts, named before the run\n\n"
    + HEADER
    + "| all | cams_global − icon_d2_global | -2.651 | [-2.877, -2.420] | **yes** | 5 of 5 "
    "| 79,384 |\n"
    "| all | icon_eu_global − ukv_global | −0.484 | [−0.664, −0.296] | **yes** | 5 of 5 "
    "| 79,384 |\n\n"
    "#### Implied capacity: month-to-month spread and seasonal swing\n\n"
    "| Product | Month-to-month spread, seasonal cycle removed | Spread minus CAMS's "
    "| December against the annual mean |\n"
    "|---|---|---|---|\n"
    "| cams | 6.5% | +0.0 [+0.0, +0.0] | -23% |\n\n"
    "#### ICON global against ICON-EU, split by ICON global's lead\n\n"
    + HEADER
    + "| ICON global lead 1 to 3 h, equal to ICON-EU's, 07–19 UTC | icon_global_global − "
    "icon_eu_global | +0.022 | [-0.032, +0.075] | no | 4 of 5 | 38,298 |\n"
    "| season winter | cams_global − icon_d2_global | -1.796 | [-2.547, -1.157] | **yes** "
    "| 4 of 4 | 13,249 |\n"
)


@pytest.fixture
def report(tmp_path: Path) -> Path:
    path = tmp_path / "report.md"
    path.write_text(REPORT)
    return path


def test_report_contrasts_reads_verbatim_report_lines_and_skips_other_tables(report: Path) -> None:
    contrasts = report_contrasts(report_path=report)

    assert contrasts.rows() == [
        (
            "Deciding contrasts, named before the run",
            "all",
            "cams_global",
            "icon_d2_global",
            -2.651,
            -2.877,
            -2.42,
            True,
            5,
            5,
            79384,
        ),
        (
            "Deciding contrasts, named before the run",
            "all",
            "icon_eu_global",
            "ukv_global",
            -0.484,
            -0.664,
            -0.296,
            True,
            5,
            5,
            79384,
        ),
        (
            "ICON global against ICON-EU, split by ICON global's lead",
            "ICON global lead 1 to 3 h, equal to ICON-EU's, 07–19 UTC",
            "icon_global_global",
            "icon_eu_global",
            0.022,
            -0.032,
            0.075,
            False,
            4,
            5,
            38298,
        ),
        (
            "ICON global against ICON-EU, split by ICON global's lead",
            "season winter",
            "cams_global",
            "icon_d2_global",
            -1.796,
            -2.547,
            -1.157,
            True,
            4,
            4,
            13249,
        ),
    ]


def test_report_contrasts_raises_on_a_row_it_cannot_read(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    path.write_text(
        HEADER + "| all | cams_global − icon_d2_global | -2.651 | -2.877 to -2.420 "
        "| **yes** | 5 of 5 | 79,384 |\n"
    )

    with pytest.raises(ValueError, match="line 3 is not a contrast row"):
        report_contrasts(report_path=path)


def test_report_contrasts_raises_when_excludes_zero_is_neither_yes_nor_no(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.md"
    path.write_text(
        HEADER + "| all | cams_global − icon_d2_global | -2.651 | [-2.877, -2.420] "
        "| maybe | 5 of 5 | 79,384 |\n"
    )

    with pytest.raises(ValueError, match="line 3 is not a contrast row"):
        report_contrasts(report_path=path)


def test_report_errors_reads_one_column_of_the_first_table(report: Path) -> None:
    assert report_errors(report_path=report, column="Global only") == {"cams": 5.054, "era5": 8.975}


def test_report_errors_raises_on_a_missing_column(report: Path) -> None:
    with pytest.raises(ValueError, match="no column 'All sites'"):
        report_errors(report_path=report, column="All sites")


def test_select_contrasts_returns_rows_in_the_order_asked_for(report: Path) -> None:
    contrasts = report_contrasts(report_path=report)
    section = "Deciding contrasts, named before the run"

    selected = select_contrasts(
        contrasts=contrasts,
        wanted=[
            ContrastKey(section, "all", "icon_eu_global", "ukv_global"),
            ContrastKey(section, "all", "cams_global", "icon_d2_global"),
        ],
    )

    assert selected["difference"].to_list() == [-0.484, -2.651]


def test_select_contrasts_names_every_missing_and_ambiguous_key(report: Path) -> None:
    contrasts = report_contrasts(report_path=report)
    duplicated = pl.concat([contrasts, contrasts.head(1)])
    section = "Deciding contrasts, named before the run"
    ambiguous = ContrastKey(section, "all", "cams_global", "icon_d2_global")
    missing = ContrastKey(section, "post", "cams_global", "icon_d2_global")

    expected = f"{ambiguous} matches 2 rows; {missing} matches 0 rows"
    with pytest.raises(ValueError, match=re.escape(expected)):
        select_contrasts(contrasts=duplicated, wanted=[ambiguous, missing])


@pytest.mark.parametrize("field", ["section", "treatment", "reference"])
def test_select_contrasts_filters_on_each_field(report: Path, field: str) -> None:
    contrasts = report_contrasts(report_path=report)
    real = {
        "section": "Deciding contrasts, named before the run",
        "scope": "all",
        "treatment": "cams_global",
        "reference": "icon_d2_global",
    }
    wrong = real | {field: "not a real value"}
    key = ContrastKey(wrong["section"], wrong["scope"], wrong["treatment"], wrong["reference"])

    with pytest.raises(ValueError, match=re.escape(f"{key} matches 0 rows")):
        select_contrasts(contrasts=contrasts, wanted=[key])


def test_flip_contrast_negates_the_estimate_and_swaps_the_bounds():
    row = pl.DataFrame(
        {
            "treatment": ["icon_eu_global"],
            "reference": ["ukv_global"],
            "difference": [-0.484],
            "lower_95": [-0.664],
            "upper_95": [-0.296],
        }
    )

    flipped = flip_contrast(contrasts=row)

    assert flipped.row(0) == ("ukv_global", "icon_eu_global", 0.484, 0.296, 0.664)


def _rows(families: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "label": [f"row {index}" for index in range(len(families))],
            "family": families,
            "difference": [-1.5, 0.25][: len(families)],
            "lower_95": [-2.0, 0.125][: len(families)],
            "upper_95": [-1.0, 0.5][: len(families)],
        }
    )


def _panel(rows: pl.DataFrame, **overrides: object) -> dict:
    arguments = {
        "rows": rows,
        "x_domain": (-3.0, 1.0),
        "x_title": "points",
        "zero_label": "same as ERA5",
        "better_label": "better than ERA5",
    } | overrides
    return interval_panel(**arguments).to_dict()  # ty: ignore[invalid-argument-type]


def _layer(spec: dict, mark: str) -> list[dict]:
    return [layer for layer in spec["layer"] if layer["mark"]["type"] == mark]


def _values(spec: dict, layer: dict) -> list[dict]:
    return spec["datasets"][layer["data"]["name"]]


def test_the_plotted_numbers_are_the_input_numbers_in_the_given_order():
    rows = _rows(["weather model", "satellite"])
    spec = _panel(rows)
    panel = spec["vconcat"][-1]
    (interval,) = [layer for layer in _layer(panel, "rule") if "x2" in layer["encoding"]]

    plotted = _values(spec, interval)

    assert [(r["label"], r["difference"], r["lower_95"], r["upper_95"]) for r in plotted] == [
        ("row 0", -1.5, -2.0, -1.0),
        ("row 1", 0.25, 0.125, 0.5),
    ]
    assert interval["encoding"]["y"]["sort"] == ["row 0", "row 1"]


def test_colour_follows_the_family_and_the_key_lists_only_families_present():
    spec = _panel(_rows(["reanalysis", "satellite"]))
    key, panel = spec["vconcat"]
    (interval,) = [layer for layer in _layer(panel, "rule") if "x2" in layer["encoding"]]
    colour = interval["encoding"]["color"]
    (text,) = _layer(key, "text")

    assert colour["scale"]["domain"][: len(FAMILY_COLOURS)] == list(FAMILY_COLOURS)
    assert colour["scale"]["range"] == [*FAMILY_COLOURS.values(), *FAMILY_COLOURS_LIGHT.values()]
    assert colour["legend"] is None
    assert [row["label"] for row in _values(spec, text)] == ["satellite", "reanalysis"]
    assert [row["colour"] for row in _values(spec, text)] == [
        FAMILY_COLOURS["satellite"],
        FAMILY_COLOURS["reanalysis"],
    ]


def test_a_panel_of_one_family_draws_no_family_key():
    spec = _panel(_rows(["weather model", "weather model"]))

    assert "vconcat" not in spec
    assert "layer" in spec


def test_the_zero_rule_sits_at_zero():
    spec = _panel(_rows(["satellite"]))
    (rule,) = [layer for layer in _layer(spec, "rule") if "x2" not in layer["encoding"]]

    assert _values(spec, rule) == [{"x": 0.0}]


@pytest.mark.parametrize(
    ("direction", "expected_text", "expected_x"),
    [("negative", "← better than ERA5", -3.0), ("positive", "better than ERA5 →", 1.0)],
)
def test_the_direction_label_points_the_better_way(
    direction: str, expected_text: str, expected_x: float
) -> None:
    spec = _panel(_rows(["satellite"]), better_direction=direction)
    (label,) = [
        layer for layer in _layer(spec, "text") if "better" in layer["encoding"]["text"]["value"]
    ]

    assert label["encoding"]["text"]["value"] == expected_text
    assert _values(spec, label) == [{"x": expected_x}]


def test_ticks_are_round_and_inside_the_range():
    assert ticks(x_domain=(-4.5, 1.0)) == [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0]
    assert ticks(x_domain=(-0.6, 0.4)) == [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4]


def test_ticks_returns_at_most_ten_values():
    assert len(ticks(x_domain=(0.0, 0.45))) == 10


def test_the_difference_is_rounded_to_three_decimal_places():
    rows = pl.DataFrame(
        {
            "label": ["row"],
            "family": ["satellite"],
            "difference": [0.12345],
            "lower_95": [0.0],
            "upper_95": [0.25],
        }
    )
    spec = _panel(rows)
    (interval,) = [layer for layer in _layer(spec, "rule") if "x2" in layer["encoding"]]

    assert _values(spec, interval)[0]["difference"] == 0.123


def _leaderboard_rows(families: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "label": [f"row {index}" for index in range(len(families))],
            "family": families,
            "value": [5.05, 7.71][: len(families)],
            "lower_95": [4.90, 7.50][: len(families)],
            "upper_95": [5.20, 7.90][: len(families)],
        }
    )


def _leaderboard(rows: pl.DataFrame) -> dict:
    return leaderboard_panel(rows=rows, x_domain=(4.0, 9.0), x_title="% of capacity").to_dict()


def test_leaderboard_plots_the_input_numbers_best_first():
    spec = _leaderboard(_leaderboard_rows(["satellite", "weather model"]))
    panel = spec["vconcat"][-1] if "vconcat" in spec else spec
    (point,) = _layer(panel, "point")

    plotted = _values(spec, panel)

    assert [(row["label"], row["value"], row["lower_95"], row["upper_95"]) for row in plotted] == [
        ("row 0", 5.05, 4.9, 5.2),
        ("row 1", 7.71, 7.5, 7.9),
    ]
    assert point["encoding"]["y"]["sort"] == ["row 0", "row 1"]


def test_leaderboard_draws_no_zero_rule_or_direction_label():
    spec = _leaderboard(_leaderboard_rows(["satellite"]))

    assert [layer["mark"]["type"] for layer in spec["layer"]] == ["rule", "point"]


def test_leaderboard_marks_carry_no_aria_text():
    spec = _leaderboard(_leaderboard_rows(["satellite"]))

    assert [layer["mark"]["aria"] for layer in spec["layer"]] == [False, False]


def test_leaderboard_of_one_family_draws_no_family_key():
    spec = _leaderboard(_leaderboard_rows(["weather model", "weather model"]))

    assert "vconcat" not in spec


def test_leaderboard_of_two_families_draws_the_family_key():
    spec = _leaderboard(_leaderboard_rows(["satellite", "weather model"]))
    key, panel = spec["vconcat"]
    (text,) = _layer(key, "text")

    assert [row["label"] for row in _values(spec, text)] == ["satellite", "weather model"]
    assert "layer" in panel


def _conditioned_leaderboard(**overrides: object) -> dict:
    rows = _leaderboard_rows(["satellite", "weather model"]).with_columns(
        condition=pl.Series(["live", "history"]), kind=pl.Series(["blend", "single"])
    )
    arguments = {
        "rows": rows,
        "x_domain": (4.0, 9.0),
        "x_title": "% of capacity",
        "conditions": ("live", "history"),
        "condition_title": "Available",
        "kinds": ("blend", "single"),
        "kind_title": "Kind",
    } | overrides
    return leaderboard_panel(**arguments).to_dict()  # ty: ignore[invalid-argument-type]


def test_leaderboard_conditions_colour_and_fill_the_points_and_replace_the_family_key():
    spec = _conditioned_leaderboard()
    condition_key, kind_key, panel = spec["vconcat"]
    filled, hollow = _layer(panel, "point")

    assert [row["label"] for row in _values(spec, filled)] == ["row 0"]
    assert [row["label"] for row in _values(spec, hollow)] == ["row 1"]
    assert (filled["mark"]["filled"], hollow["mark"]["filled"]) == (True, False)
    assert filled["encoding"]["color"]["field"] == "condition"
    assert filled["encoding"]["color"]["scale"]["range"] == list(CONDITION_COLOURS)
    assert filled["encoding"]["shape"]["field"] == "kind"
    assert [row["label"] for row in _values(spec, _layer(condition_key, "text")[0])] == [
        "live",
        "history",
    ]
    assert [row["label"] for row in _values(spec, _layer(kind_key, "text")[0])] == [
        "blend",
        "single",
    ]


def test_leaderboard_keys_false_draws_no_key():
    spec = _conditioned_leaderboard(keys=False, panel_title="Solar")

    assert "vconcat" not in spec
    assert spec["title"]["text"] == "Solar"


def test_leaderboard_rejects_more_conditions_than_colours():
    with pytest.raises(ValueError, match="too many conditions"):
        _conditioned_leaderboard(conditions=("a", "b", "c"))


def test_family_colours_map_to_the_brand_theme_swatches():
    assert FAMILY_COLOURS == {
        "satellite": ocf.BRAND_ORANGE,
        "reanalysis": ocf.DATA_SKY,
        "weather model": ocf.DATA_BLUE,
        "station observations": ocf.DATA_PURPLE,
    }
    assert FAMILY_COLOURS_LIGHT == {
        "satellite": ocf.BRAND_ORANGE_LIGHT,
        "reanalysis": ocf.DATA_SKY_LIGHT,
        "weather model": ocf.DATA_BLUE_LIGHT,
        "station observations": ocf.DATA_PURPLE_LIGHT,
    }


def test_bold_labels_planned():
    spec = _panel(_rows(["satellite"]))
    (interval,) = [layer for layer in _layer(spec, "rule") if "x2" in layer["encoding"]]
    expr = interval["encoding"]["y"]["axis"]["labelFontWeight"]["expr"]

    assert "(planned)" in expr
    assert "'bold'" in expr


def test_reference_labels_false_omits_the_text_layers():
    spec = _panel(_rows(["satellite"]), reference_labels=False)

    assert _layer(spec, "text") == []


@pytest.mark.parametrize(
    ("direction", "expected_align"), [("negative", "left"), ("positive", "right")]
)
def test_the_zero_label_aligns_away_from_the_better_direction(
    direction: str, expected_align: str
) -> None:
    spec = _panel(_rows(["satellite"]), better_direction=direction)
    (zero_text,) = [
        layer
        for layer in _layer(spec, "text")
        if layer["encoding"]["text"]["value"] == "same as ERA5"
    ]

    assert zero_text["mark"]["align"] == expected_align


def test_the_interval_draws_to_the_upper_bound():
    spec = _panel(_rows(["satellite"]))
    (interval,) = [layer for layer in _layer(spec, "rule") if "x2" in layer["encoding"]]

    assert interval["encoding"]["x2"]["field"] == "upper_95"


def _condition_panel(families: list[str]) -> tuple[dict, dict, dict, dict, dict, dict]:
    rows = pl.DataFrame(
        {
            "label": ["row"] * len(families),
            "family": families,
            "difference": [-1.5, -1.0],
            "lower_95": [-2.0, -1.5],
            "upper_95": [-1.0, -0.5],
            "condition": ["a", "b"],
        }
    )
    spec = _panel(rows, conditions=("a", "b"))
    key, panel = spec["vconcat"][-2:]
    (interval,) = [
        layer
        for layer in panel["layer"]
        if layer["mark"]["type"] == "rule" and "x2" in layer["encoding"]
    ]
    points = [layer for layer in panel["layer"] if layer["mark"]["type"] == "point"]
    filled = next(layer for layer in points if layer["mark"]["filled"])
    hollow = next(layer for layer in points if not layer["mark"]["filled"])
    (key_text,) = [layer for layer in key["layer"] if layer["mark"]["type"] == "text"]
    return spec, interval, filled, hollow, key, key_text


def test_condition_encoding_shades_shapes_and_offsets_a_second_condition():
    spec, interval, filled, hollow, _, key_text = _condition_panel(["satellite", "weather model"])

    assert [row["shade"] for row in _values(spec, interval)] == [
        "satellite",
        "weather model, light",
    ]
    assert [row["shade"] for row in _values(spec, filled)] == ["satellite"]
    assert [row["shade"] for row in _values(spec, hollow)] == ["weather model, light"]
    assert filled["encoding"]["shape"]["scale"]["range"] == ["circle", "diamond"]
    assert "yOffset" in interval["encoding"]
    assert [row["filled"] for row in _values(spec, key_text)] == [True, False]
    assert len(spec["vconcat"]) == 3


def test_a_one_family_panel_colours_its_conditions_and_keeps_their_shapes():
    spec, interval, filled, hollow, key, key_text = _condition_panel(
        ["weather model", "weather model"]
    )
    key_points = [layer for layer in key["layer"] if layer["mark"]["type"] == "point"]

    assert [row["shade"] for row in _values(spec, interval)] == ["a", "b"]
    assert interval["encoding"]["color"]["scale"]["range"] == list(CONDITION_COLOURS)
    assert interval["encoding"]["color"]["legend"] is None
    assert filled["encoding"]["shape"]["scale"]["range"] == ["circle", "diamond"]
    assert hollow["mark"]["filled"] is False
    assert [row["colour"] for layer in key_points for row in _values(spec, layer)] == list(
        CONDITION_COLOURS
    )
    assert [row["label"] for row in _values(spec, key_text)] == ["a", "b"]
    assert len(spec["vconcat"]) == 2


def test_figure_shares_the_colour_scale_across_panels():
    rows_a = pl.DataFrame(
        {
            "label": ["a"],
            "family": ["satellite"],
            "difference": [-1.0],
            "lower_95": [-1.5],
            "upper_95": [-0.5],
        }
    )
    rows_b = pl.DataFrame(
        {
            "label": ["b"],
            "family": ["reanalysis"],
            "difference": [0.2],
            "lower_95": [0.0],
            "upper_95": [0.4],
        }
    )
    panels = [
        interval_panel(
            rows=rows_a, x_domain=(-2.0, 1.0), x_title="x", zero_label="z", better_label="b"
        ),
        interval_panel(
            rows=rows_b, x_domain=(-2.0, 1.0), x_title="x", zero_label="z", better_label="b"
        ),
    ]

    spec = figure(
        panels=panels, number=1, title="t", subtitle=["s"], figure_planning=None
    ).to_dict()

    assert spec["resolve"]["scale"]["color"] == "shared"


@pytest.mark.parametrize(
    ("x_title", "direction", "expected"),
    [
        (
            "Difference (points of capacity)",
            "negative",
            "Difference (points of capacity; more negative means better than ERA5)",
        ),
        ("points", "positive", "points (more positive means better than ERA5)"),
        ("", "negative", ""),
    ],
)
def test_the_axis_title_says_which_direction_is_better(
    x_title: str, direction: str, expected: str
) -> None:
    spec = _panel(_rows(["weather model"]), x_title=x_title, better_direction=direction)
    titles = {
        " ".join(layer["encoding"]["x"]["title"])
        for layer in spec["layer"]
        if "title" in layer["encoding"].get("x", {})
    }

    assert titles == {expected}


def test_figure_stacks_panels_to_fill_the_text_column():
    panels = [_panel_chart(_rows(["satellite", "reanalysis"])) for _ in range(2)]

    spec = figure(
        panels=panels, number=1, title="t", subtitle=["s"], figure_planning=None
    ).to_dict()
    first_panel = spec["vconcat"][0]["vconcat"][-1]
    (interval,) = [
        layer
        for layer in first_panel["layer"]
        if layer["mark"]["type"] == "rule" and "x2" in layer["encoding"]
    ]

    assert len(spec["vconcat"]) == len(panels)
    assert first_panel["width"] == PLOT_WIDTH_PX
    assert LABEL_WIDTH_PX + PLOT_WIDTH_PX + 10 == CONTENT_WIDTH_PX
    assert interval["encoding"]["y"]["axis"]["minExtent"] == LABEL_WIDTH_PX
    assert spec["config"]["legend"]["orient"] == "bottom"


def _panel_chart(rows: pl.DataFrame) -> Panel:
    return interval_panel(
        rows=rows, x_domain=(-3.0, 1.0), x_title="x", zero_label="z", better_label="b"
    )


@pytest.mark.parametrize(
    ("x_domain", "direction", "align"),
    [
        ((-3.0, 1.0), "negative", "left"),
        ((-4.0, 0.5), "negative", "right"),
        ((-1.0, 2.0), "positive", "right"),
        ((-0.2, 2.5), "positive", "left"),
    ],
)
def test_the_zero_label_stays_inside_the_plot(
    x_domain: tuple[float, float], direction: str, align: str
) -> None:
    spec = _panel(_rows(["weather model"]), x_domain=x_domain, better_direction=direction)
    (zero_text,) = [
        layer
        for layer in _layer(spec, "text")
        if layer["encoding"]["text"].get("value") == "same as ERA5"
    ]

    assert zero_text["mark"]["align"] == align


def _planned_rows(planned: list[bool]) -> pl.DataFrame:
    return _rows(["satellite", "reanalysis"][: len(planned)]).with_columns(
        planned=pl.Series(planned)
    )


@pytest.mark.parametrize(
    ("planned", "expected"),
    [
        ([[True], [True, True]], "planned"),
        ([[False], [False, False]], "exploratory"),
        ([[True], [False, False]], "mixed"),
        ([[True, False]], "mixed"),
    ],
)
def test_planning_reads_every_row_of_every_panel(planned: list[list[bool]], expected: str) -> None:
    assert planning(rows=[_planned_rows(flags) for flags in planned]) == expected


def test_planning_counts_a_frame_without_the_column_as_exploratory():
    assert planning(rows=[_rows(["satellite"])]) == "exploratory"
    assert planning(rows=[_rows(["satellite"]), _planned_rows([True])]) == "mixed"


def test_planning_treats_a_null_planned_value_as_false() -> None:
    rows = _rows(["satellite"]).with_columns(planned=pl.Series([None], dtype=pl.Boolean))
    assert planning(rows=[rows]) == "exploratory"


def _labels(spec: dict) -> list[str]:
    panel = spec["vconcat"][-1]
    (interval,) = [layer for layer in _layer(panel, "rule") if "x2" in layer["encoding"]]
    return [row["label"] for row in _values(spec, interval)]


def test_a_mixed_figure_labels_each_planned_row():
    spec = _panel(_planned_rows([True, False]), figure_planning="mixed")

    assert _labels(spec) == [f"row 0{NAMED_SUFFIX}", "row 1"]


def test_a_figure_of_one_kind_labels_no_row():
    spec = _panel(_planned_rows([True, True]), figure_planning="planned")

    assert _labels(spec) == ["row 0", "row 1"]


def test_a_panel_contradicting_its_figure_raises():
    with pytest.raises(ValueError, match="all exploratory, but this panel is mixed"):
        _panel(_planned_rows([True, False]), figure_planning="exploratory")


@pytest.mark.parametrize("figure_planning", ["planned", "exploratory", "mixed"])
def test_a_figure_states_its_planning_once_in_its_subtitle(
    figure_planning: PlanningType,
) -> None:
    panels = [_panel_chart(_rows(["satellite", "reanalysis"]))]

    spec = figure(
        panels=panels,
        number=1,
        title="t",
        subtitle=["s"],
        figure_planning=figure_planning,
    ).to_dict()

    assert " ".join(spec["title"]["subtitle"]) == f"s {PLANNING_NOTES[figure_planning]}"


def test_a_figure_without_planning_adds_no_note():
    panels = [_panel_chart(_rows(["satellite", "reanalysis"]))]

    spec = figure(panels=panels, number=1, title="t", subtitle=["s"], figure_planning=None)

    assert spec.to_dict()["title"]["subtitle"] == ["s"]


def test_the_station_family_is_purple_and_named_in_the_key():
    spec = _panel(_rows(["station observations", "satellite"]))
    key, _ = spec["vconcat"]
    (text,) = _layer(key, "text")

    assert [row["label"] for row in _values(spec, text)] == ["satellite", "station observations"]
    assert [row["colour"] for row in _values(spec, text)] == [
        FAMILY_COLOURS["satellite"],
        FAMILY_COLOURS["station observations"],
    ]
    assert FAMILY_COLOURS["station observations"] == ocf.DATA_PURPLE
