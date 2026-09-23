import re
from pathlib import Path

import polars as pl
import pytest
from studies.charts import (
    FAMILY_COLOURS,
    FAMILY_COLOURS_LIGHT,
    ContrastKey,
    flip_contrast,
    interval_panel,
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
    (interval,) = [layer for layer in _layer(spec, "rule") if "x2" in layer["encoding"]]

    plotted = _values(spec, interval)

    assert [(r["label"], r["difference"], r["lower_95"], r["upper_95"]) for r in plotted] == [
        ("row 0", -1.5, -2.0, -1.0),
        ("row 1", 0.25, 0.125, 0.5),
    ]
    assert interval["encoding"]["y"]["sort"] == ["row 0", "row 1"]


def test_colour_follows_the_family_and_the_legend_lists_only_families_present():
    spec = _panel(_rows(["reanalysis"]))
    (interval,) = [layer for layer in _layer(spec, "rule") if "x2" in layer["encoding"]]
    colour = interval["encoding"]["color"]

    assert colour["scale"]["domain"][:3] == list(FAMILY_COLOURS)
    assert colour["scale"]["range"] == [*FAMILY_COLOURS.values(), *FAMILY_COLOURS_LIGHT.values()]
    assert colour["legend"]["values"] == ["reanalysis"]


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
