"""Tests for the `RowSet` options that let `past_solar_leaderboard.py` score the past-wind row sets.

The past-solar defaults are tested in `test_past_solar_leaderboard.py`. Each test here builds a
report shaped like a past-wind report (a different reference arm, a leaderboard under its own
heading, four printed decimals, a `farm-hours` heading, second-setting rows in a section of their
own), and fails on the solar parser, which reads none of those shapes.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Final

import polars as pl
import pytest
from studies.charts import (
    CONTRAST_COLUMNS_WITH_MONTHS,
    BlockArm,
    PlannedContrast,
    block_contrast_rows,
    block_leaderboard_rows,
    planned_contrast_rows,
    report_contrasts,
)
from test_past_solar_leaderboard import ARMS, METRIC, SITE_HOURS, _load, _losses

REFERENCE: Final[BlockArm] = ARMS[0]
ENS: Final[BlockArm] = ARMS[2]
ERA5: Final[BlockArm] = ARMS[1]
PLANNED: Final[PlannedContrast] = PlannedContrast(ENS, REFERENCE)
DECIDING: Final[str] = "Deciding contrasts, named before the run"
SENSITIVITY: Final[str] = "Sensitivity: the second setting"
LEADERBOARD: Final[str] = "Leaderboard at the primary setting"
CONTRAST_HEADER: Final[str] = (
    "| Scope | Contrast | ΔMAE (pp of capacity) | 95% interval | Excludes zero? "
    "| Folds agreeing | Rows |"
)


def _fine_losses() -> pl.DataFrame:
    """Return the solar tests' losses, nudged so every mean has a nonzero fourth decimal."""
    return _losses().with_columns(pl.col(METRIC) + 0.0000037 * pl.col("seed") + 0.0000011)


def _row(*, scope: str, treatment: str, reference: str, values: tuple[float, ...]) -> str:
    difference, low, high = (round(value, 4) for value in values)
    return (
        f"| {scope} | {treatment} − {reference} | {difference:+.4f} | [{low:+.4f}, {high:+.4f}] "
        f"| no | 2 of 2 | {SITE_HOURS} |"
    )


def _contrast_values(*, losses: pl.DataFrame, setting: str) -> tuple[float, ...]:
    """Return ENS minus the reference arm's difference and interval at one setting."""
    row = planned_contrast_rows(
        losses=losses, contrasts=[PLANNED], setting=setting, site_hours=SITE_HOURS, metric=METRIC
    ).row(0, named=True)
    return row["difference"], row["lower_95"], row["upper_95"]


def _report(*, losses: pl.DataFrame, tweak: str | None = None, suffix: str = "") -> str:
    """Print a report the script recomputes exactly, optionally with one edit."""
    absolute = block_leaderboard_rows(
        losses=losses, arms=ARMS, setting="pooled", site_hours=SITE_HOURS, metric=METRIC
    )
    contrast = block_contrast_rows(
        losses=losses,
        arms=[ERA5, ENS],
        reference_arm=REFERENCE.arm,
        setting="pooled",
        site_hours=SITE_HOURS,
        metric=METRIC,
    )
    primary = _contrast_values(losses=losses, setting="pooled")
    second = _contrast_values(losses=losses, setting="sensitivity")
    wrong = tuple(value + 0.01 for value in primary)
    lines = [
        f"### Test farms on {SITE_HOURS} common farm-hours of wind (2025-01-01 to 2025-02-01)",
        "",
        "| Check | Value |",
        "|---|---|",
        "| rows | 8 |",
        "",
        f"#### {LEADERBOARD}",
        "",
        "| Arm | Wind columns | MAE (pp of capacity) | 95% interval | Rows |",
        "|---|---|---|---|---|",
    ]
    by_label = {arm.label: arm.arm.removesuffix(suffix) for arm in ARMS}
    for row in absolute.iter_rows(named=True):
        value, low, high = (round(row[name], 4) for name in ("value", "lower_95", "upper_95"))
        high += 0.01 if tweak == "wrong_interval" and row["label"] == "ERA5" else 0.0
        interval = "" if tweak == "blank_interval" else f"[{low:.4f}, {high:.4f}]"
        lines.append(
            f"| `{by_label[row['label']]}` | 3 | {value:.4f} | {interval} | {SITE_HOURS} |"
        )
    lines += ["", f"#### {DECIDING}", "", CONTRAST_HEADER, "|---|---|---|---|---|---|---|"]
    lines.append(_row(scope="all", treatment=ENS.arm, reference=REFERENCE.arm, values=primary))
    if tweak != "no_exploratory_row":
        lines.append(_row(scope="all", treatment=ERA5.arm, reference=ENS.arm, values=primary))
    lines += ["", "The last row is exploratory.", "", "#### Other contrasts (exploratory)", ""]
    lines += [CONTRAST_HEADER, "|---|---|---|---|---|---|---|"]
    for row in contrast.iter_rows(named=True):
        values = (row["difference"], row["lower_95"], row["upper_95"])
        lines.append(
            _row(scope="all", treatment=row["arm"], reference=REFERENCE.arm, values=values)
        )
    lines += ["", "#### Other fit (exploratory)", "", CONTRAST_HEADER]
    lines += ["|---|---|---|---|---|---|---|"]
    lines.append(_row(scope="all", treatment=ENS.arm, reference=REFERENCE.arm, values=wrong))
    lines += ["", f"#### {SENSITIVITY}", "", CONTRAST_HEADER, "|---|---|---|---|---|---|---|"]
    lines.append(
        _row(scope="second setting", treatment=ENS.arm, reference=REFERENCE.arm, values=second)
    )
    lines += ["", "#### Elsewhere (exploratory)", "", CONTRAST_HEADER]
    lines += ["|---|---|---|---|---|---|---|"]
    lines.append(
        _row(scope="second setting", treatment=ENS.arm, reference=REFERENCE.arm, values=wrong)
    )
    return "\n".join(lines) + "\n"


def _row_set(module: Any, tmp_path: Path) -> Any:
    return module.RowSet(
        key="test",
        label="Test farms",
        directory=tmp_path,
        printed_column="MAE (pp of capacity)",
        arm_suffix="",
        leaderboard_arms=ARMS,
        contrast_arms=(ERA5, ENS),
        planned_contrasts=(PLANNED,),
        reference_arm=REFERENCE.arm,
        printed_decimals=4,
        leaderboard_section=LEADERBOARD,
        intervals="table",
        planned_section=DECIDING,
        second_planned_section=SENSITIVITY,
        second_scope="second setting",
        second_section=SENSITIVITY,
        other_fit_sections=("Other fit",),
        exploratory_in_planned=((ERA5.arm, ENS.arm),),
    )


def _score(*, tmp_path: Path, tweak: str | None = None, suffix: str = "", **changes: Any) -> Any:
    module = _load()
    losses = _fine_losses()
    report_path = tmp_path / "report.md"
    text = _report(losses=losses, tweak=tweak, suffix=suffix)
    report_path.write_text(text)
    row_set = _row_set(module, tmp_path)._replace(**changes)
    return module.score_row_set(
        row_set=row_set, losses=losses, report_text=text, report_path=report_path
    )


def test_a_wind_shaped_report_is_scored_against_its_own_reference_arm(tmp_path: Path) -> None:
    # Catches a script that contrasts every arm with `era5_global` whatever the row set says.
    result = _score(tmp_path=tmp_path)

    assert set(result.contrasts["arm"]) == {ERA5.arm, ENS.arm}
    assert result.planned["reference_arm"].to_list() == [REFERENCE.arm]
    intervals = _load().intervals_frame(results=[result])
    contrast_rows = intervals.filter(pl.col("section") == "Mean absolute error minus ERA5's")
    assert set(contrast_rows["reference"]) == {REFERENCE.arm}


def _charts_module() -> Any:
    """Load `past_solar_leaderboard_charts.py`, which reads the report and the intervals."""
    module = _load()
    script_dir = Path(module.__file__).parent
    spec = importlib.util.spec_from_file_location(
        "past_solar_leaderboard_charts", script_dir / "past_solar_leaderboard_charts.py"
    )
    assert spec is not None
    assert spec.loader is not None
    charts = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = charts
    spec.loader.exec_module(charts)
    return charts


def test_a_blocks_contrast_heading_names_its_own_reference_arm(tmp_path: Path) -> None:
    # Catches a station-style block whose contrasts against ERA5's 10 m wind are headed "minus
    # ERA5's", in the report and in the `section` column of the intervals, and a default block
    # whose heading changed.
    module = _load()
    default = _score(tmp_path=tmp_path)
    named = _score(tmp_path=tmp_path, reference_label="ERA5's 10 m wind")

    default_text = module.render_report(results=[default])
    named_text = module.render_report(results=[named])

    assert "#### Mean absolute error minus ERA5's\n" in default_text
    assert "#### Mean absolute error minus ERA5's 10 m wind\n" in named_text
    assert "#### Mean absolute error minus ERA5's\n" not in named_text
    contrast_sections = {
        section
        for section in module.intervals_frame(results=[named])["section"]
        if section.startswith("Mean absolute error minus")
    }
    assert contrast_sections == {"Mean absolute error minus ERA5's 10 m wind"}
    default_sections = set(module.intervals_frame(results=[default])["section"])
    assert module.CONTRAST_SECTION in default_sections
    assert module.CONTRAST_SECTION == "Mean absolute error minus ERA5's"


def test_the_charts_read_a_blocks_contrasts_under_its_own_heading(tmp_path: Path) -> None:
    # Catches the chart script looking for the contrasts under the fixed "minus ERA5's" heading, so
    # a block with its own reference heading is found empty or stops with a KeyError.
    module = _load()
    charts = _charts_module()
    result = _score(tmp_path=tmp_path, reference_label="ERA5's 10 m wind")
    report = charts.read_report(report_text=module.render_report(results=[result]))
    printed = report["Test farms"]
    intervals = module.intervals_frame(results=[result])

    assert "Mean absolute error minus ERA5's 10 m wind" in printed.tables
    drawn = charts.contrast_rows(
        frame=intervals,
        row_set=result.row_set,
        order=[],
        printed=printed.tables["Mean absolute error minus ERA5's 10 m wind"],
    )
    assert set(drawn["arm"]) == {ERA5.arm, ENS.arm}


def test_a_post_hoc_planned_contrast_is_marked_in_the_report_and_the_figures(
    tmp_path: Path,
) -> None:
    # Catches a planned contrast whose arms were chosen after the first run being printed and
    # drawn as an unmarked planned contrast, and a mark that reaches a contrast it was not set for.
    module = _load()
    charts = _charts_module()
    plain = _score(tmp_path=tmp_path)
    marked = _score(tmp_path=tmp_path, post_hoc_contrasts=((ENS.arm, REFERENCE.arm),))
    label = PLANNED.label

    assert plain.planned["label"].to_list() == [label]
    assert marked.planned["label"].to_list() == [f"{label} (post hoc)"]
    assert f"| {label} (post hoc) |" in module.render_report(results=[marked])
    assert f"| {label} |" in module.render_report(results=[plain])
    report = charts.read_report(report_text=module.render_report(results=[marked]))
    intervals = module.intervals_frame(results=[marked])
    drawn_planned = charts.planned_rows(
        frame=intervals,
        row_set=marked.row_set,
        printed=report["Test farms"].tables[module.PLANNED_CONTRAST_SECTION],
    )
    assert drawn_planned["label"].to_list() == [f"{label} (post hoc)"]
    assert drawn_planned["planned"].to_list() == [False]
    drawn_contrasts = charts.contrast_rows(
        frame=intervals,
        row_set=marked.row_set,
        order=[],
        printed=report["Test farms"].tables[module.CONTRAST_SECTION],
    )
    labels = dict(zip(drawn_contrasts["arm"], drawn_contrasts["label"], strict=True))
    assert labels[ENS.arm].endswith(" (post hoc)")
    assert not labels[ERA5.arm].endswith(" (post hoc)")


def test_a_farm_hours_heading_is_read() -> None:
    # Catches a heading regex that knows only "common site-hours".
    module = _load()

    assert module.read_heading(
        report_text="### T on 43,555 common farm-hours (2024-12-01 to 2026-09-10)"
    ) == (43555, "2024-12-01 to 2026-09-10")
    assert module.read_heading(
        report_text="### T on 34,156 farm-hours (2024-08-12 to 2025-12-31)"
    ) == (
        34156,
        "2024-08-12 to 2025-12-31",
    )


def test_errors_are_read_from_the_table_under_the_named_heading_not_the_first_table(
    tmp_path: Path,
) -> None:
    # Catches reading the report's first table (here a table of checks) as the table of errors.
    result = _score(tmp_path=tmp_path)

    assert result.site_hours == SITE_HOURS


def test_a_leaderboard_heading_the_report_lacks_stops_the_script(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no heading starting 'Nowhere'"):
        _score(tmp_path=tmp_path, leaderboard_section="Nowhere")


def test_a_printed_error_at_four_decimals_is_compared_at_four_decimals(tmp_path: Path) -> None:
    # Catches comparing a four-decimal report at three decimals, which never matches.
    _score(tmp_path=tmp_path)

    with pytest.raises(ValueError, match="cams_global"):
        _score(tmp_path=tmp_path, printed_decimals=3)


def test_an_interval_table_that_prints_a_column_the_row_set_did_not_declare_stops_the_script(
    tmp_path: Path,
) -> None:
    # Catches an interval parser that skips a table shape it does not know instead of failing.
    with pytest.raises(ValueError, match="declares intervals='none'"):
        _score(tmp_path=tmp_path, intervals="none")


def test_an_interval_table_the_row_set_expects_but_the_report_lacks_stops_the_script() -> None:
    module = _load()
    text = "### T\n\n| Arm | MAE |\n|---|---|\n| `a` | 1.0 |\n"

    with pytest.raises(ValueError, match="declares intervals='table'"):
        module.printed_table_intervals(
            report_text=text, section_prefix=None, column="MAE", arm_suffix="", intervals="table"
        )


def test_an_interval_cell_that_is_not_a_bracketed_pair_stops_the_script() -> None:
    module = _load()
    text = "| Arm | MAE | 95% interval |\n|---|---|---|\n| `a` | 1.0 | 0.5 to 1.5 |\n"

    with pytest.raises(ValueError, match="is not '\\[low, high\\]'"):
        module.printed_table_intervals(
            report_text=text, section_prefix=None, column="MAE", arm_suffix="", intervals="table"
        )


def test_an_arm_with_a_blank_interval_cell_is_left_out_of_the_interval_check() -> None:
    # Catches a parser that fails on a product the report prints with no interval.
    module = _load()
    text = (
        "| Product | MAE | 95% interval |\n"
        "|---|---|---|\n"
        "| a | 1.0 | [0.5, 1.5] |\n"
        "| b | 2.0 |  |\n"
    )

    printed = module.printed_table_intervals(
        report_text=text, section_prefix=None, column="MAE", arm_suffix="_wind", intervals="table"
    )

    assert printed == {"a_wind": (1.0, 0.5, 1.5)}


def test_a_report_with_no_intervals_is_checked_for_none_when_the_row_set_says_none() -> None:
    module = _load()
    text = "| Product | All sites |\n|---|---|\n| a | 1.0 |\n"

    assert (
        module.printed_table_intervals(
            report_text=text,
            section_prefix=None,
            column="All sites",
            arm_suffix="",
            intervals="none",
        )
        == {}
    )


def test_a_printed_contrast_labelled_exploratory_under_the_planned_heading_is_not_unlisted(
    tmp_path: Path,
) -> None:
    # Catches the planned-contrast check reporting a row the report itself labels exploratory.
    _score(tmp_path=tmp_path)

    with pytest.raises(ValueError, match=r"does not list it|does not reproduce"):
        _score(tmp_path=tmp_path, exploratory_in_planned=())


def test_a_section_that_refits_is_left_out_of_the_first_setting_comparison(
    tmp_path: Path,
) -> None:
    # Catches comparing the first setting with a section from another fit: its number is wrong.
    _score(tmp_path=tmp_path)

    with pytest.raises(ValueError, match="Other fit"):
        _score(tmp_path=tmp_path, other_fit_sections=())


def test_second_setting_rows_are_read_by_the_rows_own_scope_and_section(tmp_path: Path) -> None:
    # Catches a second setting looked for only under the scope `sensitivity`, and one read from
    # a section that is not the second setting's.
    _score(tmp_path=tmp_path)

    with pytest.raises(ValueError, match="Elsewhere"):
        _score(tmp_path=tmp_path, second_section="")


def test_a_second_setting_section_that_is_also_listed_as_another_fit_is_still_compared(
    tmp_path: Path,
) -> None:
    # Catches the second-setting pass dropping the very section it keeps: the wind row sets list
    # their second-setting heading among `other_fit_sections` (to keep it out of the first-setting
    # pass), so a pass that applied that list to itself compared nothing and passed.
    _score(tmp_path=tmp_path, other_fit_sections=("Other fit", SENSITIVITY))

    with pytest.raises(ValueError, match="Elsewhere"):
        _score(
            tmp_path=tmp_path,
            other_fit_sections=("Other fit", SENSITIVITY),
            second_section="",
        )


def test_a_second_setting_pass_that_compares_nothing_although_rows_exist_stops_the_script(
    tmp_path: Path,
) -> None:
    # Catches a second-setting check that silently compares zero contrasts: every printed row is
    # filtered out by the section options, so a wrong second-setting number would pass.
    module = _load()
    result = _score(tmp_path=tmp_path)
    printed = report_contrasts(report_path=tmp_path / "report.md")
    options: dict[str, Any] = {
        "contrasts": result.contrasts,
        "printed": printed,
        "site_hours": SITE_HOURS,
        "scope": "second setting",
        "column_prefix": "second_",
        "reference_arm": REFERENCE.arm,
        "decimals": 4,
    }

    assert module.check_contrasts(**options, section_prefix=SENSITIVITY) == []
    with pytest.raises(ValueError, match="none was compared"):
        module.check_contrasts(
            **options, section_prefix=SENSITIVITY, other_fit_sections=(SENSITIVITY,)
        )
    # A scope the report prints no row for is the report's own gap, not a filtered-out check.
    assert (
        module.check_contrasts(**{**options, "scope": "no such scope"}, section_prefix=SENSITIVITY)
        == []
    )


def test_a_second_setting_row_the_report_does_not_print_stops_the_script(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="planned, but the report prints no row"):
        _score(tmp_path=tmp_path, second_scope="no such scope")


def test_a_contrast_table_with_a_months_column_is_read_only_where_the_row_set_says_so(
    tmp_path: Path,
) -> None:
    # Catches contrast tables that carry a `Months` column being skipped without a word.
    module = _load()
    losses = _fine_losses()
    text = _report(losses=losses)
    header = "| " + " | ".join(CONTRAST_COLUMNS_WITH_MONTHS) + " |"
    wide = (
        "\n".join(
            header
            if line == CONTRAST_HEADER
            else f"{line} 2 |"
            if line.startswith("| all")
            else line
            for line in text.splitlines()
        ).replace("|---|---|---|---|---|---|---|", "|---|---|---|---|---|---|---|---|")
        + "\n"
    )
    report_path = tmp_path / "report.md"
    report_path.write_text(wide)
    row_set = _row_set(module, tmp_path)

    with pytest.raises(ValueError, match="no contrast table read from the report"):
        module.score_row_set(
            row_set=row_set, losses=losses, report_text=wide, report_path=report_path
        )
    module.score_row_set(
        row_set=row_set._replace(wide_contrast_tables=True),
        losses=losses,
        report_text=wide,
        report_path=report_path,
    )


def test_the_shared_run_writes_the_title_it_is_given(tmp_path: Path) -> None:
    # Catches a report whose heading says "Past-solar" whichever row sets it holds.
    module = _load()
    result = _score(tmp_path=tmp_path)

    module.write_outputs(
        results=[result], output_dir=tmp_path / "out", title="Wind title", introduction="Intro."
    )

    text = (tmp_path / "out" / "report.md").read_text()
    assert text.startswith("# Wind title\n\nIntro.\n")


def test_the_arm_suffix_is_added_to_the_names_of_the_printed_intervals(tmp_path: Path) -> None:
    # Catches interval names read without the suffix, so a product's interval is never checked:
    # the wind reports print `era5`, and the arm is `era5_wind`.
    row_set: dict[str, Any] = {"arm_suffix": "_global", "leaderboard_arms": ARMS[:2]}
    result = _score(tmp_path=tmp_path, suffix="_global", **row_set)

    assert result.site_hours == SITE_HOURS
    with pytest.raises(ValueError, match="interval"):
        _score(tmp_path=tmp_path, suffix="_global", tweak="wrong_interval", **row_set)
