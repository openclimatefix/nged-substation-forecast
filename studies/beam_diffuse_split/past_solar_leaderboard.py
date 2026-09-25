"""One leaderboard and one set of contrasts against ERA5 for the four headline past-solar row sets.

The past-solar page scores four row sets, each on its own common rows: the main rows
(`solar_long`), the extra rows (`solar_all`), the ECMWF ENS rows (`ens_past_solar`), and the
weather-station rows (`station_past_solar`). This script reads each row set's saved `pooled`
losses, without refitting anything, and writes for each

- every arm's own mean absolute error with a 95% interval,
- every arm's mean absolute error minus ERA5's with a 95% interval, and
- every planned contrast the row set's report prints, the first product minus the second (14 in
  all: 6 main, 3 extra, 2 ENS, 3 station), most of them not against ERA5, and
- the exploratory contrasts the row set's report does not print (SARAH-3 minus CAMS on the extra
  rows), at the first setting only,

all from the same month-and-seed resampling `weather_products.py` uses. Contrasts that a row
set's own report prints are labelled planned when the report names them before the run, and every
other contrast is exploratory, except the two UKV rebuilds, which the main report labels post hoc.
A contrast that has a second hyperparameter setting saved is
recomputed there if it is planned or lies near the 5% line (an interval bound within 20% of the
interval's width from zero).

**The script stops before writing anything unless every number a row set's report already prints
is reproduced at the report's own precision**: each arm's error, each printed interval, and each
printed contrast against ERA5 on the same rows, and each planned contrast, at both settings where
the second is saved. A planned contrast the report prints and this script does not list, or the
reverse, stops the script too. It writes `report.md` and `intervals.parquet` into
`SOLAR_LEADERBOARD_DIR`, which it refuses to overwrite. `--check-only` reads and verifies but
writes nothing.

Run it with `uv run studies/beam_diffuse_split/past_solar_leaderboard.py`. No generator's name,
identifier, or coordinates appears in its output.
"""

import argparse
import logging
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final, Literal, NamedTuple

import polars as pl
import station_past_solar_charts as station_charts
import weather_product_charts as main_charts
from ens_past_solar_charts import NAMES as ENS_NAMES
from sources import SOLAR_LEADERBOARD_DIR, UPDATE_OUTPUT_DIR
from studies.charts import (
    CONTRAST_COLUMNS_WITH_MONTHS,
    POST_HOC_SUFFIX,
    REPORT_PRINT_DECIMALS,
    BlockArm,
    PlannedContrast,
    ProductFamily,
    assert_matches_printed,
    block_contrast_rows,
    block_leaderboard_rows,
    planned_contrast_rows,
    report_contrasts,
    report_errors,
)
from weather_products import METRIC

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

PRINT_DECIMALS: Final[int] = REPORT_PRINT_DECIMALS
"""The precision every study report prints its numbers at."""

REFERENCE_ARM: Final[str] = "era5_global"
"""The arm every contrast is taken against."""

NEAR_LINE_SHARE: Final[float] = 0.2
"""How close to zero an interval bound must lie, as a share of the interval's width, for a
contrast to count as near the 5% line."""

PLANNED_SECTION_PREFIX: Final[str] = "Planned contrasts"
"""How the heading of a report's planned-contrast table starts, in every row set's report."""

OTHER_FIT_SECTION_PREFIXES: Final[tuple[str, ...]] = ("Leave one site out",)
"""How the headings of report sections that refit the arms on other rows start.

A section like these prints a contrast on the same arms and the same number of rows as the main
fit, but from a different fit whose losses `losses.parquet` does not hold, so its numbers are not
comparable with the recomputed ones.
"""

PLANNED_CONTRAST_SECTION: Final[str] = "Planned contrasts, first product minus second"
"""The `section` of a planned contrast in `intervals.parquet`."""

EXPLORATORY_CONTRAST_SECTION: Final[str] = "Exploratory contrasts, first product minus second"
"""The `section` of an exploratory contrast between two products in `intervals.parquet`."""

ABSOLUTE_SECTION: Final[str] = "Mean absolute error"
"""The `section` of an arm's own error in `intervals.parquet`."""

CONTRAST_SECTION: Final[str] = "Mean absolute error minus ERA5's"
"""The `section` of a contrast against ERA5 in `intervals.parquet`."""

SECOND_SETTING_SCOPE: Final[str] = "sensitivity"
"""The scope a report gives a contrast at the second hyperparameter setting."""

REPORT_TITLE: Final[str] = "Past-solar leaderboard and contrasts against ERA5"
"""The heading of the past-solar leaderboard's `report.md`."""

REPORT_INTRODUCTION: Final[str] = (
    "Every number is recomputed from the saved `pooled` losses of four row sets, by resampling "
    "whole months and a fitting seed. Each row set is scored on its own common rows, so a value "
    "is comparable within a row set and not across row sets. Mean absolute error is a percentage "
    "of each generator's 99th-percentile output. Every contrast not named before the run is "
    "exploratory, except the two UKV rebuilds, which were added after the first run: their own "
    "errors and their contrasts are post hoc. `Second setting` is the same contrast at the second "
    "hyperparameter setting, shown only for planned contrasts and contrasts near the 5% line "
    "(an interval bound within 20% of the interval's width from zero), and only where both arms "
    "have saved second-setting losses. `Exploratory contrasts` are contrasts between two "
    "products that no earlier report prints, shown at the first setting only."
)
CONTRAST_HEADER: Final[str] = (
    "| Arm | Difference (pp of capacity) | 95% interval | Planned or exploratory "
    "| Near the 5% line | Second setting |"
)

PLANNED_HEADER: Final[str] = (
    "| Contrast | Difference (pp of capacity) | 95% interval | Near the 5% line | Second setting |"
)

EXPLORATORY_HEADER: Final[str] = (
    "| Contrast | Difference (pp of capacity) | 95% interval | Near the 5% line | Second setting |"
)

_HEADING: Final[re.Pattern[str]] = re.compile(
    r"on ([\d,]+) (?:common )?(?:site|farm)-hours.*\((\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})\)"
)
INTERVAL_COLUMN: Final[str] = "95% interval"
"""The header of a table's column of 95% intervals."""

_INTERVAL_CELL: Final[re.Pattern[str]] = re.compile(r"^\[([\d.]+), ([\d.]+)\]$")
_PRINTED_REBUILD_ERRORS: Final[re.Pattern[str]] = re.compile(r"^MAE: (.+)$", re.MULTILINE)
_PRINTED_ARM_ERROR: Final[re.Pattern[str]] = re.compile(r"(\w+) (\d+\.\d+)")
_PRINTED_INTERVAL: Final[re.Pattern[str]] = re.compile(
    r"^\| (\w+) \| ([\d.]+) \| \[([\d.]+), ([\d.]+)\] \|$", re.MULTILINE
)


IntervalsType = Literal["solar", "table", "none"]
"""How a report prints each arm's 95% interval.

`solar` is a three-cell row `| arm | error | [low, high] |` anywhere in the report. `table` is a
`95% interval` column in the table that holds the errors. `none` is no interval at all.
"""


class RowSet(NamedTuple):
    """One headline row set: where its results live and which arms are scored.

    Attributes:
        key: A short name for the row set, used in the output files.
        label: The row set's name in headings.
        directory: The folder holding the row set's `losses.parquet` and `report.md`.
        printed_column: The header of the report's first table's column of errors.
        arm_suffix: What the report's first table appends to an arm's name in its key column:
            the main and extra rows print products, so `_global` is appended to reach the arm.
        leaderboard_arms: The arms whose own error is scored.
        contrast_arms: The arms contrasted with ERA5.
        planned_contrasts: The contrasts the row set's report names before the run, each the
            first product's error minus the second's.
        exploratory_contrasts: Contrasts between two products that the report does not print,
            each the first product's error minus the second's, scored at the first setting only.
        reference_arm: The arm every contrast in `contrast_arms` is taken against.
        printed_decimals: The decimal places the report prints its numbers at.
        leaderboard_section: The start of the heading above the report's table of errors, or
            `None` where that table is the first table in the report.
        intervals: How the report prints each arm's 95% interval.
        planned_section: The start of the headings of the tables that print the planned
            contrasts at the first setting.
        second_planned_section: The start of the headings of the tables that print the planned
            contrasts at the second setting, or `None` where they sit under `planned_section`.
        second_scope: The `Scope` cell of a printed row at the second setting.
        second_section: The start of the headings a row at the second setting is compared
            under; the empty string compares a row from any section.
        other_fit_sections: The starts of the headings of sections that print a contrast on the
            first setting's arms and rows from a fit whose losses `losses.parquet` does not hold.
        exploratory_in_planned: Contrasts that the report prints under `planned_section` and
            labels exploratory, each as (first arm, second arm).
        wide_contrast_tables: Whether the report's contrast tables carry a `Months` column.
        hours_unit: What a row of the row set is called in headings: solar rows are site-hours.
    """

    key: str
    label: str
    directory: Path
    printed_column: str
    arm_suffix: str
    leaderboard_arms: tuple[BlockArm, ...]
    contrast_arms: tuple[BlockArm, ...]
    planned_contrasts: tuple[PlannedContrast, ...]
    exploratory_contrasts: tuple[PlannedContrast, ...] = ()
    reference_arm: str = REFERENCE_ARM
    printed_decimals: int = PRINT_DECIMALS
    leaderboard_section: str | None = None
    intervals: IntervalsType = "solar"
    planned_section: str = PLANNED_SECTION_PREFIX
    second_planned_section: str | None = None
    second_scope: str = SECOND_SETTING_SCOPE
    second_section: str = ""
    other_fit_sections: tuple[str, ...] = OTHER_FIT_SECTION_PREFIXES
    exploratory_in_planned: tuple[tuple[str, str], ...] = ()
    wide_contrast_tables: bool = False
    hours_unit: str = "site-hours"


class RowSetResult(NamedTuple):
    """One row set's scores.

    Attributes:
        row_set: The row set scored.
        dates: The row set's first and last day, as its report's heading prints them.
        site_hours: The row set's number of common site-hours.
        absolute: The output of `block_leaderboard_rows`.
        contrasts: The output of `block_contrast_rows`, with `planning`, `near_line`, and the
            second setting's `second_difference`, `second_lower_95` and `second_upper_95` (null
            where none was computed or saved).
        planned: The output of `planned_contrast_rows`, with `near_line` and the same three
            second-setting columns.
        exploratory: The output of `planned_contrast_rows` for the row set's exploratory
            contrasts, with `planned` false and `near_line`.
    """

    row_set: RowSet
    dates: str
    site_hours: int
    absolute: pl.DataFrame
    contrasts: pl.DataFrame
    planned: pl.DataFrame
    exploratory: pl.DataFrame


def _main_arms(
    *, names: dict[str, str], families: dict[str, ProductFamily], served_name: Callable[[str], str]
) -> tuple[BlockArm, ...]:
    """Return one arm per product of a main-or-extra row set, CAMS and ERA5 as reference rows."""
    return tuple(
        BlockArm(
            arm=f"{product}_global",
            label=served_name(product),
            family=families[product],
            reference=product in ("cams", "era5"),
        )
        for product in names
    )


MAIN_ARMS: Final[tuple[BlockArm, ...]] = _main_arms(
    names=main_charts.NAMES, families=main_charts.FAMILIES, served_name=main_charts.served_name
)
EXTRA_ARMS: Final[tuple[BlockArm, ...]] = _main_arms(
    names=main_charts.ALL_PANEL_NAMES,
    families=main_charts.ALL_PANEL_FAMILIES,
    served_name=main_charts.all_served_name,
)
UKV_REBUILDS: Final[tuple[BlockArm, ...]] = (
    BlockArm("ukv_trap_global", "UKV rebuilt from its snapshots", "weather model"),
    BlockArm("ukv_pair_global", "UKV, both snapshots as separate inputs", "weather model"),
)
"""The two rebuilds of UKV's hourly value, scored and contrasted with ERA5 on the main rows only."""

POST_HOC_ARMS: Final[list[str]] = [arm.arm for arm in UKV_REBUILDS]
"""The arms whose contrast against ERA5 the main report labels post hoc, not exploratory."""

ENS_ARMS: Final[tuple[BlockArm, ...]] = (
    BlockArm("ens_mean_t3", ENS_NAMES["ens_mean_t3"], "weather model"),
    BlockArm("ens_control_t3", "ECMWF ENS (control member, T+3 band)", "weather model"),
    BlockArm("era5_global", ENS_NAMES["era5_global"], "reanalysis", reference=True),
    BlockArm("cams_global", ENS_NAMES["cams_global"], "satellite", reference=True),
)
STATION_ARMS: Final[tuple[BlockArm, ...]] = tuple(
    BlockArm(
        arm=arm,
        label=station_charts.NAMES[arm],
        family=station_charts.FAMILIES[arm],
        reference=arm in ("cams_global", "era5_global"),
    )
    for arm in station_charts.LEADERBOARD_ARMS
)


def _planned(
    *, arms: Sequence[BlockArm], pairs: Sequence[tuple[str, str]]
) -> tuple[PlannedContrast, ...]:
    """Return each (treatment, reference) pair of arm names as a `PlannedContrast`."""
    by_arm = {arm.arm: arm for arm in arms}
    return tuple(
        PlannedContrast(by_arm[treatment], by_arm[reference]) for treatment, reference in pairs
    )


MAIN_PLANNED: Final[tuple[PlannedContrast, ...]] = _planned(
    arms=MAIN_ARMS,
    pairs=(
        ("cams_global", "icon_d2_global"),
        ("icon_eu_global", "icon_d2_global"),
        ("icon_eu_global", "ukv_global"),
        ("icon_global_global", "icon_eu_global"),
        ("sarah3_global", "cams_global"),
        ("icon_dream_global", "era5_global"),
    ),
)
EXTRA_PLANNED: Final[tuple[PlannedContrast, ...]] = _planned(
    arms=EXTRA_ARMS,
    pairs=(
        ("knmi_harmonie_global", "icon_eu_global"),
        ("dmi_harmonie_global", "icon_d2_global"),
        ("ifs_hres_global", "icon_eu_global"),
    ),
)
ENS_PLANNED: Final[tuple[PlannedContrast, ...]] = _planned(
    arms=ENS_ARMS, pairs=(("ens_mean_t3", "era5_global"), ("ens_mean_t3", "cams_global"))
)
STATION_PLANNED: Final[tuple[PlannedContrast, ...]] = _planned(
    arms=(
        *STATION_ARMS,
        BlockArm(
            arm=station_charts.BLEND_CONTROL_ARM,
            label=station_charts.NAMES[station_charts.BLEND_CONTROL_ARM],
            family=station_charts.FAMILIES[station_charts.BLEND_CONTROL_ARM],
        ),
    ),
    pairs=(
        (station_charts.STATION_ARM, "cams_global"),
        (station_charts.STATION_ARM, "era5_global"),
        (station_charts.BLEND_ARM, station_charts.BLEND_CONTROL_ARM),
    ),
)
"""The planned contrasts of each row set, each the first arm's error minus the second's.

Each set is the one its row set's report prints under its planned-contrast heading, and
`score_row_set` stops unless the two agree.
"""

EXTRA_EXPLORATORY: Final[tuple[PlannedContrast, ...]] = _planned(
    arms=EXTRA_ARMS, pairs=(("sarah3_global", "cams_global"),)
)
"""SARAH-3 minus CAMS on the extra rows, which no report prints for that row set.

The main rows' report names the same contrast as planned, so Figure 1's title (CAMS has the lowest
error of the gridded products tested) rests on a number for the extra rows only if this script
prints it.
"""


def _without_era5(*, arms: tuple[BlockArm, ...]) -> tuple[BlockArm, ...]:
    """Drop ERA5, which the zero rule stands for in a contrast against ERA5."""
    return tuple(arm for arm in arms if arm.arm != REFERENCE_ARM)


ROW_SETS: Final[tuple[RowSet, ...]] = (
    RowSet(
        key="main",
        label="Main rows",
        directory=UPDATE_OUTPUT_DIR / "solar_long",
        printed_column="Global only",
        arm_suffix="_global",
        leaderboard_arms=(*MAIN_ARMS, *UKV_REBUILDS),
        contrast_arms=(*_without_era5(arms=MAIN_ARMS), *UKV_REBUILDS),
        planned_contrasts=MAIN_PLANNED,
    ),
    RowSet(
        key="extra",
        label="Extra rows",
        directory=UPDATE_OUTPUT_DIR / "solar_all",
        printed_column="Global only",
        arm_suffix="_global",
        leaderboard_arms=EXTRA_ARMS,
        contrast_arms=_without_era5(arms=EXTRA_ARMS),
        planned_contrasts=EXTRA_PLANNED,
        exploratory_contrasts=EXTRA_EXPLORATORY,
    ),
    RowSet(
        key="ens",
        label="ECMWF ENS rows",
        directory=UPDATE_OUTPUT_DIR / "ens_past_solar",
        printed_column="All sites",
        arm_suffix="",
        leaderboard_arms=ENS_ARMS,
        contrast_arms=_without_era5(arms=ENS_ARMS),
        planned_contrasts=ENS_PLANNED,
    ),
    RowSet(
        key="station",
        label="Weather-station rows",
        directory=UPDATE_OUTPUT_DIR / "station_past_solar",
        printed_column="All sites",
        arm_suffix="",
        leaderboard_arms=STATION_ARMS,
        contrast_arms=_without_era5(arms=STATION_ARMS),
        planned_contrasts=STATION_PLANNED,
    ),
)
"""The four headline row sets, in the order the leaderboard stacks them."""


def read_heading(*, report_text: str) -> tuple[int, str]:
    """Read the site-hours and the dates from a report's first heading.

    Args:
        report_text: A row set's `report.md`.

    Returns:
        The number of common site-hours or farm-hours, and the dates as `2022-12-01 to 2026-08-31`.

    Raises:
        ValueError: If no line holds both.
    """
    match = _HEADING.search(report_text)
    if match is None:
        msg = "the report has no 'on N common site-hours (first to last)' or 'farm-hours' heading"
        raise ValueError(msg)
    return int(match[1].replace(",", "")), f"{match[2]} to {match[3]}"


def printed_intervals(*, report_text: str) -> dict[str, tuple[float, float, float]]:
    """Read every arm's error and 95% interval from a report's `All sites` table.

    Args:
        report_text: A row set's `report.md`.

    Returns:
        Each arm's (error, lower bound, upper bound); empty for a report that prints no intervals.
    """
    return {
        match[1]: (float(match[2]), float(match[3]), float(match[4]))
        for match in _PRINTED_INTERVAL.finditer(report_text)
    }


def leaderboard_table(
    *, report_text: str, section_prefix: str | None
) -> tuple[tuple[str, ...], list[tuple[str, ...]]]:
    """Return the header and the rows of the table that holds a report's errors.

    Args:
        report_text: A row set's `report.md`.
        section_prefix: The start of the heading above the table, or `None` for the first table
            in the report.

    Returns:
        The table's header cells, and each body row's cells with padding and backticks removed.

    Raises:
        ValueError: If no heading starts with `section_prefix`, or no table follows it.
    """
    lines = report_text.splitlines()
    start = 0
    if section_prefix is not None:
        headings = [
            index
            for index, line in enumerate(lines)
            if line.startswith("#") and line.lstrip("#").strip().startswith(section_prefix)
        ]
        if not headings:
            msg = f"the report has no heading starting {section_prefix!r}"
            raise ValueError(msg)
        start = headings[0]
    table = []
    for line in lines[start:]:
        if line.startswith("|"):
            table.append(tuple(cell.strip().strip("`") for cell in line.strip("|").split("|")))
        elif table:
            break
    if len(table) < 3:
        msg = f"no table follows the heading starting {section_prefix!r}"
        raise ValueError(msg)
    return table[0], table[2:]


def printed_table_errors(
    *, report_text: str, section_prefix: str | None, column: str, arm_suffix: str
) -> dict[str, float]:
    """Read each arm's error from the column of a report's table of errors.

    Args:
        report_text: A row set's `report.md`.
        section_prefix: The start of the heading above the table, or `None` for the first table.
        column: The header of the column of errors.
        arm_suffix: What the key column leaves off an arm's name.

    Returns:
        Each arm's printed error.

    Raises:
        ValueError: If the table has no such column.
    """
    header, rows = leaderboard_table(report_text=report_text, section_prefix=section_prefix)
    if column not in header:
        msg = f"the table under {section_prefix!r} has no column {column!r}: {header}"
        raise ValueError(msg)
    index = header.index(column)
    return {f"{cells[0]}{arm_suffix}": float(cells[index]) for cells in rows}


def printed_table_intervals(
    *,
    report_text: str,
    section_prefix: str | None,
    column: str,
    arm_suffix: str,
    intervals: IntervalsType,
) -> dict[str, tuple[float, float, float]]:
    """Read each arm's error and 95% interval from the report, in whichever shape it prints them.

    Args:
        report_text: A row set's `report.md`.
        section_prefix: The start of the heading above the table of errors, or `None` for the
            first table.
        column: The header of the column of errors.
        arm_suffix: What the key column leaves off an arm's name.
        intervals: How the report prints intervals, as declared by the row set.

    Returns:
        Each arm's (error, lower bound, upper bound); empty where `intervals` is `none`. An arm
        whose interval cell is blank is left out.

    Raises:
        ValueError: If the report's table has a `95% interval` column that `intervals` does not
            expect, or lacks one that it does, or holds an interval cell that is not `[low, high]`.
    """
    if intervals == "solar":
        return printed_intervals(report_text=report_text)
    header, rows = leaderboard_table(report_text=report_text, section_prefix=section_prefix)
    has_column = INTERVAL_COLUMN in header
    if has_column != (intervals == "table"):
        msg = (
            f"the row set declares intervals={intervals!r} but the table under "
            f"{section_prefix!r} has header {header}"
        )
        raise ValueError(msg)
    if intervals == "none":
        return {}
    error_index, interval_index = header.index(column), header.index(INTERVAL_COLUMN)
    printed = {}
    for cells in rows:
        if not cells[interval_index]:
            continue
        match = _INTERVAL_CELL.match(cells[interval_index])
        if match is None:
            msg = f"{cells[0]}: the interval cell {cells[interval_index]!r} is not '[low, high]'"
            raise ValueError(msg)
        printed[f"{cells[0]}{arm_suffix}"] = (
            float(cells[error_index]),
            float(match[1]),
            float(match[2]),
        )
    return printed


def printed_rebuild_errors(*, report_text: str) -> dict[str, float]:
    """Read the errors a report prints on one `MAE:` line after its table of contrasts.

    The main report prints the two UKV rebuilds' errors this way (`MAE: ukv_trap_global 8.180,
    ukv_pair_global 8.125.`) and in no table.

    Args:
        report_text: A row set's `report.md`.

    Returns:
        Each arm's printed error; empty for a report with no such line.
    """
    return {
        arm: float(value)
        for line in _PRINTED_REBUILD_ERRORS.findall(report_text)
        for arm, value in _PRINTED_ARM_ERROR.findall(line)
    }


def check_intervals(
    *,
    absolute: pl.DataFrame,
    report_text: str,
    printed: dict[str, tuple[float, float, float]] | None = None,
    decimals: int = PRINT_DECIMALS,
) -> list[str]:
    """List every recomputed interval that differs from the interval the report prints.

    Args:
        absolute: `block_leaderboard_rows`'s output.
        report_text: The row set's `report.md`.
        printed: Each arm's printed (error, lower bound, upper bound), where the report prints
            them in a shape `printed_intervals` does not read; `None` reads them from
            `report_text`.
        decimals: The decimal places the report prints its numbers at.

    Returns:
        One message per difference at `decimals` places; empty where all agree or the report
        prints no intervals.
    """
    if printed is None:
        printed = printed_intervals(report_text=report_text)
    problems = []
    for row in absolute.iter_rows(named=True):
        arm = row["arm"]
        if arm not in printed:
            continue
        recomputed = tuple(round(row[name], decimals) for name in ("lower_95", "upper_95"))
        if recomputed != printed[arm][1:]:
            problems.append(
                f"{arm}: interval {recomputed} but the report prints {printed[arm][1:]}"
            )
    return problems


def check_contrasts(
    *,
    contrasts: pl.DataFrame,
    printed: pl.DataFrame,
    site_hours: int,
    scope: str = "all",
    column_prefix: str = "",
    reference_arm: str = REFERENCE_ARM,
    other_fit_sections: tuple[str, ...] = OTHER_FIT_SECTION_PREFIXES,
    section_prefix: str = "",
    decimals: int = PRINT_DECIMALS,
) -> list[str]:
    """List every recomputed contrast that differs from a printed contrast on the same rows.

    A printed row is compared where it has the same scope, treatment, reference, and number of
    rows, and sits in a section that reports the fit being checked: a section starting with one of
    `other_fit_sections` prints a different fit's numbers.

    Args:
        contrasts: `block_contrast_rows`'s output, with `arm`.
        printed: `report_contrasts`'s output for the row set's report.
        site_hours: The row set's number of site-hours; a printed row on any other rows (a season,
            a scope) is a different contrast and is skipped.
        scope: The printed scope to compare with: `all`, or `SECOND_SETTING_SCOPE`.
        column_prefix: What precedes `difference`, `lower_95` and `upper_95` in the columns to
            compare: empty for the main setting, `second_` for the second setting. A row whose
            value is null is skipped.
        reference_arm: The arm every contrast is taken against.
        other_fit_sections: The starts of the headings of sections to leave out.
        section_prefix: The start of the headings of the only sections to compare with; the
            empty string compares every section.
        decimals: The decimal places the report prints its numbers at.

    Returns:
        One message per difference at `decimals` places; a contrast the report does not print is
        not checked.

    Raises:
        ValueError: If `column_prefix` names the second setting, recomputed second-setting values
            exist, the report prints rows at the scope and section, and no contrast was compared.
    """
    names = tuple(f"{column_prefix}{name}" for name in ("difference", "lower_95", "upper_95"))
    problems = []
    compared = 0
    scored = 0
    for row in contrasts.iter_rows(named=True):
        if row[names[0]] is None:
            continue
        scored += 1
        arm = row["arm"]
        matches = printed.filter(
            pl.col("scope") == scope,
            pl.col("treatment") == arm,
            pl.col("reference") == reference_arm,
            pl.col("n_rows") == site_hours,
            pl.col("section").str.starts_with(section_prefix),
            ~pl.any_horizontal(
                pl.lit(False),
                *(pl.col("section").str.starts_with(prefix) for prefix in other_fit_sections),
            ),
        )
        compared += matches.height
        recomputed = tuple(round(row[name], decimals) for name in names)
        for match in matches.iter_rows(named=True):
            shown = (match["difference"], match["lower_95"], match["upper_95"])
            if recomputed != shown:
                problems.append(
                    f"{arm} - {reference_arm} at scope {scope}: {recomputed} but section "
                    f"{match['section']!r} prints {shown}"
                )
    if (
        column_prefix
        and scored
        and not compared
        and _holds_rows(
            printed=printed,
            scope=scope,
            reference_arm=reference_arm,
            site_hours=site_hours,
            section_prefix=section_prefix,
        )
    ):
        msg = (
            f"the report prints contrasts at scope {scope!r} under {section_prefix!r}, and "
            f"{scored} recomputed contrasts have second-setting values, but none was compared: "
            "the scope, the section prefix, or the excluded sections match no printed row"
        )
        raise ValueError(msg)
    return problems


def _holds_rows(
    *,
    printed: pl.DataFrame,
    scope: str,
    reference_arm: str,
    site_hours: int,
    section_prefix: str,
) -> bool:
    """Return whether the report prints any contrast at a scope, section and number of rows."""
    return not printed.filter(
        pl.col("scope") == scope,
        pl.col("reference") == reference_arm,
        pl.col("n_rows") == site_hours,
        pl.col("section").str.starts_with(section_prefix),
    ).is_empty()


def missing_planned_second_rows(
    *,
    contrasts: pl.DataFrame,
    printed: pl.DataFrame,
    reference_arm: str = REFERENCE_ARM,
    second_scope: str = SECOND_SETTING_SCOPE,
    section_prefix: str = "",
) -> list[str]:
    """List every planned contrast with a second setting that the report prints no row for.

    A report names each planned contrast against ERA5 at the second setting, so a planned contrast
    with none printed means the report and the script disagree about which contrast is planned.

    Args:
        contrasts: The scored contrasts, with `arm`, `planning` and `second_difference`.
        printed: `report_contrasts`'s output for the row set's report.
        reference_arm: The arm every contrast is taken against.
        second_scope: The `Scope` cell of a printed row at the second setting.
        section_prefix: The start of the headings a second-setting row is looked for under; the
            empty string looks in every section.

    Returns:
        One message per planned contrast with a second-setting value and no printed row.
    """
    on_second = printed.filter(
        pl.col("scope") == second_scope,
        pl.col("reference") == reference_arm,
        pl.col("section").str.starts_with(section_prefix),
    )
    printed_arms = set(on_second["treatment"].to_list())
    return [
        f"{row['arm']} - {reference_arm}: planned, but the report prints no row at scope "
        f"{second_scope}"
        for row in contrasts.iter_rows(named=True)
        if row["planning"] == "planned"
        and row["second_difference"] is not None
        and row["arm"] not in printed_arms
    ]


def planned_arms(
    *,
    printed: pl.DataFrame,
    reference_arm: str = REFERENCE_ARM,
    section_prefix: str = PLANNED_SECTION_PREFIX,
) -> set[str]:
    """Return the arms whose contrast against the reference arm the report names before the run.

    Args:
        printed: `report_contrasts`'s output.
        reference_arm: The arm every contrast is taken against.
        section_prefix: The start of the headings of the planned-contrast tables.

    Returns:
        Each treatment arm with a row against the reference arm, scope `all`, under a heading
        that starts with `section_prefix`.
    """
    rows = printed.filter(
        pl.col("section").str.starts_with(section_prefix),
        pl.col("scope") == "all",
        pl.col("reference") == reference_arm,
    )
    return set(rows["treatment"].to_list())


def _printed_planned_row(
    *,
    printed: pl.DataFrame,
    contrast: PlannedContrast,
    site_hours: int,
    scope: str,
    section_prefix: str = PLANNED_SECTION_PREFIX,
) -> dict[str, float] | None:
    """Return the printed row of a planned contrast, or None where the report prints none.

    Args:
        printed: `report_contrasts`'s output.
        contrast: The planned contrast.
        site_hours: The row set's number of site-hours.
        scope: The printed scope to read: `all`, or the row set's second-setting scope.
        section_prefix: The start of the headings of the tables to read from.

    Returns:
        The row's `difference`, `lower_95` and `upper_95`; None if no row matches.

    Raises:
        ValueError: If more than one row matches.
    """
    matches = printed.filter(
        pl.col("section").str.starts_with(section_prefix),
        pl.col("scope") == scope,
        pl.col("treatment") == contrast.treatment.arm,
        pl.col("reference") == contrast.reference.arm,
        pl.col("n_rows") == site_hours,
    )
    if matches.height > 1:
        msg = f"{contrast.label} is printed {matches.height} times at scope {scope}"
        raise ValueError(msg)
    if matches.is_empty():
        return None
    return matches.row(0, named=True)


def unlisted_planned_contrasts(
    *,
    contrasts: Sequence[PlannedContrast],
    printed: pl.DataFrame,
    site_hours: int,
    section_prefix: str = PLANNED_SECTION_PREFIX,
    exploratory: Sequence[tuple[str, str]] = (),
) -> list[str]:
    """List the planned contrasts a report prints that `contrasts` does not hold, and the reverse.

    Args:
        contrasts: The row set's `planned_contrasts`.
        printed: `report_contrasts`'s output.
        site_hours: The row set's number of site-hours.
        section_prefix: The start of the headings of the planned-contrast tables.
        exploratory: Contrasts that a planned-contrast table prints and labels exploratory, each
            as (first arm, second arm); they are not planned, so the script need not list them.

    Returns:
        One message per planned contrast in one place and not the other.
    """
    in_report = printed.filter(
        pl.col("section").str.starts_with(section_prefix),
        pl.col("scope") == "all",
        pl.col("n_rows") == site_hours,
    )
    reported = set(zip(in_report["treatment"], in_report["reference"], strict=True)) - set(
        exploratory
    )
    listed = {(row.treatment.arm, row.reference.arm) for row in contrasts}
    return [
        *(
            f"{t} - {r}: the report prints it as planned, but the script does not list it"
            for t, r in sorted(reported - listed)
        ),
        *(
            f"{t} - {r}: the script lists it as planned, but the report does not print it"
            for t, r in sorted(listed - reported)
        ),
    ]


def check_planned_contrasts(
    *,
    planned: pl.DataFrame,
    printed: pl.DataFrame,
    site_hours: int,
    scope: str = "all",
    column_prefix: str = "",
    section_prefix: str = PLANNED_SECTION_PREFIX,
    decimals: int = PRINT_DECIMALS,
) -> None:
    """Stop unless every planned contrast rounds to the row the report prints for it.

    Args:
        planned: `planned_contrast_rows`'s output, with `arm` and `reference_arm`.
        printed: `report_contrasts`'s output for the row set's report.
        site_hours: The row set's number of site-hours.
        scope: The printed scope to compare with: `all`, or the row set's second-setting scope.
        column_prefix: What precedes `difference`, `lower_95` and `upper_95`: empty for the main
            setting, `second_` for the second. A row whose value is null is skipped.
        section_prefix: The start of the headings of the tables to compare with.
        decimals: The decimal places the report prints its numbers at.

    Raises:
        ValueError: If a planned contrast has a value and the report prints no row for it, or a
            recomputed number differs from the printed one.
    """
    for row in planned.iter_rows(named=True):
        if row[f"{column_prefix}difference"] is None:
            continue
        contrast = PlannedContrast(
            BlockArm(row["arm"], row["label"], row["family"]),
            BlockArm(row["reference_arm"], row["label"], row["family"]),
        )
        shown = _printed_planned_row(
            printed=printed,
            contrast=contrast,
            site_hours=site_hours,
            scope=scope,
            section_prefix=section_prefix,
        )
        if shown is None:
            msg = (
                f"{row['label']} ({row['arm']} - {row['reference_arm']}): planned, but the report "
                f"prints no row at scope {scope}"
            )
            raise ValueError(msg)
        for name in ("difference", "lower_95", "upper_95"):
            assert_matches_printed(
                name=f"{row['label']} ({row['arm']} - {row['reference_arm']}) {scope} {name}",
                recomputed=row[f"{column_prefix}{name}"],
                printed=shown[name],
                decimals=decimals,
            )


def _planned_second_setting(
    *,
    planned: pl.DataFrame,
    contrasts: Sequence[PlannedContrast],
    losses: pl.DataFrame,
    site_hours: int,
) -> pl.DataFrame:
    """Add the second setting's value to each planned contrast whose two arms have one saved.

    Args:
        planned: `planned_contrast_rows`'s output, at the `pooled` setting.
        contrasts: The planned contrasts, in the order of `planned`'s rows.
        losses: The row set's `losses.parquet`.
        site_hours: The row set's number of site-hours.

    Returns:
        The rows with `second_difference`, `second_lower_95` and `second_upper_95`, null for a
        contrast with an arm that has no `sensitivity` losses.
    """
    saved = set(losses.filter(pl.col("setting") == "sensitivity")["arm"].unique().to_list())
    kept = [row for row in contrasts if {row.treatment.arm, row.reference.arm} <= saved]
    columns = ("second_difference", "second_lower_95", "second_upper_95")
    if not kept:
        return planned.with_columns(pl.lit(None, dtype=pl.Float64).alias(name) for name in columns)
    second = planned_contrast_rows(
        losses=losses,
        contrasts=kept,
        setting="sensitivity",
        site_hours=site_hours,
        metric=METRIC,
    ).select(
        "arm",
        "reference_arm",
        second_difference="difference",
        second_lower_95="lower_95",
        second_upper_95="upper_95",
    )
    return planned.join(second, on=["arm", "reference_arm"], how="left")


def near_line() -> pl.Expr:
    """Return the expression that flags an interval with a bound near zero.

    Returns:
        A boolean expression over `lower_95` and `upper_95`: True where the bound nearer zero
        lies within `NEAR_LINE_SHARE` of the interval's width from zero.
    """
    return pl.min_horizontal(
        pl.col("lower_95").abs(), pl.col("upper_95").abs()
    ) <= NEAR_LINE_SHARE * (pl.col("upper_95") - pl.col("lower_95"))


def _second_setting(
    *,
    contrasts: pl.DataFrame,
    arms: tuple[BlockArm, ...],
    losses: pl.DataFrame,
    site_hours: int,
    reference_arm: str = REFERENCE_ARM,
) -> pl.DataFrame:
    """Add the second hyperparameter setting's contrast to each planned or near-line row.

    A row gets one only where its arm and ERA5 both have saved `sensitivity` losses; every other
    row keeps nulls, because this script refits nothing.

    Args:
        contrasts: `block_contrast_rows`'s output with `arm`, `planning` and `near_line`.
        arms: The arms contrasted.
        losses: The row set's `losses.parquet`.
        site_hours: The row set's number of site-hours.
        reference_arm: The arm every contrast is taken against.

    Returns:
        The rows with `second_difference`, `second_lower_95` and `second_upper_95`.
    """
    saved = set(losses.filter(pl.col("setting") == "sensitivity")["arm"].unique().to_list())
    wanted_arms = {
        row["arm"]
        for row in contrasts.iter_rows(named=True)
        if (row["planning"] == "planned" or row["near_line"])
        and row["arm"] in saved
        and reference_arm in saved
    }
    columns = ("second_difference", "second_lower_95", "second_upper_95")
    if not wanted_arms:
        return contrasts.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(name) for name in columns
        )
    second = block_contrast_rows(
        losses=losses,
        arms=[arm for arm in arms if arm.arm in wanted_arms],
        reference_arm=reference_arm,
        setting="sensitivity",
        site_hours=site_hours,
        metric=METRIC,
    ).select(
        "arm",
        second_difference="difference",
        second_lower_95="lower_95",
        second_upper_95="upper_95",
    )
    return contrasts.join(second, on="arm", how="left")


def score_row_set(
    *, row_set: RowSet, losses: pl.DataFrame, report_text: str, report_path: Path
) -> RowSetResult:
    """Score one row set and check the scores against its report.

    Args:
        row_set: The row set to score.
        losses: The row set's `losses.parquet`.
        report_text: The row set's `report.md`.
        report_path: The path `report_text` was read from.

    Returns:
        The row set's scores.

    Raises:
        ValueError: If a leaderboard arm has no printed error, the report has no contrast table,
            or a recomputed error, interval, or contrast differs from the report's, naming every
            difference.
    """
    site_hours, dates = read_heading(report_text=report_text)
    if row_set.leaderboard_section is None and row_set.intervals == "solar":
        errors = report_errors(report_path=report_path, column=row_set.printed_column)
    else:
        errors = printed_table_errors(
            report_text=report_text,
            section_prefix=row_set.leaderboard_section,
            column=row_set.printed_column,
            arm_suffix="",
        )
    printed_errors = {
        **{f"{key}{row_set.arm_suffix}": value for key, value in errors.items()},
        **printed_rebuild_errors(report_text=report_text),
    }
    unprinted = [arm.arm for arm in row_set.leaderboard_arms if arm.arm not in printed_errors]
    if unprinted:
        msg = (
            f"{row_set.label}: the report's {row_set.printed_column!r} column prints no error "
            f"for {unprinted}; its arm names are {sorted(printed_errors)}"
        )
        raise ValueError(msg)
    printed = report_contrasts(
        report_path=report_path,
        extra_headers=(CONTRAST_COLUMNS_WITH_MONTHS,) if row_set.wide_contrast_tables else (),
    )
    if printed.is_empty():
        msg = (
            f"{row_set.label}: no contrast table read from the report; its tables may carry "
            "a `Months` column, which `RowSet.wide_contrast_tables` says"
        )
        raise ValueError(msg)
    reference_arm = row_set.reference_arm
    second_planned_section = row_set.second_planned_section or row_set.planned_section
    planned = planned_arms(
        printed=printed, reference_arm=reference_arm, section_prefix=row_set.planned_section
    )
    absolute = block_leaderboard_rows(
        losses=losses,
        arms=row_set.leaderboard_arms,
        setting="pooled",
        site_hours=site_hours,
        metric=METRIC,
        printed=printed_errors,
        decimals=row_set.printed_decimals,
    )
    contrast_arms = tuple(arm._replace(planned=arm.arm in planned) for arm in row_set.contrast_arms)
    contrasts = block_contrast_rows(
        losses=losses,
        arms=contrast_arms,
        reference_arm=reference_arm,
        setting="pooled",
        site_hours=site_hours,
        metric=METRIC,
    ).with_columns(
        planning=pl.when(pl.col("planned"))
        .then(pl.lit("planned"))
        .when(pl.col("arm").is_in(POST_HOC_ARMS))
        .then(pl.lit("post hoc"))
        .otherwise(pl.lit("exploratory")),
        near_line=near_line(),
    )
    contrasts = _second_setting(
        contrasts=contrasts,
        arms=contrast_arms,
        losses=losses,
        site_hours=site_hours,
        reference_arm=reference_arm,
    )
    planned_rows = planned_contrast_rows(
        losses=losses,
        contrasts=row_set.planned_contrasts,
        setting="pooled",
        site_hours=site_hours,
        metric=METRIC,
    ).with_columns(near_line=near_line())
    planned_rows = _planned_second_setting(
        planned=planned_rows,
        contrasts=row_set.planned_contrasts,
        losses=losses,
        site_hours=site_hours,
    )
    problems = [
        *unlisted_planned_contrasts(
            contrasts=row_set.planned_contrasts,
            printed=printed,
            site_hours=site_hours,
            section_prefix=row_set.planned_section,
            exploratory=row_set.exploratory_in_planned,
        ),
        *check_intervals(
            absolute=absolute,
            report_text=report_text,
            printed=printed_table_intervals(
                report_text=report_text,
                section_prefix=row_set.leaderboard_section,
                column=row_set.printed_column,
                arm_suffix=row_set.arm_suffix,
                intervals=row_set.intervals,
            ),
            decimals=row_set.printed_decimals,
        ),
        *check_contrasts(
            contrasts=contrasts,
            printed=printed,
            site_hours=site_hours,
            reference_arm=reference_arm,
            other_fit_sections=row_set.other_fit_sections,
            decimals=row_set.printed_decimals,
        ),
        *check_contrasts(
            contrasts=contrasts,
            printed=printed,
            site_hours=site_hours,
            scope=row_set.second_scope,
            column_prefix="second_",
            reference_arm=reference_arm,
            other_fit_sections=(),
            section_prefix=row_set.second_section,
            decimals=row_set.printed_decimals,
        ),
        *missing_planned_second_rows(
            contrasts=contrasts,
            printed=printed,
            reference_arm=reference_arm,
            second_scope=row_set.second_scope,
            section_prefix=row_set.second_section,
        ),
    ]
    if problems:
        msg = f"{row_set.label} does not reproduce its report:\n" + "\n".join(problems)
        raise ValueError(msg)
    exploratory_rows = planned_contrast_rows(
        losses=losses,
        contrasts=row_set.exploratory_contrasts,
        setting="pooled",
        site_hours=site_hours,
        metric=METRIC,
    ).with_columns(planned=pl.lit(False), near_line=near_line())
    check_planned_contrasts(
        planned=planned_rows,
        printed=printed,
        site_hours=site_hours,
        section_prefix=row_set.planned_section,
        decimals=row_set.printed_decimals,
    )
    check_planned_contrasts(
        planned=planned_rows,
        printed=printed,
        site_hours=site_hours,
        scope=row_set.second_scope,
        column_prefix="second_",
        section_prefix=second_planned_section,
        decimals=row_set.printed_decimals,
    )
    return RowSetResult(
        row_set=row_set,
        dates=dates,
        site_hours=site_hours,
        absolute=absolute,
        contrasts=contrasts,
        planned=planned_rows,
        exploratory=exploratory_rows,
    )


def _signed(value: float | None) -> str:
    """Format a difference with its sign, or a dash where there is none."""
    return "—" if value is None else f"{value:+.{PRINT_DECIMALS}f}"


def _interval(*, low: float | None, high: float | None) -> str:
    """Format a signed interval as `[low, high]`, or a dash where there is none."""
    if low is None or high is None:
        return "—"
    return f"[{low:+.{PRINT_DECIMALS}f}, {high:+.{PRINT_DECIMALS}f}]"


def _second_setting_cell(*, row: dict) -> str:
    """Format a contrast's second setting as `value [low, high]`, or one dash where it has none."""
    if row["second_difference"] is None:
        return "—"
    return (
        f"{_signed(row['second_difference'])} "
        f"{_interval(low=row['second_lower_95'], high=row['second_upper_95'])}"
    )


def _contrast_line(*, row: dict, second: str) -> str:
    """Format one row of a table of contrasts between two products."""
    return (
        f"| {row['label']} | {_signed(row['difference'])} | "
        f"{_interval(low=row['lower_95'], high=row['upper_95'])} | "
        f"{'yes' if row['near_line'] else 'no'} | {second} |"
    )


def render_report(
    *,
    results: list[RowSetResult],
    title: str = REPORT_TITLE,
    introduction: str = REPORT_INTRODUCTION,
) -> str:
    """Write the leaderboard report, one section per row set.

    Args:
        results: Each row set's scores.
        title: The report's heading.
        introduction: The paragraph under the heading.

    Returns:
        The report's markdown.
    """
    lines = [f"# {title}", "", introduction, ""]
    for result in results:
        lines += [
            (
                f"### {result.row_set.label}: {result.dates}, {result.site_hours:,} "
                f"{result.row_set.hours_unit}"
            ),
            "",
            "#### Mean absolute error",
            "",
            "| Arm | Error | 95% interval | Reference row |",
            "|---|---|---|---|",
        ]
        for row in result.absolute.iter_rows(named=True):
            post_hoc = POST_HOC_SUFFIX if row["arm"] in POST_HOC_ARMS else ""
            lines.append(
                f"| {row['label']}{post_hoc} | {row['value']:.{PRINT_DECIMALS}f} | "
                f"[{row['lower_95']:.{PRINT_DECIMALS}f}, {row['upper_95']:.{PRINT_DECIMALS}f}] | "
                f"{'yes' if row['reference'] else 'no'} |"
            )
        lines += [
            "",
            "#### Mean absolute error minus ERA5's",
            "",
            CONTRAST_HEADER,
            "|---|---|---|---|---|---|",
        ]
        for row in result.contrasts.iter_rows(named=True):
            second = _second_setting_cell(row=row)
            lines.append(
                f"| {row['label']} | {_signed(row['difference'])} | "
                f"{_interval(low=row['lower_95'], high=row['upper_95'])} | {row['planning']} | "
                f"{'yes' if row['near_line'] else 'no'} | {second} |"
            )
        lines += [
            "",
            "#### Planned contrasts, first product minus second",
            "",
            PLANNED_HEADER,
            "|---|---|---|---|---|",
        ]
        for row in result.planned.iter_rows(named=True):
            lines.append(_contrast_line(row=row, second=_second_setting_cell(row=row)))
        if not result.exploratory.is_empty():
            lines += [
                "",
                "#### Exploratory contrasts, first product minus second",
                "",
                EXPLORATORY_HEADER,
                "|---|---|---|---|---|",
            ]
            lines += [
                _contrast_line(row=row, second="—")
                for row in result.exploratory.iter_rows(named=True)
            ]
        lines.append("")
    return "\n".join(lines)


def intervals_frame(*, results: list[RowSetResult]) -> pl.DataFrame:
    """Stack every row set's scores into one long frame, in `studies.page_numbers`' convention.

    Args:
        results: Each row set's scores.

    Returns:
        One row per (row set, section, setting, arm). `section` is `Mean absolute error`,
        `Mean absolute error minus ERA5's`, `PLANNED_CONTRAST_SECTION` (a planned contrast, with its
        second arm in `reference`), or `EXPLORATORY_CONTRAST_SECTION` (the same for an exploratory
        contrast, at the `pooled` setting only); `setting` is `pooled`, or `sensitivity` for a
        contrast's second setting; `treatment` is the arm and `reference` is null for an absolute
        row. `value`, `lower` and `upper` are in percentage points of capacity at full precision,
        `level` is 95, and `n_rows` is the row set's site-hours. `row_set`, `label`, `planning`
        and `near_line` (both null for an absolute row) say which row and how the report labels it.
    """
    frames = []
    for result in results:
        common = {
            "row_set": pl.lit(result.row_set.key),
            "scope": pl.lit("all"),
            "level": pl.lit(95.0),
            "n_rows": pl.lit(result.site_hours),
        }
        contrasts = result.contrasts
        frames += [
            result.absolute.select(
                **common,
                section=pl.lit(ABSOLUTE_SECTION),
                setting=pl.lit("pooled"),
                treatment="arm",
                reference=pl.lit(None, dtype=pl.String),
                value="value",
                lower="lower_95",
                upper="upper_95",
                label="label",
                planning=pl.lit(None, dtype=pl.String),
                near_line=pl.lit(None, dtype=pl.Boolean),
            ),
            contrasts.select(
                **common,
                section=pl.lit(CONTRAST_SECTION),
                setting=pl.lit("pooled"),
                treatment="arm",
                reference=pl.lit(result.row_set.reference_arm),
                value="difference",
                lower="lower_95",
                upper="upper_95",
                label="label",
                planning="planning",
                near_line="near_line",
            ),
            contrasts.filter(pl.col("second_difference").is_not_null()).select(
                **common,
                section=pl.lit(CONTRAST_SECTION),
                setting=pl.lit(SECOND_SETTING_SCOPE),
                treatment="arm",
                reference=pl.lit(result.row_set.reference_arm),
                value="second_difference",
                lower="second_lower_95",
                upper="second_upper_95",
                label="label",
                planning="planning",
                near_line="near_line",
            ),
            result.planned.select(
                **common,
                section=pl.lit(PLANNED_CONTRAST_SECTION),
                setting=pl.lit("pooled"),
                treatment="arm",
                reference="reference_arm",
                value="difference",
                lower="lower_95",
                upper="upper_95",
                label="label",
                planning=pl.lit("planned"),
                near_line="near_line",
            ),
            result.exploratory.select(
                **common,
                section=pl.lit(EXPLORATORY_CONTRAST_SECTION),
                setting=pl.lit("pooled"),
                treatment="arm",
                reference="reference_arm",
                value="difference",
                lower="lower_95",
                upper="upper_95",
                label="label",
                planning=pl.lit("exploratory"),
                near_line="near_line",
            ),
            result.planned.filter(pl.col("second_difference").is_not_null()).select(
                **common,
                section=pl.lit(PLANNED_CONTRAST_SECTION),
                setting=pl.lit(SECOND_SETTING_SCOPE),
                treatment="arm",
                reference="reference_arm",
                value="second_difference",
                lower="second_lower_95",
                upper="second_upper_95",
                label="label",
                planning=pl.lit("planned"),
                near_line="near_line",
            ),
        ]
    return pl.concat(frames)


def write_outputs(
    *,
    results: list[RowSetResult],
    output_dir: Path,
    title: str = REPORT_TITLE,
    introduction: str = REPORT_INTRODUCTION,
) -> None:
    """Write `report.md` and `intervals.parquet` into a new folder.

    Args:
        results: Each row set's scores.
        output_dir: The folder to create.
        title: The report's heading.
        introduction: The paragraph under the heading.

    Raises:
        FileExistsError: If the folder exists, so an earlier run's numbers are never overwritten.
    """
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.md").write_text(
        render_report(results=results, title=title, introduction=introduction)
    )
    intervals_frame(results=results).write_parquet(output_dir / "intervals.parquet")


def run(
    *,
    row_sets: Sequence[RowSet],
    output_dir: Path,
    title: str,
    introduction: str,
    description: str | None,
) -> int:
    """Score each row set, check it against its report, and write the results.

    Reads the command line: `--check-only` verifies every number and writes nothing.

    Args:
        row_sets: The row sets to score, in the order the leaderboard stacks them.
        output_dir: The write-once folder for `report.md` and `intervals.parquet`.
        title: The report's heading.
        introduction: The paragraph under the report's heading.
        description: The command's help text.

    Returns:
        The process exit code: 1 where `output_dir` exists, else 0.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Read and verify every number against the reports, and write nothing.",
    )
    args = parser.parse_args()
    if not args.check_only and output_dir.exists():
        _LOG.error("%s exists; this script never overwrites it", output_dir)
        return 1
    results = []
    for row_set in row_sets:
        report_path = row_set.directory / "report.md"
        losses = pl.read_parquet(row_set.directory / "losses.parquet")
        results.append(
            score_row_set(
                row_set=row_set,
                losses=losses,
                report_text=report_path.read_text(),
                report_path=report_path,
            )
        )
        _LOG.info("%s reproduces its report", row_set.label)
    if args.check_only:
        return 0
    write_outputs(results=results, output_dir=output_dir, title=title, introduction=introduction)
    _LOG.info("wrote %s", output_dir)
    return 0


def main() -> int:
    """Score the four past-solar row sets, check them against their reports, and write them."""
    return run(
        row_sets=ROW_SETS,
        output_dir=SOLAR_LEADERBOARD_DIR,
        title=REPORT_TITLE,
        introduction=REPORT_INTRODUCTION,
        description=__doc__,
    )


if __name__ == "__main__":
    sys.exit(main())
