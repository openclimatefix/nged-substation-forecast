"""Draw the past-solar page's leaderboard (Figure 1) and contrasts (Figure 2), four row sets each.

Both figures read `past_solar_leaderboard.py`'s write-once folder, `intervals.parquet` for every
number and `report.md` for the check. **The script stops before drawing unless every number it
draws matches the number the report prints**, at the report's own precision: each arm's error and
interval, each contrast, each planned contrast, and each second-setting value. Nothing is
bootstrapped or refitted here.

Each row set is one block. Its rows are scored on that row set's own common rows, so a reader
compares products within a block only. CAMS and ERA5 are repeated in every block as hollow
reference rows. Figure 2 adds, under each block's contrasts against ERA5, that block's planned
contrasts.

Generators do not appear in either figure.

Run it with `uv run python studies/beam_diffuse_split/past_solar_leaderboard_charts.py`, after
`past_solar_leaderboard.py`. Optimise each SVG with `npx svgo@4 --multipass --precision=1
--final-newline` before committing it.
"""

import argparse
import datetime
import logging
import re
import sys
from pathlib import Path
from typing import Final, NamedTuple

import altair as alt
import polars as pl
from figure_numbers import FIGURE_NUMBERS
from past_solar_leaderboard import (
    ABSOLUTE_SECTION,
    CONTRAST_SECTION,
    PLANNED_CONTRAST_SECTION,
    POST_HOC_ARMS,
    REFERENCE_ARM,
    ROW_SETS,
    RowSet,
)
from sources import SOLAR_LEADERBOARD_DIR
from studies.charts import (
    POST_HOC_SUFFIX,
    RowSetBlock,
    assert_matches_printed,
    stacked_contrasts,
    stacked_leaderboard,
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

ASSETS_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "docs" / "studies" / "assets"
"""Where the write-up's images live."""

BLOCK_LABELS: Final[dict[str, str]] = {
    "main": "Main",
    "extra": "Extra",
    "ens": "ENS",
    "station": "Stations",
}
"""Each row set's block label, the term the page uses for the row set."""

DISPLAY_LABELS: Final[dict[str, str]] = {
    "ECMWF ENS (control member, T+3 band)": "ECMWF ENS control (T+3 band)",
    "UKV rebuilt from its snapshots": "UKV, snapshot mean",
    "UKV, both snapshots as separate inputs": "UKV, both snapshots",
}
"""Labels the report prints that are too long for one line of a block's row-label column."""

CAPACITY: Final[str] = "Capacity is each generator's 99th-percentile output."
DOTS: Final[str] = (
    "Dot: estimate. Line: 95% interval from resampling whole months and a fitting seed."
)
SCOPE: Final[str] = "Six solar farms in Lincolnshire."
BLOCKS_NOT_COMPARABLE: Final[str] = (
    "Compare products only within a block: each block is scored on its own rows, so an error in "
    "one block is not comparable with an error in another."
)

ENS_LEAD: Final[str] = (
    "ECMWF ENS is a forecast 5 to 20 hours ahead from a 00 UTC run; ERA5's radiation is 1 to 12 "
    "hours ahead."
)
STATION_SCOPE: Final[str] = (
    "The station rows rest on one pyranometer: all six generators take the same nearest radiation "
    "station, 17 to 31 km away."
)
UNEQUAL_LEADS: Final[str] = (
    "Several planned contrasts compare unequal leads: ICON global is served up to 6 hours ahead "
    "against ICON-EU's 3, and at equal leads its gap closes to 0.02 points; ICON-DREAM-EU's 1 to 3 "
    "hours against ERA5's 1 to 12; ECMWF-IFS-HRES never shorter than ICON-EU's; KNMI "
    "HARMONIE-AROME's lead not measured."
)
NOT_GRIDDED: Final[str] = (
    "In the Stations block, CAMS combined with a station observation is not a gridded product, "
    "so the title does not cover it."
)
EXTRA_FOLDS: Final[str] = (
    "Extra block: 36.9% of its hours fall in calendar months unseen in training, so its errors "
    "run about 0.12 to 0.20 points high; see Limitations."
)
NESTED_BLOCKS: Final[str] = (
    "The extra, ENS and station rows are almost entirely subsets of the main rows."
)
POST_HOC_NOTE: Final[str] = (
    "Rows marked (post hoc) were added after the first run: two ways of rebuilding UKV's hourly "
    "value from its snapshots."
)
CAMS_EXPLORATORY: Final[str] = (
    "The CAMS row is exploratory: the study plan names CAMS against ERA5 as a planned contrast on "
    "no row set."
)

_BLOCK_HEADING: Final[re.Pattern[str]] = re.compile(
    r"^### (.+?): (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2}), ([\d,]+) site-hours$"
)
_SECOND: Final[re.Pattern[str]] = re.compile(r"^(\S+) \[(\S+), (\S+)\]$")
_INTERVAL: Final[re.Pattern[str]] = re.compile(r"^\[(\S+), (\S+)\]$")


class PrintedRow(NamedTuple):
    """One row of a report table, as the report prints it."""

    value: float
    lower: float
    upper: float
    second: tuple[float, float, float] | None


class PrintedBlock(NamedTuple):
    """One row set's block of the leaderboard report."""

    first_day: str
    last_day: str
    site_hours: int
    tables: dict[str, dict[str, PrintedRow]]


def _printed_row(*, cells: list[str], second_cell: str | None) -> PrintedRow:
    """Read one report table row's value, interval, and second setting, if it prints one."""
    interval = _INTERVAL.match(cells[2])
    if interval is None:
        msg = f"not an interval: {cells[2]!r}"
        raise ValueError(msg)
    second = None
    if second_cell is not None and (match := _SECOND.match(second_cell)) is not None:
        second = (float(match[1]), float(match[2]), float(match[3]))
    return PrintedRow(float(cells[1]), float(interval[1]), float(interval[2]), second)


def read_report(*, report_text: str) -> dict[str, PrintedBlock]:
    """Read every block of the leaderboard report into its three tables.

    Args:
        report_text: The write-once `report.md`.

    Returns:
        Each block's heading text (`Main rows`) to its dates, site-hours, and tables, each table
        keyed by the section heading and then by the row's label.

    Raises:
        ValueError: If a table row's interval is malformed.
    """
    blocks: dict[str, PrintedBlock] = {}
    label = ""
    section = ""
    for line in report_text.splitlines():
        if heading := _BLOCK_HEADING.match(line):
            label = heading[1]
            blocks[label] = PrintedBlock(
                heading[2], heading[3], int(heading[4].replace(",", "")), {}
            )
        elif line.startswith("#### "):
            section = line.removeprefix("#### ")
            if label:
                blocks[label].tables[section] = {}
        elif line.startswith("| ") and section and label and not line.startswith("| Arm"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if cells[0] in ("Contrast", "---") or set(cells[0]) == {"-"}:
                continue
            second_cell = cells[-1] if section != ABSOLUTE_SECTION else None
            name = (
                cells[0].removesuffix(POST_HOC_SUFFIX) if section == ABSOLUTE_SECTION else cells[0]
            )
            blocks[label].tables[section][name] = _printed_row(cells=cells, second_cell=second_cell)
    return blocks


def _month_year(*, iso_day: str) -> str:
    """Return `2022-12-01` as `December 2022`."""
    return datetime.date.fromisoformat(iso_day).strftime("%B %Y")


def _check(*, name: str, recomputed: float | None, printed: float) -> None:
    """Stop unless a number read from `intervals.parquet` is the number the report prints."""
    if recomputed is None:
        msg = f"{name}: the report prints {printed} but intervals.parquet holds no value"
        raise ValueError(msg)
    assert_matches_printed(name=name, recomputed=recomputed, printed=printed)


def _check_row(*, name: str, row: dict[str, float | None], printed: PrintedRow) -> None:
    """Check a drawn row's value and both interval ends against the report's row."""
    _check(name=f"{name} value", recomputed=row["value"], printed=printed.value)
    _check(name=f"{name} lower", recomputed=row["lower"], printed=printed.lower)
    _check(name=f"{name} upper", recomputed=row["upper"], printed=printed.upper)


def _one(*, frame: pl.DataFrame, **matches: str) -> dict[str, float | None]:
    """Return the single row of `intervals.parquet` that matches every column given.

    Args:
        frame: The rows of `intervals.parquet`.
        **matches: Column name and the value it must equal.

    Returns:
        The one matching row, by column name.

    Raises:
        ValueError: If no row or more than one row matches.
    """
    kept = frame.filter(*(pl.col(name) == value for name, value in matches.items()))
    if kept.height != 1:
        msg = f"expected exactly one intervals row for {matches}, found {kept.height}"
        raise ValueError(msg)
    return kept.row(0, named=True)


def _second(*, frame: pl.DataFrame, section: str, arm: str, reference: str) -> dict | None:
    """Return the second-setting row of a contrast, or None where none was saved."""
    kept = frame.filter(
        pl.col("section") == section,
        pl.col("setting") == "sensitivity",
        pl.col("treatment") == arm,
        pl.col("reference") == reference,
    )
    if kept.height > 1:
        msg = f"{arm} - {reference} has {kept.height} second-setting rows in {section!r}"
        raise ValueError(msg)
    return kept.row(0, named=True) if kept.height else None


def _check_second(*, name: str, second: dict | None, printed: PrintedRow) -> None:
    """Check a second-setting value against the report's printed second setting."""
    if second is None:
        if printed.second is not None:
            msg = f"{name}: the report prints a second setting but intervals.parquet has none"
            raise ValueError(msg)
        return
    if printed.second is None:
        msg = f"{name}: intervals.parquet has a second setting but the report prints none"
        raise ValueError(msg)
    for part, drawn, shown in zip(
        ("value", "lower", "upper"),
        (second["value"], second["lower"], second["upper"]),
        printed.second,
        strict=True,
    ):
        _check(name=f"{name} second setting {part}", recomputed=drawn, printed=shown)


def absolute_rows(
    *, frame: pl.DataFrame, row_set: RowSet, printed: dict[str, PrintedRow]
) -> pl.DataFrame:
    """Return the row set's leaderboard rows, best first, each checked against the report.

    Args:
        frame: The row set's rows of `intervals.parquet`.
        row_set: The row set.
        printed: The report's absolute-error table for the row set, by label.

    Returns:
        One row per leaderboard arm with `arm`, `label`, `family`, `reference`, `planned`,
        `value`, `lower_95` and `upper_95`, in percentage points of capacity.

    Raises:
        ValueError: If an arm has no row, or a number differs from the report's.
    """
    records = []
    for arm in row_set.leaderboard_arms:
        row = _one(frame=frame, section=ABSOLUTE_SECTION, setting="pooled", treatment=arm.arm)
        _check_row(name=arm.label, row=row, printed=printed[arm.label])
        records.append(
            {
                "arm": arm.arm,
                "label": DISPLAY_LABELS.get(arm.label, arm.label)
                + (POST_HOC_SUFFIX if arm.arm in POST_HOC_ARMS else ""),
                "family": arm.family,
                "reference": arm.reference,
                "planned": False,
                "value": row["value"],
                "lower_95": row["lower"],
                "upper_95": row["upper"],
            }
        )
    return pl.DataFrame(records).sort("value")


def contrast_rows(
    *,
    frame: pl.DataFrame,
    row_set: RowSet,
    order: list[str],
    printed: dict[str, PrintedRow],
) -> pl.DataFrame:
    """Return the row set's contrasts against ERA5, each checked against the report.

    Args:
        frame: The row set's rows of `intervals.parquet`.
        row_set: The row set.
        order: The arms in leaderboard order; a contrast arm the leaderboard does not hold (a UKV
            rebuild) follows every arm it does hold.
        printed: The report's table of contrasts against ERA5, by label.

    Returns:
        One row per contrast arm with `arm`, `label`, `family`, `reference`, `planned`,
        `difference`, `lower_95`, `upper_95` and `second_difference`, in percentage points.

    Raises:
        ValueError: If an arm has no row, or a number differs from the report's.
    """
    rank = {
        arm.arm: order.index(arm.arm) if arm.arm in order else len(order) + index
        for index, arm in enumerate(row_set.contrast_arms)
    }
    ordered = sorted(row_set.contrast_arms, key=lambda arm: rank[arm.arm])
    records = []
    for arm in ordered:
        row = _one(
            frame=frame,
            section=CONTRAST_SECTION,
            setting="pooled",
            treatment=arm.arm,
            reference=REFERENCE_ARM,
        )
        name = arm.label
        _check_row(name=name, row=row, printed=printed[name])
        second = _second(
            frame=frame,
            section=CONTRAST_SECTION,
            arm=arm.arm,
            reference=REFERENCE_ARM,
        )
        _check_second(name=name, second=second, printed=printed[name])
        records.append(
            {
                "arm": arm.arm,
                "label": DISPLAY_LABELS.get(name, name)
                + (POST_HOC_SUFFIX if arm.arm in POST_HOC_ARMS else ""),
                "family": arm.family,
                "reference": arm.reference,
                "planned": row["planning"] == "planned",
                "difference": row["value"],
                "lower_95": row["lower"],
                "upper_95": row["upper"],
                "second_difference": None if second is None else second["value"],
            }
        )
    return pl.DataFrame(records, schema_overrides={"second_difference": pl.Float64})


def planned_rows(
    *, frame: pl.DataFrame, row_set: RowSet, printed: dict[str, PrintedRow]
) -> pl.DataFrame:
    """Return the row set's planned contrasts, each checked against the report.

    Args:
        frame: The row set's rows of `intervals.parquet`.
        row_set: The row set.
        printed: The report's table of planned contrasts, by label.

    Returns:
        One row per planned contrast, in the order the row set lists them, with the columns of
        `studies.charts.planned_contrast_rows` and `second_difference`.

    Raises:
        ValueError: If a contrast has no row, or a number differs from the report's.
    """
    records = []
    for contrast in row_set.planned_contrasts:
        row = _one(
            frame=frame,
            section=PLANNED_CONTRAST_SECTION,
            setting="pooled",
            treatment=contrast.treatment.arm,
            reference=contrast.reference.arm,
        )
        _check_row(name=contrast.label, row=row, printed=printed[contrast.label])
        second = _second(
            frame=frame,
            section=PLANNED_CONTRAST_SECTION,
            arm=contrast.treatment.arm,
            reference=contrast.reference.arm,
        )
        _check_second(name=contrast.label, second=second, printed=printed[contrast.label])
        records.append(
            {
                "arm": contrast.treatment.arm,
                "reference_arm": contrast.reference.arm,
                "label": contrast.label,
                "family": contrast.treatment.family,
                "reference": False,
                "planned": True,
                "difference": row["value"],
                "lower_95": row["lower"],
                "upper_95": row["upper"],
                "second_difference": None if second is None else second["value"],
            }
        )
    return pl.DataFrame(records, schema_overrides={"second_difference": pl.Float64})


def build_blocks(
    *, intervals: pl.DataFrame, report: dict[str, PrintedBlock]
) -> tuple[list[RowSetBlock], list[RowSetBlock]]:
    """Build the leaderboard's blocks and the contrast chart's blocks, checked against the report.

    Args:
        intervals: The write-once `intervals.parquet`.
        report: `read_report`'s output.

    Returns:
        The leaderboard blocks and the contrast blocks, both in the order of `ROW_SETS`.

    Raises:
        ValueError: If a row set's site-hours or a drawn number disagrees with the report.
    """
    leaderboard_blocks = []
    contrast_blocks = []
    for row_set in ROW_SETS:
        frame = intervals.filter(pl.col("row_set") == row_set.key)
        printed = report[row_set.label]
        site_hours = printed.site_hours
        if set(frame["n_rows"].to_list()) != {site_hours}:
            msg = f"{row_set.label}: intervals.parquet disagrees with {site_hours:,} site-hours"
            raise ValueError(msg)
        dates = (
            f"{_month_year(iso_day=printed.first_day)} to {_month_year(iso_day=printed.last_day)}"
        )
        absolute = absolute_rows(
            frame=frame, row_set=row_set, printed=printed.tables[ABSOLUTE_SECTION]
        )
        leaderboard_blocks.append(
            RowSetBlock(
                label=BLOCK_LABELS[row_set.key], dates=dates, site_hours=site_hours, rows=absolute
            )
        )
        contrasts = contrast_rows(
            frame=frame,
            row_set=row_set,
            order=absolute["arm"].to_list(),
            printed=printed.tables[CONTRAST_SECTION],
        )
        planned = planned_rows(
            frame=frame, row_set=row_set, printed=printed.tables[PLANNED_CONTRAST_SECTION]
        )
        contrast_blocks.append(
            RowSetBlock(
                label=BLOCK_LABELS[row_set.key],
                dates=dates,
                site_hours=site_hours,
                rows=contrasts,
                planned_rows=planned,
            )
        )
    return leaderboard_blocks, contrast_blocks


def contrasts_not_comparable(*, cams_differences: list[float]) -> str:
    """Say that a difference from ERA5 is not comparable across blocks, with the measured spread.

    Args:
        cams_differences: CAMS's error minus ERA5's in each block, in points of capacity.

    Returns:
        The subtitle sentence, quoting how far the same contrast moves between blocks.
    """
    spread = max(cams_differences) - min(cams_differences)
    return (
        "A difference from ERA5 is not comparable across blocks either: the same CAMS minus "
        f"ERA5 contrast moves by as much as {spread:.1f} points between blocks."
    )


def leaderboard_figure(*, blocks: list[RowSetBlock]) -> alt.VConcatChart:
    """Draw Figure 1, the leaderboard of the four row sets."""
    return stacked_leaderboard(
        blocks=blocks,
        number=FIGURE_NUMBERS["leaderboard"],
        title=(
            "CAMS has the lowest error of the gridded products tested on each of the four row sets"
        ),
        subtitle=[
            "Each product's own mean absolute error, sorted best first within its block.",
            BLOCKS_NOT_COMPARABLE,
            NOT_GRIDDED,
            EXTRA_FOLDS,
            (
                "Overlapping intervals do not make two products equal: the intervals are wide "
                "mainly because every product's error swings together from month to month, a "
                f"swing that Figure {FIGURE_NUMBERS['contrasts']}'s paired contrasts cancel."
            ),
            ENS_LEAD,
            STATION_SCOPE,
            POST_HOC_NOTE,
            DOTS,
            CAPACITY,
            SCOPE,
        ],
    )


def contrasts_figure(*, blocks: list[RowSetBlock]) -> alt.VConcatChart:
    """Draw Figure 2, the contrasts against ERA5 with each row set's planned contrasts."""
    cams = [
        row["difference"]
        for block in blocks
        for row in block.rows.iter_rows(named=True)
        if row["arm"] == "cams_global"
    ]
    low, high = (f"{abs(value):.1f}" for value in (max(cams), min(cams)))
    return stacked_contrasts(
        blocks=blocks,
        number=FIGURE_NUMBERS["contrasts"],
        title=f"CAMS beats ERA5 by {low} to {high} points in each block (exploratory)",
        subtitle=[
            (
                "Top panel of each block: each product's mean absolute error minus ERA5's. Lower "
                "panel: that row set's planned contrasts, the first product's error minus the "
                "second's."
            ),
            BLOCKS_NOT_COMPARABLE,
            NESTED_BLOCKS,
            contrasts_not_comparable(cams_differences=cams),
            CAMS_EXPLORATORY,
            ENS_LEAD,
            STATION_SCOPE,
            UNEQUAL_LEADS,
            DOTS,
            CAPACITY,
            SCOPE,
        ],
    )


def main() -> int:
    """Read the leaderboard folder, check every number against its report, and write two SVGs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    report = read_report(report_text=(SOLAR_LEADERBOARD_DIR / "report.md").read_text())
    intervals = pl.read_parquet(SOLAR_LEADERBOARD_DIR / "intervals.parquet")
    leaderboard_blocks, contrast_blocks = build_blocks(intervals=intervals, report=report)
    charts = {
        "sunshine_leaderboard": leaderboard_figure(blocks=leaderboard_blocks),
        "sunshine_contrasts": contrasts_figure(blocks=contrast_blocks),
    }
    for name, chart in charts.items():
        path = ASSETS_DIR / f"{name}.svg"
        chart.save(path)
        _LOG.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
