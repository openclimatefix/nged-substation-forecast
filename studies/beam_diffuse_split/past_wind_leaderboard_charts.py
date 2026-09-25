"""Draw the past-wind page's leaderboard (Figure 1) and contrasts (Figure 2), four row sets each.

Both figures read `past_wind_leaderboard.py`'s write-once folder, `intervals.parquet` for every
number and `report.md` for the check. **The script stops before drawing unless every number it
draws matches the number the report prints**, at the report's own precision. Nothing is
bootstrapped or refitted here. The reading and checking code is that of
`past_solar_leaderboard_charts.py`.

Each row set is one block: the main rows, the ICON-DREAM-EU rows, the ECMWF rows, and the
weather-station rows. A block's rows are scored on that row set's own common farm-hours, so a
reader compares arms within a block only. ERA5 is repeated in every block as a hollow reference
row. In the station block the reference is ERA5's 10 m wind, because the station measures at 10 m,
and the block's contrast panel names the arm each planned contrast is against. Figure 2 adds,
under each block's contrasts, that block's planned contrasts.

Each block's label states its wind heights, and the caption states the share of its scored rows
that fall in a calendar month with no training row in their fold, under the published folds.
`UNCOVERED_MONTH_SHARES` holds those shares. The script stops where one is unset, so a block is
never drawn without its share.

Generators do not appear in either figure.

Run it with `uv run python studies/beam_diffuse_split/past_wind_leaderboard_charts.py`, after
`past_wind_leaderboard.py`. Optimise each SVG with `npx svgo@4 --multipass --precision=1
--final-newline` before committing it.
"""

import argparse
import logging
import sys
from typing import Final, NamedTuple

import altair as alt
import polars as pl
from figure_numbers import WIND_FIGURE_NUMBERS
from past_solar_leaderboard import ABSOLUTE_SECTION, PLANNED_CONTRAST_SECTION, contrast_section
from past_solar_leaderboard_charts import (
    ASSETS_DIR,
    absolute_rows,
    contrast_rows,
    month_year,
    planned_rows,
    read_report,
)
from past_wind_leaderboard import BLOCK_SETTINGS, ROW_SETS
from sources import WIND_LEADERBOARD_DIR
from studies.charts import RowSetBlock, stacked_contrasts, stacked_leaderboard

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

BLOCK_LABELS: Final[dict[str, str]] = {
    "main": "Main",
    "icon_dream_eu": "ICON-DREAM-EU",
    "ecmwf": "ECMWF",
    "station": "Station",
}
"""Each row set's block label, the term the page uses for the row set."""

UNCOVERED_MONTH_SHARES: Final[dict[str, float | None]] = {
    "main": 16.2,
    "icon_dream_eu": 25.1,
    "ecmwf": 0.0,
    "station": 0.0,
}
"""Each block's share, in percent, of scored rows in a month with no training row in their fold.

The shares come from the folds saved with each row set's published losses (main 8,603 of 52,996
rows, ICON-DREAM-EU 12,570 of 50,041, ECMWF 0 of 43,555, station 0 of 34,156). They describe the
published folds, which rotated folds would cover. `uncovered_month_note` stops on a `None`.
"""

UNMEASURED_REFIT: Final[frozenset[str]] = frozenset({"icon_dream_eu"})
"""Blocks whose fold-covering refit has not been measured, so the caption says so."""

CAPACITY: Final[str] = "Capacity is each generator's 99th-percentile output."
DOTS: Final[str] = "Dot: estimate. Line: 95% interval from resampling whole months."
SCOPE: Final[str] = "Three wind farms in Lincolnshire."
BLOCKS_NOT_COMPARABLE: Final[str] = (
    "Compare arms only within a block: each block is scored on its own rows, so an error in one "
    "block is not comparable with an error in another."
)
REFERENCE_ROW_NOTE: Final[str] = (
    "Lighter, hollow rows are ERA5, repeated in every block as a yardstick and scored on that "
    "block's own rows; in the station block they are ERA5's 10 m wind."
)
CONTRAST_REFERENCE_ROW_NOTE: Final[str] = (
    "Every block's contrasts are against ERA5, scored on that block's own rows, except the station "
    "block's, which are against ERA5's 10 m wind."
)
STATION_SCOPE: Final[str] = (
    "The station rows rest on one nearby weather station per farm, and cover fewer months than the "
    "other blocks."
)


def uncovered_month_note(*, shares: dict[str, float | None]) -> list[str]:
    """Say, for each block, what share of its scored rows is in a month its fold never trains on.

    Args:
        shares: Each row set's key to its share in percent, or `None` where it is unset.

    Returns:
        One caption line per block, in the order of `ROW_SETS`.

    Raises:
        ValueError: If a block's share is unset.
    """
    unset = [row_set.label for row_set in ROW_SETS if shares[row_set.key] is None]
    if unset:
        msg = f"the uncovered-month share is not set for {unset}; read it from the fold report"
        raise ValueError(msg)
    return [
        (
            f"{BLOCK_LABELS[row_set.key]}: {shares[row_set.key]:.1f}% of scored rows are in a "
            "calendar month with no training row in their fold"
            + (
                "; the effect of covering those months is not yet measured."
                if row_set.key in UNMEASURED_REFIT
                else "."
            )
        )
        for row_set in ROW_SETS
    ]


class _CommonFields(NamedTuple):
    """The fields a block's leaderboard and contrast versions share."""

    label: str
    dates: str
    site_hours: int
    hours_unit: str
    reference_name: str


def _block(
    *, common: _CommonFields, rows: pl.DataFrame, planned: pl.DataFrame | None = None
) -> RowSetBlock:
    """Return a block holding `rows` and, for the contrast chart, its planned contrasts."""
    return RowSetBlock(
        label=common.label,
        dates=common.dates,
        site_hours=common.site_hours,
        rows=rows,
        planned_rows=planned,
        hours_unit=common.hours_unit,
        reference_name=common.reference_name,
    )


def build_blocks(
    *, intervals: pl.DataFrame, report: dict
) -> tuple[list[RowSetBlock], list[RowSetBlock]]:
    """Build the leaderboard's blocks and the contrast chart's blocks, checked against the report.

    Args:
        intervals: The write-once `intervals.parquet`.
        report: `read_report`'s output.

    Returns:
        The leaderboard blocks and the contrast blocks, both in the order of `ROW_SETS`.

    Raises:
        ValueError: If a row set's row count or a drawn number disagrees with the report.
    """
    leaderboard_blocks = []
    contrast_blocks = []
    for row_set in ROW_SETS:
        frame = intervals.filter(pl.col("row_set") == row_set.key)
        printed = report[row_set.label]
        if set(frame["n_rows"].to_list()) != {printed.site_hours}:
            msg = (
                f"{row_set.label}: intervals.parquet disagrees with {printed.site_hours:,} "
                f"{row_set.hours_unit}"
            )
            raise ValueError(msg)
        setting = BLOCK_SETTINGS[row_set.key]
        label = f"{BLOCK_LABELS[row_set.key]} ({setting.hub_height})"
        dates = f"{month_year(iso_day=printed.first_day)} to {month_year(iso_day=printed.last_day)}"
        absolute = absolute_rows(
            frame=frame, row_set=row_set, printed=printed.tables[ABSOLUTE_SECTION]
        )
        common = _CommonFields(
            label=label,
            dates=dates,
            site_hours=printed.site_hours,
            hours_unit=row_set.hours_unit,
            reference_name=setting.reference_name,
        )
        leaderboard_blocks.append(_block(common=common, rows=absolute))
        contrasts = contrast_rows(
            frame=frame,
            row_set=row_set,
            order=absolute["arm"].to_list(),
            printed=printed.tables[contrast_section(reference_label=row_set.reference_label)],
        )
        planned = planned_rows(
            frame=frame, row_set=row_set, printed=printed.tables[PLANNED_CONTRAST_SECTION]
        )
        contrast_blocks.append(_block(common=common, rows=contrasts, planned=planned))
    return leaderboard_blocks, contrast_blocks


def leaderboard_figure(
    *, blocks: list[RowSetBlock], shares: dict[str, float | None]
) -> alt.VConcatChart:
    """Draw Figure 1, the leaderboard of the four row sets."""
    return stacked_leaderboard(
        blocks=blocks,
        number=WIND_FIGURE_NUMBERS["leaderboard"],
        title="Mean absolute error of each weather product's wind, on four row sets",
        subtitle=[
            "Each arm's own mean absolute error, sorted best first within its block.",
            BLOCKS_NOT_COMPARABLE,
            (
                "Overlapping intervals do not make two arms equal: the intervals are wide mainly "
                "because every arm's error swings together from month to month, a swing that "
                f"Figure {WIND_FIGURE_NUMBERS['contrasts']}'s paired contrasts cancel."
            ),
            STATION_SCOPE,
            *uncovered_month_note(shares=shares),
            DOTS,
            CAPACITY,
            SCOPE,
        ],
        reference_note=REFERENCE_ROW_NOTE,
    )


def contrasts_figure(
    *, blocks: list[RowSetBlock], shares: dict[str, float | None]
) -> alt.VConcatChart:
    """Draw Figure 2, the contrasts against ERA5 with each row set's planned contrasts."""
    return stacked_contrasts(
        blocks=blocks,
        number=WIND_FIGURE_NUMBERS["contrasts"],
        title="Each arm's mean absolute error minus ERA5's, and each row set's planned contrasts",
        subtitle=[
            (
                "Top panel of each block: each arm's mean absolute error minus the reference "
                "arm's. Lower panel: that row set's planned contrasts, the first arm's error minus "
                "the second's; each row names both arms."
            ),
            BLOCKS_NOT_COMPARABLE,
            STATION_SCOPE,
            *uncovered_month_note(shares=shares),
            DOTS,
            CAPACITY,
            SCOPE,
        ],
        reference_note=CONTRAST_REFERENCE_ROW_NOTE,
    )


def main() -> int:
    """Read the leaderboard folder, check every number against its report, and write two SVGs."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args()
    report = read_report(report_text=(WIND_LEADERBOARD_DIR / "report.md").read_text())
    intervals = pl.read_parquet(WIND_LEADERBOARD_DIR / "intervals.parquet")
    leaderboard_blocks, contrast_blocks = build_blocks(intervals=intervals, report=report)
    charts = {
        "wind_leaderboard": leaderboard_figure(
            blocks=leaderboard_blocks, shares=UNCOVERED_MONTH_SHARES
        ),
        "wind_contrasts": contrasts_figure(blocks=contrast_blocks, shares=UNCOVERED_MONTH_SHARES),
    }
    for name, chart in charts.items():
        path = ASSETS_DIR / f"{name}.svg"
        chart.save(path)
        _LOG.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
