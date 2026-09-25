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

Each block's title is its row-set name, its dates, and its number of farm-hours. The caption
states each block's wind heights and any caveat, and the share of its scored rows that fall in a
calendar month with no training row in their fold, under the published folds.
`UNCOVERED_MONTH_SHARES` holds those shares. The script stops where one is unset, so a block is
never drawn without its share.

Generators do not appear in either figure.

Run it with `uv run python studies/beam_diffuse_split/past_wind_leaderboard_charts.py`, after
`past_wind_leaderboard.py`. Optimise each SVG with `npx svgo@4 --multipass --precision=1
--final-newline` before committing it.
"""

import argparse
import logging
import re
import sys
from typing import Final, NamedTuple

import altair as alt
import polars as pl
from figure_numbers import WIND_FIGURE_NUMBERS
from past_solar_leaderboard import ABSOLUTE_SECTION, contrast_section
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
from studies.charts import (
    POST_HOC_SUFFIX,
    RowSetBlock,
    stacked_contrasts,
    stacked_leaderboard,
    wrapped,
)

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

BLOCK_LABELS: Final[dict[str, str]] = {
    "main": "Main",
    "icon_dream_eu": "ICON-DREAM-EU",
    "ecmwf": "ECMWF",
    "station": "Station",
}
"""Each row set's block label, the term the page uses for the row set."""


class MonthShares(NamedTuple):
    """One block's shares of scored rows, in percent, by how their calendar month is trained on."""

    uncovered: float
    """Rows in a calendar month seen in two or more years that has no training row in their fold."""

    one_year_only: float
    """Rows in a calendar month seen in one year only, which no fold design can cover."""


UNCOVERED_MONTH_SHARES: Final[dict[str, MonthShares | None]] = {
    "main": MonthShares(uncovered=16.4, one_year_only=0.0),
    "icon_dream_eu": MonthShares(uncovered=25.1, one_year_only=0.0),
    "ecmwf": MonthShares(uncovered=0.0, one_year_only=9.4),
    "station": MonthShares(uncovered=0.0, one_year_only=42.2),
}
"""Each block's `MonthShares`, from the folds saved with its published per-row losses.

Source: `studies/era_fold_design/README.md` at commit fdddb065 on the `era-fold-design` branch
(main 8,326 of 50,734 rows, ICON-DREAM-EU 12,570 of 50,041; ECMWF 4,082 of 43,555 and station
14,411 of 34,156 in months seen in one year only). `uncovered_month_note` stops on a `None`.
"""

FOLD_COVERING_EFFECT: Final[dict[str, str]] = {
    "main": (
        "covering those months moves no planned contrast by more than 0.033 points at the primary "
        "setting and 0.052 points at the second setting; the effect on absolute errors was not "
        "measured"
    ),
    "icon_dream_eu": (
        "covering those months moves its two planned contrasts by +0.009 and -0.028 points at the "
        "primary setting, with no change of sign or of statistical significance; the effect on "
        "absolute errors was not measured"
    ),
}
"""Blocks whose fold-covering refit was measured, with the measured bound the caption states.

Source: `studies/era_fold_design/report.md` on the `era-fold-design` branch, which the folder's
README names as the full report. The ECMWF and station blocks have no uncovered rows, so no refit
was needed.
"""

POST_HOC_PLANNED_TITLE: Final[str] = "planned and post hoc contrasts"
"""What a block's lower panel is titled, after the block's label, where it holds post hoc rows."""

UNDRAWN_PLANNED_ICON_NOTE: Final[str] = (
    "The ICON contrasts that the study plan specified at 100 m are reported on the page, not "
    "drawn here."
)
"""The caption line that says where the plan's own ICON contrasts, which the figure lacks, are."""

CAPACITY: Final[str] = (
    "Capacity is each generator's 99th-percentile output, not its nameplate capacity."
)
FARM_HOURS: Final[str] = (
    "A farm-hour is one hour at one farm, so a block's count sums the hours of the three farms."
)
DOTS: Final[str] = (
    "Dot: estimate. Line: 95% interval from resampling whole calendar months, each with all three "
    "farms' rows, and a fitting seed. The interval does not cover variation between the three "
    "farms."
)
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
    "The station rows rest on one nearby weather station per farm, and cover 17 calendar months, "
    "fewer than the other blocks."
)
WIND_SECOND_SETTING_NOTE: Final[str] = (
    "Hollow triangle: the same contrast at the second hyperparameter setting, shown for planned "
    "and post hoc contrasts, and for contrasts near the 5% line where both arms have "
    "second-setting losses."
)
PLANNING_NOTE: Final[str] = (
    "Planned: a comparison written down before that block's own arms were fitted, in the study "
    "plan for the main, ECMWF, and station blocks, and after the main block's five products were "
    "scored for the ICON-DREAM-EU block. Post hoc: a planned comparison whose ICON arm was "
    "switched from 100 m to 80 m after the first run. Every other row is exploratory."
)
"""The caption line defining planned and post hoc rows, with the ICON-DREAM-EU block's plan."""
CHANCE_NOTE: Final[str] = (
    "About 1 in 20 exploratory contrasts reaches significance at the 5% level by chance. Each "
    "interval rests on 17 to 26 resampled calendar months, depending on the block, so the "
    "intervals are likely somewhat narrow."
)
CAPTION_CHARACTERS: Final[int] = 100
"""The characters a wind caption line holds, fewer than the default so no line reaches the edge."""
EACH_ARM_ERROR_NOTE: Final[str] = "Each arm's own error is in Figure {number}."


def uncovered_month_note(*, shares: dict[str, MonthShares | None]) -> list[str]:
    """Say, for each block, what share of its scored rows is in a month its fold never trains on.

    Args:
        shares: Each row set's key to its `MonthShares`, or `None` where it is unset.

    Returns:
        One caption line per block, in the order of `ROW_SETS`.

    Raises:
        ValueError: If a block's share is unset.
    """
    unset = [row_set.label for row_set in ROW_SETS if shares[row_set.key] is None]
    if unset:
        msg = f"the uncovered-month share is not set for {unset}; read it from the fold report"
        raise ValueError(msg)
    lines = []
    for row_set in ROW_SETS:
        month_shares = shares[row_set.key]
        assert month_shares is not None  # the `unset` check above rules this out
        line = (
            f"{BLOCK_LABELS[row_set.key]}: {month_shares.uncovered:.1f}% of scored rows are in a "
            "calendar month, seen in two or more years, with no training row in their fold, and "
            f"{month_shares.one_year_only:.1f}% are in a calendar month seen in one year only, "
            "which no fold design can cover"
        )
        effect = FOLD_COVERING_EFFECT.get(row_set.key)
        lines.append(f"{line}; {effect}." if effect else f"{line}.")
    return lines


def block_notes() -> list[str]:
    """State the wind heights each block's arms carry.

    Returns:
        A lead line, then one caption line for the wind heights of each block, in the order of
        `ROW_SETS`.
    """
    heights = [
        f"{BLOCK_LABELS[row_set.key]}: {BLOCK_SETTINGS[row_set.key].hub_height}."
        for row_set in ROW_SETS
    ]
    return ["Wind heights of each block's arms:", *heights]


def wind_product_caveats() -> list[str]:
    """State three caveats on the ranking of the weather products, each with a pointer.

    Returns:
        One caption line for each of ICON-D2's coverage, ICON global's steps, and HRES's lead.
    """
    return [
        (
            "ICON-D2 does not cover western Great Britain: its western edge runs from about "
            f"2\u00b0W to about 2.5\u00b0W (Figure {WIND_FIGURE_NUMBERS['domains']})."
        ),
        (
            "ICON global's worse-than-ERA5 result is mostly a pair of steps in its served wind at "
            "one generator. Once the XGBoost models are told where the steps fall, the difference "
            f"is not statistically significant (Figure {WIND_FIGURE_NUMBERS['icon_global_steps']})."
        ),
        (
            "HRES's planned lead over ERA5 is fragile: HRES minus ERA5 is -0.06 points "
            "[-0.24, +0.15] when the XGBoost model trains across IFS Cycle 49r1 without an era "
            f"cut (Figure {WIND_FIGURE_NUMBERS['reconciliation']})."
        ),
    ]


def block_caveats() -> list[str]:
    """State each block's caveat on a caption line of its own, naming the block it is about.

    Returns:
        One line for each block that has a caveat, in the order of `ROW_SETS`.
    """
    return [
        f"Caveat on the {BLOCK_LABELS[row_set.key]} block: {BLOCK_SETTINGS[row_set.key].note}"
        for row_set in ROW_SETS
        if BLOCK_SETTINGS[row_set.key].note
    ]


def _narrow(*, lines: list[str]) -> list[str]:
    """Wrap each caption line at `CAPTION_CHARACTERS`, which keeps it inside the figure's edge."""
    return [piece for line in lines for piece in wrapped(text=line, width=CAPTION_CHARACTERS)]


def _excludes_zero(*, lower: float, upper: float) -> bool:
    """Say whether a 95% interval excludes zero, the 5% significance test the figures use."""
    return lower > 0 or upper < 0


def _signed(*, value: float) -> str:
    """Format `value` in points of capacity, to three decimal places, with its sign."""
    return f"{value:+.3f}"


def significance_change_notes(*, intervals: pl.DataFrame) -> list[str]:
    """Name each contrast whose statistical significance at the 5% level differs between settings.

    A contrast is significant at a setting where its 95% interval excludes zero. Only a contrast
    with rows at both settings can change, so a contrast scored at the primary setting alone is
    skipped. A contrast printed in more than one section is named once.

    Args:
        intervals: The write-once `intervals.parquet`.

    Returns:
        One caption line per such contrast, in the order of `ROW_SETS`, naming the row set, the
        contrast, both intervals, and which setting the contrast is significant at.
    """
    notes = []
    for row_set in ROW_SETS:
        labels = {arm.arm: arm.label for arm in (*row_set.leaderboard_arms, *row_set.contrast_arms)}
        seen = set()
        frame = intervals.filter(pl.col("row_set") == row_set.key)
        for primary in frame.filter(pl.col("setting") == "pooled").iter_rows(named=True):
            pair = (primary["treatment"], primary["reference"])
            if pair in seen or primary["section"] == ABSOLUTE_SECTION:
                continue
            seconds = frame.filter(
                pl.col("section") == primary["section"],
                pl.col("setting") == "sensitivity",
                pl.col("treatment") == pair[0],
                pl.col("reference") == pair[1],
            )
            if seconds.is_empty():
                continue
            seen.add(pair)
            second = seconds.row(0, named=True)
            first_significant = _excludes_zero(lower=primary["lower"], upper=primary["upper"])
            second_significant = _excludes_zero(lower=second["lower"], upper=second["upper"])
            if first_significant == second_significant:
                continue
            verdict = (
                "statistically significant at the 5% level at the primary setting and not at the "
                "second setting"
                if first_significant
                else "not statistically significant at the 5% level at the primary setting and "
                "significant at the second setting"
            )
            notes.append(
                f"{BLOCK_LABELS[row_set.key]}: {labels[pair[0]]} minus {labels[pair[1]]} is "
                f"{_signed(value=primary['value'])} points "
                f"[{_signed(value=primary['lower'])}, {_signed(value=primary['upper'])}] at the "
                f"primary setting and {_signed(value=second['value'])} "
                f"[{_signed(value=second['lower'])}, {_signed(value=second['upper'])}] at the "
                f"second setting, so the contrast is {verdict}."
            )
    return notes


class _CommonFields(NamedTuple):
    """The fields a block's leaderboard and contrast versions share."""

    label: str
    dates: str
    site_hours: int
    hours_unit: str
    reference_name: str


def short_months(*, text: str) -> str:
    """Return `text` with each month name cut to its first three letters, so a block title fits."""
    return re.sub(r"\b([A-Z][a-z]{2})[a-z]+ (\d{4})", r"\1 \2", text)


def _block(
    *, common: _CommonFields, rows: pl.DataFrame, planned: pl.DataFrame | None = None
) -> RowSetBlock:
    """Return a block holding `rows` and, for the contrast chart, its planned contrasts."""
    post_hoc = planned is not None and planned["label"].str.ends_with(POST_HOC_SUFFIX).any()
    return RowSetBlock(
        label=common.label,
        dates=common.dates,
        site_hours=common.site_hours,
        rows=rows,
        planned_rows=planned,
        hours_unit=common.hours_unit,
        reference_name=common.reference_name,
        planned_title=POST_HOC_PLANNED_TITLE if post_hoc else "planned contrasts",
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
        label = BLOCK_LABELS[row_set.key]
        first = month_year(iso_day=printed.first_day)
        dates = short_months(text=f"{first} to {month_year(iso_day=printed.last_day)}")
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
            frame=frame, row_set=row_set, printed=printed.tables[row_set.planned_heading]
        )
        contrast_blocks.append(_block(common=common, rows=contrasts, planned=planned))
    return leaderboard_blocks, contrast_blocks


def leaderboard_figure(
    *, blocks: list[RowSetBlock], shares: dict[str, MonthShares | None]
) -> alt.VConcatChart:
    """Draw Figure 1, the leaderboard of the four row sets."""
    return stacked_leaderboard(
        blocks=blocks,
        number=WIND_FIGURE_NUMBERS["leaderboard"],
        title="Mean absolute error of each weather product's wind, on four row sets",
        subtitle=_narrow(
            lines=[
                "Each arm's own mean absolute error, sorted best first within its block.",
                BLOCKS_NOT_COMPARABLE,
                (
                    "Overlapping intervals do not show that two arms are equal. Figure "
                    f"{WIND_FIGURE_NUMBERS['contrasts']} tests each difference against ERA5 and "
                    "each "
                    "planned contrast; ICON-D2 against UKV is exploratory and is in Figure "
                    f"{WIND_FIGURE_NUMBERS['icon_d2_leads']}. The intervals are wide "
                    "mainly because every arm's error swings together from month to month, a swing "
                    f"that Figure {WIND_FIGURE_NUMBERS['contrasts']}'s paired contrasts cancel."
                ),
                STATION_SCOPE,
                *wind_product_caveats(),
                *uncovered_month_note(shares=shares),
                *block_caveats(),
                *block_notes(),
                DOTS,
                CAPACITY,
                FARM_HOURS,
                SCOPE,
            ]
        ),
        reference_note=REFERENCE_ROW_NOTE,
    )


def contrasts_figure(
    *,
    blocks: list[RowSetBlock],
    shares: dict[str, MonthShares | None],
    intervals: pl.DataFrame,
) -> alt.VConcatChart:
    """Draw Figure 2, the contrasts against ERA5 with each row set's planned contrasts."""
    return stacked_contrasts(
        blocks=blocks,
        number=WIND_FIGURE_NUMBERS["contrasts"],
        title="Each arm's mean absolute error minus ERA5's, and each row set's planned contrasts",
        subtitle=_narrow(
            lines=[
                (
                    "Top panel of each block: each arm's mean absolute error minus the reference "
                    "arm's. Lower panel: that row set's planned contrasts, the first arm's error "
                    "minus the second's; each row names both arms."
                ),
                EACH_ARM_ERROR_NOTE.format(number=WIND_FIGURE_NUMBERS["leaderboard"]),
                UNDRAWN_PLANNED_ICON_NOTE,
                BLOCKS_NOT_COMPARABLE,
                STATION_SCOPE,
                *wind_product_caveats(),
                CHANCE_NOTE,
                *uncovered_month_note(shares=shares),
                *block_caveats(),
                *block_notes(),
                *significance_change_notes(intervals=intervals),
                DOTS,
                CAPACITY,
                FARM_HOURS,
                SCOPE,
            ]
        ),
        reference_note=CONTRAST_REFERENCE_ROW_NOTE,
        colour_by_family=True,
        second_setting_note=WIND_SECOND_SETTING_NOTE,
        planning_note=PLANNING_NOTE,
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
        "wind_contrasts": contrasts_figure(
            blocks=contrast_blocks, shares=UNCOVERED_MONTH_SHARES, intervals=intervals
        ),
    }
    for name, chart in charts.items():
        path = ASSETS_DIR / f"{name}.svg"
        chart.save(path)
        _LOG.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
