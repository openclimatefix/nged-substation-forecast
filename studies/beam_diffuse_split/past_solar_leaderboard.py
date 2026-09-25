"""One leaderboard and one set of contrasts against ERA5 for the four headline past-solar row sets.

The past-solar page scores four row sets, each on its own common rows: the main rows
(`solar_long`), the extra rows (`solar_all`), the ECMWF ENS rows (`ens_past_solar`), and the
weather-station rows (`station_past_solar`). This script reads each row set's saved `pooled`
losses, without refitting anything, and writes for each

- every arm's own mean absolute error with a 95% interval, and
- every arm's mean absolute error minus ERA5's with a 95% interval,

both from the same month-and-seed resampling `weather_products.py` uses. Contrasts that a row
set's own report prints are labelled planned when the report names them before the run, and every
other contrast is exploratory, except the two UKV rebuilds, which the main report labels post hoc.
A contrast that has a second hyperparameter setting saved is
recomputed there if it is planned or lies near the 5% line (an interval bound within 20% of the
interval's width from zero).

**The script stops before writing anything unless every number a row set's report already prints
is reproduced at the report's own precision**: each arm's error, each printed interval, and each
printed contrast against ERA5 on the same rows. It writes `report.md` and `intervals.parquet` into
`SOLAR_LEADERBOARD_DIR`, which it refuses to overwrite. `--check-only` reads and verifies but
writes nothing.

Run it with `uv run studies/beam_diffuse_split/past_solar_leaderboard.py`. No generator's name,
identifier, or coordinates appears in its output.
"""

import argparse
import logging
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final, NamedTuple

import polars as pl
import station_past_solar_charts as station_charts
import weather_product_charts as main_charts
from ens_past_solar_charts import NAMES as ENS_NAMES
from sources import SOLAR_LEADERBOARD_DIR, UPDATE_OUTPUT_DIR
from studies.charts import (
    REPORT_PRINT_DECIMALS,
    BlockArm,
    ProductFamily,
    block_contrast_rows,
    block_leaderboard_rows,
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

ABSOLUTE_SECTION: Final[str] = "Mean absolute error"
"""The `section` of an arm's own error in `intervals.parquet`."""

CONTRAST_SECTION: Final[str] = "Mean absolute error minus ERA5's"
"""The `section` of a contrast against ERA5 in `intervals.parquet`."""

SECOND_SETTING_SCOPE: Final[str] = "sensitivity"
"""The scope a report gives a contrast at the second hyperparameter setting."""

REPORT_INTRODUCTION: Final[str] = (
    "Every number is recomputed from the saved `pooled` losses of four row sets, by resampling "
    "whole months and a fitting seed. Each row set is scored on its own common rows, so a value "
    "is comparable within a row set and not across row sets. Mean absolute error is a percentage "
    "of each generator's 99th-percentile output. Every contrast not named before the run is "
    "exploratory, except the two UKV rebuilds, which were added after the first run and are post "
    "hoc. `Second setting` is the same contrast at the second hyperparameter setting, "
    "shown only for planned contrasts and contrasts near the 5% line (an interval bound within "
    "20% of the interval's width from zero), and only where both arms have saved second-setting "
    "losses."
)
CONTRAST_HEADER: Final[str] = (
    "| Arm | Difference (pp of capacity) | 95% interval | Planned or exploratory "
    "| Near the 5% line | Second setting |"
)

_HEADING: Final[re.Pattern[str]] = re.compile(
    r"on ([\d,]+) common site-hours.*\((\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})\)"
)
_PRINTED_INTERVAL: Final[re.Pattern[str]] = re.compile(
    r"^\| (\w+) \| ([\d.]+) \| \[([\d.]+), ([\d.]+)\] \|$", re.MULTILINE
)


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
    """

    key: str
    label: str
    directory: Path
    printed_column: str
    arm_suffix: str
    leaderboard_arms: tuple[BlockArm, ...]
    contrast_arms: tuple[BlockArm, ...]


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
    """

    row_set: RowSet
    dates: str
    site_hours: int
    absolute: pl.DataFrame
    contrasts: pl.DataFrame


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
"""The two rebuilds of UKV's hourly value, contrasted with ERA5 on the main rows only."""

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
        leaderboard_arms=MAIN_ARMS,
        contrast_arms=(*_without_era5(arms=MAIN_ARMS), *UKV_REBUILDS),
    ),
    RowSet(
        key="extra",
        label="Extra rows",
        directory=UPDATE_OUTPUT_DIR / "solar_all",
        printed_column="Global only",
        arm_suffix="_global",
        leaderboard_arms=EXTRA_ARMS,
        contrast_arms=_without_era5(arms=EXTRA_ARMS),
    ),
    RowSet(
        key="ens",
        label="ECMWF ENS rows",
        directory=UPDATE_OUTPUT_DIR / "ens_past_solar",
        printed_column="All sites",
        arm_suffix="",
        leaderboard_arms=ENS_ARMS,
        contrast_arms=_without_era5(arms=ENS_ARMS),
    ),
    RowSet(
        key="station",
        label="Weather-station rows",
        directory=UPDATE_OUTPUT_DIR / "station_past_solar",
        printed_column="All sites",
        arm_suffix="",
        leaderboard_arms=STATION_ARMS,
        contrast_arms=_without_era5(arms=STATION_ARMS),
    ),
)
"""The four headline row sets, in the order the leaderboard stacks them."""


def read_heading(*, report_text: str) -> tuple[int, str]:
    """Read the site-hours and the dates from a report's first heading.

    Args:
        report_text: A row set's `report.md`.

    Returns:
        The number of common site-hours, and the dates as `2022-12-01 to 2026-08-31`.

    Raises:
        ValueError: If no line holds both.
    """
    match = _HEADING.search(report_text)
    if match is None:
        msg = "the report has no 'on N common site-hours (first to last)' heading"
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


def check_intervals(*, absolute: pl.DataFrame, report_text: str) -> list[str]:
    """List every recomputed interval that differs from the interval the report prints.

    Args:
        absolute: `block_leaderboard_rows`'s output.
        report_text: The row set's `report.md`.

    Returns:
        One message per difference at `PRINT_DECIMALS` places; empty where all agree or the report
        prints no intervals.
    """
    printed = printed_intervals(report_text=report_text)
    problems = []
    for row in absolute.iter_rows(named=True):
        arm = row["arm"]
        if arm not in printed:
            continue
        recomputed = tuple(round(row[name], PRINT_DECIMALS) for name in ("lower_95", "upper_95"))
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
) -> list[str]:
    """List every recomputed contrast that differs from a printed contrast on the same rows.

    A printed row is compared where it has the same scope, treatment, reference, and number of
    rows, and sits in a section that reports the main fit: a section starting with one of
    `OTHER_FIT_SECTION_PREFIXES` prints a different fit's numbers.

    Args:
        contrasts: `block_contrast_rows`'s output, with `arm`.
        printed: `report_contrasts`'s output for the row set's report.
        site_hours: The row set's number of site-hours; a printed row on any other rows (a season,
            a scope) is a different contrast and is skipped.
        scope: The printed scope to compare with: `all`, or `SECOND_SETTING_SCOPE`.
        column_prefix: What precedes `difference`, `lower_95` and `upper_95` in the columns to
            compare: empty for the main setting, `second_` for the second setting. A row whose
            value is null is skipped.

    Returns:
        One message per difference at `PRINT_DECIMALS` places; a contrast the report does not
        print is not checked.
    """
    names = tuple(f"{column_prefix}{name}" for name in ("difference", "lower_95", "upper_95"))
    problems = []
    for row in contrasts.iter_rows(named=True):
        if row[names[0]] is None:
            continue
        arm = row["arm"]
        matches = printed.filter(
            pl.col("scope") == scope,
            pl.col("treatment") == arm,
            pl.col("reference") == REFERENCE_ARM,
            pl.col("n_rows") == site_hours,
            ~pl.any_horizontal(
                pl.col("section").str.starts_with(prefix) for prefix in OTHER_FIT_SECTION_PREFIXES
            ),
        )
        recomputed = tuple(round(row[name], PRINT_DECIMALS) for name in names)
        for match in matches.iter_rows(named=True):
            shown = (match["difference"], match["lower_95"], match["upper_95"])
            if recomputed != shown:
                problems.append(
                    f"{arm} - {REFERENCE_ARM} at scope {scope}: {recomputed} but section "
                    f"{match['section']!r} prints {shown}"
                )
    return problems


def missing_planned_second_rows(*, contrasts: pl.DataFrame, printed: pl.DataFrame) -> list[str]:
    """List every planned contrast with a second setting that the report prints no row for.

    A report names each planned contrast against ERA5 at the second setting, so a planned contrast
    with none printed means the report and the script disagree about which contrast is planned.

    Args:
        contrasts: The scored contrasts, with `arm`, `planning` and `second_difference`.
        printed: `report_contrasts`'s output for the row set's report.

    Returns:
        One message per planned contrast with a second-setting value and no printed row.
    """
    on_second = printed.filter(
        pl.col("scope") == SECOND_SETTING_SCOPE, pl.col("reference") == REFERENCE_ARM
    )
    printed_arms = set(on_second["treatment"].to_list())
    return [
        f"{row['arm']} - {REFERENCE_ARM}: planned, but the report prints no row at scope "
        f"{SECOND_SETTING_SCOPE}"
        for row in contrasts.iter_rows(named=True)
        if row["planning"] == "planned"
        and row["second_difference"] is not None
        and row["arm"] not in printed_arms
    ]


def planned_arms(*, printed: pl.DataFrame) -> set[str]:
    """Return the arms whose contrast against ERA5 the report names before the run.

    Args:
        printed: `report_contrasts`'s output.

    Returns:
        Each treatment arm with a row against ERA5, scope `all`, under a heading that starts with
        `PLANNED_SECTION_PREFIX`.
    """
    rows = printed.filter(
        pl.col("section").str.starts_with(PLANNED_SECTION_PREFIX),
        pl.col("scope") == "all",
        pl.col("reference") == REFERENCE_ARM,
    )
    return set(rows["treatment"].to_list())


def is_near_line(*, lower_95: float, upper_95: float) -> bool:
    """Say whether an interval has a bound within `NEAR_LINE_SHARE` of its width from zero.

    Args:
        lower_95: The interval's lower bound.
        upper_95: The interval's upper bound.

    Returns:
        True where the bound nearer zero is within 20% of the interval's width of zero.
    """
    nearest = min(abs(lower_95), abs(upper_95))
    return nearest <= NEAR_LINE_SHARE * (upper_95 - lower_95)


def _second_setting(
    *, contrasts: pl.DataFrame, arms: tuple[BlockArm, ...], losses: pl.DataFrame, site_hours: int
) -> pl.DataFrame:
    """Add the second hyperparameter setting's contrast to each planned or near-line row.

    A row gets one only where its arm and ERA5 both have saved `sensitivity` losses; every other
    row keeps nulls, because this script refits nothing.

    Args:
        contrasts: `block_contrast_rows`'s output with `arm`, `planning` and `near_line`.
        arms: The arms contrasted.
        losses: The row set's `losses.parquet`.
        site_hours: The row set's number of site-hours.

    Returns:
        The rows with `second_difference`, `second_lower_95` and `second_upper_95`.
    """
    saved = set(losses.filter(pl.col("setting") == "sensitivity")["arm"].unique().to_list())
    wanted_arms = {
        row["arm"]
        for row in contrasts.iter_rows(named=True)
        if (row["planning"] == "planned" or row["near_line"])
        and row["arm"] in saved
        and REFERENCE_ARM in saved
    }
    columns = ("second_difference", "second_lower_95", "second_upper_95")
    if not wanted_arms:
        return contrasts.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(name) for name in columns
        )
    second = block_contrast_rows(
        losses=losses,
        arms=[arm for arm in arms if arm.arm in wanted_arms],
        reference_arm=REFERENCE_ARM,
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
        ValueError: If a leaderboard arm has no printed error, or a recomputed error, interval, or
            contrast differs from the report's, naming every difference.
    """
    site_hours, dates = read_heading(report_text=report_text)
    errors = report_errors(report_path=report_path, column=row_set.printed_column)
    printed_errors = {f"{key}{row_set.arm_suffix}": value for key, value in errors.items()}
    unprinted = [arm.arm for arm in row_set.leaderboard_arms if arm.arm not in printed_errors]
    if unprinted:
        msg = (
            f"{row_set.label}: the report's {row_set.printed_column!r} column prints no error "
            f"for {unprinted}; its arm names are {sorted(printed_errors)}"
        )
        raise ValueError(msg)
    printed = report_contrasts(report_path=report_path)
    planned = planned_arms(printed=printed)
    absolute = block_leaderboard_rows(
        losses=losses,
        arms=row_set.leaderboard_arms,
        setting="pooled",
        site_hours=site_hours,
        metric=METRIC,
        printed=printed_errors,
    )
    contrast_arms = tuple(arm._replace(planned=arm.arm in planned) for arm in row_set.contrast_arms)
    contrasts = block_contrast_rows(
        losses=losses,
        arms=contrast_arms,
        reference_arm=REFERENCE_ARM,
        setting="pooled",
        site_hours=site_hours,
        metric=METRIC,
    ).with_columns(
        planning=pl.when(pl.col("planned"))
        .then(pl.lit("planned"))
        .when(pl.col("arm").is_in(POST_HOC_ARMS))
        .then(pl.lit("post hoc"))
        .otherwise(pl.lit("exploratory")),
        near_line=pl.min_horizontal(pl.col("lower_95").abs(), pl.col("upper_95").abs())
        <= NEAR_LINE_SHARE * (pl.col("upper_95") - pl.col("lower_95")),
    )
    contrasts = _second_setting(
        contrasts=contrasts, arms=contrast_arms, losses=losses, site_hours=site_hours
    )
    problems = [
        *check_intervals(absolute=absolute, report_text=report_text),
        *check_contrasts(contrasts=contrasts, printed=printed, site_hours=site_hours),
        *check_contrasts(
            contrasts=contrasts,
            printed=printed,
            site_hours=site_hours,
            scope=SECOND_SETTING_SCOPE,
            column_prefix="second_",
        ),
        *missing_planned_second_rows(contrasts=contrasts, printed=printed),
    ]
    if problems:
        msg = f"{row_set.label} does not reproduce its report:\n" + "\n".join(problems)
        raise ValueError(msg)
    return RowSetResult(
        row_set=row_set, dates=dates, site_hours=site_hours, absolute=absolute, contrasts=contrasts
    )


def _signed(value: float | None) -> str:
    """Format a difference with its sign, or a dash where there is none."""
    return "—" if value is None else f"{value:+.{PRINT_DECIMALS}f}"


def _interval(*, low: float | None, high: float | None) -> str:
    """Format a signed interval as `[low, high]`, or a dash where there is none."""
    if low is None or high is None:
        return "—"
    return f"[{low:+.{PRINT_DECIMALS}f}, {high:+.{PRINT_DECIMALS}f}]"


def render_report(*, results: list[RowSetResult]) -> str:
    """Write the leaderboard report, one section per row set.

    Args:
        results: Each row set's scores.

    Returns:
        The report's markdown.
    """
    lines = [
        "# Past-solar leaderboard and contrasts against ERA5",
        "",
        REPORT_INTRODUCTION,
        "",
    ]
    for result in results:
        lines += [
            f"### {result.row_set.label}: {result.dates}, {result.site_hours:,} site-hours",
            "",
            "#### Mean absolute error",
            "",
            "| Arm | Error | 95% interval | Reference row |",
            "|---|---|---|---|",
        ]
        for row in result.absolute.iter_rows(named=True):
            lines.append(
                f"| {row['label']} | {row['value']:.{PRINT_DECIMALS}f} | "
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
            second = (
                f"{_signed(row['second_difference'])} "
                f"{_interval(low=row['second_lower_95'], high=row['second_upper_95'])}"
            )
            lines.append(
                f"| {row['label']} | {_signed(row['difference'])} | "
                f"{_interval(low=row['lower_95'], high=row['upper_95'])} | {row['planning']} | "
                f"{'yes' if row['near_line'] else 'no'} | {second} |"
            )
        lines.append("")
    return "\n".join(lines)


def intervals_frame(*, results: list[RowSetResult]) -> pl.DataFrame:
    """Stack every row set's scores into one long frame, in `studies.page_numbers`' convention.

    Args:
        results: Each row set's scores.

    Returns:
        One row per (row set, section, setting, arm). `section` is `Mean absolute error` or
        `Mean absolute error minus ERA5's`; `setting` is `pooled`, or `sensitivity` for a
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
                reference=pl.lit(REFERENCE_ARM),
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
                reference=pl.lit(REFERENCE_ARM),
                value="second_difference",
                lower="second_lower_95",
                upper="second_upper_95",
                label="label",
                planning="planning",
                near_line="near_line",
            ),
        ]
    return pl.concat(frames)


def write_outputs(*, results: list[RowSetResult], output_dir: Path) -> None:
    """Write `report.md` and `intervals.parquet` into a new folder.

    Args:
        results: Each row set's scores.
        output_dir: The folder to create.

    Raises:
        FileExistsError: If the folder exists, so an earlier run's numbers are never overwritten.
    """
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.md").write_text(render_report(results=results))
    intervals_frame(results=results).write_parquet(output_dir / "intervals.parquet")


def main() -> int:
    """Score the four row sets, check them against their reports, and write the results."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Read and verify every number against the reports, and write nothing.",
    )
    args = parser.parse_args()
    if not args.check_only and SOLAR_LEADERBOARD_DIR.exists():
        _LOG.error("%s exists; this script never overwrites it", SOLAR_LEADERBOARD_DIR)
        return 1
    results = []
    for row_set in ROW_SETS:
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
    write_outputs(results=results, output_dir=SOLAR_LEADERBOARD_DIR)
    _LOG.info("wrote %s", SOLAR_LEADERBOARD_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
