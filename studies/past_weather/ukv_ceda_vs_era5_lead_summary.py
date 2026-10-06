"""One table of the UKV-CEDA against ERA5 study by lead, for the page's consolidated section.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/1024>. It fits nothing. It reads
`station_intervals.parquet` (set A, which `ukv_ceda_station_scores.py` wrote) and
`intervals.parquet` (set B, which `ukv_ceda_vs_era5_fit.py` wrote), and prints every product's own
mean absolute error and the UKV-CEDA minus ERA5 contrast, with its 95% interval, at each lead from
0 to 5 hours and pooled. The absolute errors by lead are already in those two files and not in
either committed report, so this script prints them into `lead_summary.md`.

Set A rows (temperature and wind speed at four stations) use the primary score, the error after
removing each station, product, calendar month, and hour-of-day mean error. Set B rows (wind power
and solar power) use the primary hyperparameter setting. A pooled row is planned (P1 to P4). A
single-lead row is post hoc. The refit rows, each XGBoost model trained and scored on the lead-0 or
the lead 0 to 1 rows alone, are post hoc and run at the primary setting only.

Run it with `uv run python studies/past_weather/ukv_ceda_vs_era5_lead_summary.py`. `--dry-run`
prints the table and writes nothing. A fresh run stops (`refuse_to_overwrite`) while the output
exists.
"""

import argparse
import sys
from pathlib import Path
from typing import Final, NamedTuple

import polars as pl
from studies.guards import refuse_to_overwrite
from ukv_ceda_station_scores import INTERVALS_NAME as STATION_INTERVALS_NAME
from ukv_ceda_station_scores import PRIMARY_SCORE
from ukv_ceda_vs_era5_build import OUTPUT_DIR

SET_B_INTERVALS_NAME: Final[str] = "intervals.parquet"
REPORT_NAME: Final[str] = "lead_summary.md"
LEADS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 5)
PRIMARY_SETTING: Final[str] = "pooled"


INTRODUCTION: Final[str] = (
    "Each error is the mean absolute error of one product. A negative difference favours "
    "UKV-CEDA. The intervals cover month-to-month weather, and for power the fitting seed too. "
    "Temperature and wind speed (set A) are scored against four stations after removing each "
    "station, product, calendar month, and hour-of-day mean error. Power (set B) is an XGBoost "
    "model's error at the primary hyperparameter setting, in percent of capacity, and its "
    "difference is in percentage points of capacity. A refit row trains and scores both "
    "XGBoost models on those leads' rows alone."
)
HEADER: Final[str] = (
    "| Outcome | Lead (hours since the run started) | Status | ERA5 error "
    "| UKV-CEDA error | UKV-CEDA minus ERA5 [95% interval] |"
)


class Row(NamedTuple):
    """One table row: both products' errors and the contrast, in the outcome's own unit."""

    outcome: str
    scope: str
    status: str
    era5_error: float
    ukv_error: float
    difference: float
    lower: float
    upper: float


def _station_rows(*, intervals: pl.DataFrame, variable: str, outcome: str) -> list[Row]:
    """Set A rows of one variable: the pooled planned row, then each single lead."""
    at_score = intervals.filter(
        (pl.col("variable") == variable) & (pl.col("score") == PRIMARY_SCORE)
    )
    wanted = [
        ("P1" if variable == "wind" else "P2", "all", "pooled over leads 0 to 5", "planned")
    ] + [("post hoc lead", f"lead {lead} h", f"{lead}", "post hoc") for lead in LEADS]
    rows: list[Row] = []
    for label, scope, name, status in wanted:
        record = at_score.filter((pl.col("label") == label) & (pl.col("scope") == scope)).row(
            0, named=True
        )
        rows.append(
            Row(
                outcome,
                name,
                status,
                record["era5_mae"],
                record["ukv_mae"],
                record["difference"],
                record["lower_95"],
                record["upper_95"],
            )
        )
    return rows


def _power_row(
    *, intervals: pl.DataFrame, where: pl.Expr, outcome: str, name: str, status: str
) -> Row:
    """One set B row: the single record that matches `where` at the primary setting."""
    record = intervals.filter((pl.col("setting") == PRIMARY_SETTING) & where).row(0, named=True)
    return Row(
        outcome,
        name,
        status,
        record["reference_mae_pp"],
        record["treatment_mae_pp"],
        record["difference_pp"],
        record["lower_95_pp"],
        record["upper_95_pp"],
    )


def _power_rows(*, intervals: pl.DataFrame, domain: str, label: str, outcome: str) -> list[Row]:
    """Set B rows of one power outcome: pooled, each lead, then the two analysis-only refits."""
    base = pl.col("domain") == domain
    rows = [
        _power_row(
            intervals=intervals,
            where=base & (pl.col("label") == label) & (pl.col("scope") == "all"),
            outcome=outcome,
            name="pooled over leads 0 to 5",
            status="planned",
        )
    ]
    rows += [
        _power_row(
            intervals=intervals,
            where=base & (pl.col("label") == label) & (pl.col("scope") == f"lead {lead} h"),
            outcome=outcome,
            name=f"{lead}",
            status="post hoc",
        )
        for lead in LEADS
    ]
    for suffix, name in (
        ("lead0", "refit on lead 0 rows"),
        ("leads01", "refit on leads 0 to 1 rows"),
    ):
        rows.append(
            _power_row(
                intervals=intervals,
                where=(pl.col("domain") == f"{domain}_{suffix}") & (pl.col("scope") == "all"),
                outcome=outcome,
                name=name,
                status="post hoc",
            )
        )
    return rows


def _line(*, row: Row, digits: int) -> str:
    """Format one row as a markdown table line."""
    return (
        f"| {row.outcome} | {row.scope} | {row.status} | {row.era5_error:.{digits}f} | "
        f"{row.ukv_error:.{digits}f} | {row.difference:+.{digits}f} "
        f"[{row.lower:+.{digits}f}, {row.upper:+.{digits}f}] |"
    )


def report_text(*, station_intervals: pl.DataFrame, set_b_intervals: pl.DataFrame) -> str:
    """Build the report: one table by lead for the four outcomes."""
    blocks = [
        (
            _station_rows(
                intervals=station_intervals, variable="temperature", outcome="Temperature (K)"
            ),
            3,
        ),
        (
            _station_rows(intervals=station_intervals, variable="wind", outcome="Wind speed (m/s)"),
            3,
        ),
        (
            _power_rows(
                intervals=set_b_intervals,
                domain="wind",
                label="P3",
                outcome="Wind power (% of capacity)",
            ),
            3,
        ),
        (
            _power_rows(
                intervals=set_b_intervals,
                domain="solar",
                label="P4",
                outcome="Solar power (% of capacity)",
            ),
            3,
        ),
    ]
    lines = [
        "### UKV-CEDA against ERA5 by lead",
        "",
        INTRODUCTION,
        "",
        HEADER,
        "|---|---|---|---|---|---|",
    ]
    lines += [_line(row=row, digits=digits) for rows, digits in blocks for row in rows]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Print the table, and write it once to `lead_summary.md`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the table; write nothing.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()
    report = report_text(
        station_intervals=pl.read_parquet(arguments.output_dir / STATION_INTERVALS_NAME),
        set_b_intervals=pl.read_parquet(arguments.output_dir / SET_B_INTERVALS_NAME),
    )
    if not arguments.dry_run:
        path = arguments.output_dir / REPORT_NAME
        refuse_to_overwrite(paths=[path])
        path.write_text(report)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
