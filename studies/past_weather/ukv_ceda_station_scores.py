"""Set A of the UKV-CEDA against ERA5 study: score each product against the stations, no fit.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/1024>. It reads
`station_hours.parquet`, which `ukv_ceda_vs_era5_build.py` wrote, and scores ERA5 (the nearest 0.25
degree cell) and UKV-CEDA (the nearest 2 km cell, from the freshest run) against four Met Office
stations. Variables are 10 m wind speed and 2 m (ERA5) or 1.5 m (UKV-CEDA) air temperature.

**The primary score is the mean absolute error after removing each (station, product, calendar
month, hour of day) mean error.** That is the closest set A analogue of the XGBoost models of set B,
which recalibrate per generator and learn the hour-of-day bias, and for UKV-CEDA that bias includes
the lead-dependent part, because the lead equals the hour modulo 6. Two more scores sit beside it:
the mean absolute error after removing each (station, product, calendar month) mean error, and the
raw mean absolute error. Bias removal removes offsets from the station's exposure and from the
cell's orography. It does not remove the advantage that a 2 km cell has over a 0.25 degree cell when
scored against a point observation, so no result here is attributed to the product alone.

**The planned contrasts of set A are P1 and P2, and P2-lead is pre-registered.** P1 is wind speed
and P2 is air temperature, each UKV-CEDA minus ERA5 on the primary score, with a negative value
favouring UKV-CEDA. P1 never decides. P2-lead is P2 split by UKV-CEDA lead (0 to 2 hours, then 3 to
5 hours), because the stations feed the Met Office's data assimilation and the assimilated reading
has the least influence at leads 3 to 5. The early and late windows of P2 are planned too, because
P2 decides temperature. Every other row is exploratory: the per-station rows, the years, the
half-years, the windows of P1, the other two scores, and two post hoc sets: each single lead from 0
to 5 hours, and each leave-one-station-out.

**Rows.** A station-hour counts for a variable when the station has a reading and both products have
a value. ERA5 wind at one station covers 2019-09 to 2023-12 only, and the other three stations cover
2019-09 to 2025-12.

**Intervals.** Each interval resamples whole calendar months (`bootstrap_row_difference`, which has
no fitting seed), 2,000 times. A split with fewer than `MIN_MONTHS_FOR_INTERVAL` months gets a point
estimate and no interval. The intervals cover month-to-month weather, not differences between
stations or places outside one box. The 4 stations share their weather.

**The margin** is `MARGIN_STATION_SHARE` of ERA5's bias-removed error on the same rows.

Run it with `uv run python studies/past_weather/ukv_ceda_station_scores.py`. `--dry-run` prints the
report and writes nothing. A fresh run stops (`refuse_to_overwrite`) while an output exists.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final, NamedTuple, TypedDict

import numpy as np
import polars as pl
from studies.bootstrap import MIN_MONTHS_FOR_INTERVAL, bootstrap_row_difference
from studies.guards import refuse_to_overwrite
from ukv_ceda_vs_era5_build import (
    EARLY_END_MONTH,
    MARGIN_STATION_SHARE,
    OUTPUT_DIR,
    STATION_HOURS_NAME,
    contrast_reading,
)

INTERVALS_NAME: Final[str] = "station_intervals.parquet"
REPORT_NAME: Final[str] = "station_report.md"

BIAS_BY_HOUR: Final[tuple[str, ...]] = ("site", "calendar_month", "hour_of_day")
"""The groups whose mean error the primary score removes."""

BIAS_BY_MONTH: Final[tuple[str, ...]] = ("site", "calendar_month")
"""The groups whose mean error the second score removes."""

PRODUCTS: Final[tuple[str, str]] = ("era5", "ukv")

SCORES: Final[tuple[str, ...]] = ("bias_removed_hour", "bias_removed_month", "raw")
"""The three scores, the first being the primary."""

PRIMARY_SCORE: Final[str] = SCORES[0]

N_LEADS: Final[int] = 6
"""The UKV-CEDA leads, 0 to 5 hours, that a 6-hourly store serves."""

LEAD_SPLITS: Final[dict[str, tuple[int, int]]] = {"leads 0 to 2": (0, 2), "leads 3 to 5": (3, 5)}
"""The UKV-CEDA lead ranges of P2-lead, as inclusive bounds in hours."""


class Variable(NamedTuple):
    """One scored variable and its columns in `station_hours.parquet`.

    Attributes:
        name: `wind` or `temperature`.
        station: The station's column.
        era5: ERA5's column.
        ukv: UKV-CEDA's column.
        unit: The unit of the errors.
        planned: The label of the variable's planned contrast.
    """

    name: str
    station: str
    era5: str
    ukv: str
    unit: str
    planned: str


VARIABLES: Final[tuple[Variable, Variable]] = (
    Variable("wind", "station_wind_m_s", "era5_wind_m_s", "ukv_wind_m_s", "m/s", "P1"),
    Variable("temperature", "station_temp_c", "era5_temp_c", "ukv_temp_c", "K", "P2"),
)


class IntervalRecord(TypedDict):
    """One contrast, UKV-CEDA minus ERA5, as `station_intervals.parquet` holds it."""

    variable: str
    label: str
    planned: bool
    score: str
    scope: str
    era5_mae: float
    ukv_mae: float
    difference: float
    lower_95: float
    upper_95: float
    margin: float
    reading: str
    n_rows: int
    n_months: int
    enough_months: bool


def debiased(*, error: str, by: Sequence[str]) -> pl.Expr:
    """Return an error with the mean error of its group removed.

    Args:
        error: The signed error column.
        by: The columns that define a group, so the mean is taken within each group alone.

    Returns:
        The expression `error - mean(error) over the group`.
    """
    return pl.col(error) - pl.col(error).mean().over(list(by))


def scored_rows(*, frame: pl.DataFrame, variable: Variable) -> pl.DataFrame:
    """Keep the station-hours with a station reading and both products' values, and score them.

    Args:
        frame: `station_hours.parquet`.
        variable: The variable to score.

    Returns:
        The kept rows with `abs_<product>_<score>` for each product and each of `SCORES`.
    """
    columns = (variable.station, variable.era5, variable.ukv)
    present = [pl.col(column).is_not_null() & pl.col(column).is_not_nan() for column in columns]
    return (
        frame.filter(*present)
        .with_columns(
            calendar_month=pl.col("time").dt.month(),
            hour_of_day=pl.col("time").dt.hour(),
            era5_error=pl.col(variable.era5) - pl.col(variable.station),
            ukv_error=pl.col(variable.ukv) - pl.col(variable.station),
        )
        .with_columns(
            *(
                expression
                for product in PRODUCTS
                for expression in (
                    debiased(error=f"{product}_error", by=BIAS_BY_HOUR)
                    .abs()
                    .alias(f"abs_{product}_bias_removed_hour"),
                    debiased(error=f"{product}_error", by=BIAS_BY_MONTH)
                    .abs()
                    .alias(f"abs_{product}_bias_removed_month"),
                    pl.col(f"{product}_error").abs().alias(f"abs_{product}_raw"),
                )
            )
        )
    )


def interval_record(
    *, rows: pl.DataFrame, variable: Variable, label: str, planned: bool, score: str, scope: str
) -> IntervalRecord:
    """Interval one contrast, UKV-CEDA minus ERA5, on the given rows.

    Args:
        rows: Scored rows from `scored_rows`, restricted to the scope.
        variable: The variable.
        label: The contrast's label, such as `P1` or `exploratory`.
        planned: Whether the contrast was written into the plan before any result.
        score: One of `SCORES`.
        scope: What the rows are, such as `all` or `station S1`.

    Returns:
        The record. A scope with fewer than `MIN_MONTHS_FOR_INTERVAL` months holds a point estimate,
        a null interval, and the reading `no_interval`.
    """
    ukv = rows[f"abs_ukv_{score}"].to_numpy()
    era5 = rows[f"abs_era5_{score}"].to_numpy()
    months = rows["month"].to_numpy()
    n_months = len(np.unique(months))
    margin = MARGIN_STATION_SHARE * float(np.mean(rows[f"abs_era5_{PRIMARY_SCORE}"].to_numpy()))
    enough = n_months >= MIN_MONTHS_FOR_INTERVAL
    interval = bootstrap_row_difference(values=ukv - era5, months=months)
    lower, upper = (interval["lower_95"], interval["upper_95"]) if enough else (np.nan, np.nan)
    reading = (
        contrast_reading(difference=interval["difference"], lower=lower, upper=upper, margin=margin)
        if enough
        else "no_interval"
    )
    return {
        "variable": variable.name,
        "label": label,
        "planned": planned,
        "score": score,
        "scope": scope,
        "era5_mae": float(era5.mean()),
        "ukv_mae": float(ukv.mean()),
        "difference": interval["difference"],
        "lower_95": lower,
        "upper_95": upper,
        "margin": margin,
        "reading": reading,
        "n_rows": rows.height,
        "n_months": n_months,
        "enough_months": enough,
    }


def scope_rows(*, rows: pl.DataFrame) -> list[tuple[str, str, pl.DataFrame]]:
    """Cut the rows into the exploratory scopes.

    Args:
        rows: Scored rows for one variable.

    Returns:
        (kind, scope, rows) for each station, calendar year, half-year, and window.
    """
    scopes = [
        ("station", f"station {site}", rows.filter(pl.col("site") == site))
        for site in sorted(rows["site"].unique().to_list())
    ]
    scopes += [
        ("year", f"year {year}", rows.filter(pl.col("time").dt.year() == year))
        for year in sorted(rows["time"].dt.year().unique().to_list())
    ]
    winter = pl.col("time").dt.month().is_in([10, 11, 12, 1, 2, 3])
    scopes.append(("half-year", "October to March", rows.filter(winter)))
    scopes.append(("half-year", "April to September", rows.filter(~winter)))
    scopes.append(("window", "early window", rows.filter(pl.col("month") < EARLY_END_MONTH)))
    scopes.append(("window", "late window", rows.filter(pl.col("month") >= EARLY_END_MONTH)))
    return [scope for scope in scopes if scope[2].height]


def variable_records(*, frame: pl.DataFrame, variable: Variable) -> list[IntervalRecord]:
    """Interval every contrast of one variable.

    Args:
        frame: `station_hours.parquet`.
        variable: The variable.

    Returns:
        The planned contrast on all three scores, P2-lead for temperature, then the exploratory
        scopes on the primary score.
    """
    rows = scored_rows(frame=frame, variable=variable)
    records = [
        interval_record(
            rows=rows,
            variable=variable,
            label=variable.planned,
            planned=True,
            score=score,
            scope="all",
        )
        for score in SCORES
    ]
    if variable.name == "temperature":
        for scope, (low, high) in LEAD_SPLITS.items():
            records.append(
                interval_record(
                    rows=rows.filter(pl.col("lead_hours").is_between(low, high)),
                    variable=variable,
                    label="P2-lead",
                    planned=True,
                    score=PRIMARY_SCORE,
                    scope=scope,
                )
            )
    records += [
        interval_record(
            rows=rows.filter(pl.col("lead_hours") == lead),
            variable=variable,
            label="post hoc lead",
            planned=False,
            score=PRIMARY_SCORE,
            scope=f"lead {lead} h",
        )
        for lead in range(N_LEADS)
    ]
    records += [
        interval_record(
            rows=rows.filter(pl.col("site") != site),
            variable=variable,
            label="post hoc leave one out",
            planned=False,
            score=PRIMARY_SCORE,
            scope=f"without station {site}",
        )
        for site in sorted(rows["site"].unique().to_list())
        if rows["site"].n_unique() > 1
    ]
    for kind, scope, subset in scope_rows(rows=rows):
        deciding_window = variable.name == "temperature" and kind == "window"
        records.append(
            interval_record(
                rows=subset,
                variable=variable,
                label=variable.planned if deciding_window else f"exploratory {kind}",
                planned=deciding_window,
                score=PRIMARY_SCORE,
                scope=scope,
            )
        )
    return records


def _row_line(*, record: IntervalRecord) -> str:
    interval = (
        f"[{record['lower_95']:+.3f}, {record['upper_95']:+.3f}]"
        if record["enough_months"]
        else "too few months"
    )
    return (
        f"| {record['label']} | {record['scope']} | {record['score']} | {record['era5_mae']:.3f} "
        f"| {record['ukv_mae']:.3f} | {record['difference']:+.3f} | {interval} "
        f"| {record['margin']:.3f} | {record['reading']} | {record['n_rows']:,} "
        f"| {record['n_months']} |"
    )


def report_text(*, records: Sequence[IntervalRecord]) -> str:
    """Render every record as markdown tables, one per variable.

    Args:
        records: Every interval record.

    Returns:
        The report.
    """
    lines = [
        "### Set A: stations",
        "",
        (
            "Errors are in m/s for wind and in K for temperature. A negative difference favours "
            "UKV-CEDA. Rows labelled P1, P2, and P2-lead are planned. Every other row is "
            "exploratory. The reading compares the interval and the point estimate with the margin."
        ),
    ]
    for variable in VARIABLES:
        subset = [record for record in records if record["variable"] == variable.name]
        lines += [
            "",
            f"#### {variable.name.capitalize()} ({variable.unit})",
            "",
            (
                "| Label | Scope | Score | ERA5 MAE | UKV-CEDA MAE | UKV-CEDA minus ERA5 | 95% "
                "interval | Margin | Reading | Station-hours | Months |"
            ),
            "|---|---|---|---|---|---|---|---|---|---|---|",
            *(_row_line(record=record) for record in subset),
        ]
    return "\n".join(lines) + "\n"


def build_records(*, frame: pl.DataFrame) -> list[IntervalRecord]:
    """Interval every contrast of both variables.

    Args:
        frame: `station_hours.parquet`.

    Returns:
        The records.
    """
    return [
        record
        for variable in VARIABLES
        for record in variable_records(frame=frame, variable=variable)
    ]


def main() -> int:
    """Score the products against the stations, and write the intervals and the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the report; write nothing.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()
    frame = pl.read_parquet(arguments.output_dir / STATION_HOURS_NAME)
    records = build_records(frame=frame)
    report = report_text(records=records)
    if not arguments.dry_run:
        paths = [arguments.output_dir / INTERVALS_NAME, arguments.output_dir / REPORT_NAME]
        refuse_to_overwrite(paths=paths)
        pl.DataFrame(records).write_parquet(paths[0])
        paths[1].write_text(report)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
