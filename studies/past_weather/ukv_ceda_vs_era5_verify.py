"""Check the rows of the UKV-CEDA against ERA5 study before any fit.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/1024>. It reads the rows that
`ukv_ceda_vs_era5_build.py` wrote and prints four checks into `verify.md`. A fit is not run until
this script has run and the report has been read.

**Column counts.** Every arm's feature columns are printed, and the two arms of every contrast must
carry the same number of distinct columns.

**Lag scan.** Each product is compared with the station at lags of -3 to +3 hours, and the error
must be lowest at lag 0 for both products and both variables. Station wind is the mean over the 10
minutes ending 10 minutes before its label, and both products are instantaneous at the label. A
product that is lowest at another lag has a timestamp convention that differs from the station's,
and the script exits non-zero.

**Monthly steps (post hoc and exploratory).** The monthly mean of UKV-CEDA minus ERA5, UKV-CEDA
minus the station, and ERA5 minus the station, for wind and for temperature, raw and with each
calendar month's mean removed, is written to a table and a chart. `step_candidates` lists the months
that start the largest differences between the mean of the following 6 months and the mean of the
preceding 6, in units of the series' own month-to-month spread. No pass or fail threshold applies,
because none was stated before the series was seen. The maintainer reads the table and the chart,
and records any era boundary as a deviation. The report also states what the repository records
about the Met Office's PS44.

**Lead distribution.** The count of station-hours at each UKV-CEDA lead, which is 0 to 5 hours.

Run it with `uv run python studies/past_weather/ukv_ceda_vs_era5_verify.py`. `--dry-run` prints the
report and writes nothing. A fresh run stops (`refuse_to_overwrite`) while an output exists.
"""

import argparse
import sys
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import polars as pl
import ukv_ceda_station_scores as scores
from studies.guards import refuse_to_overwrite
from ukv_ceda_vs_era5_build import (
    OUTPUT_DIR,
    PRODUCTS,
    SOLAR_ROWS_NAME,
    STATION_HOURS_NAME,
    WIND_ROWS_NAME,
    check_arm_widths,
    solar_arm_columns,
    wind_arm_columns,
)

VERIFY_NAME: Final[str] = "verify.md"
VERIFY_FAILED_NAME: Final[str] = "verify_failed.md"
"""A failed run writes this, not `verify.md`, which the fit requires, so a fit cannot start."""
STEPS_NAME: Final[str] = "monthly_steps.parquet"
CHART_NAME: Final[str] = "monthly_steps.svg"

LAGS: Final[tuple[int, ...]] = (-3, -2, -1, 0, 1, 2, 3)
"""The hours by which a product is shifted against the station in the lag scan."""

STEP_WINDOW_MONTHS: Final[int] = 6
"""How many months either side of a candidate step are averaged."""

STEP_SERIES: Final[tuple[str, ...]] = ("ukv_minus_era5", "ukv_minus_station", "era5_minus_station")
"""The monthly series whose steps are listed. A step in UKV minus the station that has no matching
step in UKV minus ERA5 points at the station or at ERA5, not at UKV."""

PS44_NOTE: Final[str] = (
    "PS44, the Met Office physics version between PS43 and PS45: the repository's data-sources "
    "roadmap (`docs/roadmap/data-sources.md`, the UKV upgrades table) records that no date or "
    "content for PS44 could be found. This study did not repeat the search, so the table below is "
    "the only check for an unnamed change."
)

N_STEP_CANDIDATES: Final[int] = 3
"""How many candidate steps are listed per series."""


def lag_scan(*, frame: pl.DataFrame, variable: scores.Variable) -> pl.DataFrame:
    """Score each product against the station at each lag.

    The product at `time + lag` is compared with the station reading at `time`, on the raw
    absolute error and on the station-hours where all three values exist at every lag.

    Args:
        frame: `station_hours.parquet`.
        variable: The variable.

    Returns:
        One row per product and lag with `mae` and `n_rows`.
    """
    shifted = [
        frame.select(
            "site",
            pl.col("time").dt.offset_by(f"{-lag}h"),
            pl.lit(lag).alias("lag"),
            era5=pl.col(variable.era5),
            ukv=pl.col(variable.ukv),
        )
        for lag in LAGS
    ]
    observed = frame.select("site", "time", station=pl.col(variable.station))
    joined = (
        pl.concat(shifted)
        .join(observed, on=["site", "time"])
        .drop_nulls(["era5", "ukv", "station"])
        .filter(~pl.col("era5").is_nan(), ~pl.col("ukv").is_nan(), ~pl.col("station").is_nan())
    )
    complete = joined.group_by("site", "time").agg(n=pl.len()).filter(pl.col("n") == len(LAGS))
    kept = joined.join(complete.select("site", "time"), on=["site", "time"])
    return pl.concat(
        kept.group_by("lag")
        .agg(mae=(pl.col(product) - pl.col("station")).abs().mean(), n_rows=pl.len())
        .with_columns(product=pl.lit(product))
        for product in ("era5", "ukv")
    ).sort("product", "lag")


def lowest_lag(*, scan: pl.DataFrame, product: str) -> int:
    """Return the lag at which a product's error is lowest.

    Args:
        scan: `lag_scan`'s result.
        product: `era5` or `ukv`.

    Returns:
        The lag in hours.
    """
    rows = scan.filter(pl.col("product") == product)
    return int(rows.sort("mae")["lag"][0])


def monthly_steps(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return the monthly means of UKV minus ERA5, UKV minus station, and ERA5 minus station.

    Each series uses the station-hours where the station and both products have a value, so the
    two series of one variable rest on the same rows. Only the stations that have such rows in every
    month any station has them contribute, so a station that drops out part-way (ERA5 wind at one
    station ends in 2023-12) cannot make a step.

    Args:
        frame: `station_hours.parquet`.

    Returns:
        One row per variable and month with the three series of `STEP_SERIES` and `n_rows`.
    """
    parts: list[pl.DataFrame] = []
    for variable in scores.VARIABLES:
        columns = (variable.station, variable.era5, variable.ukv)
        valid = frame.filter(*(pl.col(c).is_not_null() & pl.col(c).is_not_nan() for c in columns))
        months_by_site = valid.group_by("site").agg(n_months=pl.col("month").n_unique())
        whole_record = months_by_site.filter(pl.col("n_months") == pl.col("n_months").max())
        parts.append(
            valid.join(whole_record.select("site"), on="site")
            .group_by("month")
            .agg(
                ukv_minus_era5=(pl.col(variable.ukv) - pl.col(variable.era5)).mean(),
                ukv_minus_station=(pl.col(variable.ukv) - pl.col(variable.station)).mean(),
                era5_minus_station=(pl.col(variable.era5) - pl.col(variable.station)).mean(),
                n_rows=pl.len(),
            )
            .with_columns(variable=pl.lit(variable.name))
        )
    return pl.concat(parts).sort("variable", "month")


def deseasonalised(*, steps: pl.DataFrame) -> pl.DataFrame:
    """Remove each calendar month's mean from every series, within each variable.

    A raw monthly series carries a seasonal cycle, which inflates a step statistic. Subtracting
    each calendar month's mean over the years leaves what the cycle does not explain.

    Args:
        steps: `monthly_steps`'s result.

    Returns:
        `steps` with a `<series>_deseasonalised` column for each of `STEP_SERIES`.
    """
    calendar_month = pl.col("month").str.slice(5, 2)
    return steps.with_columns(
        (pl.col(series) - pl.col(series).mean().over("variable", calendar_month)).alias(
            f"{series}_deseasonalised"
        )
        for series in STEP_SERIES
    )


def step_candidates(
    *, series: np.ndarray, window: int = STEP_WINDOW_MONTHS
) -> list[tuple[int, float]]:
    """Find the positions where a monthly series steps.

    For each position `i` with `window` months on both sides, the step is the mean of the `window`
    months from `i` on, minus the mean of the `window` months before `i`, divided by the standard
    deviation of the series' month-to-month differences.

    Args:
        series: The monthly values, in month order.
        window: How many months are averaged on each side.

    Returns:
        (position, size) for the `N_STEP_CANDIDATES` largest steps by absolute size, largest
        first. Empty when the series is shorter than two windows.
    """
    if len(series) < 2 * window:
        return []
    spread = float(np.std(np.diff(series)))
    steps = [
        (i, float(series[i : i + window].mean() - series[i - window : i].mean()) / spread)
        for i in range(window, len(series) - window + 1)
    ]
    return sorted(steps, key=lambda step: -abs(step[1]))[:N_STEP_CANDIDATES]


def steps_chart(*, steps: pl.DataFrame) -> alt.Chart:
    """Draw the monthly means as lines, one panel per variable.

    Args:
        steps: `monthly_steps`'s result, which holds the raw series.

    Returns:
        The chart.
    """
    long = steps.unpivot(
        list(STEP_SERIES),
        index=["variable", "month"],
        variable_name="series",
        value_name="mean",
    ).with_columns(date=pl.col("month").str.to_date("%Y-%m"))
    return (
        alt.Chart(long)
        .mark_line()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("date:T", title="Month"),
            y=alt.Y("mean:Q", title="Monthly mean (m/s for wind, K for temperature)"),
            color=alt.Color("series:N", title=""),
        )
        .properties(width=620, height=140)
        .facet(row=alt.Row("variable:N", title=""))
        .resolve_scale(y="independent")
    )


def column_lines() -> list[str]:
    """Print every arm's feature columns and check the widths.

    Returns:
        Markdown lines.
    """
    check_arm_widths()
    lines = ["#### Every arm's feature columns", ""]
    for domain, function in (("wind", wind_arm_columns), ("solar", solar_arm_columns)):
        for product in PRODUCTS:
            for shuffled in (False, True):
                columns = function(product=product, shuffled=shuffled)
                name = f"{domain}_{product}" + ("_shuffled" if shuffled else "")
                lines.append(
                    f"- `{name}` ({len(columns)} columns): {', '.join(f'`{c}`' for c in columns)}"
                )
    return lines


def report_lines(
    *, station_hours: pl.DataFrame, wind: pl.DataFrame, solar: pl.DataFrame
) -> tuple[list[str], list[str], pl.DataFrame]:
    """Run every check and render the report.

    Args:
        station_hours: Set A.
        wind: The wind rows, for the era and lead check.
        solar: The solar rows.

    Returns:
        Markdown lines, the failed checks, and the monthly steps.
    """
    failures: list[str] = []
    lines = ["### Checks before the fits", "", *column_lines(), "", "#### Lag scan", ""]
    lines += [
        "Raw mean absolute error against the station, by the hours the product is shifted.",
        "",
        "| Variable | Product | " + " | ".join(f"{lag:+d}" for lag in LAGS) + " | Lowest at |",
        "|---|---|" + "---|" * (len(LAGS) + 1),
    ]
    for variable in scores.VARIABLES:
        scan = lag_scan(frame=station_hours, variable=variable)
        for product in ("era5", "ukv"):
            rows = scan.filter(pl.col("product") == product).sort("lag")
            best = lowest_lag(scan=scan, product=product)
            if best != 0:
                failures.append(f"{variable.name} {product} is lowest at lag {best:+d}")
            maes = " | ".join(f"{value:.3f}" for value in rows["mae"])
            lines.append(f"| {variable.name} | {product} | {maes} | {best:+d} |")
    steps = deseasonalised(steps=monthly_steps(frame=station_hours))
    lines += [
        "",
        "#### Monthly steps (post hoc and exploratory, with no pass or fail threshold)",
        "",
        PS44_NOTE,
        "",
        (
            "Each series below is a monthly mean over the stations with values in every month. "
            "Step candidates are the months that start the largest differences between the mean of "
            "the following 6 months and the mean of the preceding 6, in units of the series' own "
            "month-to-month spread. A step in UKV minus the station with no matching step in UKV "
            "minus ERA5 points at the station or at ERA5. No cut-off is applied, because none was "
            "stated before the series was seen; the maintainer reads the table and the chart and "
            "records any era boundary as a deviation."
        ),
        "",
    ]
    for variable in scores.VARIABLES:
        for series in STEP_SERIES:
            for suffix in ("", "_deseasonalised"):
                rows = steps.filter(pl.col("variable") == variable.name)
                found = step_candidates(series=rows[f"{series}{suffix}"].to_numpy())
                months = rows["month"].to_list()
                text = ", ".join(f"{months[i]} ({size:+.1f})" for i, size in found)
                kind = "deseasonalised" if suffix else "raw"
                lines.append(
                    f"- {variable.name}, {series}, {kind}: largest steps start at "
                    f"{text or 'none (too short)'}."
                )
    lines += ["", "| Variable | Month | Station-hours | " + " | ".join(STEP_SERIES) + " |"]
    lines += ["|---|---|---|" + "---|" * len(STEP_SERIES)]
    lines += [
        f"| {row['variable']} | {row['month']} | {row['n_rows']:,} | "
        + " | ".join(f"{row[series]:+.3f}" for series in STEP_SERIES)
        + " |"
        for row in steps.iter_rows(named=True)
    ]
    lines += [
        "",
        "#### Lead distribution (UKV-CEDA lead of each station-hour, in hours)",
        "",
        f"- {dict(sorted(station_hours.group_by('lead_hours').len().iter_rows()))}.",
        "",
        "#### Fit rows",
        "",
        f"- Wind rows by era: {dict(sorted(wind.group_by('era_code').len().iter_rows()))}.",
        f"- Solar rows by era: {dict(sorted(solar.group_by('era_code').len().iter_rows()))}.",
    ]
    return lines, failures, steps


def main() -> int:
    """Run the checks, write the report, and exit non-zero if a lag check fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the report; write nothing.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()
    directory: Path = arguments.output_dir
    lines, failures, steps = report_lines(
        station_hours=pl.read_parquet(directory / STATION_HOURS_NAME),
        wind=pl.read_parquet(directory / WIND_ROWS_NAME),
        solar=pl.read_parquet(directory / SOLAR_ROWS_NAME),
    )
    status = (
        "Status: FAILED: " + "; ".join(failures)
        if failures
        else "Status: every check passed. The monthly-step section has no pass or fail threshold."
    )
    report = "\n".join([lines[0], "", status, *lines[1:]]) + "\n"
    if not arguments.dry_run:
        name = VERIFY_FAILED_NAME if failures else VERIFY_NAME
        paths = [directory / name, directory / STEPS_NAME, directory / CHART_NAME]
        refuse_to_overwrite(paths=paths)
        paths[0].write_text(report)
        steps.write_parquet(paths[1])
        steps_chart(steps=steps).save(paths[2])
    sys.stdout.write(report)
    if failures:
        sys.stdout.write("\nCHECKS FAILED:\n" + "\n".join(f"- {f}" for f in failures) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
