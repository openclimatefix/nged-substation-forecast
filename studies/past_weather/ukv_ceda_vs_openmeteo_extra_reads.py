"""Print two reads of the saved results that the page quotes: the first run, and wind by UTC hour.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/1051>. It fits nothing.

**The first run kept the two spans in which Open-Meteo's 10 m wind speed is built differently.**
The rerun dropped them from every wind arm after the first fit had been read. The wind contrasts
P1 and P3 were planned on all rows, so the page reports both runs, and the first run's intervals
sit in `superseded/first_run/intervals.parquet`. The script prints that run's wind P1 and P3, in all
hours and at CEDA lead 0, at both settings (the first run called the settings `pooled` and
`sensitivity`). The P3 verdict is a penalty in both runs and P1's is not, which the table shows.

**A row's CEDA lead is its UTC hour modulo 6, so lead and time of day cannot be told apart.** The
script also prints wind P1 and P3 by UTC hour from the saved losses of the rerun, at the primary
setting, with each hour's CEDA run and lead.

Nothing here prints a generator's name, identifier or coordinates. Run it with
`uv run python studies/past_weather/ukv_ceda_vs_openmeteo_extra_reads.py`. A fresh run stops
(`refuse_to_overwrite`) while an output exists.
"""

import argparse
import sys
from pathlib import Path
from typing import Final

import polars as pl
from studies.bootstrap import MIN_MONTHS_FOR_INTERVAL, bootstrap_difference
from studies.guards import refuse_to_overwrite
from ukv_ceda_vs_openmeteo_build import OUTPUT_DIR
from ukv_ceda_vs_openmeteo_fit import METRIC, PERCENTAGE_POINTS, PRIMARY_SETTING

REPORT_NAME: Final[str] = "extra_reads_report.md"
FIRST_RUN_DIR_NAME: Final[str] = "superseded/first_run"
FIRST_RUN_SETTINGS: Final[dict[str, str]] = {"pooled": "primary", "sensitivity": "second"}
"""The first run's names for the two settings, mapped to the names used now."""

WIND_CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("P1", "ceda_wind_10m", "om_wind_10m"),
    ("P3", "ceda_wind_10m_scored_on_om", "om_wind_10m"),
)
"""The wind contrasts the page reports by hour: label, treatment arm, reference arm."""

LEAD_CYCLE_HOURS: Final[int] = 6


def first_run_lines(*, first_run_dir: Path) -> list[str]:
    """Print the first run's wind P1 and P3, in all hours and at CEDA lead 0.

    Args:
        first_run_dir: The folder that holds the first run's `intervals.parquet`.

    Returns:
        Markdown lines.
    """
    records = (
        pl.read_parquet(first_run_dir / "intervals.parquet")
        .filter(
            (pl.col("domain") == "wind")
            & pl.col("label").is_in(["P1", "P3"])
            & pl.col("scope").is_in(["all", "lead 0 only"])
        )
        .with_columns(setting=pl.col("setting").replace(FIRST_RUN_SETTINGS))
        .sort("label", "scope", "setting")
    )
    lines = [
        "## The first run, which kept the two spans: wind P1 and P3",
        "",
        (
            "Difference in points of capacity, CEDA minus Open-Meteo, with the 95% interval. The "
            "scope `all` is the planned scope."
        ),
        "",
        "| Contrast | Scope | Setting | Difference | Reading | Months |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['label']} | {r['scope']} | {r['setting']} | {r['difference_pp']:+.3f} "
        f"[{r['lower_95_pp']:+.3f}, {r['upper_95_pp']:+.3f}] | {r['reading']} | {r['n_months']} |"
        for r in records.iter_rows(named=True)
    ]
    return [*lines, ""]


def by_hour_lines(*, losses: pl.DataFrame) -> list[str]:
    """Print wind P1 and P3 by UTC hour at the primary setting.

    Args:
        losses: The wind losses of the rerun.

    Returns:
        Markdown lines, one row per UTC hour with its CEDA run and lead.
    """
    primary = losses.filter(pl.col("setting") == PRIMARY_SETTING)
    lines = [
        "## Wind P1 and P3 by UTC hour, primary setting",
        "",
        (
            "A row's CEDA lead is its UTC hour modulo 6, and its CEDA run is the 00, 06, 12, or 18 "
            "UTC run at or before the hour, so lead and time of day cannot be told apart."
        ),
        "",
        "| UTC hour | CEDA run | Lead | P1 | P3 |",
        "|---|---|---|---|---|",
    ]
    for hour in range(24):
        subset = primary.filter(pl.col("time").dt.hour() == hour)
        cells = []
        for _, treatment, reference in WIND_CONTRASTS:
            interval = bootstrap_difference(
                losses=subset, treatment=treatment, reference=reference, metric=METRIC
            )
            if interval["n_months"] < MIN_MONTHS_FOR_INTERVAL:
                cells.append("(too few months)")
                continue
            cells.append(
                f"{interval['difference'] * PERCENTAGE_POINTS:+.3f} "
                f"[{interval['lower_95'] * PERCENTAGE_POINTS:+.3f}, "
                f"{interval['upper_95'] * PERCENTAGE_POINTS:+.3f}]"
            )
        run = hour // LEAD_CYCLE_HOURS * LEAD_CYCLE_HOURS
        lines.append(
            f"| {hour:02d} | {run:02d} UTC | {hour % LEAD_CYCLE_HOURS} | {cells[0]} | {cells[1]} |"
        )
    return [*lines, ""]


def main() -> int:
    """Print the first run's wind rows and the wind rows by UTC hour."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()
    directory: Path = arguments.output_dir
    path = directory / REPORT_NAME
    refuse_to_overwrite(paths=[path])
    lines = [
        "# Two reads of the saved results",
        "",
        *first_run_lines(first_run_dir=directory / FIRST_RUN_DIR_NAME),
        *by_hour_lines(losses=pl.read_parquet(directory / "losses_wind.parquet")),
    ]
    path.write_text("\n".join(lines))
    sys.stdout.write(f"Wrote {path}.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
