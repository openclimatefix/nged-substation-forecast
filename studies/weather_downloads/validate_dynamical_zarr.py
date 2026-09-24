"""Validate the GFS/GEFS parquets that `fetch_dynamical_zarr.py` wrote.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>, following the
`data-validation` skill's checklist. It checks each cached month on its own (so peak memory is one
month) and then the combined file's row count, and prints one report. A run that raises is not
evidence the data is right, and neither is a run that prints: a reader compares each printed number
with what it should be. The report holds counts and ranges only, never a coordinate.

Checks per month: the row count equals the product of the axis lengths, no duplicate key, no null,
the `NaN` count of every variable (only the two averaged radiation fields at lead time 0 may be
`NaN`), each variable's value range against a physical range, and the lead-time axis (step widths
and last lead time).

Run it with `uv run python studies/weather_downloads/validate_dynamical_zarr.py --directory
<directory under data/studies/weather>`, for example `--directory GEFS`.
"""

import argparse
import sys
from itertools import pairwise
from pathlib import Path
from typing import Final

import polars as pl
from fetch_dynamical_zarr import DATASETS, VARIABLES
from paths import WEATHER_DOWNLOADS_DIR

RANGES: Final[dict[str, tuple[float, float]]] = {
    "downward_short_wave_radiation_flux_surface": (0.0, 1500.0),
    "downward_long_wave_radiation_flux_surface": (100.0, 550.0),
    "temperature_2m": (-40.0, 45.0),
    "wind_u_10m": (-60.0, 60.0),
    "wind_v_10m": (-60.0, 60.0),
    "wind_u_100m": (-80.0, 80.0),
    "wind_v_100m": (-80.0, 80.0),
    "pressure_surface": (85_000.0, 110_000.0),
}
"""Physical bounds for the trial area, in each variable's stored unit. Rounding to 13 bits moves a
value by at most 1.2e-4 of itself, so a value outside these bounds is not a rounding artefact."""

AVERAGED_FIELDS: Final[tuple[str, ...]] = (
    "downward_short_wave_radiation_flux_surface",
    "downward_long_wave_radiation_flux_surface",
)
"""Averaged over the preceding step, so `NaN` at lead time 0."""

LEAD_STEPS_HOURS: Final[dict[str, tuple[set[int], int]]] = {
    "GFS": ({1, 3}, 384),
    "GEFS": ({3, 6}, 840),
}
"""Per model: the step widths (hours) the lead-time axis may contain, and the last lead time."""


def _validate_month(*, path: Path, label: str) -> list[str]:
    """Return one problem string per failed check on one cached month (empty if all pass)."""
    frame = pl.read_parquet(path)
    problems: list[str] = []
    key = [
        c
        for c in ("init_time", "ensemble_member", "lead_time", "lat_index", "lon_index")
        if c in frame.columns
    ]
    n_unique = frame.select(pl.struct(key).n_unique()).item()
    if n_unique != frame.height:
        problems.append(f"{frame.height - n_unique} duplicate keys")
    expected_rows = 1
    for column in key:
        expected_rows *= frame[column].n_unique()
    if expected_rows != frame.height:
        problems.append(f"{frame.height} rows, but the axes are {expected_rows} long combined")
    for variable in VARIABLES:
        column = frame[variable]
        if column.null_count():
            problems.append(f"{variable}: {column.null_count()} nulls")
        nan_rows = frame.filter(pl.col(variable).is_nan())
        if variable in AVERAGED_FIELDS:
            stray = nan_rows.filter(pl.col("lead_time") > pl.duration(hours=0)).height
            if stray:
                problems.append(f"{variable}: {stray} NaN rows at lead time above 0")
        elif nan_rows.height:
            problems.append(f"{variable}: {nan_rows.height} NaN rows")
        low, high = RANGES[variable]
        finite = frame.filter(pl.col(variable).is_finite()).select(
            low=pl.col(variable).min(), high=pl.col(variable).max()
        )
        observed_low, observed_high = finite.row(0)
        if observed_low is not None and (observed_low < low or observed_high > high):
            problems.append(f"{variable}: range {observed_low:.4g} to {observed_high:.4g}")
    leads_hours = frame["lead_time"].unique().sort().dt.total_hours().to_list()
    allowed_steps, last_lead = LEAD_STEPS_HOURS[label]
    steps = {b - a for a, b in pairwise(leads_hours)}
    if leads_hours[0] != 0 or leads_hours[-1] != last_lead or not steps <= allowed_steps:
        problems.append(f"lead axis {leads_hours[0]} to {leads_hours[-1]} h, steps {sorted(steps)}")
    print(
        f"{path.stem}: {frame.height} rows, {frame['init_time'].n_unique()} init_times, "
        f"lead steps {sorted(steps)} h, up to {leads_hours[-1]} h"
    )
    return problems


def main() -> int:
    """Validate every cached month and the combined file in one product directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, help="Product directory name.")
    arguments = parser.parse_args()
    product_dir = WEATHER_DOWNLOADS_DIR / arguments.directory
    label = next(name for name in DATASETS.values() if arguments.directory.startswith(name))
    month_paths = sorted((product_dir / "_month_cache").glob("*.parquet"))
    all_problems: list[str] = []
    month_rows = 0
    for path in month_paths:
        problems = _validate_month(path=path, label=label)
        all_problems += [f"{path.stem}: {problem}" for problem in problems]
        month_rows += pl.scan_parquet(path).select(pl.len()).collect().item()
    combined_rows = (
        pl.scan_parquet(product_dir / f"{label}.parquet").select(pl.len()).collect().item()
    )
    if combined_rows != month_rows:
        all_problems.append(f"combined file has {combined_rows} rows, months sum to {month_rows}")
    print(f"combined: {combined_rows} rows from {len(month_paths)} months")
    for problem in all_problems:
        print(f"PROBLEM {problem}")
    print("validation passed" if not all_problems else f"{len(all_problems)} problems")
    return 1 if all_problems else 0


if __name__ == "__main__":
    sys.exit(main())
