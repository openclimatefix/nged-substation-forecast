"""Validate the WeatherNext 3 run files that `fetch_weathernext3.py` wrote.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/934>, following the
`data-validation` skill's checklist. It checks each run file in `_run_cache/` on its own, so peak
memory is one run, then the checks that span runs, and prints one PASS, FAIL, or SKIP line per
check. A run that raises is not evidence the data is right, and neither is a run that prints PASS: a
check that fails names the runs it failed in. **The report never prints a row count, a cell count,
or a coordinate,** because those reveal the size of the private trial-area box.

Checks per run file: `init_time` matches the file name; the lead-time axis is 1 to 360 hours in
steps of 1 hour (48 for a run whose init hour is not 00, 06, 12, or 18 UTC); no duplicate key; no
null and no `NaN` in any column, because the store documents no missing value; each value inside a
physical range (`RANGES`); the row count equals the number of leads times the cell count recorded
in the file's parquet metadata; radiation is near zero for night-time valid hours and positive at
midday in April to September; and direct radiation does not exceed total radiation by more than a
small tolerance. Across runs: every file carries the same grid hash. The night and midday checks
catch only a gross shift such as 12 hours or a timezone error, and do not verify the exact hour
convention.

Run it with `uv run python studies/weather_downloads/validate_weathernext3.py --directory
<directory under data/studies/weather>`, for example `--directory WeatherNext3`.
"""

import argparse
import sys
from pathlib import Path
from typing import Final

import polars as pl
from fetch_weathernext3 import LONG_RUN_HOURS, LONG_RUN_LEADS, SHORT_RUN_LEADS, VARIABLES
from paths import WEATHER_DOWNLOADS_DIR

TEMPERATURE: Final[str] = VARIABLES[0]
TOTAL_RADIATION: Final[str] = "surface_solar_radiation_downwards_1hr_mean"
DIRECT_RADIATION: Final[str] = "total_sky_direct_solar_radiation_at_surface_1hr_mean"
RADIATION_FIELDS: Final[tuple[str, str]] = (TOTAL_RADIATION, DIRECT_RADIATION)

WIND_LIMIT_M_S: Final[float] = 80.0
MAX_HOURLY_RADIATION_J_M2: Final[float] = 4.0e6

RANGES: Final[dict[str, tuple[float, float]]] = {
    "temperature_2m_mean": (230.0, 320.0),
    "u_component_of_wind_10m_mean": (-WIND_LIMIT_M_S, WIND_LIMIT_M_S),
    "v_component_of_wind_10m_mean": (-WIND_LIMIT_M_S, WIND_LIMIT_M_S),
    "u_component_of_wind_100m_mean": (-WIND_LIMIT_M_S, WIND_LIMIT_M_S),
    "v_component_of_wind_100m_mean": (-WIND_LIMIT_M_S, WIND_LIMIT_M_S),
    TOTAL_RADIATION: (0.0, MAX_HOURLY_RADIATION_J_M2),
    DIRECT_RADIATION: (0.0, MAX_HOURLY_RADIATION_J_M2),
}
"""Physical bounds in each variable's stored unit (K, m/s, and J/m2 accumulated over one hour).
Rounding to 13 bits moves a value by at most 1.2e-4 of itself, so a value outside these bounds is
not a rounding artefact."""

SECONDS_PER_HOUR: Final[int] = 3600
NIGHT_HOURS: Final[tuple[int, ...]] = (1, 2, 3)
"""UTC valid hours whose accumulation window ends before sunrise everywhere in Great Britain."""
NIGHT_MEDIAN_LIMIT_J_M2: Final[float] = 5.0 * SECONDS_PER_HOUR
"""A mean of 5 W/m2 over the hour."""
MIDDAY_MEAN_MINIMUM_J_M2: Final[float] = 100.0 * SECONDS_PER_HOUR
"""A mean of 100 W/m2 over the hour."""
SUMMER_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
DIRECT_EXCESS_FRACTION: Final[float] = 0.01
DIRECT_EXCESS_FLOOR_J_M2: Final[float] = 1.0 * SECONDS_PER_HOUR
"""Direct radiation may exceed total radiation by this fraction of total plus this floor (a mean of
1 W/m2 over the hour), which covers rounding."""

Results = dict[str, list[str]]
"""Check name mapped to the runs it failed in (empty if the check passed)."""

CHECKS: Final[tuple[str, ...]] = (
    "init_matches_name",
    "lead_axis",
    "keys",
    "nulls_and_nan",
    "ranges",
    "night_total",
    "night_direct",
    "midday_total",
    "midday_direct",
    "direct_le_total",
    "row_count",
)
"""Every per-run check, so that a check no run failed still prints a PASS line."""

_SKIPPED: Final[str] = "skipped"


def _fail(*, results: Results, check: str, label: str) -> None:
    """Record that `check` failed in the run `label`."""
    results.setdefault(check, []).append(label)


def _check_axes(*, frame: pl.DataFrame, label: str, results: Results) -> None:
    """Check `init_time` matches the file name, the lead-time axis, and key uniqueness."""
    init_times = frame["init_time"].unique().to_list()
    if len(init_times) != 1 or init_times[0].strftime("%Y%m%d_%H") != label:
        _fail(results=results, check="init_matches_name", label=label)
    hour = int(label.split("_")[1])
    n_leads = LONG_RUN_LEADS if hour in LONG_RUN_HOURS else SHORT_RUN_LEADS
    leads = frame["lead_time"].unique().sort().dt.total_hours().to_list()
    if leads != list(range(1, n_leads + 1)):
        _fail(results=results, check="lead_axis", label=label)
    key = ["init_time", "lead_time", "lat_index", "lon_index"]
    if frame.select(pl.struct(key).n_unique()).item() != frame.height:
        _fail(results=results, check="keys", label=label)


def _check_values(*, frame: pl.DataFrame, label: str, results: Results) -> None:
    """Check nulls, `NaN`s, and physical ranges of every value column."""
    for variable in VARIABLES:
        if frame[variable].null_count() or frame[variable].is_nan().any():
            _fail(results=results, check="nulls_and_nan", label=label)
        low, high = RANGES[variable]
        observed = frame.filter(pl.col(variable).is_finite()).select(
            low=pl.col(variable).min(), high=pl.col(variable).max()
        )
        observed_low, observed_high = observed.row(0)
        if observed_low is not None and (observed_low < low or observed_high > high):
            _fail(results=results, check="ranges", label=label)


def _check_radiation(*, frame: pl.DataFrame, label: str, results: Results) -> None:
    """Check radiation is near zero at night, positive at summer midday, and direct <= total."""
    valid = frame.select(
        valid_time=pl.col("init_time") + pl.col("lead_time"),
        total=pl.col(TOTAL_RADIATION),
        direct=pl.col(DIRECT_RADIATION),
    ).filter(pl.col("total").is_finite() & pl.col("direct").is_finite())
    hour = pl.col("valid_time").dt.hour()
    for field in ("total", "direct"):
        night = valid.filter(hour.is_in(NIGHT_HOURS)).select(pl.col(field).median()).item()
        midday = (
            valid.filter((hour == 12) & pl.col("valid_time").dt.month().is_in(SUMMER_MONTHS))
            .select(pl.col(field).mean())
            .item()
        )
        night_check, midday_check = f"night_{field}", f"midday_{field}"
        if night is None:
            results.setdefault(night_check, []).append(_SKIPPED)
        elif night >= NIGHT_MEDIAN_LIMIT_J_M2:
            _fail(results=results, check=night_check, label=label)
        if midday is None:
            results.setdefault(midday_check, []).append(_SKIPPED)
        elif midday <= MIDDAY_MEAN_MINIMUM_J_M2 and field == "total":
            _fail(results=results, check=midday_check, label=label)
    excess = valid.filter(
        pl.col("direct")
        > pl.col("total") * (1.0 + DIRECT_EXCESS_FRACTION) + DIRECT_EXCESS_FLOOR_J_M2
    )
    if excess.height:
        _fail(results=results, check="direct_le_total", label=label)


def _validate_run(*, path: Path, results: Results) -> str | None:
    """Run every per-run check on one run file; return its grid hash from the parquet metadata."""
    label = path.stem
    frame = pl.read_parquet(path)
    metadata = pl.read_parquet_metadata(path)
    _check_axes(frame=frame, label=label, results=results)
    _check_values(frame=frame, label=label, results=results)
    _check_radiation(frame=frame, label=label, results=results)
    n_cells = int(metadata.get("cell_count", "0"))
    if frame.height != frame["lead_time"].n_unique() * n_cells:
        _fail(results=results, check="row_count", label=label)
    return metadata.get("cell_hash")


def main() -> int:
    """Validate every run file in one product directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, help="Product directory name.")
    arguments = parser.parse_args()
    paths = sorted((WEATHER_DOWNLOADS_DIR / arguments.directory / "_run_cache").glob("*.parquet"))
    results: Results = {check: [] for check in CHECKS}
    hashes: set[str | None] = set()
    for path in paths:
        hashes.add(_validate_run(path=path, results=results))
    results["cell_hash_same"] = [] if len(hashes) == 1 and None not in hashes else ["all"]

    failed = 0
    for check, failures in results.items():
        real = sorted({run for run in failures if run != _SKIPPED})
        if real:
            failed += 1
            print(f"FAIL {check}: {', '.join(real)}")
        elif failures and len(failures) == len(paths):
            print(f"SKIP {check}: no qualifying rows in any run")
        elif failures:
            print(f"PASS {check} ({len(failures)} runs skipped)")
        else:
            print(f"PASS {check}")
    print("validation passed" if not failed else f"{failed} checks failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
