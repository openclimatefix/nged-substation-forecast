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
null, `NaN`, or infinite value in any column, because the store documents no missing value;
each value inside a
physical range (`RANGES`); the row count equals the number of leads times the cell count recorded
in the file's parquet metadata; radiation is near zero for night-time valid hours and positive at
midday in April to September; and direct radiation does not exceed total radiation by more than a
small tolerance (`ranges` and `direct_le_total` fail above 0.01% of a run's rows, and print that
percentage); and no (variable, lead time) slice is constant across the crop, except dark
radiation slices. Radiation may dip to -3600 J/m2, because an ensemble mean from a machine-learning
weather model can be slightly negative at night. Across runs: every file carries the same grid
hash; the run labels cover every init hour in the first-to-last date range, except runs that
`lineage.json` lists as skipped for a missing `success` marker; the combined `WeatherNext3.parquet`
has as many rows as the run files together; and `_grid_cells.parquet` has the recorded cell count
and hash. The night and midday checks catch only a gross shift such as 12 hours or a timezone
error, and do not verify the exact hour convention.

Run it with `uv run python studies/weather_downloads/validate_weathernext3.py --directory
<directory under data/studies/weather>`, for example `--directory WeatherNext3`.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import polars as pl
import pyarrow.parquet as pq
from fetch_weathernext3 import LONG_RUN_HOURS, LONG_RUN_LEADS, SHORT_RUN_LEADS, VARIABLES
from paths import WEATHER_DOWNLOADS_DIR

TEMPERATURE: Final[str] = VARIABLES[0]
TOTAL_RADIATION: Final[str] = "surface_solar_radiation_downwards_1hr_mean"
DIRECT_RADIATION: Final[str] = "total_sky_direct_solar_radiation_at_surface_1hr_mean"
RADIATION_FIELDS: Final[tuple[str, str]] = (TOTAL_RADIATION, DIRECT_RADIATION)

WIND_LIMIT_M_S: Final[float] = 80.0
MAX_HOURLY_RADIATION_J_M2: Final[float] = 4.0e6
MIN_HOURLY_RADIATION_J_M2: Final[float] = -3600.0
"""An ensemble mean from a machine-learning weather model may dip slightly below zero at night, so
the lower bound allows -3600 J/m2 (a mean of -1 W/m2 over the hour)."""
OFFENDING_ROW_FRACTION_LIMIT: Final[float] = 1e-4
"""`ranges` and `direct_le_total` fail only when more than 0.01% of a run's rows offend, so that a
few cells of a machine-learning weather model's noise do not fail a run, while a shifted or
corrupted field, which offends in far more rows, does."""

RANGES: Final[dict[str, tuple[float, float]]] = {
    "temperature_2m_mean": (230.0, 320.0),
    "u_component_of_wind_10m_mean": (-WIND_LIMIT_M_S, WIND_LIMIT_M_S),
    "v_component_of_wind_10m_mean": (-WIND_LIMIT_M_S, WIND_LIMIT_M_S),
    "u_component_of_wind_100m_mean": (-WIND_LIMIT_M_S, WIND_LIMIT_M_S),
    "v_component_of_wind_100m_mean": (-WIND_LIMIT_M_S, WIND_LIMIT_M_S),
    TOTAL_RADIATION: (MIN_HOURLY_RADIATION_J_M2, MAX_HOURLY_RADIATION_J_M2),
    DIRECT_RADIATION: (MIN_HOURLY_RADIATION_J_M2, MAX_HOURLY_RADIATION_J_M2),
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
"""Direct radiation may exceed `max(total, 0)` by this fraction of it plus this floor (a mean of
1 W/m2 over the hour), which covers rounding and a slightly negative total."""

Results = dict[str, list[str]]
"""Check name mapped to the runs it failed in (empty if the check passed)."""

CHECKS: Final[tuple[str, ...]] = (
    "readable",
    "init_matches_name",
    "lead_axis",
    "keys",
    "nulls_and_nan",
    "ranges",
    "night_total",
    "night_direct",
    "midday_total",
    "direct_le_total",
    "row_count",
    "constant_slice",
)
"""Every per-run check, so that a check no run failed still prints a PASS line."""

_SKIPPED: Final[str] = "skipped"


def _fail(*, results: Results, check: str, label: str) -> None:
    """Record that `check` failed in the run `label`."""
    results.setdefault(check, []).append(label)


def _fail_if_many_rows(
    *, results: Results, check: str, label: str, offending: int, total: int
) -> None:
    """Record a failure of `check` if more than `OFFENDING_ROW_FRACTION_LIMIT` of rows offend.

    The failure names the run and the percentage of offending rows, never a count.
    """
    fraction = offending / total if total else 0.0
    if fraction > OFFENDING_ROW_FRACTION_LIMIT:
        _fail(results=results, check=check, label=f"{label} ({fraction:.3%} of rows)")


def _check_axes(*, frame: pl.DataFrame, label: str, results: Results) -> None:
    """Check `init_time` matches the file name, the lead-time axis, and key uniqueness."""
    init_times = frame["init_time"].unique().to_list()
    if len(init_times) != 1 or init_times[0].strftime("%Y%m%d_%H") != label:
        _fail(results=results, check="init_matches_name", label=label)
    hour = int(label.split("_")[1])
    n_leads = LONG_RUN_LEADS if hour in LONG_RUN_HOURS else SHORT_RUN_LEADS
    try:
        leads = frame["lead_time"].unique().sort().dt.total_hours().to_list()
    except pl.exceptions.PolarsError:  # for example an integer `lead_time`, not a duration
        leads = []
    if leads != list(range(1, n_leads + 1)):
        _fail(results=results, check="lead_axis", label=label)
    key = ["init_time", "lead_time", "lat_index", "lon_index"]
    if frame.select(pl.struct(key).n_unique()).item() != frame.height:
        _fail(results=results, check="keys", label=label)


def _check_values(*, frame: pl.DataFrame, label: str, results: Results) -> None:
    """Check nulls, `NaN`s, infinite values, and physical ranges of every value column."""
    for variable in VARIABLES:
        column = pl.col(variable)
        if frame[variable].null_count() or frame.select((~column.is_finite()).any()).item():
            _fail(results=results, check="nulls_and_nan", label=label)
        low, high = RANGES[variable]
        offending = frame.select(((column < low) | (column > high)).sum()).item()
        _fail_if_many_rows(
            results=results, check="ranges", label=label, offending=offending, total=frame.height
        )


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
        night_check = f"night_{field}"
        if night is None:
            results.setdefault(night_check, []).append(_SKIPPED)
        elif night >= NIGHT_MEDIAN_LIMIT_J_M2:
            _fail(results=results, check=night_check, label=label)
    midday = (
        valid.filter((hour == 12) & pl.col("valid_time").dt.month().is_in(SUMMER_MONTHS))
        .select(pl.col("total").mean())
        .item()
    )
    if midday is None:
        results.setdefault("midday_total", []).append(_SKIPPED)
    elif midday <= MIDDAY_MEAN_MINIMUM_J_M2:
        _fail(results=results, check="midday_total", label=label)
    allowed = pl.max_horizontal(pl.col("total"), 0.0) * (1.0 + DIRECT_EXCESS_FRACTION)
    excess = valid.select((pl.col("direct") > allowed + DIRECT_EXCESS_FLOOR_J_M2).sum()).item()
    _fail_if_many_rows(
        results=results,
        check="direct_le_total",
        label=label,
        offending=excess,
        total=valid.height,
    )


def _check_constant_slices(*, frame: pl.DataFrame, label: str, results: Results) -> None:
    """Check no (variable, lead time) slice has the same value in every cell of the crop.

    A constant slice is a sign of a fill value or a broken read. A radiation slice at night is
    legitimately constant (near zero everywhere), so a radiation slice is exempt when its largest
    value is below `NIGHT_MEDIAN_LIMIT_J_M2`.
    """
    for variable in VARIABLES:
        spread = frame.group_by("lead_time").agg(
            low=pl.col(variable).min(), high=pl.col(variable).max()
        )
        constant = spread.filter(pl.col("low") == pl.col("high"))
        if variable in RADIATION_FIELDS:
            constant = constant.filter(pl.col("high") >= NIGHT_MEDIAN_LIMIT_J_M2)
        if constant.height:
            _fail(results=results, check="constant_slice", label=label)


def _guarded(*, results: Results, check: str, label: str, action: Callable[[], None]) -> None:
    """Call `action`; if it raises, record a failure of `check` and carry on with other checks.

    A malformed run file (a wrong column type, a truncated file) is a finding, not a reason to
    stop validating. The report names the exception's type only.
    """
    try:
        action()
    except Exception as error:  # noqa: BLE001  # any failure to run a check is itself a finding
        _fail(results=results, check=check, label=f"{label} (check raised {type(error).__name__})")


def _validate_run(*, path: Path, results: Results) -> str | None:
    """Run every per-run check on one run file; return its grid hash from the parquet metadata."""
    label = path.stem
    try:
        frame = pl.read_parquet(path)
        metadata = pl.read_parquet_metadata(path)
    except Exception as error:  # noqa: BLE001  # an unreadable file is a finding
        _fail(
            results=results,
            check="readable",
            label=f"{label} ({type(error).__name__})",
        )
        return None
    for check, function in (
        ("lead_axis", _check_axes),
        ("nulls_and_nan", _check_values),
        ("night_total", _check_radiation),
        ("constant_slice", _check_constant_slices),
    ):
        _guarded(
            results=results,
            check=check,
            label=label,
            action=lambda function=function: function(frame=frame, label=label, results=results),
        )
    _guarded(
        results=results,
        check="row_count",
        label=label,
        action=lambda: _check_row_count(
            frame=frame, metadata=metadata, label=label, results=results
        ),
    )
    return metadata.get("cell_hash")


def _check_row_count(
    *, frame: pl.DataFrame, metadata: dict[str, str], label: str, results: Results
) -> None:
    """Check the rows equal the number of lead times times the recorded cell count."""
    n_cells = int(metadata.get("cell_count", "0"))
    if frame.height != frame["lead_time"].n_unique() * n_cells:
        _fail(results=results, check="row_count", label=label)


def _check_completeness(*, paths: list[Path], product_dir: Path, results: Results) -> None:
    """Check every init hour selected by the files' first-to-last date range has a run file.

    The selected hours are the init hours that occur in the file names. A run whose `success`
    marker was missing, and a wanted run the bucket did not list, are listed in `lineage.json` and
    are allowed to be absent.
    """
    labels = {path.stem for path in paths}
    days = sorted({label.split("_")[0] for label in labels})
    hours = sorted({label.split("_")[1] for label in labels})
    first = datetime.strptime(days[0], "%Y%m%d").replace(tzinfo=UTC).date()
    last = datetime.strptime(days[-1], "%Y%m%d").replace(tzinfo=UTC).date()
    expected = {
        f"{first + timedelta(days=offset):%Y%m%d}_{hour}"
        for offset in range((last - first).days + 1)
        for hour in hours
    }
    lineage_path = product_dir / "lineage.json"
    skipped: set[str] = set()
    if lineage_path.exists():
        note = json.loads(lineage_path.read_text())
        skipped = set(note.get("runs_skipped_missing_success_marker", [])) | set(
            note.get("runs_skipped_not_published", [])
        )
    missing = sorted(expected - labels - skipped)
    results["complete"] = missing


def _check_combined_and_grid(*, paths: list[Path], product_dir: Path, results: Results) -> None:
    """Check the combined file's rows, and the grid cells file, against the run files."""
    run_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
    combined_path = product_dir / "WeatherNext3.parquet"
    combined = pq.ParquetFile(combined_path).metadata.num_rows if combined_path.exists() else None
    results["combined_rows"] = [] if combined == run_rows else ["all"]
    metadata = pl.read_parquet_metadata(paths[0])
    grid_path = product_dir / "_grid_cells.parquet"
    try:
        grid_ok = _grid_matches(grid_path=grid_path, metadata=metadata)
    except Exception as error:  # noqa: BLE001  # a truncated or unreadable file is a finding
        results["grid_cells_match"] = [f"({type(error).__name__})"]
        return
    results["grid_cells_match"] = [] if grid_ok else ["all"]


def _grid_matches(*, grid_path: Path, metadata: dict[str, str]) -> bool:
    """Return whether `_grid_cells.parquet` has the recorded cell count and coordinate hash."""
    if not grid_path.exists():
        return False
    grid = pl.read_parquet(grid_path)
    latitudes = grid.filter(pl.col("lon_index") == 0).sort("lat_index")["latitude"]
    longitudes = grid.filter(pl.col("lat_index") == 0).sort("lon_index")["longitude"]
    digest = hashlib.sha256()
    digest.update(latitudes.to_numpy().tobytes())
    digest.update(longitudes.to_numpy().tobytes())
    return str(grid.height) == metadata.get("cell_count") and digest.hexdigest() == (
        metadata.get("cell_hash")
    )


def main() -> int:
    """Validate every run file in one product directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, help="Product directory name.")
    arguments = parser.parse_args()
    product_dir = WEATHER_DOWNLOADS_DIR / arguments.directory
    paths = sorted((product_dir / "_run_cache").glob("*.parquet"))
    results: Results = {check: [] for check in CHECKS}
    hashes: set[str | None] = set()
    for path in paths:
        hashes.add(_validate_run(path=path, results=results))
    if paths:
        _check_completeness(paths=paths, product_dir=product_dir, results=results)
        _check_combined_and_grid(paths=paths, product_dir=product_dir, results=results)
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
