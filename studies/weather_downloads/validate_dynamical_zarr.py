"""Validate the GFS/GEFS parquets that `fetch_dynamical_zarr.py` wrote.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/841>, following the
`data-validation` skill's checklist. It checks each month file on its own (so peak memory is one
month), then the checks that span months, and prints one PASS, FAIL, or SKIP line per check. A run
that raises is not evidence the data is right, and neither is a run that prints PASS: a check that
fails names the months it failed in. **The report never prints a row count, a cell count, or a
coordinate,** because those reveal the size of the private trial-area box.

Checks per month file: every `init_time` falls inside the file's month; runs are regularly spaced
(6 hours for GFS, 24 hours for GEFS), and a month that is neither the first nor the last has every
run of every day; the row count equals the product of the axis lengths; no duplicate key; no null;
`NaN` only in the two radiation fields and only at lead time 0, where those fields are all `NaN`;
each value inside a physical range; the lead-time axis starts at 0, ends at the model's last lead
time, and changes step width at the documented lead time and not later; shortwave radiation is near
zero for night-time valid hours, and positive at midday in April to September (which catches only a
gross shift such as 12 hours or a timezone error, and does not verify the exact hour convention).
On a `.partial` month, `NaN` in a value column only warns, because the newest run may still be
being written. Checks across months: the months are contiguous, every file carries the same
grid-cell hash, and the combined file's row count equals the sum of the files. Where a complete
`M.parquet` and a stale `M.partial.parquet` both exist, only the complete file is validated.

Run it with `uv run python studies/weather_downloads/validate_dynamical_zarr.py --directory
<directory under data/studies/weather>`, for example `--directory GEFS`.
"""

import argparse
import sys
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Final

import polars as pl
import pyarrow.parquet as pq
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
"""Averaged over a window ending at the valid time, so `NaN` at lead time 0."""

SHORTWAVE: Final[str] = AVERAGED_FIELDS[0]

LEAD_AXIS_HOURS: Final[dict[str, tuple[int, int, int, int]]] = {
    "GFS": (1, 3, 120, 384),
    "GEFS": (3, 6, 240, 840),
}
"""Per model: the fine step, the coarse step, the last lead time (hours) on the fine step, and the
last lead time on the axis."""

RUN_SPACING_HOURS: Final[dict[str, int]] = {"GFS": 6, "GEFS": 24}
"""Per model: the gap between successive `init_time`s (4 runs a day for GFS, 1 for GEFS)."""

NIGHT_HOURS: Final[tuple[int, ...]] = (1, 2, 3)
"""UTC valid hours whose averaging window ends before sunrise everywhere in Great Britain."""

NIGHT_MEDIAN_LIMIT_W_M2: Final[float] = 5.0
MIDDAY_MEAN_MINIMUM_W_M2: Final[float] = 100.0
SUMMER_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)

Results = dict[str, list[str]]
"""Check name mapped to the months it failed in (empty if the check passed)."""

_SKIPPED: Final[str] = "skipped"


def _fail(*, results: Results, check: str, month: str) -> None:
    """Record that `check` failed in `month`."""
    results.setdefault(check, []).append(month)


def _month_start(month: str) -> datetime:
    """Return midnight UTC on the first day of `month` (`YYYY-MM`)."""
    return datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC)


def _check_runs(
    *, frame: pl.DataFrame, month: str, label: str, edge: bool, results: Results
) -> None:
    """Check that the `init_time`s sit in `month`, are regularly spaced, and are all present."""
    start = _month_start(month)
    end = start.replace(year=start.year + (start.month == 12), month=start.month % 12 + 1)
    inits = frame["init_time"].unique().sort()
    outside = inits.filter(
        (inits < start.replace(tzinfo=None)) | (inits >= end.replace(tzinfo=None))
    )
    if outside.len():
        _fail(results=results, check="init_in_month", month=month)
    spacing = RUN_SPACING_HOURS[label]
    if set(inits.diff().drop_nulls().dt.total_hours().to_list()) - {spacing}:
        _fail(results=results, check="runs_regular", month=month)
    if not edge and inits.len() != (end - start).days * 24 // spacing:
        _fail(results=results, check="runs_full_month", month=month)


def _check_values(*, frame: pl.DataFrame, month: str, results: Results, partial: bool) -> None:
    """Check nulls, `NaN`s, and physical ranges of every value column."""
    # The newest run of a `.partial` month may still be being written, so `NaN` there only warns.
    nan_check = "nan_partial_month" if partial else "nan"
    for variable in VARIABLES:
        if frame[variable].null_count():
            _fail(results=results, check="nulls", month=month)
        nan_rows = frame.filter(pl.col(variable).is_nan())
        if variable in AVERAGED_FIELDS:
            if nan_rows.filter(pl.col("lead_time") > pl.duration(hours=0)).height:
                _fail(results=results, check=nan_check, month=month)
            at_zero = frame.filter(pl.col("lead_time") == pl.duration(hours=0))
            if not at_zero.select(pl.col(variable).is_nan().all()).item():
                _fail(results=results, check="radiation_nan_at_lead_0", month=month)
        elif nan_rows.height:
            _fail(results=results, check=nan_check, month=month)
        low, high = RANGES[variable]
        observed = frame.filter(pl.col(variable).is_finite()).select(
            low=pl.col(variable).min(), high=pl.col(variable).max()
        )
        observed_low, observed_high = observed.row(0)
        if observed_low is not None and (observed_low < low or observed_high > high):
            _fail(results=results, check="ranges", month=month)


def _check_lead_axis(*, frame: pl.DataFrame, month: str, label: str, results: Results) -> None:
    """Check the lead-time axis starts at 0, ends where documented, and changes step on time."""
    fine, coarse, change_hour, last_lead = LEAD_AXIS_HOURS[label]
    leads = frame["lead_time"].unique().sort().dt.total_hours().to_list()
    steps_ok = all(b - a == (fine if b <= change_hour else coarse) for a, b in pairwise(leads))
    if leads[0] != 0 or leads[-1] != last_lead or not steps_ok:
        _fail(results=results, check="lead_axis", month=month)


def _check_diurnal(*, frame: pl.DataFrame, month: str, results: Results) -> None:
    """Check shortwave radiation is near zero at night and positive at summer midday."""
    valid = frame.select(
        valid_time=pl.col("init_time") + pl.col("lead_time"), value=pl.col(SHORTWAVE)
    ).filter(pl.col("value").is_not_nan())
    hour = pl.col("valid_time").dt.hour()
    night = valid.filter(hour.is_in(NIGHT_HOURS)).select(pl.col("value").median()).item()
    midday = (
        valid.filter((hour == 12) & pl.col("valid_time").dt.month().is_in(SUMMER_MONTHS))
        .select(pl.col("value").mean())
        .item()
    )
    if night is None:
        results.setdefault("night", []).append(_SKIPPED)
    elif night >= NIGHT_MEDIAN_LIMIT_W_M2:
        _fail(results=results, check="night", month=month)
    if midday is None:
        results.setdefault("midday", []).append(_SKIPPED)
    elif midday <= MIDDAY_MEAN_MINIMUM_W_M2:
        _fail(results=results, check="midday", month=month)


def _validate_month(*, path: Path, month: str, label: str, edge: bool, results: Results) -> None:
    """Run every per-month check on one month file and record failures in `results`."""
    frame = pl.read_parquet(path)
    key = [
        c
        for c in ("init_time", "ensemble_member", "lead_time", "lat_index", "lon_index")
        if c in frame.columns
    ]
    if frame.select(pl.struct(key).n_unique()).item() != frame.height:
        _fail(results=results, check="keys", month=month)
    expected_rows = 1
    for column in key:
        expected_rows *= frame[column].n_unique()
    if expected_rows != frame.height:
        _fail(results=results, check="dense", month=month)
    _check_runs(frame=frame, month=month, label=label, edge=edge, results=results)
    _check_values(frame=frame, month=month, results=results, partial=".partial." in path.name)
    _check_lead_axis(frame=frame, month=month, label=label, results=results)
    _check_diurnal(frame=frame, month=month, results=results)


def main() -> int:
    """Validate every month file and the combined file in one product directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, help="Product directory name.")
    arguments = parser.parse_args()
    product_dir = WEATHER_DOWNLOADS_DIR / arguments.directory
    label = next(name for name in DATASETS.values() if arguments.directory.startswith(name))
    by_month: dict[str, Path] = {}
    # A complete `M.parquet` wins over a stale `M.partial.parquet` of the same month.
    for path in sorted((product_dir / "_month_cache").glob("*.parquet"), reverse=True):
        by_month[path.name.split(".")[0]] = path
    by_month = dict(sorted(by_month.items()))
    months = sorted(by_month)
    results: Results = {}
    row_sum = 0
    hashes: set[str | None] = set()
    for index, month in enumerate(months):
        edge = index in (0, len(months) - 1)
        _validate_month(path=by_month[month], month=month, label=label, edge=edge, results=results)
        row_sum += pq.ParquetFile(by_month[month]).metadata.num_rows
        hashes.add(pl.read_parquet_metadata(by_month[month]).get("cell_hash"))

    stamps = [_month_start(month) for month in months]
    expected = _month_range(stamps[0], stamps[-1]) if months else []
    results["months_contiguous"] = [] if expected == months else ["all"]
    results["cell_hash_same"] = [] if len(hashes) == 1 and None not in hashes else ["all"]
    combined = pq.ParquetFile(product_dir / f"{label}.parquet").metadata.num_rows
    results["combined_rows"] = [] if combined == row_sum else ["all"]

    failed = 0
    for check, failures in results.items():
        real = sorted({m for m in failures if m != _SKIPPED})
        if real and check == "nan_partial_month":
            print(f"WARN {check}: {', '.join(real)}")
        elif real:
            failed += 1
            print(f"FAIL {check}: {', '.join(real)}")
        elif failures:
            print(f"SKIP {check}: no qualifying rows in some months")
        else:
            print(f"PASS {check}")
    print("validation passed" if not failed else f"{failed} checks failed")
    return 1 if failed else 0


def _month_range(first: datetime, last: datetime) -> list[str]:
    """Return every `YYYY-MM` label from `first` to `last` inclusive."""
    count = (last.year - first.year) * 12 + last.month - first.month + 1
    labels = []
    for offset in range(count):
        year, month_index = divmod(first.year * 12 + first.month - 1 + offset, 12)
        labels.append(f"{year}-{month_index + 1:02d}")
    return labels


if __name__ == "__main__":
    sys.exit(main())
