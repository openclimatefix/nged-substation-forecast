"""Validate the ERA5 solar-variable tables that `fetch_era5_solar_variables.py` wrote.

One-off throwaway script, following the `data-validation` skill. It reads the folder's
`manifest.json` and each variable's parquet, runs the checks below on every variable, writes
`validation_<variable>.json` next to the parquet, and prints one line per check. It exits non-zero
if any check is a `FAIL`. Run it on the pilot folder after the pilot, on the first finished
variable, and again once the fetch completes.

**The checks, per variable:**

- `rows_and_keys`: the schema, the exact row count (20 cells times the hours of the span), no
  duplicate (time, cell), the 20 public cells, and a complete hourly axis in every cell.
- `physical_range`: every finite value lies within the limits in `era5_solar_variables.VARIABLES`,
  which come from physics and not from the data. The real CDS netCDF files hold unpacked `float32`
  values compressed with zlib, so the limits are widened only by the `float32` rounding of the
  largest value.
- `hour_profile`: the mean absolute hour-to-hour change by UTC hour of day, all 24 hours, with the
  seam hours marked `*` and flagged hours `!`. A seam hour (06, 07, 09, 10, 18, 19, 21, and 22 UTC
  for cloud fields, 09, 10, 21, and 22 UTC for analysed fields, 07 and 19 UTC for accumulations)
  whose value is more than 3 times the baseline is a `FLAG`: a step that may be a change of forecast
  run or analysis window. A flag does not fail the run, because a step can be real and the reader
  has to look at the profile.
- `missing_values`: any NaN in a variable that should have none is a `FAIL`. For `cbh` and `cin` the
  NaN share is reported, split by total cloud cover below and at or above 0.05.
- `accumulation`: for accumulations only. A solar accumulation that falls from one hour to the next
  at fewer than 3 hours of the day looks like a running total and fails. A clear-sky radiation or
  ultraviolet accumulation that is positive at 00, 01, or 02 UTC, when the sun is down all year at
  these latitudes, fails.
- `ssrd_against_held_copy`: with `--check-ssrd-month YYYY-MM`, one month of `ssrd` is requested from
  CDS (the one request this script can make) and compared with the held copy in `beam_diffuse/`.
  The two should agree within the `float32` rounding of the largest value.

**Row and cell counts reveal nothing private**, because the cells are the public ERA5 box, so the
script prints them.

Run it with `uv run --with netCDF4 python
studies/weather_downloads/validate_era5_solar_variables.py`, adding `--dir <folder>` to validate
`_pilot/` instead of the full folder, `--variables tcc lcc` to restrict it, and
`--check-ssrd-month 2025-06` for the `ssrd` comparison (this one needs `--with cdsapi` as well).
"""

import argparse
import calendar
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Literal

import polars as pl
from era5_solar_variables import (
    CDS_DATASET,
    CELL_COUNT,
    COST_FIELDS_PER_VARIABLE_HOUR,
    SEAM_HOURS,
    SEAM_RULES,
    VARIABLES_BY_NAME,
    Variable,
    expected_hours,
    expected_rows,
    hour_profile,
    nan_share_by_tcc,
    read_archive,
    run_chunks,
    step_flags,
)
from studies.era5_grid import AREA, GRID_LATITUDES, GRID_LONGITUDES
from studies.pv_dataset import ERA5_DIR
from studies.sources import ERA5_SOLAR_VARIABLES_DIR, SCRATCH_DIR

StatusType = Literal["PASS", "FAIL", "FLAG", "INFO"]
CheckResult = tuple[StatusType, dict[str, Any]]

FLOAT32_RELATIVE_TOLERANCE: Final[float] = 1e-6
"""Eight times `float32`'s machine epsilon (1.2e-7): the rounding allowed on the largest value."""

FALLING_HOUR_SHARE: Final[float] = 0.25
"""An hour of day counts as one where the value falls when it is lower than the hour before on more
than this share of days."""

MIN_FALLING_HOURS: Final[int] = 3
"""A genuine hourly solar total falls on most days at every hour after the afternoon peak, which is
at least 3 hours of day even in December. A running total falls only when it resets, so a solar
accumulation with fewer falling hours of day than this, away from the seam hours, looks like a
running total."""

NIGHT_HOURS_UTC: Final[tuple[int, ...]] = (0, 1, 2)
"""Stamps whose hour ends before 03 UTC, which is dark all year at 53 degrees north."""
NIGHT_ZERO_VARIABLES: Final[tuple[str, ...]] = ("ssrdc", "cdir", "uvb")
"""Accumulations of radiation that needs the sun, so they are zero at night. `strd` is not."""

HELD_SSRD_CDS_NAME: Final[str] = "surface_solar_radiation_downwards"
KEY: Final[list[str]] = ["time", "latitude", "longitude"]


def _scalar(value: Any) -> float | None:
    """Return a Polars reduction as a float, or `None` if it is missing."""
    return None if value is None else float(value)


def float32_tolerance(*, values: pl.Series) -> float:
    """Return the `float32` rounding allowed on `values`: a relative tolerance on the largest."""
    finite = values.filter(values.is_finite())
    largest = _scalar(finite.abs().max())
    return 0.0 if largest is None else FLOAT32_RELATIVE_TOLERANCE * largest


def check_rows_and_keys(
    *, frame: pl.DataFrame, first_hour: datetime, last_hour: datetime
) -> CheckResult:
    """Check the schema, row count, duplicate keys, cells, and hourly axis."""
    wanted_rows = expected_rows(first_hour=first_hour, last_hour=last_hour)
    wanted_hours = expected_hours(first_hour=first_hour, last_hour=last_hour)
    schema = {name: str(dtype) for name, dtype in frame.schema.items()}
    expected_schema = {
        "time": "Datetime(time_unit='us', time_zone='UTC')",
        "latitude": "Float64",
        "longitude": "Float64",
        "value": "Float32",
        "expver": "String",
    }
    duplicates = frame.height - frame.select(KEY).unique().height
    cells = set(frame.select("latitude", "longitude").unique().iter_rows())
    public_cells = {(lat, lon) for lat in GRID_LATITUDES for lon in GRID_LONGITUDES}
    per_cell = frame.group_by("latitude", "longitude").agg(
        pl.col("time").n_unique().alias("hours"),
        pl.col("time").min().alias("first"),
        pl.col("time").max().alias("last"),
    )
    axis_ok = (
        set(per_cell["hours"]) == {wanted_hours}
        and set(per_cell["first"]) == {first_hour}
        and set(per_cell["last"]) == {last_hour}
    )
    ok = (
        schema == expected_schema
        and frame.height == wanted_rows
        and duplicates == 0
        and cells == public_cells
        and axis_ok
    )
    return ("PASS" if ok else "FAIL"), {
        "schema": schema,
        "rows": frame.height,
        "expected_rows": wanted_rows,
        "cells": len(cells),
        "expected_cells": CELL_COUNT,
        "duplicate_keys": duplicates,
        "complete_hourly_axis_in_every_cell": axis_ok,
    }


def check_physical_range(*, frame: pl.DataFrame, variable: Variable) -> CheckResult:
    """Check finite values against the variable's physical limits, allowing float32 rounding."""
    values = frame["value"]
    finite = values.filter(values.is_finite())
    slack = float32_tolerance(values=values)
    below = 0 if variable.minimum is None else int((finite < variable.minimum - slack).sum())
    above = 0 if variable.maximum is None else int((finite > variable.maximum + slack).sum())
    return ("PASS" if below + above == 0 else "FAIL"), {
        "minimum_observed": _scalar(finite.min()),
        "maximum_observed": _scalar(finite.max()),
        "limits": [variable.minimum, variable.maximum],
        "tolerance": slack,
        "rows_below_minimum": below,
        "rows_above_maximum": above,
    }


def check_hour_profile(*, frame: pl.DataFrame, variable: Variable) -> CheckResult:
    """Report the whole 24-hour profile, mark the seam hours, and flag a step at any of them."""
    profile = hour_profile(frame=frame)
    seams = SEAM_HOURS[variable.seam_family]
    flagged = step_flags(profile=profile, seam_hours=seams, rule=SEAM_RULES[variable.seam_family])
    return ("FLAG" if flagged else "PASS"), {
        "mean_abs_hourly_change_by_utc_hour": [
            {"hour": hour, "value": value, "seam": hour in seams, "flagged": hour in flagged}
            for hour, value in enumerate(profile)
        ],
        "seam_family": variable.seam_family,
        "seam_hours_checked": list(seams),
        "rule": SEAM_RULES[variable.seam_family],
        "flagged_hours": flagged,
    }


def check_missing_values(
    *, frame: pl.DataFrame, variable: Variable, tcc: pl.DataFrame | None
) -> CheckResult:
    """Fail on NaN where none is allowed; split the NaN share by cloud cover where it is a value."""
    nan_rows = int((frame["value"].is_nan() | frame["value"].is_null()).sum())
    details: dict[str, Any] = {"nan_rows": nan_rows, "rows": frame.height}
    if tcc is not None:
        details["nan_share_by_tcc"] = nan_share_by_tcc(frame=frame, tcc=tcc)
    else:
        details["nan_share_by_tcc"] = "tcc not available in this folder"
    if variable.nan_means_no_cloud:
        return "INFO", details
    return ("PASS" if nan_rows == 0 else "FAIL"), details


def _decrease_share_by_hour(*, frame: pl.DataFrame, tolerance: float) -> list[float | None]:
    """Return, by UTC hour of the later stamp, the share of hour pairs where the value falls."""
    cell = ["latitude", "longitude"]
    paired = (
        frame.sort(*cell, "time")
        .with_columns(
            previous_time=pl.col("time").shift(1).over(cell),
            previous_value=pl.col("value").shift(1).over(cell),
        )
        .filter(
            (pl.col("time") - pl.col("previous_time") == pl.duration(hours=1))
            & pl.col("value").is_finite()
            & pl.col("previous_value").is_finite()
        )
    )
    by_hour = dict(
        paired.group_by(pl.col("time").dt.hour().alias("hour"))
        .agg((pl.col("value") < pl.col("previous_value") - tolerance).mean())
        .iter_rows()
    )
    return [by_hour.get(hour) for hour in range(24)]


def check_accumulation(*, frame: pl.DataFrame, variable: Variable) -> CheckResult:
    """Check an accumulation is an hourly total and not a running total, and is zero at night."""
    slack = float32_tolerance(values=frame["value"])
    decreases = _decrease_share_by_hour(frame=frame, tolerance=slack)
    seams = set(SEAM_HOURS[variable.seam_family])
    falling_hours = [
        hour
        for hour, share in enumerate(decreases)
        if hour not in seams and share is not None and share > FALLING_HOUR_SHARE
    ]
    looks_running = (
        variable.seam_family == "solar_accumulation" and len(falling_hours) < MIN_FALLING_HOURS
    )
    details: dict[str, Any] = {
        "decrease_share_by_utc_hour": decreases,
        "falling_hours_of_day_away_from_seams": falling_hours,
        "looks_like_running_total": looks_running,
    }
    night_ok = True
    if variable.short_name in NIGHT_ZERO_VARIABLES:
        night = frame.filter(pl.col("time").dt.hour().is_in(NIGHT_HOURS_UTC))["value"]
        night_max = _scalar(night.max())
        night_ok = night_max is None or night_max <= max(2 * slack, 1.0)
        details["maximum_at_night_hours"] = night_max
    return ("PASS" if night_ok and not looks_running else "FAIL"), details


@dataclass(frozen=True)
class SsrdCheckChunk:
    """A one-month request for `ssrd`, to compare with the held copy."""

    year: int
    month: int
    n_hours: int

    @property
    def chunk_id(self) -> str:
        """Return a filename-safe id."""
        return f"ssrd_check_{self.year}_{self.month:02d}"

    @property
    def variables(self) -> tuple[str, ...]:
        """Return the one variable."""
        return ("ssrd",)

    @property
    def variable_hours(self) -> int:
        """Return the hours requested."""
        return self.n_hours

    @property
    def cost_fields(self) -> int:
        """Return the store's field count."""
        return self.n_hours * COST_FIELDS_PER_VARIABLE_HOUR

    def request(self) -> dict[str, object]:
        """Return the CDS request body over the public box."""
        return {
            "product_type": ["reanalysis"],
            "variable": [HELD_SSRD_CDS_NAME],
            "year": [str(self.year)],
            "month": [f"{self.month:02d}"],
            "day": [f"{day:02d}" for day in range(1, 32)],
            "time": [f"{hour:02d}:00" for hour in range(24)],
            "area": list(AREA),
            "data_format": "netcdf",
            "download_format": "zip",
        }


def _download_ssrd(chunk: SsrdCheckChunk, destination: Path) -> None:
    import cdsapi  # ty: ignore[unresolved-import]  # `uv run --with cdsapi` only

    cdsapi.Client(quiet=True, progress=False).retrieve(
        CDS_DATASET, chunk.request(), str(destination)
    )


def _held_archive_for(*, year: int, month: int) -> Path | None:
    """Return the held half-year archive that covers the month, or `None`."""
    for path in sorted(ERA5_DIR.glob("era5_*.zip")):
        parts = path.stem.split("_")
        if len(parts) == 4 and int(parts[1]) == year and int(parts[2]) <= month <= int(parts[3]):
            return path
    return None


def check_ssrd_against_held_copy(*, month: str, output_dir: Path) -> CheckResult:
    """Fetch one month of `ssrd` and compare it with the held copy.

    Args:
        month: The month as `YYYY-MM`.
        output_dir: The folder whose `_checks/` receives the fresh request.

    Returns:
        `PASS` if the two agree within the sum of the two files' float32 tolerances, `INFO` if
        there is no held copy to compare with, and `FAIL` otherwise.
    """
    year, number = (int(part) for part in month.split("-"))
    held_path = _held_archive_for(year=year, month=number)
    if held_path is None:
        return "INFO", {"reason": f"no held copy covers {month} in {ERA5_DIR.name}/"}
    first = datetime(year, number, 1, tzinfo=UTC)
    n_hours = calendar.monthrange(year, number)[1] * 24
    last = first + timedelta(hours=n_hours - 1)
    chunk = SsrdCheckChunk(year=year, month=number, n_hours=n_hours)
    run_chunks(chunks=[chunk], chunk_dir=output_dir / "_checks", download=_download_ssrd, log=print)
    scratch = SCRATCH_DIR / "era5_solar_variables_check"
    fresh_all = read_archive(
        path=output_dir / "_checks" / f"{chunk.chunk_id}.zip", scratch_dir=scratch, has_expver=False
    ).frames["ssrd"]
    held_all = read_archive(path=held_path, scratch_dir=scratch, has_expver=False).frames["ssrd"]
    in_month = pl.col("time").is_between(first, last)
    fresh, held = fresh_all.filter(in_month), held_all.filter(in_month)
    joined = held.join(fresh, on=KEY, how="inner", suffix="_fresh")
    difference = (joined["value"] - joined["value_fresh"]).abs()
    tolerance = float32_tolerance(values=held_all["value"]) + float32_tolerance(
        values=fresh_all["value"]
    )
    maximum = _scalar(difference.max())
    ok = (
        held.height == fresh.height == joined.height
        and maximum is not None
        and maximum <= tolerance
    )
    return ("PASS" if ok else "FAIL"), {
        "month": month,
        "held_rows": held.height,
        "fresh_rows": fresh.height,
        "joined_rows": joined.height,
        "max_abs_difference_j_m2": maximum,
        "tolerance_j_m2": tolerance,
        "share_identical": _scalar((difference == 0).mean()),
    }


def validate_variable(
    *,
    variable: Variable,
    frame: pl.DataFrame,
    tcc: pl.DataFrame | None,
    first_hour: datetime,
    last_hour: datetime,
) -> dict[str, CheckResult]:
    """Run every check that applies to one variable."""
    checks = {
        "rows_and_keys": check_rows_and_keys(
            frame=frame, first_hour=first_hour, last_hour=last_hour
        ),
        "physical_range": check_physical_range(frame=frame, variable=variable),
        "hour_profile": check_hour_profile(frame=frame, variable=variable),
        "missing_values": check_missing_values(frame=frame, variable=variable, tcc=tcc),
    }
    if variable.kind == "accumulation":
        checks["accumulation"] = check_accumulation(frame=frame, variable=variable)
    return checks


def _profile_cell(entry: dict[str, Any]) -> str:
    """Return `hour:value`, with `*` after a seam hour and `!` after a flagged one."""
    value = entry["value"]
    text = "none" if value is None else f"{value:.3g}"
    return (
        f"{entry['hour']:02d}:{text}{'*' if entry['seam'] else ''}{'!' if entry['flagged'] else ''}"
    )


def write_record(*, output_dir: Path, name: str, checks: dict[str, CheckResult]) -> bool:
    """Write `validation_<name>.json`, print one line per check, and return whether none failed."""
    failed = any(status == "FAIL" for status, _ in checks.values())
    record = {
        "variable": name,
        "validated_at_utc": datetime.now(UTC).isoformat(),
        "status": "FAIL" if failed else "PASS",
        "checks": {key: {"status": status, **details} for key, (status, details) in checks.items()},
    }
    (output_dir / f"validation_{name}.json").write_text(json.dumps(record, indent=2, default=str))
    for key, (status, details) in checks.items():
        print(f"{status:4s} {name} {key}")
        if key == "hour_profile":
            print(
                "     "
                + " ".join(
                    _profile_cell(entry) for entry in details["mean_abs_hourly_change_by_utc_hour"]
                )
            )
    return not failed


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the chosen variables and return 1 if any check failed."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dir", type=Path, default=ERA5_SOLAR_VARIABLES_DIR)
    parser.add_argument("--variables", nargs="+", help="short names; default every one written")
    parser.add_argument("--check-ssrd-month", help="YYYY-MM; fetches one month of ssrd from CDS")
    args = parser.parse_args(argv)

    manifest = json.loads((args.dir / "manifest.json").read_text())
    first_hour = datetime.fromisoformat(manifest["scope"]["first_hour"])
    last_hour = datetime.fromisoformat(manifest["scope"]["last_hour"])
    names = args.variables or sorted(manifest["variables"])
    tcc_path = args.dir / "tcc.parquet"
    tcc = pl.read_parquet(tcc_path) if tcc_path.exists() else None

    all_ok = True
    for name in names:
        frame = pl.read_parquet(args.dir / f"{name}.parquet")
        checks = validate_variable(
            variable=VARIABLES_BY_NAME[name],
            frame=frame,
            tcc=tcc,
            first_hour=first_hour,
            last_hour=last_hour,
        )
        all_ok &= write_record(output_dir=args.dir, name=name, checks=checks)
    if args.check_ssrd_month:
        result = {
            "ssrd_against_held_copy": check_ssrd_against_held_copy(
                month=args.check_ssrd_month, output_dir=args.dir
            )
        }
        all_ok &= write_record(output_dir=args.dir, name="ssrd", checks=result)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
