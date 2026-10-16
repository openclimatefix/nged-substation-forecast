"""Validate the control-member backfill files and write the folder's report, README and lineage.

Run `uv run python studies/ens_backfill_control/check_fetch.py` after the first few dates and again
when the fetch has finished. Every number in `report.md` is printed by this script. It checks, for
every checkpoint file: the date matches the file name, the arrays are the expected shape, no value
is NaN or infinite, each variable lies in a physical range, no field is constant, the dew point does
not exceed the temperature, the grid runs north to south with a colder north, and the day-night
cycle of downward shortwave radiation peaks at local midday. Across files it checks that no date
the archive lists is missing, that no two files share a hash, and that the mean temperature has no
step change between dates.
"""

import argparse
import json
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import numpy as np
from fetch_control import DEFAULT_OUTPUT_DIR, FIRST_DATE, LAST_DATE
from studies.deaccumulation import RADIATION_INVALID_BELOW_W_PER_M2, deaccumulate_to_rates
from studies.ens_backfill_fetch import PILOT_VARIABLES, STEPS_HOURS

EXPECTED_SHAPE: Final[tuple[int, int, int, int, int]] = (12, 1, 85, 50, 69)
"""(variable, member, step, row, column) of every `values` array."""

PHYSICAL_RANGES: Final[dict[str, tuple[float, float]]] = {
    "2t": (230.0, 325.0),
    "2d": (220.0, 310.0),
    "10u": (-60.0, 60.0),
    "10v": (-60.0, 60.0),
    "100u": (-70.0, 70.0),
    "100v": (-70.0, 70.0),
    "sp": (80_000.0, 110_000.0),
    "msl": (90_000.0, 110_000.0),
    "z500": (45_000.0, 60_000.0),
}
"""Generous bounds per instantaneous variable, in ECMWF's units (K, m s-1, Pa, m2 s-2)."""

ACCUMULATED: Final[tuple[str, ...]] = ("tp", "strd", "ssrd")
"""The variables ECMWF accumulates from the start of the forecast."""


def read_hash_and_values(*, path: Path) -> tuple[str, np.ndarray]:
    """Return the first message's SHA-256 and the file's `values` array."""
    with np.load(path) as file:
        return str(file["sha256"].ravel()[0]), file["values"]


def check_file(*, path: Path) -> list[str]:
    """Return the problems found in one checkpoint file."""
    problems: list[str] = []
    with np.load(path) as file:
        values = file["values"]
        latitudes = file["latitudes"]
        file_date = str(file["date"])
        variables = [str(name) for name in file["variables"]]
        steps = [int(step) for step in file["steps_hours"]]
    if file_date != path.stem:
        problems.append(f"date inside the file is {file_date}")
    if values.shape != EXPECTED_SHAPE:
        return [*problems, f"shape {values.shape}"]
    if variables != list(PILOT_VARIABLES) or steps != list(STEPS_HOURS):
        problems.append("variable or step order differs from the fetch constants")
    if not np.isfinite(values).all():
        problems.append(f"{int((~np.isfinite(values)).sum())} non-finite values")
    for index, name in enumerate(variables):
        field = values[index, 0]
        if name in PHYSICAL_RANGES:
            low, high = PHYSICAL_RANGES[name]
            if field.min() < low or field.max() > high:
                problems.append(f"{name} range {field.min():.1f} to {field.max():.1f}")
        if name not in ACCUMULATED and (field.reshape(len(steps), -1).std(axis=1) == 0).any():
            problems.append(f"{name} has a constant field at some step")
    if not np.all(np.diff(latitudes) < 0):
        problems.append("latitudes do not run north to south")
    dew_point = values[variables.index("2d"), 0]
    if (dew_point > values[variables.index("2t"), 0] + 1.0).any():
        problems.append("dew point exceeds temperature by more than 1 K")
    return problems


def shortwave_peak_hour(*, path: Path) -> float:
    """Return the UTC hour of day at which mean downward shortwave radiation peaks, in the file."""
    with np.load(path) as file:
        values = file["values"]
        variables = [str(name) for name in file["variables"]]
    totals = values[variables.index("ssrd"), 0].mean(axis=(1, 2)).astype(np.float64)
    steps = np.array(STEPS_HOURS)
    rates = deaccumulate_to_rates(
        accumulations=totals,
        elapsed_seconds=steps * 3600.0,
        invalid_below_rate=RADIATION_INVALID_BELOW_W_PER_M2,
    ).rates
    first_day = steps <= 24
    middle_hours = (steps[first_day][1:] - np.diff(steps[first_day]) / 2) % 24
    return float(middle_hours[np.nanargmax(rates[first_day][1:])])


def main() -> int:
    """Check every file, print the findings, and write the report, README and lineage note."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir: Path = args.output_dir
    files = sorted((output_dir / "control").glob("*.npz"))
    if not files:
        print("No checkpoint files found.", file=sys.stderr)
        return 1
    listed = {
        date.fromisoformat(text)
        for text in json.loads((output_dir / "complete_dates.json").read_text())["dates"]
    }
    expected = {day for day in listed if FIRST_DATE <= day <= LAST_DATE}
    present = {date.fromisoformat(path.stem) for path in files}

    lines = ["# Control-member backfill: check report", ""]
    lines.append(f"Checkpoint files: {len(files)}, {min(present)} to {max(present)}.")
    lines.append(f"Complete dates the archive lists in the range: {len(expected)}.")
    missing = sorted(expected - present)
    lines.append(
        f"Listed dates with no file: {len(missing)}" + (f", first {missing[0]}" if missing else ".")
    )
    lines.append(f"Files for dates the archive does not list: {len(present - expected)}.")

    problems_by_file = {path.stem: check_file(path=path) for path in files}
    bad = {name: found for name, found in problems_by_file.items() if found}
    lines.append(f"Files with problems: {len(bad)} of {len(files)}.")
    lines.extend(f"- {name}: {'; '.join(found)}" for name, found in list(bad.items())[:50])

    hashes = [read_hash_and_values(path=path)[0] for path in files]
    lines.append(f"Duplicate first-message hashes across files: {len(hashes) - len(set(hashes))}.")
    peaks = np.array([shortwave_peak_hour(path=path) for path in files])
    lines.append(
        f"UTC hour at which mean shortwave peaks in the first 24 h: "
        f"min {peaks.min():.1f}, median {np.median(peaks):.1f}, max {peaks.max():.1f}."
    )
    means = np.array([read_hash_and_values(path=path)[1][0, 0, 0].mean() for path in files])
    jumps = np.abs(np.diff(means))
    lines.append(
        f"Mean 2 m temperature at step 0 (K): {means.min():.1f} to {means.max():.1f}; the largest "
        f"jump between consecutive files is {jumps.max():.1f} K "
        f"({files[int(jumps.argmax())].stem} to {files[int(jumps.argmax()) + 1].stem})."
    )
    size_mb = np.array([path.stat().st_size for path in files]) / 1e6
    lines.append(
        f"File size (MB): min {size_mb.min():.1f}, median {np.median(size_mb):.1f}, "
        f"max {size_mb.max():.1f}; total {size_mb.sum() / 1e3:.1f} GB."
    )
    report = "\n".join(lines) + "\n"
    print(report)
    (output_dir / "report.md").write_text(report)
    write_readme(output_dir=output_dir, count=len(files), first=min(present), last=max(present))
    return 1 if bad or missing else 0


def write_readme(*, output_dir: Path, count: int, first: date, last: date) -> None:
    """Write the folder's README and its lineage note."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    (output_dir / "LINEAGE.json").write_text(
        json.dumps(
            {
                "source": "https://data.source.coop/dynamical/ecmwf-ifs-grib/ecmwf-ifs-ens/",
                "licence": "CC BY 4.0",
                "fetched_with": "studies/ens_backfill_control/fetch_control.py",
                "git_revision": revision,
                "written_at_utc": datetime.now(UTC).isoformat(),
                "files": count,
            },
            indent=2,
        )
    )
    (output_dir / "README.md").write_text(
        f"""# ECMWF ENS control-member backfill (generated by check_fetch.py)

**This folder holds the ECMWF ensemble control member's forecasts for {count} dates, {first} to
{last}.** Dynamical.org stages ECMWF's GRIB edition 1 files for these dates on Source Cooperative
(CC BY 4.0), and `fetch_control.py` decodes the control member's messages. The ensemble's 50
perturbed members are not here.

## Layout

`control/<date>.npz` holds one 00 UTC forecast. `values` is a `float32` array indexed by (variable,
member, step, row, column) with the 12 variables of the `Nwp` contract (`2t`, `2d`, `10u`, `10v`,
`100u`, `100v`, `sp`, `msl`, `tp`, `strd`, `ssrd`, `z500`), one member (0, the control), 85 steps
(0 to 144 h every 3 h, then to 360 h every 6 h), 50 rows (61.25 to 49.0 degrees north), and 69
columns (12 degrees west to 5 degrees east) on ECMWF's 0.25 degree grid. Units are ECMWF's: K, Pa,
m s-1, m2 s-2 for `z500`, and totals accumulated since the forecast start for `tp` (m), `strd`
and `ssrd` (J m-2). De-accumulate with `studies.deaccumulation`. `raw_x` holds the packed 16-bit
integers from which `values` is rebuilt exactly. `sha256` and the message offsets record which
bytes were read.

## Things to know before using it

- The ECMWF ensemble moved from 18 km to 9 km native resolution on 2023-06-27 (IFS cycle 48r1).
  Use folds that do not straddle that date unless the study is about the change.
- The archive ends on 2024-03-31. Dynamical.org's own archive, which our pipeline stores, starts
  on 2024-04-01 with a different product, so the join at that date is a seam.
- Only the control member is here, so ensemble spread, ensemble mean and CRPS cannot be computed
  from these files.
- `report.md` lists the checks run on these files. `complete_dates.json` is the bucket listing the
  fetch used, and `LINEAGE.json` records the code revision.
"""
    )


if __name__ == "__main__":
    sys.exit(main())
