"""Validate the Icechunk store that `fetch_ukv_ceda.py` wrote.

One-off throwaway script for the forecast study. It opens `<product dir>/store` read-only and runs
the checks in `CHECK_NAMES`: the run spacing of the `init_time` axis, the counts of runs by status,
the value range of every variable, the pattern of NaN against the lead layout that each variable is
expected to have, the centroid of the daily shortwave curve against solar noon, the north-to-south
order of the rows, that the maximum gust is not below the instantaneous gust, and the list of gaps,
in which every run that CEDA lacks (status 3) appears.

**The checks read a sample of at most `--sample` runs, spread evenly across the archive**: the
value ranges and the shortwave cycle read complete and partial runs, and the NaN layout reads
complete runs only. The sample is small because every run holds 31 arrays of 55 steps. Pass
`--sample 0` to read every run. A range check never loosens to pass: a failure means the store or
the source is wrong.

**The script prints one PASS or FAIL line per check, and no cell count or coordinate,** because
those reveal the size and place of the private trial-area box. The measured numbers go only to
`validation.json` next to the store. The script exits non-zero when any check fails.

Run it with `uv run --with icechunk --with zarr python
studies/weather_downloads/validate_ukv_ceda.py --store-dir <product dir>`.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import zarr
from fetch_ukv_ceda import (
    CODE_VERSION,
    CYCLE_HOURS,
    FIELDS,
    N_STEPS,
    PLAIN_LAST_STEP,
    PRODUCT_NAME,
    SLOT_EPOCH,
    STATUS_COMPLETE,
    STATUS_MISSING,
    STATUS_NAMES,
    STATUS_PARTIAL,
    FieldSpec,
    UkvStore,
    _array,
)
from paths import WEATHER_DOWNLOADS_DIR

CHECK_NAMES: Final[tuple[str, ...]] = (
    "run_spacing",
    "status_counts",
    "value_ranges",
    "nan_layout",
    "shortwave_diurnal_cycle",
    "north_south_gradient",
    "gust_max_is_a_maximum",
    "gaps",
)

KELVIN: Final[float] = 273.15
VALUE_RANGES: Final[dict[str, tuple[float, float]]] = {
    "temperature_1p5m": (-40.0 + KELVIN, 50.0 + KELVIN),
    "temperature_0m": (-40.0 + KELVIN, 60.0 + KELVIN),
    "dew_point_1p5m": (-40.0 + KELVIN, 50.0 + KELVIN),
    "relative_humidity_1p5m": (0.0, 100.1),
    "visibility_1p5m": (0.0, 200000.0),
    "visibility_below_1km_probability": (0.0, 1.0),
    "precipitation_rate": (0.0, 0.1),
    "precipitation_amount": (0.0, 200.0),
    "param_0_1_230": (0.0, 100.0),
    "wind_speed_10m": (0.0, 80.0),
    "wind_direction_10m": (0.0, 360.0),
    "pressure_msl": (90000.0, 110000.0),
    "cloud_total": (0.0, 100.0),
    "cloud_low": (0.0, 100.0),
    "cloud_very_low": (0.0, 100.0),
    "cloud_medium": (0.0, 100.0),
    "cloud_high": (0.0, 100.0),
    "cloud_base_height": (0.0, 20000.0),
    "cloud_param_0_6_26": (0.0, 20000.0),
    "convective_cloud_top_height": (0.0, 20000.0),
    "snow_depth": (0.0, 10.0),
    "shortwave_down": (0.0, 1400.0),
    "longwave_down": (100.0, 600.0),
    "wind_speed_1000hpa": (0.0, 80.0),
    "wind_speed_925hpa": (0.0, 80.0),
    "wind_direction_1000hpa": (0.0, 360.0),
    "wind_direction_925hpa": (0.0, 360.0),
    "geopotential_height_1000hpa": (-500.0, 500.0),
    "geopotential_height_925hpa": (300.0, 1200.0),
    "gust_10m": (0.0, 100.0),
    "gust_10m_max": (0.0, 100.0),
}
"""The physical range of each variable as stored, in the variable's own units. Cloud is a percentage
because the source serves a percentage. Relative humidity reaches 100.1 because the source packs
values at 8 bits and serves up to 100.05 (measured in the 2026-09-20 00Z file). The range of
`param_0_1_230`, which has no known units, is a guess that a negative or huge value would break."""

NAN_ALLOWED: Final[frozenset[str]] = frozenset(
    {
        "cloud_base_height",
        "cloud_param_0_6_26",
        "convective_cloud_top_height",
        "wind_speed_1000hpa",
        "wind_direction_1000hpa",
        "geopotential_height_1000hpa",
        "wind_speed_925hpa",
        "wind_direction_925hpa",
        "geopotential_height_925hpa",
    }
)
"""Variables whose cells may be NaN at a served lead: a bitmap masks below-ground pressure levels,
cells with no cloud (cloud base is NaN under a clear sky), and cells with no convective cloud. The
layout check only demands that these variables are entirely NaN at the unserved leads."""

NIGHT_MEAN_MAX_W_M2: Final[float] = 2.0
NIGHT_HOUR_UTC: Final[int] = 0
SOLAR_NOON_TOLERANCE_HOURS: Final[float] = 0.6
"""How far the centroid of the daily shortwave curve may sit from solar noon. Weather that clouds
one half of a day moves the centroid by a few tenths of an hour, and a 1 hour lead offset moves it
by 1 hour."""
GUST_TOLERANCE: Final[float] = 2**-12
MAX_GUST_VIOLATION_FRACTION: Final[float] = 0.01


def expected_leads(spec: FieldSpec) -> set[int]:
    """The leads, in hours, at which a variable has data: its plain and `T54` files together."""
    leads: set[int] = set()
    for tag in spec.tags:
        leads.update(spec.expected_steps(tag=tag))
    return leads


def sample_slots(slots: np.ndarray, *, sample: int) -> np.ndarray:
    """Pick at most `sample` slots, spread evenly across `slots`; all of them if `sample` is 0."""
    if sample <= 0 or len(slots) <= sample:
        return slots
    positions = np.linspace(0, len(slots) - 1, sample).round().astype(int)
    return slots[positions]


def check_run_spacing(group: zarr.Group) -> tuple[bool, dict[str, Any]]:
    """Check that `init_time` is the fixed 6-hourly grid from the slot epoch, without a shift."""
    init_time = np.asarray(_array(group, "init_time")[:])
    expected = int(SLOT_EPOCH.timestamp()) + np.arange(len(init_time)) * CYCLE_HOURS * 3600
    return bool(np.array_equal(init_time, expected)), {"slots": len(init_time)}


def check_status_counts(statuses: np.ndarray, group: zarr.Group) -> tuple[bool, dict[str, Any]]:
    """Check that no run has an unknown status, and that the file counts agree with the status."""
    counts = {name: int((statuses == code).sum()) for code, name in STATUS_NAMES.items()}
    counts["never_archived"] = int((statuses == 0).sum())
    known = set(STATUS_NAMES) | {0}
    received = np.asarray(_array(group, "files_received")[:])
    expected = np.asarray(_array(group, "files_expected")[:])
    complete = statuses == STATUS_COMPLETE
    consistent = bool(np.all(received[complete] == expected[complete]))
    missing_is_empty = bool(np.all(received[statuses == STATUS_MISSING] == 0))
    ok = set(np.unique(statuses).tolist()) <= known and consistent and missing_is_empty
    versions = {str(v) for v in np.asarray(_array(group, "code_version")[:])[statuses != 0]}
    return ok, {"counts": counts, "code_versions": sorted(versions), "current": CODE_VERSION}


def read_sample(group: zarr.Group, spec: FieldSpec, slots: np.ndarray) -> np.ndarray:
    """Read one variable at `slots`, as a `(slot, step, cell)` array."""
    array = _array(group, spec.variable)
    return np.stack([np.asarray(array[int(slot)]) for slot in slots])


def check_value_ranges(group: zarr.Group, slots: np.ndarray) -> tuple[bool, dict[str, Any]]:
    """Check every variable's finite values against its physical range."""
    measured: dict[str, Any] = {}
    ok = True
    for spec in FIELDS:
        low, high = VALUE_RANGES[spec.variable]
        data = read_sample(group, spec, slots)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            measured[spec.variable] = "all NaN"
            ok = ok and spec.variable in NAN_ALLOWED
            continue
        observed = (float(finite.min()), float(finite.max()))
        measured[spec.variable] = {"min": observed[0], "max": observed[1], "range": [low, high]}
        if observed[0] < low or observed[1] > high:
            print(f"  {spec.variable}: outside its physical range")
            ok = False
    return ok, measured


def check_nan_layout(group: zarr.Group, slots: np.ndarray) -> tuple[bool, dict[str, Any]]:
    """Check that each variable is NaN at every unserved lead, and finite at every served lead."""
    ok = True
    bad: dict[str, str] = {}
    for spec in FIELDS:
        served = np.zeros(N_STEPS, dtype=bool)
        served[sorted(expected_leads(spec))] = True
        data = read_sample(group, spec, slots)
        present = np.isfinite(data)
        if present[:, ~served, :].any():
            bad[spec.variable] = "finite values at an unserved lead"
        if spec.variable not in NAN_ALLOWED and not present[:, served, :].all():
            bad[spec.variable] = bad.get(spec.variable, "") + " NaN at a served lead"
        elif spec.variable in NAN_ALLOWED and not present[:, served, :].any():
            bad[spec.variable] = "no finite value at any served lead"
    for variable, reason in bad.items():
        print(f"  {variable}: {reason.strip()}")
        ok = False
    return ok, {"variables_with_a_fault": sorted(bad)}


def equation_of_time_minutes(day_of_year: int) -> float:
    """The equation of time in minutes, to about 1 minute: apparent minus mean solar time."""
    angle = 2 * np.pi * (day_of_year - 81) / 364
    return float(9.87 * np.sin(2 * angle) - 7.53 * np.cos(angle) - 1.5 * np.sin(angle))


def check_shortwave_diurnal_cycle(
    group: zarr.Group, slots: np.ndarray
) -> tuple[bool, dict[str, Any]]:
    """Check that shortwave is centred on solar noon and near zero at midnight, by valid hour.

    The mean daily curve of shortwave, by UTC hour of the valid time, has a centroid over the
    daylight hours. That centroid must lie within `SOLAR_NOON_TOLERANCE_HOURS` of the solar noon
    of the cells' mean longitude, which the equation of time corrects for the sampled days. A lead
    offset of 1 hour or more moves the centroid outside the tolerance.
    """
    spec = next(spec for spec in FIELDS if spec.variable == "shortwave_down")
    data = read_sample(group, spec, slots)
    init_seconds = np.asarray(_array(group, "init_time")[:])[slots]
    sums = np.zeros(24)
    counts = np.zeros(24)
    equation = []
    for run, seconds in enumerate(init_seconds):
        init = datetime.fromtimestamp(int(seconds), tz=UTC)
        equation.append(equation_of_time_minutes(init.timetuple().tm_yday))
        for lead in range(PLAIN_LAST_STEP + 1):
            hour = (init.hour + lead) % 24
            sums[hour] += float(np.nanmean(data[run, lead]))
            counts[hour] += 1
    mean_by_hour = sums / np.maximum(counts, 1)
    daylight = np.arange(4, 21)
    centroid = float(np.sum(daylight * mean_by_hour[daylight]) / np.sum(mean_by_hour[daylight]))
    longitude = float(np.mean(np.asarray(_array(group, "cell_longitude")[:])))
    solar_noon = 12.0 - longitude / 15.0 - float(np.mean(equation)) / 60.0
    offset = centroid - solar_noon
    ok = (
        abs(offset) <= SOLAR_NOON_TOLERANCE_HOURS
        and mean_by_hour[NIGHT_HOUR_UTC] < NIGHT_MEAN_MAX_W_M2
        and bool(np.all(counts > 0))
    )
    return ok, {
        "centroid_minus_solar_noon_hours": round(offset, 2),
        "mean_w_m2_by_utc_hour": mean_by_hour.round(1).tolist(),
    }


def check_north_south_gradient(group: zarr.Group, slots: np.ndarray) -> tuple[bool, dict[str, Any]]:
    """Check that the stored grid runs north to south, and that temperature falls to the north.

    Latitude must fall as the row index rises. The mean temperature over the sampled runs must
    correlate negatively with latitude, which fails if the rows were read in the wrong order.
    """
    latitude = np.asarray(_array(group, "cell_latitude")[:])
    rows = np.asarray(_array(group, "cell_row")[:])
    rows_run_south = bool(latitude[rows == rows.min()].mean() > latitude[rows == rows.max()].mean())
    spec = next(spec for spec in FIELDS if spec.variable == "temperature_1p5m")
    mean_temperature = np.nanmean(
        read_sample(group, spec, slots)[:, : PLAIN_LAST_STEP + 1], axis=(0, 1)
    )
    correlation = float(np.corrcoef(latitude, mean_temperature)[0, 1])
    return rows_run_south and correlation < 0, {
        "rows_run_north_to_south": rows_run_south,
        "temperature_latitude_correlation": round(correlation, 2),
    }


def check_gust_max_is_a_maximum(
    group: zarr.Group, slots: np.ndarray
) -> tuple[bool, dict[str, Any]]:
    """Check that the maximum gust is not below the instantaneous gust at the same lead.

    The two are rounded to 13 significand bits, so a difference within `GUST_TOLERANCE` is equal.
    """
    by_name = {spec.variable: spec for spec in FIELDS}
    maximum = read_sample(group, by_name["gust_10m_max"], slots)[:, 1 : PLAIN_LAST_STEP + 1]
    instant = read_sample(group, by_name["gust_10m"], slots)[:, 1 : PLAIN_LAST_STEP + 1]
    below = float(np.mean(maximum < instant * (1 - GUST_TOLERANCE) - 0.05))
    return below < MAX_GUST_VIOLATION_FRACTION, {"fraction_max_below_instant": below}


def check_gaps(statuses: np.ndarray) -> tuple[bool, dict[str, Any]]:
    """List the ranges of runs, between the first and last archived, that are not complete.

    A gap is not a failure, since CEDA has missing runs. The check fails only if a slot inside the
    archived range was never visited, which means a run was skipped.
    """
    visited = np.flatnonzero(statuses != 0)
    if visited.size == 0:
        return False, {"gaps": []}
    window = statuses[visited[0] : visited[-1] + 1]
    never_visited = int((window == 0).sum())
    gaps: list[str] = []
    start: int | None = None
    for index, status in enumerate([*window.tolist(), STATUS_COMPLETE]):
        if status != STATUS_COMPLETE and start is None:
            start = index
        elif status == STATUS_COMPLETE and start is not None:
            first = _slot_time(int(visited[0]) + start)
            last = _slot_time(int(visited[0]) + index - 1)
            label = ",".join(
                sorted({STATUS_NAMES.get(int(s), "never") for s in window[start:index]})
            )
            gaps.append(f"{first:%Y-%m-%dT%HZ} to {last:%Y-%m-%dT%HZ} ({label})")
            start = None
    for gap in gaps:
        print(f"  gap: {gap}")
    missing = [f"{_slot_time(int(slot)):%Y-%m-%dT%HZ}" for slot in np.flatnonzero(statuses == 3)]
    return never_visited == 0, {
        "gaps": gaps,
        "never_visited_in_range": never_visited,
        "missing_on_ceda": missing,
    }


def _slot_time(slot: int) -> datetime:
    """The initialisation time of a slot."""
    return datetime.fromtimestamp(int(SLOT_EPOCH.timestamp()) + slot * CYCLE_HOURS * 3600, tz=UTC)


def main() -> int:
    """Run every check and print PASS or FAIL for each.

    Returns:
        0 if every check passed, else 1.
    """
    parser = argparse.ArgumentParser(description="Validate the UKV-CEDA Icechunk store.")
    parser.add_argument("--store-dir", type=Path, default=WEATHER_DOWNLOADS_DIR / PRODUCT_NAME)
    parser.add_argument("--sample", type=int, default=200)
    args = parser.parse_args()
    store = UkvStore.open(store_path=args.store_dir / "store")
    session = store.repository.readonly_session(branch="main")
    group = zarr.open_group(session.store, mode="r")
    statuses = store.statuses()
    readable = np.flatnonzero((statuses == STATUS_COMPLETE) | (statuses == STATUS_PARTIAL))
    complete = np.flatnonzero(statuses == STATUS_COMPLETE)
    complete_sample = sample_slots(complete, sample=args.sample)
    readable_sample = sample_slots(readable, sample=args.sample)
    no_runs = (False, {"error": "no archived run to read"})
    results = {
        "run_spacing": check_run_spacing(group),
        "status_counts": check_status_counts(statuses, group),
        "value_ranges": check_value_ranges(group, readable_sample) if len(readable) else no_runs,
        "nan_layout": check_nan_layout(group, complete_sample) if len(complete) else no_runs,
        "shortwave_diurnal_cycle": (
            check_shortwave_diurnal_cycle(group, readable_sample) if len(readable) else no_runs
        ),
        "north_south_gradient": (
            check_north_south_gradient(group, readable_sample) if len(readable) else no_runs
        ),
        "gust_max_is_a_maximum": (
            check_gust_max_is_a_maximum(group, complete_sample) if len(complete) else no_runs
        ),
        "gaps": check_gaps(statuses),
    }
    for name in CHECK_NAMES:
        print(f"{'PASS' if results[name][0] else 'FAIL'} {name}")
    measured = {name: results[name][1] for name in CHECK_NAMES}
    measured["sampled_complete_runs"] = len(complete_sample)
    measured["readable_runs"] = len(readable)
    (args.store_dir / "validation.json").write_text(json.dumps(measured, indent=2, default=str))
    return 0 if all(passed for passed, _ in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
