"""Validate the WeatherNext 3 Icechunk store that `fetch_weathernext3.py` wrote, and publish it.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/934>, following the
`data-validation` skill's checklist. It opens the repository read-only on `staging` (or on `main`
with `--branch main`), pins the branch's snapshot at the start, validates that snapshot, and prints
one PASS, FAIL, or SKIP line per check. A run that raises is not evidence the data is right, and
neither is a check that prints PASS: a check that fails names the runs it failed in. The report
never prints the bucket name.

**The validator reads every shard, so run it on a Compute Engine machine in us-east1** (or on a
workstation against `--local-store`). Reading the store from outside Google Cloud is billed as
egress.

Store checks: the array names, dtypes, shapes, chunk and shard shapes, and codecs; the `init_time`
axis is the stored range at 6-hour steps; the `lead_time` axis is 1 to 360 hours; the latitude and
longitude axes are regular, ascending, rounded to 0.1 degrees, and span the box.

Checks on each run with `run_written` set: `source_init_time` equals the `init_time` coordinate; no
`NaN` or infinite value; each value inside a physical range (`RANGES`); radiation is near zero for
night-time valid hours and positive at midday in April to September; direct radiation does not
exceed total radiation by more than a small tolerance (`ranges` and `direct_le_total` fail above
0.01% of a run's values, and print that percentage); and no (variable, lead time) slice is constant
across the box, except dark radiation slices. Radiation may dip to -3600 J/m2, because an ensemble
mean from a machine-learning weather model can be slightly negative at night. The night and midday
checks catch only a gross shift such as 12 hours or a timezone error, and do not verify the exact
hour convention.

Checks across runs: no two written runs have the same lead 1 of `temperature_2m_mean`; every slot
without `run_written` is entirely `NaN`; and every such slot inside the scope (`--start-date`,
`--end-date`, `--init-hours`, default the whole stored range and all four hours) is listed in the
skipped-run attributes.

`--compare-source` reads the source store for the first written run, the last, and 5 random ones,
and compares the whole stored slice at the first and last lead of every variable. It selects the
source cells by value from the stored `latitude` and `longitude` (with `longitude % 360`),
independently of the fetch's crop code. It needs `GOOGLE_CLOUD_PROJECT` and reads about 280 MB from
the source per run.

`--publish` validates the snapshot that `staging` pointed at when the script started, checks the
tip of `main` is an ancestor of that snapshot, and then resets `main` to it, but only if `main` is
still at the tip read at the start. Runs committed to `staging` during validation are never
published.

Run it with `uv run python studies/weather_downloads/validate_weathernext3.py --bucket <bucket>
--compare-source`, adding `--publish` once the report passes.
"""

import argparse
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

# `fetch_weathernext3` sets `ICECHUNK_LOG` before `icechunk` is imported below it.
import fetch_weathernext3 as fetch
import gcsfs
import icechunk
import numpy as np
import xarray as xr
import zarr
from fetch_weathernext3 import (
    BUCKET_PREFIX,
    DIMENSIONS,
    INIT_HOURS,
    INIT_TIME,
    LAT_MAX,
    LAT_MIN,
    LATITUDE,
    LATITUDE_DIM,
    LEAD_TIME,
    LEADS,
    LON_MAX,
    LON_MIN,
    LONGITUDE,
    LONGITUDE_DIM,
    MAIN_BRANCH,
    NOT_PUBLISHED_ATTRIBUTE,
    RUN_WRITTEN,
    SHARD_SHAPE,
    SKIPPED_ATTRIBUTE,
    SOURCE_INIT_TIME,
    STAGING_BRANCH,
    VARIABLES,
    _array,
    _filesystem,
    _label,
    _run_name,
    _storage,
    _stored_range,
)
from zarr.codecs import BloscCodec

TEMPERATURE: Final[str] = VARIABLES[0]
TOTAL_RADIATION: Final[str] = "surface_solar_radiation_downwards_1hr_mean"
DIRECT_RADIATION: Final[str] = "total_sky_direct_solar_radiation_at_surface_1hr_mean"
RADIATION_FIELDS: Final[tuple[str, str]] = (TOTAL_RADIATION, DIRECT_RADIATION)

WIND_LIMIT_M_S: Final[float] = 80.0
MAX_HOURLY_RADIATION_J_M2: Final[float] = 4.0e6
MIN_HOURLY_RADIATION_J_M2: Final[float] = -3600.0
"""An ensemble mean from a machine-learning weather model may dip slightly below zero at night, so
the lower bound allows -3600 J/m2 (a mean of -1 W/m2 over the hour)."""
OFFENDING_FRACTION_LIMIT: Final[float] = 1e-4
"""`ranges` and `direct_le_total` fail only when more than 0.01% of a run's values offend, so that a
few cells of a machine-learning weather model's noise do not fail a run, while a shifted or
corrupted field, which offends in far more values, does."""

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
MIDDAY_HOUR: Final[int] = 12
MIDDAY_MEAN_MINIMUM_J_M2: Final[float] = 100.0 * SECONDS_PER_HOUR
"""A mean of 100 W/m2 over the hour."""
SUMMER_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
DIRECT_EXCESS_FRACTION: Final[float] = 0.01
DIRECT_EXCESS_FLOOR_J_M2: Final[float] = 1.0 * SECONDS_PER_HOUR
"""Direct radiation may exceed `max(total, 0)` by this fraction of it plus this floor (a mean of
1 W/m2 over the hour), which covers rounding and a slightly negative total."""

GRID_STEP_DEGREES: Final[float] = 0.1
AXIS_TOLERANCE: Final[float] = 1e-9
"""The largest difference, in degrees, between a stored coordinate and a regular value."""

COMPARE_RELATIVE_TOLERANCE: Final[float] = 2.0**-12
"""Twice the largest relative error of rounding to 13 significand bits."""
RANDOM_COMPARE_RUNS: Final[int] = 5
"""Runs compared with the source in addition to the first and the last."""

HOURS_PER_SLOT: Final[int] = 6

Results = dict[str, list[str]]
"""Check name mapped to the runs it failed in (empty if the check passed)."""

STORE_CHECKS: Final[tuple[str, ...]] = ("structure", "axes")
RUN_CHECKS: Final[tuple[str, ...]] = (
    "source_init_time",
    "nulls_and_nan",
    "ranges",
    "night_total",
    "night_direct",
    "midday_total",
    "direct_le_total",
    "constant_slice",
)
CROSS_RUN_CHECKS: Final[tuple[str, ...]] = ("duplicate_runs", "unwritten_all_nan", "skips_cover")
CHECKS: Final[tuple[str, ...]] = (*STORE_CHECKS, *RUN_CHECKS, *CROSS_RUN_CHECKS)
"""Every check, so that a check no run failed still prints a PASS line."""

_SKIPPED: Final[str] = "skipped"
_NO_RUN: Final[str] = "store"
"""The label a store-wide failure is recorded under."""


@dataclass(frozen=True)
class Scope:
    """The runs whose presence the store must account for, as init days and init hours."""

    start: date
    end: date
    init_hours: tuple[int, ...]


def _fail(*, results: Results, check: str, label: str) -> None:
    """Record that `check` failed in the run `label`."""
    results.setdefault(check, []).append(label)


def _fail_if_many_values(
    *, results: Results, check: str, label: str, offending: int, total: int
) -> None:
    """Record a failure of `check` if more than `OFFENDING_FRACTION_LIMIT` of values offend."""
    fraction = offending / total if total else 0.0
    if fraction > OFFENDING_FRACTION_LIMIT:
        _fail(results=results, check=check, label=f"{label} ({fraction:.3%} of values)")


def _guarded(*, results: Results, check: str, label: str, action: Callable[[], None]) -> None:
    """Call `action`; if it raises, record a failure of `check` and carry on with other checks.

    A malformed store (a wrong dtype, a missing array) is a finding, not a reason to stop
    validating. The report names the exception's type only.
    """
    try:
        action()
    except Exception as error:  # noqa: BLE001  # any failure to run a check is itself a finding
        _fail(results=results, check=check, label=f"{label} (check raised {type(error).__name__})")


def _dimension_names(*, array: zarr.Array) -> tuple[str, ...]:
    """Return the dimension names of `array`, or an empty tuple if it has none."""
    names = array.metadata.to_dict().get("dimension_names")
    return tuple(str(name) for name in names) if isinstance(names, list | tuple) else ()


def _check_structure(*, root: zarr.Group, results: Results) -> None:
    """Check the array names, dtypes, shapes, chunk and shard shapes, dimensions, and codecs."""
    names = {name for name, _ in root.arrays()}
    expected_names = {
        *VARIABLES,
        INIT_TIME,
        LEAD_TIME,
        LATITUDE,
        LONGITUDE,
        RUN_WRITTEN,
        SOURCE_INIT_TIME,
    }
    if names != expected_names or list(root.groups()):
        _fail(results=results, check="structure", label="array names")
        return
    n_runs = _array(root=root, name=INIT_TIME).shape[0]
    shape = (
        n_runs,
        LEADS,
        _array(root=root, name=LATITUDE).shape[0],
        _array(root=root, name=LONGITUDE).shape[0],
    )
    for variable in VARIABLES:
        array = _array(root=root, name=variable)
        codecs = array.compressors
        problems = {
            "dtype": array.dtype != np.float32,
            "shape": array.shape != shape,
            "shards": array.shards != SHARD_SHAPE,
            "chunks": array.chunks != fetch.CHUNK_SHAPE,
            "dimensions": _dimension_names(array=array) != DIMENSIONS,
            "codec": not (
                len(codecs) == 1
                and isinstance(codecs[0], BloscCodec)
                and codecs[0].cname == "zstd"
                and codecs[0].shuffle == "shuffle"
            ),
            "fill_value": not np.isnan(array.fill_value),
        }
        for problem, present in problems.items():
            if present:
                _fail(results=results, check="structure", label=f"{variable} {problem}")
    for name, dtype in (
        (RUN_WRITTEN, np.bool_),
        (SOURCE_INIT_TIME, np.int64),
        (INIT_TIME, np.int64),
    ):
        array = _array(root=root, name=name)
        if array.dtype != dtype or array.shape != (n_runs,):
            _fail(results=results, check="structure", label=name)


def _check_axes(*, root: zarr.Group, results: Results) -> None:
    """Check the four coordinate arrays, and that the stored range agrees with `init_time`."""
    lead = np.asarray(_array(root=root, name=LEAD_TIME)[:])
    if not np.array_equal(lead, np.arange(1, LEADS + 1)):
        _fail(results=results, check="axes", label=LEAD_TIME)
    for name, low, high in ((LATITUDE, LAT_MIN, LAT_MAX), (LONGITUDE, LON_MIN, LON_MAX)):
        axis = np.asarray(_array(root=root, name=name)[:])
        regular = np.allclose(np.diff(axis), GRID_STEP_DEGREES, rtol=0.0, atol=AXIS_TOLERANCE)
        spans_box = np.allclose([axis[0], axis[-1]], [low, high], rtol=0.0, atol=AXIS_TOLERANCE)
        exact = np.array_equal(axis, np.round(axis, 1))
        if not (regular and spans_box and exact):
            _fail(results=results, check="axes", label=name)
    first_day, last_day = _stored_range(root=root)
    init_hours = np.asarray(_array(root=root, name=INIT_TIME)[:])
    start = int(datetime(first_day.year, first_day.month, first_day.day, tzinfo=UTC).timestamp())
    expected = start // SECONDS_PER_HOUR + HOURS_PER_SLOT * np.arange(len(init_hours))
    n_days = (last_day - first_day).days + 1
    if not np.array_equal(init_hours, expected) or len(init_hours) != n_days * len(INIT_HOURS):
        _fail(results=results, check="axes", label=INIT_TIME)


def _valid_times(*, root: zarr.Group, slot: int) -> np.ndarray:
    """Return the valid times of one run, as `datetime64[h]`, one per lead time."""
    init_hours = int(np.asarray(_array(root=root, name=INIT_TIME)[slot]))
    return np.datetime64(init_hours, "h") + np.arange(1, LEADS + 1) * np.timedelta64(1, "h")


def _check_values(*, values: np.ndarray, label: str, results: Results) -> None:
    """Check `NaN`s, infinite values, and physical ranges of every variable of one run."""
    for position, variable in enumerate(VARIABLES):
        array = values[position]
        if not np.isfinite(array).all():
            _fail(results=results, check="nulls_and_nan", label=label)
        low, high = RANGES[variable]
        _fail_if_many_values(
            results=results,
            check="ranges",
            label=label,
            offending=int(((array < low) | (array > high)).sum()),
            total=array.size,
        )


def _check_radiation(
    *, values: np.ndarray, valid: np.ndarray, label: str, results: Results
) -> None:
    """Check radiation is near zero at night, positive at summer midday, and direct <= total."""
    total = values[VARIABLES.index(TOTAL_RADIATION)]
    direct = values[VARIABLES.index(DIRECT_RADIATION)]
    hour = (valid - valid.astype("datetime64[D]")) // np.timedelta64(1, "h")
    month = valid.astype("datetime64[M]").astype(np.int64) % 12 + 1
    night = np.isin(hour, NIGHT_HOURS)
    for check, field in (("night_total", total), ("night_direct", direct)):
        night_values = field[night]
        night_values = night_values[np.isfinite(night_values)]
        if night_values.size == 0:
            _fail(results=results, check=check, label=_SKIPPED)
        elif np.median(night_values) >= NIGHT_MEDIAN_LIMIT_J_M2:
            _fail(results=results, check=check, label=label)
    midday = total[(hour == MIDDAY_HOUR) & np.isin(month, SUMMER_MONTHS)]
    midday = midday[np.isfinite(midday)]
    if midday.size == 0:
        _fail(results=results, check="midday_total", label=_SKIPPED)
    elif midday.mean() <= MIDDAY_MEAN_MINIMUM_J_M2:
        _fail(results=results, check="midday_total", label=label)
    allowed = np.maximum(total, 0.0) * (1.0 + DIRECT_EXCESS_FRACTION) + DIRECT_EXCESS_FLOOR_J_M2
    _fail_if_many_values(
        results=results,
        check="direct_le_total",
        label=label,
        offending=int((direct > allowed).sum()),
        total=direct.size,
    )


def _check_constant_slices(*, values: np.ndarray, label: str, results: Results) -> None:
    """Check no (variable, lead time) slice has the same value in every cell of the box.

    A constant slice is a sign of a fill value or a broken read. A radiation slice at night is
    legitimately constant (near zero everywhere), so a radiation slice is exempt when its largest
    value is below `NIGHT_MEDIAN_LIMIT_J_M2`.
    """
    for position, variable in enumerate(VARIABLES):
        flat = values[position].reshape(values.shape[1], -1)
        low = flat.min(axis=1)
        high = flat.max(axis=1)
        constant = low == high
        if variable in RADIATION_FIELDS:
            constant &= high >= NIGHT_MEDIAN_LIMIT_J_M2
        if constant.any():
            _fail(results=results, check="constant_slice", label=label)


def _load_run(*, root: zarr.Group, slot: int) -> np.ndarray:
    """Return one run as a `(variable, lead_time, latitude, longitude)` `Float32` array."""
    return np.stack([np.asarray(_array(root=root, name=name)[slot]) for name in VARIABLES])


def _validate_written_run(
    *, root: zarr.Group, slot: int, label: str, results: Results
) -> bytes | None:
    """Run every check on one written run; return the bytes of its lead 1 temperature."""
    _guarded(
        results=results,
        check="source_init_time",
        label=label,
        action=lambda: _check_source_init_time(root=root, slot=slot, label=label, results=results),
    )
    try:
        values = _load_run(root=root, slot=slot)
        valid = _valid_times(root=root, slot=slot)
    except Exception as error:  # noqa: BLE001  # an unreadable run is a finding
        _fail(results=results, check="nulls_and_nan", label=f"{label} ({type(error).__name__})")
        return None
    for check, action in (
        ("nulls_and_nan", lambda: _check_values(values=values, label=label, results=results)),
        (
            "night_total",
            lambda: _check_radiation(values=values, valid=valid, label=label, results=results),
        ),
        (
            "constant_slice",
            lambda: _check_constant_slices(values=values, label=label, results=results),
        ),
    ):
        _guarded(results=results, check=check, label=label, action=action)
    return values[VARIABLES.index(TEMPERATURE), 0].tobytes()


def _check_source_init_time(*, root: zarr.Group, slot: int, label: str, results: Results) -> None:
    """Check the init time recorded from the source store equals the `init_time` coordinate."""
    stored = _array(root=root, name=SOURCE_INIT_TIME)[slot]
    if stored != _array(root=root, name=INIT_TIME)[slot]:
        _fail(results=results, check="source_init_time", label=label)


def _check_unwritten(*, root: zarr.Group, slots: list[int], scope: Scope, results: Results) -> None:
    """Check every slot without `run_written` is all `NaN`, and listed as skipped if in `scope`."""
    first_day, _ = _stored_range(root=root)
    listed: set[str] = set()
    for attribute in (SKIPPED_ATTRIBUTE, NOT_PUBLISHED_ATTRIBUTE):
        stored = root.attrs[attribute]
        assert isinstance(stored, list), "a skipped-runs attribute is not a list"
        listed |= {str(label) for label in stored}
    for slot in slots:
        day = first_day + timedelta(days=slot // len(INIT_HOURS))
        hour = INIT_HOURS[slot % len(INIT_HOURS)]
        label = _label(day=day, hour=hour)
        if any(
            not np.isnan(np.asarray(_array(root=root, name=name)[slot])).all() for name in VARIABLES
        ):
            _fail(results=results, check="unwritten_all_nan", label=label)
        in_scope = scope.start <= day <= scope.end and hour in scope.init_hours
        if in_scope and label not in listed:
            _fail(results=results, check="skips_cover", label=label)


def _sample_slots(*, written: list[int]) -> list[int]:
    """Return the first and last written slot and `RANDOM_COMPARE_RUNS` random others."""
    if not written:
        return []
    chosen = {written[0], written[-1]}
    others = [slot for slot in written if slot not in chosen]
    chosen |= set(random.sample(others, min(RANDOM_COMPARE_RUNS, len(others))))
    return sorted(chosen)


def _compare_run_with_source(
    *, root: zarr.Group, fs: gcsfs.GCSFileSystem, slot: int, label: str, results: Results
) -> None:
    """Compare the first and last lead of every variable of one run with the source store.

    The source cells come from the stored `latitude` and `longitude`, matched by value, with the
    longitude wrapped onto the source's 0 to 360 axis.
    """
    day = datetime.strptime(label, "%Y%m%d_%H").replace(tzinfo=UTC)
    path = f"{BUCKET_PREFIX}/{_run_name(day=day.date(), hour=day.hour)}/predictions.zarr"
    source = xr.open_zarr(
        fs.get_mapper(path),
        chunks=None,
        consolidated=False,
        decode_timedelta=True,
    )
    latitudes = np.asarray(_array(root=root, name=LATITUDE)[:])
    longitudes = np.asarray(_array(root=root, name=LONGITUDE)[:]) % 360.0
    lat_positions = np.abs(source[LATITUDE_DIM].to_numpy()[:, None] - latitudes).argmin(axis=0)
    lon_positions = np.abs(source[LONGITUDE_DIM].to_numpy()[:, None] - longitudes).argmin(axis=0)
    assert np.allclose(
        source[LATITUDE_DIM].to_numpy()[lat_positions], latitudes, atol=fetch.AXIS_TOLERANCE
    ), "a stored latitude has no source cell"
    assert np.allclose(
        source[LONGITUDE_DIM].to_numpy()[lon_positions], longitudes, atol=fetch.AXIS_TOLERANCE
    ), "a stored longitude has no source cell"
    for variable in VARIABLES:
        for lead_hours in (1, LEADS):
            expected = (
                source[variable]
                .sel(lead_time=np.timedelta64(lead_hours, "h"))
                .isel({LATITUDE_DIM: lat_positions, LONGITUDE_DIM: lon_positions})
                .transpose(LATITUDE_DIM, LONGITUDE_DIM)
                .to_numpy()
            )
            stored = np.asarray(_array(root=root, name=variable)[slot, lead_hours - 1])
            if not np.allclose(stored, expected, rtol=COMPARE_RELATIVE_TOLERANCE, atol=0.0):
                _fail(results=results, check="compare_source", label=f"{label} {variable}")


def _validate(*, root: zarr.Group, scope: Scope, compare_source: bool, results: Results) -> None:
    """Run every check on the store behind `root`, filling `results`."""
    _guarded(
        results=results,
        check="structure",
        label=_NO_RUN,
        action=lambda: _check_structure(root=root, results=results),
    )
    _guarded(
        results=results,
        check="axes",
        label=_NO_RUN,
        action=lambda: _check_axes(root=root, results=results),
    )
    written_flags = np.asarray(_array(root=root, name=RUN_WRITTEN)[:])
    written = [int(slot) for slot in np.flatnonzero(written_flags)]
    unwritten = [int(slot) for slot in np.flatnonzero(~written_flags)]
    first_day, _ = _stored_range(root=root)
    fingerprints: dict[bytes, list[str]] = {}
    for slot in written:
        label = _label(
            day=first_day + timedelta(days=slot // len(INIT_HOURS)),
            hour=INIT_HOURS[slot % len(INIT_HOURS)],
        )
        fingerprint = _validate_written_run(root=root, slot=slot, label=label, results=results)
        if fingerprint is not None:
            fingerprints.setdefault(fingerprint, []).append(label)
    for labels in fingerprints.values():
        if len(labels) > 1:
            for label in labels:
                _fail(results=results, check="duplicate_runs", label=label)
    _guarded(
        results=results,
        check="unwritten_all_nan",
        label=_NO_RUN,
        action=lambda: _check_unwritten(root=root, slots=unwritten, scope=scope, results=results),
    )
    if compare_source:
        fs = _filesystem()
        for slot in _sample_slots(written=written):
            label = _label(
                day=first_day + timedelta(days=slot // len(INIT_HOURS)),
                hour=INIT_HOURS[slot % len(INIT_HOURS)],
            )
            _guarded(
                results=results,
                check="compare_source",
                label=label,
                action=lambda slot=slot, label=label: _compare_run_with_source(
                    root=root, fs=fs, slot=slot, label=label, results=results
                ),
            )


def _print_report(*, results: Results) -> int:
    """Print one line per check and return the number of failed checks."""
    failed = 0
    for check, failures in results.items():
        real = sorted({run for run in failures if run != _SKIPPED})
        if real:
            failed += 1
            print(f"FAIL {check}: {', '.join(real)}")
        elif failures:
            print(f"SKIP {check}: no qualifying values in a run")
        else:
            print(f"PASS {check}")
    return failed


def _publish(*, repo: icechunk.Repository, snapshot_id: str, main_tip: str) -> bool:
    """Reset `main` to `snapshot_id` if `main_tip` is an ancestor of it; return whether it did."""
    ancestors = {snapshot.id for snapshot in repo.ancestry(snapshot_id=snapshot_id)}
    if main_tip not in ancestors:
        print("REFUSED: the tip of main is not an ancestor of the validated snapshot")
        return False
    repo.reset_branch(MAIN_BRANCH, snapshot_id, from_snapshot_id=main_tip)
    print("published: main now points at the validated snapshot")
    return True


def _run(*, arguments: argparse.Namespace) -> int:
    """Validate the store named by `arguments`, publishing it if asked; return the exit code."""
    repo = icechunk.Repository.open(
        _storage(bucket=arguments.bucket, local_store=arguments.local_store)
    )
    main_tip = repo.lookup_branch(MAIN_BRANCH)
    snapshot_id = repo.lookup_branch(arguments.branch)
    root = zarr.open_group(repo.readonly_session(snapshot_id=snapshot_id).store, mode="r")
    first_day, last_day = _stored_range(root=root)
    scope = Scope(
        start=arguments.start_date or first_day,
        end=arguments.end_date or last_day,
        init_hours=tuple(arguments.init_hours),
    )
    results: Results = {check: [] for check in CHECKS}
    if arguments.compare_source:
        results["compare_source"] = []
    _validate(root=root, scope=scope, compare_source=arguments.compare_source, results=results)
    failed = _print_report(results=results)
    print("validation passed" if not failed else f"{failed} checks failed")
    if failed:
        return 1
    if arguments.publish:
        return 0 if _publish(repo=repo, snapshot_id=snapshot_id, main_tip=main_tip) else 1
    return 0


def main() -> int:
    """Validate the WeatherNext 3 Icechunk store, and publish it to `main` if asked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", help="Output Cloud Storage bucket, in us-east1.")
    parser.add_argument("--local-store", type=Path, help="Output directory, instead of a bucket.")
    parser.add_argument(
        "--branch", choices=(STAGING_BRANCH, MAIN_BRANCH), default=STAGING_BRANCH, help="Branch."
    )
    parser.add_argument("--start-date", type=date.fromisoformat, help="First init day in scope.")
    parser.add_argument("--end-date", type=date.fromisoformat, help="Last init day in scope.")
    parser.add_argument(
        "--init-hours", type=int, nargs="+", choices=INIT_HOURS, default=list(INIT_HOURS)
    )
    parser.add_argument("--compare-source", action="store_true", help="Compare with the source.")
    parser.add_argument("--publish", action="store_true", help="Move main to the validated tip.")
    arguments = parser.parse_args()
    if (arguments.bucket is None) == (arguments.local_store is None):
        parser.error("give exactly one of --bucket and --local-store")
    if arguments.publish and arguments.branch != STAGING_BRANCH:
        parser.error("--publish validates staging, not main")
    try:
        return _run(arguments=arguments)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return fetch.INTERRUPTED_EXIT_CODE
    except BaseException as error:  # noqa: BLE001  # a panic in the Rust core is not an Exception
        print(f"FAILED ({fetch._describe(error=error)})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
