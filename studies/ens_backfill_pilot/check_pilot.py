"""Check the pilot download against ecCodes and the physics, and write `report.md`.

Run with `uv run --with eccodes python studies/ens_backfill_pilot/check_pilot.py`. Every number in
the report is printed by this script. Section (a) needs the `eccodes` package and is skipped
without it.
"""

import argparse
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from pilot_common import (
    ACCUMULATED_VARIABLES,
    CROP_COLUMNS,
    CROP_LATITUDES,
    CROP_LONGITUDES_DEGREES_EAST,
    FIRST_ROW,
    LAST_ROW,
    PILOT_VARIABLES,
    STEPS_HOURS,
    VARIABLE_MESSAGE,
    RequestStats,
    fetch_idx,
    file_url,
    get_bytes,
    prefix_bytes,
    source_file_name,
)
from studies.deaccumulation import (
    PRECIPITATION_INVALID_BELOW_MM_PER_S,
    RADIATION_INVALID_BELOW_W_PER_M2,
    DeaccumulatedRates,
    deaccumulate_to_rates,
)
from studies.ens_grib_source import IdxEntry, find_gaps, prefix_range_header
from studies.grib1_simple import decode_values, parse_header, unpack_rows
from studies.sources import ENS_BACKFILL_PILOT_DIR, REPO_DATA_DIR

try:
    # Not a workspace dependency: run with `uv run --with eccodes`.
    import eccodes  # ty: ignore[unresolved-import]
except ImportError:
    eccodes = None

GRAVITY: float = 9.80665
FIXED_STEPS: tuple[int, ...] = (0, 3, 144, 150, 360)
RANDOM_SAMPLES_PER_DATE = 4
INTEGRITY_SAMPLES = 20
CHECK_SEED = 20260927
REPORT_PATH = Path(__file__).parent / "report.md"

lines: list[str] = []


def emit(text: str = "") -> None:
    """Print a line and keep it for the report."""
    print(text)
    lines.append(text)


Position = tuple[int, int, int]
"""Indices into the (variable, member, step) axes of a checkpoint file."""


def load_dates() -> dict[date, dict[str, np.ndarray]]:
    """Load every control-member checkpoint, keeping the arrays the checks need."""
    keys = [
        "values",
        "reference_value",
        "binary_scale",
        "decimal_scale",
        "bits_per_value",
        "sha256",
        "message_offset",
        "message_length",
        "range_bytes",
        "file_name",
        "steps_hours",
        "variables",
        "latitudes",
        "longitudes_degrees_east",
    ]
    loaded = {}
    for path in sorted((ENS_BACKFILL_PILOT_DIR / "control").glob("*.npz")):
        with np.load(path) as archive:
            loaded[date.fromisoformat(path.stem)] = {key: archive[key] for key in keys}
    return loaded


def entry_at(*, arrays: dict[str, np.ndarray], position: Position) -> IdxEntry:
    """Rebuild the idx entry of the message stored at `position`."""
    variable = PILOT_VARIABLES[position[0]]
    levtype, param, level = VARIABLE_MESSAGE[variable]
    return IdxEntry(
        parameter=param,
        level_type=levtype,
        level=level,
        step_hours=STEPS_HOURS[position[2]],
        member=0,
        offset=int(arrays["message_offset"][position]),
        length=int(arrays["message_length"][position]),
    )


def choose_decode_samples(
    *, dates: dict[date, dict[str, np.ndarray]]
) -> list[tuple[date, Position]]:
    """Choose the messages to compare with ecCodes, as the brief specifies."""
    rng = random.Random(CHECK_SEED)
    chosen: list[tuple[date, Position]] = []
    variable_index = {name: index for index, name in enumerate(PILOT_VARIABLES)}
    last_step = STEPS_HOURS.index(360)
    fully_sampled = {min(dates), date(2023, 6, 26)} & set(dates)
    for day, arrays in dates.items():
        picks: set[Position] = set()
        u10 = variable_index["10u"]
        most_negative = int(np.argmin(arrays["reference_value"][u10, 0]))
        picks.add((u10, 0, most_negative))
        for name in ("10u", "2t", "ssrd", "tp"):
            picks.add((variable_index[name], 0, last_step))
        for _ in range(RANDOM_SAMPLES_PER_DATE):
            picks.add((rng.randrange(len(PILOT_VARIABLES)), 0, rng.randrange(len(STEPS_HOURS))))
        if day in fully_sampled:
            for variable in range(len(PILOT_VARIABLES)):
                for step in FIXED_STEPS:
                    picks.add((variable, 0, STEPS_HOURS.index(step)))
        chosen.extend((day, position) for position in sorted(picks))
    return chosen


def compare_with_eccodes(
    *, day: date, arrays: dict[str, np.ndarray], position: Position, first: bool
) -> dict[str, bool | float]:
    """Fetch one whole message and compare the hand decode and the stored rows with ecCodes."""
    assert eccodes is not None
    entry = entry_at(arrays=arrays, position=position)
    file_name = str(arrays["file_name"][position])
    body = get_bytes(
        url=file_url(day=day, file_name=file_name),
        byte_range=prefix_range_header(entry=entry, prefix_bytes=entry.length),
        expected_length=entry.length,
        stats=RequestStats(),
    )
    header = parse_header(message=body)
    packed = unpack_rows(message=body, header=header, first_row=0, last_row=header.nj - 1)
    hand = decode_values(packed=packed, header=header)
    handle = eccodes.codes_new_from_message(body)
    try:
        reference = eccodes.codes_get_values(handle).reshape(header.nj, header.ni)
        names_agree = (
            eccodes.codes_get(handle, "shortName") == entry.parameter
            and eccodes.codes_get(handle, "step") == entry.step_hours
            and eccodes.codes_get(handle, "number") == 0
            and eccodes.codes_get(handle, "level") == entry.level
        )
        header_agrees = (
            eccodes.codes_get(handle, "referenceValue") == header.reference_value
            and eccodes.codes_get(handle, "binaryScaleFactor") == header.binary_scale
            and eccodes.codes_get(handle, "decimalScaleFactor") == header.decimal_scale
            and eccodes.codes_get(handle, "bitsPerValue") == header.bits_per_value
        )
        grid_agrees = True
        if first:
            latitudes = eccodes.codes_get_array(handle, "latitudes").reshape(header.nj, header.ni)
            longitudes = eccodes.codes_get_array(handle, "longitudes").reshape(header.nj, header.ni)
            grid_agrees = bool(
                np.array_equal(latitudes[FIRST_ROW : LAST_ROW + 1, 0], CROP_LATITUDES)
                and np.array_equal(longitudes[0, CROP_COLUMNS], CROP_LONGITUDES_DEGREES_EAST % 360)
            )
    finally:
        eccodes.codes_release(handle)
    band = (slice(FIRST_ROW, LAST_ROW + 1),)
    stored = arrays["values"][position]
    return {
        "hand_equals_eccodes": bool(np.array_equal(hand, reference)),
        "max_abs_difference": float(np.max(np.abs(hand - reference))),
        "stored_values_equal": bool(
            np.array_equal(reference[band[0]][:, CROP_COLUMNS].astype(np.float32), stored)
        ),
        "stored_header_equal": bool(
            (
                arrays["reference_value"][position],
                arrays["binary_scale"][position],
                arrays["decimal_scale"][position],
                arrays["bits_per_value"][position],
            )
            == (
                header.reference_value,
                header.binary_scale,
                header.decimal_scale,
                header.bits_per_value,
            )
        ),
        "length_equal": len(body) == header.total_length == entry.length,
        "eccodes_names_agree": bool(names_agree and header_agrees),
        "grid_agrees": grid_agrees,
    }


def section_decoding(*, dates: dict[date, dict[str, np.ndarray]]) -> None:
    """Section (a) and (b): compare a sample of messages with ecCodes, and check lengths."""
    emit("## (a) Hand decode against ecCodes, and (b) message lengths")
    emit()
    if eccodes is None:
        emit("SKIPPED: the eccodes package is not installed (use `uv run --with eccodes`).")
        return
    emit(f"ecCodes version {eccodes.codes_get_api_version()}.")
    samples = choose_decode_samples(dates=dates)
    emit(f"Messages compared: {len(samples)} across {len(dates)} dates, each fetched whole.")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda item: compare_with_eccodes(
                    day=item[1][0],
                    arrays=dates[item[1][0]],
                    position=item[1][1],
                    first=item[0] == 0,
                ),
                enumerate(samples),
            )
        )
    for key in results[0]:
        if key == "max_abs_difference":
            emit(
                f"Largest absolute difference, hand decode against ecCodes: "
                f"{max(r[key] for r in results)}"
            )
        else:
            failures = sum(not r[key] for r in results)
            emit(f"{key}: {len(results) - failures} of {len(results)} true, {failures} mismatches")
    negative = [
        (day, PILOT_VARIABLES[p[0]], STEPS_HOURS[p[2]], float(arrays["reference_value"][p]))
        for day, arrays in dates.items()
        for p in [(PILOT_VARIABLES.index("10u"), 0, s) for s in range(len(STEPS_HOURS))]
        if arrays["reference_value"][p] < 0
    ]
    emit(f"10u messages with a negative reference value across the pilot: {len(negative)}")
    sampled_negative = sum(
        1
        for day, p in samples
        if PILOT_VARIABLES[p[0]] == "10u" and dates[day]["reference_value"][p] < 0
    )
    emit(f"Of the compared messages, 10u with a negative reference value: {sampled_negative}")
    emit()
    emit("### (b) The idx chain")
    emit()
    for day, arrays in dates.items():
        stats = RequestStats()
        problems = 0
        mismatches = 0
        for file_name in ("cf_sfc.grib", "cf_pl.grib"):
            entries = fetch_idx(day=day, file_name=file_name, stats=stats)
            problems += len(find_gaps(entries=entries))
            by_key = {(e.parameter, e.level_type, e.level, e.step_hours): e for e in entries}
            for variable_index, variable in enumerate(PILOT_VARIABLES):
                levtype, param, level = VARIABLE_MESSAGE[variable]
                if source_file_name(member=0, levtype=levtype) != file_name:
                    continue
                for step_index, step in enumerate(STEPS_HOURS):
                    entry = by_key[(param, levtype, level, step)]
                    position = (variable_index, 0, step_index)
                    mismatches += (
                        entry.offset != arrays["message_offset"][position]
                        or entry.length != arrays["message_length"][position]
                    )
        emit(
            f"{day}: idx gaps or overlaps {problems}; "
            f"stored offset or length differs from idx {mismatches}"
        )
    emit()


def section_integrity(*, dates: dict[date, dict[str, np.ndarray]]) -> None:
    """Section (c): fetch random ranges a second time and compare their SHA-256."""
    import hashlib

    emit("## (c) Integrity: a second fetch of random ranges")
    emit()
    rng = random.Random(CHECK_SEED + 1)
    every = [
        (day, (v, 0, s))
        for day in dates
        for v in range(len(PILOT_VARIABLES))
        for s in range(len(STEPS_HOURS))
    ]
    picks = rng.sample(every, k=min(INTEGRITY_SAMPLES, len(every)))
    different = 0
    for day, position in picks:
        arrays = dates[day]
        entry = entry_at(arrays=arrays, position=position)
        length = prefix_bytes()
        body = get_bytes(
            url=file_url(day=day, file_name=str(arrays["file_name"][position])),
            byte_range=prefix_range_header(entry=entry, prefix_bytes=length),
            expected_length=length,
            stats=RequestStats(),
        )
        different += hashlib.sha256(body).hexdigest() != str(arrays["sha256"][position])
    emit(f"Ranges re-fetched: {len(picks)}; SHA-256 differs from the first fetch: {different}")
    emit()


def section_physics(*, dates: dict[date, dict[str, np.ndarray]]) -> None:
    """Section (d): minimum, maximum and mean of every variable, and the temperature units."""
    emit("## (d) Physical sanity")
    emit()
    emit("| variable | min | max | mean | non-finite |")
    emit("|---|---|---|---|---|")
    means = {}
    for index, name in enumerate(PILOT_VARIABLES):
        stacked = np.stack([arrays["values"][index, 0] for arrays in dates.values()])
        finite = stacked[np.isfinite(stacked)]
        means[name] = float(finite.mean())
        emit(
            f"| {name} | {finite.min():.6g} | {finite.max():.6g} | {means[name]:.6g} | "
            f"{stacked.size - finite.size} |"
        )
    for name in ("2t", "2d"):
        unit = "KELVIN" if means[name] > 150 else "DEGREES CELSIUS"
        emit(f"{name}: mean {means[name]:.2f} means the values are in {unit}.")
    emit(
        f"z500 mean {means['z500']:.0f} m2 s-2 is {means['z500'] / GRAVITY:.0f} m "
        "of geopotential height."
    )
    emit()


def _deaccumulate(
    *, arrays: dict[str, np.ndarray], name: str, fixed_step_seconds: bool = False
) -> DeaccumulatedRates:
    accumulation = arrays["values"][PILOT_VARIABLES.index(name), 0].astype(np.float64)
    if name == "tp":
        accumulation = accumulation * 1000
    threshold = (
        PRECIPITATION_INVALID_BELOW_MM_PER_S if name == "tp" else RADIATION_INVALID_BELOW_W_PER_M2
    )
    seconds = np.array(STEPS_HOURS) * 3600
    if fixed_step_seconds:
        seconds = np.arange(len(STEPS_HOURS)) * 10_800
    return deaccumulate_to_rates(
        accumulations=accumulation, elapsed_seconds=seconds, invalid_below_rate=threshold
    )


def section_deaccumulation(*, dates: dict[date, dict[str, np.ndarray]]) -> None:
    """Section (e): apply Dynamical's de-accumulation and report what it clamps or invalidates."""
    emit("## (e) De-accumulation")
    emit()
    emit("Dynamical.org expects a clamped fraction of 0.08 and an invalid fraction of 0.01.")
    emit("| date | variable | lead-0 max abs | clamped | invalid | drops in the accumulation |")
    emit("|---|---|---|---|---|---|")
    totals = {name: [0, 0, 0] for name in ACCUMULATED_VARIABLES}
    for day, arrays in dates.items():
        for name in ACCUMULATED_VARIABLES:
            result = _deaccumulate(arrays=arrays, name=name)
            count = result.clamped[1:].size
            accumulation = arrays["values"][PILOT_VARIABLES.index(name), 0]
            drops = int(np.sum(np.diff(accumulation.astype(np.float64), axis=0) < 0))
            lead0 = float(np.abs(accumulation[0]).max())
            emit(
                f"| {day} | {name} | {lead0:.3g} | {result.clamped.sum() / count:.4f} | "
                f"{result.invalid.sum() / count:.4f} | {drops / count:.4f} |"
            )
            totals[name][0] += int(result.clamped.sum())
            totals[name][1] += int(result.invalid.sum())
            totals[name][2] += count
    for name, (clamped, invalid, count) in totals.items():
        emit(
            f"{name} over all dates: clamped fraction {clamped / count:.4f}, "
            f"invalid fraction {invalid / count:.4f}"
        )
    emit()
    emit(
        "Step change at 144 to 150 h: mean strd rate (W m-2) from the actual elapsed "
        "seconds against"
    )
    emit("a fixed 3 h divisor. The actual-seconds rate should be continuous across the change.")
    emit("| step (h) | actual seconds | fixed 3 h |")
    emit("|---|---|---|")
    right = np.stack([_deaccumulate(arrays=a, name="strd").rates for a in dates.values()])
    wrong = np.stack(
        [
            _deaccumulate(arrays=a, name="strd", fixed_step_seconds=True).rates
            for a in dates.values()
        ]
    )
    for step in (138, 141, 144, 150, 156, 162):
        index = STEPS_HOURS.index(step)
        emit(f"| {step} | {np.nanmean(right[:, index]):.2f} | {np.nanmean(wrong[:, index]):.2f} |")
    emit()


def section_grid(*, dates: dict[date, dict[str, np.ndarray]]) -> None:
    """Section (f): compare the fetched rows and columns with the pipeline's grid cells."""
    emit("## (f) Grid registration")
    emit()
    weights = pl.read_parquet(REPO_DATA_DIR / "h3_grid_weights.parquet")
    latitudes = weights["nwp_lat"].unique().to_numpy().astype(np.float64)
    longitudes_east = np.mod(weights["nwp_lon"].unique().to_numpy().astype(np.float64), 360)
    fetched_latitudes = next(iter(dates.values()))["latitudes"]
    fetched_longitudes = next(iter(dates.values()))["longitudes_degrees_east"]
    missing_latitudes = [v for v in latitudes if not np.isclose(fetched_latitudes, v).any()]
    missing_longitudes = [v for v in longitudes_east if not np.isclose(fetched_longitudes, v).any()]
    emit(
        f"The pipeline uses {len(latitudes)} latitudes from {latitudes.min()} to "
        f"{latitudes.max()} degrees north."
    )
    emit(
        f"The pipeline uses {len(longitudes_east)} longitudes from {weights['nwp_lon'].min()} to "
        f"{weights['nwp_lon'].max()} degrees east on the -180 to 180 axis."
    )
    emit(
        f"The fetched rows run from {fetched_latitudes.max()} to "
        f"{fetched_latitudes.min()} degrees north."
    )
    emit(f"Pipeline latitudes missing from the fetched rows: {missing_latitudes}")
    emit(f"Pipeline longitudes missing from the fetched columns: {missing_longitudes}")
    emit("Section (a) checks the ecCodes latitude and longitude arrays of one message.")
    emit()


def section_validation(*, dates: dict[date, dict[str, np.ndarray]]) -> None:
    """Section (g): gaps, duplicates, NaNs, the diurnal profile and monotonic accumulations."""
    emit("## (g) Data-validation checklist")
    emit()
    steps_ok = sum(list(a["steps_hours"]) == list(STEPS_HOURS) for a in dates.values())
    emit(f"Dates whose step axis equals the 85 expected steps exactly: {steps_ok} of {len(dates)}")
    duplicates = sum(
        len(a["sha256"][:, 0].ravel()) - len(set(a["sha256"][:, 0].ravel().tolist()))
        for a in dates.values()
    )
    emit(
        f"Duplicate range hashes within a date (would mean two keys read one message): {duplicates}"
    )
    nan_count = sum(int(np.isnan(a["values"]).sum()) for a in dates.values())
    emit(f"NaN values in all stored arrays: {nan_count}")
    emit("Mean ssrd rate (W m-2) by valid hour of day, leads 3 to 24 h (period ending):")
    emit("| valid hour (UTC) | mean | maximum |")
    emit("|---|---|---|")
    rates = np.stack([_deaccumulate(arrays=a, name="ssrd").rates for a in dates.values()])
    night_maxima = []
    for step in range(3, 25, 3):
        index = STEPS_HOURS.index(step)
        emit(
            f"| {step % 24:02d} | {np.nanmean(rates[:, index]):.2f} | "
            f"{np.nanmax(rates[:, index]):.2f} |"
        )
        if step in (3, 24):
            night_maxima.append(float(np.nanmax(rates[:, index])))
    emit(f"Largest ssrd rate at valid hours 00 and 03 UTC: {max(night_maxima):.3f} W m-2")
    emit()


def main() -> int:
    """Run every section and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--offline", action="store_true", help="skip sections (a) to (c), which need the network"
    )
    args = parser.parse_args()
    started = time.monotonic()
    dates = load_dates()
    if not dates:
        print("No checkpoint files found. Run fetch_pilot.py first.", file=sys.stderr)
        return 1
    emit("# ENS backfill pilot: check report")
    emit()
    emit(f"Checkpoint files read: {len(dates)} dates, {min(dates)} to {max(dates)}.")
    emit()
    if not args.offline:
        section_decoding(dates=dates)
        section_integrity(dates=dates)
    section_physics(dates=dates)
    section_deaccumulation(dates=dates)
    section_grid(dates=dates)
    section_validation(dates=dates)
    emit("## (h) Totals")
    emit()
    files = sorted((ENS_BACKFILL_PILOT_DIR / "control").glob("*.npz"))
    emit(
        f"Checkpoint files: {len(files)}, "
        f"{sum(f.stat().st_size for f in files) / 1e6:.1f} MB on disk."
    )
    messages = sum(a["message_length"].size for a in dates.values())
    range_bytes = sum(int(a["range_bytes"].sum()) for a in dates.values())
    emit(f"Messages stored: {messages}; range bytes fetched for them: {range_bytes / 1e9:.2f} GB.")
    emit(f"The check took {time.monotonic() - started:.0f} s.")
    args.report.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
