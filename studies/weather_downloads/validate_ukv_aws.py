"""Validate the Met Office UKV files that `fetch_ukv_aws_pilot.py` writes, read-only.

One-off throwaway script, following the `data-validation` skill. It reads the day files of
`data/studies/downloads/NWP/UKV-AWS/` that the download's ledger lists as committed, and never
writes there. The report goes to a new folder under `data/studies/per_study/ukv_aws_validation/`,
which is write-once: a second run needs a different `--label`.

**What is checked, per run file and then per day and per month.**

- Presence: every committed day holds 24 run files, and every run holds 48 objects (6 leads times
  8 files) unless its `missing` list says otherwise.
- Readability: every array of every run file loads, which catches a truncated file.
- Timestamp convention: each variable's `time_s` equals run time plus lead times 3600 s,
  `lead_hours` is 0 to 5, and no file carries time bounds, so every field is an instantaneous value.
- NaN fraction and physical range per variable, with the count of values outside `PHYSICAL_RANGES`
  (which includes the non-negative shortwave components).
- Stuck fields: a temperature or wind slice with the same value in every cell, and a run whose
  lead-0 temperature field is identical to another run's (duplicates).
- Internal consistency: total shortwave minus direct minus diffuse, where total exceeds 1 W m-2,
  by month. The 2026 files close the budget to rounding, and the 2024 files did not in the pilot.
- Height levels: the files hold 33 levels before 2026-01-21 and 56 afterwards, but only the four
  heights in `fetch_ukv_aws_pilot.HUB_HEIGHTS_M` are kept, so the check is that the kept heights
  are the same on every run and that the mean wind speed at each height is continuous across
  `LEVEL_TRANSITION_DAYS`.
- Orientation: both grid axes strictly increase (row 0 is south), and the axes are identical in
  every run file.
- Attributes: the units and standard names stored with each variable are the same in every run.

**No coordinate, bound, or cell count is written to the report.** The `x` and `y` arrays are hashed
and compared, never printed.

Run it with `uv run python studies/weather_downloads/validate_ukv_aws.py` (add `--max-days 3` for
a quick look).
"""

import argparse
import datetime as dt
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from fetch_ukv_aws_pilot import ALL_FILES, LEADS_HOURS, NO_BOUNDS, PRODUCT_DIR, RUNS_PER_DAY
from studies.sources import study_dir_for

REPORT_PARENT: Final[Path] = study_dir_for(study="ukv_aws_validation")
"""The folder that holds one write-once report folder per validation run."""

OBJECTS_PER_RUN: Final[int] = len(LEADS_HOURS) * len(ALL_FILES)
SHORTWAVE_TOTAL: Final[str] = "radiation_flux_in_shortwave_total_downward_at_surface"
SHORTWAVE_DIRECT: Final[str] = "radiation_flux_in_shortwave_direct_downward_at_surface"
SHORTWAVE_DIFFUSE: Final[str] = "radiation_flux_in_shortwave_diffuse_downward_at_surface"
WIND_ON_LEVELS: Final[str] = "wind_speed_on_height_levels"
TEMPERATURE: Final[str] = "temperature_at_screen_level"

PHYSICAL_RANGES: Final[dict[str, tuple[float, float]]] = {
    TEMPERATURE: (230.0, 330.0),
    "wind_speed_at_10m": (0.0, 80.0),
    "wind_direction_at_10m": (0.0, 360.0),
    SHORTWAVE_TOTAL: (0.0, 1500.0),
    SHORTWAVE_DIRECT: (0.0, 1500.0),
    SHORTWAVE_DIFFUSE: (0.0, 1500.0),
    WIND_ON_LEVELS: (0.0, 100.0),
    "wind_direction_on_height_levels": (0.0, 360.0),
}
"""Inclusive limits, in each variable's own unit (K, m s-1, degrees, W m-2). Anything outside is
counted, not removed."""

STUCK_CHECKED: Final[tuple[str, ...]] = (TEMPERATURE, "wind_speed_at_10m", WIND_ON_LEVELS)
"""Variables whose slices should never be one constant. Shortwave is zero everywhere at night."""

LEVEL_TRANSITION_DAYS: Final[tuple[dt.date, dt.date]] = (
    dt.date(2026, 1, 20),
    dt.date(2026, 1, 22),
)
"""The days either side of the PS47 change, when the files went from 33 to 56 levels."""

RESIDUAL_THRESHOLD_W_M2: Final[float] = 10.0
DAYLIGHT_W_M2: Final[float] = 1.0
MAX_ANOMALY_ROWS: Final[int] = 60


def _hash(values: np.ndarray) -> str:
    """A short digest of an array's bytes, to compare arrays without printing them."""
    return hashlib.sha1(np.ascontiguousarray(values).tobytes(), usedforsecurity=False).hexdigest()[
        :12
    ]


def _variable_stats(*, name: str, values: np.ndarray) -> dict[str, Any]:
    """NaN count, extremes, and out-of-range count of one variable of one run."""
    low, high = PHYSICAL_RANGES[name]
    finite = values[np.isfinite(values)]
    stats: dict[str, Any] = {
        f"{name}__values": int(values.size),
        f"{name}__nan": int(values.size - finite.size),
        f"{name}__min": float(finite.min()) if finite.size else None,
        f"{name}__max": float(finite.max()) if finite.size else None,
        f"{name}__outside": int(np.count_nonzero((finite < low) | (finite > high))),
    }
    if name in STUCK_CHECKED:
        slices = values.reshape(-1, *values.shape[-2:])
        stats[f"{name}__stuck"] = int(
            sum(
                np.ptp(piece[np.isfinite(piece)]) == 0
                for piece in slices
                if np.isfinite(piece).any()
            )
        )
    return stats


def _radiation_residual(*, archive: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Mean absolute total-minus-direct-minus-diffuse residual where the sun is up."""
    total = archive[SHORTWAVE_TOTAL].astype(np.float64)
    residual = np.abs(total - archive[SHORTWAVE_DIRECT] - archive[SHORTWAVE_DIFFUSE])
    daylight = total > DAYLIGHT_W_M2
    if not daylight.any():
        return {"residual_mean": None, "residual_share_over": None, "daylight_values": 0}
    chosen = residual[daylight]
    return {
        "residual_mean": float(chosen.mean()),
        "residual_share_over": float((chosen > RESIDUAL_THRESHOLD_W_M2).mean()),
        "daylight_values": int(chosen.size),
    }


def summarise_run(*, path: Path) -> dict[str, Any]:
    """Read one run file and return one flat record of everything the checks need.

    Args:
        path: A `<run>.npz` file. It is only read.

    Returns:
        A record with `run`, `readable`, and, when the file loads, the per-run checks. A file that
        cannot be read has `readable` false and the exception's type in `error`.
    """
    run = path.stem
    record: dict[str, Any] = {"run": run, "day": run[:8], "readable": True}
    try:
        with path.open("rb") as handle, np.load(handle, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile) as error:
        return {**record, "readable": False, "error": type(error).__name__}
    run_time = int(dt.datetime.strptime(run, "%Y%m%dT%H%MZ").replace(tzinfo=dt.UTC).timestamp())
    expected_times = run_time + 3600 * np.arange(len(LEADS_HOURS))
    present = [name for name in ALL_FILES if name in arrays]
    missing = json.loads(str(arrays["missing"]))
    record |= {
        "variables": len(present),
        "absent_objects": len(missing),
        "objects": OBJECTS_PER_RUN - len(missing),
        "lead_hours_ok": arrays["lead_hours"].tolist() == list(LEADS_HOURS),
        "time_ok": all(
            np.array_equal(
                arrays[f"{name}__time_s"][~_absent_lead(arrays=arrays, name=name)],
                expected_times[~_absent_lead(arrays=arrays, name=name)],
            )
            for name in present
        ),
        "has_time_bounds": any(
            bool((arrays[f"{name}__time_bounds_s"] != NO_BOUNDS).any()) for name in present
        ),
        "x_increasing": bool(np.all(np.diff(arrays["x"]) > 0)),
        "y_increasing": bool(np.all(np.diff(arrays["y"]) > 0)),
        "axes_hash": _hash(np.concatenate([arrays["x"], arrays["y"]])),
        "heights_hash": _hash(arrays["height_m"]) if "height_m" in arrays else None,
        "temperature_hash": _hash(arrays[TEMPERATURE][0]) if TEMPERATURE in arrays else None,
        "attributes": json.dumps(
            {name: json.loads(str(arrays[f"{name}__attributes"])) for name in present},
            sort_keys=True,
        ),
    }
    for name in present:
        record |= _variable_stats(name=name, values=arrays[name])
    if all(name in arrays for name in (SHORTWAVE_TOTAL, SHORTWAVE_DIRECT, SHORTWAVE_DIFFUSE)):
        record |= _radiation_residual(archive=arrays)
    if WIND_ON_LEVELS in arrays:
        for index, mean in enumerate(np.nanmean(arrays[WIND_ON_LEVELS], axis=(0, 2, 3))):
            record[f"wind_levels_mean_{index}"] = float(mean)
    return record


def _absent_lead(*, arrays: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    """Which leads of a variable are absent: those whose stored time is the `NO_BOUNDS` marker."""
    return arrays[f"{name}__time_s"] == NO_BOUNDS


def committed_days(*, product_dir: Path) -> list[dt.date]:
    """List the days whose ledger entry exists, from the final ledger files only."""
    return sorted(
        dt.datetime.strptime(path.stem, "%Y%m%d").date()  # noqa: DTZ007
        for path in (product_dir / "ledger").glob("????????.json")
    )


def per_day_table(*, records: pl.DataFrame) -> pl.DataFrame:
    """Aggregate the run records of each day: runs, objects, NaN, range, stuck, and unreadable."""
    value_columns = [column for column in records.columns if column.endswith("__values")]
    nan_columns = [column.removesuffix("__values") + "__nan" for column in value_columns]
    outside = [column for column in records.columns if column.endswith("__outside")]
    stuck = [column for column in records.columns if column.endswith("__stuck")]
    readable = records.filter(pl.col("readable"))
    return (
        records.group_by("day")
        .agg(runs=pl.len(), unreadable=(~pl.col("readable")).sum())
        .join(
            readable.group_by("day").agg(
                objects_min=pl.col("objects").min(),
                objects_max=pl.col("objects").max(),
                absent_objects=pl.col("absent_objects").sum(),
                nan_share=pl.sum_horizontal(nan_columns).sum()
                / pl.sum_horizontal(value_columns).sum(),
                outside_range=pl.sum_horizontal(outside).sum(),
                stuck_slices=pl.sum_horizontal(stuck).sum(),
                bad_time=(~pl.col("time_ok") | ~pl.col("lead_hours_ok")).sum(),
                with_time_bounds=pl.col("has_time_bounds").sum(),
            ),
            on="day",
            how="left",
        )
        .sort("day")
    )


def radiation_by_month(*, records: pl.DataFrame) -> pl.DataFrame:
    """The daytime total-minus-direct-minus-diffuse residual of each month."""
    return (
        records.filter(pl.col("readable") & pl.col("residual_mean").is_not_null())
        .with_columns(month=pl.col("day").str.slice(0, 6))
        .group_by("month")
        .agg(
            runs=pl.len(),
            mean_abs_residual_w_m2=pl.col("residual_mean").mean(),
            worst_run_residual_w_m2=pl.col("residual_mean").max(),
            share_over_threshold=pl.col("residual_share_over").mean(),
        )
        .sort("month")
    )


def level_transition(*, records: pl.DataFrame) -> pl.DataFrame:
    """Mean wind speed at each kept height, per day, for the days around the level change."""
    first, last = (f"{day:%Y%m%d}" for day in LEVEL_TRANSITION_DAYS)
    columns = [column for column in records.columns if column.startswith("wind_levels_mean_")]
    return (
        records.filter(pl.col("readable") & (pl.col("day") >= first) & (pl.col("day") <= last))
        .group_by("day")
        .agg([pl.col(column).mean() for column in sorted(columns)])
        .sort("day")
    )


def consistency_findings(*, records: pl.DataFrame) -> list[str]:
    """Findings from the checks that compare run files with each other."""
    readable = records.filter(pl.col("readable"))
    findings: list[str] = []
    if not readable.height:
        return ["No run file could be read."]
    for column, label in (("axes_hash", "grid axes"), ("heights_hash", "kept heights")):
        distinct = readable[column].drop_nulls().n_unique()
        findings.append(f"{label}: {distinct} distinct value(s) across {readable.height} runs.")
    findings.append(
        f"Row orientation: y increases with the row index in "
        f"{int(readable['y_increasing'].sum())} of {readable.height} runs "
        f"(row 0 is south), x increases in {int(readable['x_increasing'].sum())}."
    )
    attributes = readable["attributes"].n_unique()
    findings.append(
        f"Units and standard names: {attributes} distinct attribute set(s) in all runs."
    )
    hashes = Counter(readable["temperature_hash"].drop_nulls().to_list())
    duplicates = {key: count for key, count in hashes.items() if count > 1}
    findings.append(
        f"Duplicates: {sum(duplicates.values())} runs share a lead-0 temperature field with "
        f"another run ({len(duplicates)} distinct fields)."
    )
    return findings


def runs_missing_per_day(*, records: pl.DataFrame, days: Sequence[dt.date]) -> list[str]:
    """List each committed day that does not hold exactly 24 run files."""
    counts = Counter(records["day"].to_list())
    return [
        f"{day}: {counts.get(f'{day:%Y%m%d}', 0)} run files"
        for day in days
        if counts.get(f"{day:%Y%m%d}", 0) != RUNS_PER_DAY
    ]


def _cell(value: object) -> str:
    """Format one table cell."""
    if value is None:
        return ""
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def _markdown(*, frame: pl.DataFrame) -> str:
    """Render a small frame as a markdown table."""
    header = "| " + " | ".join(frame.columns) + " |\n|" + "---|" * len(frame.columns) + "\n"
    rows = (
        "| "
        + " | ".join(
            "" if v is None else f"{v:.4g}" if isinstance(v, float) else str(v) for v in row
        )
        + " |"
        for row in frame.iter_rows()
    )
    return header + "\n".join(rows) + "\n"


def build_report(*, records: pl.DataFrame, days: Sequence[dt.date]) -> tuple[str, pl.DataFrame]:
    """Build the markdown report and the per-day table from the run records."""
    daily = per_day_table(records=records)
    anomalies = daily.filter(
        (pl.col("runs") != RUNS_PER_DAY)
        | (pl.col("unreadable") > 0)
        | (pl.col("absent_objects") > 0)
        | (pl.col("nan_share") > 0)
        | (pl.col("outside_range") > 0)
        | (pl.col("stuck_slices") > 0)
        | (pl.col("bad_time") > 0)
        | (pl.col("with_time_bounds") > 0)
    )
    sections = [
        "# UKV on AWS validation\n",
        f"Committed days read: {len(days)} ({days[0]} to {days[-1]}). "
        f"Run files read: {records.height}; unreadable: {int((~records['readable']).sum())}.\n"
        if days
        else "No committed day.\n",
        "## Consistency across runs\n",
        "\n".join(f"- {line}" for line in consistency_findings(records=records)) + "\n",
        "## Days without exactly 24 run files\n",
        "\n".join(f"- {line}" for line in runs_missing_per_day(records=records, days=days))
        or "None.",
        "\n## Days with any finding\n",
        f"{anomalies.height} of {daily.height} days. First {MAX_ANOMALY_ROWS}:\n",
        _markdown(frame=anomalies.head(MAX_ANOMALY_ROWS)),
        "## Radiation budget by month\n",
        "Daytime values where total shortwave exceeds 1 W m-2.\n",
        _markdown(frame=radiation_by_month(records=records)),
        "## Mean wind speed at the kept heights around the level change\n",
        "Columns 0 to 3 are 50, 75, 100, and 150 m.\n",
        _markdown(frame=level_transition(records=records)),
    ]
    return "\n".join(sections), daily


def main(argv: Sequence[str] | None = None) -> None:
    """Read the committed days and write a report to a new folder."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", maxsplit=1)[0])
    parser.add_argument("--max-days", type=int, help="read only the first N committed days")
    parser.add_argument(
        "--label", default=f"{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}", help="report folder name"
    )
    args = parser.parse_args(argv)
    days = committed_days(product_dir=PRODUCT_DIR)
    if args.max_days:
        days = days[: args.max_days]
    by_day: dict[str, list[Path]] = defaultdict(list)
    for path in sorted((PRODUCT_DIR / "runs").glob("*.npz")):
        by_day[path.stem[:8]].append(path)
    records = pl.DataFrame(
        [summarise_run(path=path) for day in days for path in by_day.get(f"{day:%Y%m%d}", [])],
        infer_schema_length=None,
    )
    if records.is_empty():
        message = "no committed day with run files to validate"
        raise SystemExit(message)
    report, daily = build_report(records=records, days=days)
    out_dir = REPORT_PARENT / args.label
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "report.md").write_text(report)
    daily.write_csv(out_dir / "per_day.csv")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
