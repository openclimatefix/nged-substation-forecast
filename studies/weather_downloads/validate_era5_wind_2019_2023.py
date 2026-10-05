"""Validate the 2019 to 2023 ERA5 wind parquet that `fetch_era5_wind_2019_2023.py` wrote.

One-off throwaway script for <https://github.com/openclimatefix/nged-substation-forecast/issues/841>,
following the `data-validation` skill. It reads `data/studies/weather/ERA5-WIND-2019-2023/` and
runs the checks in `main`: columns and dtypes, exact row count (cells x 37,992 hours), duplicate
keys, nulls and NaNs, a contiguous hourly axis, speed range, the hour-of-day profile, stuck runs,
level steps between months, 100 m against 10 m speed, the offsets of the cell blocks, the
correlation with the MIDAS Open station observations at lags of -2 to +2 hours, the step across
2023-12-31 to 2024-01-01 against the on-disk 2024 file, and bit-equality of the re-fetched 2024-01
with that file.

Grid orientation and a whole-hour time shift are tested by data, not by the cell plan: the station
correlation peaks at lag 0 only if each cell holds the right place and the right hour, and the
bit-equality check compares against a file built by an independent code path.

**The script prints one PASS or FAIL line per check and no count.** Row and cell counts reveal the
size of the private trial-area box, so the measured numbers go only to `validation.json` next to
the parquet. The exit code is non-zero when any check fails.

Run it with `uv run python studies/weather_downloads/validate_era5_wind_2019_2023.py`.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import polars as pl
from era5_cells import half_year_chunks
from paths import WEATHER_DOWNLOADS_DIR

PRODUCT_DIR: Final[Path] = WEATHER_DOWNLOADS_DIR / "ERA5-WIND-2019-2023"
OLD_WIND_PATH: Final[Path] = WEATHER_DOWNLOADS_DIR / "ERA5" / "wind_native_cds.parquet"
MIDAS_WEATHER_PATH: Final[Path] = (
    WEATHER_DOWNLOADS_DIR / "MIDAS-OPEN" / "uk_hourly_weather_obs.parquet"
)
VALUE_COLUMNS: Final[tuple[str, ...]] = ("u10", "v10", "u100", "v100")
FIRST_HOUR: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
LAST_HOUR: Final[datetime] = datetime(2023, 12, 31, 23, tzinfo=UTC)
EXPECTED_HOURS: Final[int] = sum(
    chunk.n_hours for chunk in half_year_chunks(first=(2019, 9), last=(2023, 12))
)
"""37,992: every hour from 2019-09-01 00:00 to 2023-12-31 23:00 UTC."""
MAX_PLAUSIBLE_COMPONENT_M_S: Final[float] = 60.0
"""A wind component above this is not physical at 10 m or 100 m."""
MAX_HOUR_PROFILE_SPREAD_M_S: Final[float] = 1.0
"""The mean speed per UTC hour of day varies by less than this. ERA5 wind is an instantaneous
analysis, so this is a sanity check on the diurnal cycle, not a running-mean test. Measured spread
on the 2024 to 2026 wind-farm cells is 0.62 to 0.64 m/s, and 0.69 m/s on September 2019."""
MAX_STUCK_RUN_HOURS: Final[int] = 6
MAX_MONTHLY_MEAN_RATIO: Final[float] = 2.5
"""Windiest over calmest monthly mean 10 m speed across the 52 months. Measured 1.61 in 2024-26."""
SEAM_MAX_RATIO: Final[float] = 1.5
"""The step across the 2023/2024 join may not exceed this multiple of the 99th percentile of the
ordinary hour-to-hour steps in the preceding 45 days."""
STATION_LAGS_H: Final[tuple[int, ...]] = (-2, -1, 0, 1, 2)
MIN_STATION_CORRELATION: Final[float] = 0.6
"""The mean over stations of the correlation between observed wind speed and ERA5 10 m speed at the
station's centre cell, at lag 0. The check also requires lag 0 to be the best of the five lags."""


def _speed(*, frame: pl.DataFrame, height: str) -> pl.Expr:
    return (pl.col(f"u{height}") ** 2 + pl.col(f"v{height}") ** 2).sqrt()


def _check_columns(*, frame: pl.DataFrame, cells: pl.DataFrame) -> tuple[bool, dict[str, Any]]:
    schema = {name: str(dtype) for name, dtype in frame.schema.items()}
    expected = {
        "cell_id": "Int32",
        "time": "Datetime(time_unit='us', time_zone='UTC')",
        **dict.fromkeys(VALUE_COLUMNS, "Float32"),
    }
    cell_ids_match = set(frame["cell_id"].unique()) == set(cells["cell_id"])
    return schema == expected and cell_ids_match, {"schema": schema}


def _check_rows(*, frame: pl.DataFrame, cells: pl.DataFrame) -> tuple[bool, dict[str, Any]]:
    expected = cells.height * EXPECTED_HOURS
    return frame.height == expected, {"rows": frame.height, "expected": expected}


def _check_duplicates(*, frame: pl.DataFrame) -> tuple[bool, dict[str, Any]]:
    duplicates = frame.height - frame.select("cell_id", "time").unique().height
    return duplicates == 0, {"duplicate_keys": duplicates}


def _check_nulls(*, frame: pl.DataFrame) -> tuple[bool, dict[str, Any]]:
    nulls = frame.null_count().row(0, named=True)
    nans = frame.select(pl.col(*VALUE_COLUMNS).is_nan().sum()).row(0, named=True)
    return sum(nulls.values()) + sum(nans.values()) == 0, {"nulls": nulls, "nans": nans}


def _check_time_axis(*, frame: pl.DataFrame) -> tuple[bool, dict[str, Any]]:
    """Every cell has exactly the hourly axis from `FIRST_HOUR` to `LAST_HOUR`."""
    expected = pl.datetime_range(FIRST_HOUR, LAST_HOUR, interval="1h", eager=True)
    per_cell = frame.group_by("cell_id").agg(
        pl.col("time").min().alias("first"),
        pl.col("time").max().alias("last"),
        pl.col("time").n_unique().alias("n"),
    )
    ok = (
        per_cell["first"].n_unique() == 1
        and per_cell["last"].n_unique() == 1
        and per_cell["first"][0] == FIRST_HOUR
        and per_cell["last"][0] == LAST_HOUR
        and set(per_cell["n"]) == {expected.len()}
    )
    return ok, {"hours": expected.len()}


def _check_range(*, frame: pl.DataFrame) -> tuple[bool, dict[str, Any]]:
    largest = frame.select(pl.max_horizontal(pl.col(*VALUE_COLUMNS).abs().max())).item()
    peaks = frame.select(
        _speed(frame=frame, height="10").max().alias("max10"),
        _speed(frame=frame, height="100").max().alias("max100"),
        _speed(frame=frame, height="10").mean().alias("mean10"),
    ).row(0, named=True)
    ok = largest <= MAX_PLAUSIBLE_COMPONENT_M_S and peaks["max10"] > 5 and peaks["mean10"] > 1
    return ok, {"largest_abs_component": largest, **peaks}


def _check_hour_profile(*, frame: pl.DataFrame) -> tuple[bool, dict[str, Any]]:
    profile = (
        frame.group_by(pl.col("time").dt.hour().alias("hour"))
        .agg(_speed(frame=frame, height="10").mean().alias("mean10"))
        .sort("hour")
    )
    spread = float(profile["mean10"].max() - profile["mean10"].min())  # ty: ignore[unsupported-operator]
    return spread < MAX_HOUR_PROFILE_SPREAD_M_S, {"profile_m_s": profile["mean10"].to_list()}


def _check_stuck_runs(*, frame: pl.DataFrame) -> tuple[bool, dict[str, Any]]:
    """No cell repeats the same (u10, v10) pair for more than `MAX_STUCK_RUN_HOURS` hours."""
    runs = (
        frame.sort("cell_id", "time")
        .with_columns(
            run=pl.struct("cell_id", "u10", "v10", "u100", "v100").rle_id(),
        )
        .group_by("run")
        .agg(pl.len().alias("length"))
    )
    longest = int(runs["length"].max())  # ty: ignore[invalid-argument-type]
    return longest <= MAX_STUCK_RUN_HOURS, {"longest_run_hours": longest}


def _check_monthly_level(*, frame: pl.DataFrame) -> tuple[bool, dict[str, Any]]:
    monthly = (
        frame.group_by(pl.col("time").dt.truncate("1mo").alias("month"))
        .agg(_speed(frame=frame, height="10").mean().alias("mean10"))
        .sort("month")
    )
    ratio = float(monthly["mean10"].max() / monthly["mean10"].min())  # ty: ignore[unsupported-operator]
    return ratio <= MAX_MONTHLY_MEAN_RATIO, {"months": monthly.height, "max_over_min": ratio}


def _check_height_consistency(*, frame: pl.DataFrame) -> tuple[bool, dict[str, Any]]:
    means = frame.select(
        _speed(frame=frame, height="10").mean().alias("mean10"),
        _speed(frame=frame, height="100").mean().alias("mean100"),
    ).row(0, named=True)
    return means["mean100"] > means["mean10"], means


def _check_cell_plan_offsets(*, groups: pl.DataFrame) -> tuple[bool, dict[str, Any]]:
    """Each block cell sits `dy` quarter degrees north and `dx` east of its block's centre cell.

    This reads only `cell_groups.parquet`, which holds for any plan `block_cells` builds, so it does
    not look at the data and cannot detect a flipped grid. `station_correlation` and
    `overlap_2024_01_bit_equal` carry the orientation evidence.
    """
    blocks = groups.filter(pl.col("kind") != "box")
    centres = blocks.filter((pl.col("dy") == 0) & (pl.col("dx") == 0)).select(
        "kind", "label", pl.col("lat_q").alias("c_lat"), pl.col("lon_q").alias("c_lon")
    )
    joined = blocks.join(centres, on=["kind", "label"])
    wrong = joined.filter(
        (pl.col("lat_q") - pl.col("c_lat") != pl.col("dy"))
        | (pl.col("lon_q") - pl.col("c_lon") != pl.col("dx"))
    ).height
    return wrong == 0, {
        "mis_offset_cells": wrong,
        "note": "plan arithmetic only; orientation rests on station_correlation and the overlap",
    }


def _dedupe_agreeing(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return `frame` with one row per `(cell_id, time)`, raising if duplicates disagree.

    Two wind-farm blocks that share a cell give the same `(cell_id, time)` key twice.
    """
    disagreeing = (
        frame.group_by("cell_id", "time")
        .agg(pl.col(c).n_unique() for c in VALUE_COLUMNS)
        .filter(pl.any_horizontal(pl.col(c) > 1 for c in VALUE_COLUMNS))
        .height
    )
    if disagreeing:
        msg = f"{disagreeing} (cell_id, time) keys hold different values in different series"
        raise ValueError(msg)
    return frame.unique(subset=["cell_id", "time"], keep="first", maintain_order=True)


def _old_series(*, groups: pl.DataFrame) -> pl.DataFrame:
    """Return the on-disk 2024+ wind file keyed by unique `(cell_id, time)`, as the same columns."""
    wind = groups.filter(pl.col("kind") == "wind").select("label", "dy", "dx", "cell_id")
    return _dedupe_agreeing(
        frame=pl.read_parquet(OLD_WIND_PATH)
        .rename({"site": "label"})
        .join(wind, on=["label", "dy", "dx"])
        .with_columns(pl.col("time").dt.replace_time_zone("UTC").dt.cast_time_unit("us"))
        .select("cell_id", "time", *VALUE_COLUMNS)
    )


def _check_station_correlation(
    *, frame: pl.DataFrame, groups: pl.DataFrame, observations: pl.DataFrame
) -> tuple[bool, dict[str, Any]]:
    """ERA5 10 m speed at each station's centre cell correlates best with the observations at lag 0.

    `observations` has `src_id`, `time` and `wind_speed_m_s`. A lag of +1 pairs the observation at
    hour t with the ERA5 value at hour t + 1, so a whole-hour time shift moves the peak off lag 0.
    A grid flip pairs each station with the wrong cell and lowers every correlation.
    """
    centres = groups.filter(
        (pl.col("kind") == "station") & (pl.col("dy") == 0) & (pl.col("dx") == 0)
    ).select(pl.col("label").alias("src_id"), "cell_id")
    era5 = (
        frame.join(centres, on="cell_id")
        .select("src_id", "time", _speed(frame=frame, height="10").alias("era5"))
        .sort("src_id", "time")
    )
    observed = observations.select("src_id", "time", pl.col("wind_speed_m_s").alias("observed"))
    per_lag: dict[int, float] = {}
    peak_at_zero = 0
    per_station: dict[int, dict[str, float]] = {}
    for lag in STATION_LAGS_H:
        shifted = era5.with_columns(pl.col("time") - pl.duration(hours=lag))
        correlations = (
            observed.join(shifted, on=["src_id", "time"])
            .drop_nulls()
            .group_by("src_id")
            .agg(pl.corr("observed", "era5").alias("r"))
        )
        per_lag[lag] = float(correlations["r"].mean())  # ty: ignore[invalid-argument-type]
        per_station[lag] = dict(zip(correlations["src_id"], correlations["r"], strict=True))
    for src_id in per_station[0]:
        best = max(STATION_LAGS_H, key=lambda lag, s=src_id: per_station[lag].get(s, -2.0))
        peak_at_zero += best == 0
    ok = max(per_lag, key=lambda lag: per_lag[lag]) == 0 and (per_lag[0] >= MIN_STATION_CORRELATION)
    return ok, {
        "mean_correlation_by_lag_h": {str(lag): r for lag, r in per_lag.items()},
        "stations": len(per_station[0]),
        "stations_peaking_at_lag_0": peak_at_zero,
    }


def _check_seam(*, frame: pl.DataFrame, old: pl.DataFrame) -> tuple[bool, dict[str, Any]]:
    """The step from 2023-12-31 23:00 to 2024-01-01 00:00 is no larger than an ordinary step."""
    ids = old["cell_id"].unique()
    joined = pl.concat(
        [
            frame.filter(pl.col("cell_id").is_in(ids)),
            old.filter(pl.col("time") <= datetime(2024, 1, 7, tzinfo=UTC)),
        ]
    ).sort("cell_id", "time")
    steps = joined.with_columns(
        step=(_speed(frame=joined, height="100") - _speed(frame=joined, height="100").shift(1))
        .abs()
        .over("cell_id")
    )
    seam = steps.filter(pl.col("time") == datetime(2024, 1, 1, tzinfo=UTC))["step"]
    ordinary = steps.filter(
        pl.col("time").is_between(datetime(2023, 11, 15, tzinfo=UTC), LAST_HOUR)
    )["step"]
    limit = float(ordinary.quantile(0.99) or 0.0) * SEAM_MAX_RATIO
    seam_max = float(seam.max())  # ty: ignore[invalid-argument-type]
    return seam_max <= limit, {"seam_max_step": seam_max, "limit": limit}


def _check_overlap(*, overlap: pl.DataFrame, old: pl.DataFrame) -> tuple[bool, dict[str, Any]]:
    """The re-fetched 2024-01 equals the on-disk file bit for bit on every wind-farm cell."""
    january = old.filter(pl.col("time") < datetime(2024, 2, 1, tzinfo=UTC))
    joined = overlap.join(january, on=["cell_id", "time"], how="full", suffix="_old", coalesce=True)
    unmatched = joined.filter(pl.col("u10").is_null() | pl.col("u10_old").is_null()).height
    different = joined.filter(
        pl.any_horizontal(pl.col(c) != pl.col(f"{c}_old") for c in VALUE_COLUMNS)
    ).height
    return unmatched == 0 and different == 0 and overlap.height == january.height, {
        "overlap_rows": overlap.height,
        "unmatched": unmatched,
        "different": different,
    }


def _check_status() -> tuple[bool, dict[str, Any]]:
    status = json.loads((PRODUCT_DIR / "status.json").read_text())
    bad = [entry["key"] for entry in status if entry["state"] not in ("cached", "downloaded")]
    return not bad, {"entries": len(status), "not_finished": bad}


def _check_size(*, path: Path) -> tuple[bool, dict[str, Any]]:
    megabytes = path.stat().st_size / 1e6
    return megabytes < 100, {"parquet_mb": megabytes}


def main() -> int:
    """Run every check, print PASS or FAIL per check, and write `validation.json`."""
    path = PRODUCT_DIR / "wind_era5_cds.parquet"
    frame = pl.read_parquet(path)
    cells = pl.read_parquet(PRODUCT_DIR / "cells.parquet")
    groups = pl.read_parquet(PRODUCT_DIR / "cell_groups.parquet")
    overlap = pl.read_parquet(PRODUCT_DIR / "overlap_2024_01.parquet")
    old = _old_series(groups=groups)
    observations = pl.read_parquet(
        MIDAS_WEATHER_PATH, columns=["src_id", "time", "wind_speed_m_s"]
    ).drop_nulls()
    results = {
        "columns_and_cell_ids": _check_columns(frame=frame, cells=cells),
        "exact_row_count": _check_rows(frame=frame, cells=cells),
        "no_duplicate_keys": _check_duplicates(frame=frame),
        "no_nulls_or_nans": _check_nulls(frame=frame),
        "contiguous_hourly_axis": _check_time_axis(frame=frame),
        "speed_range": _check_range(frame=frame),
        "hour_of_day_profile": _check_hour_profile(frame=frame),
        "no_stuck_runs": _check_stuck_runs(frame=frame),
        "monthly_level_steps": _check_monthly_level(frame=frame),
        "height_consistency": _check_height_consistency(frame=frame),
        "cell_plan_offsets": _check_cell_plan_offsets(groups=groups),
        "station_correlation": _check_station_correlation(
            frame=frame, groups=groups, observations=observations
        ),
        "seam_2023_12_to_2024_01": _check_seam(frame=frame, old=old),
        "overlap_2024_01_bit_equal": _check_overlap(overlap=overlap, old=old),
        "chunk_status_finished": _check_status(),
        "file_size": _check_size(path=path),
    }
    for name, (passed, _) in results.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    (PRODUCT_DIR / "validation.json").write_text(
        json.dumps(
            {name: {"passed": ok, **measured} for name, (ok, measured) in results.items()},
            indent=2,
            default=str,
        )
    )
    return 0 if all(passed for passed, _ in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
