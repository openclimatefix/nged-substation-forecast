r"""Build the GEFS solar inputs of experiment 00050 under `work/solar/`.

For each daily ECMWF ENS run date, the script reads the 00:00 UTC NOAA GEFS run (the same day's, or
the previous day's, as `e00050_latency.py` decided), turns its 31 members' surface shortwave
radiation into the unit PV shape `u` of experiment 00046, and writes the ensemble mean and the
control member:

1. **Area-weight the grid cells.** Each Primary's `h3_res_5` cell overlaps up to four 0.25-degree
   GEFS grid cells (`geo.h3.compute_h3_grid_weights`). The weights of the cells inside the
   downloaded crop are renormalised to sum to 1, and the share of the cell the crop covers is
   printed.
2. **De-accumulate.** GEFS radiation is a 3-hour mean at the 03, 09, 15, and 21 UTC lead times and a
   6-hour mean at the 00, 06, 12, and 18 UTC lead times (up to 240 hours; 6-hour means after that).
   The 3-hour mean ending at a 6-hourly label `L` is `2 * v(L) - v(L - 3 h)`. Means below zero are
   clipped to zero and counted.
3. **Put the series on ENS's native radiation schedule** (3-hour windows to 144 hours, 6-hour
   windows after), pass it through `ml_core.features._nwp._upsample_nwp_to_half_hourly`, which
   conserves each window's energy, and compute `u` per member and then average the members.

Outputs: `solar/gefs_train/init=<ENS run date>.parquet` (valid times up to 2025-06-30 23:30 UTC),
`solar/gefs_val/init=<date>.parquet` (the validation runs experiment 00046 scored), and, once the
training window is complete, `solar/gefs_climatology.parquet`. Columns: `site`,
`power_fcst_init_time` (the ENS run plus 9 hours), `valid_time`, `u_mean`, `u_control`.

    POLARS_MAX_THREADS=1 uv run python ml-experiments/00050-*/e00050_gefs_inputs.py \\
        [--split train|val|both] [--runs N] [--workers 4]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import e00050_common as common
import numpy as np
import polars as pl
from geo.h3 import compute_h3_grid_weights
from ml_core.features._nwp import _upsample_nwp_to_half_hourly

SHORTWAVE: Final[str] = common.GEFS_SHORTWAVE
KEYS_BY_MEMBER: Final[list[str]] = ["site", "ensemble_member"]


def grid_weights(
    *, series: pl.DataFrame, mirror: bool = False
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return each site's weight on each GEFS grid cell, and the share of the site's cell covered.

    Args:
        series: `series_table()` rows with `site` and `h3_res_5`.
        mirror: Flip the latitude index of the crop, for the grid-orientation check only.

    Returns:
        The weights (`site`, `lat_index`, `lon_index`, `weight`, renormalised over the cells the
        crop holds) and the coverage (`site`, `coverage`, the share of the H3 cell's area inside
        the crop before renormalising).

    Raises:
        ValueError: If a site has no cell inside the crop.
    """
    h3_weights = compute_h3_grid_weights(
        nwp_grid_size_degrees=common.GEFS_GRID_DEGREES, h3_index=series["h3_res_5"].to_list()
    )
    cells = pl.read_parquet(common.GEFS_GRID_PATH).with_columns(
        nwp_lat=pl.col("latitude").round(3), nwp_lon=pl.col("longitude").round(3)
    )
    if mirror:
        top = cells["lat_index"].max()
        assert isinstance(top, int)
        cells = cells.with_columns(lat_index=(top - pl.col("lat_index")).cast(pl.Int16))
    joined = (
        h3_weights.with_columns(
            h3_index=pl.col("h3_index").cast(pl.UInt64),
            nwp_lat=pl.col("nwp_lat").cast(pl.Float64).round(3),
            nwp_lon=pl.col("nwp_lon").cast(pl.Float64).round(3),
            proportion=pl.col("proportion").cast(pl.Float64),
        )
        .join(
            cells.select("nwp_lat", "nwp_lon", "lat_index", "lon_index"),
            on=["nwp_lat", "nwp_lon"],
            how="inner",
        )
        .join(series.select("site", h3_index=pl.col("h3_res_5")), on="h3_index", validate="m:1")
    )
    coverage = joined.group_by("site").agg(coverage=pl.col("proportion").sum())
    if coverage.height != series.height:
        msg = "every site needs at least one GEFS grid cell inside the crop"
        raise ValueError(msg)
    weights = joined.join(coverage, on="site").select(
        "site", "lat_index", "lon_index", weight=pl.col("proportion") / pl.col("coverage")
    )
    return weights, coverage.sort("site")


def three_hour_means(*, frame: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Return GEFS's 3-hour means, not yet clipped, at every lead time up to 240 hours.

    Args:
        frame: Columns `keys`, `lead_hours` (Int64, whole hours after the run's initialisation), and
            `value` (W/m2, the mean over the window ending at the lead time), one row per key and
            lead time.
        keys: The columns identifying one trajectory (a site and member, for example).

    Returns:
        The columns `keys`, `lead_hours`, and `mean3`. At a lead time that is 3 hours past a
        multiple of 6, `mean3` is the value itself. At a multiple of 6, `mean3` is
        `2 * v(L) - v(L - 3 h)`, which is null when the earlier value is missing.
    """
    native = frame.filter(pl.col("lead_hours") > 0)
    earlier = native.select(*keys, lead_hours=pl.col("lead_hours") + 3, previous=pl.col("value"))
    return (
        native.join(earlier, on=[*keys, "lead_hours"], how="left")
        .with_columns(
            mean3=pl.when(pl.col("lead_hours") % 6 == 3)
            .then(pl.col("value"))
            .when(pl.col("lead_hours") % 6 == 0)
            .then(2.0 * pl.col("value") - pl.col("previous"))
        )
        .filter(
            (pl.col("lead_hours") <= common.GEFS_STEP_CHANGE_HOURS) & pl.col("mean3").is_not_null()
        )
        .select(*keys, "lead_hours", "mean3")
    )


def clip_statistics(*, mean3: pl.Series) -> dict[str, float]:
    """Return how many 3-hour means fall below zero, and how much energy clipping them removes.

    Args:
        mean3: The unclipped 3-hour means in W/m2.

    Returns:
        `rows`, `nan` (NaN means), `below_minus_1` and `below_zero` (counts under those bounds),
        `minimum`, and `clipped_share` (the summed size of the negative means over the summed
        positive means).
    """
    finite = mean3.filter(mean3.is_not_nan())
    positive = float(finite.filter(finite > 0.0).sum())
    return {
        "rows": float(mean3.len()),
        "nan": float(mean3.is_nan().sum()),
        "below_minus_1": float((finite < common.MIN_PLAUSIBLE_MEAN_WM2).sum()),
        "below_zero": float((finite < 0.0).sum()),
        "minimum": common.scalar(value=finite.min()),
        "clipped_share": float(-finite.filter(finite < 0.0).sum()) / positive if positive else 0.0,
    }


def deaccumulate(*, frame: pl.DataFrame, keys: list[str]) -> tuple[pl.DataFrame, dict[str, float]]:
    """Turn GEFS window means into clipped means on ENS's native radiation schedule.

    Args:
        frame: Columns `keys`, `lead_hours`, and `value`, as for `three_hour_means`.
        keys: The columns identifying one trajectory.

    Returns:
        The table (`keys`, `lead_hours`, `value`) holding 3-hour means at leads up to 144 hours that
        are multiples of 3, 6-hour means (the mean of two 3-hour means) at multiples of 6 up to 240
        hours, and GEFS's own 6-hour means at multiples of 6 after that. Every value is at least 0.
        The statistics are those of `clip_statistics`, taken before the clip.
    """
    change = common.ENS_NATIVE_STEP_CHANGE_HOURS
    raw = three_hour_means(frame=frame, keys=keys)
    statistics = clip_statistics(mean3=raw["mean3"])
    clipped = raw.with_columns(mean3=pl.col("mean3").clip(lower_bound=0.0))
    previous_three = clipped.select(
        *keys, lead_hours=pl.col("lead_hours") + 3, mean3_previous=pl.col("mean3")
    )
    to_change = clipped.filter(
        (pl.col("lead_hours") <= change) & (pl.col("lead_hours") % 3 == 0)
    ).select(*keys, "lead_hours", value=pl.col("mean3"))
    paired = (
        clipped.filter((pl.col("lead_hours") > change) & (pl.col("lead_hours") % 6 == 0))
        .join(previous_three, on=[*keys, "lead_hours"], how="left")
        .select(*keys, "lead_hours", value=(pl.col("mean3") + pl.col("mean3_previous")) / 2.0)
    )
    beyond = frame.filter(
        (pl.col("lead_hours") > common.GEFS_STEP_CHANGE_HOURS) & (pl.col("lead_hours") % 6 == 0)
    ).select(*keys, "lead_hours", value=pl.col("value").clip(lower_bound=0.0))
    table = pl.concat([to_change, paired, beyond]).sort(*keys, "lead_hours")
    return table, statistics


def read_native_run(*, init: datetime, weights: pl.DataFrame) -> pl.DataFrame:
    """Return one GEFS run's area-weighted shortwave per site, member, and lead time.

    Args:
        init: The 00:00 UTC initialisation time of the GEFS run.
        weights: `grid_weights()` weights.

    Returns:
        Columns `site`, `ensemble_member`, `lead_hours` (Int64), and `value`; empty when the store
        holds no such run.
    """
    horizon = timedelta(hours=common.ENS_LAST_LEAD_HOURS)
    native = (
        pl.scan_parquet(common.GEFS_PATH)
        .filter(pl.col("init_time") == init.replace(tzinfo=None))
        .filter(pl.col("lead_time") <= horizon)
        .select("ensemble_member", "lead_time", "lat_index", "lon_index", SHORTWAVE)
        .collect()
    )
    if native.is_empty():
        return pl.DataFrame(schema={"site": pl.String, "ensemble_member": pl.Int8})
    return (
        native.join(weights, on=["lat_index", "lon_index"], how="inner")
        .group_by("site", "ensemble_member", "lead_time")
        .agg(value=(pl.col("weight") * pl.col(SHORTWAVE).cast(pl.Float64)).sum())
        .with_columns(lead_hours=(pl.col("lead_time").dt.total_hours()).cast(pl.Int64))
        .select("site", "ensemble_member", "lead_hours", "value")
    )


def half_hourly_ghi(
    *, run: date, series: pl.DataFrame, weights: pl.DataFrame, offset_days: int
) -> pl.DataFrame:
    """Return one ENS run date's GEFS irradiance per site and member on half-hourly valid times.

    Args:
        run: The ENS run date; the 00:00 UTC GEFS run `offset_days` earlier is read.
        series: `series_table()` rows with `latitude` and `longitude`.
        weights: `grid_weights()` weights.
        offset_days: 0 for the same day's GEFS run, 1 for the previous day's.

    Returns:
        Columns `site`, `ensemble_member`, `power_fcst_init_time` (the ENS run plus 9 hours),
        `valid_time` (half-hour ending, UTC), and `ghi` (W/m2), valid times after the origin and at
        most 351 hours later (a previous-day run stops at 360 hours after its own initialisation,
        the last lead ENS's radiation schedule allows, so it ends 24 hours sooner); empty when the
        GEFS run is missing.
    """
    ens_init = datetime(run.year, run.month, run.day, tzinfo=UTC)
    gefs_init = ens_init - timedelta(days=offset_days)
    native = read_native_run(init=gefs_init, weights=weights)
    if native.is_empty():
        return pl.DataFrame()
    schedule, _ = deaccumulate(frame=native, keys=KEYS_BY_MEMBER)
    schedule = schedule.filter(pl.col("lead_hours") <= common.ENS_LAST_LEAD_HOURS)
    origin = ens_init + timedelta(hours=9)
    ids = series.select("site", "time_series_id", "latitude", "longitude")
    lazy = (
        schedule.join(ids.select("site", "time_series_id"), on="site")
        .with_columns(
            valid_time=pl.lit(gefs_init).cast(pl.Datetime("us", "UTC"))
            + pl.duration(hours=pl.col("lead_hours")),
            nwp_init_time=pl.lit(gefs_init).cast(pl.Datetime("us", "UTC")),
        )
        .select(
            "time_series_id",
            "nwp_init_time",
            "ensemble_member",
            "valid_time",
            pl.col("value").cast(pl.Float32).alias(SHORTWAVE),
        )
        .lazy()
    )
    half_hourly = _upsample_nwp_to_half_hourly(
        lazy, solar_coordinates=ids.lazy().select("time_series_id", "latitude", "longitude")
    ).collect()
    return (
        half_hourly.join(ids.select("site", "time_series_id"), on="time_series_id")
        .with_columns(power_fcst_init_time=pl.lit(origin).cast(pl.Datetime("us", "UTC")))
        .filter(
            (pl.col("valid_time") > pl.col("power_fcst_init_time"))
            & (
                pl.col("valid_time")
                <= pl.col("power_fcst_init_time") + pl.duration(hours=common.HORIZON_HOURS)
            )
        )
        .select(
            "site",
            "ensemble_member",
            "power_fcst_init_time",
            "valid_time",
            ghi=pl.col(SHORTWAVE).cast(pl.Float64),
        )
    )


def member_u(*, rows: pl.DataFrame, latitude: float, longitude: float) -> pl.DataFrame:
    """Return the `u` of every member and valid time of one site and one run.

    Args:
        rows: One site's rows of one run with `valid_time`, `ensemble_member`, and `ghi`.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        Columns `valid_time`, `ensemble_member`, and `u`, sorted by valid time then member.

    Raises:
        ValueError: If a member lacks a valid time.
    """
    ordered = rows.sort("valid_time", "ensemble_member")
    times = ordered["valid_time"].unique(maintain_order=True)
    members = ordered["ensemble_member"].unique().sort()
    if ordered.height != times.len() * members.len():
        msg = "every member must hold every valid time"
        raise ValueError(msg)
    ghi = ordered["ghi"].to_numpy().astype(np.float64).reshape(times.len(), members.len()).T
    u = common.u_from_half_hour_ghi(stamps=times, ghi=ghi, latitude=latitude, longitude=longitude)
    return pl.DataFrame(
        {
            "valid_time": np.repeat(times.to_numpy(), members.len()),
            "ensemble_member": np.tile(members.to_numpy(), times.len()),
            "u": u.T.ravel().astype(np.float32),
        }
    ).with_columns(pl.col("valid_time").dt.replace_time_zone("UTC"))


def build_run(
    *, run: date, series: pl.DataFrame, weights: pl.DataFrame, offset_days: int, training: bool
) -> pl.DataFrame:
    """Return one run's GEFS ensemble-mean and control-member `u` per site.

    Args:
        run: The ENS run date.
        series: `series_table()` rows with `latitude` and `longitude`.
        weights: `grid_weights()` weights.
        offset_days: 0 for the same day's GEFS run, 1 for the previous day's.
        training: Keep only valid times up to the last training valid time when true.

    Returns:
        Columns `site`, `power_fcst_init_time`, `valid_time`, `u_mean`, and `u_control`; empty when
        the GEFS run is missing.
    """
    ghi = half_hourly_ghi(run=run, series=series, weights=weights, offset_days=offset_days)
    if ghi.is_empty():
        return pl.DataFrame()
    if training:
        ghi = ghi.filter(pl.col("valid_time") <= common.LAST_TRAINING_VALID)
    pieces = []
    for row in series.iter_rows(named=True):
        own = ghi.filter(pl.col("site") == row["site"])
        if own.is_empty():
            continue
        u = member_u(rows=own, latitude=row["latitude"], longitude=row["longitude"])
        pieces.append(
            u.group_by("valid_time")
            .agg(
                u_mean=pl.col("u").mean(),
                u_control=pl.col("u").filter(pl.col("ensemble_member") == 0).first(),
            )
            .with_columns(
                site=pl.lit(row["site"]),
                power_fcst_init_time=pl.lit(
                    datetime(run.year, run.month, run.day, tzinfo=UTC) + timedelta(hours=9)
                ).cast(pl.Datetime("us", "UTC")),
            )
            .select("site", "power_fcst_init_time", "valid_time", "u_mean", "u_control")
        )
    return pl.concat(pieces) if pieces else pl.DataFrame()


def _build_one(args: tuple[date, Path, int, bool]) -> tuple[str, int, float]:
    """Build and write one run's file in a worker process; skip it when the file exists."""
    run, target, offset_days, training = args
    if target.exists():
        return target.name, -1, 0.0
    started = time.monotonic()
    series = common.series_table().filter(pl.col("site").is_in(common.SITES))
    weights, _ = grid_weights(series=series)
    table = build_run(
        run=run, series=series, weights=weights, offset_days=offset_days, training=training
    )
    if table.is_empty():
        return target.name, 0, time.monotonic() - started
    common.write_atomically(frame=table, target=target)
    return target.name, table.height, time.monotonic() - started


def run_dates(*, split: str, limit: int | None) -> list[tuple[date, Path, bool]]:
    """Return the ENS run dates of a split with each one's output path and training flag."""
    jobs: list[tuple[date, Path, bool]] = []
    if split in ("train", "both"):
        day = common.FIRST_TRAINING_RUN.date()
        last = common.LAST_TRAINING_VALID.date()
        training: list[tuple[date, Path, bool]] = []
        while day <= last:
            training.append((day, common.SOLAR_DIR_TRAIN / f"init={day:%Y-%m-%d}.parquet", True))
            day += timedelta(days=1)
        jobs += training[:limit] if limit is not None else training
    if split in ("val", "both"):
        validation = [
            (
                date.fromisoformat(path.stem.removeprefix("init=")),
                common.SOLAR_DIR_VAL / path.name,
                False,
            )
            for path in sorted((common.SOURCE_SOLAR_DIR / "ens_val").glob("init=*.parquet"))
        ]
        jobs += validation[:limit] if limit is not None else validation
    return jobs


def require_complete(*, split: str) -> None:
    """Raise, naming the dates, when a split lacks the input file of any ENS run date.

    A run date has no file when the GEFS store holds no run for it, so a gap in the archive stops
    the experiment here and not as a quietly smaller set of rows later.

    Args:
        split: `train` or `val`.

    Raises:
        FileNotFoundError: If any run date of the split has no file.
    """
    missing = [run for run, target, _ in run_dates(split=split, limit=None) if not target.exists()]
    if missing:
        shown = ", ".join(f"{run:%Y-%m-%d}" for run in missing[:10])
        msg = (
            f"{len(missing)} {split} run dates have no GEFS input file (first: {shown}); "
            "build them with e00050_gefs_inputs.py, and if the archive lacks the GEFS run for "
            "a date, stop and decide how to treat that date"
        )
        raise FileNotFoundError(msg)


def write_climatology() -> None:
    """Write the GEFS month-by-half-hour climatology of `u` over the full training window.

    Neighbouring months are pooled, as for experiment 00046's ENS and CAMS climatologies, whose ENS
    cells each hold more than 62 distinct days (the check below).
    """
    source = pl.read_parquet(common.SOURCE_SOLAR_DIR / "climatology.parquet")
    daylight_cells = source.filter(pl.col("daylight"))
    assert common.scalar(value=daylight_cells["days_ens"].min()) > 62, (
        "experiment 00046 pooled its months"
    )
    require_complete(split="train")
    files = sorted(common.SOLAR_DIR_TRAIN.glob("init=*.parquet"))
    train = pl.concat([pl.read_parquet(path) for path in files])
    clim = common.climatology(
        frame=train.select("site", time=pl.col("valid_time"), u=pl.col("u_mean")),
        value="u",
        time_column="time",
        pool=True,
    )
    common.write_atomically(
        frame=clim.rename({"clim": "clim_gefs", "days": "days_gefs"}),
        target=common.SOLAR_DIR / "gefs_climatology.parquet",
    )
    print(f"wrote the GEFS climatology from {len(files)} runs")


def main() -> None:
    """Build the requested GEFS input files, four worker processes at a time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument("--runs", type=int, default=None, help="only the first N runs of a split")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    common.SOLAR_DIR.mkdir(parents=True, exist_ok=True)
    offset_days = common.gefs_run_offset_days()
    series = common.series_table().filter(pl.col("site").is_in(common.SITES))
    _, coverage = grid_weights(series=series)
    print("share of each site's H3 cell inside the GEFS crop (weights are renormalised):")
    print(coverage)
    jobs = [
        (run, target, offset_days, training)
        for run, target, training in run_dates(split=args.split, limit=args.runs)
    ]
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, rows, seconds in pool.map(_build_one, jobs):
            done += 1
            note = "exists" if rows < 0 else f"{rows:,} rows in {seconds:.1f} s"
            print(f"[{done}/{len(jobs)}] {name}: {note}", flush=True)
    if args.split in ("train", "both") and args.runs is None:
        write_climatology()


if __name__ == "__main__":
    main()
