"""Build the east, south, and west unit shapes that every solar shape of experiment 00077 combines.

Each shape is `min(1.15 * plane-of-array irradiance / 1000, 1)` for panels at a tilt of 25 degrees
facing east (90), south (180), or west (270), from the same Erbs split of global irradiance that
experiment 00046 uses. A solar shape is a linear combination of the three columns, so every
quantity built from it (climatology, anomaly, calibration slope) is built later from these tables
by `e00077_solar_inputs.py`, with no second pass over the ENS archive. This script writes under
`work/orient/`:

- `cams.parquet`: `site`, `time`, `uE`, `uS`, `uW`, from experiment 00039's frame (the CAMS
  irradiance of the six sites, reliable hours only).
- `ens_train/init=*.parquet`: for each daily 00:00 UTC ENS run of the training window, the mean
  over the 51 members of each orientation's shape, per site and valid time, with valid times up to
  `LAST_TRAINING_VALID`.
- `ens_val/init=*.parquet`: the same for each validation run of experiment 00064's row set.

    uv run python ml-experiments/00077-*/e00077_orient.py --part cams|train|val [--workers 6]
"""

import argparse
import time
from datetime import UTC, datetime, timedelta
from multiprocessing import Pool
from pathlib import Path
from typing import Final

import e00077_common as common
import numpy as np
import polars as pl
import pvlib
from ml_core.features._nwp import _upsample_nwp_to_half_hourly
from studies.pv_physics import PlantParameters, SunAndSky, plant_power_mw, solar_position
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ORIENTATION_AZIMUTHS: Final[dict[str, float]] = {"E": 90.0, "S": 180.0, "W": 270.0}
TILT_DEG: Final[float] = 25.0
DC_AC_RATIO: Final[float] = 1.15
ORIENT_DIR: Final[Path] = common.BASE_WORK_DIR / "orient"
TRAIN_DIR: Final[Path] = ORIENT_DIR / "ens_train"
VAL_DIR: Final[Path] = ORIENT_DIR / "ens_val"
ARCHIVE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR.parent / "NWP" / "nwp_model_id=ECMWF_ENS_0_25_degree"
)
SHORTWAVE_SOURCE: Final[str] = "downward_short_wave_radiation_flux_surface"
FIRST_TRAINING_RUN: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
HORIZON_HOURS: Final[int] = 351
PUBLICATION_HOURS: Final[int] = 9
COLUMNS: Final[tuple[str, ...]] = ("uE", "uS", "uW")
PLANTS: Final[dict[str, PlantParameters]] = {
    letter: PlantParameters(
        tilt_deg=TILT_DEG, azimuth_deg=azimuth, dc_capacity_mw=DC_AC_RATIO, ac_capacity_mw=1.0
    )
    for letter, azimuth in ORIENTATION_AZIMUTHS.items()
}


def unit_shapes(
    *, stamps: pl.Series, ghi: np.ndarray, latitude: float, longitude: float
) -> dict[str, np.ndarray]:
    """Return the unit shape of each orientation for half-hour mean irradiance.

    The sun is at the midpoint of the half-hour ending at each stamp, and the Erbs correlation
    splits the irradiance into beam and diffuse parts, as in experiment 00046.

    Args:
        stamps: Distinct period-ending UTC stamps, shape (n,).
        ghi: Half-hour mean global horizontal irradiance in W/m2, shape (k, n); NaN where unknown.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        `uE`, `uS`, and `uW`, each of shape (k, n), NaN where `ghi` is NaN.
    """
    copies = ghi.shape[0]
    midpoints = stamps.dt.offset_by("-15m")
    zenith, sun_azimuth = solar_position(stamps=midpoints, latitude=latitude, longitude=longitude)
    day = midpoints.dt.ordinal_day().to_numpy()
    unknown = ~np.isfinite(ghi)
    flat = np.clip(np.where(unknown, 0.0, ghi), 0.0, None).ravel().astype(np.float64)
    split = pvlib.irradiance.erbs(
        ghi=flat, zenith=np.tile(zenith, copies), datetime_or_doy=np.tile(day, copies)
    )
    sky = SunAndSky(
        half_hour_end_time=np.tile(stamps.to_numpy(), copies),
        ghi_w_m2=flat,
        dni_w_m2=np.nan_to_num(np.asarray(split["dni"]).astype(np.float64)),
        dhi_w_m2=np.nan_to_num(np.asarray(split["dhi"]).astype(np.float64)),
        zenith_deg=np.tile(zenith, copies),
        sun_azimuth_deg=np.tile(sun_azimuth, copies),
    )
    return {
        f"u{letter}": np.where(
            unknown, np.nan, plant_power_mw(sky=sky, parameters=plant).reshape(copies, -1)
        )
        for letter, plant in PLANTS.items()
    }


def cams_table(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return the three CAMS unit shapes of each site on experiment 00039's half-hour grid."""
    frame = (
        pl.read_parquet(common.FRAME_00039_PATH)
        .filter((pl.col("kind") == "substation") & pl.col("site").is_in(common.SITES))
        .select("site", "time", "ghi_u")
    )
    pieces = []
    for row in series.iter_rows(named=True):
        own = frame.filter(pl.col("site") == row["site"]).sort("time")
        ghi = own["ghi_u"].to_numpy().astype(np.float64)[None, :]
        shapes = unit_shapes(
            stamps=own["time"], ghi=ghi, latitude=row["latitude"], longitude=row["longitude"]
        )
        pieces.append(
            own.select("site", "time").with_columns(
                **{name: pl.Series(values[0]) for name, values in shapes.items()}
            )
        )
    return pl.concat(pieces)


def member_means(*, rows: pl.DataFrame, series_row: dict) -> pl.DataFrame:
    """Return the member-mean shape of each orientation at each valid time of one series and run.

    Args:
        rows: One series' rows of one run with `valid_time`, `ensemble_member`, and `shortwave`.
        series_row: The series' `latitude` and `longitude`.

    Returns:
        Columns `valid_time`, `uE`, `uS`, and `uW`.

    Raises:
        ValueError: If a member lacks a valid time.
    """
    ordered = rows.sort("valid_time", "ensemble_member")
    times = ordered["valid_time"].unique(maintain_order=True)
    members = ordered["ensemble_member"].unique().sort()
    if ordered.height != times.len() * members.len():
        msg = "every member must hold every valid time"
        raise ValueError(msg)
    ghi = ordered["shortwave"].to_numpy().astype(np.float64).reshape(times.len(), members.len()).T
    shapes = unit_shapes(
        stamps=times,
        ghi=ghi,
        latitude=series_row["latitude"],
        longitude=series_row["longitude"],
    )
    return pl.DataFrame(
        {"valid_time": times, **{n: v.mean(axis=0).astype(np.float32) for n, v in shapes.items()}}
    )


def read_run_shortwave(*, run: datetime, series: pl.DataFrame) -> pl.DataFrame:
    """Return one archive run's half-hourly `shortwave` for every member at every series' cell.

    Args:
        run: The 00:00 UTC initialisation time.
        series: `series_table()` filtered to the sites, with `h3_res_5`, `latitude`, `longitude`.

    Returns:
        Columns `KEYS`, `ensemble_member`, and `shortwave` (W/m2, half-hour mean ending at
        `valid_time`), valid times after the forecast origin (the run plus 9 hours) and at most
        351 hours later.
    """
    stamp = f"{run:%Y-%m-%d} 00:00:00.000000".replace(" ", "%20").replace(":", "%3A")
    native = (
        pl.scan_parquet(ARCHIVE_DIR / f"init_time={stamp}" / "*.parquet")
        .select("h3_index", "ensemble_member", "valid_time", SHORTWAVE_SOURCE)
        .filter(pl.col("h3_index").is_in(series["h3_res_5"].unique().to_list()))
        .join(
            series.lazy().select("time_series_id", "h3_res_5"),
            left_on="h3_index",
            right_on="h3_res_5",
            how="inner",
        )
        .drop("h3_index")
        .with_columns(nwp_init_time=pl.lit(run).cast(pl.Datetime("us", "UTC")))
    )
    coordinates = series.lazy().select("time_series_id", "latitude", "longitude")
    half_hourly = _upsample_nwp_to_half_hourly(native, solar_coordinates=coordinates).collect()
    origin = run + timedelta(hours=PUBLICATION_HOURS)
    return (
        half_hourly.rename({SHORTWAVE_SOURCE: "shortwave"})
        .with_columns(power_fcst_init_time=pl.lit(origin).cast(pl.Datetime("us", "UTC")))
        .filter(
            (pl.col("valid_time") > pl.col("power_fcst_init_time"))
            & (
                pl.col("valid_time")
                <= pl.col("power_fcst_init_time") + pl.duration(hours=HORIZON_HOURS)
            )
        )
        .select(*common.KEYS, "ensemble_member", "shortwave")
    )


def build_train_run(*, run: datetime, series: pl.DataFrame) -> pl.DataFrame:
    """Return one training run's member-mean shapes per site, training-window valid times only."""
    rows = read_run_shortwave(run=run, series=series)
    pieces = []
    for row in series.iter_rows(named=True):
        own = rows.filter(
            (pl.col("time_series_id") == row["time_series_id"])
            & (pl.col("valid_time") <= common.LAST_TRAINING_VALID)
        )
        if own.is_empty():
            continue
        pieces.append(
            member_means(rows=own, series_row=row).with_columns(
                site=pl.lit(row["site"]),
                power_fcst_init_time=pl.lit(run + timedelta(hours=PUBLICATION_HOURS)).cast(
                    pl.Datetime("us", "UTC")
                ),
            )
        )
    if not pieces:
        return pl.DataFrame()
    return pl.concat(pieces).select("site", "power_fcst_init_time", "valid_time", *COLUMNS)


def build_val_file(*, path: Path, series: pl.DataFrame) -> pl.DataFrame:
    """Return one validation run's member-mean shapes per site."""
    run = pl.read_parquet(path, columns=[*common.KEYS, "ensemble_member", "shortwave"]).filter(
        pl.col("time_series_id").is_in(series["time_series_id"])
    )
    pieces = []
    for row in series.iter_rows(named=True):
        own = run.filter(pl.col("time_series_id") == row["time_series_id"])
        pieces.append(
            member_means(rows=own, series_row=row).with_columns(
                site=pl.lit(row["site"]),
                power_fcst_init_time=pl.lit(own["power_fcst_init_time"][0]),
            )
        )
    return pl.concat(pieces).select("site", "power_fcst_init_time", "valid_time", *COLUMNS)


def _train_job(run: datetime) -> str:
    target = TRAIN_DIR / f"init={run:%Y-%m-%d}.parquet"
    if target.exists():
        return f"{target.name}: kept"
    started = time.monotonic()
    series = common.series_table().filter(pl.col("site").is_in(common.SITES))
    table = build_train_run(run=run, series=series)
    if table.is_empty():
        return f"{target.name}: no rows"
    common.write_atomically(frame=table, target=target)
    return f"{target.name}: {table.height:,} rows in {time.monotonic() - started:.1f} s"


def _val_job(path: Path) -> str:
    target = VAL_DIR / path.name
    if target.exists():
        return f"{target.name}: kept"
    started = time.monotonic()
    series = common.series_table().filter(pl.col("site").is_in(common.SITES))
    table = build_val_file(path=path, series=series)
    common.write_atomically(frame=table, target=target)
    return f"{target.name}: {table.height:,} rows in {time.monotonic() - started:.1f} s"


def main() -> None:
    """Build the CAMS table, the training runs, or the validation runs, in parallel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("cams", "train", "val"), required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None, help="only the first N jobs")
    args = parser.parse_args()
    series = common.series_table().filter(pl.col("site").is_in(common.SITES))
    if args.part == "cams":
        table = cams_table(series=series)
        common.write_atomically(frame=table, target=ORIENT_DIR / "cams.parquet")
        print(table.group_by("site").agg(pl.len(), pl.col("uS").quantile(0.99).alias("p99_uS")))
        return
    if args.part == "train":
        last = common.LAST_TRAINING_VALID.replace(hour=0, minute=0)
        runs = []
        run = FIRST_TRAINING_RUN
        while run <= last:
            runs.append(run)
            run += timedelta(days=1)
        jobs: list = runs
        function = _train_job
    else:
        jobs = sorted(common.VALIDATION_DIR.glob("init=*.parquet"))
        function = _val_job
    if args.limit is not None:
        jobs = jobs[: args.limit]
    with Pool(args.workers) as pool:
        for message in pool.imap_unordered(function, jobs):  # ty: ignore[invalid-argument-type]
            print(message, flush=True)


if __name__ == "__main__":
    main()
