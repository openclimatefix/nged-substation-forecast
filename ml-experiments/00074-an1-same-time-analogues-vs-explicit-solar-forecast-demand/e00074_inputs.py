"""Build the solar inputs of every P-roll training row for experiment 00074, under `work/inputs/`.

Experiment 00064 built the calibrated sunshine shape of each training row up to 2025-06-30, because
its model was frozen on that window. P-roll slides the window to 2026-06, so this script rebuilds
the inputs for the whole window with the same chain and with experiment 00046's climatology and
calibration slopes, which stay fixed at their 2025-06-30 values:

- `post_u/init=*.parquet`: for each daily 00:00 UTC ENS run from 2025-07-01 to 2026-06-29, the
  ensemble-mean ENS unit PV shape `u_mean` of each series at valid times up to the last refit's
  cutoff (`LAST_NEEDED_VALID`).
- `train_shape.parquet`: for each training row (series, run, valid time) of experiment 00046's
  `ens_train` files and the files above, the ensemble-mean calibrated anomaly, the CAMS climatology,
  their sum `shape_calibrated`, the month, the half-hour, and the lead band.
- `placebo_train.parquet`: the placebo twin of `anomaly_mean_calibrated`, a seeded cyclic shift
  within series, local month, half-hour, and lead band, over the whole window's rows.

`--check` compares the rows of `train_shape.parquet` up to 2025-06-30 with experiment 00064's own
table; they must agree.

    uv run python ml-experiments/00074-*/e00074_inputs.py --part post_u
    uv run python ml-experiments/00074-*/e00074_inputs.py --part shape [--check]
"""

import argparse
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00074_common as common
import numpy as np
import polars as pl
import pvlib
from ml_core.features._nwp import _upsample_nwp_to_half_hourly
from studies.pv_physics import SunAndSky, plant_power_mw, solar_position
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ARCHIVE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR.parent / "NWP" / "nwp_model_id=ECMWF_ENS_0_25_degree"
)
SHORTWAVE_SOURCE: Final[str] = "downward_short_wave_radiation_flux_surface"
FIRST_POST_RUN: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
LAST_POST_RUN: Final[datetime] = datetime(2026, 6, 29, tzinfo=UTC)
LAST_NEEDED_VALID: Final[datetime] = datetime(2026, 6, 1, tzinfo=UTC)
"""The latest refit time; no refit sees a valid time later than this less 3.5 hours."""
HORIZON_HOURS: Final[int] = 351
POST_DIR: Final[Path] = common.INPUTS_DIR / "post_u"
PUBLICATION_HOURS: Final[int] = 9
CHECK_TOLERANCE: Final[float] = 1e-6


def sites_table() -> pl.DataFrame:
    """Return the verdict series' rows of `series_table()`."""
    return common.series_table().filter(pl.col("site").is_in(list(common.SITES)))


def u_from_half_hour_ghi(
    *, stamps: pl.Series, ghi: np.ndarray, latitude: float, longitude: float
) -> np.ndarray:
    """Return the unit PV shape of half-hour mean irradiance, the chain CAMS goes through.

    The sun is at the midpoint of the half-hour ending at each stamp, the Erbs correlation splits
    the irradiance, and the generic fleet turns it into power per MW of AC capacity.

    Args:
        stamps: Distinct period-ending UTC stamps, shape (n,).
        ghi: Half-hour mean global horizontal irradiance in W/m2, shape (k, n); NaN where unknown.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        `u`, shape (k, n), NaN where `ghi` is NaN.
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
    shape = plant_power_mw(sky=sky, parameters=common.FLEET).reshape(copies, -1)
    return np.where(unknown, np.nan, shape)


def read_run_shortwave(*, run: datetime, series: pl.DataFrame) -> pl.DataFrame:
    """Return one archive run's half-hourly `shortwave` for every member at every series' cell.

    Args:
        run: The 00:00 UTC initialisation time.
        series: `sites_table()`, with `h3_res_5`, `latitude`, `longitude`.

    Returns:
        Columns `time_series_id`, `power_fcst_init_time`, `valid_time`, `ensemble_member`, and
        `shortwave` (W/m2, half-hour mean ending at `valid_time`), for valid times after the
        forecast origin (the run plus 9 hours) and at most 351 hours later.
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


def member_mean_u(*, rows: pl.DataFrame, series_row: dict) -> pl.DataFrame:
    """Return the ensemble-mean ENS `u` of every valid time of one series and one run.

    Args:
        rows: One series' rows of one run with `valid_time`, `ensemble_member`, and `shortwave`,
            every member holding every valid time.
        series_row: The series' `latitude` and `longitude`.

    Returns:
        Columns `valid_time` and `u_mean`, sorted by valid time.

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
    u = u_from_half_hour_ghi(
        stamps=times,
        ghi=ghi,
        latitude=series_row["latitude"],
        longitude=series_row["longitude"],
    )
    return pl.DataFrame({"valid_time": times, "u_mean": u.mean(axis=0).astype(np.float32)})


def build_post_run(*, run: datetime, series: pl.DataFrame) -> pl.DataFrame:
    """Return one post-window run's ensemble-mean `u` per site, up to `LAST_NEEDED_VALID`."""
    rows = read_run_shortwave(run=run, series=series)
    pieces = []
    for row in series.iter_rows(named=True):
        own = rows.filter(
            (pl.col("time_series_id") == row["time_series_id"])
            & (pl.col("valid_time") <= LAST_NEEDED_VALID)
        )
        if own.is_empty():
            continue
        pieces.append(
            member_mean_u(rows=own, series_row=row).with_columns(
                site=pl.lit(row["site"]),
                power_fcst_init_time=pl.lit(run + timedelta(hours=PUBLICATION_HOURS)).cast(
                    pl.Datetime("us", "UTC")
                ),
            )
        )
    if not pieces:
        return pl.DataFrame()
    return pl.concat(pieces).select("site", "power_fcst_init_time", "valid_time", "u_mean")


def build_post() -> None:
    """Build (or resume) the post-window ensemble-mean `u` files, one per daily run."""
    series = sites_table()
    run = FIRST_POST_RUN
    while run <= LAST_POST_RUN:
        target = POST_DIR / f"init={run:%Y-%m-%d}.parquet"
        if not target.exists():
            started = time.monotonic()
            table = build_post_run(run=run, series=series)
            if not table.is_empty():
                common.write_atomically(frame=table, target=target)
                print(
                    f"{target.name}: {table.height:,} rows in {time.monotonic() - started:.1f} s",
                    flush=True,
                )
        run += timedelta(days=1)


def shape_table(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return the calibrated ENS shape of every training row of the P-roll window.

    Args:
        series: `sites_table()`.

    Returns:
        Columns `time_series_id`, `site`, `power_fcst_init_time`, `valid_time`, `month`, `hh`,
        `band`, `anomaly_mean_calibrated` (the lead-band slope times the ensemble-mean ENS anomaly;
        a missing value counts as zero), `clim_cams`, and `shape_calibrated` (their sum).
    """
    old = pl.concat(
        [
            pl.read_parquet(path).select("site", "power_fcst_init_time", "valid_time", "u_mean")
            for path in sorted((common.E46_SOLAR_DIR / "ens_train").glob("init=*.parquet"))
        ]
    )
    new = pl.concat([pl.read_parquet(path) for path in sorted(POST_DIR.glob("init=*.parquet"))])
    ens = pl.concat([old, new]).filter(pl.col("site").is_in(list(common.SITES)))
    lead = ((ens["valid_time"] - ens["power_fcst_init_time"]).dt.total_minutes() / 60.0).to_numpy()
    climatology = pl.read_parquet(common.E46_SOLAR_DIR / "climatology.parquet").select(
        "site", "month", "hh", "clim_cams", "clim_ens"
    )
    slopes = pl.read_parquet(common.E46_SOLAR_DIR / "calibration.parquet").select(
        "site", "band", "slope"
    )
    return (
        common.with_cell(frame=ens, time_column="valid_time")
        .with_columns(band=pl.Series(common.lead_band(lead_hours=lead)))
        .join(climatology, on=["site", "month", "hh"], how="left", validate="m:1")
        .join(slopes, on=["site", "band"], how="left", validate="m:1")
        .join(series.select("site", "time_series_id"), on="site", how="left", validate="m:1")
        .with_columns(
            anomaly_mean_calibrated=(pl.col("slope") * (pl.col("u_mean") - pl.col("clim_ens")))
            .fill_nan(0.0)
            .fill_null(0.0),
            clim_cams=pl.col("clim_cams").fill_null(0.0),
        )
        .with_columns(shape_calibrated=pl.col("clim_cams") + pl.col("anomaly_mean_calibrated"))
        .select(
            "time_series_id",
            "site",
            "power_fcst_init_time",
            "valid_time",
            "month",
            "hh",
            "band",
            "anomaly_mean_calibrated",
            "clim_cams",
            "shape_calibrated",
        )
    )


def placebo_table(*, keys: pl.DataFrame) -> pl.DataFrame:
    """Exchange each row's ensemble-mean calibrated anomaly with that of another row of its group.

    The group is the series, local calendar month, UTC half-hour, and lead band. Within a group the
    rows are ordered by valid time and each takes the anomaly of the row a random number of places
    along (a cyclic shift between a quarter and three quarters of the group). Every anomaly is used
    exactly once. A plain permutation would sometimes pair two runs that forecast the same valid
    time, which would keep the timing of the sunshine.

    Args:
        keys: Rows with `site`, `month`, `hh`, `band`, `power_fcst_init_time`, `valid_time`, and
            `anomaly_mean_calibrated`.

    Returns:
        `time_series_id`, `power_fcst_init_time`, `valid_time`, and `anomaly_mean_calibrated_twin`.
    """
    ordered = keys.sort("site", "month", "hh", "band", "valid_time", "power_fcst_init_time")
    group = ordered.select(pl.struct("site", "month", "hh", "band").rank("dense").alias("group"))[
        "group"
    ].to_numpy()
    values = ordered["anomaly_mean_calibrated"].to_numpy()
    rng = np.random.default_rng([common.PLACEBO_SEED, 0])
    donor = np.arange(len(values))
    starts = np.flatnonzero(np.diff(group, prepend=-1))
    for begin, end in zip(starts, [*starts[1:], len(values)], strict=True):
        size = end - begin
        if size > 1:
            shift = int(rng.integers(max(size // 4, 1), max(3 * size // 4, 1) + 1))
            donor[begin:end] = begin + (np.arange(size) + shift) % size
    own_date = ordered["valid_time"].dt.date().to_numpy()
    print(
        f"placebo: {len(values):,} rows, "
        f"{float((donor == np.arange(len(values))).mean()):.4f} keep their own value, "
        f"{float((own_date == own_date[donor]).mean()):.4f} take a value from the same valid date"
    )
    return ordered.select(
        "time_series_id",
        "power_fcst_init_time",
        "valid_time",
        anomaly_mean_calibrated_twin=pl.Series(values[donor]),
    )


def check_against_experiment_64(*, table: pl.DataFrame) -> None:
    """Raise unless the rows up to 2025-06-30 agree with experiment 00064's `train_shape`.

    Args:
        table: `shape_table()`.

    Raises:
        ValueError: If the two tables differ in a key or in a calibrated shape.
    """
    other = pl.read_parquet(
        common.E64_WORK_DIR / "inputs" / "train_shape.parquet",
        columns=[*common.KEYS, "anomaly_mean_calibrated", "shape_calibrated"],
    )
    mine = table.filter(pl.col("valid_time") <= common.LAST_TRAINING_VALID)
    joined = mine.join(other, on=list(common.KEYS), how="full", suffix="_64", coalesce=True)
    missing = joined.filter(
        pl.col("shape_calibrated").is_null() | pl.col("shape_calibrated_64").is_null()
    ).height
    gap = (
        joined.drop_nulls("shape_calibrated")
        .drop_nulls("shape_calibrated_64")
        .select((pl.col("shape_calibrated") - pl.col("shape_calibrated_64")).abs().max())
        .item()
    )
    print(f"check: {missing} unmatched rows, largest shape difference {gap:.2e}")
    if missing or gap > CHECK_TOLERANCE:
        msg = "the rebuilt shapes do not reproduce experiment 00064's"
        raise ValueError(msg)


def main() -> None:
    """Build the requested part."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("post_u", "shape"), required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.part == "post_u":
        build_post()
        return
    table = shape_table(series=sites_table())
    print(f"shape table: {table.height:,} rows, latest valid time {table['valid_time'].max()}")
    if args.check:
        check_against_experiment_64(table=table)
    common.write_atomically(frame=table, target=common.INPUTS_DIR / "train_shape.parquet")
    common.write_atomically(
        frame=placebo_table(keys=table), target=common.INPUTS_DIR / "placebo_train.parquet"
    )


if __name__ == "__main__":
    main()
