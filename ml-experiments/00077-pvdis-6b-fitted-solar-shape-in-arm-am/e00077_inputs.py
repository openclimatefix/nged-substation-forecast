"""Build the solar inputs of one solar shape for the fits, under `work/arm/<shape>/inputs/`.

The tables of `e00077_solar_inputs.py` (CAMS `u`, climatologies, calibration slopes, validation
anomalies) are read, never written. This script writes:

- `train_shape.parquet`: for each training row (series, run, valid time; the ENS control member's
  rows of experiment 00003's `train.parquet`) the ensemble-mean calibrated anomaly
  (`anomaly_mean_calibrated`, the lead-band slope times the ensemble-mean ENS anomaly), the CAMS
  climatology (`clim_cams`) and their sum `shape_calibrated`, plus the CAMS `u` and anomaly at the
  valid time, the month, half-hour, lead band, and the daylight flag.
- `t_ref.parquet`: per series, the training-window mean over daylight rows of the cell temperature.
- `val_keys.parquet`: each validation run's distinct (series, run, valid time) with the
  ensemble-mean calibrated anomaly.
- `placebo_train.parquet`, `placebo_val.parquet`: the placebo twins' anomaly. Within a series,
  local calendar month, UTC half-hour, and lead band, the ensemble-mean calibrated anomaly of each
  forecast run is exchanged for that of another run by a seeded permutation (without replacement),
  separately in training and validation. The permutation depends only on the keys and the seed, so
  it picks the same donor runs as experiment 00064's.

    E77_ARM=blend uv run python ml-experiments/00077-*/e00077_inputs.py --part train_shape
    E77_ARM=blend uv run python ml-experiments/00077-*/e00077_inputs.py --part placebo
"""

import argparse
from pathlib import Path
from typing import Final

import e00077_common as common
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ARCHIVE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR.parent / "NWP" / "nwp_model_id=ECMWF_ENS_0_25_degree"
)
SHORTWAVE_SOURCE: Final[str] = "downward_short_wave_radiation_flux_surface"
LAG_RUN_HOURS: Final[int] = 36
"""Hours of each run kept for the lag shapes.

An ENS-sourced lagged time is at most 24 hours before the origin, which is 9 hours after the run's
initialisation, so it lies within the first 9 hours of the origin's run and the 24 to 33 hours of
the previous day's run. Keeping 36 hours covers both."""
LAG_RUNS_DIR: Final[Path] = common.INPUTS_DIR / "lag_runs"
PUBLICATION_HOURS: Final[int] = 9
GRID_START_MICROSECONDS: Final[int] = int(common.GRID_START.timestamp() * 1e6)
HALF_HOUR_MICROSECONDS: Final[int] = 1_800_000_000


def sites_table() -> pl.DataFrame:
    """Return the verdict series' `time_series_id`, `site`, `h3_res_5`, and coordinates."""
    return common.series_table().filter(pl.col("site").is_in(list(common.SITES)))


def train_shape_table(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return the calibrated ENS shape and the CAMS values of every training row of the series.

    Args:
        series: `sites_table()`.

    Returns:
        Columns `time_series_id` (Int32), `site`, `power_fcst_init_time`, `valid_time`, `month`
        (Int8), `hh` (Int8), `band` (Int64), `u_mean` (the raw ensemble-mean ENS `u`),
        `anomaly_mean_calibrated`, `clim_cams`, `shape_calibrated` (their sum), `anomaly_cams`,
        `u_cams` (null where CAMS is missing), and `daylight` (false where the cell has no CAMS
        climatology). A missing or NaN anomaly or climatology counts as zero, so the shift of that
        row is zero.

    Raises:
        ValueError: If a row is later than the last training valid time.
    """
    ens = pl.concat(
        [
            pl.read_parquet(path)
            for path in sorted((common.E46_SOLAR_DIR / "ens_train").glob("init=*.parquet"))
        ]
    ).filter(pl.col("site").is_in(list(common.SITES)))
    common.require_training_window(times=ens["valid_time"], what="ENS training rows")
    lead = ((ens["valid_time"] - ens["power_fcst_init_time"]).dt.total_minutes() / 60.0).to_numpy()
    climatology = pl.read_parquet(common.E46_SOLAR_DIR / "climatology.parquet").select(
        "site", "month", "hh", "clim_cams", "clim_ens", "daylight"
    )
    slopes = pl.read_parquet(common.E46_SOLAR_DIR / "calibration.parquet").select(
        "site", "band", "slope"
    )
    cams_anomaly = pl.read_parquet(common.E46_SOLAR_DIR / "cams_anomaly.parquet")
    cams_u = pl.read_parquet(common.E46_SOLAR_DIR / "cams_u.parquet").rename(
        {"time": "valid_time", "u": "u_cams"}
    )
    return (
        common.with_cell(frame=ens, time_column="valid_time")
        .with_columns(band=pl.Series(common.lead_band(lead_hours=lead)))
        .join(climatology, on=["site", "month", "hh"], how="left", validate="m:1")
        .join(slopes, on=["site", "band"], how="left", validate="m:1")
        .join(cams_anomaly, on=["site", "valid_time"], how="left", validate="m:1")
        .join(cams_u, on=["site", "valid_time"], how="left", validate="m:1")
        .join(series.select("site", "time_series_id"), on="site", how="left", validate="m:1")
        .with_columns(
            anomaly_mean_calibrated=(pl.col("slope") * (pl.col("u_mean") - pl.col("clim_ens")))
            .fill_nan(0.0)
            .fill_null(0.0),
            clim_cams=pl.col("clim_cams").fill_null(0.0),
            daylight=pl.col("daylight").fill_null(value=False),
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
            "u_mean",
            "anomaly_mean_calibrated",
            "clim_cams",
            "shape_calibrated",
            "anomaly_cams",
            "u_cams",
            "daylight",
        )
    )


def reference_temperature(*, shape: pl.DataFrame) -> pl.DataFrame:
    """Return, per series, the training-window mean cell temperature over daylight rows.

    Args:
        shape: `train_shape_table()`.

    Returns:
        Columns `site` and `t_ref` (the mean of the control member's 2 m temperature plus 25 K
        times `shape_calibrated`).
    """
    temperature = pl.read_parquet(
        common.ENS_TRAIN_PATH,
        columns=["time_series_id", "power_fcst_init_time", "valid_time", "temperature_2m"],
    )
    common.require_training_window(times=temperature["valid_time"], what="training temperatures")
    return (
        shape.filter(pl.col("daylight"))
        .join(
            temperature,
            on=["time_series_id", "power_fcst_init_time", "valid_time"],
            how="inner",
            validate="1:1",
        )
        .with_columns(
            cell_temperature=pl.col("temperature_2m")
            + common.CELL_TEMPERATURE_RISE_K * pl.col("shape_calibrated")
        )
        .group_by("site")
        .agg(t_ref=pl.col("cell_temperature").mean())
        .sort("site")
    )


def validation_keys(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return every validation run's distinct keys with the ensemble-mean calibrated anomaly.

    Args:
        series: `sites_table()`.

    Returns:
        Columns `time_series_id`, `site`, `power_fcst_init_time`, `valid_time`, `month`, `hh`,
        `band`, and `anomaly_mean_calibrated`, from experiment 00046's `ens_val` files (member 0's
        rows; the ensemble-mean anomaly is the same for every member).
    """
    ens = pl.concat(
        [
            pl.read_parquet(
                path,
                columns=[
                    "site",
                    "power_fcst_init_time",
                    "valid_time",
                    "ensemble_member",
                    "anomaly_mean_calibrated",
                ],
            ).filter(pl.col("site").is_in(list(common.SITES)) & (pl.col("ensemble_member") == 0))
            for path in sorted((common.E46_SOLAR_DIR / "ens_val").glob("init=*.parquet"))
        ]
    ).drop("ensemble_member")
    lead = ((ens["valid_time"] - ens["power_fcst_init_time"]).dt.total_minutes() / 60.0).to_numpy()
    return (
        common.with_cell(frame=ens, time_column="valid_time")
        .with_columns(band=pl.Series(common.lead_band(lead_hours=lead)))
        .join(series.select("site", "time_series_id"), on="site", validate="m:1")
        .with_columns(pl.col("anomaly_mean_calibrated").fill_nan(0.0).fill_null(0.0))
        .select(
            "time_series_id",
            "site",
            "power_fcst_init_time",
            "valid_time",
            "month",
            "hh",
            "band",
            "anomaly_mean_calibrated",
        )
    )


def placebo_table(*, keys: pl.DataFrame, stream: int) -> pl.DataFrame:
    """Exchange each row's ensemble-mean calibrated anomaly with that of another row of its group.

    The group is the series, local calendar month, UTC half-hour, and lead band. Within a group the
    rows are ordered by valid time and each takes the anomaly of the row a random number of places
    along (a cyclic shift between a quarter and three quarters of the group, drawn by a generator
    seeded with `PLACEBO_SEED` and `stream`). Every anomaly is used exactly once, so the
    distribution is kept. A plain random permutation would sometimes pair two runs that forecast
    the same valid time, which would keep the timing of the sunshine; the shift pairs rows of
    different days.

    Args:
        keys: Rows with `site`, `month`, `hh`, `band`, `power_fcst_init_time`, `valid_time`, and
            `anomaly_mean_calibrated`, one row per run and valid time.
        stream: 0 for training and 1 for validation, so each has its own random stream.

    Returns:
        `time_series_id`, `power_fcst_init_time`, `valid_time`, and `anomaly_mean_calibrated_twin`,
        the donor row's anomaly.
    """
    ordered = keys.sort("site", "month", "hh", "band", "valid_time", "power_fcst_init_time")
    group = ordered.select(pl.struct("site", "month", "hh", "band").rank("dense").alias("group"))[
        "group"
    ].to_numpy()
    values = ordered["anomaly_mean_calibrated"].to_numpy()
    rng = np.random.default_rng([common.PLACEBO_SEED, stream])
    donor = np.arange(len(values))
    starts = np.flatnonzero(np.diff(group, prepend=-1))
    for begin, end in zip(starts, [*starts[1:], len(values)], strict=True):
        size = end - begin
        if size > 1:
            shift = int(rng.integers(max(size // 4, 1), max(3 * size // 4, 1) + 1))
            donor[begin:end] = begin + (np.arange(size) + shift) % size
    own_date = ordered["valid_time"].dt.date().to_numpy()
    same_date = float((own_date == own_date[donor]).mean())
    print(
        f"placebo stream {stream}: {len(values):,} rows, "
        f"{float((donor == np.arange(len(values))).mean()):.4f} keep their own value, "
        f"{same_date:.4f} take a value from the same valid date"
    )
    return ordered.select(
        "time_series_id",
        "power_fcst_init_time",
        "valid_time",
        anomaly_mean_calibrated_twin=pl.Series(values[donor]),
    )


def main() -> None:
    """Build the requested part of the inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part",
        choices=("train_shape", "placebo"),
        required=True,
    )
    args = parser.parse_args()
    common.INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    series = sites_table()
    train_path = common.INPUTS_DIR / "train_shape.parquet"
    val_path = common.INPUTS_DIR / "val_keys.parquet"
    if args.part == "train_shape":
        shape = train_shape_table(series=series)
        print(
            f"training rows {shape.height:,}; null shape: {shape['shape_calibrated'].null_count()}"
        )
        common.write_atomically(frame=shape, target=train_path)
        common.write_atomically(
            frame=reference_temperature(shape=shape), target=common.INPUTS_DIR / "t_ref.parquet"
        )
        common.write_atomically(frame=validation_keys(series=series), target=val_path)
        print(pl.read_parquet(common.INPUTS_DIR / "t_ref.parquet"))
    else:
        for stream, (path, target) in enumerate(
            ((train_path, "placebo_train.parquet"), (val_path, "placebo_val.parquet"))
        ):
            common.write_atomically(
                frame=placebo_table(keys=pl.read_parquet(path), stream=stream),
                target=common.INPUTS_DIR / target,
            )


if __name__ == "__main__":
    main()
