"""Build the solar tables of one solar shape in the layout of experiment 00046's `solar/` folder.

The shape is the weights `(cE, cS, cW)` of `work/coefficients.parquet` for the arm named by the
`E77_ARM` environment variable, applied to the east, south, and west unit shapes of
`e00077_orient.py`. Every table is built from the same rules as experiment 00046's
`e00046_solar_inputs.py`, with the shape in place of the fleet's, and only from training-window
valid times:

- `cams_u.parquet`: the CAMS shape `u` of the arm, `site`, `time`, `u`.
- `climatology.parquet`: the month-by-half-hour mean of `u` for CAMS and for the ENS ensemble mean,
  with the neighbouring local months pooled when any cell holds under 30 days. The `daylight`
  column is copied from experiment 00046's climatology, so that every arm scores the same rows.
- `cams_anomaly.parquet`: CAMS `u` minus the CAMS climatology.
- `calibration.parquet`: per site and lead band, the slope through the origin of the CAMS anomaly
  on the ensemble-mean ENS anomaly, over daylight cells.
- `ens_train/init=*.parquet`: the ENS ensemble-mean `u` of each training run.
- `ens_val/init=*.parquet`: for each validation run, the calibrated ensemble-mean anomaly
  (`anomaly_mean_calibrated`, and `anomaly_calibrated`, the same value, for member 0 only: the arm
  `am` reads only the ensemble mean).

    E77_ARM=blend uv run python ml-experiments/00077-*/e00077_solar_inputs.py
"""

import time
from pathlib import Path
from typing import Final

import e00077_common as common
import e00077_orient as orient
import e00077_shares as shares
import numpy as np
import polars as pl

SOLAR_DIR: Final[Path] = common.E46_SOLAR_DIR
TRAIN_DIR: Final[Path] = SOLAR_DIR / "ens_train"
VAL_DIR: Final[Path] = SOLAR_DIR / "ens_val"


def weights_of_arm() -> dict[str, np.ndarray]:
    """Return each series' weights `(cE, cS, cW)` for this process's solar shape."""
    table = pl.read_parquet(shares.COEFFICIENTS_PATH).filter(pl.col("arm") == common.SHAPE_ARM)
    return {
        row["site"]: np.array([row["cE"], row["cS"], row["cW"]])
        for row in table.iter_rows(named=True)
    }


def combine(*, frame: pl.DataFrame, weights: dict[str, np.ndarray]) -> pl.DataFrame:
    """Return the frame with `u`, the weighted sum of `uE`, `uS`, and `uW` at each site."""
    c = pl.DataFrame(
        {
            "site": list(weights),
            "wE": [w[0] for w in weights.values()],
            "wS": [w[1] for w in weights.values()],
            "wW": [w[2] for w in weights.values()],
        }
    )
    return (
        frame.join(c, on="site", how="left", validate="m:1")
        .with_columns(
            u=pl.col("uE") * pl.col("wE")
            + pl.col("uS") * pl.col("wS")
            + pl.col("uW") * pl.col("wW")
        )
        .drop("wE", "wS", "wW", *orient.COLUMNS)
    )


def climatology(*, frame: pl.DataFrame, pool: bool) -> pl.DataFrame:
    """Return the month-by-half-hour mean of `u` over training-window valid times.

    Args:
        frame: Columns `site`, `time`, and `u`.
        pool: Pool each local month with its two neighbours when true.

    Returns:
        Columns `site`, `month`, `hh`, `clim`, and `days`.

    Raises:
        ValueError: If a row is later than the last training valid time.
    """
    training = frame.filter(pl.col("u").is_not_null() & pl.col("u").is_not_nan())
    if common.latest_time(times=training["time"]) > common.LAST_TRAINING_VALID:
        msg = "climatology rows must be training-window valid times only"
        raise ValueError(msg)
    cells = common.with_cell(frame=training, time_column="time").with_columns(
        date=pl.col("time").dt.date()
    )
    if pool:
        cells = pl.concat(
            [
                cells.with_columns(month=((pl.col("month") - 1 + shift) % 12 + 1).cast(pl.Int8))
                for shift in (-1, 0, 1)
            ]
        )
    return cells.group_by("site", "month", "hh").agg(
        clim=pl.col("u").mean(), days=pl.col("date").n_unique()
    )


def calibration_table(
    *, train: pl.DataFrame, cams_anomaly: pl.DataFrame, daylight: pl.DataFrame
) -> pl.DataFrame:
    """Return per site and lead band the slope of the CAMS anomaly on the ENS ensemble-mean anomaly.

    Args:
        train: Columns `site`, `power_fcst_init_time`, `valid_time`, and `anomaly_mean`.
        cams_anomaly: Columns `site`, `valid_time`, and `anomaly_cams`.
        daylight: Columns `site`, `month`, `hh`, and `daylight`.

    Returns:
        Columns `site`, `band`, `slope`, `correlation_mean`, and `n_rows`.

    Raises:
        ValueError: If a row is later than the last training valid time.
    """
    if common.latest_time(times=train["valid_time"]) > common.LAST_TRAINING_VALID:
        msg = "calibration rows must be training-window valid times only"
        raise ValueError(msg)
    lead_hours = (
        (train["valid_time"] - train["power_fcst_init_time"]).dt.total_minutes() / 60.0
    ).to_numpy()
    joined = (
        train.with_columns(band=pl.Series(common.lead_band(lead_hours=lead_hours)))
        .join(cams_anomaly, on=["site", "valid_time"], how="inner")
        .pipe(lambda frame: common.with_cell(frame=frame, time_column="valid_time"))
        .join(daylight, on=["site", "month", "hh"], how="left")
        .filter(pl.col("daylight") & pl.col("anomaly_mean").is_finite())
    )
    return (
        joined.group_by("site", "band")
        .agg(
            slope=(pl.col("anomaly_cams") * pl.col("anomaly_mean")).sum()
            / (pl.col("anomaly_mean") ** 2).sum(),
            correlation_mean=pl.corr("anomaly_cams", "anomaly_mean"),
            n_rows=pl.len(),
        )
        .sort("site", "band")
    )


def build_validation(
    *, weights: dict[str, np.ndarray], ens_clim: pl.DataFrame, calibration: pl.DataFrame
) -> None:
    """Write each validation run's calibrated ensemble-mean anomaly, for member 0."""
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    clim = ens_clim.select("site", "month", "hh", "clim")
    slopes = calibration.select("site", "band", "slope")
    for path in sorted(orient.VAL_DIR.glob("init=*.parquet")):
        target = VAL_DIR / path.name
        if target.exists():
            continue
        started = time.monotonic()
        frame = combine(frame=pl.read_parquet(path), weights=weights)
        origin = frame["power_fcst_init_time"]
        lead = ((frame["valid_time"] - origin).dt.total_minutes() / 60.0).to_numpy()
        table = (
            common.with_cell(frame=frame, time_column="valid_time")
            .with_columns(band=pl.Series(common.lead_band(lead_hours=lead)))
            .join(clim, on=["site", "month", "hh"], how="left", validate="m:1")
            .join(slopes, on=["site", "band"], how="left", validate="m:1")
            .with_columns(
                anomaly_mean_calibrated=(pl.col("slope") * (pl.col("u") - pl.col("clim"))).cast(
                    pl.Float64
                )
            )
            .with_columns(
                anomaly_calibrated=pl.col("anomaly_mean_calibrated"),
                ensemble_member=pl.lit(0, dtype=pl.Int8),
            )
            .select(
                "site",
                "power_fcst_init_time",
                "valid_time",
                "ensemble_member",
                "anomaly_calibrated",
                "anomaly_mean_calibrated",
            )
        )
        common.write_atomically(frame=table, target=target)
        print(f"{target.name}: {time.monotonic() - started:.1f} s", flush=True)


def main() -> None:
    """Build every table of this arm's shape, printing the checks as it goes."""
    weights = weights_of_arm()
    print(f"arm {common.SHAPE_ARM}: weights {weights}")
    SOLAR_DIR.mkdir(parents=True, exist_ok=True)
    cams = combine(frame=pl.read_parquet(orient.ORIENT_DIR / "cams.parquet"), weights=weights)
    common.write_atomically(
        frame=cams.select("site", "time", "u"), target=SOLAR_DIR / "cams_u.parquet"
    )
    train = combine(
        frame=pl.concat(
            [pl.read_parquet(p) for p in sorted(orient.TRAIN_DIR.glob("init=*.parquet"))]
        ),
        weights=weights,
    )
    cams_training = cams.filter(pl.col("time") <= common.LAST_TRAINING_VALID)
    ens_training = train.select("site", time=pl.col("valid_time"), u=pl.col("u"))
    original = pl.read_parquet(common.E46_ORIGINAL_SOLAR_DIR / "climatology.parquet")
    daylight = original.select("site", "month", "hh", "daylight")
    sunny = daylight.filter(pl.col("daylight")).select("site", "month", "hh")
    pool = any(
        float(cells.join(sunny, on=["site", "month", "hh"])["days"].min())  # ty: ignore[invalid-argument-type]
        < common.MIN_CLIMATOLOGY_DAYS
        for cells in (
            climatology(frame=cams_training, pool=False),
            climatology(frame=ens_training, pool=False),
        )
    )
    print(f"pooling the neighbouring months: {pool}")
    cams_clim = climatology(frame=cams_training, pool=pool)
    ens_clim = climatology(frame=ens_training, pool=pool)
    table = (
        cams_clim.rename({"clim": "clim_cams", "days": "days_cams"})
        .join(
            ens_clim.rename({"clim": "clim_ens", "days": "days_ens"}),
            on=["site", "month", "hh"],
            how="full",
            coalesce=True,
        )
        .join(daylight, on=["site", "month", "hh"], how="left")
    )
    common.write_atomically(frame=table, target=SOLAR_DIR / "climatology.parquet")
    cams_anomaly = (
        common.with_cell(frame=cams.rename({"time": "valid_time"}), time_column="valid_time")
        .join(cams_clim.select("site", "month", "hh", "clim"), on=["site", "month", "hh"])
        .select("site", "valid_time", anomaly_cams=pl.col("u") - pl.col("clim"))
    )
    common.write_atomically(frame=cams_anomaly, target=SOLAR_DIR / "cams_anomaly.parquet")
    ens_anomaly = (
        common.with_cell(frame=train, time_column="valid_time")
        .join(ens_clim.select("site", "month", "hh", "clim"), on=["site", "month", "hh"])
        .select(
            "site",
            "power_fcst_init_time",
            "valid_time",
            anomaly_mean=pl.col("u") - pl.col("clim"),
        )
    )
    calibration = calibration_table(train=ens_anomaly, cams_anomaly=cams_anomaly, daylight=daylight)
    with pl.Config(tbl_rows=40):
        print(calibration)
    common.write_atomically(frame=calibration, target=SOLAR_DIR / "calibration.parquet")
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(orient.TRAIN_DIR.glob("init=*.parquet")):
        combined = combine(frame=pl.read_parquet(path), weights=weights)
        common.write_atomically(
            frame=combined.select(
                "site", "power_fcst_init_time", "valid_time", u_mean=pl.col("u").cast(pl.Float32)
            ),
            target=TRAIN_DIR / path.name,
        )
    build_validation(weights=weights, ens_clim=ens_clim, calibration=calibration)
    print(f"latest training valid time used: {train['valid_time'].max()}")


if __name__ == "__main__":
    main()
