"""Fit experiment 00028's solar models and write their ENS member statistics.

For each GSP and seed, one XGBoost model forecasts PV_Live's share of its own capacity from ERA5
weather at the valid time and the sun's position (`e00028_common.PV_FEATURES`). A final model trains
on valid times from 2023-09 to before 2025-06-17. Five forward-chain models, one per quarter from
2024-04, train on valid times that end 15 days before the quarter starts. Each ENS run is passed
through the model of its valid times (the chain for calibration runs, the final model for the
validation runs), member by member, and summarised by the member mean, the member standard
deviation, and the 9 member quantiles.

Output: `work/pv_stats/<validation|calibration>/init=<date>.parquet`, skipped if they exist.

    uv run python ml-experiments/00028-pvlive-forecast-in-stage2-gsp-demand/e00028_pv.py
"""

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00028_common as common
import numpy as np
import polars as pl
import xgboost as xgb
from ml_core.features._nwp import _upsample_nwp_to_half_hourly
from studies.solar import azimuth, zenith
from studies.sources import PER_STUDY_DIR

STATS_DIR: Final[Path] = common.WORK_DIR / "pv_stats"
ARCHIVE_DIR: Final[Path] = (
    PER_STUDY_DIR.parent.parent / "NWP" / "nwp_model_id=ECMWF_ENS_0_25_degree"
)
PUBLICATION_DELAY_HOURS: Final[int] = 9
HORIZON_HOURS: Final[int] = 351
ENS_COLUMNS: Final[dict[str, str]] = {
    "temperature_2m": "temperature_2m",
    "dew_point_temperature_2m": "dew_point_temperature_2m",
    "shortwave": "downward_short_wave_radiation_flux_surface",
    "longwave": "downward_long_wave_radiation_flux_surface",
    "pressure_surface": "pressure_surface",
    "pressure_msl": "pressure_reduced_to_mean_sea_level",
    "precipitation": "precipitation_surface",
    "precipitation_type": "categorical_precipitation_type_surface",
}
"""Weather columns the archive serves directly; wind speeds are built from the components."""
KEY_COLUMNS: Final[tuple[str, ...]] = (
    "time_series_id",
    "power_fcst_init_time",
    "valid_time",
    "ensemble_member",
)


def sun_columns(*, valid_time: pl.Series, latitude: float, longitude: float) -> pl.DataFrame:
    """Return the sun's zenith and azimuth at the midpoint of each period-ending half-hour."""
    midpoint = valid_time.dt.offset_by("-15m")
    return pl.DataFrame(
        {
            "solar_zenith": zenith(stamps=midpoint, latitude=latitude, longitude=longitude),
            "solar_azimuth": azimuth(stamps=midpoint, latitude=latitude, longitude=longitude),
        }
    ).cast(pl.Float32)


def era5_matrix(*, series: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return each GSP's solar-model inputs at ERA5 weather on the dense grid."""
    weather = [c for c in common.PV_FEATURES if not c.startswith("solar_")]
    grid = pl.DataFrame({"valid_time": common.grid_times()})
    era5 = pl.read_parquet(common.ERA5_PATH)
    out = {}
    for row in series.iter_rows(named=True):
        joined = grid.join(
            era5.filter(pl.col("time_series_id") == row["time_series_id"]).select(
                "valid_time", *weather
            ),
            on="valid_time",
            how="left",
            validate="1:1",
        ).sort("valid_time")
        sun = sun_columns(
            valid_time=joined["valid_time"], latitude=row["latitude"], longitude=row["longitude"]
        )
        out[row["site"]] = (
            joined.select(weather)
            .with_columns(sun)
            .select(common.PV_FEATURES)
            .to_numpy()
            .astype(np.float32)
        )
    return out


def fit_models(*, features: np.ndarray, share: np.ndarray, seed: int) -> dict[str, xgb.Booster]:
    """Fit the final model and the five forward-chain models of one GSP and seed.

    Args:
        features: The GSP's solar-model inputs on the grid.
        share: PV_Live's share of capacity on the grid, NaN where missing.
        seed: The fitting seed.

    Returns:
        The booster of each tag: `final`, and `q0` to `q4`.
    """
    start = int(common.grid_index(times=common.TRAIN_START))
    stops = {"final": int(common.grid_index(times=common.FINAL_TRAIN_CUTOFF))}
    stops |= {
        f"q{tag}": int(common.grid_index(times=quarter - common.PURGE))
        for tag, quarter in enumerate(common.CHAIN_STARTS)
    }
    models = {}
    for tag, stop in stops.items():
        rows = np.arange(start, stop)
        rows = rows[np.isfinite(share[rows]) & np.isfinite(features[rows]).all(axis=1)]
        models[tag] = xgb.train(
            {**common.XGBOOST_PARAMETERS, "seed": seed},
            xgb.DMatrix(features[rows], label=share[rows]),
            num_boost_round=common.N_ROUNDS,
        )
    return models


def archive_run(*, run: datetime, series: pl.DataFrame) -> pl.DataFrame:
    """Return one archive run's half-hourly weather for every GSP and member.

    Args:
        run: The run's initialisation time (00:00 UTC).
        series: `common.series_table()`.

    Returns:
        Rows keyed by `KEY_COLUMNS` with the weather columns, from the run's
        `power_fcst_init_time` to 351 hours later.
    """
    stamp = f"{run:%Y-%m-%d} 00:00:00.000000".replace(" ", "%20").replace(":", "%3A")
    cells = series["h3_res_5"].unique().to_list()
    native = (
        pl.scan_parquet(ARCHIVE_DIR / f"init_time={stamp}" / "*.parquet")
        .select(
            "h3_index",
            "ensemble_member",
            "valid_time",
            *ENS_COLUMNS.values(),
            "wind_u_10m",
            "wind_v_10m",
            "wind_u_100m",
            "wind_v_100m",
        )
        .filter(pl.col("h3_index").is_in(cells))
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
    init = run + timedelta(hours=PUBLICATION_DELAY_HOURS)
    return (
        half_hourly.with_columns(
            wind_speed_10m=(pl.col("wind_u_10m") ** 2 + pl.col("wind_v_10m") ** 2).sqrt(),
            power_fcst_init_time=pl.lit(init).cast(pl.Datetime("us", "UTC")),
        )
        .filter(
            (pl.col("valid_time") > pl.col("power_fcst_init_time"))
            & (
                pl.col("valid_time")
                <= pl.col("power_fcst_init_time") + pl.duration(hours=HORIZON_HOURS + 9)
            )
        )
        .rename({source: name for name, source in ENS_COLUMNS.items()})
    )


def calibration_runs() -> list[datetime]:
    """Return every third daily run from 2024-04-01 up to 2025-06-30."""
    runs = []
    run = common.FIRST_CALIBRATION_RUN
    while run <= common.LAST_TRAINING_VALID:
        runs.append(run)
        run += timedelta(days=common.CALIBRATION_STRIDE_DAYS)
    return runs


def run_statistics(
    *,
    frame: pl.DataFrame,
    series_id: int,
    coordinates: tuple[float, float],
    models: dict[int, dict[str, xgb.Booster]],
    chained: bool,
) -> pl.DataFrame:
    """Return one GSP's solar-forecast statistics for one run, for every seed.

    Args:
        frame: The run's rows for this GSP, every member.
        series_id: The GSP's `time_series_id`.
        coordinates: Latitude and longitude, which never leave this process.
        models: Each seed's boosters by tag.
        chained: Whether to use each valid time's forward-chain model (calibration) or the final
            model (validation).

    Returns:
        One row per valid time and seed with `mean`, `std`, and the 9 member quantiles.
    """
    members = sorted(frame["ensemble_member"].unique().to_list())
    ordered = frame.sort("valid_time", "ensemble_member")
    steps = ordered["valid_time"].unique(maintain_order=True)
    count = steps.len()
    sun = sun_columns(
        valid_time=ordered["valid_time"], latitude=coordinates[0], longitude=coordinates[1]
    )
    matrix = (
        ordered.select([c for c in common.PV_FEATURES if not c.startswith("solar_")])
        .with_columns(sun)
        .select(common.PV_FEATURES)
        .to_numpy()
        .astype(np.float32)
    )
    target = np.asarray(common.grid_index(times=steps))
    if chained:
        quarter = np.clip(
            np.searchsorted(
                [int(common.grid_index(times=q)) for q in common.CHAIN_STARTS], target, side="right"
            )
            - 1,
            0,
            len(common.CHAIN_STARTS) - 1,
        )
        tags = np.array([f"q{q}" for q in quarter])
    else:
        tags = np.array(["final"] * count)
    pieces = []
    for seed, boosters in models.items():
        forecast = np.empty((count, len(members)), dtype=np.float32)
        for tag in sorted(set(tags.tolist())):
            rows = np.flatnonzero(tags == tag)
            block = matrix.reshape(count, len(members), -1)[rows].reshape(-1, matrix.shape[1])
            forecast[rows] = boosters[tag].predict(xgb.DMatrix(block)).reshape(rows.size, -1)
        summary = np.concatenate(
            [
                forecast.mean(axis=1, keepdims=True),
                forecast.std(axis=1, keepdims=True),
                np.quantile(forecast, common.LEVELS, axis=1).T,
            ],
            axis=1,
        ).astype(np.float32)
        pieces.append(
            pl.DataFrame(
                summary, schema=["mean", "std", *common.LEVEL_COLUMNS], orient="row"
            ).with_columns(
                time_series_id=pl.lit(series_id).cast(pl.Int32),
                power_fcst_init_time=pl.lit(ordered["power_fcst_init_time"][0]),
                valid_time=steps,
                seed=pl.lit(seed).cast(pl.Int8),
            )
        )
    return pl.concat(pieces)


def main() -> None:
    """Fit the solar models and write the statistics of every run, skipping existing files."""
    series = common.series_table()
    pv = common.load_pv_live(series=series)
    features = era5_matrix(series=series)
    models = {
        row["site"]: {
            seed: fit_models(
                features=features[row["site"]], share=pv[row["site"]]["share"], seed=seed
            )
            for seed in common.SEEDS
        }
        for row in series.iter_rows(named=True)
    }
    coords = {
        row["site"]: (row["latitude"], row["longitude"]) for row in series.iter_rows(named=True)
    }
    ids = dict(zip(series["site"], series["time_series_id"], strict=True))
    jobs: list[tuple[str, Path, datetime]] = [
        (
            "validation",
            path,
            datetime.fromisoformat(path.stem.removeprefix("init=")).replace(tzinfo=UTC),
        )
        for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet"))
    ] + [("calibration", Path(), run) for run in calibration_runs()]
    for split, source, run in jobs:
        target = STATS_DIR / split / f"init={run:%Y-%m-%d}.parquet"
        if target.exists():
            continue
        started = time.monotonic()
        if split == "validation":
            frame = pl.read_parquet(source).filter(
                pl.col("time_series_id").is_in(list(ids.values()))
            )
        else:
            frame = archive_run(run=run, series=series).filter(
                pl.col("valid_time") <= common.LAST_TRAINING_VALID
            )
        tables = [
            run_statistics(
                frame=frame.filter(pl.col("time_series_id") == series_id),
                series_id=series_id,
                coordinates=coords[site],
                models=models[site],
                chained=split == "calibration",
            )
            for site, series_id in ids.items()
        ]
        target.parent.mkdir(parents=True, exist_ok=True)
        pl.concat(tables).write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
        print(f"{split} {target.name}: {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
