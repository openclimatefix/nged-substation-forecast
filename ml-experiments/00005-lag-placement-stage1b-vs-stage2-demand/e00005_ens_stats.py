"""Turn each ENS run into stage 1 statistics for experiment 00005, for validation and calibration.

For every ENS run, series, configuration, and seed, this applies stage 1a to each of the 51 members
and, where the configuration has stage 1b, adds the stage 1b correction (one value per case,
computed from observed power before the run's origin). It writes the mean, standard deviation, and
9 quantiles of the 51 member forecasts for two kinds: `a` (stage 1a alone) and `ab` (stage 1a plus
the correction).

- Validation: the 73 runs of `fold_a_every5d_v1` from experiment 00003's frames, with the final
  models.
- Calibration: every third daily run from 2024-04-01 to 2025-06-30, built from the ENS archive with
  production's half-hourly resampling, keeping valid times up to 2025-06-30 23:30. Each row uses
  the forward-chain model of its valid time's quarter, whose training ends 15 days before the
  quarter starts, which is before the run's initialisation.

Output: `work/stats/<validation|calibration>/init=<date>.parquet`, skipped if they exist.

    uv run python ml-experiments/00005-lag-placement-stage1b-vs-stage2-demand/e00005_ens_stats.py
"""

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import e00005_common as common
import e00005_stage1b as stage1b
import numpy as np
import polars as pl
import xgboost as xgb
from ml_core.features._nwp import _upsample_nwp_to_half_hourly
from studies.sources import PER_STUDY_DIR

STATS_DIR: Final[Path] = common.WORK_DIR / "stats"
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


def calibration_runs() -> list[datetime]:
    """Return every third daily run from 2024-04-01 up to 2025-06-30."""
    runs = []
    run = common.FIRST_CALIBRATION_RUN
    while run <= common.LAST_TRAINING_VALID:
        runs.append(run)
        run += timedelta(days=common.CALIBRATION_STRIDE_DAYS)
    return runs


def archive_run(*, run: datetime, series: pl.DataFrame) -> pl.DataFrame:
    """Return one archive run's half-hourly weather for every series and member.

    Args:
        run: The run's initialisation time (00:00 UTC).
        series: `common.series_table()`.

    Returns:
        Rows keyed by `KEY_COLUMNS` with the weather and calendar columns, from the run's
        `power_fcst_init_time` to 351 hours later, and the lead in hours.
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
    rows = (
        half_hourly.with_columns(
            wind_speed_10m=(pl.col("wind_u_10m") ** 2 + pl.col("wind_v_10m") ** 2).sqrt(),
            wind_speed_100m=(pl.col("wind_u_100m") ** 2 + pl.col("wind_v_100m") ** 2).sqrt(),
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
    calendar = common.calendar_values(valid_time=rows["valid_time"])
    return rows.with_columns(
        hour_local=pl.Series(calendar[:, 0]), day_of_week_local=pl.Series(calendar[:, 1])
    )


class Models:
    """A cache of fitted boosters loaded from `common.model_path`."""

    def __init__(self) -> None:
        """Start with an empty cache."""
        self._cache: dict[Path, xgb.Booster] = {}

    def get(self, *, config: str, seed: int, kind: str, site: str, tag: str) -> xgb.Booster:
        """Return one booster, loading it on first use.

        Args:
            config: A key of `common.CONFIGS`.
            seed: The fitting seed.
            kind: `stage1a` or `stage1b`.
            site: The series' label.
            tag: `final` or `q0` to `q4`.

        Returns:
            The booster.
        """
        path = common.model_path(config=config, seed=seed, kind=kind, site=site, tag=tag)
        if path not in self._cache:
            self._cache[path] = xgb.Booster()
            self._cache[path].load_model(path)
        return self._cache[path]


def _summary(*, member_forecasts: np.ndarray) -> np.ndarray:
    """Return mean, standard deviation, and the 9 quantiles of each row's members.

    Args:
        member_forecasts: Shape (cases, members).

    Returns:
        Shape (cases, 11).
    """
    return np.concatenate(
        [
            member_forecasts.mean(axis=1, keepdims=True),
            member_forecasts.std(axis=1, keepdims=True),
            np.quantile(member_forecasts, common.LEVELS, axis=1).T,
        ],
        axis=1,
    ).astype(np.float32)


def run_statistics(
    *,
    frame: pl.DataFrame,
    site: str,
    series_id: int,
    powers: dict[str, np.ndarray],
    predictions: dict[str, dict[int, np.ndarray]],
    models: Models,
    chained: bool,
) -> pl.DataFrame:
    """Return the stage 1 statistics of one series in one run, for every configuration and seed.

    Args:
        frame: The run's rows for this series, every member.
        site: The series' label.
        series_id: The series' `time_series_id`.
        powers: For each configuration, this series' observed power on the grid.
        predictions: For each configuration and seed, this series' ERA5 predictions on the grid.
        models: The booster cache.
        chained: Whether to use each valid time's forward-chain model (calibration) instead of
            the final model (validation).

    Returns:
        One row per case, configuration, seed, and kind with `mean`, `std`, and the 9 quantiles.
    """
    members = sorted(frame["ensemble_member"].unique().to_list())
    ordered = frame.sort("valid_time", "ensemble_member")
    steps = ordered["valid_time"].unique(maintain_order=True)
    count = steps.len()
    target = np.asarray(common.grid_index(times=steps))
    origin = np.full(count, int(common.grid_index(times=ordered["power_fcst_init_time"][0])))
    if chained:
        quarter = (
            np.searchsorted(
                [int(common.grid_index(times=q)) for q in common.CHAIN_STARTS], target, side="right"
            )
            - 1
        )
        quarter = np.clip(quarter, 0, len(common.CHAIN_STARTS) - 1)
    pieces = []
    for config, settings in common.CONFIGS.items():
        if chained and not settings.calibrators:
            continue
        columns = common.feature_columns(config=config)
        matrix = ordered.select(columns).to_numpy().astype(np.float32)
        power = powers[config]
        for seed in common.SEEDS:
            tags = [f"q{q}" for q in quarter] if chained else ["final"] * count
            member_forecast = np.empty((count, len(members)), dtype=np.float32)
            for tag in sorted(set(tags)):
                rows = np.flatnonzero(np.array(tags) == tag)
                model = models.get(config=config, seed=seed, kind="stage1a", site=site, tag=tag)
                block = matrix.reshape(count, len(members), -1)[rows].reshape(-1, matrix.shape[1])
                member_forecast[rows] = common.predict(model=model, features=block).reshape(
                    rows.size, len(members)
                )
            kinds = {"a": member_forecast}
            if settings.stage1b:
                residual = stage1b.residuals(power=power, prediction=predictions[config][seed])
                inputs = stage1b.stage1b_matrix(
                    power=power, residual=residual, target=target, origin=origin
                )
                correction = np.empty(count, dtype=np.float32)
                for tag in sorted(set(tags)):
                    rows = np.flatnonzero(np.array(tags) == tag)
                    model = models.get(config=config, seed=seed, kind="stage1b", site=site, tag=tag)
                    correction[rows] = common.predict(model=model, features=inputs[rows])
                kinds["ab"] = member_forecast + correction[:, None]
            for kind, values in kinds.items():
                summary = _summary(member_forecasts=values)
                pieces.append(
                    pl.DataFrame(
                        summary, schema=["mean", "std", *common.LEVEL_COLUMNS], orient="row"
                    ).with_columns(
                        time_series_id=pl.lit(series_id).cast(pl.Int32),
                        power_fcst_init_time=pl.lit(ordered["power_fcst_init_time"][0]),
                        valid_time=steps,
                        config=pl.lit(config),
                        seed=pl.lit(seed).cast(pl.Int8),
                        kind=pl.lit(kind),
                    )
                )
    return pl.concat(pieces)


def main() -> None:
    """Write the statistics of every validation and calibration run, skipping existing files."""
    series = common.series_table()
    arrays = {
        config: {
            site: data.power
            for site, data in common.load_series_arrays(series=series, config=config).items()
        }
        for config in common.CONFIGS
    }
    ids = dict(zip(series["site"], series["time_series_id"], strict=True))
    era5 = {
        config: {
            seed: common.read_era5_pred(config=config, seed=seed, sites=list(ids))
            for seed in common.SEEDS
        }
        for config in common.CONFIGS
    }
    models = Models()
    jobs: list[tuple[str, Path, datetime]] = [
        (
            "validation",
            path,
            datetime.fromisoformat(path.stem.removeprefix("init=")).replace(tzinfo=UTC),
        )
        for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet"))
    ] + [("calibration", Path(f"init={run:%Y-%m-%d}.parquet"), run) for run in calibration_runs()]
    if common.SMOKE:
        jobs = jobs[:3] + jobs[-12:]
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
                site=site,
                series_id=series_id,
                powers={config: arrays[config][site] for config in common.CONFIGS},
                predictions={
                    config: {seed: era5[config][seed][site] for seed in common.SEEDS}
                    for config in common.CONFIGS
                },
                models=models,
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
