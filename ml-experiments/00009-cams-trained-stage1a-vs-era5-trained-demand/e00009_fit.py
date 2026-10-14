"""Fit experiment 00009's ERA5-trained and CAMS-trained stage 1a models and apply them to ENS.

For each arm (`era5` or `cams`), seed, and series this fits one XGBoost model on the half-hourly
valid times from 2023-09-01 to before 2025-06-17. The arms differ in one column: `era5` reads ERA5
shortwave at the series' H3 cell, `cams` reads CAMS irradiance at the series' own location. Rows
are the half-hours with an observed power and both irradiance values, so the arms train on the
same rows.

Two outputs, both under `data/ml-experiments/00009-.../work/`:

- `observed/<arm>.parquet`: each model's prediction at the observed weather (ERA5 or CAMS, as the
  arm was trained) for the validation year, a deterministic diagnostic.
- `stats/init=<date>.parquet`: for each of the 73 ENS runs of `fold_a_every5d_v1`, each arm's and
  seed's 9 empirical quantiles of the 51 members' forecasts, with the observed power over
  capacity. Both arms read the ENS member's shortwave at validation.

Existing files are kept, so a crash costs one unit.

    uv run python ml-experiments/00009-cams-trained-stage1a-vs-era5-trained-demand/e00009_fit.py
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00009_common as common
import numpy as np
import polars as pl
import xgboost as xgb

OBSERVED_DIR: Final[Path] = common.WORK_DIR / "observed"
STATS_DIR: Final[Path] = common.WORK_DIR / "stats"
KEY_COLUMNS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")


@dataclass(frozen=True)
class SeriesData:
    """One series' arrays on the dense half-hourly grid.

    Attributes:
        era5_features: Stage 1a's inputs with ERA5 shortwave, float32, one row per grid half-hour.
        cams_features: The same with CAMS irradiance in the shortwave column.
        power: Observed power over effective capacity, NaN where unobserved.
        valid_time: The grid's UTC timestamps.
    """

    era5_features: np.ndarray
    cams_features: np.ndarray
    power: np.ndarray
    valid_time: pl.Series

    def features(self, *, arm: common.ArmType) -> np.ndarray:
        """Return the arm's feature matrix."""
        return self.era5_features if arm == "era5" else self.cams_features


def load_series_data(*, series: pl.DataFrame) -> dict[str, SeriesData]:
    """Return each series' ERA5-arm and CAMS-arm features and observed power on the grid.

    Args:
        series: `common.series_table()`.

    Returns:
        Each D label's `SeriesData`.
    """
    columns = common.feature_columns()
    weather = [c for c in columns if c not in common.DERIVED]
    grid = pl.DataFrame({"valid_time": common.grid_times()})
    era5 = pl.read_parquet(common.ERA5_PATH)
    observed = pl.read_parquet(
        common.OBSERVED_PATH, columns=["time_series_id", "valid_time", "power"]
    )
    out: dict[str, SeriesData] = {}
    for row in series.iter_rows(named=True):
        series_id = row["time_series_id"]
        joined = (
            grid.join(
                era5.filter(pl.col("time_series_id") == series_id).select("valid_time", *weather),
                on="valid_time",
                how="left",
                validate="1:1",
            )
            .join(
                observed.filter(pl.col("time_series_id") == series_id).select(
                    "valid_time", pl.col("power") / row["effective_capacity_mw"]
                ),
                on="valid_time",
                how="left",
                validate="1:1",
            )
            .join(
                common.cams_half_hourly(site=row["site"]),
                on="valid_time",
                how="left",
                validate="1:1",
            )
            .sort("valid_time")
        )
        derived = common.with_derived(
            frame=joined.select("valid_time"),
            latitude=row["latitude"],
            longitude=row["longitude"],
        )
        base = joined.select(weather, "cams").with_columns(derived.select(common.DERIVED))
        era5_matrix = base.select(columns).to_numpy().astype(np.float32)
        cams_matrix = (
            base.with_columns(shortwave=pl.col("cams"))
            .select(columns)
            .to_numpy()
            .astype(np.float32)
        )
        out[row["site"]] = SeriesData(
            era5_features=era5_matrix,
            cams_features=cams_matrix,
            power=joined["power"].to_numpy().astype(np.float32),
            valid_time=joined["valid_time"],
        )
    return out


def training_rows(*, data: SeriesData) -> np.ndarray:
    """Return the grid indices both arms train on: observed power and both irradiance values."""
    first = int(common.grid_index(times=common.TRAIN_START))
    stop = int(common.grid_index(times=common.FINAL_TRAIN_CUTOFF))
    index = np.arange(first, stop)
    shortwave_at = common.feature_columns().index("shortwave")
    keep = (
        np.isfinite(data.power[first:stop])
        & np.isfinite(data.era5_features[first:stop, shortwave_at])
        & np.isfinite(data.cams_features[first:stop, shortwave_at])
    )
    return index[keep]


def fit_models(*, series: pl.DataFrame, data: dict[str, SeriesData]) -> None:
    """Fit and save every arm's, seed's, and series' booster, skipping those that exist."""
    for arm in common.ARMS:
        for seed in common.SEEDS:
            started = time.monotonic()
            for site, arrays in data.items():
                path = common.model_path(arm=arm, seed=seed, site=site)
                if path.exists():
                    continue
                rows = training_rows(data=arrays)
                model = common.fit_booster(
                    features=arrays.features(arm=arm)[rows], target=arrays.power[rows], seed=seed
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                model.save_model(path)
            print(f"fit {arm} seed {seed}: {time.monotonic() - started:.0f} s", flush=True)


def load_models(*, series: pl.DataFrame) -> dict[tuple[str, int, str], xgb.Booster]:
    """Load every saved booster, keyed by arm, seed, and D label."""
    models = {}
    for arm in common.ARMS:
        for seed in common.SEEDS:
            for site in series["site"]:
                booster = xgb.Booster()
                booster.load_model(common.model_path(arm=arm, seed=seed, site=site))
                models[arm, seed, site] = booster
    return models


def write_observed_weather(
    *,
    series: pl.DataFrame,
    data: dict[str, SeriesData],
    models: dict[tuple[str, int, str], xgb.Booster],
) -> None:
    """Write each arm's validation-year predictions at the observed weather, one file per arm."""
    first = int(common.grid_index(times=common.VALIDATION_START))
    stop = int(common.grid_index(times=common.HOLDOUT))
    ids = dict(zip(series["site"], series["time_series_id"], strict=True))
    for arm in common.ARMS:
        target = OBSERVED_DIR / f"{arm}.parquet"
        if target.exists():
            continue
        pieces = []
        for site, arrays in data.items():
            window = np.arange(first, stop)
            matrix = arrays.features(arm=arm)[window]
            for seed in common.SEEDS:
                prediction = models[arm, seed, site].predict(xgb.DMatrix(matrix))
                pieces.append(
                    pl.DataFrame(
                        {
                            "time_series_id": np.full(window.size, ids[site], dtype=np.int32),
                            "valid_time": arrays.valid_time.gather(window.tolist()),
                            "seed": np.full(window.size, seed, dtype=np.int8),
                            "prediction": prediction.astype(np.float32),
                            "actual": arrays.power[window],
                        }
                    )
                )
        target.parent.mkdir(parents=True, exist_ok=True)
        pl.concat(pieces).write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
        print(f"observed weather {arm} written", flush=True)


def run_statistics(
    *,
    run: pl.DataFrame,
    row: dict[str, object],
    arrays: SeriesData,
    models: dict[tuple[str, int, str], xgb.Booster],
) -> pl.DataFrame:
    """Return one series' quantiles in one ENS run, for every arm and seed.

    Args:
        run: The run's rows for this series, every member.
        row: The series' row of `common.series_table()`.
        arrays: The series' `SeriesData`.
        models: Every booster.

    Returns:
        One row per case, arm, and seed with the 9 quantiles and `actual`.
    """
    ordered = common.with_derived(
        frame=run, latitude=row["latitude"], longitude=row["longitude"]
    ).sort("valid_time", "ensemble_member")
    members = ordered["ensemble_member"].n_unique()
    matrix = ordered.select(common.feature_columns()).to_numpy().astype(np.float32)
    keys = ordered.select(*KEY_COLUMNS).unique(maintain_order=True)
    index = np.asarray(common.grid_index(times=keys["valid_time"]))
    actual = pl.Series("actual", arrays.power[index])
    pieces = []
    for arm in common.ARMS:
        for seed in common.SEEDS:
            forecast = (
                models[arm, seed, row["site"]].predict(xgb.DMatrix(matrix)).reshape(-1, members)
            )
            quantiles = np.quantile(forecast, common.LEVELS, axis=1).T.astype(np.float32)
            pieces.append(
                keys.with_columns(
                    pl.DataFrame(quantiles, schema=list(common.LEVEL_COLUMNS), orient="row"),
                    actual,
                    arm=pl.lit(arm),
                    seed=pl.lit(seed).cast(pl.Int8),
                )
            )
    return pl.concat(pieces)


def apply_to_ens(
    *,
    series: pl.DataFrame,
    data: dict[str, SeriesData],
    models: dict[tuple[str, int, str], xgb.Booster],
    limit_runs: int | None,
) -> None:
    """Write each ENS validation run's quantiles for every arm, series, and seed."""
    rows = {row["time_series_id"]: row for row in series.iter_rows(named=True)}
    paths = sorted(common.VALIDATION_DIR.glob("init=*.parquet"))[:limit_runs]
    for path in paths:
        target = STATS_DIR / path.name
        if target.exists():
            continue
        started = time.monotonic()
        frame = pl.read_parquet(path).filter(pl.col("time_series_id").is_in(list(rows)))
        pieces = [
            run_statistics(
                run=frame.filter(pl.col("time_series_id") == series_id),
                row=row,
                arrays=data[row["site"]],
                models=models,
            )
            for series_id, row in rows.items()
        ]
        target.parent.mkdir(parents=True, exist_ok=True)
        pl.concat(pieces).write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
        print(f"{path.name}: {time.monotonic() - started:.0f} s", flush=True)


def main() -> None:
    """Fit the models, then write the observed-weather and ENS outputs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit-runs", type=int, default=None, help="apply to this many runs only")
    args = parser.parse_args()
    series = common.series_table()
    data = load_series_data(series=series)
    fit_models(series=series, data=data)
    models = load_models(series=series)
    write_observed_weather(series=series, data=data, models=models)
    apply_to_ens(series=series, data=data, models=models, limit_runs=args.limit_runs)


if __name__ == "__main__":
    main()
