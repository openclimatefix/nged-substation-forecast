"""Fit experiment 00018's stage 2 calibrators on serving-latency recent-origin inputs.

For every seed and series, this applies experiment 00005's ERA5-trained stage 1a models to three
weather sources on the dense half-hourly grid (final ERA5, the analysis proxy, and the proxy with
CAMS shortwave), builds each arm's recent-origin inputs per forecast origin (`e00018_common.ARMS`),
fits one multi-quantile XGBoost calibrator per arm on the calibration runs' stage 1 statistics
(experiment 00005's `kind a` statistics), and writes the validation quantiles. For arm `s` it also
fits a calibrator on the calibration runs before 2025-01-01 and writes the quantile loss of the
runs from 2025-01-01, scaled by each factor `k` of the grid, so the spread factor is chosen on no
validation row.

An origin's stage 1 model is the forward-chain model of its quarter for calibration origins and the
final model for validation origins; both stop before the earliest lag time the origin reads.

Output: `work/arms/<arm>_seed<s>.parquet` (series, run, valid time, seed, 9 quantiles) and
`work/kgrid_seed<s>.parquet`.

    uv run python \
        ml-experiments/00018-staged-chain-vs-n1-serving-latency-lag-anomalies-demand/e00018_chain.py
"""

import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00018_common as common
import numpy as np
import polars as pl
import xgboost as xgb

CONFIG: Final[str] = "from2023"
STATS_DIR: Final[Path] = common.EXPERIMENT_00005_WORK_DIR / "stats"
SMOKE_SITES: Final[list[str]] = [
    site for site in os.environ.get("E00018_SMOKE_SITES", "").split(",") if site
]
"""Series labels for a smoke run, which writes under `work/smoke/` instead of `work/arms/`."""
ARMS_DIR: Final[Path] = common.WORK_DIR / ("smoke" if SMOKE_SITES else "arms")
STAT_COLUMNS: Final[tuple[str, ...]] = ("mean", "std", "q10", "q50", "q90")
K_GRID: Final[np.ndarray] = np.round(np.arange(1.0, 1.601, 0.05), 2)
K_SPLIT: Final[datetime] = datetime(2025, 1, 1, tzinfo=UTC)
CALIBRATOR_THREADS: Final[int] = 8
MIN_EARLY_ROWS: Final[int] = 5_000
"""A series with fewer observed early calibration rows has no say in the spread factor."""


def source_predictions(
    *,
    row: dict,
    arrays: common.SeriesArrays,
    proxy: pl.DataFrame,
    cams: np.ndarray,
    models: dict[str, xgb.Booster],
) -> dict[str, dict[str, np.ndarray]]:
    """Return each model's stage 1a predictions on the grid for each weather source.

    Args:
        row: The series' row of `series_table()`.
        arrays: The series' ERA5 features, power, and grid.
        proxy: The analysis proxy's rows of the series.
        cams: CAMS shortwave on the grid (NaN where absent).
        models: Booster by tag (`final`, `q0` to `q4`).

    Returns:
        `{tag: {source: predictions}}`, NaN where the source holds no weather.
    """
    columns = common.feature_columns(config=CONFIG)
    weather = [c for c in columns if c not in common.DERIVED]
    grid = pl.DataFrame({"valid_time": arrays.valid_time})
    proxy_grid = grid.join(proxy.select("valid_time", *weather), on="valid_time", how="left")
    era5 = arrays.features
    proxy_features = era5.copy()
    for name in weather:
        proxy_features[:, columns.index(name)] = proxy_grid[name].to_numpy()
    cams_features = proxy_features.copy()
    shortwave = columns.index("shortwave")
    cams_features[:, shortwave] = np.where(np.isfinite(cams), cams, proxy_features[:, shortwave])
    cams_features[~np.isfinite(cams), :] = proxy_features[~np.isfinite(cams), :]
    # CAMS rows keep the proxy's other variables; a row is NaN wherever the proxy is NaN.
    cams_features[~np.isfinite(proxy_features[:, columns.index("temperature_2m")]), :] = np.nan
    weather_index = [columns.index(name) for name in weather]
    matrices = {"E": era5, "P": proxy_features, "CP": cams_features}
    out: dict[str, dict[str, np.ndarray]] = {}
    for tag, model in models.items():
        out[tag] = {}
        for source, matrix in matrices.items():
            prediction = common.predict(model=model, features=matrix)
            prediction[~np.isfinite(matrix[:, weather_index]).all(axis=1)] = np.nan
            out[tag][source] = prediction.astype(np.float32)
    return out


def chain_tag(*, origin_time: datetime) -> str:
    """Return the stage 1 model tag of an origin: its quarter's chain model, or `final`."""
    if origin_time >= common.CHAIN_END:
        return "final"
    quarter = max(i for i, start in enumerate(common.CHAIN_STARTS) if origin_time >= start)
    return f"q{quarter}"


def matrix_for_rows(
    *,
    table: pl.DataFrame,
    power: np.ndarray,
    predictions: dict[str, dict[str, np.ndarray]],
    arm: common.Arm,
) -> np.ndarray:
    """Return the calibrator inputs for a statistics table of one series, one origin at a time."""
    base = np.concatenate(
        [
            table.select(STAT_COLUMNS).to_numpy().astype(np.float32),
            (
                (
                    np.asarray(common.grid_index(times=table["valid_time"]))
                    - np.asarray(common.grid_index(times=table["power_fcst_init_time"]))
                )
                * 0.5
            ).astype(np.float32)[:, None],
        ],
        axis=1,
    )
    target = np.asarray(common.grid_index(times=table["valid_time"]))
    origin = np.asarray(common.grid_index(times=table["power_fcst_init_time"]))
    width = len(common.recent_columns(with_residuals=arm.source is not None))
    recent = np.empty((table.height, width), dtype=np.float32)
    origins = table["power_fcst_init_time"].to_numpy()
    boundaries = np.flatnonzero(np.diff(origins.astype("int64"))) + 1
    for rows in np.split(np.arange(table.height), boundaries):
        origin_index = int(origin[rows[0]])
        origin_time = table["power_fcst_init_time"][int(rows[0])]
        residual = None
        if arm.source is not None:
            tag = chain_tag(origin_time=origin_time)
            residual = common.origin_residual(
                power=power,
                predictions=predictions[tag],
                source=arm.source,
                origin=origin_index,
                delay_steps=arm.delay_steps,
            )
        recent[rows] = common.recent_inputs(
            power=power,
            residual=residual,
            target=target[rows],
            origin=origin[rows],
            delay_steps=arm.delay_steps,
        )
    return np.concatenate([base, recent], axis=1)


def fit_calibrator(*, x: np.ndarray, y: np.ndarray, seed: int) -> xgb.Booster:
    """Fit the multi-quantile calibrator on rows with an observed target."""
    keep = np.isfinite(y)
    return xgb.train(
        {**common.CALIBRATOR_PARAMETERS, "seed": seed, "nthread": CALIBRATOR_THREADS},
        xgb.DMatrix(x[keep], label=y[keep]),
        num_boost_round=common.CALIBRATOR_ROUNDS,
    )


def select_rows(*, table: pl.DataFrame, series_id: int, seed: int) -> pl.DataFrame:
    """Return one series' `kind a` statistics for the configuration and seed, in time order."""
    return table.filter(
        (pl.col("time_series_id") == series_id)
        & (pl.col("config") == CONFIG)
        & (pl.col("seed") == seed)
        & (pl.col("kind") == "a")
    ).sort("power_fcst_init_time", "valid_time")


def read_stats(*, split: str) -> pl.DataFrame:
    """Read every run's statistics of a split (`calibration` or `validation`)."""
    return pl.concat(
        [pl.read_parquet(p) for p in sorted((STATS_DIR / split).glob("*.parquet"))]
    ).filter((pl.col("config") == CONFIG) & (pl.col("kind") == "a"))


def main() -> None:
    """Fit every arm for every seed and series, skipping a seed whose files exist."""
    series = common.series_table()
    if SMOKE_SITES:
        series = series.filter(pl.col("site").is_in(SMOKE_SITES))
    arrays = common.load_series_arrays(series=series, config=CONFIG)
    proxy_all = pl.read_parquet(common.PROXY_PATH)
    calibration = read_stats(split="calibration")
    validation = read_stats(split="validation")
    cams_all = pl.read_parquet(common.CAMS_PATH)
    grid = pl.DataFrame({"valid_time": common.grid_times()})
    ARMS_DIR.mkdir(parents=True, exist_ok=True)
    for seed in common.SEEDS:
        done = ARMS_DIR / f"seed{seed}.done"
        if done.exists():
            continue
        started = time.monotonic()
        pieces: dict[str, list[pl.DataFrame]] = {name: [] for name in common.ARMS}
        kgrid_rows = []
        for row in series.iter_rows(named=True):
            site, series_id = row["site"], row["time_series_id"]
            data = arrays[site]
            hourly = cams_all.filter(pl.col("site") == site).select("time", "ghi_w_m2")
            cams = (
                grid.with_columns(
                    hour=(pl.col("valid_time") - pl.duration(microseconds=1)).dt.truncate("1h")
                    + pl.duration(hours=1)
                )
                .join(hourly, left_on="hour", right_on="time", how="left", validate="m:1")[
                    "ghi_w_m2"
                ]
                .to_numpy()
                .astype(np.float32)
            )
            models = {}
            for tag in ("final", *(f"q{i}" for i in range(len(common.CHAIN_STARTS)))):
                booster = xgb.Booster()
                booster.load_model(
                    common.model_path(config=CONFIG, seed=seed, kind="stage1a", site=site, tag=tag)
                )
                models[tag] = booster
            predictions = source_predictions(
                row=row,
                arrays=data,
                proxy=proxy_all.filter(pl.col("time_series_id") == series_id),
                cams=cams,
                models=models,
            )
            train = select_rows(table=calibration, series_id=series_id, seed=seed)
            test = select_rows(table=validation, series_id=series_id, seed=seed)
            y_train = common.take(
                values=data.power, index=np.asarray(common.grid_index(times=train["valid_time"]))
            )
            for name, arm in common.ARMS.items():
                x_train = matrix_for_rows(
                    table=train, power=data.power, predictions=predictions, arm=arm
                )
                x_test = matrix_for_rows(
                    table=test, power=data.power, predictions=predictions, arm=arm
                )
                model = fit_calibrator(x=x_train, y=y_train, seed=seed)
                quantiles = np.sort(common.predict(model=model, features=x_test), axis=1)
                pieces[name].append(
                    test.select(
                        "time_series_id", "power_fcst_init_time", "valid_time", "seed"
                    ).with_columns(
                        pl.DataFrame(
                            quantiles.astype(np.float32),
                            schema=list(common.LEVEL_COLUMNS),
                            orient="row",
                        )
                    )
                )
                early = (train["power_fcst_init_time"] < K_SPLIT).to_numpy() & (
                    train["valid_time"] < K_SPLIT
                ).to_numpy()
                if name == "s" and np.isfinite(y_train[early]).sum() >= MIN_EARLY_ROWS:
                    early_model = fit_calibrator(x=x_train[early], y=y_train[early], seed=seed)
                    held = (train["power_fcst_init_time"] >= K_SPLIT).to_numpy() & np.isfinite(
                        y_train
                    )
                    held_q = np.sort(
                        common.predict(model=early_model, features=x_train[held]), axis=1
                    )
                    median = held_q[:, [4]]
                    for k in K_GRID:
                        scaled = median + k * (held_q - median)
                        loss = common.pinball_quantile_score(quantiles=scaled, actual=y_train[held])
                        kgrid_rows.append(
                            {
                                "seed": seed,
                                "time_series_id": series_id,
                                "k": float(k),
                                "loss_sum": float(loss.sum()),
                                "n": int(loss.size),
                            }
                        )
            print(f"seed {seed} {site}: {time.monotonic() - started:.0f} s", flush=True)
        for name, parts in pieces.items():
            target = ARMS_DIR / f"{name}_seed{seed}.parquet"
            pl.concat(parts).write_parquet(target.with_suffix(".tmp"))
            target.with_suffix(".tmp").rename(target)
        pl.DataFrame(kgrid_rows).write_parquet(ARMS_DIR / f"kgrid_seed{seed}.parquet")
        done.write_text(datetime.now().astimezone().isoformat())


if __name__ == "__main__":
    main()
