"""Fit experiment 00005's stage 2 calibrators and write their validation quantiles.

One multi-quantile XGBoost model per series, configuration, seed, and arm, trained on the
calibration runs' stage 1 statistics (`e00005_ens_stats.py`) with the observed power over effective
capacity as the target. The arms differ in the stage 1 output they summarise and in whether the
recent-origin inputs are added:

- `r0c`: stage 1a, no recent-origin inputs (the lag-free calibrator).
- `l3`: stage 1a, with the recent-origin inputs.
- `l2c`: stage 1a plus the stage 1b correction, no recent-origin inputs.
- `l2l3`: stage 1a plus the stage 1b correction, with the recent-origin inputs.

Inputs of every arm: the mean, standard deviation, and 10th, 50th, and 90th percentiles of the 51
member forecasts, and the lead in hours. No calendar column. Output quantiles are sorted.

Output: `work/calibrated/<arm>_<config>.parquet` with the validation keys, `seed`, the 9 quantile
columns, and `actual`, skipped if it exists.

    uv run python ml-experiments/00005-lag-placement-stage1b-vs-stage2-demand/e00005_calibrate.py
"""

import time
from pathlib import Path
from typing import Final

import e00005_common as common
import e00005_ens_stats as ens_stats
import e00005_stage1b as stage1b
import numpy as np
import polars as pl
import xgboost as xgb

OUT_DIR: Final[Path] = common.WORK_DIR / "calibrated"
ARMS: Final[dict[str, tuple[str, bool]]] = {
    "r0c": ("a", False),
    "l3": ("a", True),
    "l2c": ("ab", False),
    "l2l3": ("ab", True),
}
"""Each arm's stage 1 kind and whether it receives the recent-origin inputs."""
STAT_COLUMNS: Final[tuple[str, ...]] = ("mean", "std", "q10", "q50", "q90")


def _matrix(
    *,
    table: pl.DataFrame,
    power: np.ndarray,
    residual: np.ndarray,
    recent: bool,
) -> np.ndarray:
    """Return the calibrator's inputs for the rows of a statistics table of one series."""
    target = np.asarray(common.grid_index(times=table["valid_time"]))
    origin = np.asarray(common.grid_index(times=table["power_fcst_init_time"]))
    columns = [table.select(STAT_COLUMNS).to_numpy().astype(np.float32)]
    columns.append(((target - origin) * 0.5).astype(np.float32)[:, None])
    if recent:
        columns.append(
            common.recent_origin_inputs(
                power=power, residual=residual, target=target, origin=origin
            )
        )
    return np.concatenate(columns, axis=1)


def select_rows(
    *, table: pl.DataFrame, series_id: int, config: str, seed: int, kind: str
) -> pl.DataFrame:
    """Return one series' statistics for a configuration, seed, and kind, in time order."""
    return table.filter(
        (pl.col("time_series_id") == series_id)
        & (pl.col("config") == config)
        & (pl.col("seed") == seed)
        & (pl.col("kind") == kind)
    ).sort("power_fcst_init_time", "valid_time")


def main() -> None:
    """Fit and apply every calibrator, skipping arm and configuration files that exist."""
    series = common.series_table()
    ids = dict(zip(series["site"], series["time_series_id"], strict=True))
    calibration = pl.concat(
        [
            pl.read_parquet(p)
            for p in sorted((ens_stats.STATS_DIR / "calibration").glob("*.parquet"))
        ]
    )
    validation = pl.concat(
        [pl.read_parquet(p) for p in sorted((ens_stats.STATS_DIR / "validation").glob("*.parquet"))]
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for config, settings in common.CONFIGS.items():
        if not settings.calibrators:
            continue
        arrays = common.load_series_arrays(series=series, config=config)
        era5 = {
            seed: common.read_era5_pred(config=config, seed=seed, sites=list(ids))
            for seed in common.SEEDS
        }
        for arm, (kind, recent) in ARMS.items():
            target_path = OUT_DIR / f"{arm}_{config}.parquet"
            if target_path.exists():
                continue
            started = time.monotonic()
            pieces = []
            for seed in common.SEEDS:
                for site, series_id in ids.items():
                    power = arrays[site].power
                    residual = stage1b.residuals(power=power, prediction=era5[seed][site])

                    train = select_rows(
                        table=calibration, series_id=series_id, config=config, seed=seed, kind=kind
                    )
                    actual = common.take(
                        values=power, index=np.asarray(common.grid_index(times=train["valid_time"]))
                    )
                    keep = np.isfinite(actual)
                    x_train = _matrix(table=train, power=power, residual=residual, recent=recent)
                    model = xgb.train(
                        {**common.CALIBRATOR_PARAMETERS, "seed": seed},
                        xgb.DMatrix(x_train[keep], label=actual[keep]),
                        num_boost_round=common.CALIBRATOR_ROUNDS,
                    )
                    test = select_rows(
                        table=validation, series_id=series_id, config=config, seed=seed, kind=kind
                    )
                    test_actual = common.take(
                        values=power, index=np.asarray(common.grid_index(times=test["valid_time"]))
                    )
                    x_test = _matrix(table=test, power=power, residual=residual, recent=recent)
                    quantiles = np.sort(common.predict(model=model, features=x_test), axis=1)
                    pieces.append(
                        test.select(
                            "time_series_id", "power_fcst_init_time", "valid_time", "seed"
                        ).with_columns(
                            pl.DataFrame(
                                quantiles.astype(np.float32), schema=list(common.LEVEL_COLUMNS)
                            ),
                            actual=pl.Series(test_actual),
                        )
                    )
            pl.concat(pieces).write_parquet(target_path.with_suffix(".tmp"))
            target_path.with_suffix(".tmp").rename(target_path)
            print(f"{arm} {config}: {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
