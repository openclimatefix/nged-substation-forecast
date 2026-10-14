"""Fit experiment 00008's stage 2 calibrators and write their validation quantiles.

One multi-quantile XGBoost model per series, seed, and arm, trained on experiment 00005's stage 1a
statistics (read, not refitted) with observed power over effective capacity as the target. Inputs
of every arm: the mean, standard deviation, and 10th, 50th, and 90th percentiles of the 51 member
forecasts, and the lead in hours. No calendar column. The arms differ in the recent-power inputs:

- `l3`: 00005's recent-origin inputs (5 residuals, 4 power lags, 4 lag anomalies).
- `s1`: `l3` plus the 6 summaries.
- `s2`: the 5 residuals, the 4 lag anomalies, and the 6 summaries.
- `s3`: the 5 residuals and the 6 summaries.
- `s4`: the 6 summaries only.

Output: `work/calibrated/<arm>.parquet` with the validation keys, `seed`, the 9 quantile columns,
and `actual`, skipped if it exists.

    uv run python ml-experiments/00008-lag-summaries-in-stage2-calibrator-demand/e00008_fit.py
"""

import time
from pathlib import Path
from typing import Final

import e00008_common as common
import numpy as np
import polars as pl
import xgboost as xgb

OUT_DIR: Final[Path] = common.WORK_DIR / "calibrated"
ARM_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "l3": (*common.RESIDUAL_COLUMNS, *common.POWER_LAG_COLUMNS, *common.ANOMALY_COLUMNS),
    "s1": (
        *common.RESIDUAL_COLUMNS,
        *common.POWER_LAG_COLUMNS,
        *common.ANOMALY_COLUMNS,
        *common.SUMMARY_COLUMNS,
    ),
    "s2": (*common.RESIDUAL_COLUMNS, *common.ANOMALY_COLUMNS, *common.SUMMARY_COLUMNS),
    "s3": (*common.RESIDUAL_COLUMNS, *common.SUMMARY_COLUMNS),
    "s4": common.SUMMARY_COLUMNS,
}
"""Each arm's recent-power inputs, as columns of `common.RECENT_COLUMNS`."""
STAT_COLUMNS: Final[tuple[str, ...]] = ("mean", "std", "q10", "q50", "q90")
STAGE1_KIND: Final[str] = "a"
"""Every arm summarises stage 1a, the lag-free stage 1, as 00005's L3 and R0c did."""


def _matrix(
    *,
    table: pl.DataFrame,
    recent: np.ndarray,
    columns: tuple[str, ...],
    origin: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Return the calibrator's inputs for the rows of a statistics table of one series."""
    chosen = [common.RECENT_COLUMNS.index(c) for c in columns]
    return np.concatenate(
        [
            table.select(STAT_COLUMNS).to_numpy().astype(np.float32),
            ((target - origin) * 0.5).astype(np.float32)[:, None],
            recent[:, chosen],
        ],
        axis=1,
    )


def select_rows(*, table: pl.DataFrame, series_id: int, seed: int) -> pl.DataFrame:
    """Return one series' stage 1a statistics for the primary configuration and a seed."""
    return table.filter(
        (pl.col("time_series_id") == series_id)
        & (pl.col("config") == common.CONFIG)
        & (pl.col("seed") == seed)
        & (pl.col("kind") == STAGE1_KIND)
    ).sort("power_fcst_init_time", "valid_time")


def main() -> None:
    """Fit and apply every arm's calibrators, skipping arms whose file exists."""
    series = common.series_table()
    ids = dict(zip(series["site"], series["time_series_id"], strict=True))
    split = {
        name: pl.concat(
            [pl.read_parquet(p) for p in sorted((common.STATS_DIR / name).glob("*.parquet"))]
        )
        for name in ("calibration", "validation")
    }
    power = common.load_power(series=series)
    era5 = {seed: common.read_era5_pred(seed=seed, sites=list(ids)) for seed in common.SEEDS}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # The recent inputs depend on the case, not on the arm, so build them once per series and seed.
    cases = {}
    for seed in common.SEEDS:
        for site, series_id in ids.items():
            residual = (power[site] - era5[seed][site]).astype(np.float32)
            for name in ("calibration", "validation"):
                rows = select_rows(table=split[name], series_id=series_id, seed=seed)
                target = np.asarray(common.grid_index(times=rows["valid_time"]))
                origin = np.asarray(common.grid_index(times=rows["power_fcst_init_time"]))
                cases[name, seed, site] = (
                    rows,
                    target,
                    origin,
                    common.recent_inputs(
                        power=power[site], residual=residual, target=target, origin=origin
                    ),
                    common.take(values=power[site], index=target),
                )
    for arm, columns in ARM_COLUMNS.items():
        target_path = OUT_DIR / f"{arm}.parquet"
        if target_path.exists():
            continue
        started = time.monotonic()
        pieces = []
        for seed in common.SEEDS:
            for site in ids:
                rows, target, origin, recent, actual = cases["calibration", seed, site]
                keep = np.isfinite(actual)
                x_train = _matrix(
                    table=rows, recent=recent, columns=columns, origin=origin, target=target
                )
                model = xgb.train(
                    {**common.CALIBRATOR_PARAMETERS, "seed": seed},
                    xgb.DMatrix(x_train[keep], label=actual[keep]),
                    num_boost_round=common.CALIBRATOR_ROUNDS,
                )
                rows, target, origin, recent, actual = cases["validation", seed, site]
                x_test = _matrix(
                    table=rows, recent=recent, columns=columns, origin=origin, target=target
                )
                quantiles = np.sort(model.predict(xgb.DMatrix(x_test)), axis=1)
                pieces.append(
                    rows.select(
                        "time_series_id", "power_fcst_init_time", "valid_time", "seed"
                    ).with_columns(
                        pl.DataFrame(
                            quantiles.astype(np.float32),
                            schema=list(common.LEVEL_COLUMNS),
                            orient="row",
                        ),
                        actual=pl.Series(actual),
                    )
                )
        pl.concat(pieces).write_parquet(target_path.with_suffix(".tmp"))
        target_path.with_suffix(".tmp").rename(target_path)
        print(f"{arm}: {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
