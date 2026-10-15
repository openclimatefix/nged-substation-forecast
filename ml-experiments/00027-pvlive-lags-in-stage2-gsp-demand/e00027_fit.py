"""Fit experiment 00027's stage 2 calibrators for the two GSPs and write their validation quantiles.

One multi-quantile XGBoost model per series, arm, and seed, as in experiment 00005's `l3` arm: the
stage 1a member statistics, the lead in hours, and the recent-origin inputs. The arms add PV_Live
columns older than one day (`e00027_common.pv_inputs`):

- `a0`: no PV_Live.
- `a1`: PV_Live level over the series' effective capacity.
- `a2`: PV_Live share of its own capacity.
- `a3`: the delta at each position plus the PV_Live level.

Output: `work/calibrated/<arm>.parquet` with the validation keys, `seed`, the 9 quantile columns,
and `actual`, skipped if it exists.

    uv run python ml-experiments/00027-pvlive-lags-in-stage2-gsp-demand/e00027_fit.py
"""

import time
from pathlib import Path
from typing import Final

import e00027_common as common
import numpy as np
import polars as pl
import xgboost as xgb

OUT_DIR: Final[Path] = common.WORK_DIR / "calibrated"
ARMS: Final[tuple[str, ...]] = ("a0", "a1", "a2", "a3")


def _stats(*, split: str, series_ids: list[int]) -> pl.DataFrame:
    """Return the stage 1a statistics of one split for the primary window and kind `a`."""
    paths = sorted((common.STATS_DIR / split).glob("init=*.parquet"))
    return pl.concat(
        [
            pl.read_parquet(path).filter(
                (pl.col("config") == common.CONFIG)
                & (pl.col("kind") == "a")
                & pl.col("time_series_id").is_in(series_ids)
            )
            for path in paths
        ]
    )


def matrix(
    *,
    table: pl.DataFrame,
    arm: str,
    power: np.ndarray,
    residual: np.ndarray,
    pv_level: np.ndarray,
    pv_share: np.ndarray,
) -> np.ndarray:
    """Return the calibrator's inputs for the rows of one series, seed, and split."""
    target = np.asarray(common.grid_index(times=table["valid_time"]))
    origin = np.asarray(common.grid_index(times=table["power_fcst_init_time"]))
    columns = [
        table.select(common.STAT_COLUMNS).to_numpy().astype(np.float32),
        ((target - origin) * 0.5).astype(np.float32)[:, None],
        common.recent_origin_inputs(power=power, residual=residual, target=target, origin=origin),
    ]
    if arm != "a0":
        columns.append(
            common.pv_inputs(
                arm=arm,
                pv_level=pv_level,
                pv_share=pv_share,
                residual=residual,
                target=target,
                origin=origin,
            )
        )
    return np.concatenate(columns, axis=1)


def main() -> None:
    """Fit and apply every calibrator, skipping arm files that exist."""
    series = common.series_table()
    ids = dict(zip(series["site"], series["time_series_id"], strict=True))
    power = common.load_power(series=series)
    pv = common.load_pv_live(series=series)
    sites = common.all_sites()
    era5 = {seed: common.read_era5_pred(seed=seed, sites=sites) for seed in common.SEEDS}
    calibration = _stats(split="calibration", series_ids=list(ids.values()))
    validation = _stats(split="validation", series_ids=list(ids.values()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        target_path = OUT_DIR / f"{arm}.parquet"
        if target_path.exists():
            continue
        started = time.monotonic()
        pieces = []
        for seed in common.SEEDS:
            for site, series_id in ids.items():
                residual = (power[site] - era5[seed][site]).astype(np.float32)
                args = {
                    "arm": arm,
                    "power": power[site],
                    "residual": residual,
                    "pv_level": pv[site]["level"],
                    "pv_share": pv[site]["share"],
                }

                def rows(
                    frame: pl.DataFrame, *, series_id: int = series_id, seed: int = seed
                ) -> pl.DataFrame:
                    return frame.filter(
                        (pl.col("time_series_id") == series_id) & (pl.col("seed") == seed)
                    ).sort("power_fcst_init_time", "valid_time")

                train = rows(calibration)
                actual = common.take(
                    values=power[site],
                    index=np.asarray(common.grid_index(times=train["valid_time"])),
                )
                keep = np.isfinite(actual)
                x_train = matrix(table=train, **args)
                model = xgb.train(
                    {**common.CALIBRATOR_PARAMETERS, "seed": seed},
                    xgb.DMatrix(x_train[keep], label=actual[keep]),
                    num_boost_round=common.CALIBRATOR_ROUNDS,
                )
                test = rows(validation)
                test_actual = common.take(
                    values=power[site],
                    index=np.asarray(common.grid_index(times=test["valid_time"])),
                )
                quantiles = np.sort(model.predict(xgb.DMatrix(matrix(table=test, **args))), axis=1)
                pieces.append(
                    test.select(
                        "time_series_id", "power_fcst_init_time", "valid_time", "seed"
                    ).with_columns(
                        pl.DataFrame(
                            quantiles.astype(np.float32),
                            schema=list(common.LEVEL_COLUMNS),
                            orient="row",
                        ),
                        actual=pl.Series(test_actual),
                    )
                )
        pl.concat(pieces).write_parquet(target_path.with_suffix(".tmp"))
        target_path.with_suffix(".tmp").rename(target_path)
        print(f"{arm}: {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
