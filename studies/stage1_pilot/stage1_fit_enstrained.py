"""Fit the ENS-trained arms on stage 1a's columns, with and without stage 1a's prediction.

Both arms use stage 1a's columns (`stage1_inputs.FEATURES`: no power lag, no lead input, no
calendar column), the ENS control member's weather as the training weather, the training months
April 2024 to June 2025 with the chosen cleaning, and production's hyperparameters at
`colsample_bytree` 1.0, one XGBoost model per series on the GPU, target = power over effective
capacity. They differ in one input:

- `ens_trained`: the stage 1a columns only. This is the paired control for every stage 1a arm: it
  differs from `s1_memberavg` only in the training weather and the training length.
- `ens_trained_plus_s1`: the same columns plus `stage1a_prediction`. On the training rows that is
  the forward-chained out-of-fold value; on validation rows, for every ensemble member, it is the
  pre-validation model applied to that member's weather.

Outputs under `data/studies/per_study/stage1_pilot/ens_trained/<arm>/seed<s>/init=<date>.parquet`
(keys and `power_fcst`), skipped if they exist. With `--stage noise`, fits `ens_trained` twice with
one seed on one series and writes the size of the difference to `noise_floor.md`.

    uv run python studies/stage1_pilot/stage1_fit_enstrained.py --stage all
"""

import argparse
import time
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import stage1_inputs as inputs
import xgboost as xgb

OUT_DIR: Final[Path] = inputs.OUTPUT_ROOT / "ens_trained"
STAGE_DIR: Final[Path] = inputs.OUTPUT_ROOT / "stage1a"
ARMS: Final[tuple[str, ...]] = ("ens_trained", "ens_trained_plus_s1")
ENS_FEATURES_DIR: Final[Path] = inputs.INPUTS_DIR / "ens_features"
EXTRA_COLUMN: Final[str] = "stage1a_prediction"
N_ROUNDS: Final[int] = 500
PARAMETERS: Final[dict[str, float | int | str]] = {
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cuda",
    "nthread": 4,
}
"""Production's hyperparameters with every column offered to each tree."""


def arm_columns(*, arm: str, technology: str) -> list[str]:
    """Return an arm's columns for one technology."""
    base = list(inputs.FEATURES[technology])
    return base if arm == "ens_trained" else [*base, EXTRA_COLUMN]


def training_rows(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return the cleaned ENS control-member training rows, power as a fraction of capacity."""
    ids = sites["time_series_id"].to_list()
    drops = pl.read_parquet(inputs.CLEANING_DROPS).rename({"time": "valid_time"})
    return (
        pl.read_parquet(ENS_FEATURES_DIR / "train.parquet")
        .filter(pl.col("time_series_id").is_in(ids) & pl.col("power").is_finite())
        .join(drops, on=["time_series_id", "valid_time"], how="anti")
        .join(sites.select("time_series_id", "effective_capacity_mw"), on="time_series_id")
        .with_columns(power=pl.col("power") / pl.col("effective_capacity_mw"))
    )


def _matrix(*, frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    return frame.select([pl.col(c).cast(pl.Float32) for c in columns]).to_numpy()


def fit_series(*, rows: pl.DataFrame, columns: list[str], seed: int) -> xgb.Booster:
    """Fit one series' model on its training rows."""
    return xgb.train(
        {**PARAMETERS, "seed": seed},
        xgb.DMatrix(_matrix(frame=rows, columns=columns), label=rows["power"].to_numpy()),
        num_boost_round=N_ROUNDS,
    )


def _with_extra(*, frame: pl.DataFrame, prediction: pl.DataFrame, column: str) -> pl.DataFrame:
    joined = frame.join(
        prediction.select(*inputs.KEY_COLUMNS, pl.col(column).alias(EXTRA_COLUMN)),
        on=list(inputs.KEY_COLUMNS),
        how="left",
        validate="1:1",
    )
    if joined.height != frame.height or joined[EXTRA_COLUMN].null_count():
        msg = "stage 1a predictions do not cover every row"
        raise ValueError(msg)
    return joined


def run_seed(*, seed: int, sites: pl.DataFrame, training: pl.DataFrame) -> None:
    """Fit both arms for one seed and write each validation run's per-member forecasts."""
    technology = {int(r["time_series_id"]): r["technology"] for r in sites.iter_rows(named=True)}
    capacity = {
        int(r["time_series_id"]): float(r["effective_capacity_mw"])
        for r in sites.iter_rows(named=True)
    }
    val_files = sorted((ENS_FEATURES_DIR / "val").glob("init=*.parquet"))
    if all((OUT_DIR / arm / f"seed{seed}" / p.name).exists() for arm in ARMS for p in val_files):
        return
    started = time.monotonic()
    oof = pl.read_parquet(STAGE_DIR / "train_oof" / f"seed{seed}.parquet")
    train_plus = _with_extra(frame=training, prediction=oof, column="s1_oof")
    boosters = {
        (arm, series): fit_series(
            rows=train_plus.filter(pl.col("time_series_id") == series),
            columns=arm_columns(arm=arm, technology=technology[series]),
            seed=seed,
        )
        for arm in ARMS
        for series in technology
    }
    print(
        f"seed {seed}: fitted {len(boosters)} models in {time.monotonic() - started:.0f} s",
        flush=True,
    )
    ids = list(technology)
    for path in val_files:
        stage1 = pl.read_parquet(STAGE_DIR / "val" / f"seed{seed}" / path.name)
        frame = _with_extra(
            frame=pl.read_parquet(path).filter(pl.col("time_series_id").is_in(ids)),
            prediction=stage1,
            column="s1_era5",
        )
        for arm in ARMS:
            target = OUT_DIR / arm / f"seed{seed}" / path.name
            if target.exists():
                continue
            pieces = []
            for series, part in frame.partition_by("time_series_id", as_dict=True).items():
                sid = int(series[0])
                names = arm_columns(arm=arm, technology=technology[sid])
                predicted = (
                    boosters[arm, sid].predict(xgb.DMatrix(_matrix(frame=part, columns=names)))
                    * capacity[sid]
                )
                pieces.append(
                    part.select(*inputs.KEY_COLUMNS).with_columns(
                        power_fcst=pl.Series(predicted, dtype=pl.Float32)
                    )
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            pl.concat(pieces).write_parquet(target.with_suffix(".tmp"))
            target.with_suffix(".tmp").rename(target)
    print(f"seed {seed}: done in {time.monotonic() - started:.0f} s", flush=True)


def noise_floor(*, sites: pl.DataFrame, training: pl.DataFrame) -> list[str]:
    """Fit the `ens_trained` arm twice with seed 0 on one series and compare the predictions."""
    row = sites.filter(pl.col("technology") == "solar").row(0, named=True)
    series = int(row["time_series_id"])
    names = arm_columns(arm="ens_trained", technology="solar")
    rows = training.filter(pl.col("time_series_id") == series)
    first, second = (fit_series(rows=rows, columns=names, seed=0) for _ in range(2))
    capacity = float(row["effective_capacity_mw"])
    errors = []
    differences = []
    for path in sorted((ENS_FEATURES_DIR / "val").glob("init=*.parquet"))[::4]:
        frame = pl.read_parquet(path).filter(
            (pl.col("time_series_id") == series) & pl.col("power").is_finite()
        )
        matrix = xgb.DMatrix(_matrix(frame=frame, columns=names))
        a, b = first.predict(matrix), second.predict(matrix)
        actual = frame["power"].to_numpy() / capacity
        errors.append((np.abs(a - actual), np.abs(b - actual)))
        differences.append(np.abs(a - b))
    mae_first = float(np.concatenate([e[0] for e in errors]).mean()) * 100
    mae_second = float(np.concatenate([e[1] for e in errors]).mean()) * 100
    spread = np.concatenate(differences) * 100
    return [
        "# GPU repeatability",
        "",
        (
            f"One solar farm, the `ens_trained` arm, seed 0, fitted twice on the GPU and scored "
            f"on every fourth validation run ({spread.size:,} member-half-hours)."
        ),
        "",
        (
            f"- MAE of the first fit: {mae_first:.4f} points of capacity; second fit: "
            f"{mae_second:.4f}; difference {mae_second - mae_first:+.4f} points."
        ),
        (
            f"- Mean absolute difference between the two predictions: {spread.mean():.4f} points "
            f"of capacity; maximum {spread.max():.3f}."
        ),
    ]


def main() -> None:
    """Fit every seed, or measure the GPU repeatability."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("noise", "all"), required=True)
    args = parser.parse_args()
    sites = inputs.site_table()
    training = training_rows(sites=sites)
    if args.stage == "noise":
        report = noise_floor(sites=sites, training=training)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "noise_floor.md").write_text("\n".join(report) + "\n")
        print("\n".join(report))
        return
    for seed in inputs.SEEDS:
        run_seed(seed=seed, sites=sites, training=training)


if __name__ == "__main__":
    main()
