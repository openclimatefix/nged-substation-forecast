"""Refit the ENS-trained control with and without stage 1a's prediction as a 25th column.

Both arms use the fixed-pipeline control's 24 production columns, its training rows (control
member, April 2024 to June 2025, with the chosen cleaning), and production's hyperparameters at
`colsample_bytree` 1.0, one XGBoost model per series on the GPU. The only difference between them
is the extra column `stage1a_prediction`:

- `c0_refit`: the 24 columns. It reproduces the stored control from the same frames, so it checks
  this fitting loop against the stored fits and is the paired control for the extra column.
- `c0_plus_s1`: the 24 columns plus stage 1a's prediction. On the training rows it is the
  forward-chained out-of-fold value; on validation rows, for every ensemble member, it is the
  pre-validation model applied to that member's weather.
- `s1_recipe_on_ens`: stage 1a's columns, target (power over capacity), and hyperparameters, with
  the control member's ENS weather in place of ERA5 as the training weather. It separates the
  effect of training on ERA5 from the effect of the stage 1a recipe (fewer columns, no lags).

Outputs under `data/studies/per_study/stage1_pilot/ens_trained/<arm>/seed<s>/init=<date>.parquet`
(keys and `power_fcst`), skipped if they exist. With `--stage noise`, fits one arm twice with one
seed on one series and writes the size of the difference to `noise_floor.md`.

    uv run python studies/stage1_pilot/stage1_fit_enstrained.py --stage all
"""

import argparse
import json
import time
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import stage1_inputs as inputs
import xgboost as xgb

OUT_DIR: Final[Path] = inputs.OUTPUT_ROOT / "ens_trained"
STAGE_DIR: Final[Path] = inputs.OUTPUT_ROOT / "stage1a"
ARMS: Final[tuple[str, ...]] = ("c0_refit", "c0_plus_s1", "s1_recipe_on_ens")
RECIPE_ARM: Final[str] = "s1_recipe_on_ens"
ENS_FEATURES_DIR: Final[Path] = inputs.OUTPUT_ROOT / "inputs" / "ens_features"
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
"""Production's hyperparameters with every column offered to each tree, as the stored control."""


def control_columns() -> dict[str, list[str]]:
    """Return the control arm's 24 columns for PV and wind, in the stored fit's order."""
    stored = json.loads((inputs.CONTROL_FITS_DIR / "cuda_seed0_none" / "features.json").read_text())
    return {"solar": stored["PV"], "wind": stored["Wind"]}


def cleaned_training(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Read the control member's training rows, drop the chosen cleaning's rows, check counts.

    Returns:
        The cleaned training rows.
    """
    ids = sites["time_series_id"].to_list()
    frame = pl.read_parquet(inputs.FRAMES_DIR / "train.parquet").filter(
        pl.col("time_series_id").is_in(ids) & pl.col("power").is_not_null()
    )
    drops = pl.read_parquet(inputs.CLEANING_DROPS).rename({"time": "valid_time"})
    kept = frame.join(drops, on=["time_series_id", "valid_time"], how="anti")
    stored = pl.read_parquet(
        inputs.CONTROL_FITS_DIR / "cuda_seed0_none" / "training_counts.parquet"
    )
    expected = stored.filter(pl.col("time_series_id").is_in(ids)).sort("time_series_id")
    counts = kept.group_by("time_series_id").agg(rows=pl.len()).sort("time_series_id")
    if not counts["rows"].equals(expected["rows_kept"].cast(counts["rows"].dtype)):
        msg = "cleaned training rows differ from the stored control's training counts"
        raise ValueError(msg)
    return kept


def _matrix(*, frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    return frame.select([pl.col(c).cast(pl.Float32) for c in columns]).to_numpy()


def fit_series(*, rows: pl.DataFrame, columns: list[str], seed: int) -> xgb.Booster:
    """Fit one series' model on its finite-power training rows."""
    usable = rows.filter(pl.col("power").is_finite())
    return xgb.train(
        {**PARAMETERS, "seed": seed},
        xgb.DMatrix(_matrix(frame=usable, columns=columns), label=usable["power"].to_numpy()),
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


def recipe_training(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return the ENS control-member training rows in stage 1a's columns, with the same cleaning."""
    ids = sites["time_series_id"].to_list()
    drops = pl.read_parquet(inputs.CLEANING_DROPS).rename({"time": "valid_time"})
    return (
        pl.read_parquet(ENS_FEATURES_DIR / "train.parquet")
        .filter(pl.col("time_series_id").is_in(ids) & pl.col("power").is_finite())
        .join(drops, on=["time_series_id", "valid_time"], how="anti")
        .join(sites.select("time_series_id", "effective_capacity_mw"), on="time_series_id")
        .with_columns(power=pl.col("power") / pl.col("effective_capacity_mw"))
    )


def run_seed(*, seed: int, sites: pl.DataFrame, training: pl.DataFrame) -> None:
    """Fit both arms for one seed and write each validation run's per-member forecasts."""
    columns = control_columns()
    technology = {int(r["time_series_id"]): r["technology"] for r in sites.iter_rows(named=True)}
    val_files = sorted((inputs.FRAMES_DIR / "val").glob("init=*.parquet"))
    if all((OUT_DIR / arm / f"seed{seed}" / p.name).exists() for arm in ARMS for p in val_files):
        return
    started = time.monotonic()
    oof = pl.read_parquet(STAGE_DIR / "train_oof" / f"seed{seed}.parquet")
    train_plus = _with_extra(frame=training, prediction=oof, column="s1_oof")
    boosters: dict[tuple[str, int], xgb.Booster] = {}
    for series in technology:
        rows = train_plus.filter(pl.col("time_series_id") == series)
        base = columns[technology[series]]
        boosters["c0_refit", series] = fit_series(rows=rows, columns=base, seed=seed)
        boosters["c0_plus_s1", series] = fit_series(
            rows=rows, columns=[*base, EXTRA_COLUMN], seed=seed
        )
    recipe_rows = recipe_training(sites=sites)
    recipe_boosters = {
        series: fit_series(
            rows=recipe_rows.filter(pl.col("time_series_id") == series),
            columns=list(inputs.FEATURES[technology[series]]),
            seed=seed,
        )
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
            source = (
                pl.read_parquet(ENS_FEATURES_DIR / "val" / path.name).filter(
                    pl.col("time_series_id").is_in(ids)
                )
                if arm == RECIPE_ARM
                else frame
            )
            for series, part in source.partition_by("time_series_id", as_dict=True).items():
                sid = int(series[0])
                if arm == RECIPE_ARM:
                    recipe = list(inputs.FEATURES[technology[sid]])
                    predicted = recipe_boosters[sid].predict(
                        xgb.DMatrix(_matrix(frame=part, columns=recipe))
                    ) * float(
                        sites.filter(pl.col("time_series_id") == sid)[
                            "effective_capacity_mw"
                        ].item()
                    )
                    pieces.append(
                        part.select(*inputs.KEY_COLUMNS).with_columns(
                            power_fcst=pl.Series(predicted, dtype=pl.Float32)
                        )
                    )
                    continue
                names = columns[technology[sid]]
                names = names if arm == "c0_refit" else [*names, EXTRA_COLUMN]
                predicted = boosters[arm, sid].predict(
                    xgb.DMatrix(_matrix(frame=part, columns=names))
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
    """Fit the `c0_refit` arm twice with seed 0 on one series and compare the two predictions."""
    columns = control_columns()
    row = sites.filter(pl.col("technology") == "solar").row(0, named=True)
    series = int(row["time_series_id"])
    names = columns["solar"]
    rows = training.filter(pl.col("time_series_id") == series)
    first, second = (fit_series(rows=rows, columns=names, seed=0) for _ in range(2))
    errors = []
    differences = []
    for path in sorted((inputs.FRAMES_DIR / "val").glob("init=*.parquet"))[::4]:
        frame = pl.read_parquet(path).filter(
            (pl.col("time_series_id") == series) & pl.col("power").is_not_null()
        )
        matrix = xgb.DMatrix(_matrix(frame=frame, columns=names))
        a, b = first.predict(matrix), second.predict(matrix)
        capacity = row["effective_capacity_mw"]
        actual = frame["power"].to_numpy()
        errors.append((np.abs(a - actual) / capacity, np.abs(b - actual) / capacity))
        differences.append(np.abs(a - b) / capacity)
    mae_first = float(np.concatenate([e[0] for e in errors]).mean()) * 100
    mae_second = float(np.concatenate([e[1] for e in errors]).mean()) * 100
    spread = np.concatenate(differences) * 100
    return [
        "# GPU repeatability",
        "",
        (
            f"One solar farm, the `c0_refit` arm, seed 0, fitted twice on the GPU and scored on "
            f"every "
            f"fourth validation run ({spread.size:,} member-half-hours)."
        ),
        "",
        (
            f"- MAE of the first fit: {mae_first:.4f} points of capacity; second fit: "
            f"{mae_second:.4f}; "
            f"difference {mae_second - mae_first:+.4f} points."
        ),
        (
            f"- Mean absolute difference between the two predictions: {spread.mean():.4f} points "
            f"of "
            f"capacity; maximum {spread.max():.3f}."
        ),
    ]


def main() -> None:
    """Fit every seed, or measure the GPU repeatability."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("noise", "all"), required=True)
    args = parser.parse_args()
    sites = inputs.site_table()
    training = cleaned_training(sites=sites)
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
