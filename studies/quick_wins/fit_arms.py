"""Fit named arms of the quick-wins study, predict the sampled runs, and score them.

Each arm is 31 per-series XGBoost boosters, trained on the cached control-member frame (after the
arm's cleaning rule drops training rows) and run over every ensemble member of each cached
validation frame. The script writes each arm's member forecasts and its per-row scores, and skips
any arm whose scores already exist, so a crash costs one arm.

    uv run python studies/quick_wins/fit_arms.py --arms production a1_generators_without_lags \
        [--fold mid_2025_to_mid_2026] [--device cuda] [--seed 0]
"""

import argparse
import time
from datetime import timedelta
from pathlib import Path
from typing import Final

import arms as arm_definitions
import build_frames
import cleaning_rules
import inputs
import numpy as np
import polars as pl
import xgboost as xgb
from contracts.settings import Settings
from nged_data.storage import scan_cleaned_power
from per_run_scores import normalised_by_capacity, per_run_scores
from studies.power import scan_power

N_TREES: Final[int] = 500
LEARNING_RATE: Final[float] = 0.05
MAX_DEPTH: Final[int] = 6
MIN_CHILD_WEIGHT: Final[int] = 1
SUBSAMPLE: Final[float] = 0.8
"""The production hyperparameters, except `colsample_bytree`, which each arm sets."""

EARLY_STOPPING_ROUNDS: Final[int] = 50
EARLY_STOPPING_HOLDOUT: Final[timedelta] = timedelta(weeks=6)
"""Early stopping watches the last six weeks of the training window."""

MIN_TREES: Final[int] = 10
"""Fewest trees an early-stopped series is refitted with."""


def feature_matrix(*, frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Select the columns as Float32, coding an Enum column by the position of its value.

    This is production's coding (`xgboost_forecaster.forecaster._prepare_features`): the weekday
    column is an Enum, and its position in the declared list is the same in every frame. Nulls stay
    null, which XGBoost reads as missing.
    """
    return frame.select(
        [
            pl.col(c).to_physical().cast(pl.Float32)
            if isinstance(frame.schema[c], pl.Enum)
            else pl.col(c).cast(pl.Float32)
            for c in columns
        ]
    )


def arm_dir(*, fold_id: str, arm: arm_definitions.Arm, seed: int) -> Path:
    """Return the directory holding one arm's forecasts and scores."""
    return inputs.OUTPUT_ROOT / "arms" / fold_id / f"{arm.name}_seed{seed}"


def _training_drops(
    *,
    rule: cleaning_rules.CleaningRule,
    fold_id: str,
    window: inputs.FoldWindow,
    types: pl.DataFrame,
) -> pl.DataFrame:
    """Return the `(time_series_id, time)` pairs a rule drops, cached per fold and rule."""
    path = inputs.OUTPUT_ROOT / "cleaning" / fold_id / f"{rule}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    settings = Settings()
    power = (
        scan_cleaned_power(settings.cleaned_power_time_series_data_path, settings.storage_options)
        .filter(pl.col("time") <= window.train_end)
        .select("time_series_id", "time", "power")
        .collect()
    )
    dropped = cleaning_rules.rows_to_drop(
        rule=rule, power=power, types=types, train_start=window.train_start
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    dropped.write_parquet(path)
    return dropped


def _training_rows(
    *, arm: arm_definitions.Arm, window: inputs.FoldWindow, types: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read the cached training frame and drop rows the arm's cleaning rules name.

    Returns:
        The training rows with a `series_type` column, and a per-series table of how many rows
        the arm kept and dropped.
    """
    frame = (
        pl.read_parquet(build_frames.frames_dir(fold_id=window.fold_id) / "train.parquet")
        .filter(pl.col("power").is_not_null())
        .join(types, on="time_series_id")
    )
    kept = frame
    for rule in (arm.cleaning, *arm.extra_cleaning):
        if rule == "none":
            continue
        dropped = _training_drops(rule=rule, fold_id=window.fold_id, window=window, types=types)
        kept = kept.join(
            dropped.rename({"time": "valid_time"}),
            on=["time_series_id", "valid_time"],
            how="anti",
        )
    counts = (
        frame.group_by("time_series_id")
        .agg(rows_before_cleaning=pl.len())
        .join(
            kept.group_by("time_series_id").agg(rows_kept=pl.len()), on="time_series_id", how="left"
        )
        .with_columns(pl.col("rows_kept").fill_null(0))
    )
    return kept, counts


def _parameters(*, arm: arm_definitions.Arm, device: str, seed: int) -> dict[str, object]:
    """Return the XGBoost parameters: production's, with the arm's column subsampling."""
    return {
        "eta": LEARNING_RATE,
        "max_depth": MAX_DEPTH,
        "min_child_weight": MIN_CHILD_WEIGHT,
        "subsample": SUBSAMPLE,
        "colsample_bytree": arm.colsample_bytree,
        "device": device,
        "objective": arm.objective,
        "seed": seed,
    }


def _weights(
    *, group: pl.DataFrame, arm: arm_definitions.Arm, window: inputs.FoldWindow
) -> np.ndarray | None:
    """Return row weights that halve every `recency_half_life_days`, scaled to mean 1.

    Scaling to mean 1 keeps `min_child_weight` meaning what it means for unweighted rows.
    """
    if arm.recency_half_life_days is None:
        return None
    age_days = (window.train_end - group["valid_time"]).dt.total_seconds() / 86400
    weights = np.power(0.5, age_days.to_numpy() / arm.recency_half_life_days)
    return weights / weights.mean()


def _fit_series(
    *,
    group: pl.DataFrame,
    columns: list[str],
    arm: arm_definitions.Arm,
    window: inputs.FoldWindow,
    device: str,
    seed: int,
) -> xgb.Booster:
    """Fit one series' booster, with early stopping and a refit when the arm asks for it."""
    params = _parameters(arm=arm, device=device, seed=seed)
    features = feature_matrix(frame=group, columns=columns)
    label = group["power"].cast(pl.Float32)
    weights = _weights(group=group, arm=arm, window=window)
    rounds = N_TREES
    if arm.early_stopping:
        cutoff = window.train_end - EARLY_STOPPING_HOLDOUT
        is_fit = group["valid_time"] <= cutoff
        if is_fit.all() or not is_fit.any():
            msg = "early stopping needs rows both before and after the hold-out cutoff"
            raise ValueError(msg)
        fit_mask, hold_mask = is_fit.to_numpy(), (~is_fit).to_numpy()
        fit = xgb.DMatrix(
            features.filter(is_fit),
            label=label.filter(is_fit),
            weight=None if weights is None else weights[fit_mask],
        )
        hold = xgb.DMatrix(
            features.filter(~is_fit),
            label=label.filter(~is_fit),
            weight=None if weights is None else weights[hold_mask],
        )
        probe = xgb.train(
            params,
            fit,
            num_boost_round=N_TREES,
            evals=[(hold, "hold")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        rounds = max(MIN_TREES, probe.best_iteration + 1)
    return xgb.train(
        params, xgb.DMatrix(features, label=label, weight=weights), num_boost_round=rounds
    )


def _predict_and_score(
    *,
    boosters: dict[int, tuple[xgb.Booster, list[str]]],
    out_dir: Path,
    fold_id: str,
    types: pl.DataFrame,
    actuals: pl.DataFrame,
    capacity: pl.DataFrame,
) -> pl.DataFrame:
    """Predict every ensemble member of each cached run, write the forecasts, and score them."""
    (out_dir / "forecasts").mkdir(parents=True, exist_ok=True)
    parts = []
    for path in sorted((build_frames.frames_dir(fold_id=fold_id) / "val").glob("init=*.parquet")):
        frame = pl.read_parquet(path).join(types, on="time_series_id")
        pieces = []
        for (series_id,), series_rows in frame.group_by("time_series_id"):
            booster, columns = boosters[int(series_id)]
            matrix = xgb.DMatrix(feature_matrix(frame=series_rows, columns=columns))
            pieces.append(
                series_rows.select(
                    "time_series_id",
                    "power_fcst_init_time",
                    "nwp_init_time",
                    "valid_time",
                    "ensemble_member",
                ).with_columns(power_fcst=pl.Series(booster.predict(matrix), dtype=pl.Float32))
            )
        forecasts = pl.concat(pieces)
        if forecasts.height != frame.height:
            msg = f"{path.name}: predicted {forecasts.height:,} rows for {frame.height:,}"
            raise ValueError(msg)
        forecasts.write_parquet(out_dir / "forecasts" / path.name)
        parts.append(per_run_scores(forecasts=forecasts, actuals=actuals))
    return normalised_by_capacity(scores=pl.concat(parts), capacity=capacity)


def fit_arm(*, arm: arm_definitions.Arm, fold_id: str, device: str, seed: int) -> None:
    """Fit, predict, and score one arm, writing its forecasts, scores, and training-row counts."""
    out_dir = arm_dir(fold_id=fold_id, arm=arm, seed=seed)
    scores_path = out_dir / "scores.parquet"
    if scores_path.exists():
        print(f"{arm.name}: scores exist, skipping")
        return
    started = time.monotonic()
    settings = Settings()
    window = inputs.window_for(fold_id=fold_id)
    series_ids = inputs.eligible_series_ids(settings=settings, fold_id=inputs.FOLD_A)
    metadata = inputs.series_metadata(settings=settings, time_series_ids=series_ids)
    types = pl.DataFrame(metadata).select(
        "time_series_id", pl.col("time_series_type").cast(pl.String)
    )
    capacity = pl.DataFrame(inputs.effective_capacity(settings=settings))
    actuals = (
        scan_power()
        .filter(pl.col("time") >= window.val_start, pl.col("time") <= window.val_end)
        .collect()
    )

    training, counts = _training_rows(arm=arm, window=window, types=types)
    boosters: dict[int, tuple[xgb.Booster, list[str]]] = {}
    for (series_id,), group in training.group_by("time_series_id"):
        series_type = group["time_series_type"][0]
        columns = list(arm.columns(series_type))
        booster = _fit_series(
            group=group, columns=columns, arm=arm, window=window, device=device, seed=seed
        )
        boosters[int(series_id)] = (booster, columns)
    if set(boosters) != set(series_ids):
        msg = f"{arm.name}: trained {len(boosters)} of {len(series_ids)} series"
        raise ValueError(msg)
    trained_at = time.monotonic()

    scores = _predict_and_score(
        boosters=boosters,
        out_dir=out_dir,
        fold_id=fold_id,
        types=types,
        actuals=actuals,
        capacity=capacity,
    )
    counts.write_parquet(out_dir / "train_rows.parquet")
    scores.write_parquet(scores_path.with_suffix(".tmp"))
    scores_path.with_suffix(".tmp").rename(scores_path)
    print(
        f"{arm.name}: trained in {trained_at - started:.0f} s, "
        f"predicted and scored in {time.monotonic() - trained_at:.0f} s"
    )


def main() -> None:
    """Fit the named arms in order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--fold", default=inputs.FOLD_A)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    unknown = [name for name in args.arms if name not in arm_definitions.ALL_ARMS]
    if unknown:
        msg = f"unknown arms {unknown}; known: {sorted(arm_definitions.ALL_ARMS)}"
        raise SystemExit(msg)
    for name in args.arms:
        fit_arm(
            arm=arm_definitions.ALL_ARMS[name],
            fold_id=args.fold,
            device=args.device,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
