"""Refit stage 1a with 9 quantile levels, for the full ERA5 history and for the ENS months only.

The model, columns, training window, and out-of-fold chain are study 2a's stage 1a
(`stage1_fit_stage1a.py`), with the squared-error objective replaced by XGBoost's multi-quantile
loss at the levels in `stage2_common.LEVELS`, so one model per series returns 9 quantiles of power
over effective capacity. Two variants, each for seeds 0, 1, 2:

- `full`: trained on ERA5 from 2019-09 to before 2025-06-17.
- `short`: trained on ERA5 from 2024-04 only, the months the ENS-trained models train on.

Outputs under `data/studies/per_study/stage2_pilot/stage1a_q/<variant>/`, each skipped if it
exists: `val/seed<s>/init=<date>.parquet` (keys and the 9 quantile columns for every validation row
and member) and `train_oof/seed<s>.parquet` (the forward-chained out-of-fold quantiles for the ENS
training rows from 2024-07, with a `month` column).

    uv run python studies/stage1_pilot/stage2_fit_stage1a_quantile.py [--smoke]
"""

import argparse
import time
from datetime import timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import stage1_inputs as inputs
import xgboost as xgb
from stage1_fit_stage1a import (
    OOF_MONTHS,
    month_label,
    observed_training_power,
    series_columns,
    training_rows,
)
from stage2_common import ANALYSIS_PATH, ENS_FEATURES_DIR, LEVEL_COLUMNS, LEVELS, STAGE2_ROOT

OUT_DIR: Final[Path] = STAGE2_ROOT / "stage1a_q"
VARIANTS: Final[tuple[str, ...]] = ("full", "short")
TRAINING_MONTHS: Final = OOF_MONTHS[3:]
"""2024-07 to 2025-06: the months stage 2 trains on (the short variant has too little history
before 2024-07 to produce an out-of-fold value)."""
N_ROUNDS: Final[int] = 500
PARAMETERS: Final[dict[str, object]] = {
    "objective": "reg:quantileerror",
    "quantile_alpha": LEVELS,
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "tree_method": "hist",
    "device": "cuda",
    "nthread": 4,
}


def _matrix(*, frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    return frame.select([pl.col(c).cast(pl.Float32) for c in columns]).to_numpy()


def fit(*, rows: pl.DataFrame, columns: list[str], seed: int, rounds: int) -> xgb.Booster:
    """Fit one series' multi-quantile model."""
    return xgb.train(
        {**PARAMETERS, "seed": seed},
        xgb.DMatrix(_matrix(frame=rows, columns=columns), label=rows["target"].to_numpy()),
        num_boost_round=rounds,
    )


def predict(*, booster: xgb.Booster, frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Return the 9 sorted quantile columns for every row (a fraction of capacity)."""
    values = np.sort(booster.predict(xgb.DMatrix(_matrix(frame=frame, columns=columns))), axis=1)
    return pl.DataFrame(values.astype(np.float32), schema=list(LEVEL_COLUMNS))


def run(*, smoke: bool) -> None:
    """Fit every variant and seed, writing validation and out-of-fold quantiles."""
    sites = inputs.site_table()
    columns = series_columns(sites=sites)
    analysis = pl.read_parquet(ANALYSIS_PATH)
    observed = observed_training_power(sites=sites)
    val_files = sorted((ENS_FEATURES_DIR / "val").glob("init=*.parquet"))
    train = pl.read_parquet(ENS_FEATURES_DIR / "train.parquet")
    rounds = 20 if smoke else N_ROUNDS
    out_dir = OUT_DIR / "_smoke" if smoke else OUT_DIR
    for variant in VARIANTS:
        for seed in inputs.SEEDS:
            oof_path = out_dir / variant / "train_oof" / f"seed{seed}.parquet"
            todo = [
                oof_path,
                *[out_dir / variant / "val" / f"seed{seed}" / p.name for p in val_files],
            ]
            if all(p.exists() for p in todo):
                continue
            started = time.monotonic()
            pre = {
                series: fit(
                    rows=training_rows(
                        series=series,
                        analysis=analysis,
                        observed=observed,
                        before=inputs.STAGE1_TRAIN_END,
                        arm=variant if variant == "short" else "era5",
                    ),
                    columns=columns[series],
                    seed=seed,
                    rounds=rounds,
                )
                for series in columns
            }
            print(
                f"{variant} seed {seed}: pre models in {time.monotonic() - started:.0f} s",
                flush=True,
            )
            for path in val_files:
                target = out_dir / variant / "val" / f"seed{seed}" / path.name
                if target.exists():
                    continue
                frame = pl.read_parquet(path)
                pieces = []
                for key, part in frame.partition_by("time_series_id", as_dict=True).items():
                    series = int(key[0])
                    pieces.append(
                        pl.concat(
                            [
                                part.select(*inputs.KEY_COLUMNS),
                                predict(booster=pre[series], frame=part, columns=columns[series]),
                            ],
                            how="horizontal",
                        )
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                pl.concat(pieces).write_parquet(target.with_suffix(".tmp"))
                target.with_suffix(".tmp").rename(target)
                if smoke:
                    return
            if not oof_path.exists():
                pieces = []
                for month in TRAINING_MONTHS:
                    next_month = month.replace(
                        year=month.year + month.month // 12, month=month.month % 12 + 1
                    )
                    cutoff = month - timedelta(days=inputs.PURGE_DAYS)
                    in_month = train.filter(
                        (pl.col("valid_time") >= month) & (pl.col("valid_time") < next_month)
                    )
                    for series in columns:
                        fitted = fit(
                            rows=training_rows(
                                series=series,
                                analysis=analysis,
                                observed=observed,
                                before=cutoff,
                                arm=variant if variant == "short" else "era5",
                            ),
                            columns=columns[series],
                            seed=seed,
                            rounds=rounds,
                        )
                        part = in_month.filter(pl.col("time_series_id") == series)
                        pieces.append(
                            pl.concat(
                                [
                                    part.select(*inputs.KEY_COLUMNS),
                                    predict(booster=fitted, frame=part, columns=columns[series]),
                                ],
                                how="horizontal",
                            ).with_columns(month=pl.lit(month_label(month=month)))
                        )
                    print(
                        f"{variant} seed {seed}: out-of-fold {month_label(month=month)}", flush=True
                    )
                oof_path.parent.mkdir(parents=True, exist_ok=True)
                pl.concat(pieces).write_parquet(oof_path.with_suffix(".tmp"))
                oof_path.with_suffix(".tmp").rename(oof_path)
            print(f"{variant} seed {seed}: done in {time.monotonic() - started:.0f} s", flush=True)


def main() -> None:
    """Parse arguments and fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="20 rounds, one validation file")
    run(smoke=parser.parse_args().smoke)


if __name__ == "__main__":
    main()
