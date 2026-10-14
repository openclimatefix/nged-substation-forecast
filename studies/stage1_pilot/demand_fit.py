"""Fit the demand experiment's models and write each validation run's ensemble-mean forecasts.

For each variant (V0 weather only, V1 plus calendar, V2 plus calendar and power lags; see
`demand_inputs`), one XGBoost model per series and arm, with the stage 1 pilot's hyperparameters
on the GPU and target = power over effective capacity:

- `ens_trained`: trained on the ENS control-member rows of every daily run, 2024-04 to 2025-06
  (`demand_build_ens.py`).
- `ens_trained_dedup` (V1, V2): the same, one row per valid time (the shortest lead), which stops
  a model with calendar columns memorising the calendar position from the repeated valid times.
- `s1_memberavg`: trained on ERA5 from 2019-09 to 14 days before the first validation run.
- `s1_short`: the same from 2024-04 only.
- `s1_recency`, `s1_nocovid` (V0, V1): `s1_memberavg` with weights that halve every two years of
  age, and without 2020-03 to 2021-07. Exploratory.

Each model is applied to every ENS member's weather and the 51 predictions are averaged. In V2 the
ERA5-trained rows draw one lead uniformly from 0.5 to 351 hours and null the lags that lead rules
out; the ENS-trained rows carry the lags the fold A frames hold.

Validation uses the 73 runs of row set `fold_a_every5d_v1` (every fifth day from 2025-07-01).

Output: `data/studies/per_study/stage1_pilot/demand/pred_every5d/seed<s>/init=<date>.parquet`, one
row per case with the observed power, the lead, and one column `<arm>__<variant>` per model,
skipped if it exists.

    uv run python studies/stage1_pilot/demand_fit.py
"""

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import demand_inputs as demand
import numpy as np
import polars as pl
import stage1_inputs as inputs
import xgboost as xgb

PRED_DIR: Final[Path] = demand.demand_subfolder(name="pred_every5d")
ENS_FEATURES_DIR: Final[Path] = demand.demand_subfolder(name="ens_every5d")
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
"""The stage 1 pilot's hyperparameters, every column offered to each tree."""

SHORT_START: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
COVID_START: Final[datetime] = datetime(2020, 3, 1, tzinfo=UTC)
COVID_END: Final[datetime] = datetime(2021, 8, 1, tzinfo=UTC)
"""The excluded regime is 2020-03-01 up to, not including, 2021-08-01."""
RECENCY_HALF_LIFE_YEARS: Final[float] = 2.0

ARMS: Final[dict[str, tuple[str, ...]]] = {
    "V0": ("ens_trained", "s1_memberavg", "s1_short", "s1_recency", "s1_nocovid"),
    "V1": (
        "ens_trained",
        "ens_trained_dedup",
        "s1_memberavg",
        "s1_short",
        "s1_recency",
        "s1_nocovid",
    ),
    "V2": ("ens_trained", "ens_trained_dedup", "s1_memberavg", "s1_short"),
}
"""The arms fitted in each variant."""


def column_name(*, arm: str, variant: str) -> str:
    """Return the prediction column of an arm in a variant."""
    return f"{arm}__{variant}"


def era5_training_rows(*, series: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Return the ERA5 training rows: weather, calendar, target, and lags nulled by a random lead.

    Args:
        series: `demand.series_table()`.
        seed: The fitting seed, which seeds the lead drawn for each row.

    Returns:
        One row per series and half-hour from 2019-09 to 14 days before the first validation run
        that has an observed power, with `target`, `valid_time`, `time_series_id`, the weather,
        the calendar, and the lags.
    """
    weather = pl.read_parquet(demand.INPUTS_DIR / "era5_features.parquet")
    observed = pl.read_parquet(demand.INPUTS_DIR / "observed.parquet")
    rows = (
        weather.join(observed, on=["time_series_id", "valid_time"], how="inner", validate="1:1")
        .filter(pl.col("valid_time") < inputs.STAGE1_TRAIN_END)
        .join(series.select("time_series_id", "effective_capacity_mw"), on="time_series_id")
        .with_columns(
            *demand.calendar_columns(), target=pl.col("power") / pl.col("effective_capacity_mw")
        )
        .sort("time_series_id", "valid_time")
    )
    lead = np.random.default_rng(seed).uniform(
        demand.MIN_LEAD_HOURS, demand.MAX_LEAD_HOURS, size=rows.height
    )
    return rows.with_columns(
        *[
            pl.when(pl.lit(hours) <= pl.Series(lead)).then(None).otherwise(pl.col(name)).alias(name)
            for name, hours in demand.LAG_HOURS.items()
        ]
    )


def ens_training_rows(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return the ENS control-member training rows with the lead and the target."""
    return (
        pl.read_parquet(ENS_FEATURES_DIR / "train.parquet")
        .filter(pl.col("power").is_finite())
        .join(series.select("time_series_id", "effective_capacity_mw"), on="time_series_id")
        .with_columns(
            target=pl.col("power") / pl.col("effective_capacity_mw"),
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0,
        )
    )


def _matrix(*, frame: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    return frame.select([pl.col(c).cast(pl.Float32) for c in columns]).to_numpy()


def _fit(
    *, rows: pl.DataFrame, columns: tuple[str, ...], seed: int, weights: np.ndarray | None = None
) -> xgb.Booster:
    return xgb.train(
        {**PARAMETERS, "seed": seed},
        xgb.DMatrix(
            _matrix(frame=rows, columns=columns), label=rows["target"].to_numpy(), weight=weights
        ),
        num_boost_round=N_ROUNDS,
    )


def _arm_rows(
    *, arm: str, era5: pl.DataFrame, ens: pl.DataFrame
) -> tuple[pl.DataFrame, np.ndarray | None]:
    """Return an arm's training rows and sample weights for one series (or all)."""
    if arm == "ens_trained":
        return ens, None
    if arm == "ens_trained_dedup":
        return ens.sort("lead_hours").unique(
            subset=["time_series_id", "valid_time"], keep="first", maintain_order=True
        ), None
    if arm == "s1_memberavg":
        return era5, None
    if arm == "s1_short":
        return era5.filter(pl.col("valid_time") >= SHORT_START), None
    if arm == "s1_nocovid":
        return era5.filter(
            (pl.col("valid_time") < COVID_START) | (pl.col("valid_time") >= COVID_END)
        ), None
    if arm == "s1_recency":
        age_years = (inputs.STAGE1_TRAIN_END - era5["valid_time"]).dt.total_days() / 365.25
        return era5, 0.5 ** (age_years.to_numpy() / RECENCY_HALF_LIFE_YEARS)
    msg = f"unknown arm {arm}"
    raise ValueError(msg)


def run_seed(*, seed: int, series: pl.DataFrame) -> None:
    """Fit every model for one seed and write each validation run's forecasts."""
    val_files = sorted((ENS_FEATURES_DIR / "val").glob("init=*.parquet"))
    out_dir = PRED_DIR / f"seed{seed}"
    if all((out_dir / p.name).exists() for p in val_files):
        return
    started = time.monotonic()
    ids = [int(i) for i in series["time_series_id"]]
    capacity = {
        int(r["time_series_id"]): float(r["effective_capacity_mw"])
        for r in series.iter_rows(named=True)
    }
    era5 = era5_training_rows(series=series, seed=seed)
    ens = ens_training_rows(series=series)
    boosters: dict[tuple[str, str, int], xgb.Booster] = {}
    for variant, arms in ARMS.items():
        columns = demand.VARIANTS[variant]
        for arm in arms:
            source = era5 if arm.startswith("s1_") else ens
            rows, weights = _arm_rows(arm=arm, era5=source, ens=source)
            for sid in ids:
                part = rows.filter(pl.col("time_series_id") == sid)
                w = (
                    weights[(rows["time_series_id"] == sid).to_numpy()]
                    if weights is not None
                    else None
                )
                if part.height < 100:
                    msg = f"series {sid} has {part.height} rows for {arm}"
                    raise ValueError(msg)
                boosters[arm, variant, sid] = _fit(rows=part, columns=columns, seed=seed, weights=w)
        print(f"seed {seed} {variant}: fitted in {time.monotonic() - started:.0f} s", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in val_files:
        target = out_dir / path.name
        if target.exists():
            continue
        frame = pl.read_parquet(path).sort("time_series_id", "valid_time", "ensemble_member")
        predictions: dict[str, np.ndarray] = {}
        for variant, arms in ARMS.items():
            for arm in arms:
                predictions[column_name(arm=arm, variant=variant)] = np.empty(frame.height)
        offset = 0
        for (key,), part in frame.partition_by(
            "time_series_id", as_dict=True, maintain_order=True
        ).items():
            sid = int(key)
            n = part.height
            for variant, arms in ARMS.items():
                matrix = xgb.DMatrix(_matrix(frame=part, columns=demand.VARIANTS[variant]))
                for arm in arms:
                    predictions[column_name(arm=arm, variant=variant)][offset : offset + n] = (
                        boosters[arm, variant, sid].predict(matrix) * capacity[sid]
                    )
            offset += n
        cases = (
            frame.with_columns(*[pl.Series(k, v) for k, v in predictions.items()])
            .group_by("time_series_id", "power_fcst_init_time", "valid_time", maintain_order=True)
            .agg(
                actual=pl.col("power").first(),
                members=pl.len(),
                **{k: pl.col(k).mean() for k in predictions},
            )
            .with_columns(
                lead_hours=(
                    pl.col("valid_time") - pl.col("power_fcst_init_time")
                ).dt.total_minutes()
                / 60.0
            )
        )
        cases.write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
    print(f"seed {seed}: done in {time.monotonic() - started:.0f} s", flush=True)


def main() -> None:
    """Fit every seed."""
    series = demand.series_table()
    for seed in inputs.SEEDS:
        run_seed(seed=seed, series=series)


if __name__ == "__main__":
    main()
