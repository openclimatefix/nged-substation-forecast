"""Fit N1H once per series and seed, and score it with ENS and with ERA5 weather at serving time.

N1H is experiment 00019's model: one XGBoost model per series with the 9-level quantile loss,
trained on the ENS control member up to 2025-06-30 23:30, a recency half-life of 0.5 years, the
00006 columns, the lead in hours, and three power lags measured from the forecast origin. Each fit
predicts the validation runs under nine arms: `ens` (the 51 members as in 00019), `era5_all` (all
seven weather columns replaced by ERA5 at the valid time, one value for every member), and
`era5_<column>` (one weather column replaced, the other six kept from the ENS members). The
validation quantiles are the weighted quantiles of the members' 9 quantiles each.

Output: `work/arms/<arm>_seed<s>.parquet`, keyed by series, run, valid time. Existing files stay.

    uv run python ml-experiments/00022-era5-weather-ceiling-primary-demand/e00022_fit.py \
        --seeds 0 1 2 [--smoke]
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00022_common as common
import numpy as np
import polars as pl
import xgboost as xgb

ARMS_DIR: Final[Path] = common.WORK_DIR / "arms"
KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
ARM_NAMES: Final[tuple[str, ...]] = (
    "ens",
    "era5_all",
    *(f"era5_{column}" for column in common.WEATHER_COLUMNS),
)


def input_columns() -> list[str]:
    """Return the input columns in matrix order."""
    return [
        *common.feature_columns(calendar=True),
        *common.N1_EXTRA_COLUMNS,
        *common.LAG_COLUMNS,
    ]


def replaced_columns(*, arm: str) -> tuple[str, ...]:
    """Return the weather columns an arm takes from ERA5."""
    if arm == "ens":
        return ()
    if arm == "era5_all":
        return common.WEATHER_COLUMNS
    return (arm.removeprefix("era5_"),)


def add_lags(*, frame: pl.DataFrame, power: np.ndarray) -> pl.DataFrame:
    """Add the three origin lags to a frame with a run's origin and valid times."""
    lags = common.origin_lags(
        power=power,
        target=np.asarray(common.grid_index(times=frame["valid_time"])),
        origin=np.asarray(common.grid_index(times=frame["power_fcst_init_time"])),
    )
    return frame.with_columns(
        pl.DataFrame(lags, schema=list(common.LAG_COLUMNS), orient="row").cast(pl.Float32)
    )


def mixture_quantiles(*, member_quantiles: np.ndarray) -> np.ndarray:
    """Pool members' quantiles into the mixture's 9 quantiles.

    Args:
        member_quantiles: Shape (cases, members, 9), each member's quantiles at `LEVELS`.

    Returns:
        Shape (cases, 9): the weighted quantiles of the pooled values, each member weighted equally
        and each level by its atom mass, sorted along the levels so none cross.
    """
    cases, members, levels = member_quantiles.shape
    values = member_quantiles.reshape(cases, members * levels)
    weights = np.tile(common.ATOM_MASSES, members) / members
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    cumulative = np.cumsum(weights[order], axis=1)
    picks = (cumulative[:, :, None] >= (common.LEVELS - 1e-9)).argmax(axis=1)
    pooled = np.take_along_axis(sorted_values, picks, axis=1)
    return np.sort(pooled, axis=1).astype(np.float32)


def fit_series(
    *, row: dict, arrays: common.SeriesArrays, train: pl.DataFrame, seed: int
) -> xgb.Booster:
    """Fit one series' quantile model for a seed."""
    if train["ensemble_member"].unique().to_list() != [0]:
        msg = "the training frame must hold the control member (0) only"
        raise ValueError(msg)
    own = train.filter(
        (pl.col("time_series_id") == row["time_series_id"])
        & pl.col("power").is_finite()
        & (pl.col("valid_time") <= common.LAST_TRAINING_VALID)
    )
    own = common.with_derived(frame=own, latitude=row["latitude"], longitude=row["longitude"])
    own = add_lags(frame=own, power=arrays.power)
    columns = input_columns()
    age_days = (common.LAST_TRAINING_VALID - own["valid_time"]).dt.total_seconds() / 86_400
    weights = 0.5 ** (age_days.to_numpy() / common.SHORT_HALF_LIFE_DAYS)
    return xgb.train(
        {**common.QUANTILE_PARAMETERS, "seed": seed},
        xgb.DMatrix(
            own.select(columns).to_numpy().astype(np.float32),
            label=(own["power"] / row["effective_capacity_mw"]).to_numpy(),
            weight=weights,
        ),
        num_boost_round=common.N_ROUNDS,
    )


def predict_series(
    *, row: dict, arrays: common.SeriesArrays, model: xgb.Booster, arms: tuple[str, ...]
) -> dict[str, pl.DataFrame]:
    """Return the mixture quantiles of every validation run under each arm."""
    columns = input_columns()
    era5_position = {
        column: common.feature_columns(calendar=True).index(column)
        for column in common.WEATHER_COLUMNS
    }
    pieces: dict[str, list[pl.DataFrame]] = {arm: [] for arm in arms}
    for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet")):
        run = pl.read_parquet(path).filter(pl.col("time_series_id") == row["time_series_id"])
        run = common.with_derived(frame=run, latitude=row["latitude"], longitude=row["longitude"])
        run = add_lags(frame=run, power=arrays.power).sort("valid_time", "ensemble_member")
        members = run["ensemble_member"].n_unique()
        if run.height != run.select(*KEYS).n_unique() * members:
            msg = "every valid time must hold every member"
            raise ValueError(msg)
        if run["ensemble_member"].min() != 0:
            msg = "member 0, the control, must be the first member"
            raise ValueError(msg)
        matrix = run.select(columns).to_numpy().astype(np.float32)
        index = np.asarray(common.grid_index(times=run["valid_time"]))
        era5 = arrays.features[index]
        keys = run.select(*KEYS).unique(maintain_order=True)
        for arm in arms:
            swapped = replaced_columns(arm=arm)
            use = matrix.copy()
            for column in swapped:
                use[:, columns.index(column)] = era5[:, era5_position[column]]
            if arm == "era5_all":
                # Every member holds the same inputs, so one member holds the whole forecast.
                use = use.reshape(-1, members, use.shape[1])[:, :1, :].reshape(-1, use.shape[1])
                width = 1
            else:
                width = members
            raw = model.predict(xgb.DMatrix(use)).reshape(-1, width, len(common.LEVELS))
            pieces[arm].append(
                keys.with_columns(
                    pl.DataFrame(
                        mixture_quantiles(member_quantiles=raw),
                        schema=list(common.LEVEL_COLUMNS),
                        orient="row",
                    )
                )
            )
    return {arm: pl.concat(parts) for arm, parts in pieces.items()}


def main() -> None:
    """Fit once per series and seed, and write every arm."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(common.SEEDS))
    parser.add_argument("--smoke", action="store_true", help="two series, in a scratch folder")
    args = parser.parse_args()
    out_dir = ARMS_DIR.with_name("arms_smoke") if args.smoke else ARMS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    series = common.series_table()
    if args.smoke:
        series = series.head(2)
    arrays = common.load_series_arrays(series=series)
    train = pl.read_parquet(common.ENS_TRAIN_PATH)
    for seed in args.seeds:
        if all((out_dir / f"{arm}_seed{seed}.parquet").exists() for arm in ARM_NAMES):
            continue
        collected: dict[str, list[pl.DataFrame]] = {arm: [] for arm in ARM_NAMES}
        for row in series.iter_rows(named=True):
            started = time.monotonic()
            model = fit_series(row=row, arrays=arrays[row["site"]], train=train, seed=seed)
            predicted = predict_series(
                row=row, arrays=arrays[row["site"]], model=model, arms=ARM_NAMES
            )
            for arm, frame in predicted.items():
                collected[arm].append(frame)
            print(f"seed {seed} {row['site']}: {time.monotonic() - started:.0f} s", flush=True)
        for arm, pieces in collected.items():
            frame = pl.concat(pieces).with_columns(seed=pl.lit(seed).cast(pl.Int8))
            tmp = out_dir / f"{arm}_seed{seed}.tmp"
            frame.write_parquet(tmp)
            tmp.rename(out_dir / f"{arm}_seed{seed}.parquet")


if __name__ == "__main__":
    main()
