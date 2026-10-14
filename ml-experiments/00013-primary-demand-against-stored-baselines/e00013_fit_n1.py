"""Fit N1 and its variants, and write their validation quantiles.

N1 is one XGBoost model per series with the 9-level quantile loss, trained on the ENS control
member of every daily run from 2024-04-01, valid times up to 2025-06-30 23:30, with recency
weights anchored at 2025-06-30. Its columns are experiment 00006's Primary set, the lead in hours,
and three power lags measured from the forecast origin (`e00013_common.origin_lags`). Variants:

- `n1`: all columns.
- `n1_nolag`: without the three lags.
- `n1_nocal`: without hour, weekday, and days to Christmas (the no-calendar control).

The validation quantiles are the weighted quantiles of the 51 members' 9 quantiles each (the
mixture); `n1` also writes the control member's 9 quantiles alone, as the `_control` arm.

Output: `work/arms/<variant>_seed<s>.parquet` (and `n1_control_seed<s>.parquet`), keyed by series,
run, valid time. Existing files are kept.

    uv run python ml-experiments/00013-primary-demand-against-stored-baselines/e00013_fit_n1.py \
        --variant n1 --seeds 0 1 2 [--smoke]
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00013_common as common
import numpy as np
import polars as pl
import xgboost as xgb

ARMS_DIR: Final[Path] = common.WORK_DIR / "arms"
VARIANTS: Final[tuple[str, ...]] = ("n1", "n1_nolag", "n1_nocal")
KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")


def variant_columns(*, variant: str) -> list[str]:
    """Return a variant's input columns in matrix order."""
    columns = [
        *common.feature_columns(calendar=variant != "n1_nocal"),
        *common.N1_EXTRA_COLUMNS,
    ]
    if variant != "n1_nolag":
        columns += list(common.LAG_COLUMNS)
    return columns


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
    *, row: dict, arrays: common.SeriesArrays, train: pl.DataFrame, variant: str, seed: int
) -> xgb.Booster:
    """Fit one series' quantile model for a variant and seed."""
    own = train.filter(
        (pl.col("time_series_id") == row["time_series_id"])
        & pl.col("power").is_finite()
        & (pl.col("valid_time") <= common.LAST_TRAINING_VALID)
    )
    own = common.with_derived(frame=own, latitude=row["latitude"], longitude=row["longitude"])
    own = add_lags(frame=own, power=arrays.power)
    columns = variant_columns(variant=variant)
    age_days = (common.LAST_TRAINING_VALID - own["valid_time"]).dt.total_seconds() / 86_400
    weights = 0.5 ** (age_days.to_numpy() / common.RECENCY_HALF_LIFE_DAYS)
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
    *, row: dict, arrays: common.SeriesArrays, model: xgb.Booster, variant: str
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Return the mixture quantiles of every validation run, and the control member's."""
    columns = variant_columns(variant=variant)
    mixtures, controls = [], []
    for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet")):
        run = pl.read_parquet(path).filter(pl.col("time_series_id") == row["time_series_id"])
        run = common.with_derived(frame=run, latitude=row["latitude"], longitude=row["longitude"])
        run = add_lags(frame=run, power=arrays.power).sort("valid_time", "ensemble_member")
        members = run["ensemble_member"].n_unique()
        if run["ensemble_member"].min() != 0:
            msg = "member 0, the control, must be the first member"
            raise ValueError(msg)
        matrix = run.select(columns).to_numpy().astype(np.float32)
        raw = model.predict(xgb.DMatrix(matrix)).reshape(-1, members, len(common.LEVELS))
        keys = run.select(*KEYS).unique(maintain_order=True)
        mixtures.append(
            keys.with_columns(
                pl.DataFrame(
                    mixture_quantiles(member_quantiles=raw),
                    schema=list(common.LEVEL_COLUMNS),
                    orient="row",
                )
            )
        )
        controls.append(
            keys.with_columns(
                pl.DataFrame(
                    np.sort(raw[:, 0, :], axis=1),
                    schema=list(common.LEVEL_COLUMNS),
                    orient="row",
                )
            )
        )
    return pl.concat(mixtures), pl.concat(controls)


def main() -> None:
    """Fit and write the requested variant for the requested seeds."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
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
        names = [f"{args.variant}_seed{seed}"]
        if args.variant == "n1":
            names.append(f"n1_control_seed{seed}")
        if all((out_dir / f"{n}.parquet").exists() for n in names):
            continue
        mix_pieces, control_pieces = [], []
        for row in series.iter_rows(named=True):
            started = time.monotonic()
            model = fit_series(
                row=row, arrays=arrays[row["site"]], train=train, variant=args.variant, seed=seed
            )
            mixture, control = predict_series(
                row=row, arrays=arrays[row["site"]], model=model, variant=args.variant
            )
            mix_pieces.append(mixture)
            control_pieces.append(control)
            print(
                f"{args.variant} seed {seed} {row['site']}: {time.monotonic() - started:.0f} s",
                flush=True,
            )
        outputs = {names[0]: mix_pieces}
        if args.variant == "n1":
            outputs[names[1]] = control_pieces
        for name, pieces in outputs.items():
            frame = pl.concat(pieces).with_columns(seed=pl.lit(seed).cast(pl.Int8))
            tmp = out_dir / f"{name}.tmp"
            frame.write_parquet(tmp)
            tmp.rename(out_dir / f"{name}.parquet")


if __name__ == "__main__":
    main()
