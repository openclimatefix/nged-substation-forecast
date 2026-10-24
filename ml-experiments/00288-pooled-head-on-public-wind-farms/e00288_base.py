"""Fit R0 (experiment 230's A0r recipe) at each public wind farm and write the member mixtures.

For each kept farm (`work/clean/farms.parquet`), fits an XGBoost quantile model with the binned
power-curve base margin on ERA5 wind at the farm's cell before 2024-04-01, runs it on the 51 ENS
members of every hindcast and scored case, mixes the members with equal weights, and applies
experiment 33's spread factor, fitted month by month on earlier cases (pooled over the farms, with
the hindcast and scored cases in one time-ordered pass). The target is the farm's clean output over
its capacity at the same instant (`work/clean/actual_centred.parquet`).

`--history-months M` trains on the last M months before 2024-04-01 only (the learning curve);
without it the model trains on every ERA5 hour from 2019-09-01. Output, in `work/base/`:
`R0_<set>_h<M|all>.parquet` (quantiles per case), and `cases_<set>.parquet` (the cases with the
farm, lead, observed fraction, and the member summaries the heads read).

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
      uv run --no-sync python ml-experiments/00288-*/e00288_base.py [--history-months M]
"""

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00288_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import xgboost as xgb  # noqa: E402
from studies.sources import REPO_DATA_DIR  # noqa: E402

CLEAN_DIR: Final = common.WORK_DIR / "clean"
ENS_DIR: Final = common.WORK_DIR / "ens"
OUT_DIR: Final = common.WORK_DIR / "base"
ERA5_PATH: Final = REPO_DATA_DIR / "studies/downloads/reanalysis/ERA5/farm_cells_00288"
BASE_END: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
SETS: Final[tuple[str, ...]] = ("hindcast", "scored")
N_ROUNDS: Final[int] = 500
SPEED_BIN_WIDTH_M_S: Final[float] = 0.5
XGBOOST_PARAMETERS: Final[dict] = {
    "objective": "reg:quantileerror",
    "quantile_alpha": common.LEVELS,
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "tree_method": "hist",
    "device": "cuda",
    "nthread": 4,
}
SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "ws_mean",
    "ws_std",
    "ws_p10",
    "ws_p90",
    "dir_sin",
    "dir_cos",
    "temp_mean",
)
FEATURES: Final[tuple[str, ...]] = common.XGBOOST_FEATURES


def direction_features(*, u: pl.Expr, v: pl.Expr, height: str) -> list[pl.Expr]:
    """Return speed, and the sine and cosine of the direction the wind blows from."""
    bearing = ((270.0 - pl.arctan2(v, u).degrees()) % 360.0).radians()
    return [
        (u**2 + v**2).sqrt().alias(f"wind_speed_{height}m"),
        bearing.sin().alias(f"wind_dir_sin_{height}m"),
        bearing.cos().alias(f"wind_dir_cos_{height}m"),
    ]


def era5_features() -> pl.DataFrame:
    """Return ERA5's hourly features per cell: `cell_id`, `valid_time`, and `FEATURES`."""
    frame = pl.read_parquet(ERA5_PATH / "era5_farm_cells.parquet")
    return frame.select(
        "cell_id",
        pl.col("time").alias("valid_time"),
        *direction_features(u=pl.col("u10"), v=pl.col("v10"), height="10"),
        *direction_features(u=pl.col("u100"), v=pl.col("v100"), height="100"),
    ).with_columns(pl.col(FEATURES).cast(pl.Float32))


def binned_curve(*, speed: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a non-decreasing power curve of the fraction on 100 m wind speed, by 0.5 m/s bin."""
    bins = np.floor(speed / SPEED_BIN_WIDTH_M_S).astype(np.int64)
    ordered = (
        pl.DataFrame({"bin": bins, "power": target})
        .group_by("bin")
        .agg(pl.col("power").mean())
        .sort("bin")
    )
    centres = (ordered["bin"].to_numpy() + 0.5) * SPEED_BIN_WIDTH_M_S
    return centres, np.maximum.accumulate(ordered["power"].to_numpy())


def xgboost_booster(*, train: pl.DataFrame, seed: int) -> tuple[xgb.Booster, tuple]:
    """Return the fitted XGBoost booster and the binned power curve it uses as a base margin."""
    target = train["target"].to_numpy()
    speed = train["wind_speed_100m"].to_numpy()
    centres, curve = binned_curve(speed=speed, target=target)
    booster = xgb.train(
        {**XGBOOST_PARAMETERS, "seed": seed},
        xgb.DMatrix(
            train.select(FEATURES).to_numpy(),
            label=target,
            base_margin=np.tile(np.interp(speed, centres, curve)[:, None], (1, 11)),
        ),
        num_boost_round=N_ROUNDS,
    )
    return booster, (centres, curve)


def xgboost_quantiles(*, booster: xgb.Booster, curve: tuple, rows: pl.DataFrame) -> np.ndarray:
    """Return shape (rows, 11) quantiles of the XGBoost model on member rows."""
    centres, values = curve
    margin = np.tile(
        np.interp(rows["wind_speed_100m"].to_numpy(), centres, values)[:, None], (1, 11)
    )
    predicted = booster.predict(xgb.DMatrix(rows.select(FEATURES).to_numpy(), base_margin=margin))
    return np.sort(np.clip(predicted, 0.0, 1.0), axis=1).astype(np.float32)


def member_summaries(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return per-case summaries of the members: 100 m wind, mean direction, temperature."""
    speed = rows["wind_speed_100m"].to_numpy().reshape(-1, common.MEMBERS).astype(np.float64)
    sine = rows["wind_dir_sin_100m"].to_numpy().reshape(-1, common.MEMBERS).mean(axis=1)
    cosine = rows["wind_dir_cos_100m"].to_numpy().reshape(-1, common.MEMBERS).mean(axis=1)
    norm = np.maximum(np.hypot(sine, cosine), 1e-6)
    temperature = rows["temperature_c"].to_numpy().reshape(-1, common.MEMBERS).mean(axis=1)
    return pl.DataFrame(
        {
            "ws_mean": speed.mean(axis=1),
            "ws_std": speed.std(axis=1),
            "ws_p10": np.quantile(speed, 0.1, axis=1),
            "ws_p90": np.quantile(speed, 0.9, axis=1),
            "dir_sin": sine / norm,
            "dir_cos": cosine / norm,
            "temp_mean": temperature,
        }
    ).cast(pl.Float32)


def training_rows(
    *, farm: dict, era5: pl.DataFrame, actual: pl.DataFrame, months: int | None
) -> pl.DataFrame:
    """Return the farm's ERA5 hours joined to its clean output, before `BASE_END`.

    Hours before the farm's `train_start` (six weeks after its first run of 48 live half-hours) and
    hours older than `months` months before `BASE_END` are dropped.
    """
    first = farm["train_start"]
    if months is not None:
        total = BASE_END.year * 12 + BASE_END.month - 1 - months
        first = max(first, datetime(total // 12, total % 12 + 1, 1, tzinfo=UTC))
    return (
        era5.filter(pl.col("cell_id") == farm["cell_id"], pl.col("valid_time") < BASE_END)
        .join(
            actual.filter(pl.col("elexon_bmu_id") == farm["elexon_bmu_id"]).select(
                "valid_time", target="actual_norm"
            ),
            on="valid_time",
            how="inner",
        )
        .filter(pl.col("valid_time") >= first, pl.col("target").is_finite())
        .sort("valid_time")
    )


def farm_members(*, set_name: str, h3_index: int) -> pl.DataFrame:
    """Return one cell's member rows for a set, sorted by case then member."""
    paths = sorted((ENS_DIR / set_name).glob("init=*.parquet"))
    return (
        pl.concat([pl.scan_parquet(p).filter(pl.col("h3_index") == h3_index) for p in paths])
        .sort("power_fcst_init_time", "valid_time", "ensemble_member")
        .collect()
    )


def main() -> None:
    """Fit every farm's R0 and write the adjusted mixtures for each set."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-months", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    tag = "all" if args.history_months is None else str(args.history_months)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    farms = pl.read_parquet(CLEAN_DIR / "farms.parquet")
    farm_h3 = pl.read_parquet(common.WORK_DIR / "farm_h3.parquet")
    farms = farms.join(farm_h3, on="elexon_bmu_id")
    era5 = era5_features()
    actual = pl.read_parquet(CLEAN_DIR / "actual_centred.parquet")
    per_set: dict[str, tuple[list, list, list]] = {name: ([], [], []) for name in SETS}
    for farm in farms.iter_rows(named=True):
        train = training_rows(farm=farm, era5=era5, actual=actual, months=args.history_months)
        if train.height < 2000:
            print(f"{farm['elexon_bmu_id']}: only {train.height} training hours, skipped")
            continue
        booster, curve = xgboost_booster(train=train, seed=args.seed)
        farm_actual = actual.filter(pl.col("elexon_bmu_id") == farm["elexon_bmu_id"])
        for name in SETS:
            members = farm_members(set_name=name, h3_index=farm["h3_index"])
            cases = (
                members.filter(pl.col("ensemble_member") == 0)
                .select("power_fcst_init_time", "valid_time")
                .with_columns(
                    farm=pl.lit(farm["elexon_bmu_id"]),
                    lead_hours=(
                        (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
                        / 60.0
                    ),
                )
                .join(
                    farm_actual.select("valid_time", "actual_norm"),
                    on="valid_time",
                    how="left",
                )
            )
            if members.height != cases.height * common.MEMBERS:
                raise ValueError(f"{farm['elexon_bmu_id']} {name}: members do not fill the cases")
            quantiles = xgboost_quantiles(booster=booster, curve=curve, rows=members)
            mixture = common.mixture_quantiles(
                member_quantiles=quantiles.reshape(-1, common.MEMBERS, 11)
            )
            summary = member_summaries(rows=members)
            usable = cases["actual_norm"].is_finite().fill_null(value=False).to_numpy()
            per_set[name][0].append(cases.filter(pl.Series(usable)))
            per_set[name][1].append(mixture[usable])
            per_set[name][2].append(summary.filter(pl.Series(usable)))
        print(f"{farm['elexon_bmu_id']}: fitted on {train.height} hours", flush=True)
    parts = {
        name: (pl.concat(a), np.concatenate(b), pl.concat(c)) for name, (a, b, c) in per_set.items()
    }
    all_cases = pl.concat(
        [
            p[0].select("power_fcst_init_time", "valid_time", "lead_hours", "actual_norm")
            for p in parts.values()
        ]
    )
    adjusted = common.apply_spread(
        cases=all_cases, mixture=np.concatenate([p[1] for p in parts.values()]).astype(np.float64)
    )
    start = 0
    for name, (cases, mixture, summary) in parts.items():
        stop = start + cases.height
        frame = cases.select("farm", "power_fcst_init_time", "valid_time").hstack(
            pl.DataFrame(adjusted[start:stop].astype(np.float32), schema=list(common.LEVEL_COLUMNS))
        )
        write_atomic(frame=frame, path=OUT_DIR / f"R0_{name}_h{tag}_seed{args.seed}.parquet")
        path = OUT_DIR / f"cases_{name}.parquet"
        if not path.exists():
            write_atomic(
                frame=cases.hstack(summary).with_columns(
                    lead_day=common.lead_day(
                        power_fcst_init_time=pl.col("power_fcst_init_time"),
                        valid_time=pl.col("valid_time"),
                    ),
                    spread_applied=pl.Series((adjusted[start:stop] != mixture).any(axis=1)),
                ),
                path=path,
            )
        start = stop
        print(f"{name} written", flush=True)


def write_atomic(*, frame: pl.DataFrame, path: Path) -> None:
    """Write `frame` to `path` through a temporary file, so a crash leaves no partial file."""
    temporary = path.with_suffix(".tmp")
    frame.write_parquet(temporary)
    temporary.rename(path)


if __name__ == "__main__":
    main()
