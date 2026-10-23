"""Run experiment 230's A0r on the ECMWF and AIFS members and write each source's mixture.

A0r is the ERA5-trained XGBoost quantile model with the power-curve base margin (copied from
experiment 00259's R0, fitted on ERA5 valid times before 2024-04-01), here on the CPU with three
seeds. For each source (the ECMWF ensemble, the AIFS ensemble read as its 00 UTC run of the run's
date, and the AIFS run of 18 UTC the day before) every member's 11 quantiles are mixed with equal
weights, and the mixture's quantile function is stored at 199 levels. Each source's cases are those
of experiment 00259's scored set that have a finite observation and a forecast from that source.

Output, in `work/members/`: `<source>_seed<n>.parquet` (key columns and `f001` to `f199`),
`<source>_keys.parquet`, and `<source>_summary.parquet` (per-case mean and standard deviation of
the members' 100 m wind speed), for the sources `ens`, `aifs`, and `aifs_prev`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
      uv run --no-sync python ml-experiments/00262-*/e00262_members.py
"""

import os
from datetime import UTC, datetime
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00262_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import xgboost as xgb  # noqa: E402

SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
BASE_END: Final = datetime(2024, 4, 1, tzinfo=UTC)
OUT_DIR: Final = common.WORK_DIR / "members"
N_ROUNDS: Final[int] = 500
SPEED_BIN_WIDTH_M_S: Final[float] = 0.5
SOURCES: Final[dict[str, str]] = {
    "ens": "ens",
    "aifs": "aifs_same_day",
    "aifs_prev": "aifs_previous_run",
}
FINE_COLUMNS: Final[tuple[str, ...]] = tuple(f"f{k:03d}" for k in range(1, 200))
XGBOOST_PARAMETERS: Final[dict] = {
    "objective": "reg:quantileerror",
    "quantile_alpha": common.LEVELS,
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "tree_method": "hist",
    "device": "cpu",
    "nthread": 4,
}


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


def fit(*, train: pl.DataFrame, seed: int) -> tuple[xgb.Booster, tuple]:
    """Return the fitted booster and the binned power curve it uses as a base margin."""
    target = train["target"].to_numpy()
    speed = train["wind_speed_100m"].to_numpy()
    centres, curve = binned_curve(speed=speed, target=target)
    booster = xgb.train(
        {**XGBOOST_PARAMETERS, "seed": seed},
        xgb.DMatrix(
            train.select(common.XGBOOST_FEATURES).to_numpy(),
            label=target,
            base_margin=np.tile(np.interp(speed, centres, curve)[:, None], (1, 11)),
        ),
        num_boost_round=N_ROUNDS,
    )
    return booster, (centres, curve)


def predict(*, booster: xgb.Booster, curve: tuple, rows: pl.DataFrame) -> np.ndarray:
    """Return shape (rows, 11) quantiles of the booster on member rows."""
    centres, values = curve
    margin = np.tile(
        np.interp(rows["wind_speed_100m"].to_numpy(), centres, values)[:, None], (1, 11)
    )
    predicted = booster.predict(
        xgb.DMatrix(rows.select(common.XGBOOST_FEATURES).to_numpy(), base_margin=margin)
    )
    return np.sort(np.clip(predicted, 0.0, 1.0), axis=1).astype(np.float32)


def mixture_fine(*, member_quantiles: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    """Return each case's equal-weight member mixture's quantile function at `FINE_LEVELS`."""
    cases, members, levels = member_quantiles.shape
    weights = np.tile(common.ATOM_MASSES / members, members)
    fine = np.empty((cases, common.FINE_LEVELS.size), dtype=np.float32)
    for start in range(0, cases, chunk):
        flat = member_quantiles[start : start + chunk].reshape(-1, members * levels)
        order = np.argsort(flat, axis=1, kind="stable")
        values = np.take_along_axis(flat, order, axis=1)
        mass = weights[order]
        midpoint = np.cumsum(mass, axis=1) - mass / 2
        for row in range(flat.shape[0]):
            fine[start + row] = np.interp(common.FINE_LEVELS, midpoint[row], values[row])
    return fine


def check_alignment(*, members: pl.DataFrame, keys: pl.DataFrame, farm: str, source: str) -> None:
    """Raise unless the member rows hold members 0 to 50, in order, for each key row."""
    if members.height != keys.height * common.MEMBERS or not (
        members.gather_every(common.MEMBERS)
        .select(common.KEY_COLUMNS)
        .equals(keys.select(common.KEY_COLUMNS))
    ):
        msg = f"{source} {farm}: the members do not line up with the cases"
        raise ValueError(msg)
    if (
        members["ensemble_member"].to_numpy().reshape(-1, common.MEMBERS)
        != np.arange(common.MEMBERS)
    ).any():
        msg = f"{source} {farm}: a case does not hold members 0 to 50 in order"
        raise ValueError(msg)


def main() -> None:
    """Fit A0r per farm and seed, and write each source's mixture for every seed."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sites = common.site_table()
    era5_train = common.training_table(sites=sites, era5=common.era5_rows(sites=sites)).filter(
        pl.col("valid_time") < BASE_END
    )
    cases = common.cases_table(sites=sites).filter(pl.col("actual_norm").is_finite())
    ens = common.ens_member_rows(sites=sites).join(
        cases.select(*common.KEY_COLUMNS), on=list(common.KEY_COLUMNS), how="semi"
    )
    tables = {"ens": ens}
    for source, name in SOURCES.items():
        if source != "ens":
            tables[source] = pl.read_parquet(common.WORK_DIR / f"{name}.parquet")
    boosters = {}
    for row in sites.iter_rows(named=True):
        train = era5_train.filter(pl.col("time_series_id") == row["time_series_id"])
        for seed in SEEDS:
            boosters[row["site"], seed] = fit(train=train, seed=seed)
        print(f"{row['site']}: A0r fitted for seeds {SEEDS}", flush=True)
    for source, table in tables.items():
        farm_keys, farm_fine, summaries = [], {seed: [] for seed in SEEDS}, []
        for row in sites.iter_rows(named=True):
            series = int(row["time_series_id"])
            members = table.filter(pl.col("time_series_id") == series).sort(
                *common.KEY_COLUMNS, "ensemble_member"
            )
            keys = members.gather_every(common.MEMBERS).select(common.KEY_COLUMNS)
            check_alignment(members=members, keys=keys, farm=row["site"], source=source)
            speed = members["wind_speed_100m"].to_numpy().reshape(-1, common.MEMBERS)
            summaries.append(
                keys.with_columns(
                    ws_mean=pl.Series(speed.mean(axis=1)), ws_std=pl.Series(speed.std(axis=1))
                ).cast({"ws_mean": pl.Float32, "ws_std": pl.Float32})
            )
            for seed in SEEDS:
                booster, curve = boosters[row["site"], seed]
                quantiles = predict(booster=booster, curve=curve, rows=members)
                farm_fine[seed].append(
                    mixture_fine(member_quantiles=quantiles.reshape(-1, common.MEMBERS, 11))
                )
            farm_keys.append(keys)
            print(f"{source} {row['site']}: {keys.height} cases", flush=True)
        keys = pl.concat(farm_keys)
        keys.write_parquet(OUT_DIR / f"{source}_keys.parquet")
        pl.concat(summaries).write_parquet(OUT_DIR / f"{source}_summary.parquet")
        for seed in SEEDS:
            frame = pl.DataFrame(np.concatenate(farm_fine[seed]), schema=list(FINE_COLUMNS))
            path = OUT_DIR / f"{source}_seed{seed}.parquet"
            keys.hstack(frame).write_parquet(path.with_suffix(".tmp"))
            path.with_suffix(".tmp").rename(path)
        print(f"{source} written", flush=True)


if __name__ == "__main__":
    main()
