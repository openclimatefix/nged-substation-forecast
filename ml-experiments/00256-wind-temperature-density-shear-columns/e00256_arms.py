"""Fit the arms of experiment 00256 and write their forecasts.

Every arm is experiment 00230's A0r recipe (an ERA5-trained XGBoost quantile model with the
power-curve base margin, run on the 51 ENS members, mixed with equal weights, and given the spread
factor of experiment 33) with its own set of columns: T0 (six wind columns), T1 (plus air
temperature), T2 (plus air density instead), T3 (plus temperature, density, shear, and solar
elevation), and P1 (T1 with the training temperature permuted by whole calendar months within each
farm). `colsample_bytree` stays at 1.

Output: `work/arms/<arm>_mixture.parquet` and `work/arms/<arm>_spread.parquet`, keyed by case, with
`seed` and the 11 quantile columns; `work/pressure_check.json`, the stop rule's monthly mean
difference between ENS and ERA5 surface pressure.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
      uv run --no-sync python ml-experiments/00256-*/e00256_arms.py [ARM ...]
"""

import json
import os
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00256_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import xgboost as xgb  # noqa: E402

ARMS = tuple(common.ARM_FEATURES)
OUT_DIR = common.WORK_DIR / "arms"
N_ROUNDS = 500
SPEED_BIN_WIDTH_M_S = 0.5
PRESSURE_STOP_HPA = 5.0
XGBOOST_PARAMETERS = {
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
    "seed": 0,
}
PLACEBO_SEED = 256


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


def permute_temperature_by_month(*, train: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Return the training rows with `temperature_c` taken from another calendar month's rows.

    The farm's rows are cut into whole calendar months (year and month). Each month receives the
    temperatures of a different month, picked by a random permutation with no fixed point, read at
    the same position in the month and wrapped round if the donor month is shorter. The column
    keeps its distribution and its daily cycle and loses its alignment with the power.
    """
    ordered = train.sort("valid_time").with_columns(
        block=pl.col("valid_time").dt.strftime("%Y-%m"),
        position=pl.int_range(pl.len()).over(pl.col("valid_time").dt.strftime("%Y-%m")),
    )
    blocks = sorted(ordered["block"].unique().to_list())
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(blocks))
    while (order == np.arange(len(blocks))).any():
        order = rng.permutation(len(blocks))
    donors = {blocks[i]: blocks[j] for i, j in enumerate(order)}
    temperature = {
        b: g["temperature_c"].to_numpy()
        for b, g in ordered.group_by("block", maintain_order=True)
        for b in [b[0]]
    }
    permuted = np.empty(ordered.height, dtype=np.float32)
    blocks_col, position = ordered["block"].to_numpy(), ordered["position"].to_numpy()
    for block, donor in donors.items():
        rows = blocks_col == block
        values = temperature[donor]
        permuted[rows] = values[position[rows] % values.size]
    return ordered.with_columns(temperature_c=pl.Series(permuted)).drop("block", "position")


def member_quantiles(
    *, train: pl.DataFrame, members: pl.DataFrame, seed: int, features: tuple[str, ...]
) -> np.ndarray:
    """Return shape (rows, 11) quantiles of the XGBoost model with the power-curve base margin."""
    target = train["target"].to_numpy()
    speed = train["wind_speed_100m"].to_numpy()
    centres, curve = binned_curve(speed=speed, target=target)
    columns = list(features)
    booster = xgb.train(
        {**XGBOOST_PARAMETERS, "seed": seed},
        xgb.DMatrix(
            train.select(columns).to_numpy(),
            label=target,
            base_margin=np.tile(np.interp(speed, centres, curve)[:, None], (1, 11)),
        ),
        num_boost_round=N_ROUNDS,
    )
    margin = np.tile(
        np.interp(members["wind_speed_100m"].to_numpy(), centres, curve)[:, None], (1, 11)
    )
    predicted = booster.predict(xgb.DMatrix(members.select(columns).to_numpy(), base_margin=margin))
    return np.sort(np.clip(predicted, 0.0, 1.0), axis=1).astype(np.float32)


def pressure_check(*, era5: pl.DataFrame, members: pl.DataFrame, sites: pl.DataFrame) -> dict:
    """Return the monthly mean ENS minus ERA5 surface pressure (hPa) per farm, and stop if large."""
    ens = (
        members.group_by("time_series_id", "valid_time")
        .agg(pl.col("pressure_pa").mean().alias("ens"))
        .join(
            era5.select("time_series_id", "valid_time", era5_pa="pressure_pa"),
            on=["time_series_id", "valid_time"],
        )
        .join(sites.select("time_series_id", "site"), on="time_series_id")
        .with_columns(month=pl.col("valid_time").dt.strftime("%Y-%m"))
        .group_by("site", "month")
        .agg(((pl.col("ens") - pl.col("era5_pa")).mean() / 100).alias("difference_hpa"))
        .sort("site", "month")
    )
    worst = float(ens["difference_hpa"].abs().max())  # ty: ignore[invalid-argument-type]
    report = {
        "worst_abs_monthly_difference_hpa": worst,
        "monthly": ens.to_dicts(),
        "stop_rule_fires": worst > PRESSURE_STOP_HPA,
    }
    (common.WORK_DIR / "pressure_check.json").write_text(json.dumps(report, indent=1))
    print(f"pressure check: worst monthly mean difference {worst:.2f} hPa", flush=True)
    return report


def main() -> None:
    """Write each requested arm's mixture and spread-adjusted forecasts."""
    wanted = tuple(sys.argv[1:]) or ARMS
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    sites = common.site_table()
    era5 = common.era5_rows(sites=sites)
    cases = common.cases_table(sites=sites)
    members = common.ens_member_rows(sites=sites).join(
        cases.select(*common.KEY_COLUMNS), on=list(common.KEY_COLUMNS), how="semi"
    )
    check = pressure_check(era5=era5, members=members, sites=sites)
    if check["stop_rule_fires"]:
        print(
            "stop rule: pressure units or level mismatch; T2 and T3 need fixed pressure", flush=True
        )
    train_all = common.with_derived_columns(
        frame=common.training_table(sites=sites, era5=era5), sites=sites
    )
    members = common.with_derived_columns(frame=members, sites=sites)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for arm in wanted:
        features = common.ARM_FEATURES[arm]
        pieces_mix, pieces_spread = [], []
        for seed in common.SEEDS:
            mixtures, kept = [], []
            for row in sites.iter_rows(named=True):
                series = int(row["time_series_id"])
                farm_cases = cases.filter(pl.col("time_series_id") == series)
                farm_members = members.filter(pl.col("time_series_id") == series)
                one_per_case = farm_members.gather_every(common.MEMBERS).select(common.KEY_COLUMNS)
                if (
                    farm_members.height != farm_cases.height * common.MEMBERS
                    or not one_per_case.equals(farm_cases.select(common.KEY_COLUMNS))
                ):
                    msg = f"{row['site']}: the members do not line up with the cases"
                    raise ValueError(msg)
                if (
                    farm_members["ensemble_member"].to_numpy().reshape(-1, common.MEMBERS)
                    != np.arange(common.MEMBERS)
                ).any():
                    msg = f"{row['site']}: a case does not hold members 0 to 50 in order"
                    raise ValueError(msg)
                train = train_all.filter(pl.col("time_series_id") == series)
                if arm == "P1":
                    train = permute_temperature_by_month(
                        train=train, seed=PLACEBO_SEED + series + 1000 * seed
                    )
                    true_temperature = (
                        train_all.filter(pl.col("time_series_id") == series)
                        .sort("valid_time")["temperature_c"]
                        .to_numpy()
                    )
                    correlation = np.corrcoef(train["temperature_c"].to_numpy(), true_temperature)[
                        0, 1
                    ]
                    print(
                        f"P1 seed {seed} {row['site']}: correlation {correlation:.2f}", flush=True
                    )
                quantiles = member_quantiles(
                    train=train, members=farm_members, seed=seed, features=features
                )
                mixtures.append(
                    common.mixture_quantiles(
                        member_quantiles=quantiles.reshape(-1, common.MEMBERS, 11)
                    )
                )
                kept.append(farm_cases)
                print(f"{arm} seed {seed} {row['site']} fitted", flush=True)
            ordered = pl.concat(kept)
            mixture = np.concatenate(mixtures)
            spread = common.apply_spread(cases=ordered, mixture=mixture.astype(np.float64))
            keys = ordered.select(*common.KEY_COLUMNS)
            for store, values in ((pieces_mix, mixture), (pieces_spread, spread)):
                store.append(
                    keys.hstack(
                        pl.DataFrame(values.astype(np.float32), schema=list(common.LEVEL_COLUMNS))
                    ).with_columns(seed=pl.lit(seed, dtype=pl.Int32))
                )
        for name, pieces in (("mixture", pieces_mix), ("spread", pieces_spread)):
            target = OUT_DIR / f"{arm}_{name}.parquet"
            pl.concat(pieces).write_parquet(target.with_suffix(".tmp"))
            target.with_suffix(".tmp").rename(target)
        print(f"{arm} written", flush=True)


if __name__ == "__main__":
    main()
