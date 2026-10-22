"""Fit the member-versus-mean arms of experiment 00242 and write their forecasts.

For each of the three metered wind farms the same ERA5-trained model is run two ways on the ENS
validation cases: once per ensemble member with the 51 results mixed with equal weights ("members"),
and once on the ensemble mean of the weather inputs ("mean"). Models: the physics core with its
residual quantiles (B1, as in experiment 230), and the XGBoost best-guess recipe refitted with the
0.01 and 0.99 levels (A0r). A third core arm (B1d) mixes the 51 members' expected powers alone, with
no residual quantiles, so the ensemble supplies all the spread. Every arm is written twice: raw, and
after the spread factor experiment 33 fitted month by month on earlier cases.

Output: `work/arms/<arm>_raw.parquet` and `work/arms/<arm>_spread.parquet`, keyed by case, with
`seed` and the 11 quantile columns. A physics arm has no seed, so its rows repeat per seed.

    uv run --no-sync python ml-experiments/00242-*/e00242_arms.py [ARM ...]
"""

import os
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00242_common as common  # noqa: E402
import e00242_core as core_module  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import xgboost as xgb  # noqa: E402

ARMS = ("B1_members", "B1_mean", "B1d_members", "A0r_members", "A0r_mean")
STOCHASTIC_ARMS = ("A0r_members", "A0r_mean")
OUT_DIR = common.WORK_DIR / "arms"
N_ROUNDS = 500
SPEED_BIN_WIDTH_M_S = 0.5
XGBOOST_PARAMETERS = {
    "objective": "reg:quantileerror",
    "quantile_alpha": common.LEVELS,
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "tree_method": "hist",
    "device": "cuda",
    "nthread": 2,
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


def weather_arrays(*, frame: pl.DataFrame) -> core_module.WeatherArrays:
    """Return the three columns the physics core reads."""
    return core_module.WeatherArrays(
        speed_10m=frame["wind_speed_10m"].to_numpy(),
        speed_100m=frame["wind_speed_100m"].to_numpy(),
        temperature_c=frame["temperature_c"].to_numpy(),
    )


def ensemble_mean_rows(*, members: pl.DataFrame) -> pl.DataFrame:
    """Return one row per case whose weather columns are the mean over the 51 members."""
    return (
        members.group_by(common.KEY_COLUMNS, maintain_order=True)
        .agg(pl.col(common.WEATHER_COLUMNS).mean())
        .with_columns(pl.col(common.WEATHER_COLUMNS).cast(pl.Float32))
    )


def xgboost_quantiles(*, train: pl.DataFrame, rows: pl.DataFrame, seed: int) -> np.ndarray:
    """Return shape (rows, 11) quantiles of the XGBoost model with the power-curve base margin."""
    target = train["target"].to_numpy()
    speed = train["wind_speed_100m"].to_numpy()
    centres, curve = binned_curve(speed=speed, target=target)
    columns = list(common.XGBOOST_FEATURES)
    booster = xgb.train(
        {**XGBOOST_PARAMETERS, "seed": seed, "nthread": 4},
        xgb.DMatrix(
            train.select(columns).to_numpy(),
            label=target,
            base_margin=np.tile(np.interp(speed, centres, curve)[:, None], (1, 11)),
        ),
        num_boost_round=N_ROUNDS,
    )
    margin = np.tile(
        np.interp(rows["wind_speed_100m"].to_numpy(), centres, curve)[:, None], (1, 11)
    )
    predicted = booster.predict(xgb.DMatrix(rows.select(columns).to_numpy(), base_margin=margin))
    return np.sort(np.clip(predicted, 0.0, 1.0), axis=1).astype(np.float32)


def core_quantiles(
    *, train: pl.DataFrame, rows: pl.DataFrame, with_idle_layer: bool, with_residuals: bool
) -> np.ndarray:
    """Return shape (rows, 11) quantiles: the core's expected power with or without its residuals.

    Without residuals every level of a row holds the same expected power.
    """
    target = train["target"].to_numpy()
    weather = weather_arrays(frame=train)
    core = core_module.fit_core(
        weather=weather, target=target, with_idle_layer=with_idle_layer, seed=0
    )
    expected = core_module.predict_expected_power(core=core, weather=weather_arrays(frame=rows))
    if not with_residuals:
        return np.tile(expected[:, None], (1, 11)).astype(np.float32)
    residuals = core_module.ResidualQuantiles.fit(
        expected=core_module.predict_expected_power(core=core, weather=weather), observed=target
    )
    return residuals.quantiles(expected=expected)


def member_powers_mixture(*, powers: np.ndarray) -> np.ndarray:
    """Return the 11 `LEVELS` of the equal-weight mixture of per-member point forecasts.

    Args:
        powers: Shape (cases, members) expected powers.
    """
    return np.quantile(powers, common.LEVELS, axis=1).T.astype(np.float32)


def fit_arm(
    *,
    arm: str,
    era5_train: pl.DataFrame,
    members: pl.DataFrame,
    mean_rows: pl.DataFrame,
    seed: int,
    farm: str,
) -> np.ndarray:
    """Return one farm's quantiles, shape (cases, 11), for one arm and seed."""
    idle = farm != "W2"
    rows = mean_rows if arm.endswith("_mean") else members
    if arm.startswith("A0r"):
        quantiles = xgboost_quantiles(train=era5_train, rows=rows, seed=seed)
    else:
        quantiles = core_quantiles(
            train=era5_train, rows=rows, with_idle_layer=idle, with_residuals=arm != "B1d_members"
        )
    if arm == "B1d_members":
        return member_powers_mixture(powers=quantiles[:, 0].reshape(-1, common.MEMBERS))
    if arm.endswith("_members"):
        return common.mixture_quantiles(member_quantiles=quantiles.reshape(-1, common.MEMBERS, 11))
    return quantiles


def main() -> None:
    """Write each requested arm's raw and spread-adjusted forecasts."""
    wanted = tuple(sys.argv[1:]) or ARMS
    sites = common.site_table()
    era5 = common.era5_rows(sites=sites)
    era5_train = common.training_table(sites=sites, era5=era5)
    cases = common.cases_table(sites=sites)
    members = common.ens_member_rows(sites=sites).join(
        cases.select(*common.KEY_COLUMNS), on=list(common.KEY_COLUMNS), how="semi"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for arm in wanted:
        stochastic = arm in STOCHASTIC_ARMS
        pieces_raw, pieces_spread = [], []
        for seed in common.SEEDS if stochastic else common.SEEDS[:1]:
            raws, kept = [], []
            for row in sites.iter_rows(named=True):
                series, farm = int(row["time_series_id"]), row["site"]
                farm_cases = cases.filter(pl.col("time_series_id") == series)
                farm_members = members.filter(pl.col("time_series_id") == series)
                if farm_members.height != farm_cases.height * common.MEMBERS or not (
                    farm_members.gather_every(common.MEMBERS)
                    .select(common.KEY_COLUMNS)
                    .equals(farm_cases.select(common.KEY_COLUMNS))
                ):
                    msg = f"{farm}: the members do not line up with the cases"
                    raise ValueError(msg)
                if (
                    farm_members["ensemble_member"].to_numpy().reshape(-1, common.MEMBERS)
                    != np.arange(common.MEMBERS)
                ).any():
                    msg = f"{farm}: a case does not hold members 0 to 50 in order"
                    raise ValueError(msg)
                raws.append(
                    fit_arm(
                        arm=arm,
                        era5_train=era5_train.filter(pl.col("time_series_id") == series),
                        members=farm_members,
                        mean_rows=ensemble_mean_rows(members=farm_members),
                        seed=seed,
                        farm=farm,
                    )
                )
                kept.append(farm_cases)
            ordered = pl.concat(kept)
            raw = np.concatenate(raws)
            spread = common.apply_spread(cases=ordered, mixture=raw.astype(np.float64))
            keys = ordered.select(*common.KEY_COLUMNS)
            for store, values in ((pieces_raw, raw), (pieces_spread, spread)):
                frame = keys.hstack(
                    pl.DataFrame(values.astype(np.float32), schema=list(common.LEVEL_COLUMNS))
                )
                store.extend(
                    frame.with_columns(seed=pl.lit(s, dtype=pl.Int32))
                    for s in (common.SEEDS if not stochastic else (seed,))
                )
            print(f"{arm} seed {seed} done", flush=True)
        for name, pieces in (("raw", pieces_raw), ("spread", pieces_spread)):
            target = OUT_DIR / f"{arm}_{name}.parquet"
            pl.concat(pieces).write_parquet(target.with_suffix(".tmp"))
            target.with_suffix(".tmp").rename(target)


if __name__ == "__main__":
    main()
