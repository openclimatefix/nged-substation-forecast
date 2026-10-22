"""Fit the member arms of experiment 00230 and write their forecasts.

Arms: A0r (the experiment 33 recipe refitted with the 0.01 and 0.99 levels, ERA5-trained XGBoost
mixture over the 51 ENS members), A1 (A0r trained on the ENS control member's weather), B1 (the
physics core trained on ERA5), B2 (the core trained on the ENS control member), and B8 (B1 without
the idle layer). Each member's quantiles are mixed with equal weights and given the spread factor
experiment 33 fitted month by month on earlier cases.

Output: `work/arms/<arm>_mixture.parquet` and `work/arms/<arm>_spread.parquet`, keyed by case, with
`seed` and the 11 quantile columns. A physics arm has no seed, so its rows are repeated per seed.

    uv run --no-sync python ml-experiments/00230-wind-physics-core-part-1/e00230_arms.py [ARM ...]
"""

import os
import sys

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00230_common as common  # noqa: E402
import e00230_core as core_module  # noqa: E402
import e00230_neural as neural  # noqa: E402
import e00230_step0 as step0  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import xgboost as xgb  # noqa: E402

ARMS = ("A0r", "A0rT", "A1", "B1", "B2", "B8", "B9", "B5", "B6", "B14", "E0")
STOCHASTIC_ARMS = ("A0r", "A0rT", "A1", "B9", "B5", "B6", "B14")
OUT_DIR = common.WORK_DIR / "arms"
N_ROUNDS = 500


def xgboost_member_quantiles(
    *, train: pl.DataFrame, members: pl.DataFrame, seed: int, features: tuple[str, ...]
) -> np.ndarray:
    """Return shape (rows, 11) quantiles of the XGBoost model with the power-curve base margin."""
    target = train["target"].to_numpy()
    speed = train["wind_speed_100m"].to_numpy()
    centres, curve = step0.binned_curve(speed=speed, target=target)
    columns = list(features)
    booster = xgb.train(
        {**step0.XGBOOST_PARAMETERS, "seed": seed, "nthread": 4},
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


def core_member_quantiles(
    *, train: pl.DataFrame, members: pl.DataFrame, with_idle_layer: bool
) -> np.ndarray:
    """Return shape (rows, 11) quantiles of the physics core plus its residual quantiles."""
    target = train["target"].to_numpy()
    weather = step0.weather_arrays(frame=train)
    core = core_module.fit_core(
        weather=weather, target=target, with_idle_layer=with_idle_layer, seed=0
    )
    residuals = core_module.ResidualQuantiles.fit(
        expected=core_module.predict_expected_power(core=core, weather=weather), observed=target
    )
    expected = core_module.predict_expected_power(
        core=core, weather=step0.weather_arrays(frame=members)
    )
    return residuals.quantiles(expected=expected)


def hybrid_member_quantiles(
    *, arm: str, train: pl.DataFrame, members: pl.DataFrame, seed: int, with_idle_layer: bool
) -> np.ndarray:
    """Return shape (rows, 11) quantiles of B9 or B5, all built on the ERA5-trained core."""
    target = train["target"].to_numpy()
    weather = step0.weather_arrays(frame=train)
    core = core_module.fit_core(
        weather=weather, target=target, with_idle_layer=with_idle_layer, seed=0
    )
    expected_train = core_module.predict_expected_power(core=core, weather=weather)
    expected = core_module.predict_expected_power(
        core=core, weather=step0.weather_arrays(frame=members)
    )
    inputs_train = neural.network_inputs(
        frame=train, hub_speed=neural.hub_speed(core=core, frame=train)
    )
    inputs = neural.network_inputs(
        frame=members, hub_speed=neural.hub_speed(core=core, frame=members)
    )
    if arm == "B9":
        network = neural.fit_calibrator(
            inputs=inputs_train, expected=expected_train, target=target, seed=seed
        )
        calibrated_train = neural.calibrated_expected_power(
            network=network, inputs=inputs_train, expected=expected_train
        )
        residuals = core_module.ResidualQuantiles.fit(expected=calibrated_train, observed=target)
        return residuals.quantiles(
            expected=neural.calibrated_expected_power(
                network=network, inputs=inputs, expected=expected
            )
        )
    features = list(common.XGBOOST_FEATURES)
    booster = xgb.train(
        {**step0.XGBOOST_PARAMETERS, "seed": seed, "nthread": 4},
        xgb.DMatrix(
            train.select(features).to_numpy(),
            label=target,
            base_margin=np.tile(expected_train[:, None], (1, 11)),
        ),
        num_boost_round=N_ROUNDS,
    )
    predicted = booster.predict(
        xgb.DMatrix(
            members.select(features).to_numpy(),
            base_margin=np.tile(expected[:, None], (1, 11)),
        )
    )
    return np.sort(np.clip(predicted, 0.0, 1.0), axis=1).astype(np.float32)


def pure_network_member_quantiles(
    *, train: pl.DataFrame, members: pl.DataFrame, seed: int
) -> np.ndarray:
    """Return shape (rows, 11) quantiles of the network with no physics (B6)."""
    centre, scale = neural.raw_input_scaling(frame=train)
    network = neural.fit_free_network(
        inputs=neural.raw_inputs(frame=train, centre=centre, scale=scale),
        target=train["target"].to_numpy(),
        seed=seed,
    )
    return neural.free_quantiles(
        network=network, inputs=neural.raw_inputs(frame=members, centre=centre, scale=scale)
    )


def physics_feature_member_quantiles(
    *, train: pl.DataFrame, members: pl.DataFrame, seed: int, with_idle_layer: bool
) -> np.ndarray:
    """Return shape (rows, 11) quantiles of A0r with the core's columns added (B14)."""
    target = train["target"].to_numpy()
    weather = step0.weather_arrays(frame=train)
    core = core_module.fit_core(
        weather=weather, target=target, with_idle_layer=with_idle_layer, seed=0
    )
    expected_train = core_module.predict_expected_power(core=core, weather=weather)
    residuals = core_module.ResidualQuantiles.fit(expected=expected_train, observed=target)
    expected = core_module.predict_expected_power(
        core=core, weather=step0.weather_arrays(frame=members)
    )
    columns = [0, 1, 5, 9, 10]

    def design(frame: pl.DataFrame, power: np.ndarray) -> np.ndarray:
        physics = residuals.quantiles(expected=power)[:, columns]
        return np.hstack(
            [frame.select(common.XGBOOST_FEATURES).to_numpy(), power[:, None], physics]
        )

    speed = train["wind_speed_100m"].to_numpy()
    centres, curve = step0.binned_curve(speed=speed, target=target)
    booster = xgb.train(
        {**step0.XGBOOST_PARAMETERS, "seed": seed, "nthread": 4},
        xgb.DMatrix(
            design(train, expected_train),
            label=target,
            base_margin=np.tile(np.interp(speed, centres, curve)[:, None], (1, 11)),
        ),
        num_boost_round=N_ROUNDS,
    )
    margin = np.tile(
        np.interp(members["wind_speed_100m"].to_numpy(), centres, curve)[:, None], (1, 11)
    )
    predicted = booster.predict(xgb.DMatrix(design(members, expected), base_margin=margin))
    return np.sort(np.clip(predicted, 0.0, 1.0), axis=1).astype(np.float32)


def era5_with_ensemble_spread(*, members: pl.DataFrame, era5: pl.DataFrame) -> pl.DataFrame:
    """Return members whose wind is ERA5's plus the member's departure from the ensemble mean.

    The ERA5 wind speeds replace the ensemble mean of each case, keeping the ensemble's spread, so
    the arm shows how much of the ensemble's error is the mean's and how much is the spread's.
    The direction and temperature are ERA5's.
    """
    speeds = ("wind_speed_10m", "wind_speed_100m")
    joined = members.join(
        era5.select("time_series_id", "valid_time", *common.WEATHER_COLUMNS),
        on=["time_series_id", "valid_time"],
        how="left",
        suffix="_era5",
        validate="m:1",
    )
    perturbed = [
        (pl.col(f"{name}_era5") + pl.col(name) - pl.col(name).mean().over(*common.KEY_COLUMNS))
        .clip(lower_bound=0.0)
        .alias(name)
        for name in speeds
    ]
    others = [
        pl.col(f"{name}_era5").alias(name) for name in common.WEATHER_COLUMNS if name not in speeds
    ]
    return joined.select(*common.KEY_COLUMNS, "ensemble_member", *perturbed, *others)


def fit_arm(
    *,
    arm: str,
    era5_train: pl.DataFrame,
    ens_train: pl.DataFrame,
    members: pl.DataFrame,
    seed: int,
    farm: str,
) -> np.ndarray:
    """Return one farm's member quantiles for one arm and seed."""
    if arm in ("A0r", "A0rT", "A1"):
        train = ens_train if arm == "A1" else era5_train
        features = common.WEATHER_COLUMNS if arm == "A0rT" else common.XGBOOST_FEATURES
        return xgboost_member_quantiles(train=train, members=members, seed=seed, features=features)
    if arm == "B6":
        return pure_network_member_quantiles(train=era5_train, members=members, seed=seed)
    if arm == "B14":
        return physics_feature_member_quantiles(
            train=era5_train, members=members, seed=seed, with_idle_layer=farm != "W2"
        )
    if arm in ("B9", "B5"):
        return hybrid_member_quantiles(
            arm=arm,
            train=era5_train,
            members=members,
            seed=seed,
            with_idle_layer=farm != "W2",
        )
    train = ens_train if arm == "B2" else era5_train
    return core_member_quantiles(
        train=train, members=members, with_idle_layer=arm != "B8" and farm != "W2"
    )


def main() -> None:
    """Write each requested arm's mixture and spread-adjusted forecasts."""
    wanted = tuple(sys.argv[1:]) or ARMS
    sites = common.site_table()
    era5 = common.era5_rows(sites=sites)
    era5_train = common.training_table(sites=sites, era5=era5)
    ens_train = common.ens_control_training_rows(sites=sites)
    cases = common.cases_table(sites=sites)
    members = common.ens_member_rows(sites=sites).join(
        cases.select(*common.KEY_COLUMNS), on=list(common.KEY_COLUMNS), how="semi"
    )
    spread_members = era5_with_ensemble_spread(members=members, era5=era5)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for arm in wanted:
        stochastic = arm in STOCHASTIC_ARMS
        pieces_mix, pieces_spread = [], []
        for seed in common.SEEDS if stochastic else common.SEEDS[:1]:
            mixtures, kept = [], []
            for row in sites.iter_rows(named=True):
                series, farm = int(row["time_series_id"]), row["site"]
                farm_cases = cases.filter(pl.col("time_series_id") == series)
                farm_members = (spread_members if arm == "E0" else members).filter(
                    pl.col("time_series_id") == series
                )
                one_per_case = farm_members.gather_every(common.MEMBERS).select(common.KEY_COLUMNS)
                if (
                    farm_members.height != farm_cases.height * common.MEMBERS
                    or not one_per_case.equals(farm_cases.select(common.KEY_COLUMNS))
                ):
                    msg = f"{farm}: the members do not line up with the cases"
                    raise ValueError(msg)
                if (
                    farm_members["ensemble_member"].to_numpy().reshape(-1, common.MEMBERS)
                    != np.arange(common.MEMBERS)
                ).any():
                    msg = f"{farm}: a case does not hold members 0 to 50 in order"
                    raise ValueError(msg)
                quantiles = fit_arm(
                    arm=arm,
                    era5_train=era5_train.filter(pl.col("time_series_id") == series),
                    ens_train=ens_train.filter(pl.col("time_series_id") == series),
                    members=farm_members,
                    seed=seed,
                    farm=farm,
                )
                mixtures.append(
                    common.mixture_quantiles(
                        member_quantiles=quantiles.reshape(-1, common.MEMBERS, 11)
                    )
                )
                kept.append(farm_cases)
            ordered = pl.concat(kept)
            mixture = np.concatenate(mixtures)
            spread = common.apply_spread(cases=ordered, mixture=mixture.astype(np.float64))
            keys = ordered.select(*common.KEY_COLUMNS)
            for store, values in ((pieces_mix, mixture), (pieces_spread, spread)):
                frame = keys.hstack(
                    pl.DataFrame(values.astype(np.float32), schema=list(common.LEVEL_COLUMNS))
                )
                store.extend(
                    frame.with_columns(seed=pl.lit(s, dtype=pl.Int32))
                    for s in (common.SEEDS if not stochastic else (seed,))
                )
            print(f"{arm} seed {seed} done", flush=True)
        for name, pieces in (("mixture", pieces_mix), ("spread", pieces_spread)):
            target = OUT_DIR / f"{arm}_{name}.parquet"
            pl.concat(pieces).write_parquet(target.with_suffix(".tmp"))
            target.with_suffix(".tmp").rename(target)


if __name__ == "__main__":
    main()
