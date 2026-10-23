"""Fit the two base arms of experiment 00259 and write their member mixtures.

R0 is experiment 230's A0r recipe (an ERA5-trained XGBoost quantile model with the power-curve base
margin) and R1 is experiment 230's physics core with residual quantiles (B1). Both are fitted on
ERA5 valid times from 2019-09-01 to 2024-03-31, so that their output on the daily ENS hindcast runs
of 2024-04-01 to 2025-06-16 is out of sample, and the same fits score the 53 Tuesday runs of the
scored year. Each model runs on all 51 members, the 51 forecasts are mixed with equal weights, and
the spread factor of experiment 33 (fitted month by month on earlier cases only, by lead band) is
applied separately within the hindcast runs and within the scored runs.

Output, in `work/base/`: `summaries_<set>.parquet` (one row per case, with the members' wind and
temperature summaries that the heads read) and `<arm>_<set>.parquet` (the spread-adjusted quantiles
by seed), for the sets `hindcast` and `scored`. R0 has seeds 0 to 4. R1 has no seed and is written
once with seed 0.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
      uv run --no-sync python ml-experiments/00259-*/e00259_base.py [ARM ...]
"""

import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00259_common as common  # noqa: E402
import e00259_core as core_module  # noqa: E402
import e00259_hindcast as hindcast  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import xgboost as xgb  # noqa: E402

ARMS: Final[tuple[str, ...]] = ("R1", "R0")
BASE_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
BASE_END: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
"""The base models train on ERA5 valid times before this."""
OUT_DIR: Final = common.WORK_DIR / "base"
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


def xgboost_booster(*, train: pl.DataFrame, seed: int) -> tuple[xgb.Booster, tuple]:
    """Return the fitted XGBoost booster and the binned power curve it uses as a base margin."""
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


def xgboost_quantiles(*, booster: xgb.Booster, curve: tuple, rows: pl.DataFrame) -> np.ndarray:
    """Return shape (rows, 11) quantiles of the XGBoost model on member rows."""
    centres, values = curve
    margin = np.tile(
        np.interp(rows["wind_speed_100m"].to_numpy(), centres, values)[:, None], (1, 11)
    )
    predicted = booster.predict(
        xgb.DMatrix(rows.select(common.XGBOOST_FEATURES).to_numpy(), base_margin=margin)
    )
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


def hindcast_members(*, series: int) -> pl.DataFrame:
    """Return one farm's hindcast member rows, sorted by case and member."""
    paths = sorted(hindcast.HINDCAST_DIR.glob("init=*.parquet"))
    return (
        pl.concat([pl.scan_parquet(p).filter(pl.col("time_series_id") == series) for p in paths])
        .sort(*common.KEY_COLUMNS, "ensemble_member")
        .collect()
    )


def hindcast_cases(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return one row per hindcast case with the observed fraction, lead, and mask flag."""
    paths = sorted(hindcast.HINDCAST_DIR.glob("init=*.parquet"))
    keys = (
        pl.concat(
            [
                pl.scan_parquet(p).filter(pl.col("ensemble_member") == 0).select(common.KEY_COLUMNS)
                for p in paths
            ]
        )
        .collect()
        .sort(common.KEY_COLUMNS)
    )
    actual = common.observed_training_power(sites=sites).rename({"target": "actual_norm"})
    mask = common.idle_mask().with_columns(masked=pl.lit(value=True))
    return (
        keys.join(actual, on=["time_series_id", "valid_time"], how="left", validate="m:1")
        .join(mask, on=["time_series_id", "valid_time"], how="left", validate="m:1")
        .join(sites.select("time_series_id", "site"), on="time_series_id", validate="m:1")
        .with_columns(
            masked=pl.col("masked").fill_null(value=False),
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0,
            lead_day=common.lead_day(
                power_fcst_init_time=pl.col("power_fcst_init_time"), valid_time=pl.col("valid_time")
            ),
        )
        .sort(*common.KEY_COLUMNS)
    )


def check_alignment(*, members: pl.DataFrame, cases: pl.DataFrame, farm: str) -> None:
    """Raise unless the member rows hold members 0 to 50, in order, for each case in `cases`."""
    if members.height != cases.height * common.MEMBERS or not (
        members.gather_every(common.MEMBERS)
        .select(common.KEY_COLUMNS)
        .equals(cases.select(common.KEY_COLUMNS))
    ):
        msg = f"{farm}: the members do not line up with the cases"
        raise ValueError(msg)
    if (
        members["ensemble_member"].to_numpy().reshape(-1, common.MEMBERS)
        != np.arange(common.MEMBERS)
    ).any():
        msg = f"{farm}: a case does not hold members 0 to 50 in order"
        raise ValueError(msg)


def core_member_quantiles(
    *, train: pl.DataFrame, rows: pl.DataFrame, with_idle_layer: bool
) -> np.ndarray:
    """Return shape (rows, 11) quantiles of the physics core plus its residual quantiles."""
    target = train["target"].to_numpy()
    weather = weather_arrays(frame=train)
    core = core_module.fit_core(
        weather=weather, target=target, with_idle_layer=with_idle_layer, seed=0
    )
    residuals = core_module.ResidualQuantiles.fit(
        expected=core_module.predict_expected_power(core=core, weather=weather), observed=target
    )
    expected = core_module.predict_expected_power(core=core, weather=weather_arrays(frame=rows))
    return residuals.quantiles(expected=expected)


def set_mixture(
    *,
    arm: str,
    seed: int,
    sites: pl.DataFrame,
    era5_train: pl.DataFrame,
    members_for: Callable[[int], pl.DataFrame],
    cases: pl.DataFrame,
) -> tuple[pl.DataFrame, np.ndarray, pl.DataFrame]:
    """Return the usable cases, their mixture quantiles, and the members' summaries for one set."""
    mixtures, kept, summaries = [], [], []
    for row in sites.iter_rows(named=True):
        series, farm = int(row["time_series_id"]), row["site"]
        farm_cases = cases.filter(pl.col("time_series_id") == series)
        farm_members = members_for(series)
        check_alignment(members=farm_members, cases=farm_cases, farm=farm)
        train = era5_train.filter(pl.col("time_series_id") == series)
        if arm == "R0":
            booster, curve = xgboost_booster(train=train, seed=seed)
            quantiles = xgboost_quantiles(booster=booster, curve=curve, rows=farm_members)
        else:
            quantiles = core_member_quantiles(
                train=train, rows=farm_members, with_idle_layer=farm != "W2"
            )
        mixtures.append(
            common.mixture_quantiles(member_quantiles=quantiles.reshape(-1, common.MEMBERS, 11))
        )
        summaries.append(member_summaries(rows=farm_members))
        kept.append(farm_cases)
        print(f"{arm} seed {seed} {farm}: {farm_cases.height} cases", flush=True)
    ordered = pl.concat(kept)
    mixture = np.concatenate(mixtures).astype(np.float64)
    summary = pl.concat(summaries)
    usable = pl.Series(ordered["actual_norm"].is_finite().fill_null(value=False))
    return ordered.filter(usable), mixture[usable.to_numpy()], summary.filter(usable)


def write_atomic(*, frame: pl.DataFrame, path: Path) -> None:
    """Write `frame` to `path` through a temporary file, so a crash leaves no partial file."""
    temporary = path.with_suffix(".tmp")
    frame.write_parquet(temporary)
    temporary.rename(path)


def main() -> None:
    """Write each requested base arm's spread-adjusted mixtures for the hindcast and scored sets.

    The spread factor is fitted in one pass over the hindcast and scored cases together, in time
    order, so the scored months see the hindcast cases as their earlier cases. The first months of
    the hindcast hold fewer than `MIN_FIT_CASES` earlier cases and keep the unadjusted mixture;
    the summaries file flags them with `spread_applied`.
    """
    wanted = tuple(sys.argv[1:]) or ARMS
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sites = common.site_table()
    era5_train = common.training_table(sites=sites, era5=common.era5_rows(sites=sites)).filter(
        pl.col("valid_time") < BASE_END
    )
    scored_cases = common.cases_table(sites=sites)
    scored_members = common.ens_member_rows(sites=sites).join(
        scored_cases.select(*common.KEY_COLUMNS), on=list(common.KEY_COLUMNS), how="semi"
    )
    sets = {
        "hindcast": (
            lambda series: hindcast_members(series=series),
            hindcast_cases(sites=sites),
        ),
        "scored": (
            lambda series: scored_members.filter(pl.col("time_series_id") == series),
            scored_cases,
        ),
    }
    for arm in wanted:
        for seed in BASE_SEEDS if arm == "R0" else BASE_SEEDS[:1]:
            parts = {
                name: set_mixture(
                    arm=arm,
                    seed=seed,
                    sites=sites,
                    era5_train=era5_train,
                    members_for=members_for,
                    cases=cases,
                )
                for name, (members_for, cases) in sets.items()
            }
            all_cases = pl.concat(
                [
                    part[0].select(*common.KEY_COLUMNS, "lead_hours", "actual_norm")
                    for part in parts.values()
                ],
                how="vertical_relaxed",
            )
            all_mixture = np.concatenate([part[1] for part in parts.values()])
            adjusted = common.apply_spread(cases=all_cases, mixture=all_mixture)
            start = 0
            for name, (ordered, mixture, summary) in parts.items():
                stop = start + ordered.height
                frame = ordered.select(*common.KEY_COLUMNS).hstack(
                    pl.DataFrame(
                        adjusted[start:stop].astype(np.float32), schema=list(common.LEVEL_COLUMNS)
                    )
                )
                write_atomic(
                    frame=frame.with_columns(seed=pl.lit(seed, dtype=pl.Int32)),
                    path=OUT_DIR / f"{arm}_{name}_seed{seed}.parquet",
                )
                if not (OUT_DIR / f"summaries_{name}.parquet").exists():
                    write_atomic(
                        frame=ordered.hstack(summary).with_columns(
                            spread_applied=pl.Series((adjusted[start:stop] != mixture).any(axis=1))
                        ),
                        path=OUT_DIR / f"summaries_{name}.parquet",
                    )
                start = stop
                print(f"{arm} seed {seed} {name} written", flush=True)


if __name__ == "__main__":
    main()
