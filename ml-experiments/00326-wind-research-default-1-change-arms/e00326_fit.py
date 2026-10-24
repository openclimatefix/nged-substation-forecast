"""Fit experiment 00326's arms for one farm and store their member mixtures.

Mixtures are stored in `work/mix/<arm>_s<seed>_<hindcast|scored>_<farm>.npy`, shape (cases, 11),
the equal-weight mixture of the 51 members' quantiles before the spread factor, in experiment
00303's layout (`e00303_fit.py`). Arms: `default` (the six-column XGBoost refit, the reproduction
gate's subject), `a` (plus temperature and density), `b` (plus hour and day of year), `d` (the
default's booster on members with the 100 m speed multiplied by the hindcast factor), `d_rep`
(the same with the replicate's factor, hindcast set only), and the placebos `a_pl`, `b_pl`
(month-shuffled columns) and `c_pl` (the adapter's effective speed, permuted across rows).
Arm `c` is experiment 00303's stored P8 and is not refitted.

    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
      env PYTHONPATH=ml-experiments/00303-wind-adapter-in-production-setting \
      uv run --no-sync python ml-experiments/00326-*/e00326_fit.py --farms W1
"""

import argparse
import os
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "2")
os.environ.setdefault("E00303_TORCH_THREADS", "2")

import e00303_ens as ens  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00303_fit as fit303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00303_models as models  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00326_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import xgboost as xgb  # noqa: E402


def train_columns(*, data: fit303.Hindcast) -> dict[str, np.ndarray]:
    """Return the ERA5 training rows' columns, named as `e00303_fit.member_columns` names them."""
    train = data.train
    columns = {
        "speed_10m": train["wind_speed_10m"].to_numpy(),
        "speed_100m": train["wind_speed_100m"].to_numpy(),
        "temperature_c": train["temperature_c"].to_numpy(),
        "dir_sin_10m": train["wind_dir_sin_10m"].to_numpy(),
        "dir_cos_10m": train["wind_dir_cos_10m"].to_numpy(),
        "direction_sin": train["wind_dir_sin_100m"].to_numpy(),
        "direction_cos": train["wind_dir_cos_100m"].to_numpy(),
        "hour": train["hour"].to_numpy(),
        "doy": train["doy"].to_numpy(),
        "lead_day": np.zeros(train.height, dtype=np.float32),
        "spread": np.zeros(train.height, dtype=np.float32),
    }
    columns["density"] = common.air_density(temperature_c=columns["temperature_c"])
    return columns


def member_set(
    *, columns: dict[str, np.ndarray], valid_case: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return a member set's columns with `doy` and `density` added, and each row's valid time."""
    valid = np.repeat(valid_case, common.MEMBERS)
    doy = pl.Series(valid_case).dt.ordinal_day().to_numpy().astype(np.float32)
    out = {
        **columns,
        "doy": np.repeat(doy, common.MEMBERS),
        "density": common.air_density(temperature_c=columns["temperature_c"]),
    }
    return out, valid


def placebo(
    *, columns: dict[str, np.ndarray], names: tuple[str, ...], valid: np.ndarray, seed: int
) -> dict[str, np.ndarray]:
    """Return `columns` with `names` replaced by the month-shuffled placebo values."""
    source = common.month_shuffle_index(valid=valid, seed=seed)
    return {**columns, **{name: columns[name][source] for name in names}}


def matrix(
    *,
    columns: dict[str, np.ndarray],
    extra: tuple[str, ...] = (),
    effective: np.ndarray | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Return the XGBoost matrix: the six wind columns (100 m speed times `scale`), then extras."""
    parts = [columns[name] for name in fit303.XGB_COLUMN_ORDER]
    if scale != 1.0:
        parts[1] = parts[1] * np.float32(scale)
    parts += [columns[name] for name in extra]
    if effective is not None:
        parts.append(effective)
    return np.column_stack(parts).astype(np.float32)


def fit_booster(
    *, train: pl.DataFrame, features: np.ndarray, seed: int
) -> tuple[xgb.Booster, np.ndarray, np.ndarray]:
    """Fit the XGBoost quantile model with the binned-curve base margin; return it and the curve."""
    target = train["target"].to_numpy()
    centres, curve = models.binned_curve(speed=train["wind_speed_100m"].to_numpy(), target=target)
    base = np.interp(train["wind_speed_100m"].to_numpy(), centres, curve)
    booster = xgb.train(
        {**models.XGBOOST_PARAMETERS, "seed": seed},
        xgb.DMatrix(features, label=target, base_margin=np.tile(base[:, None], (1, 11))),
        num_boost_round=models.N_ROUNDS,
    )
    return booster, centres, curve


def mixture(
    *,
    booster: xgb.Booster,
    centres: np.ndarray,
    curve: np.ndarray,
    columns: dict[str, np.ndarray],
    extra: tuple[str, ...] = (),
    effective: np.ndarray | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Return the cases' equal-weight member mixtures (cases, 11) of the booster on `columns`."""
    size = columns["speed_100m"].size
    member = np.empty((size, 11), dtype=np.float32)
    for start in range(0, size, fit303.PREDICT_CHUNK):
        window = slice(start, start + fit303.PREDICT_CHUNK)
        chunk = {name: values[window] for name, values in columns.items()}
        member[window] = fit303._xgb_member(
            booster=booster,
            features=matrix(
                columns=chunk,
                extra=extra,
                effective=None if effective is None else effective[window],
                scale=scale,
            ),
            speed100=chunk["speed_100m"] * np.float32(scale),
            centres=centres,
            curve=curve,
        )
    return fit303.mixture_of_member_quantiles(member_quantiles=member)


def stored(*, arm: str, seed: int, set_name: str, farm: str, array: np.ndarray) -> None:
    """Write one mixture atomically."""
    common.MIX_DIR.mkdir(parents=True, exist_ok=True)
    ens.write_atomic(path=common.MIX_DIR / f"{arm}_s{seed}_{set_name}_{farm}.npy", array=array)


def present(*, arm: str, farm: str) -> bool:
    """Return whether every mixture of `arm` for `farm` is stored."""
    sets = ("hindcast",) if arm == "d_rep" else ("hindcast", "scored")
    return all(
        (common.MIX_DIR / f"{arm}_s{seed}_{name}_{farm}.npy").exists()
        for seed in common.SEEDS
        for name in sets
    )


def run_farm(*, farm: str, arms: list[str]) -> None:
    """Fit and store the requested arms for one farm."""
    data = fit303.Hindcast(farm=farm)
    cases = common.c303.cases_table(sites=data.sites).filter(pl.col("site") == farm)
    sets = {
        "hindcast": member_set(columns=data.hind_columns, valid_case=data.valid),
        "scored": member_set(
            columns=data.scored_columns,
            valid_case=cases["valid_time"].dt.replace_time_zone(None).to_numpy(),
        ),
    }
    train = train_columns(data=data)
    train_valid = data.train["valid_time"].dt.replace_time_zone(None).to_numpy()
    factor = common.speed_factor(valid_before=None)
    factor_rep = common.speed_factor(valid_before=common.REPLICATE_START)
    print(f"{farm}: factor {factor:.4f}, replicate factor {factor_rep:.4f}", flush=True)
    pending = [a for a in arms if not present(arm=a, farm=farm)]
    need_default = {"default", "d", "d_rep"} & set(pending)
    for seed in common.SEEDS:
        started = time.time()
        if need_default:
            booster, centres, curve = fit_booster(
                train=data.train, features=matrix(columns=train), seed=seed
            )
            for arm, scale, names in (
                ("default", 1.0, ("hindcast", "scored")),
                ("d", factor, ("hindcast", "scored")),
                ("d_rep", factor_rep, ("hindcast",)),
            ):
                if arm in pending:
                    for name in names:
                        stored(
                            arm=arm,
                            seed=seed,
                            set_name=name,
                            farm=farm,
                            array=mixture(
                                booster=booster,
                                centres=centres,
                                curve=curve,
                                columns=sets[name][0],
                                scale=scale,
                            ),
                        )
        for arm, extra in common.EXTRA_COLUMNS.items():
            for variant in (arm, f"{arm}_pl"):
                if variant not in pending:
                    continue
                shuffled = variant.endswith("_pl")
                train_set = (
                    placebo(
                        columns=train,
                        names=extra,
                        valid=train_valid,
                        seed=common.SHUFFLE_SEED + seed,
                    )
                    if shuffled
                    else train
                )
                booster, centres, curve = fit_booster(
                    train=data.train, features=matrix(columns=train_set, extra=extra), seed=seed
                )
                for name, (columns, valid) in sets.items():
                    use = (
                        placebo(
                            columns=columns,
                            names=extra,
                            valid=valid,
                            seed=common.SHUFFLE_SEED + seed,
                        )
                        if shuffled
                        else columns
                    )
                    stored(
                        arm=variant,
                        seed=seed,
                        set_name=name,
                        farm=farm,
                        array=mixture(
                            booster=booster, centres=centres, curve=curve, columns=use, extra=extra
                        ),
                    )
        if "c_pl" in pending:
            core, _ = fit303.fit_era5_core(
                train=data.train, adapter="net", seed=seed, with_idle_layer=data.with_idle_layer
            )
            rng = np.random.default_rng(common.SHUFFLE_SEED + seed)
            effective_train = fit303.predict_columns(core=core, columns=train, speed=True)
            effective_train = effective_train[rng.permutation(effective_train.size)]
            booster, centres, curve = fit_booster(
                train=data.train,
                features=matrix(columns=train, effective=effective_train),
                seed=seed,
            )
            for name, (columns, _) in sets.items():
                effective = fit303.predict_columns(core=core, columns=columns, speed=True)
                effective = effective[rng.permutation(effective.size)]
                stored(
                    arm="c_pl",
                    seed=seed,
                    set_name=name,
                    farm=farm,
                    array=mixture(
                        booster=booster,
                        centres=centres,
                        curve=curve,
                        columns=columns,
                        effective=effective,
                    ),
                )
        print(f"{farm} seed {seed}: {time.time() - started:.0f} s", flush=True)


def main() -> None:
    """Fit the requested arms for the requested farms."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--farms", default=",".join(common.FARMS))
    parser.add_argument("--arms", default=",".join(common.FIT_ARMS))
    args = parser.parse_args()
    for farm in args.farms.split(","):
        run_farm(farm=farm, arms=args.arms.split(","))


if __name__ == "__main__":
    main()
