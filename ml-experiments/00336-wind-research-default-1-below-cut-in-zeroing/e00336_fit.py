"""Fit experiment 00336's arms for one farm and store their member mixtures.

Mixtures are stored in `work/mix/<arm>_s<seed>_<hindcast|scored>_<farm>.npy`, shape (cases, 11),
the equal-weight mixture of the 51 members' quantiles before the spread factor, in experiment
00303's layout. Per-case statistics (ensemble-mean and ensemble-maximum 100 m speed from the member
rows, and the ERA5 100 m speed at the valid time) are stored in `work/stats/`.

Arms: `default` (the six-column XGBoost refit, the reproduction gate's subject), `x1m*` (each
member's own speed below the cut-in zeroes that member before pooling; the default booster),
`x2*` (ERA5 rows below cut-in dropped from training), `x3` (target zeroed on those rows), and the
placebos `x2pl` and `x3pl` (the same share of rows chosen at random within the month).

    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
      env PYTHONPATH=<the 00303 and 00326 experiment folders, colon separated> \
      uv run --no-sync python ml-experiments/00336-*/e00336_fit.py --farms W1
"""

import argparse
import os
import time
from types import SimpleNamespace

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_ens as ens  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00303_fit as fit303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00326_fit as fit326  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00336_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import xgboost as xgb  # noqa: E402

MEMBERS = common.MEMBERS


def store_stats(*, farm: str, data: fit303.Hindcast, cases: pl.DataFrame) -> None:
    """Write the per-case member speed statistics and the ERA5 speed at the valid time."""
    common.STATS_DIR.mkdir(parents=True, exist_ok=True)
    era5 = data.era5.select("time_series_id", "valid_time", "wind_speed_100m")
    keys = {"hindcast": data.hind_keys, "scored": cases.select(*common.c303.KEY_COLUMNS)}
    columns = {"hindcast": data.hind_columns, "scored": data.scored_columns}
    for name, frame in keys.items():
        speed = columns[name]["speed_100m"].reshape(-1, MEMBERS)
        if speed.shape[0] != frame.height:
            msg = f"{farm} {name}: {speed.shape[0]} member groups for {frame.height} cases"
            raise ValueError(msg)
        ens.write_atomic(
            path=common.stats_path(kind="speed", set_name=name, farm=farm),
            array=np.column_stack([speed.mean(axis=1), speed.max(axis=1)]).astype(np.float32),
        )
        joined = frame.join(
            era5,
            on=["time_series_id", "valid_time"],
            how="left",
            validate="m:1",
            maintain_order="left",
        )
        ens.write_atomic(
            path=common.stats_path(kind="era5", set_name=name, farm=farm),
            array=joined["wind_speed_100m"].to_numpy().astype(np.float32),
        )


def member_quantiles(
    *, booster: xgb.Booster, centres: np.ndarray, curve: np.ndarray, columns: dict[str, np.ndarray]
) -> np.ndarray:
    """Return every member row's 11 quantiles (rows, 11), as `e00326_fit.mixture` computes them."""
    size = columns["speed_100m"].size
    member = np.empty((size, 11), dtype=np.float32)
    for start in range(0, size, fit303.PREDICT_CHUNK):
        window = slice(start, start + fit303.PREDICT_CHUNK)
        chunk = {name: values[window] for name, values in columns.items()}
        member[window] = fit303._xgb_member(
            booster=booster,
            features=fit326.matrix(columns=chunk),
            speed100=chunk["speed_100m"],
            centres=centres,
            curve=curve,
        )
    return member


def pool(*, member: np.ndarray, speed: np.ndarray, cut_in: float | None) -> np.ndarray:
    """Return the cases' mixtures (cases, 11), with members below `cut_in` set to zero power."""
    step = fit303.CASE_CHUNK * MEMBERS
    pieces = []
    for start in range(0, member.shape[0], step):
        chunk = member[start : start + step]
        if cut_in is not None:
            chunk = chunk.copy()
            chunk[speed[start : start + step] < cut_in] = 0.0
        pieces.append(fit303.mixture_of_member_quantiles(member_quantiles=chunk))
    return np.concatenate(pieces)


def training_variant(*, arm: str, train: pl.DataFrame, farm: str, seed: int) -> pl.DataFrame:
    """Return the ERA5 training frame of a training arm (`x2*` drops rows, `x3*` zeroes targets)."""
    kind, rest = arm[:2], arm[2:]
    speed = train["wind_speed_100m"].to_numpy()
    suffix = rest if rest in common.CUT_IN_VARIANTS else ""
    flag = speed < common.cut_in_for(farm=farm, suffix=suffix)
    if rest == "e":
        masked = train.join(
            common.idle_mask_frame(),
            on=["time_series_id", "valid_time"],
            how="left",
            validate="m:1",
            maintain_order="left",
        )["masked"].fill_null(value=False)
        flag = flag | masked.to_numpy()
    if rest == "pl":
        month = common.month_labels(
            valid_time=train["valid_time"].dt.replace_time_zone(None).to_numpy()
        )
        flag = common.random_like(flag=flag, group=month, seed=common.PLACEBO_SEED + seed)
    if kind == "x2":
        return train.filter(pl.Series(~flag))
    target = pl.when(pl.Series(flag)).then(0.0).otherwise(pl.col("target"))
    return train.with_columns(target.cast(train.schema["target"]).alias("target"))


def store(*, arm: str, seed: int, set_name: str, farm: str, array: np.ndarray) -> None:
    """Write one mixture atomically."""
    common.MIX_DIR.mkdir(parents=True, exist_ok=True)
    ens.write_atomic(
        path=common.MIX_DIR / f"{arm}_s{seed}_{set_name}_{farm}.npy", array=array.astype(np.float32)
    )


def present(*, arm: str, farm: str) -> bool:
    """Return whether every mixture of `arm` for `farm` is stored."""
    return all(
        (common.MIX_DIR / f"{arm}_s{seed}_{name}_{farm}.npy").exists()
        for seed in common.SEEDS
        for name in ("hindcast", "scored")
    )


def run_farm(*, farm: str, arms: list[str]) -> None:
    """Fit and store the requested arms for one farm."""
    data = fit303.Hindcast(farm=farm)
    cases = common.c303.cases_table(sites=data.sites).filter(pl.col("site") == farm)
    store_stats(farm=farm, data=data, cases=cases)
    sets = {"hindcast": data.hind_columns, "scored": data.scored_columns}
    pending = [a for a in arms if not present(arm=a, farm=farm)]
    print(f"{farm}: cut-in {common.cut_in_speeds()[farm]:.3f} m/s; pending {pending}", flush=True)
    for seed in common.SEEDS:
        started = time.time()
        member_arms = [a for a in pending if a == "default" or a.startswith("x1m")]
        if member_arms:
            columns = fit326.train_columns(data=data)
            booster, centres, curve = fit326.fit_booster(
                train=data.train, features=fit326.matrix(columns=columns), seed=seed
            )
            for name, cols in sets.items():
                member = member_quantiles(
                    booster=booster, centres=centres, curve=curve, columns=cols
                )
                for arm in member_arms:
                    cut = (
                        None
                        if arm == "default"
                        else common.cut_in_for(farm=farm, suffix=arm.removeprefix("x1m"))
                    )
                    store(
                        arm=arm,
                        seed=seed,
                        set_name=name,
                        farm=farm,
                        array=pool(member=member, speed=cols["speed_100m"], cut_in=cut),
                    )
                del member
        for arm in (a for a in pending if a[:2] in ("x2", "x3")):
            train = training_variant(arm=arm, train=data.train, farm=farm, seed=seed)
            namespace = SimpleNamespace(train=train)
            columns = fit326.train_columns(data=namespace)  # type: ignore[arg-type]
            booster, centres, curve = fit326.fit_booster(
                train=train, features=fit326.matrix(columns=columns), seed=seed
            )
            print(
                f"{farm} {arm} seed {seed}: {train.height} of {data.train.height} rows", flush=True
            )
            for name, cols in sets.items():
                store(
                    arm=arm,
                    seed=seed,
                    set_name=name,
                    farm=farm,
                    array=fit326.mixture(
                        booster=booster, centres=centres, curve=curve, columns=cols
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
