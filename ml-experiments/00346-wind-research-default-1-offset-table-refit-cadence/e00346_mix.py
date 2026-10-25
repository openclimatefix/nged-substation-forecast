"""Refit the ERA5-trained XGBoost model on CPU and store the default's member mixtures.

For one farm, fits the booster of experiment 00336's `default` arm (six wind columns, ERA5 before
2024-04-01, seeds 0 to 2) on CPU, then turns the member rows of every daily run (experiment 00259's
hindcast files, 2024-04-01 to 2025-06-16, and this experiment's files, 2025-06-17 to 2026-10-23)
into the equal-weight mixture of the 51 members' quantiles, before the spread factor. All runs go
through the same boosters so that the hindcast and the later runs are one consistent set.

Output, in `work/mix/`: `keys_<farm>.parquet` (key columns, in mixture order, written by seed 0) and
`default_s<seed>_<farm>.npy` (cases, 11). One process per farm and seed, one thread each.

    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
      env PYTHONPATH=<see e00346_common> uv run --no-sync python e00346_mix.py --farm W1
"""

import argparse
import os
import time
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import e00303_ens as ens  # isort: skip  # ty: ignore[unresolved-import]
import e00303_fit as fit303  # isort: skip  # ty: ignore[unresolved-import]
import e00303_models as models  # isort: skip  # ty: ignore[unresolved-import]
import e00326_fit as fit326  # isort: skip  # ty: ignore[unresolved-import]
import e00336_fit as fit336  # isort: skip  # ty: ignore[unresolved-import]
import e00346_common as common
import numpy as np
import polars as pl

CHUNK_RUNS = 20
models.XGBOOST_PARAMETERS.update(device="cpu", nthread=1)


def run_files() -> list:
    """Return every daily run's member file in run order."""
    old = sorted(common.c303.HINDCAST_DIR.glob("init=*.parquet"))
    new = sorted(common.MEMBERS_DIR.glob("init=*.parquet"))
    return [*old, *new]


def main() -> None:
    """Fit the farm's three boosters and store the mixtures of every run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--farm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    farm, only_seed = args.farm, args.seed
    common.MIX_DIR.mkdir(parents=True, exist_ok=True)
    sites = common.c303.site_table().filter(pl.col("site") == farm)
    series = int(sites["time_series_id"][0])
    era5 = common.c303.era5_rows(sites=sites)
    train = models.add_derived_columns(
        frame=common.c303.training_table(sites=sites, era5=era5)
    ).filter(pl.col("valid_time") < common.c303.BASE_END)
    namespace = SimpleNamespace(train=train)
    columns = fit326.train_columns(data=namespace)  # type: ignore[arg-type]
    started = time.time()
    boosters = {
        seed: fit326.fit_booster(train=train, features=fit326.matrix(columns=columns), seed=seed)
        for seed in (only_seed,)
    }
    print(
        f"{farm} seed {only_seed}: booster on {train.height} rows in {time.time() - started:.0f} s",
        flush=True,
    )
    files = run_files()
    pieces: dict[int, list[np.ndarray]] = {only_seed: []}
    keys = []
    for start in range(0, len(files), CHUNK_RUNS):
        rows = (
            pl.concat(
                [
                    pl.scan_parquet(p).filter(pl.col("time_series_id") == series)
                    for p in files[start : start + CHUNK_RUNS]
                ]
            )
            .sort(*common.c303.KEY_COLUMNS, "ensemble_member")
            .collect()
        )
        chunk_keys = rows.gather_every(common.c303.MEMBERS).select(common.c303.KEY_COLUMNS)
        ens.check_alignment(members=rows, cases=chunk_keys, farm=farm)
        keys.append(chunk_keys)
        member_cols = fit303.member_columns(rows=ens.add_member_columns(rows=rows))
        for seed, (booster, centres, curve) in boosters.items():
            member = fit336.member_quantiles(
                booster=booster, centres=centres, curve=curve, columns=member_cols
            )
            pieces[seed].append(
                fit336.pool(member=member, speed=member_cols["speed_100m"], cut_in=None)
            )
        print(
            f"{farm}: {start + CHUNK_RUNS}/{len(files)} runs, {time.time() - started:.0f} s",
            flush=True,
        )
    if only_seed == common.SEEDS[0]:
        pl.concat(keys).write_parquet(common.MIX_DIR / f"keys_{farm}.parquet")
    for seed in (only_seed,):
        ens.write_atomic(
            path=common.MIX_DIR / f"default_s{seed}_{farm}.npy",
            array=np.concatenate(pieces[seed]).astype(np.float32),
        )
    print(f"{farm}: done in {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
