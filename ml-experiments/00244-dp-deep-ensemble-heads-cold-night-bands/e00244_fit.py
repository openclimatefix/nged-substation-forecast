"""Experiment 244: train the deep ensembles of the pooled correction head.

A job is one network: (head set, network kind, scheme, seed). Head sets: `full` (trained to the
scored year, as experiments 200 and 211), `cfx` (holds out 2024-07 to 2024-12) and `cfy` (holds out
2025-01 to 2025-06), the two cross-fit sets that give out-of-sample spreads for the replicate year.
Network kinds: `ref` (no error inputs) and `err` (error inputs kept). Schemes: `seed` (a different
seed) and `bag` (a different seed on a resample of the training months, with replacement).
Each job writes its state dict atomically to `work/nets/<set>_<kind>_<scheme>_<seed>.pt`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python e00244_fit.py --workers 5
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Final

import e00244_data as data_module
import e00244_model as model_module
import numpy as np
import torch  # ty: ignore[unresolved-import]

N_SEEDS: Final[int] = 20
HEAD_SETS: Final[tuple[str, ...]] = ("full", "cfx", "cfy")
KINDS: Final[tuple[str, ...]] = ("ref", "err")
SCHEMES: Final[tuple[str, ...]] = ("seed", "bag")
TRAIN_END: Final[np.datetime64] = np.datetime64("2025-06-28", "us")
PURGE: Final[np.timedelta64] = np.timedelta64(15, "D")
VALID_DEFAULT: Final = (np.datetime64("2025-04-28", "us"), np.datetime64("2025-06-24", "us"))
VALID_CFY: Final = (np.datetime64("2024-10-01", "us"), np.datetime64("2024-11-10", "us"))
HOLD: Final[dict[str, tuple[np.datetime64, np.datetime64]]] = {
    "cfx": (np.datetime64("2024-07-01", "us"), np.datetime64("2025-01-01", "us")),
    "cfy": (np.datetime64("2025-01-01", "us"), np.datetime64("2025-07-01", "us")),
}
SCORED_START: Final[np.datetime64] = np.datetime64("2025-07-02", "us")
ERROR_COLUMNS: Final[tuple[int, ...]] = (*range(18), 29, 30)
"""The 18 error columns, the Kalman level, and the Kalman anomaly: zeroed in the `ref` network."""
WORK_DIR = data_module.WORK_DIR
NETS_DIR = WORK_DIR / "nets"

_STATE: dict = {}


def valid_block(*, head_set: str) -> tuple[np.datetime64, np.datetime64]:
    """The early-stopping block of a head set."""
    return VALID_CFY if head_set == "cfy" else VALID_DEFAULT


def masks(*, f: dict[str, np.ndarray], head_set: str) -> tuple[np.ndarray, np.ndarray]:
    """Boolean train and early-stopping masks of one substation's rows for a head set."""
    init = f["init"]
    low, high = valid_block(head_set=head_set)
    valid = (init >= low) & (init < high)
    drop = (init >= low - PURGE) & (init < high + PURGE)
    if head_set in HOLD:
        hold_low, hold_high = HOLD[head_set]
        drop |= (init >= hold_low - PURGE) & (init < hold_high + PURGE)
    return (init < TRAIN_END) & ~drop, valid


def load_state() -> dict:
    """Features, design matrices, and site codes, loaded once per process."""
    if not _STATE:
        feats, labels, _ = data_module_load()
        _STATE["feats"] = feats
        _STATE["labels"] = labels
        _STATE["x"] = [model_module.design(f=f) for f in feats]
    return _STATE


def data_module_load() -> tuple[list[dict[str, np.ndarray]], list[str], None]:
    """Features per substation in series-id order, with labels (no masks)."""
    import polars as pl

    labels = {}
    for path in sorted(data_module.SRC_DIR.glob("d_scored_D*.parquet")):
        if "spread" in path.name:
            continue
        sid = int(pl.read_parquet(path, columns=["time_series_id"])["time_series_id"][0])
        labels[sid] = path.stem.split("_")[-1]
    ids = sorted(labels)
    feats = []
    for i in ids:
        site = data_module.load_site(time_series_id=i)
        feats.append(data_module.build_features(site=site, error_grid=site.error_grid))
    return feats, [labels[i] for i in ids], None


def training_data(
    *, head_set: str, kind: str, scheme: str, seed: int
) -> tuple[model_module.Data, model_module.Data]:
    """The (training, early-stopping) arrays for one job."""
    state = load_state()
    feats = state["feats"]
    zero = list(ERROR_COLUMNS) if kind == "ref" else []
    parts = {"train": [], "valid": []}
    for k, f in enumerate(feats):
        train_mask, valid_mask = masks(f=f, head_set=head_set)
        x = state["x"][k].copy()
        x[:, zero] = 0.0
        for name, mask in (("train", train_mask), ("valid", valid_mask)):
            parts[name].append(
                (
                    x[mask],
                    np.full(int(mask.sum()), k),
                    f["target"][mask],
                    f["q_offsets"][mask],
                    f["init"][mask].astype("datetime64[M]"),
                )
            )

    def stack(name: str) -> tuple[np.ndarray, ...]:
        return tuple(np.concatenate([p[i] for p in parts[name]]) for i in range(5))

    tx, ts, ty, to, tm = stack("train")
    if scheme == "bag":
        rng = np.random.default_rng(1000 + seed)
        months = np.unique(tm)
        drawn = rng.choice(months, size=len(months), replace=True)
        index = np.concatenate([np.flatnonzero(tm == m) for m in drawn])
        tx, ts, ty, to = tx[index], ts[index], ty[index], to[index]
    vx, vs, vy, vo, _ = stack("valid")
    return (
        model_module.Data(x=tx, site=ts, y=ty, offsets=to),
        model_module.Data(x=vx, site=vs, y=vy, offsets=vo),
    )


def run_job(job: tuple[str, str, str, int]) -> str:
    """Train one network and save its state dict; skip it if the file exists."""
    head_set, kind, scheme, seed = job
    path = NETS_DIR / f"{head_set}_{kind}_{scheme}_{seed}.pt"
    if path.exists():
        return f"{path.name}: exists"
    torch.set_num_threads(1)
    start = time.time()
    train, valid = training_data(head_set=head_set, kind=kind, scheme=scheme, seed=seed)
    n_sites = len(load_state()["feats"])
    net = model_module.fit(train=train, valid=valid, n_sites=n_sites, seed=seed, device="cpu")
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    torch.save(net.state_dict(), temporary)
    temporary.rename(path)
    return f"{path.name}: {len(train.y)} rows, {time.time() - start:.0f} s"


def main() -> None:
    """Run every job on a pool of single-thread workers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--head-sets", nargs="*", default=list(HEAD_SETS))
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()
    NETS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        (h, k, s, seed)
        for seed in range(args.seeds)
        for h in args.head_sets
        for k in KINDS
        for s in SCHEMES
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(run_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
