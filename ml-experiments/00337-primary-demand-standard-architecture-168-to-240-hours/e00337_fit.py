# ruff: noqa: E501, PLR0915
"""Fit experiment 330's standard-architecture heads (STD and STD-A) on the 168-to-240-hour rows.

The network, mixture quantiles, and training settings are experiment 330's. The rows, pool, and
early-stopping split follow experiment 330's rules on the rows `e00337_prep.py` writes: a refit on
the first day of month m trains on rows whose origin is at least 240 hours before the cut, within
24 months of it, and stops early on every fourth month of the pool. Output:
`work/fits/<head>_cut<YYYY-MM>_seed<s>.parquet`.

    source /tmp/pp337.sh   # or the PYTHONPATH in e00337_common.py
    systemd-run --user --scope -p MemoryMax=20G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00337-*/e00337_fit.py --workers 12
"""

import argparse
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

import e00311_common as common  # isort: skip  # ty: ignore[unresolved-import]
import e00311_fit as base  # isort: skip  # ty: ignore[unresolved-import]
import e00330_common as c330  # isort: skip  # ty: ignore[unresolved-import]
import e00330_fit as fit330  # isort: skip  # ty: ignore[unresolved-import]
import e00337_common as own  # isort: skip


def run_job(task: tuple[str, str, int]) -> str:
    """Fit one (head, cut, seed) network and write its test-row forecasts."""
    head, label, seed = task
    target = own.FIT_DIR / f"{head}_cut{label}_seed{seed}.parquet"
    if target.exists():
        return f"{head} {label} seed {seed} exists"
    torch.set_num_threads(fit330.THREADS)
    torch.manual_seed(seed)
    started = time.monotonic()
    rows = pl.read_parquet(own.PREP_DIR / "rows.parquet")
    x_all = np.load(own.PREP_DIR / "x.npy", mmap_mode="r")
    mean = np.load(own.PREP_DIR / "mean.npy", mmap_mode="r")
    scale = np.load(own.PREP_DIR / "scale.npy", mmap_mode="r")
    actual = np.load(own.PREP_DIR / "actual.npy", mmap_mode="r")
    cut = base.month_start(label=label)
    origin = rows["power_fcst_init_time"]
    pool = ((origin + timedelta(hours=own.MAX_LEAD_HOURS)) < cut) & (
        origin >= base.months_before(when=cut, months=common.POOL_MONTHS)
    )
    pool_np = pool.to_numpy()
    pool_valid_max = rows["valid_time"].filter(pool).max()
    assert isinstance(pool_valid_max, datetime), "the training pool is empty"
    assert pool_valid_max < cut, "a training target reaches the refit date"
    test_np = ((rows["month_key"] == label) & (rows["period"] == "scored")).to_numpy()
    months = sorted(rows.filter(pool)["month_key"].unique().to_list())
    val_months = {m for i, m in enumerate(months) if i % common.VAL_MONTH_STRIDE == 3}
    is_val = rows["month_key"].is_in(list(val_months)).to_numpy()
    train_idx = np.flatnonzero(pool_np & ~is_val)
    val_idx = np.flatnonzero(pool_np & is_val)
    test_idx = np.flatnonzero(test_np)
    x_train = np.asarray(x_all[train_idx], dtype=np.float64)
    mu, sd = x_train.mean(axis=0), x_train.std(axis=0) + 1e-6

    def tensors(idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        x = np.clip((np.asarray(x_all[idx], dtype=np.float64) - mu) / sd, -base.INPUT_CLIP_SD, base.INPUT_CLIP_SD)  # fmt: skip
        z = (np.asarray(actual[idx]) - np.asarray(mean[idx])) / np.asarray(scale[idx])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(z, dtype=torch.float32)

    x_fit, z_fit = tensors(train_idx)
    x_val, z_val = tensors(val_idx)
    x_test, _ = tensors(test_idx)
    model = fit330.StandardNet(n_inputs=x_fit.shape[1], head=head)
    optimiser = torch.optim.AdamW(model.parameters(), lr=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY)  # fmt: skip
    generator = torch.Generator().manual_seed(seed)
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epochs in range(1, base.MAX_EPOCHS + 1):  # noqa: B007
        model.train()
        order = torch.randperm(len(x_fit), generator=generator)
        for begin in range(0, len(order), base.BATCH):
            idx = order[begin : begin + base.BATCH]
            loss = model.loss(x=x_fit[idx], z=z_fit[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            score = float(model.loss(x=x_val, z=z_val))
        if score < best:
            best, stale = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= base.PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    centre = np.asarray(mean[test_idx])[:, None]
    width = np.asarray(scale[test_idx])[:, None]
    if head == "std":
        q = centre + width * fit330._mixture_quantiles(model=model, x=x_test)
        columns = [pl.Series(name, q[:, k].astype(np.float32)) for k, name in enumerate(common.QUANTILES)]  # fmt: skip
    else:
        with torch.no_grad():
            z_m = model(x_test)[:, 0].numpy().astype(np.float64)
        columns = [pl.Series("q50", (centre[:, 0] + width[:, 0] * z_m).astype(np.float32))]
    part = rows[test_idx].select("site", "power_fcst_init_time", "valid_time").with_columns(
        *columns,
        n_train=pl.lit(len(train_idx)),
        epochs=pl.lit(epochs),
        best_val=pl.lit(best),
        seconds=pl.lit(time.monotonic() - started),
    )  # fmt: skip
    own.FIT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    part.write_parquet(temporary)
    temporary.rename(target)
    return f"{head} {label} seed {seed}: n_train {len(train_idx)}, {epochs} epochs, {time.monotonic() - started:.0f} s"  # fmt: skip


def main() -> None:
    """Fit every requested (head, cut, seed) with a spawn-based process pool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuts", nargs="+", default=list(common.SCORED_CUTS))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--heads", nargs="+", default=list(c330.HEADS))
    args = parser.parse_args()
    tasks = [(h, c, s) for c in args.cuts for h in args.heads for s in c330.SEEDS]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        for message in pool.map(run_job, tasks):
            print(message, flush=True)


if __name__ == "__main__":
    main()
