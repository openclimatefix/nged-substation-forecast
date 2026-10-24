# ruff: noqa: E501, PLR0915
"""Fit the standard-architecture heads (STD and STD-A) at each monthly refit date.

The cuts, pool, early-stopping split, input matrix, and band scale are experiment 311's. The
network feeds the columns in `ADAPTER_COLUMNS` through a two-layer adapter, concatenates the result
with all inputs, and passes them through a two-layer trunk. `std` ends in a three-component Gaussian
mixture on `(actual - arm D mean) / scale` and is trained on the negative log-likelihood; `stda`
ends in one median output trained on the absolute error. Output:
`work/fits/<head>_cut<YYYY-MM>_seed<s>.parquet`, with the 11 quantiles (`std`) or the median
(`stda`) in points.

    export PYTHONPATH=ml-experiments/00278-stack-retraining-cadence:ml-experiments/00311-*:ml-experiments/00314-*:ml-experiments/00330-*
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00330-*/e00330_fit.py --cuts 2025-07 ... --workers 6
"""

import argparse
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from typing import Final

import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

import e00311_common as common  # isort: skip  # ty: ignore[unresolved-import]
import e00311_fit as base  # isort: skip  # ty: ignore[unresolved-import]
import e00330_common as own  # isort: skip

ADAPTER_HIDDEN: Final[int] = 16
ADAPTER_OUT: Final[int] = 8
THREADS: Final[int] = 1


class StandardNet(nn.Module):
    """Input adapter, trunk, and either a mixture head or a median head."""

    def __init__(self, *, n_inputs: int, head: str) -> None:
        """Build the network; the last layer starts at zero (a standard normal-like output)."""
        super().__init__()
        self.head = head
        self.adapter = nn.Sequential(
            nn.Linear(len(own.ADAPTER_COLUMNS), ADAPTER_HIDDEN), nn.GELU(),
            nn.Linear(ADAPTER_HIDDEN, ADAPTER_OUT),
        )  # fmt: skip
        n_out = 3 * base.COMPONENTS if head == "std" else 1
        self.trunk = nn.Sequential(
            nn.Linear(n_inputs + ADAPTER_OUT, base.HIDDEN), nn.GELU(),
            nn.Linear(base.HIDDEN, base.HIDDEN), nn.GELU(),
            nn.Linear(base.HIDDEN, n_out),
        )  # fmt: skip
        last = self.trunk[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            bias = [0.0] * 3 + [-0.5, 0.0, 0.5] + [0.0] * 3 if head == "std" else [0.0]
            last.bias.copy_(torch.tensor(bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw outputs."""
        adapted = self.adapter(x[:, list(own.ADAPTER_COLUMNS)])
        return self.trunk(torch.cat([x, adapted], dim=1))

    def parameters_of(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the mixture's log weights, means, and standard deviations (head `std`)."""
        out = self(x)
        k = base.COMPONENTS
        return (
            torch.log_softmax(out[:, :k], dim=1),
            out[:, k : 2 * k],
            torch.exp(out[:, 2 * k :].clamp(-3.0, 3.0)),
        )

    def loss(self, *, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return the training loss: the mixture's NLL, or the mean absolute error."""
        if self.head == "stda":
            return (z - self(x)[:, 0]).abs().mean()
        log_w, mean, std = self.parameters_of(x)
        log_pdf = -0.5 * ((z[:, None] - mean) / std) ** 2 - torch.log(std)
        return -torch.logsumexp(log_w + log_pdf, dim=1).mean() + 0.5 * float(np.log(2 * np.pi))


def run_job(task: tuple[str, str, int]) -> str:
    """Fit one (head, cut, seed) network and write its test-row forecasts."""
    head, label, seed = task
    target = own.FIT_DIR / f"{head}_cut{label}_seed{seed}.parquet"
    if target.exists():
        return f"{head} {label} seed {seed} exists"
    torch.set_num_threads(THREADS)
    torch.manual_seed(seed)
    started = time.monotonic()
    prep = own.E311_DATA_DIR / "prep"
    rows = pl.read_parquet(prep / "rows.parquet")
    x_all = np.load(prep / "x.npy", mmap_mode="r")
    mean = np.load(prep / "mean.npy", mmap_mode="r")
    scale = np.load(prep / "scale.npy", mmap_mode="r")
    actual = np.load(prep / "actual.npy", mmap_mode="r")
    cut = base.month_start(label=label)
    origin = rows["power_fcst_init_time"]
    pool = ((origin + timedelta(hours=common.MAX_LEAD_HOURS)) < cut) & (
        origin >= base.months_before(when=cut, months=common.POOL_MONTHS)
    )
    pool_np = pool.to_numpy()
    pool_valid_max = rows["valid_time"].filter(pool).max()
    assert isinstance(pool_valid_max, datetime), "the training pool is empty"
    assert pool_valid_max < cut, "a training target reaches the refit date"
    test_np = (rows["month_key"] == label).to_numpy()
    if label in common.REPLICATE_CUTS:
        test_np &= (rows["period"] == "replicate").to_numpy()
    months = sorted(rows.filter(pool)["month_key"].unique().to_list())
    val_months = {m for i, m in enumerate(months) if i % common.VAL_MONTH_STRIDE == 3}
    is_val = rows["month_key"].is_in(list(val_months)).to_numpy()
    train_idx = np.flatnonzero(pool_np & ~is_val)
    val_idx = np.flatnonzero(pool_np & is_val)
    test_idx = np.flatnonzero(test_np)
    x_train = np.asarray(x_all[train_idx], dtype=np.float64)
    mu, sd = x_train.mean(axis=0), x_train.std(axis=0) + 1e-6

    def tensors(idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        x = np.clip(
            (np.asarray(x_all[idx], dtype=np.float64) - mu) / sd,
            -base.INPUT_CLIP_SD,
            base.INPUT_CLIP_SD,
        )
        z = (np.asarray(actual[idx]) - np.asarray(mean[idx])) / np.asarray(scale[idx])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(z, dtype=torch.float32)

    x_fit, z_fit = tensors(train_idx)
    x_val, z_val = tensors(val_idx)
    x_test, _ = tensors(test_idx)
    model = StandardNet(n_inputs=x_fit.shape[1], head=head)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY
    )
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
        z_q = _mixture_quantiles(model=model, x=x_test)
        q = centre + width * z_q
        columns = [
            pl.Series(name, q[:, k].astype(np.float32)) for k, name in enumerate(common.QUANTILES)
        ]
    else:
        with torch.no_grad():
            z_m = model(x_test)[:, 0].numpy().astype(np.float64)
        columns = [pl.Series("q50", (centre[:, 0] + width[:, 0] * z_m).astype(np.float32))]
    part = rows[test_idx].select("site", "run", "valid_time").with_columns(
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
    return f"{head} {label} seed {seed}: {epochs} epochs, {time.monotonic() - started:.0f} s"


def _mixture_quantiles(*, model: StandardNet, x: torch.Tensor) -> np.ndarray:
    """Return the mixture's 11 quantiles in scaled units by bisection on its CDF."""
    from scipy.special import ndtr

    out = np.empty((len(x), len(common.LEVELS)))
    chunk = 1 << 16
    for start in range(0, len(x), chunk):
        with torch.no_grad():
            log_w, mean, std = model.parameters_of(x[start : start + chunk])
        w, mean, std = (t.numpy().astype(np.float64) for t in (log_w.exp(), mean, std))
        for k, level in enumerate(common.LEVELS):
            lo = np.full(len(w), -base.Z_LIMIT)
            hi = np.full(len(w), base.Z_LIMIT)
            for _ in range(45):
                mid = 0.5 * (lo + hi)
                cdf = (w * ndtr((mid[:, None] - mean) / std)).sum(axis=1)
                below = cdf < level
                lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
            out[start : start + len(w), k] = 0.5 * (lo + hi)
    return out


def main() -> None:
    """Fit every requested (head, cut, seed) with a spawn-based process pool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuts", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--heads", nargs="+", default=list(own.HEADS))
    args = parser.parse_args()
    tasks = [(h, c, s) for c in args.cuts for h in args.heads for s in own.SEEDS]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        for message in pool.map(run_job, tasks):
            print(message, flush=True)


if __name__ == "__main__":
    main()
