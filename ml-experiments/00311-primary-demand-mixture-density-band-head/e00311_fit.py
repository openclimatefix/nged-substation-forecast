"""Fit the mixture-density band head at each monthly refit date and write its 11 quantiles.

For a refit at month start `c`, the pool is every row whose origin lies in the 24 months before `c`
and whose longest target (origin plus 168 hours) lies before `c`. Every fourth month of the pool is
held out for early stopping. The head is a three-component Gaussian mixture on
`(actual - arm D mean) / scale`, where `scale` is the stored 5% to 95% width over 3.29. The test
rows are those whose origin month is `c`. Output: `work/fits/cut<YYYY-MM>_seed<s>.parquet`.

    PYTHONPATH=ml-experiments/00278-stack-retraining-cadence \
        systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00311-*/e00311_fit.py --cuts 2025-07 ... --workers 6
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Final

import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from scipy.special import ndtr
from torch import nn  # ty: ignore[unresolved-import]

import e00311_common as common  # isort: skip

COMPONENTS: Final[int] = 3
HIDDEN: Final[int] = 64
BATCH: Final[int] = 16384
MAX_EPOCHS: Final[int] = 40
PATIENCE: Final[int] = 6
LEARNING_RATE: Final[float] = 5e-4
WEIGHT_DECAY: Final[float] = 1e-3
INPUT_CLIP_SD: Final[float] = 5.0
Z_LIMIT: Final[float] = 40.0


class MixtureHead(nn.Module):
    """Three-component Gaussian mixture on the scaled residual of arm D's mean."""

    def __init__(self, *, n_inputs: int) -> None:
        """Build the network; the last layer starts at zero with a standard normal-like mixture."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, 3 * COMPONENTS),
        )  # fmt: skip
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.copy_(torch.tensor([0.0] * 3 + [-0.5, 0.0, 0.5] + [0.0] * 3))

    def parameters_of(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the log weights, means, and standard deviations."""
        out = self.net(x)
        return (
            torch.log_softmax(out[:, :COMPONENTS], dim=1),
            out[:, COMPONENTS : 2 * COMPONENTS],
            torch.exp(out[:, 2 * COMPONENTS :].clamp(-3.0, 3.0)),
        )

    def nll(self, *, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return the mean negative log-likelihood of the scaled residuals."""
        log_w, mean, std = self.parameters_of(x)
        log_pdf = -0.5 * ((z[:, None] - mean) / std) ** 2 - torch.log(std)
        return -torch.logsumexp(log_w + log_pdf, dim=1).mean() + 0.5 * float(np.log(2 * np.pi))


def mixture_quantiles(*, model: MixtureHead, x: torch.Tensor) -> np.ndarray:
    """Return the mixture's 11 quantiles in scaled units by bisection on its CDF."""
    out = np.empty((len(x), len(common.LEVELS)))
    for start in range(0, len(x), 1 << 16):
        with torch.no_grad():
            log_w, mean, std = model.parameters_of(x[start : start + (1 << 16)])
        w, mean, std = (t.numpy().astype(np.float64) for t in (log_w.exp(), mean, std))
        for k, level in enumerate(common.LEVELS):
            lo = np.full(len(w), -Z_LIMIT)
            hi = np.full(len(w), Z_LIMIT)
            for _ in range(45):
                mid = 0.5 * (lo + hi)
                cdf = (w * ndtr((mid[:, None] - mean) / std)).sum(axis=1)
                below = cdf < level
                lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
            out[start : start + len(w), k] = 0.5 * (lo + hi)
    return out


def month_start(*, label: str) -> datetime:
    """Return 00:00 UTC on the first day of a `YYYY-MM` month."""
    year, month = (int(part) for part in label.split("-"))
    return datetime(year, month, 1, tzinfo=UTC)


def months_before(*, when: datetime, months: int) -> datetime:
    """Return the month start `months` months before `when`."""
    index = when.year * 12 + (when.month - 1) - months
    return datetime(index // 12, index % 12 + 1, 1, tzinfo=UTC)


def run_job(task: tuple[str, int]) -> str:  # noqa: PLR0915
    """Fit one (cut, seed) head and write its test-row quantiles."""
    label, seed = task
    target = common.FIT_DIR / f"cut{label}_seed{seed}.parquet"
    if target.exists():
        return f"{label} seed {seed} exists"
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    started = time.monotonic()
    rows = pl.read_parquet(common.PREP_DIR / "rows.parquet")
    x_all = np.load(common.PREP_DIR / "x.npy", mmap_mode="r")
    mean = np.load(common.PREP_DIR / "mean.npy", mmap_mode="r")
    scale = np.load(common.PREP_DIR / "scale.npy", mmap_mode="r")
    actual = np.load(common.PREP_DIR / "actual.npy", mmap_mode="r")
    cut = month_start(label=label)
    origin = rows["power_fcst_init_time"]
    pool = ((origin + timedelta(hours=common.MAX_LEAD_HOURS)) < cut) & (
        origin >= months_before(when=cut, months=common.POOL_MONTHS)
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
            (np.asarray(x_all[idx], dtype=np.float64) - mu) / sd, -INPUT_CLIP_SD, INPUT_CLIP_SD
        )
        z = (np.asarray(actual[idx]) - np.asarray(mean[idx])) / np.asarray(scale[idx])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(z, dtype=torch.float32)

    x_fit, z_fit = tensors(train_idx)
    x_val, z_val = tensors(val_idx)
    x_test, _ = tensors(test_idx)
    model = MixtureHead(n_inputs=x_fit.shape[1])
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed)
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    history: list[float] = []
    for epochs in range(1, MAX_EPOCHS + 1):  # noqa: B007
        model.train()
        order = torch.randperm(len(x_fit), generator=generator)
        for begin in range(0, len(order), BATCH):
            idx = order[begin : begin + BATCH]
            loss = model.nll(x=x_fit[idx], z=z_fit[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            score = float(model.nll(x=x_val, z=z_val))
        history.append(round(score, 4))
        if score < best:
            best, stale = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    z_q = mixture_quantiles(model=model, x=x_test)
    q = np.asarray(mean[test_idx])[:, None] + np.asarray(scale[test_idx])[:, None] * z_q
    part = rows[test_idx].select("site", "run", "valid_time").with_columns(
        *[pl.Series(name, q[:, k].astype(np.float32)) for k, name in enumerate(common.QUANTILES)],
        n_train=pl.lit(len(train_idx)),
        epochs=pl.lit(epochs),
        best_val=pl.lit(best),
        seconds=pl.lit(time.monotonic() - started),
    )  # fmt: skip
    common.FIT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    part.write_parquet(temporary)
    temporary.rename(target)
    return (
        f"{label} seed {seed}: {epochs} epochs, {time.monotonic() - started:.0f} s, val {history}"
    )


def main() -> None:
    """Fit every requested (cut, seed) with a process pool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuts", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    tasks = [(label, seed) for label in args.cuts for seed in common.SEEDS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for message in pool.map(run_job, tasks):
            print(message, flush=True)


if __name__ == "__main__":
    main()
