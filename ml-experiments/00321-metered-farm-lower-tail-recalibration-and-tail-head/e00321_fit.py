# ruff: noqa: E501
"""Fit the zero-inflated head (arm 2), the tail-targeted head (arm 3), and arm 4's zero-history fit.

Rows, settings, farms, and refits are experiment 310's `Setting`. The zero-inflated head is a
three-component Gaussian mixture on the residual from the plant plus a logit for an exact-zero
half-hour; its quantile at level t is 0 when t is at most the zero probability and the mixture's
quantile at (t - pi) / (1 - pi) otherwise. The tail head is a small network that outputs offsets
from the plant forecast at 0.01, 0.02, 0.05, 0.95, 0.98, and 0.99, trained pooled on all bands with
the pinball loss and a non-crossing penalty.

    PYTHONPATH=ml-experiments/00310-metered-farm-long-lead-scoring:ml-experiments/00318-metered-farm-mdn-head-more-features \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00321-*/e00321_fit.py [zi|pooled|band|setting|farm|YYYY-MM ...]
"""

import itertools
import multiprocessing
import os
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Final

import e00310_common as c310  # ty: ignore[unresolved-import]
import e00310_fit as f310  # ty: ignore[unresolved-import]
import e00318_common as c318  # ty: ignore[unresolved-import]
import e00318_fit as f318  # ty: ignore[unresolved-import]
import e00321_common as common
import numpy as np
import polars as pl
import torch
from scipy.special import ndtr
from studies.tail_calibration import level_column
from torch import nn

N_PROCESSES: Final[int] = 8
SHUFFLE_SEED_OFFSET: Final[int] = 2_000


class ZeroInflatedHead(nn.Module):
    """Three-component Gaussian mixture on the residual plus a logit for an exact-zero half-hour."""

    def __init__(self, n_in: int) -> None:
        """Build the network with one extra output, the exact-zero logit."""
        super().__init__()
        k = f310.COMPONENTS
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 3 * k + 1)
        )
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.copy_(
                torch.tensor([0.0] * 3 + [-0.1, 0.0, 0.1] + [float(np.log(0.08))] * 3 + [-3.5])
            )

    def parameters_of(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the log mixture weights, means, standard deviations, and zero logit."""
        k = f310.COMPONENTS
        out = self.net(x)
        return (
            torch.log_softmax(out[:, :k], dim=1),
            out[:, k : 2 * k],
            torch.exp(out[:, 2 * k : 3 * k].clamp(-6.0, 1.0)),
            out[:, 3 * k],
        )

    def nll(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the negative log likelihood of the zero-inflated mixture."""
        log_w, mean, std, logit = self.parameters_of(batch["x"])
        z = (batch["actual"] - batch["mu"])[:, None]
        log_pdf = -0.5 * ((z - mean) / std) ** 2 - torch.log(std) - 0.5 * float(np.log(2 * np.pi))
        log_mix = torch.logsumexp(log_w + log_pdf, dim=1)
        is_zero = (batch["actual"] <= common.ZERO_OUTPUT).float()
        log_pi = nn.functional.logsigmoid(logit)
        log_not = nn.functional.logsigmoid(-logit)
        return -(is_zero * log_pi + (1.0 - is_zero) * (log_not + log_mix)).mean()


class TailHead(nn.Module):
    """Offsets from the plant forecast at the tail levels, trained with the pinball loss."""

    def __init__(self, n_in: int, init_offsets: np.ndarray) -> None:
        """Build the network with its last bias at the training residuals' quantiles."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, len(common.TAIL_TRAIN_LEVELS)),
        )
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.copy_(torch.tensor(init_offsets, dtype=torch.float32))

    def quantiles(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the plant forecast plus the network's offsets at the six training levels."""
        return batch["mu"][:, None] + self.net(batch["x"])

    def loss(self, batch: dict[str, torch.Tensor], *, penalty: bool) -> torch.Tensor:
        """Return the pinball loss, plus the non-crossing penalty when `penalty` is true."""
        q = self.quantiles(batch)
        level = torch.tensor(common.TAIL_TRAIN_LEVELS, dtype=torch.float32)
        error = batch["actual"][:, None] - q
        pin = torch.maximum(level * error, (level - 1.0) * error).mean()
        if not penalty:
            return pin
        crossing = torch.relu(q[:, :-1] - q[:, 1:]).mean()
        return pin + common.NON_CROSSING_WEIGHT * crossing


def zi_quantiles(
    *, model: ZeroInflatedHead, batch: dict[str, torch.Tensor], ceiling: float
) -> np.ndarray:
    """Return the zero-inflated mixture's 11 quantiles by bisection, clamped to [0, ceiling]."""
    with torch.no_grad():
        log_w, mean, std, logit = model.parameters_of(batch["x"])
    w, mean, std = (t.cpu().numpy().astype(np.float64) for t in (log_w.exp(), mean, std))
    pi = torch.sigmoid(logit).cpu().numpy().astype(np.float64)
    mu = batch["mu"].cpu().numpy().astype(np.float64)
    out = np.empty((len(mu), len(c310.LEVELS)))
    for k, level in enumerate(c310.LEVELS):
        target = np.clip((level - pi) / (1.0 - pi), 1e-6, 1.0 - 1e-6)
        lo, hi = np.full(len(mu), -2.0), np.full(len(mu), 2.0)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            cdf = (w * ndtr((mid[:, None] - mean) / std)).sum(axis=1)
            below = cdf < target
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        value = np.minimum(np.maximum(mu + 0.5 * (lo + hi), 0.0), ceiling)
        out[:, k] = np.where(level <= pi, 0.0, value)
    return out


def standardise(*, train: pl.DataFrame, inputs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return the training inputs' mean and standard deviation."""
    x = train.select(inputs).to_numpy().astype(np.float64)
    return x.mean(axis=0), x.std(axis=0) + 1e-6


def fit_loop(
    *,
    model: nn.Module,
    fit_b: dict[str, torch.Tensor],
    val_b: dict[str, torch.Tensor],
    loss_fn: Callable[[nn.Module, dict[str, torch.Tensor], bool], torch.Tensor],
    seed: int,
) -> None:
    """Train with AdamW and keep the epoch with the lowest validation loss."""
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=f310.LEARNING_RATE, weight_decay=f310.WEIGHT_DECAY
    )
    generator = torch.Generator().manual_seed(seed)
    best, best_state = float("inf"), None
    n = len(fit_b["mu"])
    for _ in range(f310.EPOCHS):
        model.train()
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, f310.BATCH):
            idx = order[start : start + f310.BATCH]
            loss = loss_fn(model, {k: v[idx] for k, v in fit_b.items()}, True)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            score = float(loss_fn(model, val_b, False))
        if score < best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()


def shuffle_columns(*, frame: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Return the frame with the control columns permuted jointly across rows."""
    order = rng.permutation(frame.height)
    return frame.with_columns(
        pl.Series(c, frame[c].to_numpy()[order]) for c in common.SHUFFLED_COLUMNS
    )


def train_zi(
    *,
    train: pl.DataFrame,
    test: pl.DataFrame,
    ceiling: float,
    seed: int,
) -> np.ndarray:
    """Fit the zero-inflated head on `train` and return `test`'s (rows, 11) quantiles."""
    torch.manual_seed(seed)
    cols = list(common.ZI_INPUTS)
    mean, std = standardise(train=train, inputs=cols)
    is_val = train["is_val"].to_numpy()
    fit_b = f318.make_batch(
        frame=train, inputs=cols, mean=mean, std=std, plant_base=True, rows=~is_val
    )
    val_b = f318.make_batch(
        frame=train, inputs=cols, mean=mean, std=std, plant_base=True, rows=is_val
    )
    model = ZeroInflatedHead(len(cols))
    fit_loop(
        model=model, fit_b=fit_b, val_b=val_b, loss_fn=lambda _m, b, _t: model.nll(b), seed=seed
    )
    test_b = f318.make_batch(
        frame=test, inputs=cols, mean=mean, std=std, plant_base=True, rows=None
    )
    return zi_quantiles(model=model, batch=test_b, ceiling=ceiling)


def train_tail(
    *,
    train: pl.DataFrame,
    tests: dict[str, pl.DataFrame],
    ceilings: dict[str, float],
    shuffle: bool,
    seed: int,
) -> dict[str, np.ndarray]:
    """Fit the tail head on pooled rows and return each band's (rows, 4) stored-level quantiles."""
    torch.manual_seed(seed)
    if shuffle:
        rng = np.random.default_rng(SHUFFLE_SEED_OFFSET + seed)
        train = shuffle_columns(frame=train, rng=rng)
        tests = {k: shuffle_columns(frame=v, rng=rng) for k, v in tests.items()}
    cols = list(common.TAIL_INPUTS)
    mean, std = standardise(train=train, inputs=cols)
    is_val = train["is_val"].to_numpy()
    residual = (train["output"] - train["mu"]).to_numpy()
    init = np.quantile(residual, common.TAIL_TRAIN_LEVELS)
    fit_b = f318.make_batch(
        frame=train, inputs=cols, mean=mean, std=std, plant_base=True, rows=~is_val
    )
    val_b = f318.make_batch(
        frame=train, inputs=cols, mean=mean, std=std, plant_base=True, rows=is_val
    )
    model = TailHead(len(cols), init)
    fit_loop(
        model=model,
        fit_b=fit_b,
        val_b=val_b,
        loss_fn=lambda _m, b, penalty: model.loss(b, penalty=penalty),
        seed=seed,
    )
    keep = [common.TAIL_TRAIN_LEVELS.index(lv) for lv in common.TAIL_STORED_LEVELS]
    out = {}
    for key, frame in tests.items():
        batch = f318.make_batch(
            frame=frame, inputs=cols, mean=mean, std=std, plant_base=True, rows=None
        )
        with torch.no_grad():
            q = model.quantiles(batch).numpy().astype(np.float64)[:, keep]
        out[key] = np.minimum(np.maximum(q, 0.0), ceilings[key])
    return out


def out_path(*, band: str, setting: str, site: str, arm: str, seed: int, start: datetime) -> Path:
    """Return a job's output file."""
    return common.PRED_DIR / band / setting / f"{arm}_s{seed}_{site}_{start:%Y-%m}.parquet"


def write(
    *,
    st: f310.Setting,
    band: str,
    setting: str,
    arm: str,
    seed: int,
    start: datetime,
    columns: dict[str, np.ndarray],
) -> None:
    """Write a job's test rows atomically; `columns` maps quantile column names to values."""
    path = out_path(band=band, setting=setting, site=st.site, arm=arm, seed=seed, start=start)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = st.test.select("time", "init_time", "lead_hours").with_columns(
        time_series_id=pl.lit(st.site),
        arm=pl.lit(arm),
        seed=pl.lit(seed),
        setting=pl.lit(setting),
        band=pl.lit(band),
        actual=st.test["output"].cast(pl.Float64),
        **{name: pl.Series(values) for name, values in columns.items()},
    )
    tmp = path.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)


def all_levels(*, q: np.ndarray) -> dict[str, np.ndarray]:
    """Map a (rows, 11) array to the standard quantile column names."""
    return {level_column(level=lv): q[:, i] for i, lv in enumerate(c310.LEVELS)}


def tail_levels(*, q: np.ndarray) -> dict[str, np.ndarray]:
    """Map a (rows, 4) array to the four stored tail column names."""
    return {level_column(level=lv): q[:, i] for i, lv in enumerate(common.TAIL_STORED_LEVELS)}


def seeds_for(setting: str) -> tuple[int, ...]:
    """Return the neural seeds for a history setting."""
    return common.OWN_SEEDS if setting == "own_history" else common.ZERO_SEEDS


def run_zi_job(task: tuple[str, str, str, datetime]) -> str:
    """Fit the zero-inflated head for one (band, setting, farm, refit)."""
    torch.set_num_threads(1)
    band, setting, site, start = task
    started = time.time()
    todo = [
        s
        for s in seeds_for(setting)
        if not out_path(
            band=band, setting=setting, site=site, arm="zi", seed=s, start=start
        ).exists()
    ]
    if not todo:
        return f"zi {band} {setting} {site} {start:%Y-%m} skipped"
    st = f318.new_setting(band=band, setting=setting, site=site, start=start)
    for seed in todo:
        q = train_zi(train=st.train, test=st.test, ceiling=st.ceiling, seed=seed)
        write(
            st=st,
            band=band,
            setting=setting,
            arm="zi",
            seed=seed,
            start=start,
            columns=all_levels(q=q),
        )
    return f"zi {band} {setting} {site} {start:%Y-%m} {time.time() - started:.0f}s"


def run_pooled_job(task: tuple[str, str, datetime]) -> str:
    """Fit the tail head (and its control) and arm 4's zero-history network for one (setting, farm, refit)."""
    torch.set_num_threads(1)
    setting, site, start = task
    started = time.time()
    own = setting == "own_history"
    plan: list[tuple[str, int]] = [("tail", s) for s in seeds_for(setting)]
    if own:
        plan += [("tail_shuffled", s) for s in common.CONTROL_SEEDS]
    else:
        plan += [("pure_lead_pooled", s) for s in common.ZERO_SEEDS]
    todo = [
        (a, s)
        for a, s in plan
        if not all(
            out_path(band=b, setting=setting, site=site, arm=a, seed=s, start=start).exists()
            for b in c310.BAND_ORDER
        )
    ]
    if not todo:
        return f"pooled {setting} {site} {start:%Y-%m} skipped"
    sts = {
        b: f318.new_setting(band=b, setting=setting, site=site, start=start)
        for b in c310.BAND_ORDER
    }
    train = pl.concat([st.train for st in sts.values()])
    tests = {b: st.test for b, st in sts.items()}
    ceilings = {b: st.ceiling for b, st in sts.items()}
    for arm, seed in todo:
        if arm == "pure_lead_pooled":
            quantiles = f318.train_head(
                train=train,
                tests=tests,
                ceilings=ceilings,
                inputs=(*c318.PURE_COLUMNS, common.PURE_LEAD_INPUT_EXTRA),
                plant_base=False,
                shuffle=False,
                seed=seed,
            )
            mapper = lambda q: all_levels(q=q)  # noqa: E731
        else:
            quantiles = train_tail(
                train=train,
                tests=tests,
                ceilings=ceilings,
                shuffle=arm == "tail_shuffled",
                seed=seed,
            )
            mapper = lambda q: tail_levels(q=q)  # noqa: E731
        for b, st in sts.items():
            write(
                st=st,
                band=b,
                setting=setting,
                arm=arm,
                seed=seed,
                start=start,
                columns=mapper(quantiles[b]),
            )
    return f"pooled {setting} {site} {start:%Y-%m} {time.time() - started:.0f}s"


def run_any(task: tuple) -> str:
    """Dispatch a zero-inflated job or a pooled job."""
    return run_zi_job(task) if len(task) == 4 else run_pooled_job(task)


def main() -> None:
    """Run all jobs in parallel; `zi` or `pooled` restricts the job kind, other arguments filter."""
    args = set(sys.argv[1:])
    kinds = {a for a in args if a in ("zi", "pooled")}
    keep = args - kinds
    bands, settings = set(c310.BAND_ORDER), set(c310.SETTINGS)
    farms, months = set(c310.FARMS), {f"{s:%Y-%m}" for s in f310.REFIT_STARTS}
    groups = (bands, settings, farms, months)

    def matches(labels: tuple[str, ...]) -> bool:
        return all(not (keep & g) or bool(set(labels) & keep & g) for g in groups)

    pooled = [
        t
        for t in itertools.product(c310.SETTINGS, c310.FARMS, f310.REFIT_STARTS)
        if not (keep & bands) and matches((t[0], t[1], f"{t[2]:%Y-%m}"))
    ]
    zi = [
        t
        for t in itertools.product(
            reversed(c310.BAND_ORDER), c310.SETTINGS, c310.FARMS, f310.REFIT_STARTS
        )
        if matches((t[0], t[1], t[2], f"{t[3]:%Y-%m}"))
    ]
    tasks: list[tuple] = []
    if not kinds or "pooled" in kinds:
        tasks += pooled
    if not kinds or "zi" in kinds:
        tasks += zi
    with multiprocessing.get_context("spawn").Pool(N_PROCESSES) as pool:
        for done in pool.imap_unordered(run_any, tasks):
            print("finished", done, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
