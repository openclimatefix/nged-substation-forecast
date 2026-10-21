"""The sequence network, its loss, and its training with month-block early stopping.

The network reads 15 day vectors with a gated recurrent unit, adds a substation embedding and the
target day's features, and emits per half-hour of day D+1 a median correction `c` and a lower and
an upper log-scale. The 11 quantiles are `c + scale * (K0 quantile - K0 median)`, with the scale
chosen by side, so they cannot cross because K0's own offsets are increasing.
"""

from dataclasses import dataclass
from typing import Final

import e00142_data as data_module
import numpy as np
import torch
from torch import nn

LEVELS: Final[torch.Tensor] = torch.tensor(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
MEDIAN_INDEX: Final[int] = 5
HIDDEN: Final[int] = 32
EMBEDDING: Final[int] = 4
HEAD: Final[int] = 64
MAX_EPOCHS: Final[int] = 150
PATIENCE: Final[int] = 15
BATCH: Final[int] = 128
LEARNING_RATE: Final[float] = 3e-3
WEIGHT_DECAY: Final[float] = 1e-2
DROPOUT: Final[float] = 0.1
CORRECTION_SCALE: Final[float] = 3.0
LOG_SCALE_LIMIT: Final[float] = 0.7
PURGE_DAYS: Final[int] = 15


class SequenceNet(nn.Module):
    """Gated recurrent unit over day vectors with a head on the target day's features."""

    def __init__(self, *, n_sites: int, use_history: bool = True) -> None:
        """Build the network; `use_history=False` zeroes the day vectors' residual inputs."""
        super().__init__()
        self.use_history = use_history
        self.gru = nn.GRU(data_module.N_DAY_FEATURES, HIDDEN, batch_first=True)
        self.embedding = nn.Embedding(n_sites, EMBEDDING)
        last = nn.Linear(HEAD, 3 * data_module.N_BINS)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN + EMBEDDING + data_module.N_TARGET_FEATURES, HEAD),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            last,
        )
        self.skip = nn.Linear(3 * data_module.N_BINS, data_module.N_BINS)
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(
        self, *, history: torch.Tensor, target: torch.Tensor, site: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the median correction, lower and upper scale, each shape (batch, 48)."""
        if not self.use_history:
            history = history.clone()
            history[:, :, : 2 * data_module.N_BINS] = 0.0
        n = data_module.N_BINS
        residual, seen = history[:, :14, :n], history[:, :14, n : 2 * n]
        profile = residual.sum(dim=1) / seen.sum(dim=1).clamp(min=1.0)
        skip = self.skip(torch.cat([profile, history[:, 13, :n], history[:, 14, :n]], dim=1))
        _, hidden = self.gru(history)
        features = torch.cat([hidden[0], self.embedding(site), target], dim=1)
        out = self.head(features)
        correction = CORRECTION_SCALE * torch.tanh(out[:, :n]) + data_module.RESIDUAL_SCALE * skip
        lower = torch.exp(LOG_SCALE_LIMIT * torch.tanh(out[:, n : 2 * n]))
        upper = torch.exp(LOG_SCALE_LIMIT * torch.tanh(out[:, 2 * n :]))
        return correction, lower, upper


def quantiles_of(
    *, correction: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Return the 11 quantile offsets from K0's median, shape (batch, 48, 11)."""
    scale = torch.where(
        torch.arange(11, device=offsets.device) < MEDIAN_INDEX,
        lower[:, :, None],
        upper[:, :, None],
    )
    return correction[:, :, None] + scale * offsets


def pinball(*, y: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """Return the mean pinball loss over the readings that exist, in points."""
    levels = LEVELS.to(y.device)
    mask = torch.isfinite(y)
    error = torch.where(mask, y, 0.0)[:, :, None] - quantiles
    loss = torch.maximum(levels * error, (levels - 1.0) * error).mean(dim=2)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def median_loss(*, y: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
    """Return the mean absolute error of the median correction."""
    mask = torch.isfinite(y)
    error = (torch.where(mask, y, 0.0) - correction).abs()
    return (error * mask).sum() / mask.sum().clamp(min=1)


@dataclass(frozen=True)
class Pooled:
    """Samples of every substation stacked, with each sample's site index."""

    history: np.ndarray
    target: np.ndarray
    offsets: np.ndarray
    y: np.ndarray
    site: np.ndarray
    target_day: np.ndarray
    month_code: np.ndarray
    period: np.ndarray
    where: np.ndarray
    """Shape (n, 2): the site's index in the pooled list and the sample's index within it."""


def pool(*, samples: list[data_module.Samples]) -> Pooled:
    """Stack the substations' samples."""
    site = np.concatenate([np.full(len(s.run), i) for i, s in enumerate(samples)])
    where = np.concatenate(
        [
            np.stack([np.full(len(s.run), i), np.arange(len(s.run))], axis=1)
            for i, s in enumerate(samples)
        ]
    )
    cat = lambda name: np.concatenate([getattr(s, name) for s in samples])  # noqa: E731
    return Pooled(
        history=cat("history"),
        target=cat("target"),
        offsets=cat("k0_offsets"),
        y=cat("y"),
        site=site,
        target_day=cat("target_day"),
        month_code=cat("month_code"),
        period=cat("period"),
        where=where,
    )


def _batch(*, pooled: Pooled, index: np.ndarray, device: str) -> dict[str, torch.Tensor]:
    t = lambda a, dtype=torch.float32: torch.as_tensor(a[index], dtype=dtype, device=device)  # noqa: E731
    return {
        "history": t(pooled.history),
        "target": t(pooled.target),
        "offsets": t(pooled.offsets),
        "y": t(pooled.y),
        "site": t(pooled.site, torch.long),
    }


def _loss(*, net: SequenceNet, batch: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    correction, lower, upper = net(
        history=batch["history"], target=batch["target"], site=batch["site"]
    )
    if mode == "median":
        return median_loss(y=batch["y"], correction=correction)
    quantiles = quantiles_of(
        correction=correction, lower=lower, upper=upper, offsets=batch["offsets"]
    )
    return pinball(y=batch["y"], quantiles=quantiles)


@torch.no_grad()
def evaluate(
    *, net: SequenceNet, pooled: Pooled, index: np.ndarray, mode: str, device: str
) -> float:
    """Return the mean loss over samples, in batches."""
    net.eval()
    total, count = 0.0, 0
    for start in range(0, len(index), 512):
        chunk = index[start : start + 512]
        n = int(np.isfinite(pooled.y[chunk]).sum())
        total += (
            float(
                _loss(net=net, batch=_batch(pooled=pooled, index=chunk, device=device), mode=mode)
            )
            * n
        )
        count += n
    return total / max(count, 1)


def fit(
    *,
    pooled: Pooled,
    train: np.ndarray,
    validation: np.ndarray,
    n_sites: int,
    mode: str,
    seed: int,
    device: str,
    use_history: bool = True,
) -> tuple[SequenceNet, dict]:
    """Train one network with early stopping on `validation`; return the best and its history."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = SequenceNet(n_sites=n_sites, use_history=use_history).to(device)
    optimiser = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best, best_state, stale, log = np.inf, None, 0, []
    for _epoch in range(MAX_EPOCHS):
        net.train()
        order = rng.permutation(train)
        for start in range(0, len(order), BATCH):
            batch = _batch(pooled=pooled, index=order[start : start + BATCH], device=device)
            optimiser.zero_grad()
            _loss(net=net, batch=batch, mode=mode).backward()
            optimiser.step()
        valid = evaluate(net=net, pooled=pooled, index=validation, mode=mode, device=device)
        log.append(valid)
        if valid < best - 1e-5:
            best, stale = valid, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    assert best_state is not None
    net.load_state_dict(best_state)
    train_loss = evaluate(net=net, pooled=pooled, index=train, mode=mode, device=device)
    return net, {"epochs": len(log), "validation": best, "train": train_loss}


def month_blocks(*, month_codes: np.ndarray, n_folds: int) -> list[np.ndarray]:
    """Split the sorted distinct months into `n_folds` contiguous blocks of month codes."""
    months = np.unique(month_codes)
    return [np.array(block) for block in np.array_split(months, n_folds)]


def purged_split(
    *, pooled: Pooled, candidates: np.ndarray, block: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train, validation) indices for a held-out block.

    Training leaves out `PURGE_DAYS` either side of the block.
    """
    in_block = np.isin(pooled.month_code[candidates], block)
    validation = candidates[in_block]
    low, high = pooled.target_day[validation].min(), pooled.target_day[validation].max()
    near = (pooled.target_day[candidates] >= low - PURGE_DAYS) & (
        pooled.target_day[candidates] <= high + PURGE_DAYS
    )
    return candidates[~near], validation


@torch.no_grad()
def predict(
    *, nets: list[SequenceNet], pooled: Pooled, index: np.ndarray, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the nets' mean correction, lower and upper scale, each shape (len(index), 48)."""
    outs = []
    for net in nets:
        net.eval()
        batch = _batch(pooled=pooled, index=index, device=device)
        c, lo, up = net(history=batch["history"], target=batch["target"], site=batch["site"])
        outs.append([c.cpu().numpy(), lo.cpu().numpy(), up.cpu().numpy()])
    return tuple(np.mean([o[i] for o in outs], axis=0) for i in range(3))  # ty: ignore[invalid-return-type]
