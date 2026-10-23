"""A small multilayer perceptron that corrects arm D's quantiles.

The network reads an arm's inputs and returns two numbers per row: an additive correction `delta`
to the median (points) and a log scale `s` on the band widths. The corrected quantiles are
`q50 + delta + exp(s) * (q_k - q50)` for each of the 11 levels. The last layer starts at zero, so
training starts from arm D exactly. The loss is twice the mean pinball loss over the 11 levels, the
quantile score of experiment 87, in points of effective capacity.
"""

from dataclasses import dataclass
from typing import Final

import e00278_h_common as common
import numpy as np
import polars as pl
import torch
from torch import nn

LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
MEDIAN_INDEX: Final[int] = 5
HIDDEN: Final[int] = 64
BATCH: Final[int] = 16384
MAX_EPOCHS: Final[int] = 40
PATIENCE: Final[int] = 4
LEARNING_RATE: Final[float] = 2e-3
WEIGHT_DECAY: Final[float] = 1e-3
LOG_SCALE_LIMIT: Final[float] = 1.0
INPUT_CLIP_SD: Final[float] = 5.0
"""Standardised inputs are clipped here: a lagged-power spike of 140 points otherwise sends arm 4's
median far outside the data."""


class Corrector(nn.Module):
    """Two hidden layers and a zero-initialised output layer."""

    def __init__(self, *, n_inputs: int) -> None:
        """Build the network."""
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_inputs, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
        )
        self.head = nn.Linear(HIDDEN, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the corrected quantiles (rows, 11) and the raw output (rows, 2)."""
        out = self.head(self.body(x))
        delta = out[:, 0:1]
        scale = torch.exp(torch.clamp(out[:, 1:2], -LOG_SCALE_LIMIT, LOG_SCALE_LIMIT))
        median = q[:, MEDIAN_INDEX : MEDIAN_INDEX + 1]
        return median + delta + scale * (q - median), out


def pinball(*, actual: torch.Tensor, quantiles: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Return twice the mean pinball loss over the levels, per row."""
    error = actual[:, None] - quantiles
    return 2.0 * torch.maximum(levels * error, (levels - 1.0) * error).mean(dim=1)


@dataclass(frozen=True)
class FitResult:
    """A fitted network's corrected quantiles on every row and its training history."""

    quantiles: np.ndarray
    delta: np.ndarray
    log_scale: np.ndarray
    epochs: int
    best_val: float
    base_val: float


def _predict_all(
    *, model: Corrector, xt: torch.Tensor, qt: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Return the corrected quantiles and the raw outputs on every row."""
    quantiles, raw = [], []
    with torch.no_grad():
        for start in range(0, len(xt), 1 << 17):
            corrected, out = model(xt[start : start + (1 << 17)], qt[start : start + (1 << 17)])
            quantiles.append(corrected.cpu().numpy())
            raw.append(out.cpu().numpy())
    return np.concatenate(quantiles), np.concatenate(raw)


def fit_predict(
    *,
    x: np.ndarray,
    q: np.ndarray,
    actual: np.ndarray,
    split: np.ndarray,
    seed: int,
    device: str = "cuda",
) -> FitResult:
    """Train on the `train` rows, stop on the `val` rows, and predict every row.

    Args:
        x: The unstandardised inputs, shape (rows, d).
        q: Arm D's 11 quantiles, shape (rows, 11), in points.
        actual: The target, in points.
        split: `train`, `val`, or `test` per row.
        seed: The seed for the initial weights and the batch order.
        device: The torch device.

    Returns:
        The corrected quantiles on all rows and the fit's history.
    """
    torch.manual_seed(seed)
    train = split == "train"
    mu, sd = x[train].mean(axis=0), x[train].std(axis=0) + 1e-6
    standardised = np.clip((x - mu) / sd, -INPUT_CLIP_SD, INPUT_CLIP_SD)
    xt = torch.tensor(standardised, dtype=torch.float32, device=device)
    qt = torch.tensor(q, dtype=torch.float32, device=device)
    yt = torch.tensor(actual, dtype=torch.float32, device=device)
    levels = torch.tensor(LEVELS, dtype=torch.float32, device=device)[None]
    train_idx = torch.tensor(np.flatnonzero(train), device=device)
    val_idx = torch.tensor(np.flatnonzero(split == "val"), device=device)
    model = Corrector(n_inputs=x.shape[1]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def loss_on(index: torch.Tensor) -> float:
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(index), 1 << 17):
                batch = index[start : start + (1 << 17)]
                corrected, _ = model(xt[batch], qt[batch])
                total += pinball(actual=yt[batch], quantiles=corrected, levels=levels).sum().item()
                count += len(batch)
        return total / count

    with torch.no_grad():
        base_val = float(
            pinball(actual=yt[val_idx], quantiles=qt[val_idx], levels=levels).mean().item()
        )
    best, best_state, bad, epochs = np.inf, None, 0, 0
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), device=device, generator=generator)]
        for start in range(0, len(order), BATCH):
            batch = order[start : start + BATCH]
            corrected, _ = model(xt[batch], qt[batch])
            loss = pinball(actual=yt[batch], quantiles=corrected, levels=levels).mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        val = loss_on(val_idx)
        epochs = epoch + 1
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    quantiles, raw_all = _predict_all(model=model, xt=xt, qt=qt)
    return FitResult(
        quantiles=quantiles,
        delta=raw_all[:, 0],
        log_scale=np.clip(raw_all[:, 1], -LOG_SCALE_LIMIT, LOG_SCALE_LIMIT),
        epochs=epochs,
        best_val=best,
        base_val=base_val,
    )


def quantile_columns() -> list[str]:
    """Return the 11 quantile column names."""
    return list(common.QUANTILES)


def quantile_matrix(*, frame: pl.DataFrame) -> np.ndarray:
    """Return arm D's quantiles, shape (rows, 11), float32."""
    return frame.select(quantile_columns()).to_numpy().astype(np.float32)


def row_scores(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return the quantile score of each row, in points."""
    error = actual[:, None] - quantiles
    return 2.0 * np.maximum(LEVELS[None] * error, (LEVELS[None] - 1.0) * error).mean(axis=1)
