"""The GAM-stage correction network: a small multilayer perceptron fitted by the Huber loss.

The network reads the convex model's own forecast and the weather and calendar at the target, and
returns an additive correction to the convex model's mean. The last layer starts at zero, so
training starts from the uncorrected convex model. The Huber loss has a 10-point threshold: the
Kalman filter that follows estimates a mean error, but the convex model's residuals include load
transfers of 100 points or more, which a squared-error loss would chase. The spread is refitted
afterwards from the corrected forecasts' residuals.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from torch import nn

HIDDEN: Final[int] = 64
BATCH: Final[int] = 16384
MAX_EPOCHS: Final[int] = 120
PATIENCE: Final[int] = 6
LEARNING_RATE: Final[float] = 2e-3
WEIGHT_DECAY: Final[float] = 1e-3
INPUT_CLIP_SD: Final[float] = 5.0
HUBER_DELTA: Final[float] = 10.0
"""Residuals beyond this many points count linearly, not quadratically."""


def huber(*, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the mean Huber loss."""
    return nn.functional.huber_loss(prediction, target, delta=HUBER_DELTA)


class Corrector(nn.Module):
    """Two hidden layers and a zero-initialised output layer."""

    def __init__(self, *, n_inputs: int) -> None:
        """Build the network."""
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_inputs, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN), nn.GELU()
        )
        self.head = nn.Linear(HIDDEN, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the correction per row, shape (rows,)."""
        return self.head(self.body(x))[:, 0]


@dataclass(frozen=True)
class Fitted:
    """A fitted network with its input standardisation.

    Attributes:
        model: The trained network, in evaluation mode.
        mean: The training inputs' column means.
        scale: The training inputs' column standard deviations.
        epochs: The number of epochs run.
        best_val: The best validation Huber loss.
        base_val: The validation Huber loss of a zero correction.
    """

    model: Corrector
    mean: np.ndarray
    scale: np.ndarray
    epochs: int
    best_val: float
    base_val: float

    def predict(self, *, x: np.ndarray) -> np.ndarray:
        """Return the correction for each row of `x`, in points."""
        z = np.clip((x - self.mean) / self.scale, -INPUT_CLIP_SD, INPUT_CLIP_SD)
        out = []
        with torch.no_grad():
            for start in range(0, len(z), 1 << 17):
                chunk = torch.tensor(z[start : start + (1 << 17)], dtype=torch.float32)
                out.append(self.model(chunk).numpy())
        return np.concatenate(out).astype(np.float64)


def fit(*, x: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray, seed: int) -> Fitted:
    """Train on the `train` rows and stop on the `val` rows.

    Args:
        x: The unstandardised inputs, shape (rows, d).
        y: The target (observed minus the convex model's forecast), in points.
        train: A boolean mask of the training rows.
        val: A boolean mask of the early-stopping rows.
        seed: The seed for the initial weights and the batch order.

    Returns:
        The fitted network.
    """
    torch.manual_seed(seed)
    mean, scale = x[train].mean(axis=0), x[train].std(axis=0) + 1e-6
    z = torch.tensor(
        np.clip((x - mean) / scale, -INPUT_CLIP_SD, INPUT_CLIP_SD), dtype=torch.float32
    )
    t = torch.tensor(y, dtype=torch.float32)
    train_idx, val_idx = torch.tensor(np.flatnonzero(train)), torch.tensor(np.flatnonzero(val))
    model = Corrector(n_inputs=x.shape[1])
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def loss_on(index: torch.Tensor) -> float:
        model.eval()
        with torch.no_grad():
            return float(huber(prediction=model(z[index]), target=t[index]))

    base_val = float(huber(prediction=torch.zeros_like(t[val_idx]), target=t[val_idx]))
    best, best_state, bad, epochs = np.inf, None, 0, 0
    generator = torch.Generator()
    generator.manual_seed(seed)
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), generator=generator)]
        for start in range(0, len(order), BATCH):
            batch = order[start : start + BATCH]
            loss = huber(prediction=model(z[batch]), target=t[batch])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        epochs = epoch + 1
        val_loss = loss_on(val_idx)
        if val_loss < best - 1e-5:
            best, bad = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is None:
        msg = "the network never produced a finite validation loss"
        raise RuntimeError(msg)
    model.load_state_dict(best_state)
    model.eval()
    return Fitted(
        model=model, mean=mean, scale=scale, epochs=epochs, best_val=best, base_val=base_val
    )
