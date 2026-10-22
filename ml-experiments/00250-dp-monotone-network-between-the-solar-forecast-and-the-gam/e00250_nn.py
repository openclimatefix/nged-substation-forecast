"""A small neural network that is monotone in the solar forecast and saturates.

The network maps the solar term `s` (the unit-free share of the held capacity) and side inputs `x`
to the solar output `g` that the demand model should subtract:

    raw = s * softplus(a + x.u) + sum_j softplus(v_j) * softplus(softplus(w_j) * s + x.P_j + b_j)
    g   = G * (1 - exp(-relu(raw) / G))

`raw` is non-decreasing in `s` because every weight on the solar path is made non-negative with a
softplus and every activation is non-decreasing. `g` is bounded by `G`, and the value at `s = 0`
is subtracted so that nothing is removed at night. Training starts from `g = s` with a small
hidden contribution.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

HIDDEN: Final[int] = 16
LEARNING_RATE: Final[float] = 5e-3
WEIGHT_DECAY: Final[float] = 1e-3
MAX_EPOCHS: Final[int] = 80
PATIENCE: Final[int] = 8
BATCH: Final[int] = 4096
VALIDATION_SHARE: Final[float] = 0.15
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
SOFTPLUS_ONE: Final[float] = 0.5413


class MonotoneSolar(nn.Module):
    """Monotone, saturating map from the solar term and side inputs to the subtracted output."""

    def __init__(self, *, n_side: int) -> None:
        """Build the network with the identity as its starting point."""
        super().__init__()
        self.skew = nn.Linear(n_side, 1)
        nn.init.zeros_(self.skew.weight)
        nn.init.constant_(self.skew.bias, SOFTPLUS_ONE)
        self.solar_weight = nn.Parameter(torch.zeros(HIDDEN) + SOFTPLUS_ONE)
        self.side = nn.Linear(n_side, HIDDEN)
        self.output_weight = nn.Parameter(torch.full((HIDDEN,), -4.0))
        self.log_ceiling = nn.Parameter(torch.tensor(1.5))

    def saturated(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Return the saturating output before the zero point is subtracted."""
        linear = s * nn.functional.softplus(self.skew(x)).squeeze(-1)
        hidden = nn.functional.softplus(
            nn.functional.softplus(self.solar_weight) * s[:, None] + self.side(x)
        )
        raw = linear + (hidden * nn.functional.softplus(self.output_weight)).sum(dim=-1)
        ceiling = nn.functional.softplus(self.log_ceiling) * 10.0
        return ceiling * (1.0 - torch.exp(-torch.relu(raw) / ceiling))

    def forward(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Return `g` for solar terms `s` (rows,) and side inputs `x` (rows, n_side).

        The output at `s = 0` is subtracted, so the network removes nothing at night.
        """
        return self.saturated(s, x) - self.saturated(torch.zeros_like(s), x)


@dataclass(frozen=True)
class Fitted:
    """A trained ensemble of networks (one per seed)."""

    models: tuple[MonotoneSolar, ...]

    def predict(self, *, s: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return the seed-averaged output `g` for the rows."""
        with torch.no_grad():
            tensors = (
                torch.as_tensor(s, dtype=torch.float32),
                torch.as_tensor(x, dtype=torch.float32),
            )
            return np.mean([m(*tensors).numpy() for m in self.models], axis=0).astype(np.float64)


def fit_network(*, s: np.ndarray, x: np.ndarray, base: np.ndarray, target: np.ndarray) -> Fitted:
    """Train one network per seed on the mean absolute error of `target - (base - g)`.

    All arrays are unit-free (points divided by 100). The last `VALIDATION_SHARE` of the rows, in
    the order given (ascending in time), stop the training early.
    """
    torch.set_num_threads(1)
    n = len(s)
    split = int(n * (1.0 - VALIDATION_SHARE))
    tensors = [torch.as_tensor(a, dtype=torch.float32) for a in (s, x, base, target)]
    models = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = MonotoneSolar(n_side=x.shape[1])
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        best, best_state, stale = np.inf, None, 0
        generator = torch.Generator().manual_seed(seed)
        for _ in range(MAX_EPOCHS):
            order = torch.randperm(split, generator=generator)
            for start in range(0, split, BATCH):
                batch = order[start : start + BATCH]
                s_b, x_b, base_b, target_b = (t[batch] for t in tensors)
                loss = (target_b - (base_b - model(s_b, x_b))).abs().mean()
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            with torch.no_grad():
                s_v, x_v, base_v, target_v = (t[split:] for t in tensors)
                validation = float((target_v - (base_v - model(s_v, x_v))).abs().mean())
            if validation < best - 1e-6:
                best, stale = validation, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        models.append(model.eval())
    return Fitted(models=tuple(models))
