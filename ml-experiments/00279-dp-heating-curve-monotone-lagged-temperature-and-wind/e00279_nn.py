"""A small neural network for the heating term, monotone in every temperature and in the wind.

The network maps the cold excess `c_k = (15.5 - T_k) / 10` of three daily mean temperatures, the
wind speed over 5 m/s, and a group (hour band, day type, half of the hour band) to the heating
power the demand model should add:

    h1 = softplus(W1 [c, w] + e_group + b1)           W1 >= 0
    h2 = softplus(W2 h1 + b2)                          W2 >= 0
    f  = v . h2                                        v >= 0
    g  = softplus(s_group) * relu(f(c, w) - f(c_ref, w))

Every weight on the temperature and wind path is non-negative and every activation is
non-decreasing, so `g` never falls as any temperature falls or the wind rises. The reference
point `c_ref` is 18 C on every temperature at the same wind speed, so `g` is exactly zero when the
weather is warm. The network is first fitted to reproduce the model's own heating term, then to the
residual target, and the epoch with the lowest validation error (the distilled start included) is
kept.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

HIDDEN: Final[int] = 16
LEARNING_RATE: Final[float] = 5e-3
WEIGHT_DECAY: Final[float] = 1e-4
DISTIL_EPOCHS: Final[int] = 30
MAX_EPOCHS: Final[int] = 60
PATIENCE: Final[int] = 8
BATCH: Final[int] = 2048
VALIDATION_SHARE: Final[float] = 0.15
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
N_GROUPS: Final[int] = 16
THRESHOLD_C: Final[float] = 15.5
REFERENCE_C: Final[float] = 18.0
TEMPERATURE_SCALE_C: Final[float] = 10.0
WIND_SCALE_M_S: Final[float] = 5.0
SOFTPLUS_ONE: Final[float] = 0.5413
REFERENCE_EXCESS: Final[float] = (THRESHOLD_C - REFERENCE_C) / TEMPERATURE_SCALE_C
"""The cold excess of the reference weather, 18 C: negative, so the network is zero when warm."""


def cold_excess(*, temperature_c: np.ndarray) -> np.ndarray:
    """Return the network's temperature inputs, `(15.5 - T) / 10`, positive when cold."""
    return (THRESHOLD_C - temperature_c) / TEMPERATURE_SCALE_C


class MonotoneHeating(nn.Module):
    """Heating power, non-decreasing in the cold excess of every temperature and in the wind."""

    def __init__(self, *, n_temperatures: int, use_wind: bool) -> None:
        """Build the network with small random first-layer weights."""
        super().__init__()
        self.n_temperatures = n_temperatures
        n_inputs = n_temperatures + int(use_wind)
        self.first = nn.Parameter(torch.zeros(HIDDEN, n_inputs))
        self.group_bias = nn.Embedding(N_GROUPS, HIDDEN)
        nn.init.normal_(self.group_bias.weight, std=0.05)
        self.first_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.second = nn.Parameter(torch.zeros(HIDDEN, HIDDEN) - 2.0)
        self.second_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.output = nn.Parameter(torch.zeros(HIDDEN) - 1.0)
        self.group_scale = nn.Parameter(torch.zeros(N_GROUPS) + SOFTPLUS_ONE)
        with torch.no_grad():
            self.first.add_(torch.randn_like(self.first) * 0.1)

    def raw(self, x: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        """Return `f` for inputs `x` (rows, n_inputs) in group `group` (rows,)."""
        h1 = nn.functional.softplus(
            x @ nn.functional.softplus(self.first).T + self.group_bias(group) + self.first_bias
        )
        h2 = nn.functional.softplus(h1 @ nn.functional.softplus(self.second).T + self.second_bias)
        return h2 @ nn.functional.softplus(self.output)

    def forward(self, x: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        """Return the heating power `g` of the rows, zero when every temperature is warm."""
        reference = x.clone()
        reference[:, : self.n_temperatures] = REFERENCE_EXCESS
        gain = nn.functional.softplus(self.group_scale)[group]
        return gain * torch.relu(self.raw(x, group) - self.raw(reference, group))


@dataclass(frozen=True)
class Fitted:
    """A trained ensemble of networks, one per seed."""

    models: tuple[MonotoneHeating, ...]
    scale: float

    def predict(self, *, x: np.ndarray, group: np.ndarray) -> np.ndarray:
        """Return the seed-averaged heating power of the rows, in the target's units."""
        with torch.no_grad():
            tensors = (
                torch.as_tensor(x, dtype=torch.float32),
                torch.as_tensor(group, dtype=torch.int64),
            )
            return self.scale * np.mean([m(*tensors).numpy() for m in self.models], axis=0).astype(
                np.float64
            )


def inputs(*, temperatures_c: np.ndarray, wind_m_s: np.ndarray | None) -> np.ndarray:
    """Return the network input matrix: cold excesses first, then the wind, if used."""
    columns = [
        cold_excess(temperature_c=temperatures_c[:, k]) for k in range(temperatures_c.shape[1])
    ]
    if wind_m_s is not None:
        columns.append(wind_m_s / WIND_SCALE_M_S)
    return np.stack(columns, axis=1)


def _train(
    *,
    model: MonotoneHeating,
    tensors: tuple[torch.Tensor, ...],
    split: int,
    epochs: int,
    seed: int,
    keep_best: bool,
    learning_rate: float,
) -> None:
    """Train `model` on the first `split` rows for `epochs`, early-stopping on the rest."""
    x, group, target = tensors
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed)

    def validation() -> float:
        with torch.no_grad():
            return float(((target[split:] - model(x[split:], group[split:])) ** 2).mean())

    best, best_state, stale = validation(), {k: v.clone() for k, v in model.state_dict().items()}, 0
    for _ in range(epochs):
        order = torch.randperm(split, generator=generator)
        for start in range(0, split, BATCH):
            batch = order[start : start + BATCH]
            loss = ((target[batch] - model(x[batch], group[batch])) ** 2).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        current = validation()
        if current < best - 1e-9:
            best, stale = current, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if keep_best and stale >= PATIENCE:
                break
    if keep_best:
        model.load_state_dict(best_state)


def fit_network(
    *,
    x: np.ndarray,
    group: np.ndarray,
    own_heating: np.ndarray,
    target: np.ndarray,
    use_wind: bool,
    distil_epochs: int = DISTIL_EPOCHS,
    epochs: int = MAX_EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> Fitted:
    """Train one network per seed: first on `own_heating`, then on `target`.

    Args:
        x: Network inputs, shape (rows, n_inputs), ascending in time.
        group: Group of each row, 0 to 15.
        own_heating: The convex model's own heating term at the rows, in points (the distillation
            target).
        target: The target minus the model's non-heating terms, in points.
        use_wind: Whether the last column of `x` is the wind speed.
        distil_epochs: Epochs spent reproducing `own_heating`.
        epochs: Maximum epochs spent on `target`.
        learning_rate: The Adam learning rate.

    Returns:
        The trained networks.
    """
    torch.set_num_threads(1)
    n = len(x)
    split = int(n * (1.0 - VALIDATION_SHARE))
    n_temperatures = x.shape[1] - int(use_wind)
    x_t = torch.as_tensor(x, dtype=torch.float32)
    group_t = torch.as_tensor(group, dtype=torch.int64)
    scale = float(np.std(target)) or 1.0
    models = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = MonotoneHeating(n_temperatures=n_temperatures, use_wind=use_wind)
        for goal, n_epochs, keep in (
            (own_heating, distil_epochs, False),
            (target, epochs, True),
        ):
            _train(
                model=model,
                tensors=(x_t, group_t, torch.as_tensor(goal / scale, dtype=torch.float32)),
                split=split,
                epochs=n_epochs,
                seed=seed,
                keep_best=keep,
                learning_rate=learning_rate,
            )
        models.append(model.eval())
    return Fitted(models=tuple(models), scale=scale)
