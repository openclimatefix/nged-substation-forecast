"""The pooled correction head: an MLP over recent errors and calendar, trained with pinball loss.

It emits a median correction `c` and a log-scale `s`; the 11 quantiles are
`mean + c + exp(s) * (arm D quantile - arm D median)`, so they cannot cross.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from e00200_data import N_RECENT
from torch import nn

LEVELS: Final[torch.Tensor] = torch.tensor(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
ERROR_SCALE: Final[float] = 5.0
HIDDEN: Final[int] = 128
EMBEDDING: Final[int] = 4
MAX_EPOCHS: Final[int] = 40
PATIENCE: Final[int] = 5
BATCH: Final[int] = 1024
LEARNING_RATE: Final[float] = 2e-3
WEIGHT_DECAY: Final[float] = 1e-2
LOG_SCALE_LIMIT: Final[float] = 0.7
N_FEATURES: Final[int] = N_RECENT + 3 + 1 + 1 + 2 + 7 + 1 + 2


def design(*, f: dict[str, np.ndarray]) -> np.ndarray:
    """Stack the numeric inputs into one matrix."""
    angle = 2 * np.pi * f["hod"] / 48.0
    dow = np.eye(7, dtype=np.float32)[f["dow"]]
    return np.column_stack(
        [
            f["recent"] / ERROR_SCALE,
            f["recent_ok"],
            f["profile"] / ERROR_SCALE,
            f["profile_n"],
            f["weekly"] / ERROR_SCALE,
            f["weekly_n"],
            f["day_recent"] / ERROR_SCALE,
            (f["lead"] - 30.0) / 10.0,
            np.sin(angle),
            np.cos(angle),
            dow,
            f["mean"] / 100.0,
        ]
    ).astype(np.float32)


class Head(nn.Module):
    """Two hidden layers, a site embedding, a median correction and a log-scale."""

    def __init__(self, *, n_sites: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_sites, EMBEDDING)
        self.body = nn.Sequential(
            nn.Linear(N_FEATURES + EMBEDDING, HIDDEN),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, 2),
        )
        self.skip = nn.Linear(N_FEATURES, 1)

    def forward(self, *, x: torch.Tensor, site: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.body(torch.cat([x, self.embedding(site)], dim=1))
        correction = (out[:, 0] + self.skip(x)[:, 0]) * ERROR_SCALE
        log_scale = LOG_SCALE_LIMIT * torch.tanh(out[:, 1])
        return correction, log_scale


def pinball(
    *, y: torch.Tensor, correction: torch.Tensor, log_scale: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Mean pinball loss over the 11 quantiles."""
    pred = correction[:, None] + torch.exp(log_scale)[:, None] * offsets
    diff = y[:, None] - pred
    levels = LEVELS.to(y.device)
    return torch.maximum(levels * diff, (levels - 1) * diff).mean()


@dataclass
class Data:
    """Tensors of one set of rows."""

    x: np.ndarray
    site: np.ndarray
    y: np.ndarray
    offsets: np.ndarray


def fit(*, train: Data, valid: Data, n_sites: int, seed: int, device: str) -> Head:
    """Train with early stopping on `valid` (a purged time block), return the best head."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Head(n_sites=n_sites).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def tensors(d: Data) -> tuple[torch.Tensor, ...]:
        return (
            torch.from_numpy(d.x).to(device),
            torch.from_numpy(d.site).to(device),
            torch.from_numpy(d.y).to(device),
            torch.from_numpy(d.offsets).to(device),
        )

    tx, ts, ty, to = tensors(train)
    vx, vs, vy, vo = tensors(valid)
    best, best_state, stale = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        net.train()
        order = torch.from_numpy(rng.permutation(len(ty))).to(device)
        for start in range(0, len(order), BATCH):
            b = order[start : start + BATCH]
            c, s = net(x=tx[b], site=ts[b])
            loss = pinball(y=ty[b], correction=c, log_scale=s, offsets=to[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            c, s = net(x=vx, site=vs)
            val = float(pinball(y=vy, correction=c, log_scale=s, offsets=vo))
        if val < best - 1e-5:
            best, stale = val, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    assert best_state is not None
    net.load_state_dict(best_state)
    return net.eval()


def predict(
    *, net: Head, x: np.ndarray, site: np.ndarray, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """Median correction and log-scale for rows."""
    with torch.no_grad():
        c, s = net(x=torch.from_numpy(x).to(device), site=torch.from_numpy(site).to(device))
    return c.cpu().numpy(), s.cpu().numpy()
