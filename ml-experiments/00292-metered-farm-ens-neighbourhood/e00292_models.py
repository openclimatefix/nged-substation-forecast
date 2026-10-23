# ruff: noqa: E501, D102, D107
"""The neural heads of experiment 292: H5 (`QuantileHead`) and H5 with a convolution over the 19 cells.

Every hybrid starts as the plant: its last layer is zero-initialised, so before training its
forecast equals the plant's forecast plus the tuning-month offsets. Powers are over capacity.
"""

import torch
from torch import nn


def mlp(*, n_in: int, n_out: int, hidden: int = 64, zero_last: bool = False) -> nn.Sequential:
    """Return a two-hidden-layer perceptron; the last layer is zeroed for a head on the plant."""
    net = nn.Sequential(
        nn.Linear(n_in, hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, n_out),
    )
    if zero_last:
        last = net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    return net


def finish(*, centre: torch.Tensor, offsets: torch.Tensor, ceiling: torch.Tensor) -> torch.Tensor:
    """Return the quantiles: centre plus sorted offsets, clamped to [0, ceiling]."""
    q = centre[:, None] + torch.sort(offsets, dim=1).values
    return torch.minimum(torch.clamp(q, min=0.0), ceiling[:, None])


class QuantileHead(nn.Module):
    """H5: quantiles = clamp(plant + sorted(offsets + 11 network corrections))."""

    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.net = mlp(n_in=n_in, n_out=11, zero_last=True)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        delta = 0.3 * torch.tanh(self.net(batch["head"]))
        return finish(
            centre=batch["mu"], offsets=batch["offsets"] + delta, ceiling=batch["ceiling"]
        )


class ConvQuantileHead(nn.Module):
    """N2: H5 plus a small convolution over the 5 by 5 layout of the farm's 19 cells.

    The image channels are member-mean irradiance, member spread, presence, and the cell's member
    mean minus the own cell's. Two 3 by 3 convolutions and a linear layer give 16 features that join
    the tabular inputs before the zero-initialised 11-offset output layer.
    """

    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(8 * 25, 16),
            nn.GELU(),
        )
        self.net = mlp(n_in=n_in + 16, n_out=11, zero_last=True)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        z = torch.cat([batch["head"], self.conv(batch["image"])], dim=1)
        delta = 0.3 * torch.tanh(self.net(z))
        return finish(
            centre=batch["mu"], offsets=batch["offsets"] + delta, ceiling=batch["ceiling"]
        )
