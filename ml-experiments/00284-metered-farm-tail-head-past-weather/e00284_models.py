# ruff: noqa: D102, D107
"""The tail head of experiment 284: a correction to the spread-binned plant at six tail levels.

Its last layer is zero-initialised, so before training its tail quantiles equal the spread-binned
plant's. Powers are over capacity.
"""

import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

N_TAIL: int = 6


class TailHead(nn.Module):
    """quantiles = clamp(plant + sorted(binned offsets + 0.3 * tanh(network)), 0, ceiling)."""

    def __init__(self, n_in: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, N_TAIL),
        )
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def unsorted_offsets(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["offsets"] + 0.3 * torch.tanh(self.net(batch["head"]))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        q = batch["mu"][:, None] + torch.sort(self.unsorted_offsets(batch), dim=1).values
        return torch.minimum(torch.clamp(q, min=0.0), batch["ceiling"][:, None])

    def crossing_penalty(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the mean size of a decrease along the levels before sorting."""
        return torch.relu(-torch.diff(self.unsorted_offsets(batch), dim=1)).mean()

    def crossed(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return (torch.diff(self.unsorted_offsets(batch), dim=1) < 0).any(dim=1)
