# ruff: noqa: D102, D107
"""The H5 network of experiment 275, copied from experiment 243: the plant plus 11 quantile offsets.

Its last layer is zero-initialised, so before training its forecast equals the plant's forecast plus
the tuning-row offsets. Powers are over capacity.
"""

import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]


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


class QuantileHead(nn.Module):
    """H5: quantiles = clamp(plant + sorted(offsets + 11 network corrections), 0, ceiling)."""

    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.net = mlp(n_in=n_in, n_out=11, zero_last=True)

    def unsorted_offsets(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the offsets before sorting; a decrease along the levels is a quantile crossing."""
        return batch["offsets"] + 0.3 * torch.tanh(self.net(batch["head"]))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        q = batch["mu"][:, None] + torch.sort(self.unsorted_offsets(batch), dim=1).values
        return torch.minimum(torch.clamp(q, min=0.0), batch["ceiling"][:, None])

    def crossed(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return a boolean per row: some quantile is below the one at the next lower level."""
        return (torch.diff(self.unsorted_offsets(batch), dim=1) < 0).any(dim=1)
