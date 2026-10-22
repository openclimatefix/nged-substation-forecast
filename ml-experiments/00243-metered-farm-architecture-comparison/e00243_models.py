# ruff: noqa: E501, D102, D107
"""The neural architectures of experiment 243, all outputting 11 non-crossing, non-negative quantiles.

Every hybrid starts as the plant: its last layer is zero-initialised, so before training its
forecast equals the plant's forecast plus the tuning-month offsets. Powers are over capacity.
"""

import math

import torch
from torch import nn

TILT_RAD = math.radians(35.0)
PANEL_AZIMUTH_RAD = math.radians(180.0)
GROUND_ALBEDO = 0.2
HEAT_PER_KELVIN = 0.004
CELL_RISE_K = 25.0


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


class PureNetwork(nn.Module):
    """A network with no physics: raw inputs to 11 increasing quantiles in (0, ceiling)."""

    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.net = mlp(n_in=n_in, n_out=11)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.net(batch["raw"])
        u = z[:, :1] + torch.cumsum(
            torch.cat([torch.zeros_like(z[:, :1]), nn.functional.softplus(z[:, 1:]) * 0.5], dim=1),
            dim=1,
        )
        return batch["ceiling"][:, None] * torch.sigmoid(u - 3.0)


class ScaleHead(nn.Module):
    """H1: quantiles = clamp(scale * plant + offsets), the scale in (0.5, 1.5) from a network."""

    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.net = mlp(n_in=n_in, n_out=1, zero_last=True)

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 + torch.sigmoid(self.net(x)[:, 0])

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        s = self.scale(batch["head"])
        return finish(centre=s * batch["mu"], offsets=batch["offsets"], ceiling=batch["ceiling"])


class ResidualHead(nn.Module):
    """H2 and H6: quantiles = clamp(plant + correction + offsets), the correction in +-0.3 capacity."""

    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.net = mlp(n_in=n_in, n_out=1, zero_last=True)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        delta = 0.3 * torch.tanh(self.net(batch["head"])[:, 0])
        return finish(
            centre=batch["mu"] + delta, offsets=batch["offsets"], ceiling=batch["ceiling"]
        )


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


def differentiable_plant(
    *,
    dni: torch.Tensor,
    dhi: torch.Tensor,
    temperature: torch.Tensor,
    zenith_rad: torch.Tensor,
    sun_azimuth_rad: torch.Tensor,
    scale: torch.Tensor,
    ceiling: torch.Tensor,
) -> torch.Tensor:
    """Return the capped fleet-shape plant power in torch, from beam, diffuse, and temperature."""
    cos_incidence = torch.cos(zenith_rad) * math.cos(TILT_RAD) + torch.sin(zenith_rad) * math.sin(
        TILT_RAD
    ) * torch.cos(sun_azimuth_rad - PANEL_AZIMUTH_RAD)
    ghi = dni * torch.cos(zenith_rad) + dhi
    poa = (
        dni * torch.clamp(cos_incidence, min=0.0)
        + dhi * (1 + math.cos(TILT_RAD)) / 2
        + ghi * GROUND_ALBEDO * (1 - math.cos(TILT_RAD)) / 2
    )
    unit = torch.clamp(poa, min=0.0) / 1000.0
    cell = temperature + CELL_RISE_K * unit
    derating = torch.clamp(1.0 - HEAT_PER_KELVIN * (cell - 25.0), min=0.0)
    return torch.minimum(scale * unit * derating, ceiling)


class Adapter(nn.Module):
    """H3 (and H4 with `with_scale_head`): a network adjusts beam, diffuse, and temperature, then the plant."""

    def __init__(self, n_in: int, *, with_scale_head: bool, n_head_in: int) -> None:
        super().__init__()
        self.net = mlp(n_in=n_in, n_out=3, zero_last=True)
        self.head = ScaleHead(n_head_in) if with_scale_head else None

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        a = self.net(batch["raw"])
        beam_factor = 0.5 + torch.sigmoid(a[:, 0])
        diffuse_factor = 0.5 + torch.sigmoid(a[:, 1])
        temperature = batch["temperature"] + 4.0 * torch.tanh(a[:, 2])
        power = differentiable_plant(
            dni=batch["dni"] * beam_factor,
            dhi=batch["dhi"] * diffuse_factor,
            temperature=temperature,
            zenith_rad=batch["zenith_rad"],
            sun_azimuth_rad=batch["sun_azimuth_rad"],
            scale=batch["scale"],
            ceiling=batch["ceiling"],
        )
        if self.head is not None:
            power = torch.minimum(self.head.scale(batch["head"]) * power, batch["ceiling"])
        return finish(centre=power, offsets=batch["offsets"], ceiling=batch["ceiling"])
