# ruff: noqa: E501, D102, D107
"""The adapter models of experiment 272: additive curves and a small neural network on the plant.

Both adapters output g(x) and h(x). The plant (the fleet-shape plant of experiment 243, with the
per-farm scale and ceiling frozen) receives the ensemble-mean beam and diffuse irradiance times
exp(g) and a cell temperature raised by h. Both start at zero, so an untrained adapter returns the
plant. The 11 quantiles are the adapted plant output plus a learned table of 11 offsets per bin of
the unadapted plant forecast, sorted. Powers are over capacity.
"""

import math

import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

TILT_RAD = math.radians(35.0)
PANEL_AZIMUTH_RAD = math.radians(180.0)
GROUND_ALBEDO = 0.2
HEAT_PER_KELVIN = 0.004
CELL_RISE_K = 25.0
LOG_FACTOR_BOUND = 0.4
"""The irradiance factor exp(g) stays between exp(-0.4) = 0.67 and exp(0.4) = 1.49."""
TEMPERATURE_BOUND_K = 4.0
N_KNOTS = 8
HIDDEN = 32


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


class AdditiveCurves(nn.Module):
    """One piecewise-linear curve per variable for each of g and h, summed; zero at initialisation."""

    def __init__(self, knots: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("knots", knots)  # (n_vars, N_KNOTS), increasing
        self.values = nn.Parameter(torch.zeros(knots.shape[0], knots.shape[1], 2))

    def curve(self, x: torch.Tensor, j: int) -> torch.Tensor:
        """Return the (rows, 2) value of variable j's curves at x[:, j]."""
        knots = self.knots[j]
        xj = torch.minimum(torch.maximum(x[:, j], knots[0]), knots[-1])
        hi = torch.bucketize(xj, knots[1:-1]) + 1
        lo = hi - 1
        w = (xj - knots[lo]) / (knots[hi] - knots[lo]).clamp(min=1e-6)
        return (1 - w)[:, None] * self.values[j, lo] + w[:, None] * self.values[j, hi]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum(self.curve(x, j) for j in range(x.shape[1]))

    def penalty(self) -> torch.Tensor:
        """Return the squared second difference of every curve."""
        second = self.values[:, 2:] - 2 * self.values[:, 1:-1] + self.values[:, :-2]
        return (second**2).mean()

    def curve_values(self, x: torch.Tensor, j: int) -> torch.Tensor:
        return self.curve(x, j)


class NeuralCorrection(nn.Module):
    """Two hidden layers of 32 units; the last layer is zeroed so the plant is the start."""

    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, 2),
        )
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def penalty(self) -> torch.Tensor:
        return torch.zeros(())


class PlantAdapter(nn.Module):
    """The adapted plant: corrected inputs through the frozen plant, then the offset table."""

    def __init__(self, *, correction: nn.Module, offsets: torch.Tensor) -> None:
        super().__init__()
        self.correction = correction
        self.offsets = nn.Parameter(offsets.clone())  # (bins, 11)
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.log_scale.requires_grad_(False)

    def gh(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.correction(batch["x"])
        return LOG_FACTOR_BOUND * torch.tanh(out[:, 0]), TEMPERATURE_BOUND_K * torch.tanh(out[:, 1])

    def power(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        g, h = self.gh(batch)
        factor = torch.exp(g)
        return differentiable_plant(
            dni=batch["dni"] * factor,
            dhi=batch["dhi"] * factor,
            temperature=batch["temperature"] + h,
            zenith_rad=batch["zenith_rad"],
            sun_azimuth_rad=batch["sun_azimuth_rad"],
            scale=batch["scale"] * torch.exp(self.log_scale),
            ceiling=batch["ceiling"],
        )

    def quantiles(
        self, batch: dict[str, torch.Tensor], power: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (sorted quantiles, share of rows whose unsorted quantiles cross)."""
        centre = self.power(batch) if power is None else power
        raw = centre[:, None] + self.offsets[batch["bin"]]
        crossed = (raw[:, 1:] < raw[:, :-1]).any(dim=1).float()
        q = torch.minimum(
            torch.clamp(torch.sort(raw, dim=1).values, min=0.0), batch["ceiling"][:, None]
        )
        return q, crossed

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.quantiles(batch)[0]


def mlp(*, n_in: int, n_out: int, zero_last: bool) -> nn.Sequential:
    """Return a two-hidden-layer perceptron; the last layer is zeroed for a head on the plant."""
    net = nn.Sequential(
        nn.Linear(n_in, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, n_out)
    )
    if zero_last:
        last = net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    return net


class AdapterWithHead(nn.Module):
    """A6: the trained adapter (frozen) plus experiment 243's H5-form head, 11 corrections."""

    def __init__(self, *, adapter: PlantAdapter, n_head_in: int) -> None:
        super().__init__()
        self.adapter = adapter
        for p in self.adapter.parameters():
            p.requires_grad_(False)
        self.net = mlp(n_in=n_head_in, n_out=11, zero_last=True)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.quantiles(batch)[0]

    def quantiles(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        power = self.adapter.power(batch)
        head = torch.cat([power[:, None], batch["head"]], dim=1)
        delta = 0.3 * torch.tanh(self.net(head))
        raw = power[:, None] + self.adapter.offsets[batch["bin"]] + delta
        crossed = (raw[:, 1:] < raw[:, :-1]).any(dim=1).float()
        q = torch.minimum(
            torch.clamp(torch.sort(raw, dim=1).values, min=0.0), batch["ceiling"][:, None]
        )
        return q, crossed


class PureNetwork(nn.Module):
    """The blind arm: raw inputs and the full variable list to 11 increasing quantiles, no plant."""

    def __init__(self, n_in: int) -> None:
        super().__init__()
        self.net = mlp(n_in=n_in, n_out=11, zero_last=False)

    def quantiles(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.net(batch["blind"])
        steps = nn.functional.softplus(z[:, 1:]) * 0.5
        u = z[:, :1] + torch.cumsum(torch.cat([torch.zeros_like(z[:, :1]), steps], dim=1), dim=1)
        q = batch["ceiling"][:, None] * torch.sigmoid(u - 3.0)
        return q, torch.zeros(len(q), device=q.device)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.quantiles(batch)[0]
