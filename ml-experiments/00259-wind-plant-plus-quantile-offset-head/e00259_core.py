"""The wind-farm physics core, fitted by gradient descent, and its residual-quantile layer.

One farm's core turns the ERA5 or ENS 10 m and 100 m wind speeds and the 2 m temperature into an
expected power as a fraction of capacity: a shear exponent from the two speeds takes the 100 m
speed to a fitted hub height, an air-density factor adjusts it, a smooth monotone power curve with a
logistic cut-out and a Gaussian blur (the 25 km weather cell's sub-grid wind) gives the active
power, and an optional idle layer turns active power into apparent power for the MVA series. The
spread of the observation about the expected power comes from `ResidualQuantiles`.
"""

import os
from dataclasses import dataclass
from typing import Final, Self

import e00259_common as common
import numpy as np
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

GAUSS_HERMITE_POINTS: Final[int] = 7
HUB_HEIGHT_PRIOR_M: Final[float] = 65.0
HUB_HEIGHT_PRIOR_LOG_STANDARD_DEVIATION: Final[float] = 0.2
HUB_HEIGHT_RANGE_M: Final[tuple[float, float]] = (30.0, 140.0)
DEFAULT_SHEAR_EXPONENT: Final[float] = 0.14
SHEAR_RANGE: Final[tuple[float, float]] = (0.0, 0.5)
MIN_SHEAR_SPEED_M_S: Final[float] = 0.5
FIT_STEPS: Final[int] = 800
LEARNING_RATE: Final[float] = 0.05
RESIDUAL_BINS: Final[int] = 14
IDLE_FRACTION_FLOOR: Final[float] = 0.0
THREADS: Final[int] = int(os.environ.get("E00259_TORCH_THREADS", "1"))


def _sigmoid_range(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Return `raw` squashed into the open interval from `low` to `high`."""
    return low + (high - low) * torch.sigmoid(raw)


def _inverse_sigmoid_range(value: float, low: float, high: float) -> float:
    """Return the raw value whose `_sigmoid_range` is `value`."""
    unit = (value - low) / (high - low)
    return float(np.log(unit / (1 - unit)))


def _inverse_softplus(value: float) -> float:
    """Return the raw value whose softplus is `value`."""
    return float(np.log(np.expm1(value)))


@dataclass(frozen=True)
class WeatherArrays:
    """Weather columns of equal length, as float32 NumPy arrays.

    Attributes:
        speed_10m: Wind speed at 10 m, m/s.
        speed_100m: Wind speed at 100 m, m/s.
        temperature_c: 2 m air temperature, degrees Celsius.
    """

    speed_10m: np.ndarray
    speed_100m: np.ndarray
    temperature_c: np.ndarray

    def as_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the three columns as torch tensors."""
        return (
            torch.as_tensor(self.speed_10m, dtype=torch.float32),
            torch.as_tensor(self.speed_100m, dtype=torch.float32),
            torch.as_tensor(self.temperature_c, dtype=torch.float32),
        )


class WindCore(nn.Module):
    """One farm's physics core: weather in, expected power as a fraction of capacity out."""

    blur_nodes: torch.Tensor
    blur_weights: torch.Tensor

    def __init__(self, *, with_idle_layer: bool, idle_floor: float) -> None:
        """Set the starting parameters.

        Args:
            with_idle_layer: Whether the apparent-power idle layer is part of the model.
            idle_floor: The starting idle floor, a fraction of capacity (the median power observed
                at ERA5 100 m wind below `IDLE_WIND_M_S`).
        """
        super().__init__()
        self.with_idle_layer = with_idle_layer
        self.raw_hub_height = nn.Parameter(
            torch.tensor(
                _inverse_sigmoid_range(HUB_HEIGHT_PRIOR_M, *HUB_HEIGHT_RANGE_M), dtype=torch.float32
            )
        )
        self.raw_cut_in = nn.Parameter(torch.tensor(_inverse_sigmoid_range(3.0, 1.5, 6.0)))
        self.raw_ramp_width = nn.Parameter(torch.tensor(_inverse_sigmoid_range(8.0, 3.0, 15.0)))
        self.raw_shape = nn.Parameter(torch.tensor(_inverse_sigmoid_range(2.5, 1.0, 5.0)))
        self.raw_cut_out = nn.Parameter(torch.tensor(_inverse_sigmoid_range(27.0, 20.0, 35.0)))
        self.raw_blur = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self.raw_scale = nn.Parameter(torch.tensor(_inverse_softplus(0.95)))
        self.raw_idle_floor = nn.Parameter(torch.tensor(_inverse_softplus(max(idle_floor, 1e-3))))
        self.raw_dip = nn.Parameter(torch.tensor(_inverse_softplus(0.005)))
        nodes, weights = np.polynomial.hermite.hermgauss(GAUSS_HERMITE_POINTS)
        self.register_buffer(
            "blur_nodes", torch.as_tensor(np.sqrt(2.0) * nodes, dtype=torch.float32)
        )
        self.register_buffer(
            "blur_weights", torch.as_tensor(weights / np.sqrt(np.pi), dtype=torch.float32)
        )

    @property
    def hub_height_m(self) -> torch.Tensor:
        """Return the fitted hub height in metres."""
        return _sigmoid_range(self.raw_hub_height, *HUB_HEIGHT_RANGE_M)

    @property
    def cut_in_m_s(self) -> torch.Tensor:
        """Return the fitted cut-in speed in m/s."""
        return _sigmoid_range(self.raw_cut_in, 1.5, 6.0)

    @property
    def rated_m_s(self) -> torch.Tensor:
        """Return the fitted speed at which the curve reaches rated power, in m/s."""
        return self.cut_in_m_s + _sigmoid_range(self.raw_ramp_width, 3.0, 15.0)

    def hub_height_speed(
        self, *, speed_10m: torch.Tensor, speed_100m: torch.Tensor, temperature_c: torch.Tensor
    ) -> torch.Tensor:
        """Return the density-adjusted wind speed at the fitted hub height, in m/s."""
        usable = (speed_10m > MIN_SHEAR_SPEED_M_S) & (speed_100m > MIN_SHEAR_SPEED_M_S)
        safe_10m = torch.where(usable, speed_10m, torch.ones_like(speed_10m))
        safe_100m = torch.where(usable, speed_100m, torch.ones_like(speed_100m))
        exponent = torch.log(safe_100m / safe_10m) / np.log(10.0)
        exponent = torch.where(usable, exponent, torch.full_like(exponent, DEFAULT_SHEAR_EXPONENT))
        exponent = exponent.clamp(*SHEAR_RANGE)
        speed = speed_100m * (self.hub_height_m / 100.0) ** exponent
        density = common.FIXED_SURFACE_PRESSURE_PA / (
            common.DRY_AIR_GAS_CONSTANT * (temperature_c + 273.15)
        )
        return speed * (density / common.STANDARD_AIR_DENSITY_KG_M3) ** (1.0 / 3.0)

    def active_power(self, *, speed: torch.Tensor) -> torch.Tensor:
        """Return the blurred monotone power curve at `speed`, a fraction of capacity."""
        scale = nn.functional.softplus(self.raw_scale)
        blur = nn.functional.softplus(self.raw_blur)
        shifted = (speed[:, None] + blur * self.blur_nodes[None, :]).clamp(min=0.0)
        ramp = ((shifted - self.cut_in_m_s) / (self.rated_m_s - self.cut_in_m_s)).clamp(0.0, 1.0)
        shape = _sigmoid_range(self.raw_shape, 1.0, 5.0)
        cut_out = torch.sigmoid((_sigmoid_range(self.raw_cut_out, 20.0, 35.0) - shifted) / 1.0)
        return scale * ((ramp**shape) * cut_out * self.blur_weights[None, :]).sum(dim=1)

    def forward(self, *, weather: WeatherArrays) -> torch.Tensor:
        """Return the expected power as a fraction of capacity for each row of `weather`."""
        speed_10m, speed_100m, temperature_c = weather.as_tensors()
        speed = self.hub_height_speed(
            speed_10m=speed_10m, speed_100m=speed_100m, temperature_c=temperature_c
        )
        active = self.active_power(speed=speed)
        if not self.with_idle_layer:
            return active
        floor = nn.functional.softplus(self.raw_idle_floor)
        dip = nn.functional.softplus(self.raw_dip)
        position = speed / self.cut_in_m_s
        return torch.sqrt(active**2 + floor**2) - dip * position * torch.exp(1.0 - position)


def idle_floor_start(*, speed_100m: np.ndarray, target: np.ndarray) -> float:
    """Return the median observed fraction at 100 m wind below `common.IDLE_WIND_M_S`."""
    idle = speed_100m < common.IDLE_WIND_M_S
    return float(np.median(target[idle])) if idle.any() else IDLE_FRACTION_FLOOR


def fit_core(
    *, weather: WeatherArrays, target: np.ndarray, with_idle_layer: bool, seed: int
) -> WindCore:
    """Fit one farm's core by full-batch Adam on the L1 loss of the observed fraction.

    A log-normal prior (median `HUB_HEIGHT_PRIOR_M`) enters the loss with a weight of one over the
    number of rows, so it is negligible: the fitted hub height is an effective parameter that
    absorbs ERA5's wind bias at the farm, not a physical hub height.

    Args:
        weather: The training rows' weather.
        target: The observed power as a fraction of capacity, one per row.
        with_idle_layer: Whether to fit the idle layer.
        seed: The seed of the (here deterministic) starting parameters.

    Returns:
        The fitted core, in evaluation mode.
    """
    torch.manual_seed(seed)
    torch.set_num_threads(THREADS)
    core = WindCore(
        with_idle_layer=with_idle_layer,
        idle_floor=idle_floor_start(speed_100m=weather.speed_100m, target=target),
    )
    goal = torch.as_tensor(target, dtype=torch.float32)
    optimiser = torch.optim.Adam(core.parameters(), lr=LEARNING_RATE)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=FIT_STEPS)
    log_prior = np.log(HUB_HEIGHT_PRIOR_M)
    for _ in range(FIT_STEPS):
        optimiser.zero_grad()
        loss = (core(weather=weather) - goal).abs().mean()
        penalty = (
            (torch.log(core.hub_height_m) - log_prior) ** 2
            / (2 * HUB_HEIGHT_PRIOR_LOG_STANDARD_DEVIATION**2)
            / goal.numel()
        )
        (loss + penalty).backward()
        optimiser.step()
        schedule.step()
    return core.eval()


def predict_expected_power(
    *, core: WindCore, weather: WeatherArrays, chunk: int = 500_000
) -> np.ndarray:
    """Return the core's expected power fraction for every row, in float32."""
    with torch.no_grad():
        pieces = []
        for start in range(0, weather.speed_100m.size, chunk):
            part = WeatherArrays(
                speed_10m=weather.speed_10m[start : start + chunk],
                speed_100m=weather.speed_100m[start : start + chunk],
                temperature_c=weather.temperature_c[start : start + chunk],
            )
            pieces.append(core(weather=part).numpy())
    return np.concatenate(pieces)


@dataclass(frozen=True)
class ResidualQuantiles:
    """Empirical quantiles of observed minus expected power, in bins of the expected power.

    Attributes:
        centres: Mean expected power of each bin, ascending, shape (bins,).
        offsets: Residual quantile at each of `common.LEVELS` per bin, shape (bins, 11).
    """

    centres: np.ndarray
    offsets: np.ndarray

    @classmethod
    def fit(cls, *, expected: np.ndarray, observed: np.ndarray) -> Self:
        """Return the residual quantiles of `observed - expected` per bin of `expected`."""
        order = np.argsort(expected, kind="stable")
        centres, offsets = [], []
        for chunk in np.array_split(order, RESIDUAL_BINS):
            centres.append(float(expected[chunk].mean()))
            offsets.append(np.quantile(observed[chunk] - expected[chunk], common.LEVELS))
        return cls(centres=np.array(centres), offsets=np.array(offsets))

    def quantiles(self, *, expected: np.ndarray) -> np.ndarray:
        """Return shape (rows, 11) quantiles: expected power plus interpolated offsets, clipped.

        Args:
            expected: The expected power fraction per row.

        Returns:
            Sorted quantiles of the observed fraction, clipped to 0 to 1.
        """
        columns = [
            np.interp(expected, self.centres, self.offsets[:, index])
            for index in range(len(common.LEVELS))
        ]
        out = expected[:, None] + np.stack(columns, axis=1)
        return np.sort(np.clip(out, 0.0, 1.0), axis=1).astype(np.float32)
