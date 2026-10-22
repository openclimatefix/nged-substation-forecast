"""The small neural components of experiment 00230: the power-curve calibrator and the free network.

The calibrator (B3) adds a bounded correction to the physics core's expected power. The free network
(B6) has no physics: it maps the same inputs straight to 11 monotone quantiles.
"""

from typing import Final

import e00230_common as common
import e00230_core as core_module
import numpy as np
import polars as pl
import torch
from torch import nn

CALIBRATOR_CAP: Final[float] = 0.15
"""The calibrator's output is capped at plus or minus this fraction of capacity."""
CALIBRATOR_HIDDEN: Final[int] = 16
CALIBRATOR_STEPS: Final[int] = 400
CALIBRATOR_LEARNING_RATE: Final[float] = 0.01
CALIBRATOR_WEIGHT_DECAY: Final[float] = 1e-3
FREE_HIDDEN: Final[int] = 64
FREE_STEPS: Final[int] = 600
FREE_LEARNING_RATE: Final[float] = 0.01
FREE_WEIGHT_DECAY: Final[float] = 1e-4
INPUT_COLUMNS: Final[int] = 6
MAX_LOG_RATIO: Final[float] = 1.5


def network_inputs(*, frame: pl.DataFrame, hub_speed: np.ndarray) -> np.ndarray:
    """Return shape (rows, 6) float32 inputs: wind speed, direction, density, and stability proxy.

    Columns: the density-adjusted hub-height wind speed (m/s over 10), the sine and cosine of the
    100 m wind direction, the cube-root air-density factor from the 2 m temperature, the log of the
    100 m to 10 m speed ratio (a lapse-rate stability proxy, clipped), and the 100 m speed over 10.
    """
    speed_10m = np.maximum(frame["wind_speed_10m"].to_numpy(), 0.5)
    speed_100m = np.maximum(frame["wind_speed_100m"].to_numpy(), 0.5)
    density = common.FIXED_SURFACE_PRESSURE_PA / (
        common.DRY_AIR_GAS_CONSTANT * (frame["temperature_c"].to_numpy() + 273.15)
    )
    return np.stack(
        [
            hub_speed / 10.0,
            frame["wind_dir_sin_100m"].to_numpy(),
            frame["wind_dir_cos_100m"].to_numpy(),
            (density / common.STANDARD_AIR_DENSITY_KG_M3) ** (1.0 / 3.0),
            np.clip(np.log(speed_100m / speed_10m), -MAX_LOG_RATIO, MAX_LOG_RATIO),
            speed_100m / 10.0,
        ],
        axis=1,
    ).astype(np.float32)


def hub_speed(*, core: core_module.WindCore, frame: pl.DataFrame) -> np.ndarray:
    """Return the core's density-adjusted hub-height wind speed for every row of `frame`."""
    weather = core_module.WeatherArrays(
        speed_10m=frame["wind_speed_10m"].to_numpy(),
        speed_100m=frame["wind_speed_100m"].to_numpy(),
        temperature_c=frame["temperature_c"].to_numpy(),
    )
    with torch.no_grad():
        tensors = weather.as_tensors()
        return core.hub_height_speed(
            speed_10m=tensors[0], speed_100m=tensors[1], temperature_c=tensors[2]
        ).numpy()


class Calibrator(nn.Module):
    """A two-hidden-layer network whose output is `CALIBRATOR_CAP` times a tanh."""

    def __init__(self) -> None:
        """Start the final layer at zero so the calibrated forecast starts equal to the core's."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(INPUT_COLUMNS, CALIBRATOR_HIDDEN),
            nn.Tanh(),
            nn.Linear(CALIBRATOR_HIDDEN, CALIBRATOR_HIDDEN),
            nn.Tanh(),
            nn.Linear(CALIBRATOR_HIDDEN, 1),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the bounded correction to the expected power fraction, one per row."""
        return CALIBRATOR_CAP * torch.tanh(self.layers(inputs).squeeze(-1))


def fit_calibrator(
    *, inputs: np.ndarray, expected: np.ndarray, target: np.ndarray, seed: int
) -> Calibrator:
    """Fit the calibrator on the L1 loss of the corrected expected power against the target."""
    torch.manual_seed(seed)
    torch.set_num_threads(core_module.THREADS)
    network = Calibrator()
    features = torch.as_tensor(inputs)
    base = torch.as_tensor(expected, dtype=torch.float32)
    goal = torch.as_tensor(target, dtype=torch.float32)
    optimiser = torch.optim.Adam(
        network.parameters(), lr=CALIBRATOR_LEARNING_RATE, weight_decay=CALIBRATOR_WEIGHT_DECAY
    )
    for _ in range(CALIBRATOR_STEPS):
        optimiser.zero_grad()
        (base + network(features) - goal).abs().mean().backward()
        optimiser.step()
    return network.eval()


def calibrated_expected_power(
    *, network: Calibrator, inputs: np.ndarray, expected: np.ndarray
) -> np.ndarray:
    """Return the expected power plus the calibrator's correction, in float32."""
    with torch.no_grad():
        return expected + network(torch.as_tensor(inputs)).numpy()


class FreeQuantileNetwork(nn.Module):
    """A network from the inputs to 11 non-crossing quantiles of the power fraction."""

    def __init__(self) -> None:
        """Build two hidden layers and 11 outputs: a median and 10 positive steps."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(INPUT_COLUMNS, FREE_HIDDEN),
            nn.Tanh(),
            nn.Linear(FREE_HIDDEN, FREE_HIDDEN),
            nn.Tanh(),
            nn.Linear(FREE_HIDDEN, len(common.LEVELS)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return shape (rows, 11) quantiles: a sigmoid median plus cumulative softplus steps."""
        raw = self.layers(inputs)
        median = torch.sigmoid(raw[:, common.MEDIAN_INDEX : common.MEDIAN_INDEX + 1])
        steps = nn.functional.softplus(raw) * 0.1
        below = torch.flip(
            torch.cumsum(torch.flip(steps[:, : common.MEDIAN_INDEX], dims=[1]), dim=1), dims=[1]
        )
        above = torch.cumsum(steps[:, common.MEDIAN_INDEX + 1 :], dim=1)
        return torch.cat([median - below, median, median + above], dim=1)


def fit_free_network(*, inputs: np.ndarray, target: np.ndarray, seed: int) -> FreeQuantileNetwork:
    """Fit the free network on the pinball loss summed over the 11 levels."""
    torch.manual_seed(seed)
    torch.set_num_threads(core_module.THREADS)
    network = FreeQuantileNetwork()
    features = torch.as_tensor(inputs)
    goal = torch.as_tensor(target, dtype=torch.float32)[:, None]
    levels = torch.as_tensor(common.LEVELS, dtype=torch.float32)[None, :]
    optimiser = torch.optim.Adam(
        network.parameters(), lr=FREE_LEARNING_RATE, weight_decay=FREE_WEIGHT_DECAY
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=FREE_STEPS)
    for _ in range(FREE_STEPS):
        optimiser.zero_grad()
        difference = goal - network(features)
        loss = torch.maximum(levels * difference, (levels - 1) * difference).mean()
        loss.backward()
        optimiser.step()
        schedule.step()
    return network.eval()


def free_quantiles(*, network: FreeQuantileNetwork, inputs: np.ndarray) -> np.ndarray:
    """Return shape (rows, 11) float32 quantiles clipped to 0 to 1."""
    with torch.no_grad():
        return np.clip(network(torch.as_tensor(inputs)).numpy(), 0.0, 1.0).astype(np.float32)
