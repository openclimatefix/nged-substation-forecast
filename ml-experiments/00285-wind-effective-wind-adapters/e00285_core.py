"""The wind-farm physics core and its variants for experiment 00285.

Copied from experiment 00276's `e00276_core.py` (itself copied from experiment 00230), with
effective-wind adapters added (`CoreOptions.adapter`): a shear term, direction sectors, and small
neural adapters that scale and offset the hub-height wind speed before the power curve. The
options of experiment 00276 remain: no
shear (`shear=False`), a free air-density exponent, direction-dependent speed terms, a per-group
scale (per year or per month), a monotone neural power curve, and a core fitted directly on pinball
loss with a hub-speed-dependent spread (`fit_core_pinball`).
"""

import os
from dataclasses import dataclass
from typing import Final, Self

import e00285_common as common
import numpy as np
import torch
from torch import nn

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
THREADS: Final[int] = int(os.environ.get("E00285_TORCH_THREADS", "1"))
NET_WIDTH: Final[int] = 12
NET_STEPS: Final[int] = 1_500
NET_LEARNING_RATE: Final[float] = 0.02
PINBALL_STEPS: Final[int] = 600
PINBALL_LEARNING_RATE: Final[float] = 0.01
SPREAD_KNOTS: Final[np.ndarray] = np.linspace(0.0, 30.0, 16)
"""Hub speeds, m/s, at which the pinball core's spread offsets are free."""
SCALE_FIT_STEPS: Final[int] = 300
ADAPTER_STEPS: Final[int] = 1_200
ADAPTER_LEARNING_RATE: Final[float] = 0.01
ADAPTER_WEIGHT_DECAY: Final[float] = 1e-4
ADAPTER_WIDTH: Final[int] = 32
SCALE_RANGE: Final[tuple[float, float]] = (0.8, 1.25)
OFFSET_LIMIT_M_S: Final[float] = 1.0
SECTOR_COUNT: Final[int] = 8
ADAPTERS: Final[tuple[str, ...]] = (
    "none",
    "shear_k",
    "sectors",
    "shear_k_sectors",
    "net",
    "net_no_hour",
    "blind",
)
NET_INPUT_COUNT: Final[dict[str, int]] = {"net": 7, "net_no_hour": 5, "blind": 1}


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
class CoreOptions:
    """Which terms the core has beyond experiment 230's.

    Attributes:
        shear: Whether the shear exponent comes from the 10 m and 100 m speeds (else a fixed 0.14).
        free_density_exponent: Whether the air-density exponent is fitted (else the fixed 1/3).
        direction_terms: Whether the speed is scaled by two harmonics of the 100 m direction.
        group_count: Number of groups with their own multiplicative scale (0 for none).
        monotone_net: Whether a monotone neural curve replaces the parametric power curve.
        adapter: The effective-wind adapter (one of `ADAPTERS`): `shear_k` (extra speed ratio to a
            fitted power), `sectors` (a scale per direction sector), `shear_k_sectors` (both),
            `net` (a small network on direction, speed ratio, temperature, and hour), `net_no_hour`
            (the same without the hour), or `blind` (a network on the 100 m speed alone).
    """

    shear: bool = True
    free_density_exponent: bool = False
    direction_terms: bool = False
    group_count: int = 0
    monotone_net: bool = False
    adapter: str = "none"


DEFAULT_OPTIONS: Final[CoreOptions] = CoreOptions()


@dataclass(frozen=True)
class WeatherArrays:
    """Weather columns of equal length, as NumPy arrays.

    Attributes:
        speed_10m: Wind speed at 10 m, m/s.
        speed_100m: Wind speed at 100 m, m/s.
        temperature_c: 2 m air temperature, degrees Celsius.
        direction_sin: Sine of the 100 m wind direction (needed by `direction_terms`).
        direction_cos: Cosine of the 100 m wind direction.
        group: Integer group of each row (needed by `group_count`).
        hour: Hour of day (0 to 24) of each row (needed by the `net` adapter).
    """

    speed_10m: np.ndarray
    speed_100m: np.ndarray
    temperature_c: np.ndarray
    direction_sin: np.ndarray | None = None
    direction_cos: np.ndarray | None = None
    group: np.ndarray | None = None
    hour: np.ndarray | None = None

    def take(self, index: slice) -> Self:
        """Return the rows in `index`."""

        def cut(column: np.ndarray | None) -> np.ndarray | None:
            return None if column is None else column[index]

        return type(self)(
            speed_10m=self.speed_10m[index],
            speed_100m=self.speed_100m[index],
            temperature_c=self.temperature_c[index],
            direction_sin=cut(self.direction_sin),
            direction_cos=cut(self.direction_cos),
            group=cut(self.group),
            hour=cut(self.hour),
        )


class WindCore(nn.Module):
    """One farm's physics core: weather in, expected power as a fraction of capacity out."""

    blur_nodes: torch.Tensor
    blur_weights: torch.Tensor

    def __init__(
        self, *, with_idle_layer: bool, idle_floor: float, options: CoreOptions = DEFAULT_OPTIONS
    ) -> None:
        """Set the starting parameters.

        Args:
            with_idle_layer: Whether the apparent-power idle layer is part of the model.
            idle_floor: The starting idle floor, a fraction of capacity.
            options: Which extra terms the core has.
        """
        super().__init__()
        self.with_idle_layer = with_idle_layer
        self.options = options
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
        if options.free_density_exponent:
            self.raw_density_exponent = nn.Parameter(torch.tensor(_inverse_softplus(1.0 / 3.0)))
        if options.direction_terms:
            self.direction_coefficients = nn.Parameter(torch.zeros(4))
        if options.group_count:
            self.group_log_scale = nn.Parameter(torch.zeros(options.group_count))
        if options.adapter in ("shear_k", "shear_k_sectors"):
            self.shear_adapter_exponent = nn.Parameter(torch.zeros(()))
        if options.adapter in ("sectors", "shear_k_sectors"):
            self.sector_log_scale = nn.Parameter(torch.zeros(SECTOR_COUNT))
        if options.adapter in NET_INPUT_COUNT:
            self.adapter_net = nn.Sequential(
                nn.Linear(NET_INPUT_COUNT[options.adapter], ADAPTER_WIDTH),
                nn.Tanh(),
                nn.Linear(ADAPTER_WIDTH, ADAPTER_WIDTH),
                nn.Tanh(),
                nn.Linear(ADAPTER_WIDTH, 2),
            )
            last = self.adapter_net[-1]
            if not isinstance(last, nn.Linear):
                raise TypeError
            nn.init.zeros_(last.weight)
            with torch.no_grad():
                last.bias.copy_(torch.tensor([_inverse_sigmoid_range(1.0, *SCALE_RANGE), 0.0]))
        if options.monotone_net:
            generator = torch.Generator().manual_seed(0)
            self.net_raw_1 = nn.Parameter(
                torch.randn(1, NET_WIDTH, generator=generator) * 0.5 - 1.0
            )
            self.net_raw_2 = nn.Parameter(
                torch.randn(NET_WIDTH, NET_WIDTH, generator=generator) * 0.3 - 2.0
            )
            self.net_raw_3 = nn.Parameter(
                torch.randn(NET_WIDTH, 1, generator=generator) * 0.3 - 1.0
            )
            self.net_bias_1 = nn.Parameter(torch.linspace(-2.0, 0.0, NET_WIDTH))
            self.net_bias_2 = nn.Parameter(torch.zeros(NET_WIDTH))
            self.net_bias_3 = nn.Parameter(torch.tensor(-1.0))

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

    def hub_height_speed(self, *, weather: WeatherArrays) -> torch.Tensor:
        """Return the adjusted wind speed at the fitted hub height, in m/s."""
        speed_10m = torch.as_tensor(weather.speed_10m, dtype=torch.float32)
        speed_100m = torch.as_tensor(weather.speed_100m, dtype=torch.float32)
        temperature_c = torch.as_tensor(weather.temperature_c, dtype=torch.float32)
        if self.options.shear:
            usable = (speed_10m > MIN_SHEAR_SPEED_M_S) & (speed_100m > MIN_SHEAR_SPEED_M_S)
            safe_10m = torch.where(usable, speed_10m, torch.ones_like(speed_10m))
            safe_100m = torch.where(usable, speed_100m, torch.ones_like(speed_100m))
            exponent = torch.log(safe_100m / safe_10m) / np.log(10.0)
            exponent = torch.where(
                usable, exponent, torch.full_like(exponent, DEFAULT_SHEAR_EXPONENT)
            )
            exponent = exponent.clamp(*SHEAR_RANGE)
        else:
            exponent = torch.full_like(speed_100m, DEFAULT_SHEAR_EXPONENT)
        speed = speed_100m * (self.hub_height_m / 100.0) ** exponent
        density = common.FIXED_SURFACE_PRESSURE_PA / (
            common.DRY_AIR_GAS_CONSTANT * (temperature_c + 273.15)
        )
        density_exponent = (
            nn.functional.softplus(self.raw_density_exponent)
            if self.options.free_density_exponent
            else 1.0 / 3.0
        )
        speed = speed * (density / common.STANDARD_AIR_DENSITY_KG_M3) ** density_exponent
        if self.options.direction_terms:
            sin1 = torch.as_tensor(weather.direction_sin, dtype=torch.float32)
            cos1 = torch.as_tensor(weather.direction_cos, dtype=torch.float32)
            c = self.direction_coefficients
            harmonics = (
                c[0] * sin1 + c[1] * cos1 + c[2] * 2 * sin1 * cos1 + c[3] * (cos1**2 - sin1**2)
            )
            speed = speed * torch.exp(harmonics)
        return self._adapt(speed=speed, weather=weather)

    def _adapt(self, *, speed: torch.Tensor, weather: WeatherArrays) -> torch.Tensor:
        """Return `speed` after the effective-wind adapter (a no-op for `adapter="none"`)."""
        adapter = self.options.adapter
        if adapter == "none":
            return speed
        speed_10m = torch.as_tensor(weather.speed_10m, dtype=torch.float32)
        speed_100m = torch.as_tensor(weather.speed_100m, dtype=torch.float32)
        usable = (speed_10m > MIN_SHEAR_SPEED_M_S) & (speed_100m > MIN_SHEAR_SPEED_M_S)
        ratio = torch.where(
            usable,
            speed_10m / torch.where(usable, speed_100m, torch.ones_like(speed_100m)),
            torch.full_like(speed_10m, 0.5),
        ).clamp(0.05, 1.5)
        if adapter in ("shear_k", "shear_k_sectors"):
            speed = speed * ratio**self.shear_adapter_exponent
        if adapter in ("sectors", "shear_k_sectors"):
            sin1 = torch.as_tensor(weather.direction_sin, dtype=torch.float32)
            cos1 = torch.as_tensor(weather.direction_cos, dtype=torch.float32)
            degrees = torch.rad2deg(torch.atan2(sin1, cos1)) % 360.0
            sector = (
                torch.floor((degrees + 180.0 / SECTOR_COUNT) / (360.0 / SECTOR_COUNT)).long()
                % SECTOR_COUNT
            )
            speed = speed * torch.exp(self.sector_log_scale[sector])
        if adapter in NET_INPUT_COUNT:
            if adapter == "blind":
                inputs = (speed_100m / 15.0)[:, None]
            else:
                sin1 = torch.as_tensor(weather.direction_sin, dtype=torch.float32)
                cos1 = torch.as_tensor(weather.direction_cos, dtype=torch.float32)
                temperature = (
                    torch.as_tensor(weather.temperature_c, dtype=torch.float32) - 10.0
                ) / 10.0
                columns = [sin1, cos1, ratio, temperature, (speed_100m / 15.0)]
                if adapter == "net":
                    angle = torch.as_tensor(weather.hour, dtype=torch.float32) * (2 * np.pi / 24.0)
                    columns += [torch.sin(angle), torch.cos(angle)]
                inputs = torch.stack(columns, dim=1)
            raw = self.adapter_net(inputs)
            scale = _sigmoid_range(raw[:, 0], *SCALE_RANGE)
            speed = speed * scale + OFFSET_LIMIT_M_S * torch.tanh(raw[:, 1])
        return speed.clamp(min=0.0)

    def _net_curve(self, speed: torch.Tensor) -> torch.Tensor:
        """Return the monotone neural curve at `speed`, zero at zero speed."""

        def run(x: torch.Tensor) -> torch.Tensor:
            h = torch.tanh(
                x[:, None] / 15.0 @ nn.functional.softplus(self.net_raw_1) + self.net_bias_1
            )
            h = torch.tanh(h @ nn.functional.softplus(self.net_raw_2) + self.net_bias_2)
            return torch.sigmoid(
                (h @ nn.functional.softplus(self.net_raw_3)).squeeze(1) + self.net_bias_3
            )

        zero = run(torch.zeros(1))
        return nn.functional.softplus(self.raw_scale) * (run(speed) - zero).clamp(min=0.0)

    def active_power(self, *, speed: torch.Tensor) -> torch.Tensor:
        """Return the power curve at `speed`, a fraction of capacity."""
        if self.options.monotone_net:
            return self._net_curve(speed)
        scale = nn.functional.softplus(self.raw_scale)
        blur = nn.functional.softplus(self.raw_blur)
        shifted = (speed[:, None] + blur * self.blur_nodes[None, :]).clamp(min=0.0)
        ramp = ((shifted - self.cut_in_m_s) / (self.rated_m_s - self.cut_in_m_s)).clamp(0.0, 1.0)
        shape = _sigmoid_range(self.raw_shape, 1.0, 5.0)
        cut_out = torch.sigmoid((_sigmoid_range(self.raw_cut_out, 20.0, 35.0) - shifted) / 1.0)
        return scale * ((ramp**shape) * cut_out * self.blur_weights[None, :]).sum(dim=1)

    def forward(self, *, weather: WeatherArrays) -> torch.Tensor:
        """Return the expected power as a fraction of capacity for each row of `weather`."""
        speed = self.hub_height_speed(weather=weather)
        active = self.active_power(speed=speed)
        if self.options.group_count:
            group = torch.as_tensor(weather.group, dtype=torch.long)
            active = active * torch.exp(self.group_log_scale[group])
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
    *,
    weather: WeatherArrays,
    target: np.ndarray,
    with_idle_layer: bool,
    seed: int,
    options: CoreOptions = DEFAULT_OPTIONS,
) -> WindCore:
    """Fit one farm's core by full-batch Adam on the L1 loss of the observed fraction.

    A log-normal prior (median `HUB_HEIGHT_PRIOR_M`) enters the loss with a weight of one over the
    number of rows, so it is negligible.
    """
    torch.manual_seed(seed)
    torch.set_num_threads(THREADS)
    core = WindCore(
        with_idle_layer=with_idle_layer,
        idle_floor=idle_floor_start(speed_100m=weather.speed_100m, target=target),
        options=options,
    )
    goal = torch.as_tensor(target, dtype=torch.float32)
    steps = NET_STEPS if options.monotone_net else FIT_STEPS
    rate = NET_LEARNING_RATE if options.monotone_net else LEARNING_RATE
    adapter_parameters = [
        p
        for name, p in core.named_parameters()
        if name.startswith(("adapter_net", "sector_log", "shear_adapter"))
    ]
    other_parameters = [
        p
        for name, p in core.named_parameters()
        if not name.startswith(("adapter_net", "sector_log", "shear_adapter"))
    ]
    if options.adapter != "none":
        steps = ADAPTER_STEPS
    optimiser = torch.optim.AdamW(
        [
            {"params": other_parameters, "lr": rate, "weight_decay": 0.0},
            {
                "params": adapter_parameters,
                "lr": ADAPTER_LEARNING_RATE,
                "weight_decay": ADAPTER_WEIGHT_DECAY,
            },
        ]
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps)
    log_prior = np.log(HUB_HEIGHT_PRIOR_M)
    for _ in range(steps):
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
        pieces = [
            core(weather=weather.take(slice(start, start + chunk))).numpy()
            for start in range(0, weather.speed_100m.size, chunk)
        ]
    return np.concatenate(pieces)


def predict_hub_speed(
    *, core: WindCore, weather: WeatherArrays, chunk: int = 500_000
) -> np.ndarray:
    """Return the core's adjusted hub-height speed for every row, in float32."""
    with torch.no_grad():
        pieces = [
            core.hub_height_speed(weather=weather.take(slice(start, start + chunk))).numpy()
            for start in range(0, weather.speed_100m.size, chunk)
        ]
    return np.concatenate(pieces)


@dataclass(frozen=True)
class ResidualQuantiles:
    """Empirical quantiles of observed minus expected power, in bins of a binning variable.

    Attributes:
        centres: Mean binning variable of each bin, ascending, shape (bins,).
        offsets: Residual quantile at each of `common.LEVELS` per bin, shape (bins, 11).
    """

    centres: np.ndarray
    offsets: np.ndarray

    @classmethod
    def fit(
        cls, *, expected: np.ndarray, observed: np.ndarray, bin_variable: np.ndarray | None = None
    ) -> Self:
        """Return the residual quantiles of `observed - expected` per bin of `bin_variable`.

        `bin_variable` defaults to `expected`.
        """
        variable = expected if bin_variable is None else bin_variable
        order = np.argsort(variable, kind="stable")
        centres, offsets = [], []
        for chunk in np.array_split(order, RESIDUAL_BINS):
            centres.append(float(variable[chunk].mean()))
            offsets.append(np.quantile(observed[chunk] - expected[chunk], common.LEVELS))
        return cls(centres=np.array(centres), offsets=np.array(offsets))

    def quantiles(
        self, *, expected: np.ndarray, bin_variable: np.ndarray | None = None
    ) -> np.ndarray:
        """Return shape (rows, 11) quantiles: expected power plus interpolated offsets, clipped."""
        variable = expected if bin_variable is None else bin_variable
        columns = [
            np.interp(variable, self.centres, self.offsets[:, index])
            for index in range(len(common.LEVELS))
        ]
        out = expected[:, None] + np.stack(columns, axis=1)
        return np.sort(np.clip(out, 0.0, 1.0), axis=1).astype(np.float32)


class PinballSpread(nn.Module):
    """Quantile offsets about the median, piecewise linear in hub speed, ordered by construction."""

    def __init__(self, *, start_offsets: np.ndarray) -> None:
        """Start from `start_offsets`, shape (knots, 11): the empirical residual quantiles."""
        super().__init__()
        median = common.MEDIAN_INDEX
        self.median = nn.Parameter(torch.as_tensor(start_offsets[:, median], dtype=torch.float32))
        down = -np.diff(start_offsets[:, : median + 1][:, ::-1], axis=1)
        up = np.diff(start_offsets[:, median:], axis=1)
        self.raw_down = nn.Parameter(
            torch.as_tensor(np.log(np.expm1(np.maximum(down, 1e-3))), dtype=torch.float32)
        )
        self.raw_up = nn.Parameter(
            torch.as_tensor(np.log(np.expm1(np.maximum(up, 1e-3))), dtype=torch.float32)
        )

    def table(self) -> torch.Tensor:
        """Return shape (knots, 11) offsets at the knots."""
        down = self.median[:, None] - torch.cumsum(nn.functional.softplus(self.raw_down), dim=1)
        up = self.median[:, None] + torch.cumsum(nn.functional.softplus(self.raw_up), dim=1)
        return torch.cat([down.flip(1), self.median[:, None], up], dim=1)

    def forward(self, *, speed: torch.Tensor) -> torch.Tensor:
        """Return shape (rows, 11) offsets at `speed`."""
        knots = torch.as_tensor(SPREAD_KNOTS, dtype=torch.float32)
        step = float(knots[1] - knots[0])
        position = (speed / step).clamp(0.0, knots.numel() - 1.0001)
        low = position.floor().long()
        fraction = (position - low)[:, None]
        table = self.table()
        return table[low] * (1 - fraction) + table[low + 1] * fraction


@dataclass(frozen=True)
class PinballCore:
    """A core fitted jointly with its spread on pinball loss."""

    core: WindCore
    spread: PinballSpread

    def quantiles(self, *, weather: WeatherArrays, chunk: int = 500_000) -> np.ndarray:
        """Return shape (rows, 11) sorted quantiles clipped to 0 to 1."""
        pieces = []
        with torch.no_grad():
            for start in range(0, weather.speed_100m.size, chunk):
                part = weather.take(slice(start, start + chunk))
                mu = self.core(weather=part)
                speed = self.core.hub_height_speed(weather=part)
                pieces.append((mu[:, None] + self.spread(speed=speed)).numpy())
        out = np.concatenate(pieces)
        return np.sort(np.clip(out, 0.0, 1.0), axis=1).astype(np.float32)


def fit_core_pinball(
    *, weather: WeatherArrays, target: np.ndarray, with_idle_layer: bool, seed: int
) -> PinballCore:
    """Fit the core and a hub-speed-dependent spread together by minimising pinball loss.

    The core starts from its L1 fit, the spread from the empirical residual quantiles in 2 m/s
    windows of hub speed, and both are then refitted on the mass-weighted pinball loss at the 11
    levels, which is close to the nine-level loss the quantile score measures.
    """
    core = fit_core(weather=weather, target=target, with_idle_layer=with_idle_layer, seed=seed)
    mu = predict_expected_power(core=core, weather=weather)
    speed = predict_hub_speed(core=core, weather=weather)
    start = np.zeros((SPREAD_KNOTS.size, len(common.LEVELS)))
    for index, knot in enumerate(SPREAD_KNOTS):
        near = np.abs(speed - knot) <= 1.5
        residual = (target - mu)[near] if near.sum() >= 200 else (target - mu)
        start[index] = np.quantile(residual, common.LEVELS)
    start = np.maximum.accumulate(start, axis=1)
    spread = PinballSpread(start_offsets=start)
    params = list(core.parameters()) + list(spread.parameters())
    optimiser = torch.optim.Adam(params, lr=PINBALL_LEARNING_RATE)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=PINBALL_STEPS)
    goal = torch.as_tensor(target, dtype=torch.float32)[:, None]
    levels = torch.as_tensor(common.LEVELS, dtype=torch.float32)[None, :]
    masses = torch.as_tensor(common.ATOM_MASSES, dtype=torch.float32)[None, :]
    core.train()
    for _ in range(PINBALL_STEPS):
        optimiser.zero_grad()
        mu_t = core(weather=weather)
        speed_t = core.hub_height_speed(weather=weather)
        predicted = mu_t[:, None] + spread(speed=speed_t)
        difference = goal - predicted
        pinball = torch.where(difference >= 0, levels * difference, (levels - 1) * difference)
        (pinball * masses).sum(dim=1).mean().backward()
        optimiser.step()
        schedule.step()
    return PinballCore(core=core.eval(), spread=spread.eval())


def fit_speed_scale(*, weather: WeatherArrays, target: np.ndarray) -> float:
    """Return the factor on both ERA5 speeds that best fits a fixed generic power curve.

    The reference curve is the core at its starting parameters (hub height 65 m, cut-in 3 m/s, ramp
    8 m/s, shape 2.5, cut-out 27 m/s, blur 1 m/s, scale 0.95, no idle layer). Only the factor is
    fitted, by Adam on the L1 loss; the result is a farm-specific stand-in for the mismatch
    between the 25 km grid cell and the farm.
    """
    torch.set_num_threads(THREADS)
    reference = WindCore(with_idle_layer=False, idle_floor=1e-3)
    for p in reference.parameters():
        p.requires_grad_(False)
    log_factor = nn.Parameter(torch.zeros(()))
    optimiser = torch.optim.Adam([log_factor], lr=0.03)
    goal = torch.as_tensor(target, dtype=torch.float32)
    base = reference.hub_height_speed(weather=weather).detach()
    for _ in range(SCALE_FIT_STEPS):
        optimiser.zero_grad()
        loss = (reference.active_power(speed=base * torch.exp(log_factor)) - goal).abs().mean()
        loss.backward()
        optimiser.step()
    return float(torch.exp(log_factor.detach()))
