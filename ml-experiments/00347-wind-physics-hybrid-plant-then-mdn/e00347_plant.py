"""The wind-farm plant model: a speed-to-power curve with priors and a direction-dependent loss.

One farm's plant turns the ERA5 100 m wind speed and direction into an expected power as a fraction
of the farm's effective capacity. The curve is `scale * g(x)` with `x = (v - cut_in) / (rated -
cut_in)`, where `g` rises as a cubic, joins a flat top through a soft minimum of learned width, and
a logistic cut-out of learned width multiplies the whole curve. The optional direction term
multiplies the wind speed by `f(direction) = 1 - sum_k depth_k * exp(kappa_k * (cos(direction -
centre_k) - 1))`, three von Mises dips whose depths are held between 0 and 15% and pulled towards 0
by a prior.

Fitting is maximum a posteriori: a Laplace likelihood on the observed fraction (so the median is
the fitted quantity; the loss is the pinball loss at 0.5) plus Gaussian priors. The likelihood is
divided by `HALF_HOURS_PER_DAY` so that one day, not one half-hour, counts as one independent
observation (speed errors from ERA5 last for hours to days).
"""

import os
from dataclasses import dataclass, field, replace
from typing import Final

import numpy as np
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

HALF_HOURS_PER_DAY: Final[int] = 48
DIP_COUNT: Final[int] = 3
FIT_STEPS: Final[int] = 1_200
LEARNING_RATE: Final[float] = 0.05
LOWER_SMOOTHING: Final[float] = 0.03
"""Width (in units of the cut-in to rated span) of the smooth start of the cubic."""
DEPTH_RANGE: Final[tuple[float, float]] = (0.0, 0.15)
KAPPA_RANGE: Final[tuple[float, float]] = (1.0, 30.0)
DEPTH_PRIOR_SD: Final[float] = 0.04
THREADS: Final[int] = int(os.environ.get("E00347_TORCH_THREADS", "1"))

PRIOR_CUT_IN: Final[tuple[float, float]] = (3.0, 0.5)
PRIOR_RATED: Final[tuple[float, float]] = (12.5, 1.5)
PRIOR_CUT_OUT: Final[tuple[float, float]] = (25.0, 2.0)
PRIOR_SATURATION_WIDTH: Final[tuple[float, float]] = (0.08, 0.05)
PRIOR_CUT_OUT_WIDTH: Final[tuple[float, float]] = (1.0, 0.5)
PRIOR_BLUR: Final[tuple[float, float]] = (1.0, 0.7)
GAUSS_HERMITE_POINTS: Final[int] = 7
PRIOR_HUB_HEIGHT_M: Final[float] = 65.0
PRIOR_HUB_HEIGHT_LOG_SD: Final[float] = 0.2
HUB_HEIGHT_RANGE_M: Final[tuple[float, float]] = (25.0, 140.0)
SHEAR_RANGE: Final[tuple[float, float]] = (0.0, 0.5)
DEFAULT_SHEAR_EXPONENT: Final[float] = 0.14
MIN_SHEAR_SPEED_M_S: Final[float] = 0.5
STANDARD_AIR_DENSITY_KG_M3: Final[float] = 1.225
DRY_AIR_GAS_CONSTANT: Final[float] = 287.05
FIXED_SURFACE_PRESSURE_PA: Final[float] = 100_000.0
SCALE_PRIOR_RELATIVE_SD: Final[float] = 0.15


@dataclass(frozen=True)
class Weather:
    """The plant's weather inputs, equal-length float arrays.

    Attributes:
        speed_100m: ERA5 or ensemble 100 m wind speed, m/s.
        speed_10m: 10 m wind speed, m/s (sets the shear exponent when the plant has a hub height).
        temperature_c: 2 m air temperature in degrees Celsius (sets the air density).
        direction_rad: 100 m wind direction in radians.
    """

    speed_100m: np.ndarray
    speed_10m: np.ndarray
    temperature_c: np.ndarray
    direction_rad: np.ndarray

    def take(self, index: np.ndarray) -> Weather:
        """Return the rows selected by `index`."""
        return Weather(
            speed_100m=self.speed_100m[index],
            speed_10m=self.speed_10m[index],
            temperature_c=self.temperature_c[index],
            direction_rad=self.direction_rad[index],
        )

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (speed_100m, speed_10m, temperature_c, direction) as float32 tensors."""
        return tuple(  # ty: ignore[invalid-return-type]
            torch.as_tensor(a, dtype=torch.float32)
            for a in (self.speed_100m, self.speed_10m, self.temperature_c, self.direction_rad)
        )


def _range(raw: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Return `raw` squashed into the open interval from `low` to `high`."""
    return low + (high - low) * torch.sigmoid(raw)


def _inverse_range(value: float, low: float, high: float) -> float:
    """Return the raw value whose `_range` is `value`."""
    unit = min(max((value - low) / (high - low), 1e-4), 1 - 1e-4)
    return float(np.log(unit / (1 - unit)))


def _inverse_softplus(value: float) -> float:
    """Return the raw value whose softplus is `value`."""
    return float(np.log(np.expm1(max(value, 1e-4))))


CUT_IN_RANGE: Final[tuple[float, float]] = (1.5, 5.5)
RATED_RANGE: Final[tuple[float, float]] = (8.0, 18.0)
CUT_OUT_RANGE: Final[tuple[float, float]] = (18.0, 40.0)
CUT_OUT_WIDTH_RANGE: Final[tuple[float, float]] = (0.3, 3.0)
SATURATION_RANGE: Final[tuple[float, float]] = (0.01, 0.4)
BLUR_RANGE: Final[tuple[float, float]] = (0.05, 4.0)


@dataclass(frozen=True)
class PlantParameters:
    """The plant's parameters in physical units.

    Attributes:
        cut_in: Speed (m/s) where the cubic starts.
        rated: Speed (m/s) where the cubic reaches the flat top (the middle of the soft join).
        cut_out: Speed (m/s) at which the logistic cut-out halves the output.
        cut_out_width: Logistic width (m/s) of the cut-out.
        saturation_width: Width of the soft join, in units of the cut-in to rated span.
        scale: The flat-top output as a fraction of effective capacity.
        blur: Standard deviation (m/s) of the Gaussian blur over wind speed that stands for the
            25 km ERA5 cell's sub-grid wind (the curve is averaged over speeds `v + blur * z`).
        hub_height: Effective hub height in metres, or `None` for a plant on the 100 m speed alone.
            With a hub height, the speed is the 100 m speed taken to that height by the shear
            exponent of the 10 m and 100 m speeds and multiplied by the cube root of the air
            density ratio.
        depths: Depth of each direction dip (0 to 0.15 of the wind speed).
        centres_deg: Centre of each dip in degrees.
        kappas: Von Mises concentration of each dip (larger is narrower).
    """

    cut_in: float
    rated: float
    cut_out: float
    cut_out_width: float
    saturation_width: float
    scale: float
    blur: float = PRIOR_BLUR[0]
    hub_height: float | None = None
    depths: tuple[float, ...] = (0.0,) * DIP_COUNT
    centres_deg: tuple[float, ...] = (0.0, 120.0, 240.0)
    kappas: tuple[float, ...] = (4.0,) * DIP_COUNT

    def as_dict(self) -> dict[str, float | list[float] | None]:
        """Return the parameters as a JSON-ready dictionary."""
        return {
            "cut_in": self.cut_in,
            "rated": self.rated,
            "cut_out": self.cut_out,
            "cut_out_width": self.cut_out_width,
            "saturation_width": self.saturation_width,
            "scale": self.scale,
            "blur": self.blur,
            "hub_height": self.hub_height,
            "depths": list(self.depths),
            "centres_deg": list(self.centres_deg),
            "kappas": list(self.kappas),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> PlantParameters:
        """Rebuild the parameters from `as_dict`."""
        return cls(
            cut_in=payload["cut_in"],
            rated=payload["rated"],
            cut_out=payload["cut_out"],
            cut_out_width=payload["cut_out_width"],
            saturation_width=payload["saturation_width"],
            scale=payload["scale"],
            blur=payload["blur"],
            hub_height=payload["hub_height"],
            depths=tuple(payload["depths"]),
            centres_deg=tuple(payload["centres_deg"]),
            kappas=tuple(payload["kappas"]),
        )


def prior_parameters(*, prior_scale: float, shear_density: bool) -> PlantParameters:
    """Return the parameters at the prior means (no direction term), the untrained plant."""
    return PlantParameters(
        hub_height=PRIOR_HUB_HEIGHT_M if shear_density else None,
        cut_in=PRIOR_CUT_IN[0],
        rated=PRIOR_RATED[0],
        cut_out=PRIOR_CUT_OUT[0],
        cut_out_width=PRIOR_CUT_OUT_WIDTH[0],
        saturation_width=PRIOR_SATURATION_WIDTH[0],
        scale=prior_scale,
    )


class Plant(nn.Module):
    """One farm's plant model, in unconstrained raw parameters."""

    blur_nodes: torch.Tensor
    blur_weights: torch.Tensor

    def __init__(self, *, start: PlantParameters, prior_scale: float, with_direction: bool) -> None:
        """Set the starting parameters.

        Args:
            start: The parameters to start from.
            prior_scale: The prior mean of the flat-top output (fraction of effective capacity).
            with_direction: Whether the direction term is part of the model.
        """
        super().__init__()
        self.with_direction = with_direction
        self.prior_scale = prior_scale
        self.raw_cut_in = nn.Parameter(torch.tensor(_inverse_range(start.cut_in, *CUT_IN_RANGE)))
        self.raw_rated = nn.Parameter(torch.tensor(_inverse_range(start.rated, *RATED_RANGE)))
        self.raw_cut_out = nn.Parameter(torch.tensor(_inverse_range(start.cut_out, *CUT_OUT_RANGE)))
        self.raw_cut_out_width = nn.Parameter(
            torch.tensor(_inverse_range(start.cut_out_width, *CUT_OUT_WIDTH_RANGE))
        )
        self.raw_saturation = nn.Parameter(
            torch.tensor(_inverse_range(start.saturation_width, *SATURATION_RANGE))
        )
        self.raw_scale = nn.Parameter(torch.tensor(_inverse_softplus(start.scale)))
        self.raw_blur = nn.Parameter(torch.tensor(_inverse_range(start.blur, *BLUR_RANGE)))
        self.shear_density = start.hub_height is not None
        self.raw_hub_height = nn.Parameter(
            torch.tensor(
                _inverse_range(start.hub_height or PRIOR_HUB_HEIGHT_M, *HUB_HEIGHT_RANGE_M)
            ),
            requires_grad=self.shear_density,
        )
        nodes, weights = np.polynomial.hermite.hermgauss(GAUSS_HERMITE_POINTS)
        self.register_buffer(
            "blur_nodes", torch.as_tensor(np.sqrt(2.0) * nodes, dtype=torch.float32)
        )
        self.register_buffer(
            "blur_weights", torch.as_tensor(weights / np.sqrt(np.pi), dtype=torch.float32)
        )
        self.raw_depths = nn.Parameter(
            torch.tensor([_inverse_range(d, *DEPTH_RANGE) for d in start.depths])
        )
        self.centres = nn.Parameter(
            torch.tensor(np.radians(start.centres_deg), dtype=torch.float32)
        )
        self.raw_kappas = nn.Parameter(
            torch.tensor([_inverse_range(k, *KAPPA_RANGE) for k in start.kappas])
        )

    @property
    def cut_in(self) -> torch.Tensor:
        """Return the cut-in speed."""
        return _range(self.raw_cut_in, *CUT_IN_RANGE)

    @property
    def rated(self) -> torch.Tensor:
        """Return the rated speed, kept above the cut-in speed by 2 m/s."""
        return torch.maximum(_range(self.raw_rated, *RATED_RANGE), self.cut_in + 2.0)

    @property
    def cut_out(self) -> torch.Tensor:
        """Return the cut-out speed."""
        return _range(self.raw_cut_out, *CUT_OUT_RANGE)

    @property
    def cut_out_width(self) -> torch.Tensor:
        """Return the width of the cut-out."""
        return _range(self.raw_cut_out_width, *CUT_OUT_WIDTH_RANGE)

    @property
    def saturation_width(self) -> torch.Tensor:
        """Return the width of the soft join to the flat top."""
        return _range(self.raw_saturation, *SATURATION_RANGE)

    @property
    def scale(self) -> torch.Tensor:
        """Return the flat-top output."""
        return nn.functional.softplus(self.raw_scale)

    @property
    def hub_height(self) -> torch.Tensor:
        """Return the effective hub height in metres."""
        return _range(self.raw_hub_height, *HUB_HEIGHT_RANGE_M)

    def hub_speed(self, *, weather: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Return the wind speed the curve sees, before the direction factor and the blur."""
        speed_100m, speed_10m, temperature_c, _ = weather
        if not self.shear_density:
            return speed_100m
        usable = (speed_10m > MIN_SHEAR_SPEED_M_S) & (speed_100m > MIN_SHEAR_SPEED_M_S)
        safe_10m = torch.where(usable, speed_10m, torch.ones_like(speed_10m))
        safe_100m = torch.where(usable, speed_100m, torch.ones_like(speed_100m))
        exponent = torch.log(safe_100m / safe_10m) / float(np.log(10.0))
        exponent = torch.where(usable, exponent, torch.full_like(exponent, DEFAULT_SHEAR_EXPONENT))
        exponent = exponent.clamp(*SHEAR_RANGE)
        speed = speed_100m * (self.hub_height / 100.0) ** exponent
        density = FIXED_SURFACE_PRESSURE_PA / (DRY_AIR_GAS_CONSTANT * (temperature_c + 273.15))
        return speed * (density / STANDARD_AIR_DENSITY_KG_M3) ** (1.0 / 3.0)

    @property
    def blur(self) -> torch.Tensor:
        """Return the blur standard deviation in m/s."""
        return _range(self.raw_blur, *BLUR_RANGE)

    @property
    def depths(self) -> torch.Tensor:
        """Return the dip depths."""
        return _range(self.raw_depths, *DEPTH_RANGE)

    @property
    def kappas(self) -> torch.Tensor:
        """Return the dip concentrations."""
        return _range(self.raw_kappas, *KAPPA_RANGE)

    def direction_factor(self, *, direction: torch.Tensor) -> torch.Tensor:
        """Return `f(direction)`, the factor on wind speed (1 where there is no loss)."""
        angle = direction[:, None] - self.centres[None, :]
        dips = self.depths[None, :] * torch.exp(self.kappas[None, :] * (torch.cos(angle) - 1.0))
        return (1.0 - dips.sum(dim=1)).clamp(min=0.5)

    def curve(self, *, speed: torch.Tensor) -> torch.Tensor:
        """Return the blurred expected power at `speed` (m/s), a fraction of effective capacity."""
        shifted = (speed[:, None] + self.blur * self.blur_nodes[None, :]).clamp(min=0.0)
        return (self.sharp_curve(speed=shifted) * self.blur_weights[None, :]).sum(dim=1)

    def sharp_curve(self, *, speed: torch.Tensor) -> torch.Tensor:
        """Return the unblurred curve at `speed`."""
        x = (speed - self.cut_in) / (self.rated - self.cut_in)
        width = self.saturation_width
        capped = 1.0 - width * nn.functional.softplus((1.0 - x) / width)
        lower = LOWER_SMOOTHING * nn.functional.softplus(capped / LOWER_SMOOTHING)
        cut_out = torch.sigmoid((self.cut_out - speed) / self.cut_out_width)
        return self.scale * lower**3 * cut_out

    def forward(self, *, weather: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Return the expected power for each row of `weather` (`Weather.tensors()`)."""
        speed = self.hub_speed(weather=weather)
        if self.with_direction:
            speed = speed * self.direction_factor(direction=weather[3])
        return self.curve(speed=speed)

    def log_prior(self) -> torch.Tensor:
        """Return the negative log prior density (up to a constant)."""
        total = torch.zeros(())
        for value, (mean, sd) in (
            (self.cut_in, PRIOR_CUT_IN),
            (self.rated, PRIOR_RATED),
            (self.cut_out, PRIOR_CUT_OUT),
            (self.saturation_width, PRIOR_SATURATION_WIDTH),
            (self.cut_out_width, PRIOR_CUT_OUT_WIDTH),
            (self.blur, PRIOR_BLUR),
            (self.scale, (self.prior_scale, SCALE_PRIOR_RELATIVE_SD * self.prior_scale)),
        ):
            total = total + 0.5 * ((value - mean) / sd) ** 2
        if self.shear_density:
            total = (
                total
                + 0.5
                * (
                    (torch.log(self.hub_height) - np.log(PRIOR_HUB_HEIGHT_M))
                    / PRIOR_HUB_HEIGHT_LOG_SD
                )
                ** 2
            )
        if self.with_direction:
            total = total + 0.5 * ((self.depths / DEPTH_PRIOR_SD) ** 2).sum()
        return total

    def parameters_physical(self) -> PlantParameters:
        """Return the current parameters in physical units."""
        return PlantParameters(
            cut_in=float(self.cut_in),
            rated=float(self.rated),
            cut_out=float(self.cut_out),
            cut_out_width=float(self.cut_out_width),
            saturation_width=float(self.saturation_width),
            scale=float(self.scale),
            blur=float(self.blur),
            hub_height=float(self.hub_height) if self.shear_density else None,
            depths=tuple(float(d) for d in self.depths),
            centres_deg=tuple(float(np.degrees(c) % 360.0) for c in self.centres.detach()),
            kappas=tuple(float(k) for k in self.kappas),
        )


def expected_power(
    *, parameters: PlantParameters, weather: Weather, prior_scale: float
) -> np.ndarray:
    """Return the plant's expected power (fraction of effective capacity) for each row.

    The direction term is applied whenever any depth is non-zero.
    """
    with_direction = any(depth > 0 for depth in parameters.depths)
    model = Plant(start=parameters, prior_scale=prior_scale, with_direction=with_direction)
    with torch.no_grad():
        return model(weather=weather.tensors()).numpy().astype(np.float64)


@dataclass(frozen=True)
class FitResult:
    """A fitted plant.

    Attributes:
        parameters: The fitted parameters.
        laplace_scale: The Laplace scale `b` used in the likelihood.
        loss_history: Total loss at a few checkpoints (for diagnosing a failed fit).
    """

    parameters: PlantParameters
    laplace_scale: float
    loss_history: list[float] = field(default_factory=list)


def fit_plant(
    *,
    weather: Weather,
    target: np.ndarray,
    weight: np.ndarray,
    prior_scale: float,
    with_direction: bool,
    start: PlantParameters,
    steps: int = FIT_STEPS,
) -> FitResult:
    """Fit one farm's plant by maximum a posteriori with Adam.

    Args:
        weather: The training rows' weather.
        target: Observed power as a fraction of effective capacity.
        weight: Per-row likelihood weight (0 drops the row; the experiment 210 mask and the
            Stage 3b flagged blocks enter here).
        prior_scale: Prior mean of the flat-top output.
        with_direction: Whether to include the direction term.
        start: The starting parameters (use `prior_parameters` for the untrained plant).
        steps: Adam steps.

    Returns:
        The fitted plant.
    """
    torch.set_num_threads(THREADS)
    model = Plant(start=start, prior_scale=prior_scale, with_direction=with_direction)
    inputs = weather.tensors()
    goal = torch.as_tensor(target, dtype=torch.float32)
    weight_t = torch.as_tensor(weight, dtype=torch.float32)
    with torch.no_grad():
        start_error = (model(weather=inputs) - goal).abs()
        laplace_scale = float((start_error * weight_t).sum() / weight_t.sum())
    optimiser = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps)
    history = []
    normaliser = float(weight_t.sum()) / HALF_HOURS_PER_DAY
    for step in range(steps):
        optimiser.zero_grad()
        error = (model(weather=inputs) - goal).abs()
        likelihood = (error * weight_t).sum() / laplace_scale / HALF_HOURS_PER_DAY
        loss = (likelihood + model.log_prior()) / normaliser
        loss.backward()
        optimiser.step()
        schedule.step()
        if step % 300 == 0 or step == steps - 1:
            history.append(float(loss.detach()))
    with torch.no_grad():
        parameters = model.parameters_physical()
    return FitResult(parameters=parameters, laplace_scale=laplace_scale, loss_history=history)


def log_slope(*, parameters: PlantParameters, weather: Weather, prior_scale: float) -> np.ndarray:
    """Return `d ln(power) / d ln(speed)` of the plant (direction term off) by a central difference.

    The speed is the 100 m speed scaled by 1.01 and 0.99 with the other inputs held, so the slope
    includes the shear and density mapping.
    """
    plain = replace(parameters, depths=(0.0,) * DIP_COUNT)
    results = []
    for factor in (1.01, 0.99):
        scaled = Weather(
            speed_100m=weather.speed_100m * factor,
            speed_10m=weather.speed_10m,
            temperature_c=weather.temperature_c,
            direction_rad=weather.direction_rad,
        )
        results.append(
            np.log(
                np.maximum(
                    expected_power(parameters=plain, weather=scaled, prior_scale=prior_scale), 1e-9
                )
            )
        )
    return (results[0] - results[1]) / (np.log(1.01) - np.log(0.99))


POSTERIOR_PARAMETERS: Final[tuple[str, ...]] = (
    "cut_in",
    "rated",
    "cut_out",
    "cut_out_width",
    "saturation_width",
    "scale",
    "blur",
    "hub_height",
)
"""Parameters of the no-direction plant, for the Laplace approximation to the posterior."""


def posterior_sd(
    *,
    parameters: PlantParameters,
    weather: Weather,
    weight: np.ndarray,
    prior_scale: float,
    laplace_scale: float,
) -> dict[str, float]:
    """Return the Laplace-approximation posterior standard deviation of each no-direction parameter.

    The information is the Gauss-Newton matrix of the Laplace likelihood (a Laplace distribution of
    scale `b` has Fisher information `1 / b^2` for its location), divided by `HALF_HOURS_PER_DAY`
    as in the fit, plus the diagonal prior precision. Jacobians are central differences in physical
    units.

    Args:
        parameters: The fitted parameters (direction depths must be zero).
        weather: The training weather.
        weight: The per-row likelihood weight used in the fit.
        prior_scale: Prior mean of the flat-top output.
        laplace_scale: The Laplace scale `b` the fit used.

    Returns:
        The posterior standard deviation by parameter name, in the parameter's own unit.
    """
    names = [n for n in POSTERIOR_PARAMETERS if getattr(parameters, n) is not None]
    columns = []
    for name in names:
        value = float(getattr(parameters, name))
        step = 1e-3 * max(abs(value), 1.0)
        ends = []
        for sign in (1, -1):
            moved = replace(parameters, **{name: value + sign * step})
            ends.append(expected_power(parameters=moved, weather=weather, prior_scale=prior_scale))
        columns.append((ends[0] - ends[1]) / (2 * step))
    jacobian = np.stack(columns, axis=1) * np.sqrt(weight)[:, None]
    information = jacobian.T @ jacobian / laplace_scale**2 / HALF_HOURS_PER_DAY
    prior_sd = {
        "cut_in": PRIOR_CUT_IN[1],
        "rated": PRIOR_RATED[1],
        "cut_out": PRIOR_CUT_OUT[1],
        "cut_out_width": PRIOR_CUT_OUT_WIDTH[1],
        "saturation_width": PRIOR_SATURATION_WIDTH[1],
        "scale": SCALE_PRIOR_RELATIVE_SD * prior_scale,
        "blur": PRIOR_BLUR[1],
        "hub_height": PRIOR_HUB_HEIGHT_LOG_SD * (parameters.hub_height or PRIOR_HUB_HEIGHT_M),
    }
    information += np.diag([1.0 / prior_sd[n] ** 2 for n in names])
    covariance = np.linalg.inv(information)
    return {n: float(np.sqrt(covariance[i, i])) for i, n in enumerate(names)}
