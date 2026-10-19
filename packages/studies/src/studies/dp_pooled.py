"""The pooled PyTorch configuration of the differentiable-physics (DP) series, and its switches.

`studies.dp_gam` fits one convex model per substation. This module holds the model that Parts 3 to
6 grow from: one model for many substations, fitted by gradient descent, with shared daily shapes,
a shared calendar and weather module, an optional interaction network, a solar term per substation
and season-year, and a per-row observation operator (`studies.dp_operator`).

**Every later module is a switch in `DpConfig`.** The switches for modules that are not built
(the variational posterior, the smooth clip, the unmetered generator node, and drift in the pooled
model) raise `NotImplementedError` when turned on, so a later part replaces one raise with code and
changes no caller. `ResidualStates` and `WeatherInputLayer` are typed stubs for the per-site
residual states and the weather-input layer per product.

**The model, in points of effective capacity:**

    reading = observe( D - gate * C[site, season-year] * u )
    D = level[site] + mix[site] @ shared_shapes[day type, half-hour] + deviation[site, day type,
        half-hour] + holiday[period, band] + tent and harmonics terms + lighting ramps
        + softplus(slope[site, band]) * relu(threshold[site] - mixed smoothed temperature) / 10
        + interaction_network(calendar and weather inputs)

- The shared daily shapes are `K` curves over day type and half-hour. Each substation mixes them
  with softmax weights and adds a deviation that the penalty shrinks towards zero.
- The thermal-inertia mix is a softmax over the four smoothed temperatures, shared by all
  substations. The heating threshold is per substation with a prior around 15.5 degrees.
- No irradiance and no time trend enters the shared modules, and the interaction network's last
  layer starts at zero, so the solar term and the drift are not absorbed by it at the start.
- The capacity is the cumulative sum of softplus increments over the season-years, so it never
  decreases.
"""

import time
from dataclasses import dataclass, field
from typing import Final, Protocol

import numpy as np
import torch
from torch import nn

from studies.dp_calendar import (
    BAND_NAMES,
    DAY_TYPES,
    HALF_HOURS_PER_DAY,
    HEATING_THRESHOLD_C,
    HOLIDAY_PERIODS,
    SMOOTHING_HOURS,
    lighting_ramps,
)
from studies.dp_features import FeatureCache
from studies.dp_gam import DriftType, derated_unit_shape
from studies.dp_operator import (
    ApparentPowerParameters,
    observe,
    real_power,
    tan_phi_from_power_factor,
)

N_BANDS: Final[int] = len(BAND_NAMES)
N_PROFILE: Final[int] = len(DAY_TYPES) * HALF_HOURS_PER_DAY
N_PERIODS: Final[int] = len(HOLIDAY_PERIODS)
HEATING_SCALE_C: Final[float] = 10.0
THRESHOLD_PRIOR_SCALE_C: Final[float] = 2.0
"""The prior on a substation's heating threshold is a normal around 15.5 degrees, this wide."""
INITIAL_POWER_FACTOR: Final[float] = 0.97
"""The power factor that the MVA substations' `tan(phi)` starts from."""
INTERACTION_INPUTS: Final[int] = 4 + 2 + 2 + len(DAY_TYPES) + 1 + 1
"""Harmonics, tent, half-hour sine and cosine, day-type one-hot, mixed temperature, elevation."""


@dataclass(frozen=True)
class DpConfig:
    """Switches for every module of the DP model.

    Attributes:
        shared_shapes: The number of shared daily shapes `K`; 0 gives each substation its own
            profile only.
        interaction_hidden: The hidden-layer widths of the interaction network; empty means off.
        variational: Mean-field variational posterior over site-level parameters (not built).
        drift: Drift form of the pooled model (only `none` is built).
        clip: The smooth export-limit clip of the solar term (not built).
        generator_node: The unmetered generator node (not built).
    """

    shared_shapes: int = 4
    interaction_hidden: tuple[int, ...] = ()
    variational: bool = False
    drift: DriftType = "none"
    clip: bool = False
    generator_node: bool = False

    def check_supported(self) -> None:
        """Raise if a switch that is not built is on.

        Raises:
            NotImplementedError: Naming the first switch that is on but not built.
        """
        for name, value in (
            ("variational", self.variational),
            ("clip", self.clip),
            ("generator_node", self.generator_node),
            ("drift", self.drift != "none"),
        ):
            if value:
                msg = f"the {name} module is not built yet"
                raise NotImplementedError(msg)


@dataclass
class ResidualStates:
    """Per-site residual states, kept addressable for a later assimilation filter.

    Attributes:
        level: The persistent residual level of each substation and hour band, in points.
        anomaly: The transient residual anomaly of each substation and hour band, in points.
    """

    level: dict[str, np.ndarray] = field(default_factory=dict)
    anomaly: dict[str, np.ndarray] = field(default_factory=dict)


class WeatherInputLayer(Protocol):
    """The weather-input layer of one weather product, which a later part fills in."""

    name: str

    def __call__(self, *, batch: PooledBatch) -> torch.Tensor:
        """Return the product's weather inputs for the rows of `batch`."""
        ...


@dataclass
class PooledBatch:
    """The rows of a pooled fit as tensors on one device.

    Attributes:
        site: The substation index of each row.
        day_type: The day-type index.
        half_hour: The local half-hour of day.
        band: The hour band.
        holiday: The holiday-period index, `N_PERIODS` on an ordinary day.
        tent: The tent and signed tent, shape (N, 2).
        harmonics: The day-of-year harmonics, shape (N, 4).
        smoothed_temperature: Trailing-mean temperatures, shape (N, 4).
        lighting: The darkness ramps, shape (N, 4).
        elevation: The sun's elevation in degrees.
        pv_shape: The derated unit PV shape.
        season_slot: The index into the flattened (substation, season-year) capacity table.
        solar_gate: True where the solar term applies.
        is_apparent: True where the meter reads apparent power (MVA substations).
        y: The reading in points.
        site_labels: The substation labels, in index order.
        first_season_year: The season-year of capacity slot 0.
        n_season_years: The number of season-years per substation.
    """

    site: torch.Tensor
    day_type: torch.Tensor
    half_hour: torch.Tensor
    band: torch.Tensor
    holiday: torch.Tensor
    tent: torch.Tensor
    harmonics: torch.Tensor
    smoothed_temperature: torch.Tensor
    lighting: torch.Tensor
    elevation: torch.Tensor
    pv_shape: torch.Tensor
    season_slot: torch.Tensor
    solar_gate: torch.Tensor
    is_apparent: torch.Tensor
    y: torch.Tensor
    site_labels: tuple[str, ...]
    first_season_year: int
    n_season_years: int

    @property
    def n_rows(self) -> int:
        """Return the number of rows."""
        return int(self.y.shape[0])


def build_pooled_batch(
    *,
    cache: FeatureCache,
    sites: tuple[str, ...],
    cut_end: np.datetime64,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PooledBatch:
    """Gather the pooled rows: default rows of MW substations and dark default rows of MVA ones.

    Args:
        cache: The cache.
        sites: The substations to pool; their units decide the operator.
        cut_end: Rows ending after this instant are left out.
        device: Where the tensors live.
        dtype: The floating-point type of the real-valued tensors.

    Returns:
        The batch.
    """
    columns: dict[str, list[np.ndarray]] = {key: [] for key in _BATCH_KEYS}
    years = cache.calendar.season_year
    first_year = int(years.min())
    n_years = int(years.max()) - first_year + 1
    for position, label in enumerate(sites):
        features = cache.sites[label]
        apparent = features.table.units == "MVA"
        keep = (
            features.default_rows
            & features.present
            & np.isfinite(features.y)
            & np.isfinite(features.pv_unit_shape)
            & np.isfinite(features.smoothed_temperature_c).all(axis=1)
            & (cache.end_times <= cut_end)
        )
        if apparent:
            keep &= features.dark
        rows = np.flatnonzero(keep)
        n = len(rows)
        calendar = cache.calendar
        holiday = calendar.holiday_period[rows].astype(np.int64)
        columns["site"].append(np.full(n, position, dtype=np.int64))
        columns["day_type"].append(calendar.day_type[rows].astype(np.int64))
        columns["half_hour"].append(calendar.half_hour[rows].astype(np.int64))
        columns["band"].append(calendar.band[rows].astype(np.int64))
        columns["holiday"].append(np.where(holiday < 0, N_PERIODS, holiday))
        columns["tent"].append(np.stack([calendar.tent[rows], calendar.signed_tent[rows]], axis=1))
        columns["harmonics"].append(calendar.harmonics[rows])
        columns["smoothed_temperature"].append(features.smoothed_temperature_c[rows])
        columns["lighting"].append(lighting_ramps(elevation_deg=features.elevation_deg[rows]))
        columns["elevation"].append(features.elevation_deg[rows])
        columns["pv_shape"].append(derated_unit_shape(features=features, index=rows))
        columns["season_slot"].append(
            position * n_years + (calendar.season_year[rows].astype(np.int64) - first_year)
        )
        columns["solar_gate"].append(np.full(n, not apparent))
        columns["is_apparent"].append(np.full(n, apparent))
        columns["y"].append(features.y[rows])
    joined = {key: np.concatenate(values) for key, values in columns.items()}
    real = {"tent", "harmonics", "smoothed_temperature", "lighting", "elevation", "pv_shape", "y"}

    def to_tensor(key: str) -> torch.Tensor:
        array = joined[key]
        if key in real:
            return torch.as_tensor(array, dtype=dtype, device=device)
        return torch.as_tensor(array, device=device)

    return PooledBatch(
        site_labels=sites,
        first_season_year=first_year,
        n_season_years=n_years,
        **{key: to_tensor(key) for key in _BATCH_KEYS},
    )


_BATCH_KEYS: Final[tuple[str, ...]] = (
    "site",
    "day_type",
    "half_hour",
    "band",
    "holiday",
    "tent",
    "harmonics",
    "smoothed_temperature",
    "lighting",
    "elevation",
    "pv_shape",
    "season_slot",
    "solar_gate",
    "is_apparent",
    "y",
)


class PooledDemandModel(nn.Module):
    """The pooled demand-minus-solar model, in points of effective capacity."""

    def __init__(
        self,
        *,
        config: DpConfig,
        n_sites: int,
        n_season_years: int,
        initial_level: torch.Tensor,
    ) -> None:
        """Create the parameters.

        Args:
            config: Which modules are on.
            n_sites: The number of substations.
            n_season_years: The number of season-years in the capacity table.
            initial_level: The starting level of each substation, in points.
        """
        super().__init__()
        config.check_supported()
        self.config = config
        self.n_sites = n_sites
        self.n_season_years = n_season_years
        k = max(config.shared_shapes, 1)
        self.level = nn.Parameter(initial_level.clone())
        self.shapes = nn.Parameter(0.1 * torch.randn(k, N_PROFILE))
        self.site_logits = nn.Parameter(torch.zeros(n_sites, k))
        self.deviation = nn.Parameter(torch.zeros(n_sites, N_PROFILE))
        self.holiday = nn.Parameter(torch.zeros(N_PERIODS + 1, N_BANDS))
        self.tent_coef = nn.Parameter(torch.zeros(2, N_BANDS))
        self.harmonic_coef = nn.Parameter(torch.zeros(4, N_BANDS))
        self.lighting_coef = nn.Parameter(torch.zeros(4))
        self.inertia_logits = nn.Parameter(torch.zeros(len(SMOOTHING_HOURS)))
        self.threshold = nn.Parameter(torch.full((n_sites,), HEATING_THRESHOLD_C))
        self.slope_raw = nn.Parameter(torch.full((n_sites, N_BANDS), -2.0))
        self.capacity_raw = nn.Parameter(torch.full((n_sites, n_season_years), -3.0))
        self.tan_phi = nn.Parameter(
            torch.full((n_sites,), tan_phi_from_power_factor(power_factor=INITIAL_POWER_FACTOR))
        )
        self.q0 = nn.Parameter(torch.zeros(n_sites))
        self.interaction = self._interaction_network(hidden=config.interaction_hidden)

    @staticmethod
    def _interaction_network(*, hidden: tuple[int, ...]) -> nn.Sequential | None:
        """Build the interaction network, whose last layer starts at zero."""
        if not hidden:
            return None
        layers: list[nn.Module] = []
        width = INTERACTION_INPUTS
        for size in hidden:
            layers += [nn.Linear(width, size), nn.Tanh()]
            width = size
        last = nn.Linear(width, 1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        return nn.Sequential(*layers, last)

    def capacities(self) -> torch.Tensor:
        """Return the capacity of each substation and season-year, shape (sites, years)."""
        return torch.cumsum(nn.functional.softplus(self.capacity_raw), dim=1)

    def demand(self, *, batch: PooledBatch) -> torch.Tensor:
        """Return the modelled demand of each row, in points."""
        profile = batch.day_type * HALF_HOURS_PER_DAY + batch.half_hour
        mix = torch.softmax(self.site_logits, dim=1)[batch.site]
        shared = (mix * self.shapes[:, profile].T).sum(dim=1) if self.config.shared_shapes else 0.0
        value = self.level[batch.site] + shared + self.deviation[batch.site, profile]
        value = value + self.holiday[batch.holiday, batch.band]
        value = value + (batch.tent * self.tent_coef[:, batch.band].T).sum(dim=1)
        value = value + (batch.harmonics * self.harmonic_coef[:, batch.band].T).sum(dim=1)
        value = value + batch.lighting @ self.lighting_coef
        inertia = torch.softmax(self.inertia_logits, dim=0)
        mixed = batch.smoothed_temperature @ inertia
        hinge = torch.relu(self.threshold[batch.site] - mixed) / HEATING_SCALE_C
        value = value + nn.functional.softplus(self.slope_raw[batch.site, batch.band]) * hinge
        if self.interaction is not None:
            value = (
                value + self.interaction(self._interaction_inputs(batch=batch, mixed=mixed))[:, 0]
            )
        return value

    @staticmethod
    def _interaction_inputs(*, batch: PooledBatch, mixed: torch.Tensor) -> torch.Tensor:
        """Stack the calendar and weather inputs of the interaction network."""
        angle = 2.0 * torch.pi * batch.half_hour.to(mixed.dtype) / HALF_HOURS_PER_DAY
        day_type = nn.functional.one_hot(batch.day_type, num_classes=len(DAY_TYPES)).to(mixed.dtype)
        return torch.cat(
            [
                batch.harmonics,
                batch.tent,
                torch.sin(angle)[:, None],
                torch.cos(angle)[:, None],
                day_type,
                ((mixed - HEATING_THRESHOLD_C) / HEATING_SCALE_C)[:, None],
                (batch.elevation / 60.0)[:, None],
            ],
            dim=1,
        )

    def forward(self, batch: PooledBatch) -> torch.Tensor:
        """Return the meter reading of each row, in points."""
        demand = self.demand(batch=batch)
        solar = self.capacities().reshape(-1)[batch.season_slot] * batch.pv_shape
        real = real_power(demand=demand, solar=solar, solar_gate=batch.solar_gate)
        apparent = ApparentPowerParameters(tan_phi=self.tan_phi[batch.site], q0=self.q0[batch.site])
        reading = observe(kind="identity", real=real, demand=demand)
        apparent_reading = observe(kind="apparent", real=real, demand=demand, parameters=apparent)
        return torch.where(batch.is_apparent, apparent_reading, reading)

    def penalty(self) -> torch.Tensor:
        """Return the weight decay and the prior on the heating threshold."""
        decay = (
            1e-4 * (self.shapes**2).sum()
            + 1e-3 * (self.deviation**2).sum()
            + 1e-3 * (self.holiday**2).sum()
        )
        prior = (((self.threshold - HEATING_THRESHOLD_C) / THRESHOLD_PRIOR_SCALE_C) ** 2).sum()
        if self.interaction is not None:
            decay = decay + 1e-5 * sum((p**2).sum() for p in self.interaction.parameters())
        return decay + prior


@dataclass(frozen=True)
class PooledFitResult:
    """What a pooled fit reports.

    Attributes:
        losses: The loss at every step.
        seconds: Wall-clock seconds of the optimisation loop.
        peak_memory_bytes: The peak GPU memory allocated, or 0 on the CPU.
    """

    losses: list[float]
    seconds: float
    peak_memory_bytes: int


def fit_pooled(
    *,
    model: PooledDemandModel,
    batch: PooledBatch,
    steps: int,
    learning_rate: float = 0.05,
) -> PooledFitResult:
    """Fit by full-batch Adam on the mean squared error plus the penalty.

    Args:
        model: The model, already on the batch's device.
        batch: The rows.
        steps: The number of optimiser steps.
        learning_rate: Adam's learning rate.

    Returns:
        The loss history, the wall-clock time, and the peak GPU memory.
    """
    device = batch.y.device
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    losses: list[torch.Tensor] = []
    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        residual = model(batch) - batch.y
        loss = (residual**2).mean() + model.penalty() / batch.n_rows
        loss.backward()
        optimiser.step()
        losses.append(loss.detach())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    history = [float(value) for value in torch.stack(losses).cpu()]
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return PooledFitResult(losses=history, seconds=seconds, peak_memory_bytes=peak)
