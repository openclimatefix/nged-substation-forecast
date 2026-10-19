"""The pooled PyTorch configuration of the differentiable-physics (DP) series, and its switches.

`studies.dp_gam` fits one convex model per substation. This module holds the model that Parts 3 to
6 grow from: one model for many substations, fitted by gradient descent, with shared daily shapes,
a shared calendar and weather module, an optional interaction network, a solar term per substation
and season-year, and a per-row observation operator (`studies.dp_operator`).

**Every later module is a switch in `DpConfig`.** The switches for modules that are not built
(the smooth clip, the unmetered generator node, and drift in the pooled model) raise
`NotImplementedError` when turned on, so a later part replaces one raise with code and changes no
caller. `ResidualStates` and `WeatherInputLayer` are typed stubs for the per-site
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

**Site-level parameters and their prior.** Six families of parameters belong to a substation: the
level, the style loadings (`site_logits`), the deviation, the heating threshold, the heating slope,
and the capacity increments. Each family has a population mean and a population scale, and a
substation's value is drawn around the population mean. Three site priors exist. `decay` is the
Part 2 configuration (fixed weight decay and the threshold prior). `map` fixes the population scales
and finds the maximum a posteriori fit. `variational` gives every site-level value a mean-field
Gaussian posterior, learns the population scales through the evidence lower bound, and forecasts
with the posterior mean. The shared network weights are point estimates with weight decay in all
three.
"""

import time
from dataclasses import dataclass, field
from typing import Final, Literal, Protocol

import numpy as np
import torch
from torch import nn

from studies.dp_calendar import (
    DAY_TYPES,
    HALF_HOURS_PER_DAY,
    HEATING_THRESHOLD_C,
    HOLIDAY_PERIODS,
    SMOOTHING_HOURS,
    lighting_ramps,
)
from studies.dp_features import FeatureCache
from studies.dp_gam import HEATING_SCALE_C, N_BANDS, DriftType, derated_unit_shape
from studies.dp_operator import (
    ApparentPowerParameters,
    observe,
    real_power,
    tan_phi_from_power_factor,
)

N_PROFILE: Final[int] = len(DAY_TYPES) * HALF_HOURS_PER_DAY
"""Entries of a daily profile: one per day type and local half-hour."""
N_PERIODS: Final[int] = len(HOLIDAY_PERIODS)
"""The number of named holiday periods (one more embedding row stands for an ordinary day)."""
THRESHOLD_PRIOR_SCALE_C: Final[float] = 2.0
"""The prior on a substation's heating threshold is a normal around 15.5 degrees, this wide."""
MIN_TRAINED_SLOT_ROWS: Final[int] = 1440
"""A site-year with fewer training rows (30 days of half-hours) counts as untrained.

A cut on 1 July leaves a few dark half-hours in the next season-year, which must not count as a
trained year: its capacity increment and centring mean would rest on them.
"""
INITIAL_POWER_FACTOR: Final[float] = 0.97
"""The power factor that the MVA substations' `tan(phi)` starts from."""
SITE_FAMILIES: Final[tuple[str, ...]] = (
    "level",
    "site_logits",
    "deviation",
    "threshold",
    "slope_raw",
    "capacity_raw",
)
"""The site-level parameter families; each has a population mean and a population scale."""
POPULATION_FAMILIES: Final[tuple[str, ...]] = (*SITE_FAMILIES, "capacity_first")
"""The families with a population distribution.

`capacity_first` is the first trained season-year's capacity increment of a substation, which is
its whole capacity; `capacity_raw` then holds the yearly growth increments that follow. They are
separate families because the first increment is about 10 times the growth increments.
"""
MEAN_LEARNED_FAMILIES: Final[tuple[str, ...]] = (
    "level",
    "threshold",
    "slope_raw",
    "capacity_raw",
    "capacity_first",
)
"""Families whose population mean is learned; the loadings and the deviation centre on zero."""
SitePrior = Literal["decay", "map", "variational"]
INITIAL_POSTERIOR_LOG_SD: Final[float] = -3.0
"""A site-level posterior starts this tight, so the first draws sit near the point fit."""
INTERACTION_INPUTS: Final[int] = 4 + 2 + 2 + len(DAY_TYPES) + 1
"""Harmonics, tent, half-hour sine and cosine, day-type one-hot, and the mixed temperature.

The sun's elevation is not an input: the lighting ramps already carry how dark it is, and an
elevation input would let the network learn the fleet-average solar curve and compete with the
solar term.
"""


@dataclass(frozen=True)
class PopulationScales:
    """The population scale of each site-level family: the spread of sites around their mean.

    The values are standard deviations in each family's own units (points for the level and
    the deviation, degrees for the threshold, and the softplus or softmax inputs for the rest).

    Attributes:
        level: The scale of the substations' levels.
        site_logits: The scale of the style loadings.
        deviation: The scale of the per-site deviation from the shared shapes.
        threshold: The scale of the heating thresholds.
        slope_raw: The scale of the heating slopes before the softplus.
        capacity_raw: The scale of the growth increments after a substation's first year.
        capacity_first: The scale of the first trained season-year's capacity increment.
    """

    level: float = 5.0
    site_logits: float = 1.0
    deviation: float = 0.3
    threshold: float = THRESHOLD_PRIOR_SCALE_C
    slope_raw: float = 1.0
    capacity_raw: float = 1.0
    capacity_first: float = 8.0

    def as_dict(self) -> dict[str, float]:
        """Return the scales keyed by family name."""
        return {name: float(getattr(self, name)) for name in POPULATION_FAMILIES}


@dataclass(frozen=True)
class DpConfig:
    """Switches for every module of the DP model.

    Attributes:
        shared_shapes: The number of shared daily shapes `K`; 0 gives each substation its own
            profile only.
        interaction_hidden: The hidden-layer widths of the interaction network; empty means off.
        variational: Mean-field variational posterior over the site-level parameters, with
            population scales learned through the evidence lower bound.
        population_scales: The population scales. With `variational` on they are the starting
            values; with it off, giving them selects the MAP fit with these scales held fixed.
        drift: Drift form of the pooled model (only `none` is built).
        clip: The smooth export-limit clip of the solar term (not built).
        generator_node: The unmetered generator node (not built).
    """

    shared_shapes: int = 4
    interaction_hidden: tuple[int, ...] = ()
    variational: bool = False
    population_scales: PopulationScales | None = None
    drift: DriftType = "none"
    clip: bool = False
    generator_node: bool = False

    @property
    def site_prior(self) -> SitePrior:
        """Return which prior the site-level parameters have."""
        if self.variational:
            return "variational"
        return "map" if self.population_scales is not None else "decay"

    def check_supported(self) -> None:
        """Raise if a switch that is not built is on.

        Raises:
            NotImplementedError: Naming the first switch that is on but not built.
        """
        for name, value in (
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
        quarter: The calendar quarter of each row (0 to 3).
        row: The row of the feature cache's time grid.
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
    quarter: torch.Tensor
    row: torch.Tensor
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
    after: np.datetime64 | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    all_present_rows: bool = False,
) -> PooledBatch:
    """Gather the pooled rows: default rows of MW substations and dark default rows of MVA ones.

    Args:
        cache: The cache.
        sites: The substations to pool; their units decide the operator.
        cut_end: Rows ending after this instant are left out.
        after: If given, rows ending at or before this instant are left out too, which gathers the
            rows of a forecast year.
        device: Where the tensors live.
        dtype: The floating-point type of the real-valued tensors.
        all_present_rows: Keep every row that has a reading, including those inside a switching
            mask or the lockdown window, which scoring wants beside the default rows.

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
            (features.present if all_present_rows else features.default_rows & features.present)
            & np.isfinite(features.y)
            & np.isfinite(features.pv_unit_shape)
            & np.isfinite(features.smoothed_temperature_c).all(axis=1)
            & (cache.end_times <= cut_end)
        )
        if after is not None:
            keep &= cache.end_times > after
        if apparent:
            keep &= features.dark
        rows = np.flatnonzero(keep)
        if len(rows) and cache.end_times[rows].max() > cut_end:
            msg = f"{label} has a pooled row that ends after the cut: lookahead"
            raise ValueError(msg)
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
        columns["quarter"].append(calendar.quarter[rows].astype(np.int64))
        columns["row"].append(rows.astype(np.int64))
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
    "quarter",
    "row",
)


class PooledDemandModel(nn.Module):
    """The pooled demand-minus-solar model, in points of effective capacity."""

    trained: torch.Tensor
    """1 for each substation and season-year that has training rows, else 0 (a buffer)."""
    centre_means: torch.Tensor
    """The mean of the centred terms over each substation and season-year, from training."""
    site_active: torch.Tensor
    """1 for each substation that has training rows, else 0 (a buffer)."""
    solar_sites: torch.Tensor
    """1 for each substation that has training rows with the solar term on, else 0 (a buffer)."""
    capacity_prior: torch.Tensor
    """A site-specific centre for the first capacity increment, before the softplus; `NaN` for none.

    The population mean centres a site with `NaN`. The values are inputs, not fitted parameters:
    they carry a prior capacity, such as an earlier vintage's, into the fit.
    """

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
        self.register_buffer("trained", torch.ones(n_sites, n_season_years))
        self.register_buffer("centre_means", torch.zeros(n_sites * n_season_years))
        self.register_buffer("site_active", torch.ones(n_sites))
        self.register_buffer("solar_sites", torch.ones(n_sites))
        self.register_buffer("capacity_prior", torch.full((n_sites,), float("nan")))
        self._build_site_prior(initial_level=initial_level)

    def _build_site_prior(self, *, initial_level: torch.Tensor) -> None:
        """Create the population parameters and the posterior spreads of the site-level families."""
        mode = self.config.site_prior
        self.log_noise = nn.Parameter(torch.zeros(()), requires_grad=mode != "decay")
        scales = (self.config.population_scales or PopulationScales()).as_dict()
        self.population_mean = nn.ParameterDict()
        self.population_log_scale = nn.ParameterDict()
        self.posterior_log_sd = nn.ParameterDict()
        for name in POPULATION_FAMILIES:
            site_values = getattr(self, "capacity_raw" if name == "capacity_first" else name)
            if name in MEAN_LEARNED_FAMILIES:
                start = initial_level.mean() if name == "level" else site_values.detach().mean()
                self.population_mean[name] = nn.Parameter(
                    start.detach().clone().reshape(()), requires_grad=mode != "decay"
                )
            self.population_log_scale[name] = nn.Parameter(
                torch.tensor(float(np.log(scales[name]))), requires_grad=mode == "variational"
            )
            if name in SITE_FAMILIES:
                self.posterior_log_sd[name] = nn.Parameter(
                    torch.full_like(site_values, INITIAL_POSTERIOR_LOG_SD),
                    requires_grad=mode == "variational",
                )

    def _site(self, name: str) -> torch.Tensor:
        """Return a site-level family: the mean, or in training a draw from its posterior."""
        mean: torch.Tensor = getattr(self, name)
        if self.config.variational and self.training:
            return mean + torch.exp(self.posterior_log_sd[name]) * torch.randn_like(mean)
        return mean

    def site_prior_term(self) -> torch.Tensor:
        """Return the prior on the site-level parameters, in nats.

        With `variational` on this is the Kullback-Leibler divergence of the posterior from the
        population distribution, summed over the substations and season-years that have training
        rows (and, for the capacity, solar rows). With the MAP fit it is the Gaussian log prior at
        the point values. The heating threshold's population mean carries a normal prior of
        15.5 +- 2 degrees in both.
        """
        term = (
            ((self.population_mean["threshold"] - HEATING_THRESHOLD_C) / THRESHOLD_PRIOR_SCALE_C)
            ** 2
        ) / 2
        first, growth = self.capacity_masks()
        for name in SITE_FAMILIES:
            mean: torch.Tensor = getattr(self, name)
            log_sd = self.posterior_log_sd[name]
            if name == "capacity_raw":
                term = term + self._family_term(
                    name="capacity_first", mean=mean, log_sd=log_sd, active=first
                )
                term = term + self._family_term(
                    name="capacity_raw", mean=mean, log_sd=log_sd, active=growth
                )
            else:
                active = self.site_active.to(mean.dtype)
                if mean.dim() == 2:
                    active = active[:, None].expand_as(mean)
                term = term + self._family_term(name=name, mean=mean, log_sd=log_sd, active=active)
        return term

    def _family_term(
        self, *, name: str, mean: torch.Tensor, log_sd: torch.Tensor, active: torch.Tensor
    ) -> torch.Tensor:
        """Return one family's prior term over the entries where `active` is 1."""
        centre = self.population_mean.get(name, 0.0)
        if name == "capacity_first":
            own = self.capacity_prior.to(mean.dtype)[:, None]
            centre = torch.where(torch.isnan(own), centre, own)
        log_scale = self.population_log_scale[name]
        scale_sq = torch.exp(2.0 * log_scale)
        squared = (mean - centre) ** 2
        if self.config.variational:
            kl = log_scale - log_sd + (torch.exp(2.0 * log_sd) + squared) / (2.0 * scale_sq) - 0.5
            return (kl * active).sum()
        return (squared / (2.0 * scale_sq) * active).sum()

    def capacity_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masks of the first and the later capacity increments of the solar substations.

        The first trained season-year's increment is a substation's whole capacity. Only
        substations with solar training rows count, so the MVA substations, whose capacity gets no
        gradient, do not pull the population towards their starting value.
        """
        trained = self.trained
        first = ((torch.cumsum(trained, dim=1) == 1) & (trained > 0)).to(trained.dtype)
        solar = self.solar_sites.to(trained.dtype)[:, None]
        return first * solar, (trained - first) * solar

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
        return torch.cumsum(
            nn.functional.softplus(self._site("capacity_raw")) * self.trained, dim=1
        )

    def finalise_centres(self, *, batch: PooledBatch) -> None:
        """Store the final centring means of the training rows, and carry them to later years.

        One forward pass in training mode with the final parameters refreshes the means of every
        site-year that has training rows. A site-year without training rows, which is a forecast
        year, then takes the mean of the site's last trained season-year, so a forecast row is
        shifted by a training mean and never by a mean of the forecast rows.

        Args:
            batch: The training rows.
        """
        was_training = self.training
        self.train()
        with torch.no_grad():
            self(batch)
            self.mark_trained_years(batch=batch)
            means = self.centre_means.reshape(self.n_sites, self.n_season_years)
            seen = torch.zeros(self.n_sites, dtype=torch.bool, device=means.device)
            last = torch.zeros(self.n_sites, dtype=means.dtype, device=means.device)
            for year in range(self.n_season_years):
                trained = self.trained[:, year] > 0
                last = torch.where(trained, means[:, year], last)
                seen = seen | trained
                means[:, year] = torch.where(trained | ~seen, means[:, year], last)
        self.train(was_training)

    def mark_trained_years(
        self, *, batch: PooledBatch, min_rows: int = MIN_TRAINED_SLOT_ROWS
    ) -> None:
        """Allow a capacity increment only in the season-years with enough training rows.

        An untrained season-year then adds nothing, so its capacity is the last trained
        season-year's: the vintage rule's hold, which a forecast of a later season-year needs.
        A substation with no training rows is marked inactive, so the site prior leaves it alone.

        Args:
            batch: The training rows.
            min_rows: The fewest rows of a site-year that count as trained.
        """
        counts = torch.bincount(batch.season_slot, minlength=self.n_sites * self.n_season_years)
        trained = (counts >= min_rows).to(self.trained.dtype)
        self.trained.copy_(trained.reshape(self.n_sites, self.n_season_years))
        per_site = torch.bincount(batch.site, minlength=self.n_sites)
        self.site_active.copy_((per_site > 0).to(self.site_active.dtype))
        solar = torch.bincount(batch.site[batch.solar_gate], minlength=self.n_sites)
        self.solar_sites.copy_((solar > 0).to(self.solar_sites.dtype))

    def start_from_batch(self, *, batch: PooledBatch) -> None:
        """Start the levels from the training rows only.

        Each substation's level starts at the mean reading of its own training rows, and the
        population level starts at the mean over substations that have rows. A substation with no
        rows starts at the population level. Nothing outside `batch` reaches a starting value, so
        a held-out substation's readings cannot enter the population through them.

        Args:
            batch: The training rows.
        """
        counts = torch.bincount(batch.site, minlength=self.n_sites).to(batch.y.dtype)
        sums = torch.zeros(self.n_sites, dtype=batch.y.dtype, device=batch.y.device)
        means = sums.index_add(0, batch.site, batch.y) / counts.clamp(min=1)
        has_rows = counts > 0
        pooled = means[has_rows].mean() if bool(has_rows.any()) else means.mean()
        with torch.no_grad():
            self.level.copy_(torch.where(has_rows, means, pooled))
            self.population_mean["level"].copy_(pooled)

    def calendar_terms(self, *, batch: PooledBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the shared calendar, lighting, and interaction terms before centring.

        Args:
            batch: The rows.

        Returns:
            The uncentred terms of each row, and the mixed smoothed temperature.
        """
        calendar = self.holiday[batch.holiday, batch.band]
        calendar = calendar + (batch.tent * self.tent_coef[:, batch.band].T).sum(dim=1)
        calendar = calendar + (batch.harmonics * self.harmonic_coef[:, batch.band].T).sum(dim=1)
        calendar = calendar + batch.lighting @ self.lighting_coef
        inertia = torch.softmax(self.inertia_logits, dim=0)
        mixed = batch.smoothed_temperature @ inertia
        if self.interaction is not None:
            calendar = (
                calendar
                + self.interaction(self._interaction_inputs(batch=batch, mixed=mixed))[:, 0]
            )
        return calendar, mixed

    def demand(self, *, batch: PooledBatch) -> torch.Tensor:
        """Return the modelled demand of each row, in points."""
        profile = batch.day_type * HALF_HOURS_PER_DAY + batch.half_hour
        level, deviation = self._site("level"), self._site("deviation")
        threshold, slope = self._site("threshold"), self._site("slope_raw")
        mix = torch.softmax(self._site("site_logits"), dim=1)[batch.site]
        shared = (mix * self.shapes[:, profile].T).sum(dim=1) if self.config.shared_shapes else 0.0
        value = level[batch.site] + shared + deviation[batch.site, profile]
        calendar, mixed = self.calendar_terms(batch=batch)
        hinge = torch.relu(threshold[batch.site] - mixed) / HEATING_SCALE_C
        value = value + nn.functional.softplus(slope[batch.site, batch.band]) * hinge
        return value + self._centred(values=calendar, slot=batch.season_slot)

    def _centred(self, *, values: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        """Subtract the mean of `values` over each substation and season-year.

        The shared calendar, lighting, and interaction terms are centred per site-year, so that
        neither a level nor a drift enters them: those belong to the substation's level and, in
        Part 4, to its drift. In training mode the means come from the batch itself and are stored;
        in evaluation mode the stored means are used, which are the training rows' and never a
        forecast row's.
        """
        n_slots = self.n_sites * self.n_season_years
        if not self.training:
            return values - self.centre_means[slot]
        counts = torch.bincount(slot, minlength=n_slots).clamp(min=1).to(values.dtype)
        sums = torch.zeros(n_slots, dtype=values.dtype, device=values.device)
        means = sums.index_add(0, slot, values) / counts
        with torch.no_grad():
            self.centre_means.copy_(means.detach())
        return values - means[slot]

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
        """Return the weight decay, and in the `decay` prior the deviation and threshold priors."""
        decay = 1e-4 * (self.shapes**2).sum() + 1e-3 * (self.holiday**2).sum()
        if self.interaction is not None:
            decay = decay + 1e-5 * sum((p**2).sum() for p in self.interaction.parameters())
        if self.config.site_prior != "decay":
            return decay
        prior = (((self.threshold - HEATING_THRESHOLD_C) / THRESHOLD_PRIOR_SCALE_C) ** 2).sum()
        return decay + 1e-3 * (self.deviation**2).sum() + prior

    def loss(self, *, batch: PooledBatch) -> torch.Tensor:
        """Return the training loss of `batch`.

        The `decay` prior gives the mean squared error plus the penalty per row. The `map` and
        `variational` priors give the negative evidence lower bound per row, with a learned
        Gaussian noise scale: the squared error scaled by twice the noise variance, the log noise
        scale, and the weight decay and the site prior's nats per row. The weight decay is not
        scaled by the noise variance, so it has the same strength in all three site priors.
        """
        residual = self(batch) - batch.y
        penalty = self.penalty()
        n_rows = batch.n_rows
        if self.config.site_prior == "decay":
            return (residual**2).mean() + penalty / n_rows
        noise_variance = torch.exp(2.0 * self.log_noise)
        fit = (residual**2).mean() / (2.0 * noise_variance)
        return fit + self.log_noise + (penalty + self.site_prior_term()) / n_rows


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
    final_rate_fraction: float = 1.0,
) -> PooledFitResult:
    """Fit by full-batch Adam on the model's loss (`PooledDemandModel.loss`).

    The levels start from the training rows' own means (`start_from_batch`), whatever
    `initial_level` the model was built with. Adam is not deterministic on the GPU, because
    `index_add` adds in an unspecified order; the tests that need exact agreement run on the CPU.

    Args:
        model: The model, already on the batch's device.
        batch: The rows.
        steps: The number of optimiser steps.
        learning_rate: Adam's starting learning rate.
        final_rate_fraction: The fraction of `learning_rate` left at the last step; the rate
            follows a cosine from 1 to this fraction, and 1.0 keeps it constant.

    Returns:
        The loss history, the wall-clock time, and the peak GPU memory.
    """
    device = batch.y.device
    model.to(device=device, dtype=batch.y.dtype)
    model.mark_trained_years(batch=batch)
    model.start_from_batch(batch=batch)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(steps, 1), eta_min=learning_rate * final_rate_fraction
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    losses: list[torch.Tensor] = []
    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        loss = model.loss(batch=batch)
        loss.backward()
        optimiser.step()
        schedule.step()
        losses.append(loss.detach())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    model.finalise_centres(batch=batch)
    history = [float(value) for value in torch.stack(losses).cpu()]
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return PooledFitResult(losses=history, seconds=seconds, peak_memory_bytes=peak)
