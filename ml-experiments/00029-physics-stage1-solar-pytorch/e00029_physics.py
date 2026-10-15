"""A differentiable PV model with probabilistic priors, fitted by stochastic variational inference.

The model turns global horizontal irradiance, the sun's position, and air temperature into the AC
power of one plant, as a fraction of the plant's effective capacity. The chain is an Erbs split of
global into beam and diffuse light (done by the caller, because it has no parameters), an
isotropic-sky transposition onto a plane of fitted tilt and azimuth, a ground reflection, a
temperature derate, DC capacity, and a smooth clip at the AC limit. The AC limit is the DC capacity
times the AC-to-DC ratio, which lies between 0 and 1.

Each plant has four fitted parameters: tilt, azimuth, the logarithm of the DC capacity, and a logit
of the AC-to-DC ratio. Each has a normal prior and a mean-field normal posterior. The fit maximises
an evidence lower bound whose likelihood is tempered by `TEMPERING`, because half-hourly residuals
within a day are strongly autocorrelated and an untempered posterior would be far too narrow.
Many fits run in one batch: a batch row is one plant starting from one initial orientation.
"""

from dataclasses import dataclass
from math import log
from typing import Final

import numpy as np
import pvlib
import torch  # ty: ignore[unresolved-import]
from torch.nn import functional as F  # noqa: N812  # ty: ignore[unresolved-import]

GROUND_ALBEDO: Final[float] = 0.2
TEMPERATURE_COEFFICIENT: Final[float] = 0.004
"""Fractional loss of DC power per kelvin of cell temperature above 25 degrees Celsius."""
CELL_HEATING_K_PER_W_M2: Final[float] = 0.03
"""Cell temperature above air temperature per watt per square metre of plane irradiance."""
CLIP_SHARPNESS: Final[float] = 50.0
"""The smooth clip's sharpness as a multiple of one over the AC limit."""
TEMPERING: Final[float] = 1.0 / 24.0
"""The likelihood's weight: roughly one independent draw per day of 48 half-hours, halved."""

PRIOR_MEAN: Final[tuple[float, float, float, float]] = (
    30.0,
    180.0,
    log(1.2),
    log(1.0 / 1.3 / (1.0 - 1.0 / 1.3)),
)
"""Tilt (degrees), azimuth (degrees clockwise from north), log DC capacity, logit AC-to-DC ratio."""
PRIOR_SD: Final[tuple[float, float, float, float]] = (20.0, 40.0, 0.4, 0.7)
"""The same widths as `studies.pv_fit.LOOSE_PRIORS` for tilt and azimuth, from general knowledge."""
START_TILTS_DEG: Final[tuple[float, ...]] = (15.0, 30.0, 45.0)
START_AZIMUTHS_DEG: Final[tuple[float, ...]] = (150.0, 180.0, 210.0)
HUBER_DELTA: Final[float] = 1.5
STEPS: Final[int] = 1500
LEARNING_RATE: Final[float] = 0.05
POSTERIOR_SAMPLES: Final[int] = 400


@dataclass(frozen=True)
class SkyTensors:
    """Per-row inputs of a batch of fits, each of shape (batch, rows).

    Attributes:
        ghi: Global horizontal irradiance, W/m2.
        dni: Direct normal irradiance from the Erbs split, W/m2.
        dhi: Diffuse horizontal irradiance from the Erbs split, W/m2.
        zenith: Solar zenith, radians.
        sun_azimuth: Solar azimuth, radians clockwise from north.
        temperature: Air temperature, degrees Celsius.
    """

    ghi: torch.Tensor
    dni: torch.Tensor
    dhi: torch.Tensor
    zenith: torch.Tensor
    sun_azimuth: torch.Tensor
    temperature: torch.Tensor


def erbs_split(
    *, ghi: np.ndarray, zenith_deg: np.ndarray, day_of_year: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the direct normal and diffuse horizontal irradiance of an Erbs split of `ghi`."""
    ghi = np.clip(np.nan_to_num(np.asarray(ghi, dtype=np.float64)), 0.0, None)
    split = pvlib.irradiance.erbs(ghi=ghi, zenith=zenith_deg, datetime_or_doy=day_of_year)
    return (
        np.nan_to_num(np.asarray(split["dni"], dtype=np.float64)),
        np.nan_to_num(np.asarray(split["dhi"], dtype=np.float64)),
    )


def sky_tensors(
    *,
    ghi: np.ndarray,
    zenith_deg: np.ndarray,
    sun_azimuth_deg: np.ndarray,
    day_of_year: np.ndarray,
    temperature_c: np.ndarray,
    device: torch.device,
) -> SkyTensors:
    """Build a one-row batch of sky tensors from 1-D NumPy arrays."""
    dni, dhi = erbs_split(ghi=ghi, zenith_deg=zenith_deg, day_of_year=day_of_year)

    def tensor(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(values, dtype=np.float32)[None], device=device)

    return SkyTensors(
        ghi=tensor(np.clip(np.nan_to_num(ghi), 0.0, None)),
        dni=tensor(dni),
        dhi=tensor(dhi),
        zenith=tensor(np.radians(zenith_deg)),
        sun_azimuth=tensor(np.radians(sun_azimuth_deg)),
        temperature=tensor(np.nan_to_num(temperature_c, nan=10.0)),
    )


def physical_parameters(*, u: torch.Tensor) -> dict[str, torch.Tensor]:
    """Turn standardised parameters `u` of shape (..., 4) into physical ones.

    Args:
        u: The standardised parameters, in units of the prior's standard deviation.

    Returns:
        `tilt` and `azimuth` in degrees, `dc` (DC capacity, fraction of effective capacity), `ac`
        (AC limit, the same unit), and `ratio` (DC over AC).
    """
    mean = u.new_tensor(PRIOR_MEAN)
    sd = u.new_tensor(PRIOR_SD)
    phi = mean + sd * u
    tilt = phi[..., 0].clamp(0.0, 89.0)
    dc = torch.exp(phi[..., 2])
    ac_share = torch.sigmoid(phi[..., 3])
    return {
        "tilt": tilt,
        "azimuth": phi[..., 1],
        "dc": dc,
        "ac": dc * ac_share,
        "ratio": 1 / ac_share,
    }


def soft_clip(*, power: torch.Tensor, limit: torch.Tensor) -> torch.Tensor:
    """Return a differentiable `min(power, limit)`, linear well below the limit."""
    beta = CLIP_SHARPNESS / limit
    return limit - F.softplus(beta * (limit - power)) / beta


def plant_power(*, sky: SkyTensors, params: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the AC power (fraction of effective capacity) of plants with given parameters.

    Args:
        sky: Inputs of shape (batch, rows).
        params: Physical parameters, each of shape (batch,) or (samples, batch).

    Returns:
        Power of shape (batch, rows), or (samples, batch, rows) when the parameters carry samples.
    """
    extra = params["tilt"].dim() - 1
    shape = (*params["tilt"].shape, 1)

    def p(name: str) -> torch.Tensor:
        return params[name].reshape(shape)

    def s(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape((1,) * extra + tensor.shape) if extra else tensor

    tilt = torch.deg2rad(p("tilt"))
    panel_azimuth = torch.deg2rad(p("azimuth"))
    zenith, sun_azimuth = s(sky.zenith), s(sky.sun_azimuth)
    cos_incidence = torch.cos(zenith) * torch.cos(tilt) + torch.sin(zenith) * torch.sin(
        tilt
    ) * torch.cos(sun_azimuth - panel_azimuth)
    beam = s(sky.dni) * cos_incidence.clamp(min=0.0)
    sky_diffuse = s(sky.dhi) * (1 + torch.cos(tilt)) / 2
    ground = s(sky.ghi) * GROUND_ALBEDO * (1 - torch.cos(tilt)) / 2
    poa = (beam + sky_diffuse + ground).clamp(min=0.0)
    cell_excess = s(sky.temperature) + CELL_HEATING_K_PER_W_M2 * poa - 25.0
    derate = (1 - TEMPERATURE_COEFFICIENT * cell_excess).clamp(min=0.0)
    dc_power = p("dc") * poa / 1000.0 * derate
    return soft_clip(power=dc_power, limit=p("ac"))


@dataclass(frozen=True)
class FitSummary:
    """The result of fitting a batch of plants, one entry per plant.

    Attributes:
        point: Physical parameters at the posterior mean of the standardised parameters, each of
            shape (plants,).
        draws: Physical parameters of posterior draws, each of shape (draws, plants).
        noise: The fitted noise scale, a fraction of effective capacity.
        elbo: The best start's final loss (lower is better).
    """

    point: dict[str, np.ndarray]
    draws: dict[str, np.ndarray]
    noise: np.ndarray
    elbo: np.ndarray

    def interval(self, *, name: str, level: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        """Return the central `level` posterior interval of a parameter for every plant."""
        tail = (1 - level) / 2
        return (
            np.quantile(self.draws[name], tail, axis=0),
            np.quantile(self.draws[name], 1 - tail, axis=0),
        )


def _start_means(*, device: torch.device) -> torch.Tensor:
    """Return the standardised initial means of the multi-start grid, shape (starts, 4)."""
    rows = [
        [
            (tilt - PRIOR_MEAN[0]) / PRIOR_SD[0],
            (azimuth - PRIOR_MEAN[1]) / PRIOR_SD[1],
            0.0,
            0.0,
        ]
        for tilt in START_TILTS_DEG
        for azimuth in START_AZIMUTHS_DEG
    ]
    return torch.tensor(rows, device=device, dtype=torch.float32)


def fit_plants(
    *,
    sky: SkyTensors,
    target: torch.Tensor,
    mask: torch.Tensor,
    seed: int,
    steps: int = STEPS,
) -> FitSummary:
    """Fit a batch of plants, each from every start of the orientation grid.

    Args:
        sky: Inputs of shape (plants, rows).
        target: Observed power as a fraction of effective capacity, shape (plants, rows).
        mask: True for rows the fit may use (daylight, observed, with CAMS irradiance).
        seed: The random seed of the sampling.
        steps: Optimiser steps.

    Returns:
        The fit of the best start for each plant.
    """
    device = target.device
    generator = torch.Generator(device=device).manual_seed(seed)
    plants = target.shape[0]
    starts = _start_means(device=device)
    n_starts = starts.shape[0]

    def tile(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.repeat_interleave(n_starts, dim=0)

    tiled = SkyTensors(
        *(
            tile(t)
            for t in (sky.ghi, sky.dni, sky.dhi, sky.zenith, sky.sun_azimuth, sky.temperature)
        )
    )
    y, valid = tile(target), tile(mask).float()
    mean = starts.repeat(plants, 1).clone().requires_grad_(True)
    log_sd = torch.full_like(mean, log(0.3)).requires_grad_(True)
    log_noise = torch.full((plants * n_starts,), log(0.05), device=device).requires_grad_(True)
    optimiser = torch.optim.Adam([mean, log_sd, log_noise], lr=LEARNING_RATE)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=steps, eta_min=0.002)

    def loss_terms(*, draws: int) -> torch.Tensor:
        sd = torch.exp(log_sd)
        noise = torch.randn((draws, *mean.shape), device=device, generator=generator)
        u = mean + sd * noise
        prediction = plant_power(sky=tiled, params=physical_parameters(u=u))
        scale = torch.exp(log_noise)[None, :, None]
        z = (prediction - y[None]) / scale
        robust = HUBER_DELTA**2 * (torch.sqrt(1 + (z / HUBER_DELTA) ** 2) - 1)
        nll = ((robust + log_noise[None, :, None]) * valid[None]).sum(dim=2).mean(dim=0)
        kl = 0.5 * (sd**2 + mean**2 - 1 - 2 * log_sd).sum(dim=1)
        return TEMPERING * nll + kl

    for _ in range(steps):
        optimiser.zero_grad()
        loss_terms(draws=2).sum().backward()
        optimiser.step()
        schedule.step()
    with torch.no_grad():
        final = loss_terms(draws=64).reshape(plants, n_starts)
        best = final.argmin(dim=1)
        index = torch.arange(plants, device=device) * n_starts + best
        best_mean, best_sd = mean[index], torch.exp(log_sd[index])
        draw_u = best_mean + best_sd * torch.randn(
            (POSTERIOR_SAMPLES, plants, 4), device=device, generator=generator
        )
        draws = {k: v.cpu().numpy() for k, v in physical_parameters(u=draw_u).items()}
        point = {k: v.cpu().numpy() for k, v in physical_parameters(u=best_mean).items()}
        return FitSummary(
            point=point,
            draws=draws,
            noise=torch.exp(log_noise[index]).cpu().numpy(),
            elbo=final.gather(1, best[:, None])[:, 0].cpu().numpy(),
        )
