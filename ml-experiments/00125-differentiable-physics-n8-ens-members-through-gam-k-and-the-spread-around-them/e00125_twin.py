"""A synthetic twin of a site's tuning-month runs, with known weather and model errors.

The truth weather W is ERA5 temperature and CAMS irradiance. Member weather is `W_m = W + e_m`, with
`e_m` a zero-mean AR(1) series (correlation time 12 hours) whose standard deviation grows with the
lead from a known `σ_w`; the truth is a fresh draw `W + e_0`, so truth and members are exchangeable
by construction. The twin's analysis proxy is the truth plus a temperature error of 0.1 K. Fake
demand is GAM-K's forecast on the truth draw plus AR(1) noise `η` with a known standard deviation.

The twin therefore confirms the arithmetic and the code of the arms. It does not show that the real
ECMWF ensemble behaves like this.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final

import e00125_arms as arms
import e00125_cache as cache_module
import e00125_common as common
import e00125_ens as ens
import numpy as np
from studies.dp_ens import member_means
from studies.dp_ens_weather import Calibration, EnsSiteRun, RunWindow
from studies.dp_features import FeatureCache
from studies.dp_gam import GamFit

SIGMA_T_K: Final[tuple[float, float]] = (0.6, 2.0)
"""Temperature error sd at 12 hours and at 168 hours, in kelvin."""
SIGMA_G_RATIO: Final[tuple[float, float]] = (0.04, 0.20)
"""Irradiance error sd as a share of irradiance, at 12 hours and at 168 hours."""
CORRELATION_HOURS: Final[float] = 12.0
PROXY_ERROR_K: Final[float] = 0.1
N_TWIN_MEMBERS: Final[int] = 51
N_MONTE_CARLO: Final[int] = 500
MONTE_CARLO_STRIDE: Final[int] = 6
"""One run in this many gets the 500-draw reference."""


@dataclass(frozen=True)
class Twin:
    """The twin of one site.

    Attributes:
        runs: The twin's runs, with the same rows as the real cache's tuning runs.
        reference_variance: Per MC run, the variance over 500 draws of GAM-K on the weather, at
            each kept row.
        reference_variance_half: The same with the weather error halved.
        reference_run: The index into `runs` that each MC run belongs to.
        eta_sd: The injected standard deviation of the demand noise `η` in points.
    """

    runs: list[cache_module.RunCache]
    reference_variance: list[np.ndarray]
    reference_variance_half: list[np.ndarray]
    reference_run: list[int]
    eta_sd: float


def error_sd(*, lead_hours: np.ndarray, ends: tuple[float, float]) -> np.ndarray:
    """Return the error sd at each lead: constant to 12 hours, linear to 168 hours, then flat."""
    return np.interp(lead_hours, [12.0, 168.0], list(ends))


def ar1(*, rng: np.random.Generator, shape: tuple[int, int], sd: np.ndarray) -> np.ndarray:
    """Return zero-mean AR(1) series with the marginal sd at each step.

    Args:
        rng: The random number generator.
        shape: (series, steps).
        sd: The marginal standard deviation at each step.

    Returns:
        Shape `shape`.
    """
    phi = float(np.exp(-0.5 / CORRELATION_HOURS))
    noise = rng.standard_normal(shape)
    out = np.empty(shape)
    out[:, 0] = noise[:, 0]
    for step in range(1, shape[1]):
        out[:, step] = phi * out[:, step - 1] + np.sqrt(1.0 - phi**2) * noise[:, step]
    return out * sd[None, :]


def gam_on_weather(
    *,
    cache: FeatureCache,
    setup: ens.SiteSetup,
    fits: Sequence[GamFit],
    index: int,
    temperature_c: np.ndarray,
    shortwave_w_m2: np.ndarray,
) -> np.ndarray:
    """Return GAM-K's forecast of a run for each row of weather, with the calibration left alone.

    Args:
        cache: The feature cache.
        setup: The site's setup, whose calibration is the identity.
        fits: The monthly fits.
        index: The run's index.
        temperature_c: Shape (draws, window).
        shortwave_w_m2: Shape (draws, window).

    Returns:
        Shape (draws, window), in points.
    """
    real = setup.runs[index]
    twin = EnsSiteRun(
        origin_position=real.origin_position,
        temperature_c=temperature_c,
        shortwave_w_m2=shortwave_w_m2,
    )
    return member_means(
        cache=cache,
        context=setup.context,
        run=twin,
        origin=setup.origins[index],
        fits=fits,
        members=list(range(len(temperature_c))),
    ).means


def build_twin(
    *, site: str, seed: int, member_scale: float = 1.0, max_runs: int | None = None
) -> Twin:
    """Build the twin of a site's tuning runs.

    Args:
        site: The substation's label.
        seed: The base seed; run `i` uses `seed + i`.
        member_scale: Multiplies the members' weather error (the truth keeps its full error), 0.5
            for the halved-spread twin of gate test 2(e).
        max_runs: Build only this many runs, for a quick test.

    Returns:
        The twin.
    """
    cache = common.load_cache()
    setup = ens.load_setup(cache=cache, site=site).with_calibration(
        calibration=Calibration.identity()
    )
    fits = arms.load_arm_fits(arm=arms.ARMS["gamk"], site=site)
    real = cache_module.read_cache(site=site, period="tuning")
    if max_runs is not None:
        real = real[:max_runs]
    features = cache.sites[site]
    eta_sd, eta_phi = _noise_model(runs=real)
    runs, reference, reference_half, reference_run = [], [], [], []
    for position, source in enumerate(real):
        rng = np.random.default_rng(seed + source.run)
        window = RunWindow.build(cache=cache, context=setup.context, run=setup.runs[source.run])
        target = np.arange(window.window.start, window.window.stop)
        valid = cache.end_times[target]
        keep = np.isin(valid, source.valid_time)
        lead = np.arange(1, len(target) + 1) / 2.0
        sd_t = error_sd(lead_hours=lead, ends=SIGMA_T_K)
        sd_g = error_sd(lead_hours=lead, ends=SIGMA_G_RATIO)
        truth_t = features.temperature_c[target]
        truth_g = features.ghi_w_m2[target]

        def draws(*, n: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
            temperature = truth_t[None, :] + scale * ar1(rng=rng, shape=(n, len(target)), sd=sd_t)  # noqa: B023
            ratio = scale * ar1(rng=rng, shape=(n, len(target)), sd=sd_g)  # noqa: B023
            return temperature, np.maximum(truth_g[None, :] * (1.0 + ratio), 0.0)  # noqa: B023

        member_t, member_g = draws(n=N_TWIN_MEMBERS, scale=member_scale)
        truth_draw_t, truth_draw_g = draws(n=1, scale=1.0)
        proxy_t = truth_draw_t + PROXY_ERROR_K * rng.standard_normal(truth_draw_t.shape)
        stacked = gam_on_weather(
            cache=cache,
            setup=setup,
            fits=fits,
            index=source.run,
            temperature_c=np.concatenate([member_t, truth_draw_t, proxy_t]),
            shortwave_w_m2=np.concatenate([member_g, truth_draw_g, truth_draw_g]),
        )
        means, truth_mean, proxy_mean = (
            stacked[:N_TWIN_MEMBERS],
            stacked[N_TWIN_MEMBERS],
            stacked[-1],
        )
        eta = ar1_scalar(rng=rng, steps=len(target), sd=eta_sd, phi=eta_phi)
        actual = truth_mean + eta
        runs.append(
            replace(
                source,
                members=list(range(N_TWIN_MEMBERS)),
                corrected=means[:, keep],
                proxy=proxy_mean[keep],
                level=np.zeros(int(keep.sum())),
                anomaly=np.zeros(int(keep.sum())),
                actual=actual[keep],
                switching_mask=np.zeros(int(keep.sum()), dtype=bool),
            )
        )
        if position % MONTE_CARLO_STRIDE == 0:
            for scale, store in ((1.0, reference), (0.5, reference_half)):
                mc_t, mc_g = draws(n=N_MONTE_CARLO, scale=scale)
                mc = gam_on_weather(
                    cache=cache,
                    setup=setup,
                    fits=fits,
                    index=source.run,
                    temperature_c=mc_t,
                    shortwave_w_m2=mc_g,
                )
                store.append(mc.var(axis=0, ddof=1)[keep])
            reference_run.append(position)
    return Twin(
        runs=runs,
        reference_variance=reference,
        reference_variance_half=reference_half,
        reference_run=reference_run,
        eta_sd=eta_sd,
    )


def _noise_model(*, runs: Sequence[cache_module.RunCache]) -> tuple[float, float]:
    """Return the sd and lag-1 correlation of the real proxy-driven residuals outside switching."""
    residuals, pairs = [], []
    for run in runs:
        keep = ~run.switching_mask & np.isfinite(run.proxy)
        residual = np.where(keep, run.actual - run.proxy, np.nan)
        residuals.append(residual[keep])
        pairs.append(np.column_stack([residual[:-1], residual[1:]]))
    pooled = np.concatenate(residuals)
    both = np.concatenate(pairs)
    both = both[np.isfinite(both).all(axis=1)]
    return float(pooled.std()), float(np.corrcoef(both[:, 0], both[:, 1])[0, 1])


def ar1_scalar(*, rng: np.random.Generator, steps: int, sd: float, phi: float) -> np.ndarray:
    """Return one AR(1) series with a constant marginal sd and lag-1 correlation `phi`."""
    noise = rng.standard_normal(steps)
    out = np.empty(steps)
    out[0] = noise[0]
    for step in range(1, steps):
        out[step] = phi * out[step - 1] + np.sqrt(1.0 - phi**2) * noise[step]
    return out * sd
