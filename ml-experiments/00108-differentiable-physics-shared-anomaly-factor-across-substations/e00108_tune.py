"""Tune the factor on the tuning year (2023-07-01 to 2024-06-30) and freeze it.

Everything here reads the tuning year's out-of-sample residuals only:

1. each decision site's loading `beta_s`: the Huber slope of its scaled residual on its
   leave-one-out factor with a fitted intercept, shrunk halfway to the mean over the five decision
   sites (D9 has no tuning-year fit and takes the mean);
2. the local filter's hyperparameters, fitted by marginal likelihood on `e_s = r_s - beta_s
   sigma_{s,b} F_{-s}`, one set per site under a prior shared across sites (D9 takes the mean of
   the five sites' unconstrained values), exactly as experiment 106 fitted `a0`;
3. the shared filter's hyperparameters: one set for all sites, fitted by marginal likelihood on
   the five decision sites' factors together under the same prior's starting point.

A placebo draw repeats steps 1 and 3 on its shifted factor and keeps the real local set.

    uv run python ml-experiments/00108-*/e00108_tune.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00108_common as common
import e00108_factor as factor_lib
import e00108_pipeline as pipeline
import numpy as np
from scipy.optimize import minimize
from studies.dp_assimilation import (
    BOUNDS,
    PRIOR_SPREAD,
    PRIOR_START,
    FilterHyperparameters,
    fit_filter_hyperparameters,
    site_log_likelihood,
)
from studies.dp_features import FeatureCache

TUNING_PATH: Final[Path] = common.WORK_DIR / "tuning_real.json"


@dataclass(frozen=True)
class Tuning:
    """The frozen factor settings.

    Attributes:
        raw_beta: Each decision site's unshrunk Huber loading.
        beta: Each site's shrunk loading.
        local_hyper: Each site's local filter hyperparameters.
        shared_hyper: The shared filter's hyperparameters.
    """

    raw_beta: dict[str, float]
    beta: dict[str, float]
    local_hyper: dict[str, FilterHyperparameters]
    shared_hyper: FilterHyperparameters

    def to_json(self) -> dict:
        """Return the settings as JSON-ready values."""
        return {
            "raw_beta": self.raw_beta,
            "beta": self.beta,
            "local_hyper": {site: h.as_array().tolist() for site, h in self.local_hyper.items()},
            "shared_hyper": self.shared_hyper.as_array().tolist(),
        }

    @staticmethod
    def from_json(*, stored: dict) -> Tuning:
        """Rebuild the settings from `to_json`'s output."""
        return Tuning(
            raw_beta=stored["raw_beta"],
            beta=stored["beta"],
            local_hyper={
                site: FilterHyperparameters.from_array(values=np.array(values))
                for site, values in stored["local_hyper"].items()
            },
            shared_hyper=FilterHyperparameters.from_array(values=np.array(stored["shared_hyper"])),
        )


def tuning_window(*, cache: FeatureCache) -> slice:
    """Return the grid positions of the tuning year."""
    cut = common.cut_of(year=common.TUNING_YEAR)
    return factor_lib.window_positions(cache=cache, after=cut, through=common.year_end(cut=cut))


def fit_betas(
    *, cache: FeatureCache, panel: factor_lib.Panel, factor: dict[str, np.ndarray]
) -> tuple[dict[str, float], dict[str, float]]:
    """Return the raw and the shrunk loadings, fitted on the tuning year."""
    window = tuning_window(cache=cache)
    raw = {
        site: factor_lib.huber_loading(
            scaled=panel.scaled[site][window], factor=factor[site][window]
        )
        for site in common.DECISION_SITES
    }
    return raw, factor_lib.shrink_loadings(raw=raw, sites=common.DECISION_SITES)


def fit_shared_hyper(
    *, cache: FeatureCache, factor: dict[str, np.ndarray]
) -> FilterHyperparameters:
    """Fit one set of hyperparameters for the shared filter on the five decision factors."""
    window = tuning_window(cache=cache)
    streams = [
        factor_lib.keep_window(stream=factor[site], window=window) for site in common.DECISION_SITES
    ]
    band = cache.calendar.band
    prior_mean = np.array(PRIOR_START)
    prior_spread = np.array(PRIOR_SPREAD)

    def negative_posterior(values: np.ndarray) -> float:
        hyper = FilterHyperparameters.from_array(values=values)
        likelihood = sum(
            site_log_likelihood(residual=stream, band=band, hyper=hyper) for stream in streams
        )
        prior = -0.5 * float((((values - prior_mean) / prior_spread) ** 2).sum())
        return -(likelihood + prior)

    result = minimize(negative_posterior, x0=prior_mean, method="L-BFGS-B", bounds=BOUNDS)
    if not result.success:
        msg = f"the shared filter's hyperparameter fit did not converge: {result.message}"
        raise ValueError(msg)
    return FilterHyperparameters.from_array(values=result.x)


def fit_local_hyper(
    *,
    cache: FeatureCache,
    panel: factor_lib.Panel,
    factor: dict[str, np.ndarray],
    beta: dict[str, float],
) -> dict[str, FilterHyperparameters]:
    """Fit the local filters' hyperparameters on `e_s` over the tuning year."""
    window = tuning_window(cache=cache)
    streams = {
        site: factor_lib.keep_window(
            stream=pipeline.split_stream(
                panel=panel, site=site, beta=beta[site], factor=factor[site]
            ),
            window=window,
        )
        for site in common.DECISION_SITES
    }
    cut = common.cut_of(year=common.TUNING_YEAR)
    fitted = fit_filter_hyperparameters(
        residuals=streams,
        band=cache.calendar.band,
        end_times=cache.end_times,
        window=(common.to_64(moment=cut), common.to_64(moment=common.year_end(cut=cut))),
    )
    values = {site: hyper.as_array() for site, hyper in fitted.items()}
    values["D9"] = np.mean(list(values.values()), axis=0)
    return {site: FilterHyperparameters.from_array(values=array) for site, array in values.items()}


def tune(
    *,
    cache: FeatureCache,
    panel: factor_lib.Panel,
    factor: dict[str, np.ndarray],
    local_hyper: dict[str, FilterHyperparameters] | None,
) -> Tuning:
    """Fit the loadings, the shared hyperparameters, and (if not given) the local ones."""
    raw, beta = fit_betas(cache=cache, panel=panel, factor=factor)
    shared = fit_shared_hyper(cache=cache, factor=factor)
    if local_hyper is None:
        local_hyper = fit_local_hyper(cache=cache, panel=panel, factor=factor, beta=beta)
    return Tuning(raw_beta=raw, beta=beta, local_hyper=local_hyper, shared_hyper=shared)


def real_setup(
    *, tuning: Tuning, panel: factor_lib.Panel, name: str = "real"
) -> pipeline.FactorSetup:
    """Return the real factor's setup from frozen settings."""
    return pipeline.FactorSetup(
        name=name,
        factor=panel.factor,
        beta=tuning.beta,
        local_hyper=tuning.local_hyper,
        shared_hyper=tuning.shared_hyper,
    )


def load_real_tuning() -> Tuning:
    """Read the frozen settings written by `main`."""
    return Tuning.from_json(stored=json.loads(TUNING_PATH.read_text()))


def main() -> None:
    """Fit and freeze the real factor's settings, and print them beside experiment 106's `a0`."""
    cache = common.load_cache()
    all_fits = factor_lib.load_fits(cache=cache)
    panel = factor_lib.build_panel(cache=cache, all_fits=all_fits)
    tuning = tune(cache=cache, panel=panel, factor=panel.factor, local_hyper=None)
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    TUNING_PATH.write_text(json.dumps(tuning.to_json(), indent=2) + "\n")
    a0 = pipeline.a0_hyper_from_part4()
    print(
        "loadings (raw Huber slope -> shrunk): "
        + ", ".join(
            f"{site} {tuning.raw_beta.get(site, float('nan')):.3f} -> {tuning.beta[site]:.3f}"
            for site in tuning.beta
        )
    )
    print("hyperparameters (log q, logit phi, log sigma_z, log sigma_eps)")
    for site, hyper in tuning.local_hyper.items():
        print(f"  local  {site:>4} " + " ".join(f"{v:7.3f}" for v in hyper.as_array()))
        print(f"  a0     {site:>4} " + " ".join(f"{v:7.3f}" for v in a0[site].as_array()))
    print("  shared    all " + " ".join(f"{v:7.3f}" for v in tuning.shared_hyper.as_array()))
    print(
        f"  shared: q {tuning.shared_hyper.q:.4f}/day, phi {tuning.shared_hyper.phi:.4f}, "
        f"sigma_z {tuning.shared_hyper.sigma_z:.3f}, sigma_eps {tuning.shared_hyper.sigma_eps:.3f}"
    )


if __name__ == "__main__":
    main()
