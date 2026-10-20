"""Fit each arm's noise matrix on the tuning months, for experiment 118.

For each fitted arm the script fits, for the six scored sites together, the per-site random-walk
variances of the arm's moving weights and the observation noise, by maximum a posteriori under a
prior shared across sites (`studies.dp_effects_filter.fit_effects_noise_sites`). The anomaly's
factor and size are experiment 111's fitted values at each site. The fit sees the residual stream
up to 2025-07-01 only, and the likelihood counts the tuning months' readings only; the state runs
warm through the replicate months.

The output is one JSON file per arm under `work/noise/`, holding for each site the fitted `q`, the
observation noise, the regressor scales, which weights were set by the prior, and the number of
active half-hours of each weight in the likelihood.

    uv run python ml-experiments/00118-*/e00118_noise.py [--workers 6] [--arms KE KEZ ...]
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import e00118_arms as arms
import e00118_common as common
import e00118_streams as streams
import numpy as np
import torch
from studies.dp_effects_filter import (
    SiteStream,
    day_ahead_targets,
    fit_effects_noise_sites,
    relative_step_per_year,
)

NOISE_DIR: Final[Path] = common.WORK_DIR / "noise"
SITES: Final[tuple[str, ...]] = common.SCORED_SITES


def noise_path(*, arm: str) -> Path:
    """Return the file that holds an arm's fitted noise."""
    return NOISE_DIR / f"{arm}.json"


def read_noise(*, arm: str) -> dict:
    """Return an arm's stored noise: per-site fits and the final shared prior."""
    return json.loads(noise_path(arm=arm).read_text())


def site_streams(*, site: str) -> streams.SiteStreams:
    """Build a site's streams from experiment 111's inputs."""
    cache = common.load_cache()
    setup = common.load_setup(cache=cache, site=site)
    return streams.build_site_streams(cache=cache, setup=setup, fits=common.load_fits(site=site))


def fit_arm(name: str) -> str:
    """Fit one arm's noise at every scored site and store it.

    Args:
        name: The arm's name, a key of `arms.FITTED_ARMS`.

    Returns:
        A progress line.
    """
    torch.set_num_threads(1)
    started = time.monotonic()
    arm = arms.FITTED_ARMS[name]
    cache = common.load_cache()
    hypers = common.tuned_111()["filter_hyper"]["gamk"]
    inputs: dict[str, SiteStream] = {}
    scales: dict[str, np.ndarray] = {}
    for site in SITES:
        data = site_streams(site=site)
        noise = arms.gk_noise(hyper=hypers[site], scale=data.scale)
        sigma_eps_start = noise.sigma_eps
        if arm.inflated:
            sigma_eps_start = read_noise(arm="KE")["sites"][site]["sigma_eps"] * (
                arms.INFLATION_VARIANCE**0.5
            )
        resid = data.fit_residual(cache=cache, skip_holidays=arm.skip_holidays)
        scored = data.tuning_mask & np.isfinite(resid)
        inputs[site] = SiteStream(
            residual=resid,
            regressors=data.standardised,
            scale=data.scale,
            likelihood_mask=scored,
            phi=noise.phi,
            sigma_z=noise.sigma_z,
            sigma_eps_start=sigma_eps_start,
            lead_matched=day_ahead_targets(end_times=cache.end_times, rows=np.flatnonzero(scored)),
        )
        scales[site] = data.scale
    fitted, prior_mean, prior_spread = fit_effects_noise_sites(
        streams=inputs,
        band=data.band,
        variant=arm.variant,
        sigma_eps_spread=arms.INFLATED_SIGMA_EPS_SPREAD if arm.inflated else 1.0,
    )
    content = {
        "arm": name,
        "prior_mean": prior_mean.tolist(),
        "prior_spread": prior_spread.tolist(),
        "sites": {
            site: {
                "q": fit.noise.q.tolist(),
                "log_q": [float(v) if np.isfinite(v) else None for v in fit.log_q],
                "sigma_eps": fit.noise.sigma_eps,
                "phi": fit.noise.phi,
                "sigma_z": fit.noise.sigma_z,
                "initial_sd": fit.noise.initial_sd.tolist(),
                "scale": scales[site].tolist(),
                "set_by_prior": fit.set_by_prior.tolist(),
                "n_active": [int(v) for v in fit.n_active],
                "log_posterior": fit.log_posterior,
                "converged": fit.converged,
                "relative_step_per_year": relative_step_per_year(
                    noise=fit.noise, scale=scales[site]
                ).tolist(),
            }
            for site, fit in fitted.items()
        },
    }
    common.write_json_atomic(path=noise_path(arm=name), content=content)
    return f"{name}: {len(fitted)} sites, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Fit the arms in parallel; KE-infl waits for KE."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[name for name, arm in arms.FITTED_ARMS.items() if not arm.one_step],
    )
    args = parser.parse_args()
    first = [name for name in args.arms if name != "KE-infl"]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(fit_arm, first):
            print(line, flush=True)
    if "KE-infl" in args.arms:
        print(fit_arm("KE-infl"), flush=True)


if __name__ == "__main__":
    main()
