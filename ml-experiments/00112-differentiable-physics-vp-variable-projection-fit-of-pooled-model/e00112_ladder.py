"""The equivalence check refitted by variable projection (L1), and the convex control (L0).

Experiment 105's equivalence arm `s_equiv_all` was the per-site DP set up like the convex model: no
interaction network, no shared daily shapes, a weak penalty on the deviation, and the capacity
pinned to Part 2's convex path, fitted on every usable year by gradient descent. It scored 0.25
points worse than the convex model G-all. L1 here fits the same configuration by variable
projection on the natural-scale objective: one substation at a time, the capacity pinned to the
convex path fitted at the same cut, a deviation scale of 100, the heating threshold started from
12 to 20 degrees in steps of 2, and the start with the lowest objective kept. L1 minus
`s_equiv_all` is the optimiser's share of the 0.25, and it also contains the change from the
pre-softplus slope prior and the learned noise scale to the natural-scale prior and a fixed noise
scale. The heating form (one hinge at a learned threshold on a learned temperature mix, against the
convex model's one slope per smoothing and band at 15.5 degrees) can no longer be separated from
the optimiser: the share of the 0.25 that remains after L1 is the heating form and the rest of the
specification together.

L0 is Part 2's convex model itself (`g_all_centred`), scored by the report to confirm 3.072 points
at the five decision sites.

    uv run --no-sync python ml-experiments/00112-*/e00112_ladder.py
"""

import argparse
import json
import time
from datetime import datetime
from typing import Final

import e00112_common as common
import e00112_forecast as forecast
import e00112_gradient as gradient
import e00112_vp as vp
import numpy as np
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import GamSpec, TrainingWindow, fit_gam, predict_gam, training_rows
from studies.dp_onboarding import PooledResiduals, fit_centred_spread, pooled_residuals
from studies.dp_pooled import DpConfig, PooledBatch, PooledDemandModel, PopulationScales
from studies.dp_pooled_vp import VpSettings, fit_pooled_vp, freeze_operator_for_vp

WEAK_DEVIATION_SCALE: Final[float] = 100.0
THRESHOLD_STARTS_C: Final[tuple[float, ...]] = (12.0, 14.0, 16.0, 18.0, 20.0)
LADDER_FALLBACK_STEPS: Final[int] = 400


def convex_fit(*, cache: FeatureCache, site: str, cut: datetime):  # noqa: ANN201
    """Return Part 2's convex model (all years, no drift, tuned penalties) fitted at `cut`."""
    tuned = json.loads((common.PART_2_DIR / "work" / "tuned.json").read_text())
    spec = GamSpec(
        drift="none",
        ridge=tuned["ridge"]["chosen"],
        drift_step=tuned["drift_step"]["chosen"],
        step_penalty=tuned["step_penalty"]["chosen"],
    )
    return fit_gam(
        cache=cache,
        site=site,
        specs=[spec],
        cut=cut,
        window=TrainingWindow(),
        season_years=common.SEASON_YEARS,
    )


def convex_override(*, cache: FeatureCache, site: str, cut: datetime, batch: PooledBatch):  # noqa: ANN201
    """Return the convex capacity path of `site` fitted at `cut`, shape (1, years), in points."""
    fit = convex_fit(cache=cache, site=site, cut=cut)
    by_year = {int(year): float(value) for year, value in fit.capacity_points().items()}
    path = np.full((1, batch.n_season_years), np.nan)
    previous = 0.0
    for offset in range(batch.n_season_years):
        previous = by_year.get(batch.first_season_year + offset, previous)
        path[0, offset] = previous
    return torch.as_tensor(path, dtype=batch.y.dtype, device=batch.y.device)


def convex_sd(*, cache: FeatureCache, sites: tuple[str, ...], cut: datetime) -> float:
    """Return the root mean square training residual of the convex model on all years, in points."""
    squares = []
    for site in sites:
        fit = convex_fit(cache=cache, site=site, cut=cut)
        index = training_rows(
            features=cache.sites[site], cache=cache, cut=cut, window=TrainingWindow()
        )
        residual = cache.sites[site].y[index] - predict_gam(cache=cache, fit=fit, index=index)
        squares.append(residual**2)
    return float(np.sqrt(np.mean(np.concatenate(squares))))


def equivalence_model(*, batch: PooledBatch, override: torch.Tensor, threshold: float):  # noqa: ANN201
    """Return the s_equiv_all configuration as a model started at `threshold` degrees."""
    torch.manual_seed(0)
    model = PooledDemandModel(
        config=DpConfig(
            shared_shapes=0, population_scales=PopulationScales(deviation=WEAK_DEVIATION_SCALE)
        ),
        n_sites=1,
        n_season_years=batch.n_season_years,
        initial_level=torch.full((1,), 50.0),
    ).to(device=batch.y.device, dtype=batch.y.dtype)
    model.capacity_override = override
    with torch.no_grad():
        model.threshold.fill_(threshold)
        model.population_mean["threshold"].fill_(threshold)
    freeze_operator_for_vp(model=model)
    return model


def record_ladder(*, site: str, cut: datetime, objectives: dict, seconds: float) -> None:
    """Store a site's objectives by threshold start and the seconds the fits took."""
    path = common.WORK_DIR / "ladder_objectives.json"
    table = json.loads(path.read_text()) if path.exists() else {}
    table[f"{site}__{cut.date().isoformat()}"] = {
        "objectives": {str(k): v for k, v in objectives.items()},
        "seconds": seconds,
    }
    common.write_json(name="ladder_objectives.json", payload=table)


def fit_site(*, cache: FeatureCache, site: str, cut: datetime, sigma: float):  # noqa: ANN201
    """Fit L1 for one substation from every threshold start; return the lowest-objective model."""
    batch = common.pooled_batch_for(
        cache=cache,
        sites=(site,),
        cut=cut,
        months=None,
        device=vp.DEVICE,
        dtype=common.BATCH_DTYPE,
    )
    override = convex_override(cache=cache, site=site, cut=cut, batch=batch)
    settings = VpSettings(noise_sd=sigma, deviation=WEAK_DEVIATION_SCALE)
    best = None
    objectives = {}
    started = time.perf_counter()
    for threshold in THRESHOLD_STARTS_C:
        model = equivalence_model(batch=batch, override=override, threshold=threshold)
        result = fit_pooled_vp(
            model=model,
            batch=batch,
            settings=settings,
            adam_steps=LADDER_FALLBACK_STEPS,
            warm_start_steps=0,
        )
        objectives[threshold] = result.objective
        if best is None or result.objective < best[1].objective:
            best = (model, result, batch)
    assert best is not None
    seconds = time.perf_counter() - started
    print(f"L1 {site} at {cut.date()}: objectives by start {objectives}", flush=True)
    record_ladder(site=site, cut=cut, objectives=objectives, seconds=seconds)
    return best


def l1_forecasts(*, cache: FeatureCache) -> dict[str, gradient.SiteForecast]:
    """Return L1's forecasts of the scored year, each substation with its own replica spread."""
    cut, cut_r, through = common.FINAL_CUT, common.REPLICATE_CUT, common.SCORING_END
    sigma = convex_sd(cache=cache, sites=common.SCORED_SITES, cut=cut)
    sigma_r = convex_sd(cache=cache, sites=common.SCORED_SITES, cut=cut_r)
    residuals: dict[str, PooledResiduals] = {}
    models = {}
    for site in common.SCORED_SITES:
        model, _, _ = fit_site(cache=cache, site=site, cut=cut, sigma=sigma)
        models[site] = model
        try:
            model_r, _, _ = fit_site(cache=cache, site=site, cut=cut_r, sigma=sigma_r)
        except torch.linalg.LinAlgError:
            # A substation with 9 months of history at the replica cut gives a singular system;
            # its spread comes from the other substations' replica residuals.
            print(f"L1 {site}: replica fit failed (singular), pooled spread used", flush=True)
            continue
        replica_year = forecast.score_batch(
            cache=cache, sites=(site,), cut=cut_r, through=cut, for_spread=True
        )
        residuals[site] = pooled_residuals(model=model_r, batch=replica_year)
    out: dict[str, gradient.SiteForecast] = {}
    for site, model in models.items():
        use = [residuals[site]] if site in residuals else list(residuals.values())
        spread = fit_centred_spread(
            residual=np.concatenate([r.residual for r in use]),
            band=np.concatenate([r.band for r in use]),
            quarter=np.concatenate([r.quarter for r in use]),
        )
        score = forecast.score_batch(cache=cache, sites=(site,), cut=cut, through=through)
        out |= gradient.forecasts_by_site(
            model=model, score=score, replica=None, spread_for={site: spread}
        )
    return out


def main() -> None:
    """Store L1's prediction set."""
    common.require_gate_passed()
    argparse.ArgumentParser().parse_args()
    cache = common.load_cache()
    forecast.store(cache=cache, name="vp_l1_equiv_all", forecasts=l1_forecasts(cache=cache))


if __name__ == "__main__":
    main()
