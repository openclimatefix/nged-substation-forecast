"""The blocking known-answer gate of experiment 112, before any replicate-year or scored-year fit.

Module tests 1 to 5 of the plan are `packages/studies/tests/test_dp_pooled_vp.py`. This script runs
the other three, each fitted at the replicate cut (2024-07-01) with the interaction network on. Every
fit starts cold and is saved, so a rerun loads it. The results go to `work/gate.json`, which the fit,
forecast, onboarding, profile, and ladder scripts check before they run.

**Test 2, a synthetic fleet.** Eight sites on the real calendar and weather tensors of the MW
Primaries, 2 years each, with K = 2 shared daily shapes of known loadings, a 14 degree threshold, a
known thermal-inertia mix, non-zero shared holiday, tent, harmonic, and lighting terms, a capacity
path with a 3 to 8 point growth increment in the last season-year, no network term, and residuals
block-bootstrapped by week from the per-site convex model's training residuals at a different site.
Pass if every threshold is within 1 degree, every slope in every band within 15%, every season-year
capacity within 10%, every growth increment within 1.5 points, and the network's centred
summer-midday output under 0.1 points.

**Test 3, the outer loop converges** on the fleet from three starts (seed 0; seed 1; thresholds at
17 degrees with uniform loadings): the final objective gradient is under 1e-4, the objectives agree
within 1 nat per 10,000 rows, and each threshold agrees with a 0.1 degree grid profile within 0.2
degrees.

**Test 4, injected farms**, one at a time at D10, D11, D14, and D16: the recovered capacity increment
over the injected one has a median in [0.9, 1.1] (experiment 105's value was 0.78), and the other
scored substations' capacities move under 5%. The shared modules' midday movement is reported
beside a second-seed base fit's, made once on the first farm's rows. Only ratios appear.

    uv run --no-sync python ml-experiments/00112-*/e00112_gate.py [--tests 2 3 4]
"""

import argparse
import json
from dataclasses import replace
from datetime import datetime
from typing import Any, Final

import e00112_common as common
import e00112_vp as vp
import numpy as np
import polars as pl
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import GamSpec, TrainingWindow, fit_gam, predict_gam, training_rows
from studies.dp_onboarding import select_rows, with_targets
from studies.dp_pooled import DpConfig, PooledBatch, PooledDemandModel
from studies.dp_pooled_vp import (
    VpSettings,
    _evaluate,
    make_vp_problem,
    phi_parameters,
    trainable_only,
)

GATE_CUT: Final[datetime] = common.REPLICATE_CUT
REAL_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
"""Real substations of the synthetic fleet; D9 holds 9 months of the window and is left out."""
N_COPIES: Final[int] = 3
"""Relabelled copies of the first real sites, so the fleet has 8 substations."""
TRUE_THRESHOLD_C: Final[float] = 14.0
TRUE_INERTIA: Final[tuple[float, ...]] = (0.2, 0.1, 0.3, 0.4)
TRUE_SLOPE: Final[float] = 8.0
INJECTED_SITES: Final[tuple[str, ...]] = ("D10", "D11", "D14", "D16")
FARM_OF_SITE: Final[dict[str, str]] = {"D10": "A", "D11": "B", "D14": "C", "D16": "D"}
INJECTED_POINTS: Final[float] = 10.0
"""The injected farm's output at its 99th percentile, in points of effective capacity."""
SUMMER_MONTHS: Final[range] = range(4, 10)
MIDDAY_HALF_HOURS: Final[range] = range(20, 29)
WEEK_SLOTS: Final[int] = 336
"""Half-hours in a week."""
GATE_FALLBACK_STEPS: Final[int] = 400
GRID_STEP_C: Final[float] = 0.1
GRID_HALF_WIDTH_C: Final[float] = 1.0


def inverse_softplus(value: np.ndarray | float) -> np.ndarray:
    """Return `x` with `softplus(x) = value`."""
    return np.log(np.expm1(np.asarray(value, dtype=np.float64)))


def relabel_site(*, batch: PooledBatch, source: int, target: int) -> PooledBatch:
    """Return the rows of site `source` as site `target` of the same layout."""
    rows = select_rows(batch=batch, keep=batch.site == source)
    year = rows.season_slot % batch.n_season_years
    return replace(
        rows,
        site=torch.full_like(rows.site, target),
        season_slot=target * batch.n_season_years + year,
    )


def concat_batches(*, parts: list[PooledBatch], labels: tuple[str, ...]) -> PooledBatch:
    """Concatenate batches that share a layout, with the given site labels."""
    names = (
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
    fields = {name: torch.cat([getattr(part, name) for part in parts]) for name in names}
    return replace(parts[0], site_labels=labels, **fields)


def smooth_shapes(*, generator: np.random.Generator, k: int) -> np.ndarray:
    """Return `k` smooth daily shapes over day type and half-hour, a few harmonics each."""
    half_hour = np.arange(48)
    shapes = np.zeros((k, 4, 48))
    for i in range(k):
        for day_type in range(4):
            for harmonic in (1, 2, 3):
                amplitude = generator.normal(scale=3.0 / harmonic)
                phase = generator.uniform(0, 2 * np.pi)
                shapes[i, day_type] += amplitude * np.sin(
                    2 * np.pi * harmonic * half_hour / 48 + phase
                )
    return shapes.reshape(k, -1)


def convex_residual_weeks(*, cache: FeatureCache, site: str) -> dict[int, np.ndarray]:
    """Return the per-site convex model's training residuals of `site`, one array per week.

    Each array has 336 half-hour slots, with `NaN` where the site has no training row. The weeks
    are numbered from the first training row.
    """
    tuned = json.loads((common.PART_2_DIR / "work" / "tuned.json").read_text())
    spec = GamSpec(
        drift="none",
        ridge=tuned["ridge"]["chosen"],
        drift_step=tuned["drift_step"]["chosen"],
        step_penalty=tuned["step_penalty"]["chosen"],
    )
    window = TrainingWindow(months=common.WINDOW_MONTHS)
    fit = fit_gam(
        cache=cache,
        site=site,
        specs=[spec],
        cut=GATE_CUT,
        window=window,
        season_years=common.SEASON_YEARS,
    )
    index = training_rows(features=cache.sites[site], cache=cache, cut=GATE_CUT, window=window)
    residual = cache.sites[site].y[index] - predict_gam(cache=cache, fit=fit, index=index)
    return week_blocks(times=cache.end_times[index], values=residual)


def week_blocks(*, times: np.ndarray, values: np.ndarray) -> dict[int, np.ndarray]:
    """Cut a series into week arrays of 336 half-hour slots, with `NaN` where it has no row."""
    origin = times.min().astype("datetime64[D]")
    offset = (times - origin) / np.timedelta64(30, "m")
    week = (offset // WEEK_SLOTS).astype(int)
    slot = (offset % WEEK_SLOTS).astype(int)
    blocks: dict[int, np.ndarray] = {}
    for w in np.unique(week):
        block = np.full(WEEK_SLOTS, np.nan)
        keep = week == w
        block[slot[keep]] = values[keep]
        blocks[int(w)] = block
    return blocks


def bootstrap_noise(
    *,
    times: np.ndarray,
    weeks: dict[int, np.ndarray],
    generator: np.random.Generator,
) -> np.ndarray:
    """Return noise for rows at `times`: for each week of rows, a random week of `weeks`.

    A missing value (`NaN`) in the chosen week is replaced by 0, the residuals' mean.
    """
    origin = times.min().astype("datetime64[D]")
    offset = (times - origin) / np.timedelta64(30, "m")
    week = (offset // WEEK_SLOTS).astype(int)
    slot = (offset % WEEK_SLOTS).astype(int)
    pool = list(weeks.values())
    out = np.zeros(len(times))
    for w in np.unique(week):
        chosen = pool[int(generator.integers(0, len(pool)))]
        rows = week == w
        out[rows] = np.nan_to_num(chosen[slot[rows]], nan=0.0)
    return out


def new_truth(*, batch: PooledBatch, generator: np.random.Generator) -> PooledDemandModel:
    """Return the truth model of the synthetic fleet, with every parameter set."""
    n_sites = len(batch.site_labels)
    torch.manual_seed(0)
    truth = PooledDemandModel(
        config=DpConfig(shared_shapes=2),
        n_sites=n_sites,
        n_season_years=batch.n_season_years,
        initial_level=torch.zeros(n_sites),
    ).to(device=batch.y.device, dtype=batch.y.dtype)
    truth.mark_trained_years(batch=batch)
    slopes = TRUE_SLOPE * (1.0 + 0.1 * generator.uniform(-1, 1, n_sites))
    first_capacity = generator.uniform(5.0, 25.0, n_sites)
    growth = generator.uniform(3.0, 8.0, n_sites)
    raw = np.full((n_sites, batch.n_season_years), float(inverse_softplus(0.01)))
    trained = (truth.trained > 0).cpu().numpy()
    for site in range(n_sites):
        years = np.flatnonzero(trained[site])
        raw[site, years[0]] = inverse_softplus(first_capacity[site])
        raw[site, years[-1]] = inverse_softplus(growth[site])

    def tensor(array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=batch.y.dtype, device=batch.y.device)

    with torch.no_grad():
        truth.shapes.copy_(tensor(smooth_shapes(generator=generator, k=2)))
        truth.site_logits.copy_(tensor(generator.normal(size=(n_sites, 2))))
        truth.level.copy_(tensor(generator.uniform(40, 70, n_sites)))
        truth.threshold.fill_(TRUE_THRESHOLD_C)
        truth.inertia_logits.copy_(tensor(np.log(np.array(TRUE_INERTIA))))
        truth.slope_raw.copy_(tensor(inverse_softplus(slopes))[:, None].expand(-1, 4))
        truth.capacity_raw.copy_(tensor(raw))
        truth.holiday.copy_(tensor(0.5 * generator.normal(size=tuple(truth.holiday.shape))))
        truth.tent_coef.copy_(tensor(0.5 * generator.normal(size=tuple(truth.tent_coef.shape))))
        truth.harmonic_coef.copy_(
            tensor(0.8 * generator.normal(size=tuple(truth.harmonic_coef.shape)))
        )
        truth.lighting_coef.copy_(
            tensor(0.5 * generator.normal(size=tuple(truth.lighting_coef.shape)))
        )
    return truth


def build_synthetic(*, cache: FeatureCache) -> tuple[PooledBatch, PooledDemandModel, float]:
    """Return the eight-site synthetic batch with its targets, the truth model, and the noise sd."""
    base = common.pooled_batch_for(
        cache=cache, sites=REAL_SITES, cut=GATE_CUT, device=vp.DEVICE, dtype=common.BATCH_DTYPE
    )
    parts = [relabel_site(batch=base, source=i, target=i) for i in range(len(REAL_SITES))]
    parts += [
        relabel_site(batch=base, source=src, target=len(REAL_SITES) + i)
        for i, src in enumerate(range(N_COPIES))
    ]
    labels = (*REAL_SITES, *(f"N{i + 1}" for i in range(N_COPIES)))
    batch = concat_batches(parts=parts, labels=labels)
    batch = replace(
        batch,
        solar_gate=torch.ones_like(batch.solar_gate),
        is_apparent=torch.zeros_like(batch.is_apparent),
    )
    generator = np.random.default_rng(0)
    truth = new_truth(batch=batch, generator=generator)
    truth.train()
    with torch.no_grad():
        clean = truth(batch)
    weeks = {site: convex_residual_weeks(cache=cache, site=site) for site in REAL_SITES}
    times = cache.end_times[batch.row.cpu().numpy()]
    noise = np.zeros(batch.n_rows)
    site_of_row = batch.site.cpu().numpy()
    for index in range(len(labels)):
        rows = site_of_row == index
        other = REAL_SITES[(index + 1) % len(REAL_SITES)]
        noise[rows] = bootstrap_noise(times=times[rows], weeks=weeks[other], generator=generator)
    noisy = clean + torch.as_tensor(noise, dtype=clean.dtype, device=clean.device)
    return with_targets(batch=batch, y=noisy), truth, float(noise.std())


def summer_midday(*, batch: PooledBatch, cache: FeatureCache) -> torch.Tensor:
    """Return a boolean tensor marking summer middays (10:00 to 14:00 local, April to September)."""
    end = cache.end_times[batch.row.cpu().numpy()]
    month = end.astype("datetime64[M]").astype(int) % 12 + 1
    in_summer = np.isin(month, list(SUMMER_MONTHS))
    in_midday = np.isin(batch.half_hour.cpu().numpy(), list(MIDDAY_HALF_HOURS))
    return torch.as_tensor(in_summer & in_midday, device=batch.y.device)


def module_midday_output(
    *, model: PooledDemandModel, batch: PooledBatch, mask: torch.Tensor
) -> dict[str, float]:
    """Return the mean midday output of each shared module, centred per site-year, in points."""
    model.eval()
    n_slots = model.centre_means.numel()
    counts = torch.bincount(batch.season_slot, minlength=n_slots).clamp(min=1)
    with torch.no_grad():
        holiday = model.holiday[batch.holiday, batch.band]
        tent = (batch.tent * model.tent_coef[:, batch.band].T).sum(dim=1)
        harmonics = (batch.harmonics * model.harmonic_coef[:, batch.band].T).sum(dim=1)
        lighting = batch.lighting @ model.lighting_coef
        total, _ = model.calendar_terms(batch=batch)
        network = total - holiday - tent - harmonics - lighting
        out = {}
        for name, values in (
            ("holiday", holiday),
            ("tent", tent),
            ("harmonics", harmonics),
            ("lighting", lighting),
            ("interaction_network", network),
        ):
            sums = torch.zeros(n_slots, dtype=values.dtype, device=values.device)
            means = sums.index_add(0, batch.season_slot, values) / counts.to(values.dtype)
            out[name] = float((values - means[batch.season_slot])[mask].mean())
    return out


def network_midday_size(
    *, model: PooledDemandModel, batch: PooledBatch, mask: torch.Tensor
) -> float:
    """Return the largest absolute centred summer-midday mean of the network, over substations."""
    model.eval()
    n_slots = model.centre_means.numel()
    counts = torch.bincount(batch.season_slot, minlength=n_slots).clamp(min=1)
    with torch.no_grad():
        total, _ = model.calendar_terms(batch=batch)
        holiday = model.holiday[batch.holiday, batch.band]
        tent = (batch.tent * model.tent_coef[:, batch.band].T).sum(dim=1)
        harmonics = (batch.harmonics * model.harmonic_coef[:, batch.band].T).sum(dim=1)
        lighting = batch.lighting @ model.lighting_coef
        network = total - holiday - tent - harmonics - lighting
        sums = torch.zeros(n_slots, dtype=network.dtype, device=network.device)
        means = sums.index_add(0, batch.season_slot, network) / counts.to(network.dtype)
        centred = network - means[batch.season_slot]
        sizes = [
            abs(float(centred[mask & (batch.site == site)].mean()))
            for site in range(len(batch.site_labels))
        ]
    return max(sizes)


def uniform_loadings_at_17(model: PooledDemandModel) -> None:
    """Start every threshold at 17 degrees with uniform style loadings (test 3's third start)."""
    with torch.no_grad():
        model.threshold.fill_(common.START_THRESHOLD_C)
        model.population_mean["threshold"].fill_(common.START_THRESHOLD_C)
        model.site_logits.zero_()


def fit_synthetic(
    *, batch: PooledBatch, noise_sd: float, name: str, start: str, third: bool = False
) -> tuple[PooledDemandModel, dict]:
    """Fit the fleet from a start (saved, so a rerun loads it); `third` is test 3's third start."""
    settings = VpSettings(noise_sd=noise_sd, deviation=common.tuned()["map_deviation"])
    return vp.fit_cached(
        name=f"gate_fleet_{name}",
        cut=GATE_CUT,
        batch=batch,
        settings=settings,
        start=start,
        fallback_steps=GATE_FALLBACK_STEPS,
        prepare=uniform_loadings_at_17 if third else None,
    )


def recovery(
    *, model: PooledDemandModel, truth: PooledDemandModel, batch: PooledBatch, cache: FeatureCache
) -> dict[str, Any]:
    """Return how well `model` recovered the truth's thresholds, slopes, capacities, and network."""
    n_sites = len(batch.site_labels)
    slope_fit = torch.nn.functional.softplus(model.slope_raw).detach().cpu().numpy()
    slope_true = torch.nn.functional.softplus(truth.slope_raw).detach().cpu().numpy()
    capacity_fit = model.capacities().detach().cpu().numpy()
    capacity_true = truth.capacities().detach().cpu().numpy()
    trained = (truth.trained > 0).cpu().numpy()
    capacity_error, growth_error = [], []
    for site in range(n_sites):
        years = np.flatnonzero(trained[site])
        capacity_error += [
            float(abs(capacity_fit[site, y] / capacity_true[site, y] - 1)) for y in years
        ]
        growth_fit = capacity_fit[site, years[-1]] - capacity_fit[site, years[0]]
        growth_true = capacity_true[site, years[-1]] - capacity_true[site, years[0]]
        growth_error.append(float(abs(growth_fit - growth_true)))
    mask = summer_midday(batch=batch, cache=cache)
    return {
        "threshold_error_c": [
            float(abs(t - TRUE_THRESHOLD_C)) for t in model.threshold.detach().cpu().tolist()
        ],
        "slope_relative_error": [
            float(np.max(np.abs(slope_fit[i] / slope_true[i] - 1))) for i in range(n_sites)
        ],
        "capacity_relative_error": capacity_error,
        "growth_increment_error_points": growth_error,
        "network_midday_points": network_midday_size(model=model, batch=batch, mask=mask),
    }


def passes_recovery(result: dict[str, Any]) -> bool:
    """Return whether a recovery meets the plan's thresholds."""
    return bool(
        max(result["threshold_error_c"]) < 1.0
        and max(result["slope_relative_error"]) < 0.15
        and max(result["capacity_relative_error"]) < 0.10
        and max(result["growth_increment_error_points"]) < 1.5
        and result["network_midday_points"] < 0.1
    )


def threshold_profile_error(
    *, model: PooledDemandModel, batch: PooledBatch, noise_sd: float
) -> list[float]:
    """Return, per substation, the distance of its threshold from the grid-profile minimum.

    The profile moves one substation's threshold over a 0.1 degree grid around the fitted value,
    holding every other non-linear parameter fixed and solving the linear coefficients exactly.
    """
    settings = VpSettings(noise_sd=noise_sd, deviation=common.tuned()["map_deviation"])
    problem = make_vp_problem(model=model, batch=batch, settings=settings)
    phi = phi_parameters(model=model)
    errors = []
    grid = np.arange(-GRID_HALF_WIDTH_C, GRID_HALF_WIDTH_C + 1e-9, GRID_STEP_C)
    with trainable_only(model=model, parameters=phi):
        for site in range(len(batch.site_labels)):
            fitted = float(model.threshold[site])
            values = []
            for shift in grid:
                with torch.no_grad():
                    model.threshold[site] = fitted + float(shift)
                values.append(_evaluate(problem=problem, phi=phi)[0])
            with torch.no_grad():
                model.threshold[site] = fitted
            errors.append(float(abs(grid[int(np.argmin(values))])))
    return errors


def gate_2_and_3(*, cache: FeatureCache) -> dict[str, Any]:
    """Return the synthetic-fleet recovery (test 2) and the three-start convergence (test 3)."""
    batch, truth, noise_sd = build_synthetic(cache=cache)
    fits = {
        "s0": fit_synthetic(batch=batch, noise_sd=noise_sd, name="s0", start="s0"),
        "s1": fit_synthetic(batch=batch, noise_sd=noise_sd, name="s1", start="s1"),
        "s17_uniform": fit_synthetic(
            batch=batch, noise_sd=noise_sd, name="s17_uniform", start="s0", third=True
        ),
    }
    recoveries = {
        name: recovery(model=model, truth=truth, batch=batch, cache=cache)
        for name, (model, _) in fits.items()
    }
    objectives = {name: record["objective"] for name, (_, record) in fits.items()}
    rows = batch.n_rows
    spread = (max(objectives.values()) - min(objectives.values())) / rows * 1e4
    profile = {
        name: threshold_profile_error(model=model, batch=batch, noise_sd=noise_sd)
        for name, (model, _) in fits.items()
    }
    return {
        "noise_sd": noise_sd,
        "test_2": {
            "recovery": recoveries,
            "passed": all(passes_recovery(r) for r in recoveries.values()),
        },
        "test_3": {
            "objectives": objectives,
            "objective_spread_nats_per_10000_rows": spread,
            "gradient_max_abs": {name: r["gradient_max_abs"] for name, (_, r) in fits.items()},
            "converged": {name: r["converged"] for name, (_, r) in fits.items()},
            "threshold_profile_error_c": profile,
            "seconds": {name: r["seconds"] for name, (_, r) in fits.items()},
            "passed": bool(
                all(r["gradient_max_abs"] < 1e-4 for _, r in fits.values())
                and spread < 1.0
                and all(max(e) <= 0.2 for e in profile.values())
            ),
        },
    }


def farm_series(*, farm: str, end_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a farm's normalised output at the rows' end times, and where it has a reading."""
    table = (
        pl.read_parquet(common.FARMS_FILE)
        .filter(pl.col("farm") == farm)
        .select("time", "farm_norm")
        .drop_nulls()
        .sort("time")
    )
    times = table["time"].dt.replace_time_zone(None).dt.cast_time_unit("us").to_numpy()
    values = table["farm_norm"].to_numpy().astype(np.float64)
    position = np.clip(np.searchsorted(times, end_times), 0, len(times) - 1)
    found = times[position] == end_times
    return np.where(found, values[position], 0.0), found


def p99_unit_shape(*, batch: PooledBatch, site: int) -> float:
    """Return the 99th percentile of a site's daylight PV shape."""
    shape = batch.pv_shape[batch.site == site]
    daylight = shape[shape > 0.01]
    return float(torch.quantile(daylight.float().cpu(), 0.99))


def gate_4_injected(*, cache: FeatureCache) -> dict[str, Any]:
    """Return the injected-farm recovery.

    A farm reports for about 30% of the window, so each injection fit keeps the injected
    substation's rows only where the farm has a reading, and is compared with a base fit on exactly
    the same rows. Every fit starts cold.
    """
    batch = common.pooled_batch_for(
        cache=cache, sites=common.POOL, cut=GATE_CUT, device=vp.DEVICE, dtype=common.BATCH_DTYPE
    )
    sigma = common.read_sigma(cut=GATE_CUT, round_name="round1")
    settings = VpSettings(noise_sd=sigma, deviation=common.tuned()["map_deviation"])
    end = cache.end_times[batch.row.cpu().numpy()]
    solar_sites = [s for s in common.SCORED_SITES if s in batch.site_labels]
    ratios, others, by_module, floor = {}, {}, {}, {}
    for site in INJECTED_SITES:
        index = batch.site_labels.index(site)
        site_rows = batch.site == index
        values, found = farm_series(farm=FARM_OF_SITE[site], end_times=end)
        farm = torch.as_tensor(values, dtype=batch.y.dtype, device=batch.y.device)
        has_farm = torch.as_tensor(found, device=batch.y.device)
        keep = ~site_rows | has_farm
        rows = select_rows(batch=batch, keep=keep)
        injected_y = rows.y - INJECTED_POINTS * torch.where(
            rows.site == index, farm[keep], torch.zeros_like(farm[keep])
        )
        tag = site.lower()
        base, _ = vp.fit_cached(
            name=f"gate_farm_{tag}_base",
            cut=GATE_CUT,
            batch=rows,
            settings=settings,
            start="s0",
            fallback_steps=GATE_FALLBACK_STEPS,
        )
        second = None
        if site == INJECTED_SITES[0]:
            second, _ = vp.fit_cached(
                name=f"gate_farm_{tag}_second",
                cut=GATE_CUT,
                batch=rows,
                settings=settings,
                start="s1",
                fallback_steps=GATE_FALLBACK_STEPS,
            )
        injected, _ = vp.fit_cached(
            name=f"gate_farm_{tag}_injected",
            cut=GATE_CUT,
            batch=with_targets(batch=rows, y=injected_y),
            settings=settings,
            start="s0",
            fallback_steps=GATE_FALLBACK_STEPS,
        )
        base_cap = base.capacities()[:, -1].detach().cpu().numpy()
        cap = injected.capacities()[:, -1].detach().cpu().numpy()
        recovered = (cap[index] - base_cap[index]) * p99_unit_shape(batch=rows, site=index)
        ratios[site] = float(recovered / INJECTED_POINTS)
        others[site] = float(
            max(
                abs(cap[batch.site_labels.index(o)] / base_cap[batch.site_labels.index(o)] - 1)
                for o in solar_sites
                if o != site
            )
        )
        mask_rows = summer_midday(batch=rows, cache=cache)
        before = module_midday_output(model=base, batch=rows, mask=mask_rows)
        after = module_midday_output(model=injected, batch=rows, mask=mask_rows)
        by_module[site] = {name: after[name] - before[name] for name in before}
        if second is not None:
            noise = module_midday_output(model=second, batch=rows, mask=mask_rows)
            floor[site] = {name: noise[name] - before[name] for name in before}
    median = float(np.median(list(ratios.values())))
    return {
        "recovery_ratio": ratios,
        "median_ratio": median,
        "largest_other_capacity_change": others,
        "shared_midday_change_by_module_points": by_module,
        "second_seed_base_midday_change_by_module_points": floor,
        "passed": bool(0.9 <= median <= 1.1 and max(others.values()) < 0.05),
    }


def main() -> None:
    """Run the requested tests and write `work/gate.json`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", nargs="+", type=int, default=[2, 3, 4], choices=[2, 3, 4])
    arguments = parser.parse_args()
    cache = common.load_cache()
    out: dict[str, Any] = {}
    if 2 in arguments.tests or 3 in arguments.tests:
        out |= gate_2_and_3(cache=cache)
    if 4 in arguments.tests:
        out["test_4"] = gate_4_injected(cache=cache)
    previous = json.loads(common.GATE_PATH.read_text()) if common.GATE_PATH.exists() else {}
    common.write_json(name="gate.json", payload=previous | out)
    for name in ("test_2", "test_3", "test_4"):
        if name in out:
            print(name, "passed" if out[name]["passed"] else "FAILED", flush=True)


if __name__ == "__main__":
    main()
