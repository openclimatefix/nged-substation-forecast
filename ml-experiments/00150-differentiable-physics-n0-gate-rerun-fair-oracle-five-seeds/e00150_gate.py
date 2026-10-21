r"""The rerun of the known-answer gate: can the pooled fit and its tuning rule find a known truth?

Experiment 150's rerun of experiment 109's gate (`e00109_gate.py`, whose helpers are copied here).
The synthetic fleet is experiment 109's: eight synthetic substations on the real calendar and
weather of the MW Primaries, two years each, a known fleet mean `mu`, known deviations with family
scales `tau_f`, and noise taken as whole weeks of residuals from a different real substation. The
gate contains no generator data. One invocation runs one seed:

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00150-*/e00150_gate.py --seed 20261101 --workers 8
    uv run --no-sync python ml-experiments/00150-*/e00150_gate.py --summarise

**Differences from experiment 109's gate** (the plan's pre-registered lines):

- Line (b) compares the pooled fit with a fair oracle, the joint posterior mean that knows `mu`,
  every `tau_f` and the noise variance and estimates every coefficient, the capacity included
  (experiment 109's oracle was given the true capacity).
- The pooled fit's penalty set to 0 and to 1,000,000 rows are controls that must fail line (b).
- Line (d) compares the tuned penalty with the best penalty found by scoring the same fits against
  the noise-free truth. The two extra fleets (every `tau_f` ten times larger or a tenth) are gone.
- The seeds are fixed in advance and every seed runs the same definition.
"""

import argparse
import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import e00150_arms as arms
import e00150_common as common
import e00150_tune as tune
import numpy as np
import polars as pl
from studies.dp_calendar import HALF_HOURS_PER_DAY, SMOOTHING_HOURS
from studies.dp_features import FeatureCache, save_feature_cache
from studies.dp_gam import (
    N_BANDS,
    GamLayout,
    derated_unit_shape,
    design_matrix,
    predict_gam,
    training_rows,
)
from studies.dp_gam_pooled import (
    POOLED_FAMILIES,
    PooledFamilyType,
    PoolingStrength,
    centring_matrix,
    fit_pooled,
)
from studies.dp_pinned_capacity import pinned_capacity

GATE_DIR: Final[Path] = common.WORK_DIR / "gate"
N_SYNTHETIC: Final[int] = 8
WEATHER_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
REAL_SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
SYNTHETIC_LABELS: Final[tuple[str, ...]] = tuple(f"S{i + 1}" for i in range(N_SYNTHETIC))
TUNING_DECISION: Final[tuple[str, ...]] = SYNTHETIC_LABELS[:5]
WEEK_ROWS: Final[int] = 7 * HALF_HOURS_PER_DAY
FLEET_CUT: Final[datetime] = common.SCORED_CUT
SEEDS: Final[tuple[int, ...]] = (20261101, 20261102, 20261103, 20261104, 20261105)
"""The five seeds, fixed in the plan before any run."""
TRUTH_FILE: Final[str] = "truth.npz"
"""The noise-free series of each synthetic substation, saved beside its fleet."""
CONTROL_LAMBDAS: Final[dict[str, float]] = {"no_pooling": 0.0, "complete_pooling": 1_000_000.0}
SLOPE_TOLERANCE: Final[float] = 0.15
RMSE_FACTOR: Final[float] = 1.2


def _g_fits(*, cache: FeatureCache, vintages: pl.DataFrame) -> tuple[Any, dict[str, np.ndarray]]:
    """Fit G at the 2025-07-01 cut for the six real MW Primaries; return the fit and residuals.

    The residuals are those of the training window (the two years before the cut), by site, on
    the grid, `NaN` where the site has no training row.
    """
    config = arms.ArmConfig(name="g", pool=REAL_SITES, tag="gate_real")
    built = arms.build_arm(
        cache=cache, vintages=vintages, cut=FLEET_CUT, config=config, held_out=None, history=None
    )
    joint = fit_pooled(
        cache=built.cache,
        sites=built.sites,
        cut=FLEET_CUT,
        strengths=[PoolingStrength.uniform(value=0.0)],
    )
    residuals = {}
    for site in built.sites:
        index = training_rows(
            features=cache.sites[site.site], cache=cache, cut=FLEET_CUT, window=site.window
        )
        series = np.full(len(cache.end_times), np.nan)
        series[index] = cache.sites[site.site].y[index] - predict_gam(
            cache=built.cache, fit=joint.fits[site.site], index=index
        )
        residuals[site.site] = series
    return joint, residuals


def _pooled_vector(*, theta: np.ndarray, layout: GamLayout) -> np.ndarray:
    """Return one substation's pooled coefficients, in the order of `POOLED_FAMILIES`."""
    return np.concatenate([theta[layout.slices[family]] for family in POOLED_FAMILIES])


def _family_truth(
    *, joint: Any, layout: GamLayout, multiplier: float, generator: np.random.Generator
) -> tuple[dict[str, np.ndarray], dict[str, float], list[np.ndarray]]:
    """Return the fleet mean, the family scales, and each synthetic substation's truth.

    The truth is the pooled coefficients alone, because the synthetic capacity is pinned: the
    layout has no other columns.
    """
    if layout.slices["capacity"].stop != layout.slices["capacity"].start:
        msg = "the gate's real fits must have a pinned capacity"
        raise ValueError(msg)
    thetas = np.stack(
        [_pooled_vector(theta=joint.fits[s].theta[0], layout=layout) for s in REAL_SITES]
    )
    sizes = {f: layout.slices[f].stop - layout.slices[f].start for f in POOLED_FAMILIES}
    offsets = dict(zip(POOLED_FAMILIES, np.cumsum([0, *sizes.values()])[:-1], strict=True))

    def block(family: PooledFamilyType) -> slice:
        return slice(int(offsets[family]), int(offsets[family]) + sizes[family])

    mean_fit: dict[str, np.ndarray] = {}
    scale: dict[str, float] = {}
    for family in POOLED_FAMILIES:
        part = thetas[:, block(family)]
        centre = centring_matrix(family=family, size=sizes[family])
        mean_fit[family] = part.mean(axis=0)
        deviation = (part - part.mean(axis=0)) @ centre.T
        scale[family] = float(np.sqrt((deviation**2).mean()))
    truths = []
    for number in range(N_SYNTHETIC):
        real = thetas[number % len(REAL_SITES)]
        theta = np.zeros(thetas.shape[1])
        for family in POOLED_FAMILIES:
            centre = centring_matrix(family=family, size=sizes[family])
            noise = multiplier * scale[family] * generator.normal(size=sizes[family])
            if family == "profile":
                band_means = real[block(family)] - centre @ real[block(family)]
                theta[block(family)] = centre @ (mean_fit[family] + noise) + band_means
            else:
                theta[block(family)] = mean_fit[family] + noise
            if family == "heating":
                theta[block(family)] = np.maximum(theta[block(family)], 0.0)
        truths.append(theta)
    return mean_fit, scale, truths


def _noise_bank(*, residuals: dict[str, np.ndarray]) -> dict[str, list[np.ndarray]]:
    """Return each real site's whole weeks of training-window residuals (mostly observed)."""
    bank = {}
    for site, series in residuals.items():
        weeks = []
        for start in range(0, len(series) - WEEK_ROWS, WEEK_ROWS):
            block = series[start : start + WEEK_ROWS]
            if np.isfinite(block).mean() >= 0.9:
                weeks.append(np.nan_to_num(block, nan=0.0))
        bank[site] = weeks
    return bank


def heating_design(
    *, cache: FeatureCache, layout: GamLayout, label: str, cut: datetime = FLEET_CUT
) -> np.ndarray:
    """Return the heating columns of the design over the substation's two-year window at the cut.

    The gate measures shrinkage on the heating contribution to the prediction over these rows,
    because the four smoothing windows of a band are nearly collinear: their coefficients can be
    shrunk hard while the predictions barely move.
    """
    features = cache.sites[label]
    index = training_rows(features=features, cache=cache, cut=cut, window=common.FULL_WINDOW)
    x = design_matrix(cache=cache, features=features, layout=layout, index=index)
    return x[:, layout.slices["heating"]]


def contribution_slope(
    *, estimated: dict[str, np.ndarray], truth: dict[str, np.ndarray], design: dict[str, np.ndarray]
) -> float:
    """Return the slope of the estimated heating contribution on the true one, through the origin.

    Args:
        estimated: The estimated deviation of each substation's heating coefficients.
        truth: The true deviation of each substation's heating coefficients.
        design: Each substation's heating design columns over its two-year window.

    Returns:
        The pooled slope over every substation's rows.
    """
    numerator = sum(
        float(((design[s] @ truth[s]) * (design[s] @ estimated[s])).sum()) for s in truth
    )
    denominator = sum(float(((design[s] @ truth[s]) ** 2).sum()) for s in truth)
    return numerator / denominator


def band_totals(*, heating: np.ndarray) -> np.ndarray:
    """Return the heating slopes summed over the smoothing windows, one per hour band."""
    return heating.reshape(len(SMOOTHING_HOURS), N_BANDS).sum(axis=0)


def slope_through_origin(*, estimated: np.ndarray, truth: np.ndarray) -> float:
    """Return the least-squares slope of `estimated` on `truth` through the origin."""
    return float((estimated * truth).sum() / (truth * truth).sum())


def _fleet_fit(
    *,
    source: arms.Source,
    variant: str,
    strength: PoolingStrength,
    held_out: str | None,
    history: common.History | None,
    alone: bool = False,
    ridge: float | None = None,
    rebuild_scale: bool = True,
    cut: datetime = FLEET_CUT,
) -> tuple[Any, arms.Built]:
    """Fit the synthetic fleet at the 2025-07-01 cut, optionally with one short-history site."""
    cache, vintages = arms.load_source(source=source)
    config = arms.ArmConfig(
        name="gate",
        strength=strength,
        history=history,
        tag=f"gate_{variant}",
        pool=SYNTHETIC_LABELS,
        alone=alone,
        short_ridge=ridge,
        rebuild_scale=rebuild_scale,
    )
    built = arms.build_arm(
        cache=cache,
        vintages=vintages,
        cut=cut,
        config=config,
        held_out=held_out,
        history=history,
    )
    joint = fit_pooled(cache=built.cache, sites=built.sites, cut=cut, strengths=[strength])
    return joint, built


def fleet_family(*, joint: Any, layout: GamLayout, family: str) -> np.ndarray:
    """Return one pooled family's slice of the fitted fleet mean."""
    offset = 0
    for name in POOLED_FAMILIES:
        size = layout.slices[name].stop - layout.slices[name].start
        if name == family:
            return joint.fleet_mean[0][offset : offset + size]
        offset += size
    msg = f"{family} is not a pooled family"
    raise ValueError(msg)


def capacity_check(
    *,
    source: arms.Source,
    variant: str,
    truth: dict[str, Any],
    tuned: dict[str, Any],
    strength: PoolingStrength,
) -> dict[str, Any]:
    """Return line (e): the capacity recovered from a short window, pooled and per site."""
    ratios: dict[str, list[float]] = {"4w": [], "3m": [], "g_4w": [], "g_3m": []}
    for label in SYNTHETIC_LABELS:
        for key, history in (("4w", common.FOUR_WEEKS), ("3m", common.THREE_MONTHS)):
            joint, built = _fleet_fit(
                source=source,
                variant=variant,
                strength=strength,
                held_out=label,
                history=history,
            )
            ratios[key].append(_capacity_ratio(joint=joint, built=built, truth=truth, label=label))
            alone, alone_built = _fleet_fit(
                source=source,
                variant=variant,
                strength=PoolingStrength.uniform(value=0.0),
                held_out=label,
                history=history,
                alone=True,
                ridge=tuned["g_short_ridge"][history.label],
            )
            ratios[f"g_{key}"].append(
                _capacity_ratio(joint=alone, built=alone_built, truth=truth, label=label)
            )
    within = {key: int(sum(abs(r - 1.0) <= 0.25 for r in values)) for key, values in ratios.items()}
    return {
        "passed": bool(within["4w"] >= 6 and within["3m"] >= 6),
        "within_25_percent_of_8": within,
        "ratios": ratios,
    }


def _capacity_ratio(*, joint: Any, built: arms.Built, truth: dict[str, Any], label: str) -> float:
    """Return the held-out site's fitted capacity over its true capacity, in original points."""
    fit = joint.fits[label]
    fitted = fit.capacity_points()[fit.last_season_year]
    transform = built.transform
    original = fitted / (transform.scale if transform is not None else 1.0)
    return float(original / truth["capacity_points"][label])


def build_variant(
    *, cache: FeatureCache, vintages: pl.DataFrame, directory: Path, seed: int
) -> tuple[arms.Source, dict[str, Any]]:
    """Build one synthetic fleet, save it, and return its source and its truth.

    Args:
        cache: The real cache.
        vintages: The real capacity vintages.
        directory: Where the fleet's files are saved.
        seed: The seed of the fleet.

    Returns:
        The source of the saved synthetic cache and vintages, and the truth: per-site coefficients,
        the family scales, the noise standard deviation, and the pinned capacities in points.
    """
    generator = np.random.default_rng(seed)
    multiplier = 1.0
    joint, residuals = _g_fits(cache=cache, vintages=vintages)
    layout = joint.fits[REAL_SITES[0]].layout
    mean_fit, scale, truths = _family_truth(
        joint=joint, layout=layout, multiplier=multiplier, generator=generator
    )
    bank = _noise_bank(residuals=residuals)
    n_rows = len(cache.end_times)
    sites = {}
    clean: dict[str, np.ndarray] = {}
    vintage_rows = []
    capacity_truth = {}
    first_week, last_week = date(2019, 9, 2), date(2026, 6, 29)
    weeks = pl.date_range(first_week, last_week, "1w", eager=True)
    for number, label in enumerate(SYNTHETIC_LABELS):
        base = cache.sites[WEATHER_SITES[number % len(WEATHER_SITES)]]
        real_pinned = pinned_capacity(vintages=vintages, table=base.table, cut=FLEET_CUT)
        points = real_pinned.held_points
        capacity_truth[label] = points
        table = replace(base.table, site=label, time_series_id=9000 + number)
        usable = np.flatnonzero(
            np.isfinite(base.pv_unit_shape) & np.isfinite(base.smoothed_temperature_c).all(axis=1)
        )
        y = np.full(n_rows, np.nan)
        for start in range(0, len(usable), 20000):
            rows = usable[start : start + 20000]
            y[rows] = design_matrix(cache=cache, features=base, layout=layout, index=rows) @ truths[
                number
            ] - points * derated_unit_shape(features=base, index=rows)
        donors = [site for site in bank if site != WEATHER_SITES[number % len(WEATHER_SITES)]]
        clean[label] = y.copy()
        for start in range(0, n_rows, WEEK_ROWS):
            donor = bank[donors[int(generator.integers(len(donors)))]]
            block = donor[int(generator.integers(len(donor)))]
            stop = min(start + WEEK_ROWS, n_rows)
            y[start:stop] = y[start:stop] + block[: stop - start]
        sites[label] = replace(
            base,
            table=table,
            y=y,
            present=np.ones(n_rows, dtype=bool),
            switching_mask=np.zeros(n_rows, dtype=bool),
            in_pool=np.ones(n_rows, dtype=bool),
            default_rows=cache.sites["D11"].default_rows.copy(),
        )
        vintage_rows.append(
            pl.DataFrame(
                {
                    "time_series_id": 9000 + number,
                    "variant": "frozen",
                    "origin_monday": date(2019, 9, 2),
                    "week_start": weeks,
                    "capacity_mw": points * table.effective_capacity_mw / 100.0,
                    "unobserved": False,
                }
            ).with_columns(pl.col("time_series_id").cast(pl.Int32))
        )
    synthetic = replace(
        cache,
        key=f"gate_seed{seed}",
        sites=sites,
        node_ids={label: (label,) for label in SYNTHETIC_LABELS},
    )
    directory.mkdir(parents=True, exist_ok=True)
    cache_path = directory / "fleet.npz"
    vintages_path = directory / "fleet_vintages.parquet"
    np.savez(directory / TRUTH_FILE, **clean)  # ty: ignore[invalid-argument-type]
    save_feature_cache(cache=synthetic, path=cache_path)
    pl.concat(vintage_rows).write_parquet(vintages_path)
    noise_variance = float(
        np.nanvar(np.concatenate([np.concatenate(w) for w in bank.values() if w]))
    )
    truth = {
        "thetas": truths,
        "mean_fit": mean_fit,
        "scale": {family: multiplier * value for family, value in scale.items()},
        "noise_variance": noise_variance,
        "capacity_points": capacity_truth,
        "layout": layout,
    }
    return arms.Source(cache_path=cache_path, vintages_path=vintages_path), truth


def fair_oracle(
    *, cache: FeatureCache, layout: GamLayout, label: str, truth: dict[str, Any], cut: datetime
) -> tuple[np.ndarray, float]:
    """Return the fair oracle's heating slopes and capacity for one substation's 4 weeks.

    The oracle is the posterior mean of a Gaussian model. It knows the realised mean of the eight
    substations' true coefficients (the fleet mean after the heating slopes were clipped at zero),
    each family's true scale `tau_f`, and the noise variance (independent noise of the true
    variance). It estimates every coefficient jointly from the same rows as the pooled fit, with
    the capacity free (a flat prior), so no coefficient is fixed at its true value. The prior on
    each family's deviations has precision `1 / tau_f^2` (on the profile's shape only, so the band
    means stay free).

    Args:
        cache: The synthetic cache.
        layout: The design's columns, with the capacity pinned (no capacity column).
        label: The substation.
        truth: The truth of the synthetic fleet.
        cut: The cut.

    Returns:
        The posterior mean of the heating slopes and of the capacity in points.
    """
    features = cache.sites[label]
    window = common.FOUR_WEEKS.window(cut=cut)
    index = training_rows(features=features, cache=cache, cut=cut, window=window)
    x = design_matrix(cache=cache, features=features, layout=layout, index=index)
    shape = derated_unit_shape(features=features, index=index)
    # The target is `x @ theta - capacity * shape`, so the capacity's column is `-shape`.
    z = np.column_stack([x, -shape])
    n_columns = layout.n_columns
    prior_mean = np.zeros(n_columns + 1)
    precision = np.zeros((n_columns + 1, n_columns + 1))
    for family in POOLED_FAMILIES:
        block = layout.slices[family]
        centre = centring_matrix(family=family, size=block.stop - block.start)
        prior_mean[block] = np.mean([theta[block] for theta in truth["thetas"]], axis=0)
        precision[block, block] = centre.T @ centre / truth["scale"][family] ** 2
    variance = truth["noise_variance"]
    posterior = np.linalg.solve(
        z.T @ z / variance + precision + 1e-6 * np.eye(n_columns + 1),
        z.T @ features.y[index] / variance + precision @ prior_mean,
    )
    return posterior[layout.slices["heating"]], float(posterior[-1])


def four_week_heating(
    *,
    source: arms.Source,
    variant: str,
    truth: dict[str, Any],
    strength: PoolingStrength,
    designs: dict[str, np.ndarray],
    with_oracle: bool,
    cut: datetime = FLEET_CUT,
) -> dict[str, Any]:
    """Return the pooled fit's shrinkage of the heating slopes on 4 June weeks, and the oracle's.

    Each substation takes a turn as the one that holds only its last 4 weeks; the other seven keep
    their two years. The scale is not rebuilt, to isolate shrinkage.

    Args:
        source: The synthetic fleet.
        variant: Names the fleet, so stored replicas never mix.
        truth: The fleet's truth.
        strength: The pooled fit's penalty.
        designs: Each substation's heating design over its two-year window.
        with_oracle: True to also compute the fair oracle.
        cut: The cut; the 4 weeks held are the 4 weeks before it.

    Returns:
        The slope of the fitted deviation on the true deviation (on the heating contribution), the
        heating root mean square error, and, with the oracle, its slope, error, and capacity ratios.
    """
    layout: GamLayout = truth["layout"]
    heating = layout.slices["heating"]
    true_heating = np.stack([theta[heating] for theta in truth["thetas"]])
    true_mean = true_heating.mean(axis=0)
    true_by_site = dict(zip(SYNTHETIC_LABELS, true_heating - true_mean, strict=True))
    deltas, errors = {}, []
    oracle_delta, oracle_errors, capacity_ratios = {}, [], []
    for number, label in enumerate(SYNTHETIC_LABELS):
        joint, built = _fleet_fit(
            source=source,
            variant=variant,
            strength=strength,
            held_out=label,
            history=common.FOUR_WEEKS,
            rebuild_scale=False,
            cut=cut,
        )
        fitted = joint.fits[label].theta[0, heating]
        deltas[label] = fitted - fleet_family(joint=joint, layout=layout, family="heating")
        errors.append(fitted - true_heating[number])
        if with_oracle:
            slopes, capacity = fair_oracle(
                cache=built.cache, layout=layout, label=label, truth=truth, cut=cut
            )
            oracle_delta[label] = slopes - true_mean
            oracle_errors.append(slopes - true_heating[number])
            capacity_ratios.append(capacity / truth["capacity_points"][label])
    result: dict[str, Any] = {
        "slope_on_heating_contribution": contribution_slope(
            estimated=deltas, truth=true_by_site, design=designs
        ),
        "rmse": float(np.sqrt((np.stack(errors) ** 2).mean())),
    }
    if with_oracle:
        result["oracle_slope_on_heating_contribution"] = contribution_slope(
            estimated=oracle_delta, truth=true_by_site, design=designs
        )
        result["oracle_rmse"] = float(np.sqrt((np.stack(oracle_errors) ** 2).mean()))
        result["oracle_capacity_ratios"] = capacity_ratios
    return result


def passes_line_b(*, run: dict[str, Any], oracle: dict[str, Any]) -> bool:
    """Return True if the slope is near the oracle's and the error within 1.2 times its."""
    return bool(
        abs(run["slope_on_heating_contribution"] - oracle["oracle_slope_on_heating_contribution"])
        <= SLOPE_TOLERANCE
        and run["rmse"] <= RMSE_FACTOR * oracle["oracle_rmse"]
    )


def shrinkage_checks(
    *, source: arms.Source, variant: str, truth: dict[str, Any], tuned: dict[str, Any]
) -> dict[str, Any]:
    """Return lines (a), (b) with its controls, (c), and (e)."""
    layout: GamLayout = truth["layout"]
    heating = layout.slices["heating"]
    strength = PoolingStrength.uniform(value=tuned["lambda_star"])
    cache, _ = arms.load_source(source=source)
    true_heating = np.stack([theta[heating] for theta in truth["thetas"]])
    true_mean = true_heating.mean(axis=0)
    true_delta = true_heating - true_mean

    full, _ = _fleet_fit(
        source=source, variant=variant, strength=strength, held_out=None, history=None
    )
    band_true = band_totals(heating=true_mean)
    band_hat = band_totals(heating=fleet_family(joint=full, layout=layout, family="heating"))
    band_error = np.abs(band_hat - band_true) / np.abs(band_true)
    profile = layout.slices["profile"]
    centre = centring_matrix(family="profile", size=profile.stop - profile.start)
    true_profile = np.mean([centre @ theta[profile] for theta in truth["thetas"]], axis=0)
    fitted_profile = centre @ fleet_family(joint=full, layout=layout, family="profile")
    profile_rms = float(np.sqrt(((fitted_profile - true_profile) ** 2).mean()))
    checks: dict[str, Any] = {
        "a_fleet_mean": {
            "passed": bool((band_error <= 0.10).all() and profile_rms <= 0.5),
            "band_relative_error": band_error.tolist(),
            "profile_rms_points": profile_rms,
        }
    }
    designs = {
        label: heating_design(cache=cache, layout=layout, label=label) for label in SYNTHETIC_LABELS
    }
    mu_heating = fleet_family(joint=full, layout=layout, family="heating")
    full_delta = {s: full.fits[s].theta[0, heating] - mu_heating for s in SYNTHETIC_LABELS}
    true_by_site = dict(zip(SYNTHETIC_LABELS, true_delta, strict=True))
    slope_full = contribution_slope(estimated=full_delta, truth=true_by_site, design=designs)
    checks["c_full_history_keeps_own"] = {
        "passed": slope_full >= 0.85,
        "slope_on_heating_contribution": slope_full,
    }

    pooled = four_week_heating(
        source=source,
        variant=variant,
        truth=truth,
        strength=strength,
        designs=designs,
        with_oracle=True,
    )
    pooled["passed"] = passes_line_b(run=pooled, oracle=pooled)
    controls = {}
    for name, value in CONTROL_LAMBDAS.items():
        control = four_week_heating(
            source=source,
            variant=variant,
            truth=truth,
            strength=PoolingStrength.uniform(value=value),
            designs=designs,
            with_oracle=False,
        )
        control["passes_line_b"] = passes_line_b(run=control, oracle=pooled)
        controls[name] = control
    checks["b_shrinkage_on_4_weeks"] = {
        "passed": pooled["passed"],
        "reference": "fair oracle: joint posterior mean, capacity estimated, true mu, tau, sigma",
        "pooled": pooled,
        "controls": controls,
        "controls_fail": bool(not any(c["passes_line_b"] for c in controls.values())),
    }
    checks["e_capacity_from_a_short_window"] = capacity_check(
        source=source, variant=variant, truth=truth, tuned=tuned, strength=strength
    )
    return checks


def tuning_check(*, tuned: dict[str, Any]) -> dict[str, Any]:
    """Return line (d): the tuned penalty is within one grid step of the truth-best penalty."""
    grid = list(common.LAMBDA_GRID)
    chosen, best = tuned["lambda_star"], tuned["lambda_truth_best"]
    steps = abs(grid.index(chosen) - grid.index(best))
    return {
        "d_tuning_within_one_step": {
            "passed": bool(steps <= 1),
            "lambda_star": chosen,
            "lambda_truth_best": best,
            "grid_steps_apart": steps,
            "lambda_star_mae": tuned["lambda_star_mae"],
            "own_best": tuned["own_best"],
        }
    }


def winter_line_b(*, seed: int) -> dict[str, Any]:
    """Return line (b) and its controls with the 4 held weeks in January (post hoc).

    The 4 June weeks of the pre-registered line carry almost no heating signal, so this reruns
    the line with the cut at 2025-02-01, using the seed's tuned penalty and the same fleet.
    """
    directory = GATE_DIR / f"seed{seed}"
    variant = f"seed{seed}"
    cut = common.WINTER_CUT
    cache, vintages = arms.load_source(source=arms.Source())
    source, truth = build_variant(cache=cache, vintages=vintages, directory=directory, seed=seed)
    lam = json.loads((GATE_DIR / f"gate_seed{seed}.json").read_text())["lambda_star"]
    fleet, _ = arms.load_source(source=source)
    layout: GamLayout = truth["layout"]
    designs = {
        label: heating_design(cache=fleet, layout=layout, label=label, cut=cut)
        for label in SYNTHETIC_LABELS
    }
    pooled = four_week_heating(
        source=source,
        variant=f"{variant}_winter",
        truth=truth,
        strength=PoolingStrength.uniform(value=lam),
        designs=designs,
        with_oracle=True,
        cut=cut,
    )
    pooled["passed"] = passes_line_b(run=pooled, oracle=pooled)
    controls = {}
    for name, value in CONTROL_LAMBDAS.items():
        control = four_week_heating(
            source=source,
            variant=f"{variant}_winter",
            truth=truth,
            strength=PoolingStrength.uniform(value=value),
            designs=designs,
            with_oracle=False,
            cut=cut,
        )
        control["passes_line_b"] = passes_line_b(run=control, oracle=pooled)
        controls[name] = control
    return {"seed": seed, "lambda": lam, "pooled": pooled, "controls": controls}


def run_seed(*, seed: int, workers: int) -> dict[str, Any]:
    """Build one seed's fleet, tune on it, run every line, and return the results."""
    directory = GATE_DIR / f"seed{seed}"
    variant = f"seed{seed}"
    cache, vintages = arms.load_source(source=arms.Source())
    source, truth = build_variant(cache=cache, vintages=vintages, directory=directory, seed=seed)
    jobs = tune.grid_jobs(
        source=source,
        cut_name="tuning",
        pool=SYNTHETIC_LABELS,
        decision=TUNING_DECISION,
        tag=f"gate_{variant}",
        truth_path=directory / TRUTH_FILE,
    )
    table = tune.run_grid(jobs=jobs, workers=workers)
    table.write_parquet(directory / "tuning.parquet")
    tuned = tune.choose(table=table)
    print(f"seed {seed}: lambda* = {tuned['lambda_star']:g}, best = {tuned['lambda_truth_best']:g}")
    checks = shrinkage_checks(source=source, variant=variant, truth=truth, tuned=tuned)
    checks.update(tuning_check(tuned=tuned))
    return {"seed": seed, "lambda_star": tuned["lambda_star"], "checks": checks}


def summarise() -> dict[str, Any]:
    """Count each line's passes over the seeds that have run, and apply the decision rule."""
    runs = [json.loads(p.read_text()) for p in sorted(GATE_DIR.glob("gate_seed*.json"))]
    lines = {
        "a": lambda r: r["checks"]["a_fleet_mean"]["passed"],
        "b": lambda r: r["checks"]["b_shrinkage_on_4_weeks"]["passed"],
        "b_controls_fail": lambda r: r["checks"]["b_shrinkage_on_4_weeks"]["controls_fail"],
        "c": lambda r: r["checks"]["c_full_history_keeps_own"]["passed"],
        "d": lambda r: r["checks"]["d_tuning_within_one_step"]["passed"],
        "e": lambda r: r["checks"]["e_capacity_from_a_short_window"]["passed"],
    }
    counts = {name: sum(bool(f(r)) for r in runs) for name, f in lines.items()}
    for control in CONTROL_LAMBDAS:
        counts[f"control_{control}_passes_b"] = sum(
            r["checks"]["b_shrinkage_on_4_weeks"]["controls"][control]["passes_line_b"]
            for r in runs
        )
    n = len(runs)
    if n < len(SEEDS):
        return {"n_seeds": n, "verdict": "incomplete: not every seed has run"}
    required = 4
    gate = all(counts[k] >= required for k in ("a", "b", "c", "e")) and (
        counts["b_controls_fail"] >= required
    )
    stable = counts["d"] >= required
    cannot_tell = any(counts[f"control_{c}_passes_b"] >= 2 for c in CONTROL_LAMBDAS)
    verdict = (
        "inconclusive"
        if cannot_tell
        else "supported"
        if gate and stable
        else "partly supported"
        if gate or stable
        else "not supported"
    )
    result = {
        "n_seeds": n,
        "pass_counts": counts,
        "gate_passed": gate,
        "tuning_stable": stable,
        "verdict": verdict,
        "per_seed": {r["seed"]: {name: bool(f(r)) for name, f in lines.items()} for r in runs},
    }
    (GATE_DIR / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Run one seed of the gate, or summarise the seeds that have run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--summarise", action="store_true")
    parser.add_argument("--winter", action="store_true", help="post hoc line (b) in January")
    arguments = parser.parse_args()
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    if arguments.summarise:
        print(json.dumps(summarise(), indent=2))
        return
    if arguments.winter and arguments.seed is not None:
        result = winter_line_b(seed=arguments.seed)
        path = GATE_DIR / f"winter_seed{arguments.seed}.json"
        path.write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(json.dumps(result, indent=2, default=str))
        return
    if arguments.seed is None:
        parser.error("give --seed or --summarise")
    started = datetime.now(UTC)
    result = run_seed(seed=arguments.seed, workers=arguments.workers)
    result["minutes"] = (datetime.now(UTC) - started) / timedelta(minutes=1)
    path = GATE_DIR / f"gate_seed{arguments.seed}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, default=str) + "\n")
    temporary.replace(path)
    print(json.dumps(result["checks"], indent=2, default=str))
    print(f"seed {arguments.seed} done in {result['minutes']:.1f} minutes")


if __name__ == "__main__":
    main()
