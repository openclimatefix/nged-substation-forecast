"""The known-answer gate: can the pooled fit and its tuning rule recover a fleet whose truth we set?

**Blocking.** No replicate-year or scored-year fit runs until every blocking line passes. The gate
is synthetic and contains no generator data.

**The synthetic fleet.** Eight synthetic substations on the real calendar and weather of the MW
Primaries (D5, D10, D11, D14, D16, with the first three reused), two years each. The truth of
substation `i` is the fleet mean `mu` plus a deviation `delta[i, f]` drawn normal with family scale
`tau_f`. `mu` is the per-site convex model's (G's) fleet average at the 2025-07-01 cut over the six
real MW Primaries, and `tau_f` is G's between-site spread in family `f`. Heating slopes are clipped
at zero, and the profile keeps a real substation's band means. The capacity is pinned and synthetic,
with solar output from CAMS irradiance and no farm data. The noise is whole weeks block-bootstrapped
from G's training-window residuals at a different substation, never from a replica or scored year,
so it keeps the multi-day persistence that makes `lambda` hard to tune.

**Lines (blocking unless marked).**

- (a) Fleet mean: the fitted `mu`'s heating slopes (summed over the four smoothing windows) are
  within 10% of the mean of the eight true slopes in every band, and its centred profile is within
  0.5 points root mean square.
- (b) Shrinkage on 4 June weeks: the slope of the fitted deviation on the true deviation over the
  heating coefficients is within 0.15 of the closed-form shrinkage factor, and the heating root
  mean square error across substations is at most 1.2 times the oracle posterior mean's.
- (c) Full history keeps its own coefficients: with 2 years that slope is at least 0.85.
- (d) The tuning rule responds to the fleet: with every `tau_f` ten times larger the tuned `lambda`
  falls at least tenfold and the all-rows score stays within 0.02 points of G's; with every
  `tau_f` a tenth the tuned `lambda` rises at least tenfold.
- (e) Capacity from a short window: the 4-week and the 3-month pooled fits recover the held-out
  substation's capacity within 25% at 6 or more of 8 substations. G's ratios are reported.
- Reported, not blocking: G-4w's heating error, and the 4-week pooled fit against G-4w with every
  `tau_f` a tenth (these pass by construction).

    uv run python ml-experiments/00109-*/e00109_gate.py [--workers 6]
"""

import argparse
import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import e00109_arms as arms
import e00109_common as common
import e00109_tune as tune
import numpy as np
import polars as pl
from studies.dp_calendar import HALF_HOURS_PER_DAY, SMOOTHING_HOURS
from studies.dp_features import FeatureCache, save_feature_cache
from studies.dp_gam import (
    N_BANDS,
    GamLayout,
    TrainingWindow,
    derated_unit_shape,
    design_matrix,
    predict_gam,
    training_rows,
)
from studies.dp_gam_pooled import (
    POOLED_FAMILIES,
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
VARIANTS: Final[dict[str, float]] = {"base": 1.0, "tau_x10": 10.0, "tau_x0.1": 0.1}
WEEK_ROWS: Final[int] = 7 * HALF_HOURS_PER_DAY
FLEET_CUT: Final[datetime] = common.SCORED_CUT
SEED: Final[int] = 20261019


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

    def block(family: str) -> slice:
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


def build_variant(
    *, cache: FeatureCache, vintages: pl.DataFrame, variant: str, multiplier: float
) -> tuple[arms.Source, dict[str, Any]]:
    """Build one synthetic fleet, save it, and return its source and its truth.

    Args:
        cache: The real cache.
        vintages: The real capacity vintages.
        variant: The variant's name, used in file names.
        multiplier: The factor on every family scale `tau_f`.

    Returns:
        The source of the saved synthetic cache and vintages, and the truth: per-site coefficients,
        the family scales, the noise standard deviation, and the pinned capacities in points.
    """
    generator = np.random.default_rng(SEED)
    joint, residuals = _g_fits(cache=cache, vintages=vintages)
    layout = joint.fits[REAL_SITES[0]].layout
    mean_fit, scale, truths = _family_truth(
        joint=joint, layout=layout, multiplier=multiplier, generator=generator
    )
    bank = _noise_bank(residuals=residuals)
    n_rows = len(cache.end_times)
    sites = {}
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
        key=f"gate_{variant}",
        sites=sites,
        node_ids={label: (label,) for label in SYNTHETIC_LABELS},
    )
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = GATE_DIR / f"{variant}.npz"
    vintages_path = GATE_DIR / f"{variant}_vintages.parquet"
    save_feature_cache(cache=synthetic, path=cache_path)
    pl.concat(vintage_rows).write_parquet(vintages_path)
    noise_variance = float(
        np.nanvar(np.concatenate([np.concatenate(w) for w in bank.values() if w]))
    )
    truth = {
        "thetas": truths,
        "mean_fit": mean_fit,
        "scale": scale,
        "noise_variance": noise_variance,
        "capacity_points": capacity_truth,
        "layout": layout,
    }
    return arms.Source(cache_path=cache_path, vintages_path=vintages_path), truth


def heating_design(*, cache: FeatureCache, layout: GamLayout, label: str) -> np.ndarray:
    """Return the heating columns of the design over the substation's two-year window at the cut.

    The gate measures shrinkage on the heating contribution to the prediction over these rows,
    because the four smoothing windows of a band are nearly collinear: their coefficients can be
    shrunk hard while the predictions barely move.
    """
    features = cache.sites[label]
    index = training_rows(features=features, cache=cache, cut=FLEET_CUT, window=common.FULL_WINDOW)
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


def posterior_factor_and_oracle(
    *,
    cache: FeatureCache,
    layout: GamLayout,
    label: str,
    truth_theta: np.ndarray,
    mean_heating: np.ndarray,
    tau_heating: float,
    noise_variance: float,
    window: TrainingWindow,
    cut: datetime,
    pinned_points: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the closed-form shrinkage matrix and the oracle posterior-mean heating slopes.

    The oracle knows every other coefficient, the fleet mean of the heating slopes, `tau`, and the
    noise variance.

    Args:
        cache: The synthetic cache.
        layout: The design's columns.
        label: The substation.
        truth_theta: The substation's true coefficients.
        mean_heating: The fleet mean of the heating slopes the oracle shrinks towards.
        tau_heating: The true family scale of the heating slopes.
        noise_variance: The noise variance.
        window: The substation's window.
        cut: The cut.
        pinned_points: The synthetic capacity in points.

    Returns:
        The shrinkage matrix `S` (so the posterior mean is `S theta_ols + (I - S) mean`), and the
        posterior mean.
    """
    features = cache.sites[label]
    index = training_rows(features=features, cache=cache, cut=cut, window=window)
    x = design_matrix(cache=cache, features=features, layout=layout, index=index)
    heating = layout.slices["heating"]
    rest = np.ones(layout.n_columns, dtype=bool)
    rest[heating] = False
    residual = (
        features.y[index]
        + pinned_points * derated_unit_shape(features=features, index=index)
        - x[:, rest] @ truth_theta[rest]
    )
    x_h = x[:, heating]
    precision = x_h.T @ x_h / noise_variance
    prior = np.eye(heating.stop - heating.start) / tau_heating**2
    posterior = np.linalg.solve(
        precision + prior, x_h.T @ residual / noise_variance + prior @ mean_heating
    )
    shrink = np.linalg.solve(precision + prior, precision)
    return shrink, posterior


def run_checks(*, workers: int) -> dict[str, Any]:
    """Run every line of the gate and return the results.

    Args:
        workers: Worker processes for the tuning grids.

    Returns:
        A dict with each line's numbers and `passed`.
    """
    cache, vintages = arms.load_source(source=arms.Source())
    sources, truths, tuned = {}, {}, {}
    for variant, multiplier in VARIANTS.items():
        sources[variant], truths[variant] = build_variant(
            cache=cache, vintages=vintages, variant=variant, multiplier=multiplier
        )
        jobs = tune.grid_jobs(
            source=sources[variant],
            cut_name="tuning",
            pool=SYNTHETIC_LABELS,
            decision=TUNING_DECISION,
            tag=f"gate_{variant}",
        )
        table = tune.run_grid(jobs=jobs, workers=workers)
        table.write_parquet(GATE_DIR / f"tuning_{variant}.parquet")
        tuned[variant] = tune.choose(table=table)
        print(f"{variant}: lambda* = {tuned[variant]['lambda_star']:g}", flush=True)
    result = {"lambda_star": {v: t["lambda_star"] for v, t in tuned.items()}}
    result["checks"] = shrinkage_checks(sources=sources, truths=truths, tuned=tuned)
    result["checks"].update(tuning_checks(tuned=tuned))
    blocking = [
        value["passed"] for key, value in result["checks"].items() if value.get("blocking", True)
    ]
    result["passed"] = all(blocking)
    return result


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
        cut=FLEET_CUT,
        config=config,
        held_out=held_out,
        history=history,
    )
    joint = fit_pooled(cache=built.cache, sites=built.sites, cut=FLEET_CUT, strengths=[strength])
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


def shrinkage_checks(
    *, sources: dict[str, arms.Source], truths: dict[str, Any], tuned: dict[str, Any]
) -> dict[str, Any]:
    """Return lines (a), (b), (c), (e), and the reported discrimination checks."""
    truth = truths["base"]
    layout: GamLayout = truth["layout"]
    heating = layout.slices["heating"]
    strength = PoolingStrength.uniform(value=tuned["base"]["lambda_star"])
    true_heating = np.stack([theta[heating] for theta in truth["thetas"]])
    true_mean = true_heating.mean(axis=0)
    true_delta = true_heating - true_mean
    base_source = sources["base"]

    full, _ = _fleet_fit(
        source=base_source, variant="base", strength=strength, held_out=None, history=None
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
    mu_heating = fleet_family(joint=full, layout=layout, family="heating")
    designs = {
        label: heating_design(cache=cache, layout=layout, label=label) for label in SYNTHETIC_LABELS
    }
    true_by_site = dict(zip(SYNTHETIC_LABELS, true_delta, strict=True))
    full_delta = {s: full.fits[s].theta[0, heating] - mu_heating for s in SYNTHETIC_LABELS}
    slope_full = contribution_slope(estimated=full_delta, truth=true_by_site, design=designs)
    slope_full_coefficients = slope_through_origin(
        estimated=np.stack([full_delta[s] for s in SYNTHETIC_LABELS]), truth=true_delta
    )
    checks["c_full_history_keeps_own"] = {
        "passed": slope_full >= 0.85,
        "slope_on_heating_contribution": slope_full,
        "slope_on_coefficients_reported": slope_full_coefficients,
    }

    deltas, errors, oracle_errors, shrinks = {}, [], [], {}
    ratios: dict[str, list[float]] = {"4w": [], "3m": [], "g_4w": [], "g_3m": []}
    for number, label in enumerate(SYNTHETIC_LABELS):
        joint, built = _fleet_fit(
            source=base_source,
            variant="base",
            strength=strength,
            held_out=label,
            history=common.FOUR_WEEKS,
            rebuild_scale=False,
        )
        fitted = joint.fits[label].theta[0, heating]
        deltas[label] = fitted - fleet_family(joint=joint, layout=layout, family="heating")
        errors.append(fitted - true_heating[number])
        shrink, posterior = posterior_factor_and_oracle(
            cache=built.cache,
            layout=layout,
            label=label,
            truth_theta=truth["thetas"][number],
            mean_heating=true_mean,
            tau_heating=truth["scale"]["heating"],
            noise_variance=truth["noise_variance"],
            window=common.FOUR_WEEKS.window(cut=FLEET_CUT),
            cut=FLEET_CUT,
            pinned_points=truth["capacity_points"][label],
        )
        shrinks[label] = shrink
        oracle_errors.append(posterior - true_heating[number])
    slope_short = contribution_slope(estimated=deltas, truth=true_by_site, design=designs)
    expected = {s: shrinks[s] @ true_by_site[s] for s in SYNTHETIC_LABELS}
    closed_form = contribution_slope(estimated=expected, truth=true_by_site, design=designs)
    rmse = float(np.sqrt((np.stack(errors) ** 2).mean()))
    oracle_rmse = float(np.sqrt((np.stack(oracle_errors) ** 2).mean()))
    checks["b_shrinkage_on_4_weeks"] = {
        "passed": bool(abs(slope_short - closed_form) <= 0.15 and rmse <= 1.2 * oracle_rmse),
        "slope_on_heating_contribution": slope_short,
        "closed_form_slope": closed_form,
        "rmse": rmse,
        "oracle_rmse": oracle_rmse,
        "note": "scale not rebuilt, to isolate shrinkage",
    }
    for label in SYNTHETIC_LABELS:
        for key, history in (("4w", common.FOUR_WEEKS), ("3m", common.THREE_MONTHS)):
            joint, built = _fleet_fit(
                source=base_source,
                variant="base",
                strength=strength,
                held_out=label,
                history=history,
            )
            ratios[key].append(_capacity_ratio(joint=joint, built=built, truth=truth, label=label))
            alone, alone_built = _fleet_fit(
                source=base_source,
                variant="base",
                strength=PoolingStrength.uniform(value=0.0),
                held_out=label,
                history=history,
                alone=True,
                ridge=tuned["base"]["g_short_ridge"][history.label],
            )
            ratios[f"g_{key}"].append(
                _capacity_ratio(joint=alone, built=alone_built, truth=truth, label=label)
            )
    within = {key: int(sum(abs(r - 1.0) <= 0.25 for r in values)) for key, values in ratios.items()}
    checks["e_capacity_from_a_short_window"] = {
        "passed": bool(within["4w"] >= 6 and within["3m"] >= 6),
        "within_25_percent_of_8": within,
        "ratios": ratios,
    }
    checks["discrimination_g_4w_heating"] = {
        "passed": True,
        "blocking": False,
        "note": "reported only; passes by construction",
        "g_4w_heating_rmse": _g_short_heating_rmse(
            source=base_source, variant="base", truth=truth, tuned=tuned["base"]
        ),
    }
    checks["discrimination_small_tau"] = {
        "passed": True,
        "blocking": False,
        "note": "reported only; passes by construction",
        **_small_tau_discrimination(
            source=sources["tau_x0.1"],
            truth=truths["tau_x0.1"],
            lam=tuned["tau_x0.1"]["lambda_star"],
            ridge=tuned["tau_x0.1"]["g_short_ridge"]["4w"],
        ),
    }
    return checks


def _capacity_ratio(*, joint: Any, built: arms.Built, truth: dict[str, Any], label: str) -> float:
    """Return the held-out site's fitted capacity over its true capacity, in original points."""
    fit = joint.fits[label]
    fitted = fit.capacity_points()[fit.last_season_year]
    transform = built.transform
    original = fitted / (transform.scale if transform is not None else 1.0)
    return float(original / truth["capacity_points"][label])


def _g_short_heating_rmse(
    *, source: arms.Source, variant: str, truth: dict[str, Any], tuned: dict[str, Any]
) -> float:
    """Return G-4w's heating root mean square error across the synthetic substations."""
    layout: GamLayout = truth["layout"]
    heating = layout.slices["heating"]
    errors = []
    for number, label in enumerate(SYNTHETIC_LABELS):
        alone, _ = _fleet_fit(
            source=source,
            variant=variant,
            strength=PoolingStrength.uniform(value=0.0),
            held_out=label,
            history=common.FOUR_WEEKS,
            alone=True,
            ridge=tuned["g_short_ridge"]["4w"],
        )
        errors.append(alone.fits[label].theta[0, heating] - truth["thetas"][number][heating])
    return float(np.sqrt((np.stack(errors) ** 2).mean()))


def _small_tau_discrimination(
    *, source: arms.Source, truth: dict[str, Any], lam: float, ridge: float
) -> dict[str, float]:
    """Return the 4-week heating error of the pooled and the per-site fit at a tenth of tau."""
    layout: GamLayout = truth["layout"]
    heating = layout.slices["heating"]
    pooled, single = [], []
    for number, label in enumerate(SYNTHETIC_LABELS):
        joint, _ = _fleet_fit(
            source=source,
            variant="tau_x0.1",
            strength=PoolingStrength.uniform(value=lam),
            held_out=label,
            history=common.FOUR_WEEKS,
        )
        alone, _ = _fleet_fit(
            source=source,
            variant="tau_x0.1",
            strength=PoolingStrength.uniform(value=0.0),
            held_out=label,
            history=common.FOUR_WEEKS,
            alone=True,
            ridge=ridge,
        )
        pooled.append(joint.fits[label].theta[0, heating] - truth["thetas"][number][heating])
        single.append(alone.fits[label].theta[0, heating] - truth["thetas"][number][heating])
    return {
        "n_4w_heating_rmse": float(np.sqrt((np.stack(pooled) ** 2).mean())),
        "g_4w_heating_rmse": float(np.sqrt((np.stack(single) ** 2).mean())),
    }


def tuning_checks(*, tuned: dict[str, Any]) -> dict[str, Any]:
    """Return line (d): the tuning rule responds to the fleet's between-site spread."""
    base = tuned["base"]["lambda_star"]
    large = tuned["tau_x10"]
    small = tuned["tau_x0.1"]["lambda_star"]
    large_row = next(r for r in large["summary"] if r["lam"] == large["lambda_star"])
    return {
        "d_tuning_responds": {
            "passed": bool(
                large["lambda_star"] * 10.0 <= base
                and large_row["all_delta"] <= 0.02
                and small >= base * 10.0
            ),
            "lambda_base": base,
            "lambda_tau_x10": large["lambda_star"],
            "lambda_tau_x0.1": small,
            "all_rows_delta_tau_x10": large_row["all_delta"],
        }
    }


def main() -> None:
    """Run the gate and write `work/gate/gate.json`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    result = run_checks(workers=arguments.workers)
    result["minutes"] = (datetime.now(UTC) - started) / timedelta(minutes=1)
    (GATE_DIR / "gate.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result["checks"], indent=2, default=str))
    print(f"gate passed: {result['passed']} in {result['minutes']:.1f} minutes")


if __name__ == "__main__":
    main()
