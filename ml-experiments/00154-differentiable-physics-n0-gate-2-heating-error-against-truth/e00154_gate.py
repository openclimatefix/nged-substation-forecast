r"""The second known-answer gate: does the partly pooled convex model recover a known heating curve?

Experiment 154 scores the heating error against the injected truth on 4 winter weeks, the test that
experiment 150 (`../00150-*/e00150_gate.py`, whose helpers are copied here) proposed. The synthetic
fleet is experiment 150's, rebuilt from the same five seeds: eight synthetic substations on the real
calendar and weather of the MW Primaries, a known fleet-mean heating response, known per-substation
heating offsets, and noise taken as whole weeks of residuals from a different real substation. The
gate contains no generator data. The deviation penalty of each seed is the one experiment 150 tuned
(read from its stored gate file), so this script needs no tuning and no torch. One invocation runs
one seed:

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00154-*/e00154_gate.py --seed 20261101
    uv run --no-sync python ml-experiments/00154-*/e00154_gate.py --summarise

Each of the eight substations takes a turn as the one that holds only the 4 weeks before 2025-02-01;
the other seven keep two years. The heating error is the root mean square, over the winter rows
(November to March) of the held-out substation's two-year window, of the fitted heating contribution
minus the true one.
"""

import argparse
import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import e00154_arms as arms
import e00154_common as common
import numpy as np
import polars as pl
from studies.dp_calendar import HALF_HOURS_PER_DAY
from studies.dp_features import FeatureCache, save_feature_cache
from studies.dp_gam import (
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
WEEK_ROWS: Final[int] = 7 * HALF_HOURS_PER_DAY
FLEET_CUT: Final[datetime] = common.SCORED_CUT
SEEDS: Final[tuple[int, ...]] = (20261101, 20261102, 20261103, 20261104, 20261105)
"""The five seeds, fixed in the plan before any run."""
TRUTH_FILE: Final[str] = "truth.npz"
"""The noise-free series of each synthetic substation, saved beside its fleet."""
CONTROL_LAMBDAS: Final[dict[str, float]] = {"no_pooling": 0.0, "complete_pooling": 1_000_000.0}


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


WINTER_CUT: Final[datetime] = common.WINTER_CUT
WINTER_MONTHS: Final[tuple[int, ...]] = (11, 12, 1, 2, 3)
ORACLE_FACTOR: Final[float] = 1.1
"""W2: the pooled fit's heating error may exceed the fair oracle's by at most 10%."""
CURVE_FACTOR: Final[float] = 1.1
"""W1: the pooled fleet-curve error may exceed the per-site average's by at most 10%."""
SIGN_SITES_NEEDED: Final[int] = 6
"""W3: the pooled offset has the true sign at this many of the 8 substations."""
SEEDS_NEEDED: Final[int] = 4
ARMS: Final[tuple[str, ...]] = ("pooled", "no_pooling", "complete_pooling")
GATE_FILE_OF_EXPERIMENT_150: Final[Path] = (
    common.DATA_DIR.parent
    / "00150-differentiable-physics-n0-gate-rerun-fair-oracle-five-seeds"
    / "work"
    / "gate"
)
"""Where experiment 150 stored each seed's tuned deviation penalty (private, never committed)."""


def tuned_penalty(*, seed: int) -> float:
    """Return the deviation penalty that experiment 150 tuned on this seed's fleet."""
    path = GATE_FILE_OF_EXPERIMENT_150 / f"gate_seed{seed}.json"
    return float(json.loads(path.read_text())["lambda_star"])


def winter_design(*, cache: FeatureCache, layout: GamLayout, label: str) -> np.ndarray:
    """Return the heating columns over the winter rows of the substation's two-year window."""
    features = cache.sites[label]
    index = training_rows(features=features, cache=cache, cut=WINTER_CUT, window=common.FULL_WINDOW)
    months = pl.Series(cache.end_times[index]).dt.month().to_numpy()
    index = index[np.isin(months, WINTER_MONTHS)]
    x = design_matrix(cache=cache, features=features, layout=layout, index=index)
    return x[:, layout.slices["heating"]]


def contribution_rmse(*, design: np.ndarray, error: np.ndarray) -> float:
    """Return the root mean square heating contribution of a coefficient error, in points."""
    return float(np.sqrt(((design @ error) ** 2).mean()))


def contribution_mean(*, design: np.ndarray, coefficients: np.ndarray) -> float:
    """Return the mean heating contribution over the winter rows, in points."""
    return float((design @ coefficients).mean())


def run_seed(*, seed: int) -> dict[str, Any]:
    """Build one seed's fleet and score the heating error of every arm over the eight turns.

    Args:
        seed: One of `SEEDS`.

    Returns:
        Per held-out substation: each arm's heating error and offset, the fair oracle's, and the
        fleet-curve errors; plus the seed's penalty and the true offsets.
    """
    directory = GATE_DIR / f"seed{seed}"
    cache, vintages = arms.load_source(source=arms.Source())
    source, truth = build_variant(cache=cache, vintages=vintages, directory=directory, seed=seed)
    fleet, _ = arms.load_source(source=source)
    layout: GamLayout = truth["layout"]
    heating = layout.slices["heating"]
    lam = tuned_penalty(seed=seed)
    true_heating = np.stack([theta[heating] for theta in truth["thetas"]])
    true_mean = true_heating.mean(axis=0)
    designs = {s: winter_design(cache=fleet, layout=layout, label=s) for s in SYNTHETIC_LABELS}
    strengths = {
        "pooled": lam,
        "no_pooling": CONTROL_LAMBDAS["no_pooling"],
        "complete_pooling": CONTROL_LAMBDAS["complete_pooling"],
    }
    turns = []
    for number, label in enumerate(SYNTHETIC_LABELS):
        design = designs[label]
        true_offset = contribution_mean(
            design=design, coefficients=true_heating[number] - true_mean
        )
        turn: dict[str, Any] = {
            "held_out": label,
            "true_offset": true_offset,
            "true_contribution_mean": contribution_mean(
                design=design, coefficients=true_heating[number]
            ),
            "true_contribution_rms": float(np.sqrt(((design @ true_heating[number]) ** 2).mean())),
        }
        for arm, value in strengths.items():
            joint, built = _fleet_fit(
                source=source,
                variant=f"seed{seed}_e154",
                strength=PoolingStrength.uniform(value=value),
                held_out=label,
                history=common.FOUR_WEEKS,
                rebuild_scale=False,
                cut=WINTER_CUT,
            )
            fitted = joint.fits[label].theta[0, heating]
            mu_hat = fleet_family(joint=joint, layout=layout, family="heating")
            per_site_mean = np.mean(
                [joint.fits[s].theta[0, heating] for s in SYNTHETIC_LABELS], axis=0
            )
            curve = {
                "fleet_curve_rmse": float(
                    np.sqrt(
                        np.mean(
                            [
                                contribution_rmse(design=designs[s], error=mu_hat - true_mean) ** 2
                                for s in SYNTHETIC_LABELS
                            ]
                        )
                    )
                ),
                "per_site_mean_curve_rmse": float(
                    np.sqrt(
                        np.mean(
                            [
                                contribution_rmse(
                                    design=designs[s], error=per_site_mean - true_mean
                                )
                                ** 2
                                for s in SYNTHETIC_LABELS
                            ]
                        )
                    )
                ),
            }
            turn[arm] = {
                "heating_rmse": contribution_rmse(
                    design=design, error=fitted - true_heating[number]
                ),
                "coefficient_rmse": float(np.sqrt(((fitted - true_heating[number]) ** 2).mean())),
                "offset": contribution_mean(design=design, coefficients=fitted - mu_hat),
                **curve,
            }
            if arm == "pooled":
                slopes, capacity = fair_oracle(
                    cache=built.cache, layout=layout, label=label, truth=truth, cut=WINTER_CUT
                )
                turn["oracle"] = {
                    "heating_rmse": contribution_rmse(
                        design=design, error=slopes - true_heating[number]
                    ),
                    "coefficient_rmse": float(
                        np.sqrt(((slopes - true_heating[number]) ** 2).mean())
                    ),
                    "offset": contribution_mean(design=design, coefficients=slopes - true_mean),
                    "capacity_ratio": capacity / truth["capacity_points"][label],
                }
        turns.append(turn)
        print(f"seed {seed}: {label} done")
    return {"seed": seed, "lambda": lam, "turns": turns}


def score_seed(*, run: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered lines W1 to W3 to one seed's turns.

    Args:
        run: The output of `run_seed`.

    Returns:
        The seed's pooled-root-mean-square errors, the lines' pass flags, and the comparison
        figures each line uses.
    """

    def rms(arm: str) -> float:
        return float(np.sqrt(np.mean([t[arm]["heating_rmse"] ** 2 for t in run["turns"]])))

    def curve(arm: str, key: str) -> float:
        return float(np.sqrt(np.mean([t[arm][key] ** 2 for t in run["turns"]])))

    errors = {arm: rms(arm) for arm in (*ARMS, "oracle")}
    right_sign = sum(
        np.sign(t["pooled"]["offset"]) == np.sign(t["true_offset"]) for t in run["turns"]
    )
    offset_slope = float(
        sum(t["pooled"]["offset"] * t["true_offset"] for t in run["turns"])
        / sum(t["true_offset"] ** 2 for t in run["turns"])
    )
    return {
        "seed": run["seed"],
        "errors": errors,
        "fleet_curve_rmse": curve("pooled", "fleet_curve_rmse"),
        "per_site_mean_curve_rmse": curve("no_pooling", "per_site_mean_curve_rmse"),
        "sites_with_right_offset_sign": int(right_sign),
        "offset_slope": offset_slope,
        "W1_shared_curve": bool(
            curve("pooled", "fleet_curve_rmse")
            <= CURVE_FACTOR * curve("no_pooling", "per_site_mean_curve_rmse")
        ),
        "W2_close_to_fair_oracle": bool(errors["pooled"] <= ORACLE_FACTOR * errors["oracle"]),
        "W2b_beats_both_controls": bool(
            errors["pooled"] < errors["no_pooling"]
            and errors["pooled"] < errors["complete_pooling"]
        ),
        "W3_offset_sign": bool(right_sign >= SIGN_SITES_NEEDED and offset_slope > 0.0),
        "control_beats_pooled": [
            arm for arm in ("no_pooling", "complete_pooling") if errors[arm] <= errors["pooled"]
        ],
    }


def summarise() -> dict[str, Any]:
    """Score every seed that has run, count each line's passes, and apply the decision rule."""
    runs = [json.loads(p.read_text()) for p in sorted(GATE_DIR.glob("e154_seed*.json"))]
    scored = [score_seed(run=run) for run in runs]
    lines = (
        "W1_shared_curve",
        "W2_close_to_fair_oracle",
        "W2b_beats_both_controls",
        "W3_offset_sign",
    )
    counts = {line: sum(s[line] for s in scored) for line in lines}
    if len(scored) < len(SEEDS):
        return {"n_seeds": len(scored), "verdict": "incomplete: not every seed has run"}
    held = [line for line, count in counts.items() if count >= SEEDS_NEEDED]
    if counts["W2b_beats_both_controls"] < SEEDS_NEEDED:
        verdict = "not supported"
    elif len(held) == len(lines):
        verdict = "supported"
    elif len(held) == len(lines) - 1:
        verdict = "partly supported"
    else:
        verdict = "inconclusive"
    result = {"n_seeds": len(scored), "pass_counts": counts, "verdict": verdict, "per_seed": scored}
    (GATE_DIR / "e154_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Run one seed of the gate, or summarise the seeds that have run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--summarise", action="store_true")
    arguments = parser.parse_args()
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    if arguments.summarise:
        print(json.dumps(summarise(), indent=2))
        return
    if arguments.seed is None:
        parser.error("give --seed or --summarise")
    started = datetime.now(UTC)
    result = run_seed(seed=arguments.seed)
    result["minutes"] = (datetime.now(UTC) - started) / timedelta(minutes=1)
    path = GATE_DIR / f"e154_seed{arguments.seed}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, default=str) + "\n")
    temporary.replace(path)
    print(json.dumps(score_seed(run=result), indent=2, default=str))
    print(f"seed {arguments.seed} done in {result['minutes']:.1f} minutes")


if __name__ == "__main__":
    main()
