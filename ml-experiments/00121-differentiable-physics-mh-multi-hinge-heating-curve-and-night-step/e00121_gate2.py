"""Gate 2 of experiment 121 (MH power through the filter) and the pooled-curve coverage check.

**The synthetic substations.** The truth is B0's real fit at the 2024-07-01 cut with its heating
block replaced by a five-hinge curve: each of B0's 16 heating slopes is spread over the hinges at
10, 12, 14, 16, and 18 degrees C with shares `(0.2 a, 0.2 a, 0.2, 0.2, 0.2)`, where the amplitude
`a` scales the cold-end hinges (10 and 12). **The single-hinge truth** is B0's fit itself (all the
slope at 15.5 degrees C). The noise is whole donor months of the other decision substations' B0
residuals (`e00121_noise`), inflated by the site's ratio of out-of-sample to in-sample RMSE for B0
on the tuning year, on the training rows and on the held-out year alike, so no 2024-25 residual
enters. The synthetic target is net power plus the pinned solar output, so the solar term cancels.

**Every score runs through the filter.** An arm is fitted at the 2023-07-01 and 2024-07-01 cuts on
the synthetic target. Its replica residuals (the 2023-24 year) give the post-filter lead spread,
and its 2024-25 residuals run through the Kalman filter with B0-K's tuned hyperparameters
(`decisions.json`). The oracle forecast is the true mean, with the filter run on the noise alone.

**The amplitude is set on 5 pilot draws** (one per substation, seeds apart from the main draws) by
bisection so that the oracle gain is 0.03 points: oracle gain = (score of B0-K minus score of the
oracle with B0-K's filter) under the five-hinge truth, minus the same difference under the
single-hinge truth.

**Pass** if (a) the paired mean of MH-K minus B0-K over the 100 draws (20 per substation) under
the five-hinge truth is at most -0.015 and its 80% interval over draws lies wholly below zero, and
(b) under the single-hinge truth the paired mean is at most +0.01. **Pooled-curve coverage:** in
the five-hinge draws, the 80% month-block interval of the fitted heating response at 5 degrees C
divided by the response at 12 degrees C (summed over the four bands and four smoothings) is
checked for coverage of the truth's value.

    uv run python ml-experiments/00121-*/e00121_gate2.py run [--workers 5]
    uv run python ml-experiments/00121-*/e00121_gate2.py summary
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Final

import e00121_common as common
import e00121_design as design
import e00121_noise as noise
import e00121_pipeline as pipeline
import numpy as np
import polars as pl
import torch
from studies.dp_assimilation import (
    DEFAULT_DELAY_HOURS,
    FilterHyperparameters,
    correction_at,
    day_ahead_initialisation,
    post_filter_lead_spread,
    usable_cutoff_positions,
)
from studies.dp_features import FeatureCache
from studies.dp_gam import HEATING_SCALE_C, GamLayout, ReplicaResiduals, ResidualSpread
from studies.dp_gam import design_matrix as gam_design_matrix
from studies.dp_pinned_capacity import read_vintages

N_DRAWS_PER_SITE: Final[int] = 20
N_PILOT_DRAWS: Final[int] = 5
TARGET_GAIN_POINTS: Final[float] = 0.03
MAX_FIVE_HINGE_MEAN: Final[float] = -0.015
MAX_SINGLE_HINGE_LOSS: Final[float] = 0.01
AMPLITUDE_BRACKET: Final[tuple[float, float]] = (0.0, 40.0)
BISECTION_STEPS: Final[int] = 12
GATE_SEED_TAG: Final[int] = 2
BOOTSTRAP_SEED_TAG: Final[int] = 7
PILOT_DRAW_ID: Final[int] = 99
N_BOOTSTRAP_DRAWS: Final[int] = 2000
COVERAGE_RANGE: Final[tuple[float, float]] = (0.70, 0.90)
INTERVAL_PERCENTILES: Final[tuple[float, float]] = (10.0, 90.0)
RESPONSE_COLD_C: Final[float] = 5.0
RESPONSE_REFERENCE_C: Final[float] = 12.0
MIN_RESPONSE: Final[float] = 1e-9

_STATE: dict[Any, Any] = {}


def verified_decisions() -> dict[str, Any]:
    """Return `decisions.json` after checking its hash against `decisions.sha256`.

    Returns:
        The decisions.

    Raises:
        ValueError: If the file no longer matches its recorded hash.
    """
    import hashlib

    text = (common.WORK_DIR / "decisions.json").read_text()
    recorded = (common.WORK_DIR / "decisions.sha256").read_text().strip()
    if hashlib.sha256(text.encode()).hexdigest() != recorded:
        msg = "decisions.json no longer matches the hash recorded before the gates"
        raise ValueError(msg)
    return json.loads(text)


def stratified_counts(
    *, quarter_of_month: np.ndarray, n_resamples: int, rng: np.random.Generator
) -> np.ndarray:
    """Return month counts of a month-block bootstrap that keeps every calendar quarter's size."""
    counts = np.zeros((n_resamples, len(quarter_of_month)), dtype=np.float64)
    for quarter in range(4):
        members = np.flatnonzero(quarter_of_month == quarter)
        if len(members) == 0:
            continue
        draws = rng.integers(0, len(members), size=(n_resamples, len(members)))
        for resample in range(n_resamples):
            counts[resample, members] = np.bincount(draws[resample], minlength=len(members))
    return counts


def hinge_shares(*, amplitude: float) -> np.ndarray:
    """Return the share of each B0 heating slope that goes to each of the five hinges."""
    return np.array([0.2 * amplitude, 0.2 * amplitude, 0.2, 0.2, 0.2])


def response_at(*, thresholds: np.ndarray, coefficients: np.ndarray, temperature_c: float) -> float:
    """Return the summed heating response at a temperature, in the units of the design.

    Args:
        thresholds: The hinge threshold of each coefficient block, shape (n_blocks,).
        coefficients: The coefficients, shape (n_blocks, 16).
        temperature_c: The temperature.

    Returns:
        The sum over blocks and columns of `coefficient * relu(threshold - T) / scale`.
    """
    ramp = np.clip(thresholds - temperature_c, 0.0, None) / HEATING_SCALE_C
    return float((coefficients.sum(axis=1) * ramp).sum())


def response_ratio(*, thresholds: np.ndarray, coefficients: np.ndarray) -> float:
    """Return the response at 5 degrees C over the response at 12 degrees C, or NaN."""
    cold = response_at(
        thresholds=thresholds, coefficients=coefficients, temperature_c=RESPONSE_COLD_C
    )
    reference = response_at(
        thresholds=thresholds, coefficients=coefficients, temperature_c=RESPONSE_REFERENCE_C
    )
    return cold / reference if abs(reference) > MIN_RESPONSE else float("nan")


@dataclass
class SiteSetup:
    """Everything about one substation that does not change from draw to draw.

    Attributes:
        site: The substation's label.
        layout: B0's columns.
        theta_b0: B0's real coefficients at the main cut.
        inflation: The ratio of B0's out-of-sample to in-sample RMSE on the tuning year.
        rows: The union of every row set, ascending.
        sets: The grid rows of each row set.
        positions: The position in `rows` of each row set.
        matrices: Each arm's design on each row set.
        truth_matrix: The five-hinge truth's design (MH-5's columns) on `rows`.
        grams: Each arm's Gram matrix on the two training sets.
        penalties: Each arm's penalty on the two training sets.
        mh_months: The main training set's per-month Gram matrices for MH.
        mh_quarters: The quarter label of each month in `mh_months`.
        mh_month_of_row: The month, as an index into `mh_months`, of each training row.
        hyper: B0-K's filter hyperparameters at the site.
    """

    site: str
    layout: GamLayout
    theta_b0: np.ndarray
    inflation: float
    rows: np.ndarray
    sets: dict[str, np.ndarray]
    positions: dict[str, np.ndarray]
    matrices: dict[tuple[str, str], np.ndarray]
    truth_matrix: np.ndarray
    grams: dict[tuple[str, str], torch.Tensor]
    penalties: dict[tuple[str, str], design.ArmPenalty]
    mh_months: design.MonthGrams
    mh_quarters: np.ndarray
    mh_month_of_row: np.ndarray
    hyper: FilterHyperparameters


def build_setup(
    *, cache: FeatureCache, vintages: Any, site: str, hyper_values: list[float]
) -> SiteSetup:
    """Build one substation's fixed synthetic-data inputs."""
    main = design.build_base(cache=cache, vintages=vintages, site=site, cut=common.REPLICATE_CUT)
    rep = design.build_base(cache=cache, vintages=vintages, site=site, cut=common.TUNING_CUT)
    theta_b0 = design.fit_arm(cache=cache, base=main, arm=design.B0).theta
    inflation = noise.tuning_inflation(cache=cache, vintages=vintages, site=site)
    test_rep = common.scoring_index(
        cache=cache,
        site=site,
        after=common.TUNING_CUT,
        through=common.year_end(cut=common.TUNING_CUT),
    )
    sets = {
        "train_main": main.index,
        "train_rep": rep.index,
        "test_main": common.scoring_index(
            cache=cache,
            site=site,
            after=common.REPLICATE_CUT,
            through=common.year_end(cut=common.REPLICATE_CUT),
        ),
        "test_rep": test_rep,
    }
    rows = np.unique(np.concatenate(list(sets.values())))
    positions = {name: np.searchsorted(rows, index) for name, index in sets.items()}
    features = cache.sites[site]
    matrices: dict[tuple[str, str], np.ndarray] = {}
    grams: dict[tuple[str, str], torch.Tensor] = {}
    penalties: dict[tuple[str, str], design.ArmPenalty] = {}
    for arm_name in ("b0", "mh"):
        arm = design.all_arms()[arm_name]
        for name, index in sets.items():
            x0 = gam_design_matrix(cache=cache, features=features, layout=main.layout, index=index)
            matrices[(arm_name, name)] = design.arm_matrix(
                cache=cache, layout=main.layout, site=site, index=index, x0=x0, arm=arm
            )
        for name, base in (("train_main", main), ("train_rep", rep)):
            x = matrices[(arm_name, name)]
            xt = torch.as_tensor(x)
            grams[(arm_name, name)] = xt.T @ xt
            penalties[(arm_name, name)] = design.penalty_for(cache=cache, base=base, arm=arm, x=x)
    all_x0 = gam_design_matrix(cache=cache, features=features, layout=main.layout, index=rows)
    matrices[("b0", "all")] = all_x0
    truth_matrix = design.arm_matrix(
        cache=cache, layout=main.layout, site=site, index=rows, x0=all_x0, arm=design.MH5
    )
    mh_months = design.MonthGrams.build(
        x=matrices[("mh", "train_main")],
        month_of_row=main.month_of_row,
        n_months=len(main.months),
    )
    return SiteSetup(
        site=site,
        layout=main.layout,
        theta_b0=theta_b0,
        inflation=inflation,
        rows=rows,
        sets=sets,
        positions=positions,
        matrices=matrices,
        truth_matrix=truth_matrix,
        grams=grams,
        penalties=penalties,
        mh_months=mh_months,
        mh_quarters=main.quarter_of_month,
        mh_month_of_row=main.month_of_row,
        hyper=FilterHyperparameters.from_array(values=np.array(hyper_values)),
    )


def five_hinge_theta(*, setup: SiteSetup, amplitude: float) -> np.ndarray:
    """Return the truth's coefficients in MH-5's columns (B0's kept columns, then five blocks)."""
    heating = setup.layout.slices["heating"]
    keep = [j for j in range(len(setup.theta_b0)) if not heating.start <= j < heating.stop]
    slopes = setup.theta_b0[heating]
    blocks = [share * slopes for share in hinge_shares(amplitude=amplitude)]
    return np.concatenate([setup.theta_b0[keep], *blocks])


def fit_means(
    *, setup: SiteSetup, arm_name: str, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit an arm to the synthetic target at both cuts; return the two forecasts and the main theta.

    Args:
        setup: The substation's inputs.
        arm_name: `b0` or `mh`.
        z: The synthetic target on every row of the setup.

    Returns:
        The replica year's mean, the held-out year's mean, and the coefficients at the main cut.
    """
    thetas = {}
    for cut_name in ("rep", "main"):
        x_train = setup.matrices[(arm_name, f"train_{cut_name}")]
        target = z[setup.positions[f"train_{cut_name}"]]
        gram = setup.grams[(arm_name, f"train_{cut_name}")][None]
        rhs = torch.as_tensor(x_train.T @ target)[None]
        thetas[cut_name] = design.solve(
            gram=gram, rhs=rhs, penalty=setup.penalties[(arm_name, f"train_{cut_name}")]
        )[0]
    rep_mean = setup.matrices[(arm_name, "test_rep")] @ thetas["rep"]
    main_mean = setup.matrices[(arm_name, "test_main")] @ thetas["main"]
    return rep_mean, main_mean, thetas["main"]


def scores(
    *,
    cache: FeatureCache,
    setup: SiteSetup,
    z: np.ndarray,
    rep_mean: np.ndarray,
    main_mean: np.ndarray,
) -> tuple[float, float]:
    """Return the unfiltered and the filtered mean quantile score on the held-out year.

    Args:
        cache: The feature cache.
        setup: The substation's inputs.
        z: The synthetic target on every row of the setup.
        rep_mean: The replica year's mean forecast.
        main_mean: The held-out year's mean forecast.

    Returns:
        The two scores, outside the switching masks.
    """
    rep_index = setup.sets["test_rep"]
    main_index = setup.sets["test_main"]
    features = cache.sites[setup.site]
    band = cache.calendar.band
    quarter = cache.calendar.quarter
    residual = z[setup.positions["test_rep"]] - rep_mean
    replica = ReplicaResiduals(
        residuals=residual,
        band=band[rep_index],
        quarter=quarter[rep_index],
        index=rep_index,
        fit_cut=common.TUNING_CUT,
        switching=features.switching_mask[rep_index],
    )
    keep_rep = ~replica.switching
    spread = ResidualSpread.fit(
        residuals=residual[keep_rep],
        band=band[rep_index][keep_rep],
        quarter=quarter[rep_index][keep_rep],
        centre=True,
    )
    actual = z[setup.positions["test_main"]]
    keep = ~features.switching_mask[main_index]
    plain = spread.quantiles(mean=main_mean, band=band[main_index], quarter=quarter[main_index])
    build = pipeline.trace_builder(hyper=setup.hyper)
    lead = post_filter_lead_spread(
        end_times=cache.end_times, replica=replica, run=build, delay_hours=DEFAULT_DELAY_HOURS
    )
    stream = np.full(len(cache.end_times), np.nan)
    stream[rep_index] = residual
    stream[main_index] = actual - main_mean
    trace = build(residual=stream, band=band)
    cutoff = usable_cutoff_positions(
        end_times=cache.end_times,
        initialisation=day_ahead_initialisation(
            end_times=cache.end_times[main_index], days_ahead=1
        ),
        delay_hours=DEFAULT_DELAY_HOURS,
    )
    correction = correction_at(trace=trace, cutoff=cutoff, target=main_index, band=band[main_index])
    filtered_mean = main_mean + correction.total
    filtered = lead.quantiles(
        mean=filtered_mean, band=band[main_index], quarter=quarter[main_index], lead_band=0
    )
    return (
        common.as_float(common.quantile_score(actual=actual[keep], quantiles=plain[keep]).mean()),
        common.as_float(
            common.quantile_score(actual=actual[keep], quantiles=filtered[keep]).mean()
        ),
    )


def arm_scores(*, setup: SiteSetup, arm_name: str, z: np.ndarray) -> tuple[float, float]:
    """Fit an arm and return its unfiltered and filtered scores."""
    rep_mean, main_mean, _ = fit_means(setup=setup, arm_name=arm_name, z=z)
    return scores(cache=_STATE["cache"], setup=setup, z=z, rep_mean=rep_mean, main_mean=main_mean)


def oracle_scores(
    *, setup: SiteSetup, truth_mean: np.ndarray, noise_values: np.ndarray, z: np.ndarray
) -> tuple[float, float]:
    """Return the true-parameter forecast's scores, with the filter and spread run on the noise."""
    return scores(
        cache=_STATE["cache"],
        setup=setup,
        z=z,
        rep_mean=z[setup.positions["test_rep"]] - noise_values[setup.positions["test_rep"]],
        main_mean=truth_mean[setup.positions["test_main"]],
    )


def draw_noise(*, setup: SiteSetup, site_position: int, draw: int) -> np.ndarray:
    """Draw the inflated real noise for one draw."""
    rng = np.random.default_rng([common.EXPERIMENT_ID, GATE_SEED_TAG, site_position, draw])
    values = _STATE["pool"].draw(site=setup.site, index=setup.rows, rng=rng)
    return setup.inflation * values


def truths(*, setup: SiteSetup, amplitude: float) -> dict[str, np.ndarray]:
    """Return the two truths' mean on every row of the setup."""
    return {
        "five": setup.truth_matrix @ five_hinge_theta(setup=setup, amplitude=amplitude),
        "single": setup.matrices[("b0", "all")] @ setup.theta_b0,
    }


def pooled_curve(*, setup: SiteSetup, z: np.ndarray, amplitude: float, seed: int) -> dict[str, Any]:
    """Return MH's pooled response ratio at 5 and 12 degrees C and its 80% month-block interval."""
    thresholds = np.array([common.BASE_THRESHOLD_C, *common.MH_THRESHOLDS_C])
    p0 = setup.matrices[("b0", "train_main")].shape[1]
    heating = setup.layout.slices["heating"]
    x = setup.matrices[("mh", "train_main")]
    target = z[setup.positions["train_main"]]
    rng = np.random.default_rng([common.EXPERIMENT_ID, BOOTSTRAP_SEED_TAG, seed])
    counts = np.concatenate(
        [
            np.ones((1, len(setup.mh_quarters))),
            stratified_counts(
                quarter_of_month=setup.mh_quarters, n_resamples=common.N_BOOTSTRAP, rng=rng
            ),
        ]
    )
    cross = setup.mh_months.cross_products(x=x, target=target)
    theta = design.solve(
        gram=setup.mh_months.weighted(counts=counts),
        rhs=torch.as_tensor(counts) @ cross,
        penalty=setup.penalties[("mh", "train_main")],
    )

    def ratio(row: np.ndarray) -> float:
        blocks = np.vstack([row[heating], row[p0:].reshape(len(common.MH_THRESHOLDS_C), -1)])
        return response_ratio(thresholds=thresholds, coefficients=blocks)

    values = np.array([ratio(row) for row in theta])
    truth_blocks = np.array(
        [share * setup.theta_b0[heating] for share in hinge_shares(amplitude=amplitude)]
    )
    truth = response_ratio(thresholds=np.array(common.MH_THRESHOLDS_C), coefficients=truth_blocks)
    estimate = values[0]
    resamples = values[1:]
    widen = float(np.sqrt(len(setup.mh_quarters) / (len(setup.mh_quarters) - 1)))
    if not np.isfinite(resamples).all() or not np.isfinite(estimate):
        return {
            "truth": truth,
            "estimate": float(estimate),
            "low": float("nan"),
            "high": float("nan"),
        }
    low_p, high_p = np.percentile(resamples, INTERVAL_PERCENTILES)
    return {
        "truth": truth,
        "estimate": float(estimate),
        "low": float(estimate - widen * (estimate - low_p)),
        "high": float(estimate + widen * (high_p - estimate)),
    }


def one_draw(
    *, setup: SiteSetup, site_position: int, draw: int, amplitude: float
) -> dict[str, Any]:
    """Run one draw under both truths with the same noise and return the scores."""
    noise_values = draw_noise(setup=setup, site_position=site_position, draw=draw)
    out: dict[str, Any] = {"noise_sd": float(noise_values.std())}
    truth_means = truths(setup=setup, amplitude=amplitude)
    for kind, truth_mean in truth_means.items():
        z = truth_mean + noise_values
        for arm_name in ("b0", "mh"):
            plain, filtered = arm_scores(setup=setup, arm_name=arm_name, z=z)
            out[f"{kind}_{arm_name}"] = plain
            out[f"{kind}_{arm_name}_k"] = filtered
        plain, filtered = oracle_scores(
            setup=setup, truth_mean=truth_mean, noise_values=noise_values, z=z
        )
        out[f"{kind}_oracle"] = plain
        out[f"{kind}_oracle_k"] = filtered
    if amplitude > 0.0:
        z_five = truth_means["five"] + noise_values
        curve = pooled_curve(
            setup=setup, z=z_five, amplitude=amplitude, seed=site_position * 1000 + draw
        )
        out.update({f"curve_{key}": value for key, value in curve.items()})
    return out


def _init_worker() -> None:
    """Load the cache, vintages, donors, and decisions once per worker process."""
    common.configure_threads()
    cache = common.load_cache()
    _STATE["cache"] = cache
    _STATE["vintages"] = read_vintages()
    _STATE["pool"] = noise.build_pool(cache=cache)
    _STATE["decisions"] = verified_decisions()


def _setup(*, site: str) -> SiteSetup:
    """Build one substation's setup, cached for this process."""
    if site not in _STATE:
        _STATE[site] = build_setup(
            cache=_STATE["cache"],
            vintages=_STATE["vintages"],
            site=site,
            hyper_values=_STATE["decisions"]["filter_hyper"]["b0"][site],
        )
    return _STATE[site]


def _pilot_job(job: tuple[str, int, float]) -> float:
    """Return one pilot draw's oracle gain at an amplitude, net of the single-hinge truth's."""
    site, position, amplitude = job
    setup = _setup(site=site)
    noise_values = draw_noise(setup=setup, site_position=position, draw=PILOT_DRAW_ID)
    gaps = {}
    for kind, truth_mean in truths(setup=setup, amplitude=amplitude).items():
        z = truth_mean + noise_values
        _, b0_filtered = arm_scores(setup=setup, arm_name="b0", z=z)
        _, oracle_filtered = oracle_scores(
            setup=setup, truth_mean=truth_mean, noise_values=noise_values, z=z
        )
        gaps[kind] = b0_filtered - oracle_filtered
    return gaps["five"] - gaps["single"]


def calibrate_amplitude(*, pool: ProcessPoolExecutor) -> dict[str, float]:
    """Return the amplitude at which the pilot oracle gain equals `TARGET_GAIN_POINTS`."""

    def gain(amplitude: float) -> float:
        jobs = [(site, position, amplitude) for position, site in enumerate(common.DECISION_SITES)]
        return float(np.mean(list(pool.map(_pilot_job, jobs))))

    low, high = AMPLITUDE_BRACKET
    gain_low = gain(low)
    print(f"pilot oracle gain at amplitude {low}: {gain_low:.4f}", flush=True)
    if gain_low >= TARGET_GAIN_POINTS:
        msg = f"amplitude {low} already gives a pilot gain of {gain_low:.4f} points"
        raise ValueError(msg)
    gain_high = gain(high)
    if gain_high < TARGET_GAIN_POINTS:
        msg = f"even amplitude {high} gives a pilot gain of only {gain_high:.4f} points"
        raise ValueError(msg)
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (low + high)
        if gain(middle) < TARGET_GAIN_POINTS:
            low = middle
        else:
            high = middle
    amplitude = 0.5 * (low + high)
    return {"amplitude": amplitude, "pilot_gain": gain(amplitude)}


def _site_job(job: tuple[str, int, float]) -> list[dict[str, Any]]:
    """Run one substation's main draws."""
    site, position, amplitude = job
    setup = _setup(site=site)
    rows = []
    for draw in range(N_DRAWS_PER_SITE):
        scores_ = one_draw(setup=setup, site_position=position, draw=draw, amplitude=amplitude)
        rows.append({"site": site, "draw": draw, **scores_})
        print(f"{site} draw {draw} done", flush=True)
    return rows


def run(*, workers: int) -> None:
    """Calibrate the amplitude, run the 100 draws, and store them."""
    verified_decisions()
    common.load_cache()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        calibration = calibrate_amplitude(pool=pool)
        print(
            f"amplitude {calibration['amplitude']:.4f} gives a pilot gain of "
            f"{calibration['pilot_gain']:.4f}",
            flush=True,
        )
        (common.WORK_DIR / "gate2_amplitude.json").write_text(
            json.dumps(calibration, indent=2) + "\n"
        )
        jobs = [
            (site, position, calibration["amplitude"])
            for position, site in enumerate(common.DECISION_SITES)
        ]
        results = list(pool.map(_site_job, jobs))
    pl.DataFrame([row for site_rows in results for row in site_rows]).write_parquet(
        common.WORK_DIR / "gate2_draws.parquet"
    )


def summary() -> dict[str, Any]:
    """Print and return the gate's criteria from the stored draws."""
    frame = pl.read_parquet(common.WORK_DIR / "gate2_draws.parquet")
    calibration = json.loads((common.WORK_DIR / "gate2_amplitude.json").read_text())
    rng = np.random.default_rng(common.EXPERIMENT_ID)
    report: dict[str, Any] = {**calibration}
    for kind in ("five", "single"):
        difference = (frame[f"{kind}_mh_k"] - frame[f"{kind}_b0_k"]).to_numpy()
        resampled = rng.choice(difference, size=(N_BOOTSTRAP_DRAWS, len(difference))).mean(axis=1)
        gap_b0 = (frame[f"{kind}_b0_k"] - frame[f"{kind}_oracle_k"]).to_numpy()
        gap_mh = (frame[f"{kind}_mh_k"] - frame[f"{kind}_oracle_k"]).to_numpy()
        report[kind] = {
            "mean_mh_k_minus_b0_k": float(difference.mean()),
            "interval_80": [
                float(np.percentile(resampled, 10)),
                float(np.percentile(resampled, 90)),
            ],
            "unfiltered_mean_mh_minus_b0": common.as_float(
                (frame[f"{kind}_mh"] - frame[f"{kind}_b0"]).mean()
            ),
            "share_mh_k_better": float((difference < 0.0).mean()),
            "oracle_gap_b0_k": float(gap_b0.mean()),
            "oracle_gap_mh_k": float(gap_mh.mean()),
            "wins_by_site": {
                site: int(
                    (
                        frame.filter(pl.col("site") == site)[f"{kind}_mh_k"]
                        < frame.filter(pl.col("site") == site)[f"{kind}_b0_k"]
                    ).sum()
                )
                for site in common.DECISION_SITES
            },
        }

    def mean_of(column: str) -> float:
        return common.as_float(frame[column].mean())

    oracle_gain_k = (
        mean_of("five_b0_k")
        - mean_of("five_oracle_k")
        - mean_of("single_b0_k")
        + mean_of("single_oracle_k")
    )
    oracle_gain_plain = (
        mean_of("five_b0")
        - mean_of("five_oracle")
        - mean_of("single_b0")
        + mean_of("single_oracle")
    )
    report["oracle_gain_filtered"] = oracle_gain_k
    report["oracle_gain_unfiltered"] = oracle_gain_plain
    gained = -report["five"]["mean_mh_k_minus_b0_k"]
    report["share_of_oracle_gain_recovered"] = gained / report["oracle_gain_filtered"]
    report["a_pass"] = bool(
        report["five"]["mean_mh_k_minus_b0_k"] <= MAX_FIVE_HINGE_MEAN
        and report["five"]["interval_80"][1] < 0.0
    )
    report["b_pass"] = bool(report["single"]["mean_mh_k_minus_b0_k"] <= MAX_SINGLE_HINGE_LOSS)
    report["passed"] = bool(report["a_pass"] and report["b_pass"])
    covered = (frame["curve_low"] <= frame["curve_truth"]) & (
        frame["curve_truth"] <= frame["curve_high"]
    )
    report["pooled_curve_coverage"] = common.as_float(covered.fill_null(False).mean())
    report["pooled_curve_finite_share"] = common.as_float(frame["curve_low"].is_finite().mean())
    report["curve_coverage_in_range"] = bool(
        COVERAGE_RANGE[0] <= report["pooled_curve_coverage"] <= COVERAGE_RANGE[1]
    )
    report["noise_sd_mean"] = common.as_float(frame["noise_sd"].mean())
    (common.WORK_DIR / "gate2.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    """Run the draws or print the summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "summary"])
    parser.add_argument("--workers", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.action == "run":
        run(workers=arguments.workers)
    summary()


if __name__ == "__main__":
    main()
