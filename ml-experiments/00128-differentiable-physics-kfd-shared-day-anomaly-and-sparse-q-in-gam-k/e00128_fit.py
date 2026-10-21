"""Fit the hyperparameters of every filtering arm of experiment 128 on the tuning months.

**KD** has six unconstrained values per substation, fitted together by maximum a posteriori under a
normal prior: log q (one level variance shared by the four bands), logit phi, log sigma_z, and log
sigma_eps with experiment 111's bounds and a prior whose mean and spread are the mean and spread of
experiment 111's six fitted substations (one round, spread at least 0.25), and logit phi_d and log
sigma_d with the bounds -3 to 2.2 and log 0.02 to log 6 and a weak prior (mean 0, spread 1.5) each.
The likelihood is the one-step marginal likelihood of the tuning months' readings, with the state
running warm through the other months of the stream cut at 2025-07-01. **KD-da** is the same fit by
the day-ahead likelihood of `studies.dp_effects_filter` (each tuning reading scored from the state
at its day-ahead cutoff).

**KS and KDS** give each band its own level variance `q_b`, found by a greedy search that starts
every `q_b` at exactly zero (`sparse_search`): it adds the band with the largest single-band gain in
the log likelihood over the grid `0.001 * 2**k` (k = 0 to 11) points squared per day while that gain
is at least the threshold, then re-optimises each active band on the grid and removes a band whose
removal costs less than the threshold. KS holds experiment 111's `phi`, `sigma_z`, and `sigma_eps`,
and switches d off. KDS holds all five other values of KD. The thresholds are 2 nats, with 1 and 4
for the sensitivity arms KDS-1n and KDS-4n. KDS-floor never lets a `q_b` fall below 1e-4. KDS-da
holds the five values of KD-da.

The script stores one JSON file per arm under `work/hyper/` and `work/fit_status.json`, which holds
the stop rules the plan names for the fits.

    uv run python ml-experiments/00128-*/e00128_fit.py [--workers 6] [--sites D5 D10 ...]
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Final

import e00128_common as common
import e00128_filter as joint
import e00128_streams as streams
import numpy as np
from scipy.optimize import minimize
from studies.dp_assimilation import BOUNDS as GAMK_BOUNDS
from studies.dp_assimilation import MIN_PRIOR_SPREAD
from studies.dp_effects_filter import DayAheadTargets, day_ahead_targets

HYPER_DIR: Final[Path] = common.WORK_DIR / "hyper"
FIT_STATUS_PATH: Final[Path] = common.WORK_DIR / "fit_status.json"
BOUNDS: Final[tuple[tuple[float, float], ...]] = (
    *GAMK_BOUNDS,
    (-3.0, 2.2),
    (log(0.02), log(6.0)),
)
"""Bounds on KD's six unconstrained values: experiment 111's four, then logit `phi_d` (0.05 to 0.90)
and log `sigma_d` (0.02 to 6 points)."""
SIGMA_D_FLOOR: Final[float] = 0.02
NEW_PRIOR_MEAN: Final[tuple[float, float]] = (0.0, 0.0)
"""logit(0.5) and log(1): `phi_d` of 0.5 and `sigma_d` of 1 point."""
NEW_PRIOR_SPREAD: Final[tuple[float, float]] = (1.5, 1.5)
FINITE_DIFFERENCE_STEP: Final[float] = 1e-4
MAX_ITERATIONS: Final[int] = 200
Q_GRID: Final[tuple[float, ...]] = tuple(0.001 * 2.0**k for k in range(12))
"""The level variances (points squared per day) the sparse search tries: 0.001 to about 2.05."""
THRESHOLD_NATS: Final[float] = 2.0
Q_FLOOR: Final[float] = 1e-4
SPARSE_ARMS: Final[dict[str, dict]] = {
    "KS": {"threshold": THRESHOLD_NATS, "base": "gk", "floor": 0.0},
    "KDS": {"threshold": THRESHOLD_NATS, "base": "KD", "floor": 0.0},
    "KDS-1n": {"threshold": 1.0, "base": "KD", "floor": 0.0},
    "KDS-4n": {"threshold": 4.0, "base": "KD", "floor": 0.0},
    "KDS-floor": {"threshold": THRESHOLD_NATS, "base": "KD", "floor": Q_FLOOR},
    "KDS-da": {"threshold": THRESHOLD_NATS, "base": "KD-da", "floor": 0.0},
}
"""Each sparse arm's threshold, the arm whose five other values it holds, and the floor on `q_b`."""
FIT_ARMS: Final[tuple[str, ...]] = ("KD", "KD-da", *SPARSE_ARMS)
"""Every arm with fitted hyperparameters, in the order the script fits them."""
SITES: Final[tuple[str, ...]] = common.SCORED_SITES


@dataclass(frozen=True)
class StreamInputs:
    """What a fit reads from a stream.

    Attributes:
        residual: The residual stream the filter runs on (cut at the fit end), `NaN` where absent.
        band: The hour band of each grid row.
        day: The local day of each grid row.
        likelihood_mask: True on the rows that count: finite residuals in the tuning months.
        day_ahead: The tuning readings with a day-ahead cutoff, or `None` if not built.
    """

    residual: np.ndarray
    band: np.ndarray
    day: np.ndarray
    likelihood_mask: np.ndarray
    day_ahead: DayAheadTargets | None = None


def stream_inputs(
    *, data: streams.SiteStreams, end_times: np.ndarray, with_day_ahead: bool = False
) -> StreamInputs:
    """Build a site's fit inputs from its real streams (cut at the fit end)."""
    from studies.dp_features import naive_utc

    residual = np.where(end_times <= naive_utc(moment=streams.FIT_END), data.residual, np.nan)
    mask = data.tuning_mask & np.isfinite(residual)
    return StreamInputs(
        residual=residual,
        band=data.band,
        day=data.day,
        likelihood_mask=mask,
        day_ahead=day_ahead_targets(end_times=end_times, rows=np.flatnonzero(mask))
        if with_day_ahead
        else None,
    )


def gamk_prior() -> tuple[np.ndarray, np.ndarray]:
    """Return the shared prior (mean and spread) of KD's six unconstrained values.

    The first four are the mean and standard deviation (spread at least 0.25) of experiment 111's
    fitted values over the six scored substations; the last two are the weak priors of the new
    values.
    """
    fitted = np.array([common.gamk_hyper(site=site) for site in common.SCORED_SITES])
    mean = np.r_[fitted.mean(axis=0), NEW_PRIOR_MEAN]
    spread = np.r_[np.maximum(fitted.std(axis=0, ddof=1), MIN_PRIOR_SPREAD), NEW_PRIOR_SPREAD]
    return mean, spread


@dataclass(frozen=True)
class FitResult:
    """One substation's fitted KD.

    Attributes:
        params: The fitted constants.
        vector: The six unconstrained values.
        log_posterior: The objective at the optimum (likelihood plus log prior).
        converged: Whether the optimiser reported success.
        message: The optimiser's message.
    """

    params: joint.JointParams
    vector: np.ndarray
    log_posterior: float
    converged: bool
    message: str


def fit_kd(
    *,
    inputs: StreamInputs,
    start: np.ndarray,
    prior_mean: np.ndarray,
    prior_spread: np.ndarray,
    objective: str = "one_step",
) -> FitResult:
    """Fit KD's six unconstrained values by maximum a posteriori.

    Args:
        inputs: The stream and the likelihood rows.
        start: The six starting values.
        prior_mean: The prior mean of the six values.
        prior_spread: The prior standard deviation of the six values.
        objective: `one_step` or `day_ahead` (needs `inputs.day_ahead`).

    Returns:
        The fit.

    Raises:
        ValueError: If the day-ahead objective is asked for without day-ahead targets, or the
            objective is unknown.
    """
    if objective not in ("one_step", "day_ahead"):
        msg = f"unknown objective {objective!r}"
        raise ValueError(msg)
    if objective == "day_ahead" and inputs.day_ahead is None:
        msg = "the day-ahead objective needs day-ahead targets"
        raise ValueError(msg)

    def negative_posterior(values: np.ndarray) -> float:
        trace = joint.run_joint_filter(
            residual=inputs.residual,
            band=inputs.band,
            day=inputs.day,
            params=joint.params_from_vector(values=values),
            likelihood_mask=inputs.likelihood_mask,
            keep_trace=False,
            day_ahead=inputs.day_ahead if objective == "day_ahead" else None,
        )
        likelihood = (
            trace.log_likelihood if objective == "one_step" else trace.day_ahead_log_likelihood
        )
        prior = -0.5 * float((((values - prior_mean) / prior_spread) ** 2).sum())
        return -(likelihood + prior)

    result = minimize(
        negative_posterior,
        x0=start,
        method="L-BFGS-B",
        bounds=BOUNDS,
        options={"eps": FINITE_DIFFERENCE_STEP, "maxiter": MAX_ITERATIONS},
    )
    if not np.isfinite(result.fun):
        msg = f"the hyperparameter fit returned a non-finite objective: {result.message}"
        raise ValueError(msg)
    return FitResult(
        params=joint.params_from_vector(values=result.x),
        vector=np.asarray(result.x, dtype=np.float64),
        log_posterior=-float(result.fun),
        converged=bool(result.success),
        message=str(result.message),
    )


def start_vector(*, site: str) -> np.ndarray:
    """Return KD's starting values: experiment 111's four, then `phi_d` 0.5 and `sigma_d` 1."""
    return np.r_[np.array(common.gamk_hyper(site=site)), NEW_PRIOR_MEAN]


@dataclass(frozen=True)
class SparseResult:
    """The outcome of one sparse search.

    Attributes:
        q: The level variance of each band; `floor` (0 by default) for a band left out.
        active: True for the bands the search kept.
        log_likelihood: The one-step log likelihood at `q`.
        base_log_likelihood: The log likelihood with every band at the floor.
        steps: One record per step of the search (the band added, its gain, the grid value).
        evaluations: How many filter passes the search ran.
    """

    q: np.ndarray
    active: np.ndarray
    log_likelihood: float
    base_log_likelihood: float
    steps: list[dict]
    evaluations: int


def sparse_search(
    *,
    inputs: StreamInputs,
    base: joint.JointParams,
    threshold: float,
    floor: float = 0.0,
    grid: tuple[float, ...] = Q_GRID,
) -> SparseResult:
    """Search for a sparse set of level variances, each zero unless the data justify it.

    Args:
        inputs: The stream and the likelihood rows.
        base: The constants held fixed (everything but `q`).
        threshold: The smallest log-likelihood gain, in nats, that admits or keeps a band.
        floor: The value a band takes when it is left out (0, or 1e-4 for KDS-floor).
        grid: The level variances tried for a band.

    Returns:
        The search's result.
    """
    cache: dict[tuple[float, ...], float] = {}

    def likelihood(q: np.ndarray) -> float:
        key = tuple(float(v) for v in q)
        if key not in cache:
            cache[key] = joint.run_joint_filter(
                residual=inputs.residual,
                band=inputs.band,
                day=inputs.day,
                params=base.with_q(q=q),
                likelihood_mask=inputs.likelihood_mask,
                keep_trace=False,
            ).log_likelihood
        return cache[key]

    def best_value(q: np.ndarray, band: int) -> tuple[float, float]:
        """Return the grid value of a band with the highest likelihood and that likelihood."""
        trials = []
        for value in grid:
            trial = q.copy()
            trial[band] = value
            trials.append((likelihood(trial), value))
        return max(trials)

    q = np.full(joint.N_BANDS, floor)
    active = np.zeros(joint.N_BANDS, dtype=bool)
    base_ll = current = likelihood(q)
    steps: list[dict] = []
    while not active.all():
        gains = {band: best_value(q, band) for band in range(joint.N_BANDS) if not active[band]}
        band, (best_ll, value) = max(gains.items(), key=lambda item: item[1][0])
        if best_ll - current < threshold:
            break
        q[band] = value
        active[band] = True
        steps.append({"band": band, "gain": best_ll - current, "q": value, "phase": "add"})
        current = best_ll
    for _ in range(2):
        for band in np.flatnonzero(active):
            best_ll, value = best_value(q, int(band))
            q[band] = value
            current = best_ll
        for band in np.flatnonzero(active):
            without = q.copy()
            without[band] = floor
            cost = current - likelihood(without)
            if cost < threshold:
                q = without
                active[band] = False
                current = likelihood(q)
                steps.append({"band": int(band), "gain": -cost, "q": floor, "phase": "remove"})
    return SparseResult(
        q=q,
        active=active,
        log_likelihood=current,
        base_log_likelihood=base_ll,
        steps=steps,
        evaluations=len(cache),
    )


def base_for(*, name: str, site: str, fitted: dict[str, FitResult]) -> joint.JointParams:
    """Return the constants a sparse arm holds: GAM-K's (d off) or a fitted KD's."""
    if name == "gk":
        return joint.gamk_params(hyper=common.gamk_hyper(site=site))
    return fitted[name].params


def fit_site(job: tuple[str, tuple[str, ...]]) -> dict:
    """Fit the arms at one substation.

    Args:
        job: The substation's label and the arms to fit.

    Returns:
        A map from arm name to its fitted constants and fit record.
    """
    site, wanted = job
    started = time.monotonic()
    cache = common.load_cache()
    data = streams.load_site_streams(site=site)
    inputs = stream_inputs(data=data, end_times=cache.end_times, with_day_ahead=True)
    prior_mean, prior_spread = gamk_prior()
    fitted: dict[str, FitResult] = {}
    out: dict = {}
    for name in ("KD", "KD-da"):
        if name in wanted or any(
            SPARSE_ARMS[arm]["base"] == name for arm in wanted if arm in SPARSE_ARMS
        ):
            fitted[name] = fit_kd(
                inputs=inputs,
                start=start_vector(site=site),
                prior_mean=prior_mean,
                prior_spread=prior_spread,
                objective="day_ahead" if name == "KD-da" else "one_step",
            )
    for name, result in fitted.items():
        if name in wanted:
            out[name] = {
                "params": result.params.to_json(),
                "log_posterior": result.log_posterior,
                "converged": result.converged,
                "message": result.message,
                "vector": result.vector.tolist(),
                "d_share_night": night_d_share(inputs=inputs, params=result.params),
            }
    for name, spec in SPARSE_ARMS.items():
        if name not in wanted:
            continue
        base = base_for(name=spec["base"], site=site, fitted=fitted)
        sparse = sparse_search(
            inputs=inputs, base=base, threshold=spec["threshold"], floor=spec["floor"]
        )
        out[name] = {
            "params": base.with_q(q=sparse.q).to_json(),
            "active": sparse.active.tolist(),
            "log_likelihood": sparse.log_likelihood,
            "base_log_likelihood": sparse.base_log_likelihood,
            "steps": sparse.steps,
            "evaluations": sparse.evaluations,
            "threshold": spec["threshold"],
            "floor": spec["floor"],
            "d_share_night": night_d_share(inputs=inputs, params=base.with_q(q=sparse.q)),
        }
    out["_seconds"] = time.monotonic() - started
    return {"site": site, "arms": out}


def night_d_share(*, inputs: StreamInputs, params: joint.JointParams) -> dict | None:
    """Return the quantiles of the share of each night innovation that goes to d.

    The share is d's Kalman gain over the sum of the gains of the night level, the night anomaly,
    and d, over the tuning months' night readings. `None` if d is off.
    """
    if params.sigma_d <= 0.0:
        return None
    trace = joint.run_joint_filter(
        residual=inputs.residual,
        band=inputs.band,
        day=inputs.day,
        params=params,
        likelihood_mask=inputs.likelihood_mask,
    )
    night = (inputs.band[trace.positions] == 0) & inputs.likelihood_mask[trace.positions]
    share = trace.d_share[night]
    return {
        "mean": float(share.mean()),
        "p10": float(np.percentile(share, 10)),
        "p50": float(np.percentile(share, 50)),
        "p90": float(np.percentile(share, 90)),
        "n": int(night.sum()),
    }


def hyper_path(*, arm: str) -> Path:
    """Return the file that holds an arm's fitted constants for every substation."""
    return HYPER_DIR / f"{arm}.json"


def read_params(*, arm: str, site: str) -> joint.JointParams:
    """Return an arm's fitted constants at a substation."""
    content = json.loads(hyper_path(arm=arm).read_text())
    return joint.JointParams.from_json(content=content["sites"][site]["params"])


def stop_rules(*, results: dict[str, dict[str, dict]]) -> dict:
    """Evaluate the plan's stop rules on the fits of the decision substations.

    Args:
        results: Arm name to site to the stored fit record.

    Returns:
        `stop` (true if the coordinator must see the fits before any scored run), the reasons, and
        `d_inert` (true if KD's `sigma_d` sits at its floor at 4 or more decision substations).
    """
    reasons = []
    for arm in ("KS", "KDS"):
        for site in common.DECISION_SITES:
            record = results.get(arm, {}).get(site)
            if record is not None and not any(record["active"]):
                reasons.append(f"{arm} at {site}: all four q_b are zero, the levels are frozen")
    inert = [
        site
        for site in common.DECISION_SITES
        if (record := results.get("KD", {}).get(site)) is not None
        and record["params"]["sigma_d"] <= SIGMA_D_FLOOR * 1.001
    ]
    return {
        "stop": bool(reasons),
        "reasons": reasons,
        "d_inert": len(inert) >= 4,
        "sites_with_sigma_d_at_floor": inert,
        "not_converged": [
            f"{arm} at {site}"
            for arm, by_site in results.items()
            for site, record in by_site.items()
            if record.get("converged") is False
        ],
    }


def main() -> None:
    """Fit the arms at every substation in parallel and write the files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sites", nargs="+", default=list(SITES))
    parser.add_argument("--arms", nargs="+", default=list(FIT_ARMS), choices=FIT_ARMS)
    args = parser.parse_args()
    started = time.monotonic()
    jobs = [(site, tuple(args.arms)) for site in args.sites]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        outputs = list(pool.map(fit_site, jobs))
    for arm in args.arms:
        previous = (
            json.loads(hyper_path(arm=arm).read_text()) if hyper_path(arm=arm).exists() else {}
        )
        sites = previous.get("sites", {})
        for output in outputs:
            sites[output["site"]] = output["arms"][arm]
        common.write_json_atomic(path=hyper_path(arm=arm), content={"arm": arm, "sites": sites})
    stored = {
        arm: json.loads(hyper_path(arm=arm).read_text())["sites"]
        for arm in FIT_ARMS
        if hyper_path(arm=arm).exists()
    }
    status = stop_rules(results=stored)
    common.write_json_atomic(path=FIT_STATUS_PATH, content=status)
    print(json.dumps(status, indent=2))
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
