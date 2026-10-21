"""The controls of experiment 128, run on the tuning months only, with the plan's stop rules.

- **p_premise:** per decision substation, the mean pairwise correlation across the four hour bands
  of the daily means of GAM-K's standardised one-step innovations, and the lag-1 autocorrelation
  (within months) of the mean of those four daily means. **Stop** if the mean over substations of
  the cross-band correlation is below 0.10. A lag-1 autocorrelation below 0.3 is reported and does
  not stop the experiment (the page then says that the C1 prediction was implausible).
- **p1, nesting:** (a) the joint filter with d switched off (`sigma_d = 0`) and every `q_b` equal
  to GAM-K's `q` reproduces `studies.dp_assimilation.run_kalman`'s levels, anomalies, and
  likelihood to 1e-9; (b) the forecast frames of arm GK reproduce experiment 111's stored
  `gamk_tuning_<site>.parquet` (quantiles to 1e-4; the stored level and anomaly are float32, so
  they agree to 1e-5). **Stop** on failure.
- **p2, causality:** changing every reading after the cutoff, and every ensemble run issued after
  the origin, leaves the joint filter's state, the correction, and the members unchanged. **Stop**
  on failure.
- **p_day:** every band's half-hours fall on one local date; the local-day index matches the
  Europe/London dates across the clock changes of 2024-10-27 and 2025-03-30 (days of 25 and 23
  hours); and the joint filter equals an independent dense reference filter that counts midnights
  from the local dates, on the fortnight around each clock change, to 1e-9; and the forecast
  corrections (`correction_from_state` and `day_ahead_corrections`) equal the dense reference's
  state propagated to the targets, with `sigma_d` above 0. **Stop** on failure.
- **p_ident:** on streams simulated from KD with a known `sigma_d` and `phi_d`, the refitted
  `sigma_d` lies within a factor of 2 of the truth in at least 80% of the draws; a permanent common
  level step of 3 points held for 60 days ends mostly in the levels for KD (d below 25% of the step
  after 30 days), and mostly in d for KDS when all `q_b` are zero (d above half the step). **Stop**
  on failure.
- **p_shift:** KD refitted on real tuning-year streams with each band shifted circularly by an
  independent random number of days returns `sigma_d` below 0.1 points at 4 or more of the 5
  decision substations, where the unshifted fit returns it above. A failure gives `inconclusive`
  and triggers an audit; it does not stop the experiment.
- **p_sparse:** on AR(1) streams with true `q_b = 0`, a band keeps a nonzero `q_b` at 2 nats in at
  most 10% of draws; on streams with a true `q_b` of 0.05, a band is kept in at least 80% of
  draws; and every band left out has `q` exactly 0.

Each control writes `work/control_<name>.json`, which holds `passed` and, for the controls with a
stop rule, `stop`. The script exits with status 1 after printing `STOP` when a stop rule fires.

    uv run python ml-experiments/00128-*/e00128_controls.py p_premise p1 p2 p_day
    uv run python ml-experiments/00128-*/e00128_controls.py p_ident p_shift p_sparse [--draws 30]
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import date
from typing import Final
from zoneinfo import ZoneInfo

import e00128_common as common
import e00128_filter as joint
import e00128_fit as fit
import e00128_forecast as fc
import e00128_gate as gate
import e00128_score as score_module
import e00128_streams as streams
import numpy as np
import polars as pl
from studies.dp_assimilation import CorrectionTrace, FilterHyperparameters, run_kalman
from studies.dp_calendar import BAND_FIRST_HOURS
from studies.dp_ens import choose_fit
from studies.dp_features import FeatureCache, naive_utc

PREMISE_MIN_CORRELATION: Final[float] = 0.10
PREMISE_PLAUSIBLE_CORRELATION: Final[float] = 0.25
PREMISE_MIN_LAG1: Final[float] = 0.3
P1_FILTER_TOLERANCE: Final[float] = 1e-9
P1_LIKELIHOOD_TOLERANCE: Final[float] = 1e-6
P1_QUANTILE_TOLERANCE: Final[float] = 1e-4
P1_STORED_FLOAT32_TOLERANCE: Final[float] = 1e-5
P_DAY_TOLERANCE: Final[float] = 1e-9
CHECK_RUNS: Final[int] = 40
CLOCK_CHANGE_WINDOWS: Final[tuple[tuple[date, date], ...]] = (
    (date(2024, 10, 20), date(2024, 11, 3)),
    (date(2025, 3, 23), date(2025, 4, 6)),
)
"""Fortnight windows around the clock changes: first date and last date."""
CLOCK_CHANGE_HALF_HOURS: Final[dict[date, int]] = {date(2024, 10, 27): 50, date(2025, 3, 30): 46}
"""The number of half-hours in the local day of each clock change."""
IDENT_SIGMA_D: Final[float] = 1.5
IDENT_PHI_D: Final[float] = 0.6
IDENT_FACTOR: Final[float] = 2.0
IDENT_SHARE: Final[float] = 0.8
STEP_POINTS: Final[float] = 3.0
STEP_START_DAY: Final[int] = 120
STEP_DAYS: Final[int] = 60
STEP_READ_AFTER_DAYS: Final[int] = 30
KD_D_SHARE_MAX: Final[float] = 0.25
KDS_D_SHARE_MIN: Final[float] = 0.5
SHIFT_SIGMA_D_MAX: Final[float] = 0.1
SHIFT_SITES_NEEDED: Final[int] = 4
SPARSE_FALSE_MAX: Final[float] = 0.10
SPARSE_TRUE_MIN: Final[float] = 0.80
SPARSE_TRUE_Q: Final[float] = 0.05
TEST_PARAMS: Final[joint.JointParams] = joint.JointParams(
    q=np.array([0.1, 0.0, 0.05, 0.0]),
    phi=0.85,
    sigma_z=2.5,
    sigma_eps=0.3,
    phi_d=0.7,
    sigma_d=1.5,
)
"""Constants for the plumbing controls, which do not depend on any fit."""


def site_data(*, site: str) -> tuple[FeatureCache, common.SiteSetup, list, streams.SiteStreams]:
    """Load a site's cache, setup, fits, and streams."""
    cache = common.load_cache()
    setup = common.load_setup(cache=cache, site=site)
    fits = common.load_fits(site=site)
    return cache, setup, fits, streams.build_site_streams(cache=cache, setup=setup, fits=fits)


def p_premise() -> dict:
    """Measure the cross-band correlation of daily innovations and the lag-1 autocorrelation."""
    result: dict = {}
    for site in gate.SITES:
        cache, _, _, data = site_data(site=site)
        residual = data.fit_residual(cache=cache)
        trace = joint.run_joint_filter(
            residual=residual,
            band=data.band,
            day=data.day,
            params=joint.gamk_params(hyper=common.gamk_hyper(site=site)),
        )
        counted = data.tuning_mask[trace.positions]
        table = pl.DataFrame(
            {
                "day": trace.day[counted],
                "band": data.band[trace.positions][counted],
                "z": (trace.innovation / np.sqrt(trace.innovation_variance))[counted],
            }
        )
        daily = table.group_by(["day", "band"]).agg(mean=pl.col("z").mean())
        wide = daily.pivot(on="band", index="day", values="mean").sort("day").drop_nulls()
        columns = [str(b) for b in range(joint.N_BANDS)]
        matrix = wide.select(columns).to_numpy()
        correlation = np.corrcoef(matrix.T)
        pairs = [correlation[i, j] for i in range(4) for j in range(i + 1, 4)]
        shared = matrix.mean(axis=1)
        days = wide["day"].to_numpy()
        consecutive = np.flatnonzero(np.diff(days) == 1)
        first, second = shared[consecutive], shared[consecutive + 1]
        lag1 = float(np.corrcoef(first, second)[0, 1])
        result[site] = {
            "cross_band_correlation": float(np.mean(pairs)),
            "lag1_autocorrelation": lag1,
            "days": len(days),
            "lag1_pairs": len(consecutive),
        }
        print(site, result[site], flush=True)
    mean_correlation = float(np.mean([r["cross_band_correlation"] for r in result.values()]))
    mean_lag1 = float(np.mean([r["lag1_autocorrelation"] for r in result.values()]))
    result["summary"] = {
        "mean_cross_band_correlation": mean_correlation,
        "mean_lag1_autocorrelation": mean_lag1,
        "premise_plausible": bool(
            mean_correlation >= PREMISE_PLAUSIBLE_CORRELATION and mean_lag1 >= PREMISE_MIN_LAG1
        ),
        "c1_prediction_implausible": bool(mean_lag1 < PREMISE_MIN_LAG1),
    }
    result["stop"] = bool(mean_correlation < PREMISE_MIN_CORRELATION)
    result["passed"] = not result["stop"]
    return result


def level_anomaly_difference(*, trace: joint.JointTrace, reference: CorrectionTrace) -> float:
    """Return the largest gap between the joint trace and `run_kalman`'s levels and anomalies."""
    worst = 0.0
    for b in range(joint.N_BANDS):
        index = np.searchsorted(trace.positions, reference.positions[b])
        worst = max(
            worst,
            float(np.abs(trace.state[index, b] - reference.level[b]).max()),
            float(np.abs(trace.state[index, joint.N_BANDS + b] - reference.anomaly[b]).max()),
        )
    return worst


def p1_quantiles(*, site: str) -> dict:
    """Compare arm GK's frames with experiment 111's stored frames on the first tuning runs."""
    cache, setup, fits, data = site_data(site=site)
    filters = fc.build_filters(cache=cache, data=data, wanted=("GK",))
    stored = pl.read_parquet(common.E111_WORK / "dp_preds" / f"gamk_tuning_{site}.parquet")
    spread = common.load_gamk_spread(site=site)
    quantile_difference = stored_difference = 0.0
    joined_rows = 0
    for index in setup.indices(periods={"tuning"})[:CHECK_RUNS]:
        members = fc.run_members(cache=cache, setup=setup, fits=fits, index=index)
        if members is None:
            continue
        corrected = fc.corrected_members(
            cache=cache,
            setup=setup,
            data=data,
            filtered=filters["GK"],
            members=members,
            day_mode="none",
        )
        quantiles = fc.quantiles_of(
            cache=cache, members=members, corrected=corrected.means, spread=spread
        )
        frame = fc.frame_of(
            cache=cache,
            setup=setup,
            members=members,
            corrected=corrected,
            quantiles=quantiles,
            arm="GK",
        )
        if frame is None:
            continue
        joined = frame.join(
            stored.filter(pl.col("run") == index), on=["run", "valid_time"], suffix="_111"
        )
        joined_rows += joined.height
        for column in ("q01", "q25", "q50", "q75", "q99"):
            quantile_difference = max(
                quantile_difference,
                score_module.max_of((joined[column] - joined[f"{column}_111"]).abs()),
            )
        for column in ("level", "anomaly"):
            stored_difference = max(
                stored_difference,
                score_module.max_of((joined[column] - joined[f"{column}_111"]).abs()),
            )
    return {
        "quantiles_vs_stored": quantile_difference,
        "level_anomaly_vs_stored_float32": stored_difference,
        "rows_compared": joined_rows,
        "passed": bool(
            joined_rows > 0
            and quantile_difference < P1_QUANTILE_TOLERANCE
            and stored_difference < P1_STORED_FLOAT32_TOLERANCE
        ),
    }


def p1() -> dict:
    """Check nesting against `run_kalman` and against experiment 111's stored frames."""
    result: dict = {}
    for site in common.SCORED_SITES:
        _, _, _, data = site_data(site=site)
        hyper = common.gamk_hyper(site=site)
        reference = run_kalman(
            residual=data.residual,
            band=data.band,
            hyper=FilterHyperparameters(*hyper),
        )
        gamk = joint.gamk_params(hyper=hyper)
        kd_off = replace(
            joint.params_from_vector(
                values=np.array([hyper[0], hyper[1], hyper[2], hyper[3], 0.0, 0.0])
            ),
            sigma_d=0.0,
        )
        ks_equal = gamk.with_q(q=np.full(joint.N_BANDS, float(gamk.q[0])))
        differences = {}
        for name, params in (("GK", gamk), ("KD_sigma_d_0", kd_off), ("KS_equal_q", ks_equal)):
            trace = joint.run_joint_filter(
                residual=data.residual, band=data.band, day=data.day, params=params
            )
            differences[name] = {
                "level_anomaly": level_anomaly_difference(trace=trace, reference=reference),
                "log_likelihood": abs(trace.log_likelihood - reference.log_likelihood),
            }
        quantiles = p1_quantiles(site=site)
        result[site] = {
            "filter": differences,
            **quantiles,
            "passed": bool(
                quantiles["passed"]
                and all(
                    d["level_anomaly"] < P1_FILTER_TOLERANCE
                    and d["log_likelihood"] < P1_LIKELIHOOD_TOLERANCE
                    for d in differences.values()
                )
            ),
        }
        print(site, result[site], flush=True)
    result["passed"] = all(r["passed"] for r in result.values())
    result["stop"] = not result["passed"]
    return result


def p2() -> dict:
    """Check that a forecast ignores later readings and runs."""
    site = "D10"
    cache, setup, fits, data = site_data(site=site)
    index = setup.indices(periods={"tuning"})[60]
    origin = setup.origins[index]
    cutoff = fc.cutoff_of(cache=cache, origin=origin)
    later_readings = data.residual.copy()
    later_readings[cutoff + 1 :] += 25.0
    outcome = {}
    for name, params in (
        ("GK", joint.gamk_params(hyper=common.gamk_hyper(site=site))),
        ("KDS_test_constants", TEST_PARAMS),
    ):
        states = []
        for residual in (data.residual, later_readings):
            trace = joint.run_joint_filter(
                residual=residual, band=data.band, day=data.day, params=params
            )
            state = joint.state_at(trace=trace, cutoff=cutoff)
            target = np.arange(cutoff, cutoff + 96)
            correction = joint.correction_from_state(
                state=state,
                params=params,
                band=data.band[target],
                target=target,
                target_day=data.day[target],
            )
            states.append((state.x, correction.total))
        outcome[name] = bool(
            np.array_equal(states[0][0], states[1][0])
            and np.array_equal(states[0][1], states[1][1])
        )
    members = fc.run_members(cache=cache, setup=setup, fits=fits, index=index)
    later = replace(
        setup,
        runs=[
            run
            if position <= index
            else replace(
                run, temperature_c=run.temperature_c + 9.0, shortwave_w_m2=run.shortwave_w_m2 * 0.1
            )
            for position, run in enumerate(setup.runs)
        ],
    )
    members_later = fc.run_members(cache=cache, setup=later, fits=fits, index=index)
    fit_used = choose_fit(fits=fits, origin=origin)
    outcome["members_unchanged_by_later_runs"] = bool(
        members is not None
        and members_later is not None
        and np.array_equal(members.base_mean, members_later.base_mean)
    )
    beyond = data.fit_residual(cache=cache)[cache.end_times > naive_utc(moment=streams.FIT_END)]
    outcome["fit_stream_empty_after_cut"] = bool(not np.isfinite(beyond).any())
    outcome["fit_cut_not_after_origin"] = bool(fit_used.cut <= origin)
    outcome["passed"] = all(outcome.values())
    outcome["stop"] = not outcome["passed"]
    print(outcome)
    return outcome


def reference_filter(
    *,
    values: np.ndarray,
    positions: np.ndarray,
    bands: np.ndarray,
    dates: np.ndarray,
    params: joint.JointParams,
) -> np.ndarray:
    """Run an independent dense filter that counts midnights from the local dates.

    The reference builds the transition matrix and the noise matrix explicitly at every reading and
    builds the update from dense products, so it shares no propagation or update code with
    `e00128_filter`. A band's level starts to move at the band's first reading.

    Args:
        values: The readings.
        positions: The grid row of each reading.
        bands: The hour band of each reading.
        dates: The local date of each reading, `datetime64[D]`.
        params: The filter's constants.

    Returns:
        The state after each reading, shape (readings, 9).
    """
    n = joint.N_STATE
    x = np.zeros(n)
    covariance = np.diag(
        [*([joint.INITIAL_LEVEL_VARIANCE] * 4), *([params.stationary_z] * 4), params.stationary_d]
    )
    started = np.zeros(4, dtype=bool)
    out = np.empty((len(values), n))
    for k in range(len(values)):
        if k:
            gap = int(positions[k] - positions[k - 1])
            midnights = int((dates[k] - dates[k - 1]).astype(int))
            transition = np.eye(n)
            noise = np.zeros((n, n))
            for b in range(4):
                transition[4 + b, 4 + b] = params.phi**gap
                noise[4 + b, 4 + b] = params.stationary_z * (1.0 - params.phi ** (2 * gap))
                if started[b]:
                    noise[b, b] = params.q[b] / 48.0 * gap
            transition[8, 8] = params.phi_d**midnights
            noise[8, 8] = params.stationary_d * (1.0 - params.phi_d ** (2 * midnights))
            x = transition @ x
            covariance = transition @ covariance @ transition.T + noise
        h = np.zeros(n)
        h[bands[k]] = h[4 + bands[k]] = h[8] = 1.0
        variance = h @ covariance @ h + params.sigma_eps**2
        gain = covariance @ h / variance
        x = x + gain * (values[k] - h @ x)
        covariance = covariance - np.outer(gain, h @ covariance)
        started[bands[k]] = True
        out[k] = x
    return out


def reference_forecast_error(
    *,
    trace: joint.JointTrace,
    reference_state: np.ndarray,
    window: np.ndarray,
    data: streams.SiteStreams,
    dates: np.ndarray,
    params: joint.JointParams,
) -> float:
    """Return the largest gap between the code's corrections and the dense reference's.

    For five cutoffs spread over the window, the reference propagates its own dense state to each of
    the next 100 grid rows with an explicit transition matrix, counting midnights from the local
    dates, and sums the level, anomaly, and day-level entries of the band. Both
    `correction_from_state` and `day_ahead_corrections` must give the same totals.

    Args:
        trace: The filter's trace on the window.
        reference_state: The dense reference's state after each reading of the window.
        window: The grid row of each reading.
        data: The site's streams.
        dates: The local date of every grid row, `datetime64[D]`.
        params: The filter's constants.

    Returns:
        The largest absolute gap, in points.
    """
    worst = 0.0
    for k in np.linspace(len(window) // 10, len(window) - 1, num=5).astype(int):
        targets = np.arange(window[k] + 1, min(window[k] + 101, len(dates)))
        band = data.band[targets]
        expected = np.empty(len(targets))
        for i, row in enumerate(targets):
            gap = int(row - window[k])
            midnights = int((dates[row] - dates[window[k]]).astype(int))
            propagated = reference_state[k].copy()
            propagated[4:8] *= params.phi**gap
            propagated[8] *= params.phi_d**midnights
            expected[i] = propagated[band[i]] + propagated[4 + band[i]] + propagated[8]
        from_state = joint.correction_from_state(
            state=joint.state_at(trace=trace, cutoff=int(window[k])),
            params=params,
            band=band,
            target=targets,
            target_day=data.day[targets],
        )
        day_ahead = joint.day_ahead_corrections(
            trace=trace,
            rows=targets,
            cutoffs=np.full(len(targets), window[k]),
            band=data.band,
            day=data.day,
        )
        worst = max(
            worst,
            float(np.abs(from_state.total - expected).max()),
            float(np.abs(day_ahead.total - expected).max()),
        )
    return worst


def p_day() -> dict:
    """Check the bands, the local days, and the midnight counts across the clock changes."""
    cache = common.load_cache()
    calendar = cache.calendar
    expected_band = (
        np.searchsorted(np.array(BAND_FIRST_HOURS), calendar.half_hour // 2, side="right") - 1
    )
    day = common.local_day_index(cache=cache)
    wrap = np.r_[False, calendar.half_hour[1:] == 0]
    increments = np.r_[0, np.diff(day)]
    outcome: dict = {
        "bands_match_local_hours": bool(np.array_equal(expected_band, calendar.band)),
        "day_rises_exactly_at_local_midnight": bool(np.array_equal(increments == 1, wrap)),
        "day_never_rises_by_more_than_one": bool(increments.max() <= 1),
    }
    zone = ZoneInfo("Europe/London")
    starts = (cache.end_times - np.timedelta64(30, "m")).astype("datetime64[us]")
    lengths = {}
    mismatches = 0
    for first, last in CLOCK_CHANGE_WINDOWS:
        inside = np.flatnonzero(
            (calendar.local_date >= np.datetime64(first))
            & (calendar.local_date <= np.datetime64(last))
        )
        for row in inside[:: max(len(inside) // 400, 1)]:
            moment = starts[row].astype(object).replace(tzinfo=ZoneInfo("UTC")).astimezone(zone)
            mismatches += int(moment.date() != calendar.local_date[row].astype(object))
        for change, count in CLOCK_CHANGE_HALF_HOURS.items():
            if first <= change <= last:
                lengths[str(change)] = int((calendar.local_date == np.datetime64(change)).sum())
                outcome[f"{change}_has_{count}_half_hours"] = lengths[str(change)] == count
    outcome["local_dates_match_zoneinfo"] = mismatches == 0
    site = "D10"
    data = streams.load_site_streams(site=site)
    residual = data.fit_residual(cache=cache)
    worst = 0.0
    worst_forecast = 0.0
    for first, last in CLOCK_CHANGE_WINDOWS:
        window = np.flatnonzero(
            (calendar.local_date >= np.datetime64(first))
            & (calendar.local_date <= np.datetime64(last))
            & np.isfinite(residual)
        )
        if len(window) == 0:
            continue
        trace = joint.run_joint_filter(
            residual=np.where(np.isin(np.arange(len(residual)), window), residual, np.nan),
            band=data.band,
            day=data.day,
            params=TEST_PARAMS,
        )
        expected = reference_filter(
            values=residual[window],
            positions=window,
            bands=data.band[window],
            dates=calendar.local_date[window],
            params=TEST_PARAMS,
        )
        worst = max(worst, float(np.abs(trace.state - expected).max()))
        worst_forecast = max(
            worst_forecast,
            reference_forecast_error(
                trace=trace,
                reference_state=expected,
                window=window,
                data=data,
                dates=calendar.local_date,
                params=TEST_PARAMS,
            ),
        )
    outcome["joint_filter_vs_dense_reference"] = worst
    outcome["reference_agrees"] = bool(worst < P_DAY_TOLERANCE)
    outcome["forecast_vs_dense_reference"] = worst_forecast
    outcome["forecast_agrees"] = bool(worst_forecast < P_DAY_TOLERANCE)
    outcome["passed"] = all(value for key, value in outcome.items() if isinstance(value, bool))
    outcome["stop"] = not outcome["passed"]
    print(outcome)
    return outcome


def known_kd_params(*, inputs: gate.GateInputs) -> joint.JointParams:
    """Return the constants P-ident simulates from: GAM-K's with a known `sigma_d` and `phi_d`."""
    return replace(inputs.gk, phi_d=IDENT_PHI_D, sigma_d=IDENT_SIGMA_D)


def ident_draw(job: tuple[str, int]) -> dict:
    """Simulate a stream from KD, refit KD, and read the permanent-step test."""
    site, draw = job
    inputs = gate.gate_inputs(site)
    truth = known_kd_params(inputs=inputs)
    generator = np.random.default_rng(gate.draw_seed(parts=("p_ident", site, str(draw))))
    stream = joint.simulate_stream(
        params=truth,
        band=inputs.band,
        day=inputs.day,
        present=np.isfinite(inputs.real_residual),
        generator=generator,
    )
    prior_mean, prior_spread = fit.gamk_prior()
    refit = fit.fit_kd(
        inputs=fit.StreamInputs(
            residual=stream,
            band=inputs.band,
            day=inputs.day,
            likelihood_mask=inputs.likelihood_mask & np.isfinite(stream),
        ),
        start=fit.start_vector(site=site),
        prior_mean=prior_mean,
        prior_spread=prior_spread,
    )
    stepped = stream + np.where(
        (inputs.day >= STEP_START_DAY) & (inputs.day < STEP_START_DAY + STEP_DAYS), STEP_POINTS, 0.0
    )
    read_day = STEP_START_DAY + STEP_READ_AFTER_DAYS
    shares = {}
    for name, params in (("KD", truth), ("KDS_q_zero", truth.with_q(q=np.zeros(joint.N_BANDS)))):
        trace = joint.run_joint_filter(
            residual=stepped, band=inputs.band, day=inputs.day, params=params
        )
        row = int(np.searchsorted(trace.day, read_day, side="right")) - 1
        shares[name] = {
            "d_share": float(trace.state[row, joint.D_INDEX] / STEP_POINTS),
            "level_share": float(trace.state[row, : joint.N_BANDS].mean() / STEP_POINTS),
        }
    return {
        "site": site,
        "draw": draw,
        "sigma_d_fit": refit.params.sigma_d,
        "within_factor": bool(
            truth.sigma_d / IDENT_FACTOR <= refit.params.sigma_d <= truth.sigma_d * IDENT_FACTOR
        ),
        "steps": shares,
    }


def p_ident(*, draws: int, workers: int) -> dict:
    """Check that KD recovers a known `sigma_d` and that a permanent step goes where expected."""
    jobs = [(site, draw) for site in gate.SITES for draw in range(draws)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        outputs = list(pool.map(ident_draw, jobs))
    share_within = float(np.mean([o["within_factor"] for o in outputs]))
    kd_d_share = float(np.mean([o["steps"]["KD"]["d_share"] for o in outputs]))
    kds_d_share = float(np.mean([o["steps"]["KDS_q_zero"]["d_share"] for o in outputs]))
    result = {
        "true_sigma_d": IDENT_SIGMA_D,
        "true_phi_d": IDENT_PHI_D,
        "draws": len(outputs),
        "share_within_factor_2": share_within,
        "kd_step_d_share": kd_d_share,
        "kd_step_level_share": float(np.mean([o["steps"]["KD"]["level_share"] for o in outputs])),
        "kds_q_zero_step_d_share": kds_d_share,
        "median_sigma_d_fit": float(np.median([o["sigma_d_fit"] for o in outputs])),
        "outputs": outputs,
    }
    result["passed"] = bool(
        share_within >= IDENT_SHARE
        and kd_d_share < KD_D_SHARE_MAX
        and kds_d_share > KDS_D_SHARE_MIN
    )
    result["stop"] = not result["passed"]
    return result


def shift_bands(*, inputs: gate.GateInputs, generator: np.random.Generator) -> np.ndarray:
    """Return the real residual with each band shifted circularly by its own random number of days.

    A band's readings of local day `i` are replaced by the same band's readings of day `i + s`
    (wrapping round the stream), cut or padded with `NaN` to the length of day `i`'s own readings.
    """
    out = np.full(len(inputs.real_residual), np.nan)
    for b in range(joint.N_BANDS):
        rows = np.flatnonzero(inputs.band == b)
        days = inputs.day[rows]
        unique = np.unique(days)
        by_day = {int(d): rows[days == d] for d in unique}
        shift = int(generator.integers(1, len(unique)))
        for position, d in enumerate(unique):
            target = by_day[int(d)]
            source = by_day[int(unique[(position + shift) % len(unique)])]
            values = inputs.real_residual[source][: len(target)]
            out[target[: len(values)]] = values
    return out


def shift_draw(job: tuple[str, int]) -> dict:
    """Refit KD on one band-shifted stream (shift 0 is the unshifted stream)."""
    site, draw = job
    inputs = gate.gate_inputs(site)
    generator = np.random.default_rng(gate.draw_seed(parts=("p_shift", site, str(draw))))
    stream = inputs.real_residual if draw == 0 else shift_bands(inputs=inputs, generator=generator)
    prior_mean, prior_spread = fit.gamk_prior()
    refit = fit.fit_kd(
        inputs=fit.StreamInputs(
            residual=stream,
            band=inputs.band,
            day=inputs.day,
            likelihood_mask=inputs.likelihood_mask & np.isfinite(stream),
        ),
        start=joint.vector_from_params(params=inputs.real_kd),
        prior_mean=prior_mean,
        prior_spread=prior_spread,
    )
    return {"site": site, "draw": draw, "sigma_d": refit.params.sigma_d}


def p_shift(*, draws: int, workers: int) -> dict:
    """Check that KD finds no day-level state once the cross-band structure is destroyed."""
    jobs = [(site, draw) for site in gate.SITES for draw in range(draws + 1)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        outputs = list(pool.map(shift_draw, jobs))
    per_site = {}
    for site in gate.SITES:
        shifted = [o["sigma_d"] for o in outputs if o["site"] == site and o["draw"] > 0]
        unshifted = next(o["sigma_d"] for o in outputs if o["site"] == site and o["draw"] == 0)
        per_site[site] = {
            "unshifted_sigma_d": unshifted,
            "median_shifted_sigma_d": float(np.median(shifted)),
            "share_shifted_below_0.1": float(np.mean(np.array(shifted) < SHIFT_SIGMA_D_MAX)),
            "ok": bool(np.median(shifted) < SHIFT_SIGMA_D_MAX and unshifted > SHIFT_SIGMA_D_MAX),
        }
    result = {"sites": per_site, "shifts_per_site": draws}
    result["passed"] = sum(s["ok"] for s in per_site.values()) >= SHIFT_SITES_NEEDED
    result["stop"] = False
    return result


def sparse_draw(job: tuple[str, int, float]) -> dict:
    """Search the level variances of one AR(1) stream with a known `q_b` in every band."""
    site, draw, true_q = job
    inputs = gate.gate_inputs(site)
    truth = inputs.gk.with_q(q=np.full(joint.N_BANDS, true_q))
    generator = np.random.default_rng(
        gate.draw_seed(parts=("p_sparse", site, str(true_q), str(draw)))
    )
    stream = joint.simulate_stream(
        params=truth,
        band=inputs.band,
        day=inputs.day,
        present=np.isfinite(inputs.real_residual),
        generator=generator,
    )
    result = fit.sparse_search(
        inputs=fit.StreamInputs(
            residual=stream,
            band=inputs.band,
            day=inputs.day,
            likelihood_mask=inputs.likelihood_mask & np.isfinite(stream),
        ),
        base=truth,
        threshold=fit.THRESHOLD_NATS,
    )
    return {
        "site": site,
        "draw": draw,
        "true_q": true_q,
        "active": result.active.tolist(),
        "q_of_left_out_bands": result.q[~result.active].tolist(),
    }


def p_sparse(*, draws: int, workers: int) -> dict:
    """Check the sparse search keeps a band with a true level variance and drops one without."""
    jobs = [
        (site, draw, true_q)
        for true_q in (0.0, SPARSE_TRUE_Q)
        for site in gate.SITES
        for draw in range(draws)
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        outputs = list(pool.map(sparse_draw, jobs))
    result: dict = {"draws_per_site": draws}
    left_out = [q for output in outputs for q in output["q_of_left_out_bands"]]
    result["left_out_bands_have_q_exactly_zero"] = all(q == 0.0 for q in left_out)
    for true_q in (0.0, SPARSE_TRUE_Q):
        active = np.array([o["active"] for o in outputs if o["true_q"] == true_q])
        result[f"true_q_{true_q}"] = {
            "share_kept_by_band": active.mean(axis=0).tolist(),
            "share_any_band_kept": float(active.any(axis=1).mean()),
        }
    false_rate = max(result["true_q_0.0"]["share_kept_by_band"])
    true_rate = min(result[f"true_q_{SPARSE_TRUE_Q}"]["share_kept_by_band"])
    result["passed"] = bool(
        false_rate <= SPARSE_FALSE_MAX
        and true_rate >= SPARSE_TRUE_MIN
        and result["left_out_bands_have_q_exactly_zero"]
    )
    result["stop"] = False
    return result


def main() -> None:
    """Run the chosen controls, write their results, and exit with status 1 on a stop rule."""
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("p_premise", "p1", "p2", "p_day", "p_ident", "p_shift", "p_sparse")
    parser.add_argument("controls", nargs="+", choices=names)
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    defaults = {"p_ident": 30, "p_shift": 20, "p_sparse": 30}
    stopped = []
    for name in args.controls:
        draws = args.draws or defaults.get(name, 0)
        if name == "p_ident":
            content = p_ident(draws=draws, workers=args.workers)
        elif name == "p_shift":
            content = p_shift(draws=draws, workers=args.workers)
        elif name == "p_sparse":
            content = p_sparse(draws=draws, workers=args.workers)
        else:
            content = {"p_premise": p_premise, "p1": p1, "p2": p2, "p_day": p_day}[name]()
        common.write_json_atomic(path=common.WORK_DIR / f"control_{name}.json", content=content)
        print(f"{name}: written; passed={content['passed']}", flush=True)
        if content.get("stop"):
            stopped.append(name)
    if stopped:
        print(f"STOP: {', '.join(stopped)} fired a stop rule; report before the gate.")
        print(json.dumps({"stopped": stopped}))
        sys.exit(1)


if __name__ == "__main__":
    main()
