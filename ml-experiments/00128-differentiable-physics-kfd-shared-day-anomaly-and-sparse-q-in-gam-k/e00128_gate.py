"""The known-answer gate of experiment 128: can the filter find a level that really moved?

Each stream is built on the real PV regressor and the real tuning-year residuals of one decision
substation (2024-07-01 to 2025-06-30). The script injects a known change shaped like a rise in
solar-panel capacity, adds noise, refits KD, KS, and KDS on the synthetic stream (starting from the
real fit and with the real fit's prior), runs GK, KD, KS, and KDS over it, and stores the arms'
end-of-day midday level, night level, and d, and their day-ahead absolute error.

**Scenarios.** S0 injects nothing. S1 raises the PV weight from 1 to 1.30 over 180 days from
2024-07-01, so the stream gains `0.30 * clip(days / 180) * pv_regressor` at each half-hour. S1x3 is
three times that. **Noise forms.** `real`: a circular month-block resample of the substation's real
tuning-year residuals (12 draws per substation), which keeps the cross-band structure. `ar1`: the
filter's own model without a day-level state (an AR(1) anomaly per hour band at GAM-K's `phi`,
`sigma_z`, and `sigma_eps`; 6 draws per substation, run as S0, S1, and S1x3 so that the reference
rate exists at whichever amplitude decides). **Each S1 and S1x3 stream is paired with an S0
stream on the identical noise realisation**: the seed depends on the substation, the form, and the
draw, and not on the scenario.

**Truth.** The truth path is `I_k = deviation_k * R_k`, where `R_k` is the mean of the PV regressor
over the midday-band half-hours of the 30 days up to local day `k`. The read day `k*` is the day
from 30 to 200 on which `|I_k|` peaks, computed per substation from the regressors before any draw
runs (`truth_path`).

The `power` step writes `work/gate_power.json`: the peak of `|I_k|` in posterior standard deviations
of the midday level (primary) and of the midday level plus d (secondary) on day `k*`, from KD's
filter on the real tuning stream.

    uv run python ml-experiments/00128-*/e00128_gate.py power
    uv run python ml-experiments/00128-*/e00128_gate.py run [--workers 6] [--scenarios S0 S1 S1x3]
    uv run python ml-experiments/00128-*/e00128_gate.py report
"""

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Final

import e00128_common as common
import e00128_filter as joint
import e00128_fit as fit
import e00128_streams as streams
import numpy as np
from studies.dp_assimilation import DEFAULT_DELAY_HOURS, day_ahead_initialisation

GATE_DIR: Final[Path] = common.WORK_DIR / "gate"
GATE_JSON: Final[Path] = common.WORK_DIR / "gate.json"
POWER_JSON: Final[Path] = common.WORK_DIR / "gate_power.json"
STREAM_START: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
STREAM_END: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
PV_RAMP_DAYS: Final[float] = 180.0
PV_CHANGE: Final[float] = 0.30
SETTLE_DAYS: Final[int] = 60
"""Days at the start of a stream that the no-cross-talk and no-false-movement readings skip."""
READ_WINDOW: Final[tuple[int, int]] = (30, 200)
"""The days between which the truth path peaks."""
TRUTH_WINDOW_DAYS: Final[int] = 30
READ_AVERAGE_DAYS: Final[int] = 7
MIDDAY_BAND: Final[int] = 2
NIGHT_BAND: Final[int] = 0
SCENARIOS: Final[dict[str, float]] = {"S0": 0.0, "S1": 1.0, "S1x3": 3.0}
"""The share of the real change injected by each scenario."""
FORMS: Final[tuple[str, ...]] = ("real", "ar1")
DRAWS: Final[dict[str, int]] = {"real": 12, "ar1": 6}
"""Noise draws per substation and form. Each is run as every scenario of the form."""
SCENARIOS_OF_FORM: Final[dict[str, tuple[str, ...]]] = {
    "real": ("S0", "S1", "S1x3"),
    "ar1": ("S0", "S1", "S1x3"),
}
GATE_ARMS: Final[tuple[str, ...]] = ("GK", "KD", "KS", "KDS")
SITES: Final[tuple[str, ...]] = common.DECISION_SITES
SEED_NAMESPACE: Final[str] = "e128"


@dataclass(frozen=True)
class GateInputs:
    """A site's real stream and constants over the tuning year, on the rows of the stream.

    Attributes:
        site: The substation's label.
        rows: The grid rows of the stream (2024-07-01 to 2025-06-30).
        pv: The PV regressor on the stream rows, `NaN` where absent.
        real_residual: The site's real residuals on the stream rows, `NaN` where absent.
        band: The hour band of each stream row.
        day: The local day of each stream row, counted from the stream's first row.
        days_float: Days since 2024-07-01 00:00 UTC of each row's end, as a float.
        month: The calendar month (`datetime64[M]`) of each stream row.
        likelihood_mask: True on the stream rows of the tuning months.
        n_days: The number of local days the stream spans.
        day_ahead_cutoff: The newest stream index each row's day-ahead forecast may see, -1 where
            there is none.
        gk: GAM-K's constants at the site.
        real_kd: The real fit's constants of KD at the site: the refit's start.
    """

    site: str
    rows: np.ndarray
    pv: np.ndarray
    real_residual: np.ndarray
    band: np.ndarray
    day: np.ndarray
    days_float: np.ndarray
    month: np.ndarray
    likelihood_mask: np.ndarray
    n_days: int
    day_ahead_cutoff: np.ndarray
    gk: joint.JointParams
    real_kd: joint.JointParams


@lru_cache(maxsize=8)
def gate_inputs(site: str) -> GateInputs:
    """Build a site's gate inputs from experiment 111's data and the real fits."""
    cache = common.load_cache()
    data = streams.load_site_streams(site=site)
    ends = cache.end_times
    rows = np.flatnonzero(
        (ends > np.datetime64(STREAM_START.replace(tzinfo=None), "us"))
        & (ends <= np.datetime64(STREAM_END.replace(tzinfo=None), "us"))
    )
    start64 = np.datetime64(STREAM_START.replace(tzinfo=None), "us")
    initialisation = day_ahead_initialisation(end_times=ends[rows], days_ahead=1)
    delay = np.timedelta64(round(DEFAULT_DELAY_HOURS * 3600 * 1_000_000), "us")
    cutoff_grid = np.searchsorted(ends, initialisation - delay, side="right") - 1
    day = data.day[rows] - data.day[rows[0]]
    return GateInputs(
        site=site,
        rows=rows,
        pv=data.pv_regressor[rows],
        real_residual=data.residual[rows],
        band=data.band[rows],
        day=day,
        days_float=((ends[rows] - start64) / np.timedelta64(1, "D")).astype(np.float64),
        month=(ends[rows] - np.timedelta64(30, "m")).astype("datetime64[M]"),
        likelihood_mask=common.in_months(cache=cache, months=common.TUNING_MONTHS)[rows],
        n_days=int(day.max()) + 1,
        day_ahead_cutoff=np.where(cutoff_grid >= rows[0], cutoff_grid - rows[0], -1),
        gk=joint.gamk_params(hyper=common.gamk_hyper(site=site)),
        real_kd=fit.read_params(arm="KD", site=site),
    )


def draw_seed(*, parts: tuple[str, ...]) -> int:
    """Return a seed that depends only on the draw's name, so a parallel run repeats exactly."""
    return int(hashlib.sha256("|".join((SEED_NAMESPACE, *parts)).encode()).hexdigest()[:12], 16)


def injected_deviation(*, inputs: GateInputs, amplitude: float) -> np.ndarray:
    """Return the PV weight's deviation from 1 at each stream row."""
    return amplitude * PV_CHANGE * np.clip(inputs.days_float / PV_RAMP_DAYS, 0.0, 1.0)


def ar1_noise(*, inputs: GateInputs, generator: np.random.Generator) -> np.ndarray:
    """Simulate the filter's own noise model without a level or d: an AR(1) anomaly per band."""
    params = joint.JointParams(
        q=np.zeros(joint.N_BANDS),
        phi=inputs.gk.phi,
        sigma_z=inputs.gk.sigma_z,
        sigma_eps=inputs.gk.sigma_eps,
        phi_d=0.5,
        sigma_d=0.0,
    )
    return joint.simulate_stream(
        params=params,
        band=inputs.band,
        day=inputs.day,
        present=np.isfinite(inputs.real_residual),
        generator=generator,
    )


def block_noise(*, inputs: GateInputs, generator: np.random.Generator) -> np.ndarray:
    """Resample the real residuals by whole calendar months, with replacement, keeping the position.

    The residual at a position in a month comes from the same position (day and half-hour) of
    another month of the stream, wrapped round if that month is shorter. A missing real residual
    stays missing.
    """
    labels = np.unique(inputs.month)
    members = {label: np.flatnonzero(inputs.month == label) for label in labels}
    out = np.full(len(inputs.rows), np.nan)
    for label in labels:
        target = members[label]
        source = members[labels[generator.integers(len(labels))]]
        out[target] = inputs.real_residual[source[np.arange(len(target)) % len(source)]]
    return out


def noise_of(*, inputs: GateInputs, form: str, draw: int) -> np.ndarray:
    """Return the noise realisation of a draw, which every scenario of the draw shares."""
    generator = np.random.default_rng(draw_seed(parts=(inputs.site, form, str(draw))))
    return (
        ar1_noise(inputs=inputs, generator=generator)
        if form == "ar1"
        else block_noise(inputs=inputs, generator=generator)
    )


def synthetic_stream(*, inputs: GateInputs, noise: np.ndarray, scenario: str) -> np.ndarray:
    """Return the residual stream of a scenario: the injected change plus the noise."""
    deviation = injected_deviation(inputs=inputs, amplitude=SCENARIOS[scenario])
    stream = deviation * inputs.pv + noise
    return np.where(np.isfinite(inputs.pv), stream, np.nan)


def truth_path(*, inputs: GateInputs, amplitude: float = 1.0) -> np.ndarray:
    """Return `I_k` at the end of each local day, shape (days,).

    `I_k` is the injected deviation at the end of day `k` times the mean PV regressor over the
    midday-band half-hours of the 30 days up to day `k`.
    """
    midday = (inputs.band == MIDDAY_BAND) & np.isfinite(inputs.pv)
    out = np.full(inputs.n_days, np.nan)
    for k in range(inputs.n_days):
        window = midday & (inputs.day > k - TRUTH_WINDOW_DAYS) & (inputs.day <= k)
        if window.any():
            deviation = amplitude * PV_CHANGE * min((k + 1) / PV_RAMP_DAYS, 1.0)
            out[k] = deviation * inputs.pv[window].mean()
    return out


def read_day(*, truth: np.ndarray) -> int:
    """Return the day from 30 to 200 on which the truth path's size peaks."""
    low, high = READ_WINDOW
    return low + int(np.nanargmax(np.abs(truth[low : high + 1])))


def refit_arms(*, inputs: GateInputs, stream: np.ndarray) -> dict[str, joint.JointParams]:
    """Refit KD, KS, and KDS on a synthetic stream, as the real pipeline fits them.

    KD starts from the real fit and uses the real fit's prior. KS holds GAM-K's other values, and
    KDS holds the refitted KD's, each searching its level variances from zero at 2 nats.
    """
    stream_inputs = fit.StreamInputs(
        residual=stream,
        band=inputs.band,
        day=inputs.day,
        likelihood_mask=inputs.likelihood_mask & np.isfinite(stream),
    )
    prior_mean, prior_spread = fit.gamk_prior()
    kd = fit.fit_kd(
        inputs=stream_inputs,
        start=joint.vector_from_params(params=inputs.real_kd),
        prior_mean=prior_mean,
        prior_spread=prior_spread,
    )
    ks = fit.sparse_search(inputs=stream_inputs, base=inputs.gk, threshold=fit.THRESHOLD_NATS)
    kds = fit.sparse_search(inputs=stream_inputs, base=kd.params, threshold=fit.THRESHOLD_NATS)
    return {
        "GK": inputs.gk,
        "KD": kd.params,
        "KS": inputs.gk.with_q(q=ks.q),
        "KDS": kd.params.with_q(q=kds.q),
    }


def day_ahead_mae(*, trace: joint.JointTrace, stream: np.ndarray, inputs: GateInputs) -> float:
    """Return a filter's mean absolute day-ahead error on a stream, after the settling days."""
    rows = np.flatnonzero(
        np.isfinite(stream) & (inputs.day > SETTLE_DAYS) & (inputs.day_ahead_cutoff >= 0)
    )
    correction = joint.day_ahead_corrections(
        trace=trace,
        rows=rows,
        cutoffs=inputs.day_ahead_cutoff[rows],
        band=inputs.band,
        day=inputs.day,
    )
    return float(np.mean(np.abs(stream[rows] - correction.total)))


def arm_record(*, params: joint.JointParams, stream: np.ndarray, inputs: GateInputs) -> dict:
    """Run one arm's filter on a stream and return the series the readings use."""
    trace = joint.run_joint_filter(residual=stream, band=inputs.band, day=inputs.day, params=params)
    n_days = inputs.n_days

    def daily(values: np.ndarray) -> list[float]:
        return joint.end_of_day_series(trace=trace, values=values, n_days=n_days).tolist()

    state = trace.state
    return {
        "mid": daily(state[:, MIDDAY_BAND]),
        "night": daily(state[:, NIGHT_BAND]),
        "d": daily(state[:, joint.D_INDEX]),
        "mae": day_ahead_mae(trace=trace, stream=stream, inputs=inputs),
        "q": params.q.tolist(),
        "sigma_d": params.sigma_d,
        "phi_d": params.phi_d,
    }


def stream_path(*, site: str, form: str, scenario: str, draw: int) -> Path:
    """Return the file that holds one stream's record."""
    return GATE_DIR / form / scenario / f"{site}_{draw:03d}.json"


def run_stream(job: tuple[str, str, str, int]) -> str:
    """Run one synthetic stream and store its record, unless a file for it exists.

    Args:
        job: Site, noise form, scenario, and draw index.

    Returns:
        The path of the stream's file.
    """
    site, form, scenario, draw = job
    path = stream_path(site=site, form=form, scenario=scenario, draw=draw)
    if path.exists():
        return str(path)
    inputs = gate_inputs(site)
    stream = synthetic_stream(
        inputs=inputs, noise=noise_of(inputs=inputs, form=form, draw=draw), scenario=scenario
    )
    fitted = refit_arms(inputs=inputs, stream=stream)
    content = {
        "site": site,
        "form": form,
        "scenario": scenario,
        "draw": draw,
        "arms": {
            arm: arm_record(params=params, stream=stream, inputs=inputs)
            for arm, params in fitted.items()
        },
    }
    common.write_json_atomic(path=path, content=content)
    return str(path)


def power_check() -> dict:
    """Return, per site, the peak of the truth path in posterior standard deviations.

    The posterior standard deviation of the midday level (primary) and of the midday level plus d
    (secondary) on the read day comes from KD's real fit run on the site's real tuning-year stream.
    """
    out = {}
    for site in SITES:
        inputs = gate_inputs(site)
        truth = truth_path(inputs=inputs)
        k_star = read_day(truth=truth)
        trace = joint.run_joint_filter(
            residual=inputs.real_residual,
            band=inputs.band,
            day=inputs.day,
            params=inputs.real_kd,
        )
        row = int(np.searchsorted(trace.day, k_star, side="right")) - 1
        sd_level = trace.sd[row, MIDDAY_BAND]
        variance_sum = (
            sd_level**2
            + trace.sd[row, joint.D_INDEX] ** 2
            + 2.0 * trace.cov_level_d[row, MIDDAY_BAND]
        )
        peak = float(np.abs(truth[k_star]))
        out[site] = {
            "read_day": k_star,
            "peak_s1": peak,
            "sd_primary": float(sd_level),
            "sd_secondary": float(np.sqrt(variance_sum)),
            "primary_in_sds": peak / float(sd_level),
            "secondary_in_sds": peak / float(np.sqrt(variance_sum)),
        }
    return out


def jobs_for(*, scenarios: list[str], draws: dict[str, int]) -> list[tuple[str, str, str, int]]:
    """Return every stream of the scenarios, over the noise forms and substations."""
    return [
        (site, form, scenario, draw)
        for form in FORMS
        for scenario in SCENARIOS_OF_FORM[form]
        if scenario in scenarios
        for site in SITES
        for draw in range(draws[form])
    ]


def main() -> None:
    """Run the power check, the streams, or the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("power", "run", "report"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--real-draws", type=int, default=DRAWS["real"])
    parser.add_argument("--ar1-draws", type=int, default=DRAWS["ar1"])
    args = parser.parse_args()
    started = time.monotonic()
    if args.step == "power":
        result = power_check()
        common.write_json_atomic(path=POWER_JSON, content=result)
        print(json.dumps(result, indent=2))
    elif args.step == "run":
        jobs = jobs_for(
            scenarios=args.scenarios, draws={"real": args.real_draws, "ar1": args.ar1_draws}
        )
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for count, _ in enumerate(pool.map(run_stream, jobs, chunksize=1), start=1):
                print(
                    f"{count}/{len(jobs)} streams, {time.monotonic() - started:.0f} s", flush=True
                )
    else:
        import e00128_gate_report as report

        report.main()
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
