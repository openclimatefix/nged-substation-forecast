"""The known-answer gate of experiment 118: can the effects filter find a weight that really moved?

Each draw builds a synthetic residual stream on the real regressors of one decision site over the
tuning year (2024-07-01 to 2025-06-30), injects a known path for the PV and heating weights, adds
noise, refits the noise matrix (the whole pipeline: the arm KEZ's variant, the site's anomaly factor
and size, a prior taken from the real KEZ fit), runs the filter, and records the filtered weights
and the day-ahead error of KEZ and GK on the stream.

**Scenarios.** S0 clean (all weights 1); S1 the PV weight rises from 1 to 1.30 over 180 days from
2024-07-01; S2 the four heating weights fall from 1 to 0.70 over 90 days from 2024-10-01; S3 both.
`a` scales the injected change (1 is the real amplitude). **Noise forms.** (a) the filter's own
model: an AR(1) anomaly per hour band at the site's fitted factor and size plus white noise; (b) a
circular month-block resample of the site's real tuning-year residuals. **Regressor conditions.**
`exact`: the filter sees the regressors that made the stream (the analysis proxy's); `eiv`
(errors in variables): the stream is made from the ERA5 and CAMS regressors and the filter sees the
analysis proxy's.

    uv run python ml-experiments/00118-*/e00118_gate.py run [--workers 6] [--draws 20]
    uv run python ml-experiments/00118-*/e00118_gate.py report
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

import e00118_arms as arms
import e00118_common as common
import e00118_noise as noise_io
import e00118_streams as streams
import numpy as np
import torch
from studies.dp_assimilation import DEFAULT_DELAY_HOURS, day_ahead_initialisation
from studies.dp_effects_filter import (
    HEATING,
    HOLIDAY_INDEX,
    LEVEL,
    N_BANDS,
    N_WEIGHTS,
    PV_INDEX,
    DayAheadTargets,
    EffectsTrace,
    FilterNoise,
    fit_effects_noise,
    run_effects_filter,
    stitched_effect_regressors,
)

GATE_DIR: Final[Path] = common.WORK_DIR / "gate"
GATE_JSON: Final[Path] = common.WORK_DIR / "gate.json"
STREAM_START: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
STREAM_END: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
PV_RAMP_START: Final[datetime] = datetime(2024, 7, 1, tzinfo=UTC)
PV_RAMP_DAYS: Final[float] = 180.0
PV_CHANGE: Final[float] = 0.30
PV_READ_DAY: Final[int] = 200
HEATING_RAMP_START: Final[datetime] = datetime(2024, 10, 1, tzinfo=UTC)
HEATING_RAMP_DAYS: Final[float] = 90.0
HEATING_CHANGE: Final[float] = -0.30
HEATING_READ_DAY: Final[int] = 120
SETTLE_DAYS: Final[int] = 60
"""Days at the start of a stream that the no-false-movement and cross-talk readings skip."""
PASS_RATE: Final[float] = 0.80
"""A reading passes if at least this share of draws pass it."""
SCENARIOS: Final[dict[str, tuple[float, float]]] = {
    "S0": (0.0, 0.0),
    "S1": (1.0, 0.0),
    "S2": (0.0, 1.0),
    "S3": (1.0, 1.0),
}
"""The PV and heating amplitudes (shares of the real change) of each scenario."""
EXTRA_SCENARIOS: Final[dict[str, tuple[float, float]]] = {
    "S1x2": (2.0, 0.0),
    "S2x2": (0.0, 2.0),
    "S3x2": (2.0, 2.0),
    "S1a10": (0.1, 0.0),
    "S1a20": (0.2, 0.0),
    "S2a10": (0.0, 0.1),
    "S2a20": (0.0, 0.2),
}
"""Doubled amplitudes (added when a weight is underpowered) and the 10% and 20% rows."""
NOISE_FORMS: Final[tuple[str, ...]] = ("ar1", "blocks")
CONDITIONS: Final[tuple[str, ...]] = ("exact", "eiv")
SITES: Final[tuple[str, ...]] = common.DECISION_SITES
SQRT_DAY_STEPS: Final[float] = 1.0


@dataclass(frozen=True)
class GateInputs:
    """A site's real regressors and noise over the tuning year, on the rows of the stream.

    Attributes:
        site: The substation's label.
        rows: The grid rows of the stream (2024-07-01 to 2025-06-30).
        seen: The regressors the filter sees (analysis proxy), shape (rows, 14).
        true: The regressors that make the stream under the `eiv` condition (ERA5 and CAMS).
        scale: The regressor scales of the real tuning stream, shape (14,).
        real_residual: The site's real residuals on the stream rows, `NaN` where absent.
        band: The hour band of each stream row.
        likelihood_mask: True on the stream rows of the tuning months.
        days: The day index (days since 2024-07-01) of each stream row.
        noise: The real KEZ noise at the site.
        prior_mean: The shared prior mean of `log q` from the real KEZ fit.
        prior_spread: The shared prior spread.
        gk_noise: GAM-K's noise at the site.
        day_ahead_cutoff: The newest usable grid position (stream index) of each row's day-ahead
            forecast, -1 where there is none.
        holiday_events: Stream indices where each holiday event's last active half-hour falls.
    """

    site: str
    rows: np.ndarray
    seen: np.ndarray
    true: np.ndarray
    scale: np.ndarray
    real_residual: np.ndarray
    band: np.ndarray
    likelihood_mask: np.ndarray
    days: np.ndarray
    noise: FilterNoise
    prior_mean: np.ndarray
    prior_spread: np.ndarray
    gk_noise: FilterNoise
    day_ahead_cutoff: np.ndarray
    holiday_events: np.ndarray


@lru_cache(maxsize=8)
def gate_inputs(site: str) -> GateInputs:
    """Build a site's gate inputs from experiment 111's data and the real KEZ fit."""
    cache = common.load_cache()
    setup = common.load_setup(cache=cache, site=site)
    fits = common.load_fits(site=site)
    data = streams.build_site_streams(cache=cache, setup=setup, fits=fits)
    era5 = stitched_effect_regressors(cache=cache, fits=fits, site=site)
    ends = cache.end_times
    rows = np.flatnonzero(
        (ends > np.datetime64(STREAM_START.replace(tzinfo=None), "us"))
        & (ends <= np.datetime64(STREAM_END.replace(tzinfo=None), "us"))
    )
    stored = noise_io.read_noise(arm="KEZ")
    site_noise = stored["sites"][site]
    noise = FilterNoise(
        q=np.array(site_noise["q"]),
        phi=site_noise["phi"],
        sigma_z=site_noise["sigma_z"],
        sigma_eps=site_noise["sigma_eps"],
        initial_sd=np.array(site_noise["initial_sd"]),
        use_anomaly=True,
    )
    hyper = common.tuned_111()["filter_hyper"]["gamk"][site]
    start64 = np.datetime64(STREAM_START.replace(tzinfo=None), "us")
    days = ((ends[rows] - start64) / np.timedelta64(1, "D")).astype(np.float64)
    initialisation = day_ahead_initialisation(end_times=ends[rows], days_ahead=1)
    delay = np.timedelta64(round(DEFAULT_DELAY_HOURS * 3600 * 1_000_000), "us")
    cutoff_grid = np.searchsorted(ends, initialisation - delay, side="right") - 1
    cutoff = np.where(cutoff_grid >= rows[0], cutoff_grid - rows[0], -1)
    holiday = np.abs(data.contributions[rows, HOLIDAY_INDEX]) > 0
    starts = np.flatnonzero(np.diff(np.r_[0, holiday.astype(int)]) == -1)
    return GateInputs(
        site=site,
        rows=rows,
        seen=data.contributions[rows],
        true=era5.contributions[rows],
        scale=data.scale,
        real_residual=data.residual[rows],
        band=data.band[rows],
        likelihood_mask=common.in_months(cache=cache, months=common.TUNING_MONTHS)[rows],
        days=days,
        noise=noise,
        prior_mean=np.array(stored["prior_mean"]),
        prior_spread=np.array(stored["prior_spread"]),
        gk_noise=arms.gk_noise(hyper=hyper, scale=data.scale),
        day_ahead_cutoff=cutoff,
        holiday_events=starts,
    )


def draw_seed(*, parts: tuple[str, ...]) -> int:
    """Return a seed that depends only on the draw's name, so a parallel run repeats exactly."""
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:12], 16)


def injected_weights(
    *, inputs: GateInputs, pv_amplitude: float, heating_amplitude: float
) -> np.ndarray:
    """Return the deviation of each of the 14 weights at each stream row, shape (rows, 14)."""
    deviation = np.zeros((len(inputs.rows), N_WEIGHTS))
    pv_days = (
        inputs.days - (PV_RAMP_START - STREAM_START).total_seconds() / 86400.0
    ) / PV_RAMP_DAYS
    deviation[:, PV_INDEX] = pv_amplitude * PV_CHANGE * np.clip(pv_days, 0.0, 1.0)
    heating_days = (
        inputs.days - (HEATING_RAMP_START - STREAM_START).total_seconds() / 86400.0
    ) / HEATING_RAMP_DAYS
    deviation[:, HEATING] = (
        heating_amplitude * HEATING_CHANGE * np.clip(heating_days, 0.0, 1.0)[:, None]
    )
    return deviation


def ar1_noise(*, inputs: GateInputs, generator: np.random.Generator) -> np.ndarray:
    """Simulate the filter's own noise model: an AR(1) anomaly per hour band plus white noise."""
    noise = inputs.noise
    n = len(inputs.rows)
    out = np.empty(n)
    stationary = noise.sigma_z / np.sqrt(1.0 - noise.phi**2)
    for band in range(N_BANDS):
        positions = np.flatnonzero(inputs.band == band)
        gaps = np.diff(positions, prepend=positions[0])
        decay = noise.phi**gaps
        shocks = generator.normal(size=len(positions))
        z = np.empty(len(positions))
        previous = stationary * generator.normal()
        for k in range(len(positions)):
            previous = decay[k] * previous + stationary * np.sqrt(1.0 - decay[k] ** 2) * shocks[k]
            z[k] = previous
        out[positions] = z
    return out + noise.sigma_eps * generator.normal(size=n)


def block_noise(*, inputs: GateInputs, generator: np.random.Generator) -> np.ndarray:
    """Resample the real residuals by whole calendar months, with replacement, keeping the position.

    The residual at a position in a month comes from the same position (day and half-hour) of
    another month of the stream, wrapped round if that month is shorter. A missing real residual
    stays missing.
    """
    months = (
        (inputs.days * 86400.0).astype("timedelta64[s]") + np.datetime64("2024-07-01", "s")
    ).astype("datetime64[M]")
    labels = np.unique(months)
    members = {label: np.flatnonzero(months == label) for label in labels}
    out = np.full(len(inputs.rows), np.nan)
    for label in labels:
        target = members[label]
        source = members[labels[generator.integers(len(labels))]]
        out[target] = inputs.real_residual[source[np.arange(len(target)) % len(source)]]
    return out


def synthetic_stream(
    *, inputs: GateInputs, scenario: str, form: str, condition: str, draw: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a draw's residual stream and the regressors the filter sees, on the stream rows.

    Args:
        inputs: The site's gate inputs.
        scenario: A key of `SCENARIOS` or `EXTRA_SCENARIOS`.
        form: `ar1` or `blocks`.
        condition: `exact` or `eiv`.
        draw: The draw's index.

    Returns:
        The residual stream (with the real stream's missing readings, `NaN` where the regressors
        are missing) and the filter's regressors, both of length `len(inputs.rows)`.
    """
    amplitudes = {**SCENARIOS, **EXTRA_SCENARIOS}[scenario]
    generator = np.random.default_rng(
        draw_seed(parts=(inputs.site, scenario, form, condition, str(draw)))
    )
    noise = (
        ar1_noise(inputs=inputs, generator=generator)
        if form == "ar1"
        else block_noise(inputs=inputs, generator=generator)
    )
    true = inputs.true if condition == "eiv" else inputs.seen
    deviation = injected_weights(
        inputs=inputs, pv_amplitude=amplitudes[0], heating_amplitude=amplitudes[1]
    )
    stream = (deviation * true).sum(axis=1) + noise
    if form == "ar1":
        stream = np.where(np.isfinite(inputs.real_residual), stream, np.nan)
    stream = np.where(
        np.isfinite(inputs.seen).all(axis=1) & np.isfinite(true).all(axis=1), stream, np.nan
    )
    return stream, inputs.seen


def daily_weights(
    *, trace_positions: np.ndarray, state: np.ndarray, inputs: GateInputs
) -> np.ndarray:
    """Return the filtered weights (not deviations) at the end of each day, shape (days, 14).

    A day with no reading holds the previous day's value.
    """
    n_days = int(np.ceil(inputs.days.max()))
    day_ends = np.arange(1, n_days + 1)
    out = np.ones((n_days, N_WEIGHTS))
    day_of_position = inputs.days[trace_positions]
    found = np.searchsorted(day_of_position, day_ends, side="right") - 1
    deviation = state[:, :N_WEIGHTS] / inputs.scale
    for d, row in enumerate(found):
        if row >= 0:
            out[d] = 1.0 + deviation[row]
            out[d, LEVEL] = deviation[row, LEVEL]
        elif d:
            out[d] = out[d - 1]
    return out


def day_ahead_mae(
    *,
    stream: np.ndarray,
    regressors: np.ndarray,
    trace: EffectsTrace,
    inputs: GateInputs,
    phi: float,
) -> float:
    """Return a filter's mean absolute day-ahead error on a stream, after the settling days."""
    scaled = regressors / inputs.scale
    errors = []
    positions = trace.positions
    for row in np.flatnonzero(np.isfinite(stream) & (inputs.days > SETTLE_DAYS)):
        cutoff = inputs.day_ahead_cutoff[row]
        found = int(np.searchsorted(positions, cutoff, side="right")) - 1
        if cutoff < 0 or found < 0:
            continue
        state = trace.state[found]
        band = inputs.band[row]
        decay = phi ** max(row - int(positions[found]), 0)
        correction = scaled[row] @ state[:N_WEIGHTS] + decay * state[N_WEIGHTS + band]
        errors.append(abs(stream[row] - correction))
    return float(np.mean(errors))


def run_draw(job: tuple[str, str, str, str, int]) -> str:
    """Run one draw and store its metrics, unless a file for it exists.

    Args:
        job: Site, scenario, noise form, regressor condition, and draw index.

    Returns:
        The path of the draw's file.
    """
    site, scenario, form, condition, draw = job
    path = GATE_DIR / scenario / form / condition / f"{site}_{draw:03d}.json"
    if path.exists():
        return str(path)
    torch.set_num_threads(1)
    inputs = gate_inputs(site)
    stream, seen = synthetic_stream(
        inputs=inputs, scenario=scenario, form=form, condition=condition, draw=draw
    )
    standardised = seen / inputs.scale
    fitted = fit_effects_noise(
        residual=stream,
        regressors=standardised,
        band=inputs.band,
        scale=inputs.scale,
        variant=arms.FITTED_ARMS["KEZ"].variant,
        phi=inputs.noise.phi,
        sigma_z=inputs.noise.sigma_z,
        sigma_eps_start=inputs.noise.sigma_eps,
        likelihood_mask=inputs.likelihood_mask & np.isfinite(stream),
        prior_mean=inputs.prior_mean,
        prior_spread=inputs.prior_spread,
        lead_matched=day_ahead_scored(inputs=inputs, stream=stream),
    )
    kez = run_effects_filter(
        residual=stream, regressors=standardised, band=inputs.band, noise=fitted.noise
    )
    gk = run_effects_filter(
        residual=stream, regressors=standardised, band=inputs.band, noise=inputs.gk_noise
    )
    weights = daily_weights(trace_positions=kez.positions, state=kez.state, inputs=inputs)
    content = {
        "site": site,
        "scenario": scenario,
        "form": form,
        "condition": condition,
        "draw": draw,
        "q": fitted.noise.q.tolist(),
        "converged": fitted.converged,
        "pv_weight": weights[:, PV_INDEX].tolist(),
        "heating_weights": weights[:, HEATING].tolist(),
        "holiday_weight": weights[:, HOLIDAY_INDEX].tolist(),
        "level_deviation": weights[:, LEVEL].tolist(),
        "mae_kez": day_ahead_mae(
            stream=stream, regressors=seen, trace=kez, inputs=inputs, phi=fitted.noise.phi
        ),
        "mae_gk": day_ahead_mae(
            stream=stream, regressors=seen, trace=gk, inputs=inputs, phi=inputs.gk_noise.phi
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content))
    temporary.rename(path)
    return str(path)


def day_ahead_scored(*, inputs: GateInputs, stream: np.ndarray) -> DayAheadTargets:
    """Return the stream's tuning-month readings with a day-ahead cutoff, and their cutoffs."""
    rows = np.flatnonzero(
        inputs.likelihood_mask & np.isfinite(stream) & (inputs.day_ahead_cutoff >= 0)
    )
    return DayAheadTargets(target_rows=rows, cutoff_rows=inputs.day_ahead_cutoff[rows])


def power_check() -> dict:
    """Return, per site, the injected change in posterior standard deviations at the read days.

    The posterior standard deviation of the PV weight at day 200 and of the mean heating weight at
    day 120 comes from the real KEZ noise run on the site's real tuning-year stream. The heating
    sd is the root mean square over the four bands (the bands' weights are treated as independent,
    which overstates the sd of their mean).
    """
    out = {}
    for site in SITES:
        inputs = gate_inputs(site)
        stream = inputs.real_residual
        trace = run_effects_filter(
            residual=stream,
            regressors=inputs.seen / inputs.scale,
            band=inputs.band,
            noise=inputs.noise,
        )
        row_pv = int(np.searchsorted(inputs.days[trace.positions], PV_READ_DAY, side="right")) - 1
        row_heating = (
            int(np.searchsorted(inputs.days[trace.positions], 92 + HEATING_READ_DAY, side="right"))
            - 1
        )
        sd_pv = trace.sd[row_pv, PV_INDEX] / inputs.scale[PV_INDEX]
        sd_heating = float(
            np.sqrt(np.mean((trace.sd[row_heating, HEATING] / inputs.scale[HEATING]) ** 2))
        )
        out[site] = {
            "sd_pv": float(sd_pv),
            "sd_heating": sd_heating,
            "pv_in_sds": PV_CHANGE / float(sd_pv),
            "heating_in_sds": abs(HEATING_CHANGE) / sd_heating,
        }
    return out


def jobs_for(*, scenarios: list[str], draws: int) -> list[tuple[str, str, str, str, int]]:
    """Return every draw of the scenarios, over the sites, noise forms, and conditions."""
    return [
        (site, scenario, form, condition, draw)
        for scenario in scenarios
        for form in NOISE_FORMS
        for condition in CONDITIONS
        for site in SITES
        for draw in range(draws)
    ]


def main() -> None:
    """Run the draws or print the readings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("power", "run", "report"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    args = parser.parse_args()
    started = time.monotonic()
    if args.step == "power":
        result = power_check()
        common.write_json_atomic(path=common.WORK_DIR / "gate_power.json", content=result)
        print(json.dumps(result, indent=2))
    elif args.step == "run":
        jobs = jobs_for(scenarios=args.scenarios, draws=args.draws)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for count, _ in enumerate(pool.map(run_draw, jobs, chunksize=4), start=1):
                if count % 50 == 0:
                    print(
                        f"{count}/{len(jobs)} draws, {time.monotonic() - started:.0f} s", flush=True
                    )
    else:
        import e00118_gate_report as report

        report.main()
    print(f"done in {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    main()
