"""The known-answer gate of experiment 106, run on the convex model after tuning.

A failed blocking test stops scoring until it is fixed. The gate runs at the settings
`e00106_tune.py` chose. The convex model has no interaction network, so the part of the plan's
test 4 that asks for the network's midday movement does not apply, and the plan's note about
running every test with the network in its decision setting is moot here.

1. **Synthetic drift recovery (blocking).** Eight synthetic series are built on the real calendar,
   weather, and solar tensors of the five decision sites from 2019-09 to 2025-06: a 10-point ramp
   up at two, a 10-point ramp down at two, a +8-point step on a known date at two, a 3-point rise
   in the last 3 months before the cut at one, and no drift at one. The power is the real fit's
   own terms plus the known drift path (each band scaled by `BAND_SCALE` for the ramps) and
   independent noise at the fit's residual spread. D1 must recover each path within 1.5 points
   root mean square (after centring each band), hold the late rise to within 1.5 points at the cut,
   and the D1-plus-steps arm must place each true step within one knot and 25% of its size and add
   under 1 point of steps at the no-drift series.
2. **The filter tracks a seasonal misfit causally and does not invent one (blocking, part a).** Into
   white-noise residuals a known 3-point annual cycle is injected. The level must follow the cycle
   with its amplitude within 25%, lag it by at most 7 days, and never lead it; on the same
   residuals without the cycle the level's annual harmonic must be under 0.5 points. Part b is
   reported only: the annual harmonic of the filter's level on the real tuning-year residuals of
   A0, and its correlation with the 7-day mean temperature, per site.
3. **Injected transfers (blocking).** Compact transfers are injected into the tuning year's
   residuals at D10, D11, D14, and D16 only where no real hand-filtered switching mask lies within
   48 hours. A transfer takes a share of the site's own power (so it follows the site's daily
   shape), the share drifting between 0.5 and 2 times its start, over 4 to 8 days. The share of the
   moved power that the filter's state carries must reach 50% by the end of day 2, lie between
   0.6 and 1.3 from day 2 to the end, and fall under 25% within 3 days of the end. The shapes follow
   experiment 43's injection harness, which a script cannot import. The filter on the real
   hand-filtered transfers is reported beside the test.
4. **The drift does not take the solar term (blocking).** A synthetic solar farm, the derated unit
   shape of another decision site times 30% of the site's pinned capacity, is switched on from
   2023-07-01 at D10, D11, D14, and D16 and subtracted from the real power. With the capacity
   pinned, the drift outside the midday band must move by under 0.5 points. Reported: the midday
   band's movement, and the recovered-to-injected capacity ratio of the fitted-capacity arm against
   [0.75, 1.25]. The farm is synthetic, so no generator's output appears here.

    uv run python ml-experiments/00106-*/e00106_gate.py
"""

import json
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from typing import Final

import e00106_common as common
import e00106_g as g
import numpy as np
import polars as pl
from studies.dp_assimilation import (
    DEFAULT_DELAY_HOURS,
    CorrectionTrace,
    FilterHyperparameters,
    run_kalman,
    usable_cutoff_positions,
)
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    N_BANDS,
    GamFit,
    GamSpec,
    TrainingWindow,
    derated_unit_shape,
    design_matrix,
    fit_gam,
    pinned_solar_points,
    training_rows,
)
from studies.dp_pinned_capacity import PinnedCapacity, pinned_capacity, read_vintages
from studies.sources import ML_EXPERIMENTS_DATA_DIR

BAND_SCALE: Final[tuple[float, ...]] = (1.0, 0.6, 1.0, 0.8)
"""How strongly each hour band carries a ramp, so the paths differ by band."""
NIGHT: Final[int] = 0
MIDDAY: Final[int] = 2
RMS_LIMIT: Final[float] = 1.5
HELD_LIMIT: Final[float] = 1.5
STEP_SIZE: Final[float] = 8.0
STEP_SIZE_TOLERANCE: Final[float] = 0.25
NO_DRIFT_STEP_LIMIT: Final[float] = 1.0
CYCLE_AMPLITUDE: Final[float] = 3.0
CYCLE_AMPLITUDE_TOLERANCE: Final[float] = 0.25
CYCLE_LAG_LIMIT_DAYS: Final[float] = 7.0
NO_CYCLE_LIMIT: Final[float] = 0.5
WHITE_NOISE_POINTS: Final[float] = 2.0
FARM_SHARE: Final[float] = 0.3
FARM_START: Final[datetime] = datetime(2023, 7, 1, tzinfo=UTC)
FARM_DRIFT_LIMIT: Final[float] = 0.5
CAPACITY_RATIO_RANGE: Final[tuple[float, float]] = (0.75, 1.25)
TRANSFER_SITES: Final[tuple[str, ...]] = ("D10", "D11", "D14", "D16")
TRANSFERS_PER_SITE: Final[int] = 6
TRANSFER_CLEARANCE_HOURS: Final[int] = 48
STEPS_PER_HOUR: Final[int] = 2


@dataclass(frozen=True)
class Truth:
    """A synthetic drift path.

    Attributes:
        label: The series' label.
        base_site: The real site whose tensors the series uses.
        kind: `ramp_up`, `ramp_down`, `step`, `late_rise`, or `none`.
        step_date: The step's date, for a step.
    """

    label: str
    base_site: str
    kind: str
    step_date: datetime | None = None


SYNTHETIC: Final[tuple[Truth, ...]] = (
    Truth("S1", "D5", "ramp_up"),
    Truth("S2", "D10", "ramp_down"),
    Truth("S3", "D11", "ramp_up"),
    Truth("S4", "D14", "ramp_down"),
    Truth("S5", "D16", "step", datetime(2023, 3, 1, tzinfo=UTC)),
    Truth("S6", "D5", "step", datetime(2022, 6, 15, tzinfo=UTC)),
    Truth("S7", "D10", "late_rise"),
    Truth("S8", "D11", "none"),
)


def days_since_epoch(*, end_times: np.ndarray) -> np.ndarray:
    """Return the days since 1970-01-01 of half-hour ends."""
    return end_times.astype("datetime64[s]").astype(np.int64) / 86400.0


def truth_level(
    *, truth: Truth, days: np.ndarray, band: np.ndarray, first: float, cut: float
) -> np.ndarray:
    """Return the true drift at the given days in the given bands, in points.

    Args:
        truth: The synthetic series.
        days: Days since the epoch.
        band: The hour band of each day.
        first: The first training day.
        cut: The cut's day.

    Returns:
        The drift in points.

    Raises:
        ValueError: If the kind is unknown.
    """
    scale = np.array(BAND_SCALE)[band]
    if truth.kind == "ramp_up":
        return scale * 10.0 * np.clip((days - first) / (cut - first), 0.0, 1.0)
    if truth.kind == "ramp_down":
        return -scale * 10.0 * np.clip((days - first) / (cut - first), 0.0, 1.0)
    if truth.kind == "step" and truth.step_date is not None:
        step_day = truth.step_date.timestamp() / 86400.0
        return STEP_SIZE * (days >= step_day)
    if truth.kind == "late_rise":
        return 3.0 * np.clip((days - (cut - 91.0)) / 91.0, 0.0, 1.0)
    if truth.kind == "none":
        return np.zeros(len(days))
    msg = f"unknown synthetic drift kind {truth.kind}"
    raise ValueError(msg)


def _synthetic_cache(
    *, cache: FeatureCache, vintages: pl.DataFrame, truth: Truth, spec: GamSpec, seed: int
) -> FeatureCache:
    """Return a cache whose power for `truth.label` is the real fit's terms plus a known drift."""
    cut = common.SCORED_CUT
    features = cache.sites[truth.base_site]
    pinned = pinned_capacity(vintages=vintages, table=features.table, cut=cut)
    real = fit_gam(
        cache=cache,
        site=truth.base_site,
        specs=[spec],
        cut=cut,
        window=TrainingWindow(),
        season_years=common.SEASON_YEARS,
        pinned=pinned,
    )
    rows = training_rows(features=features, cache=cache, cut=cut, window=TrainingWindow())
    theta = real.theta[0].copy()
    theta[real.layout.slices["drift"]] = 0.0
    design = design_matrix(cache=cache, features=features, layout=real.layout, index=rows)
    base = design @ theta - pinned_solar_points(
        cache=cache, features=features, pinned=pinned, index=rows
    )
    noise_sd = float(
        np.std(
            features.y[rows]
            - (
                design @ real.theta[0]
                - pinned_solar_points(cache=cache, features=features, pinned=pinned, index=rows)
            )
        )
    )
    days = days_since_epoch(end_times=cache.end_times[rows])
    drift = truth_level(
        truth=truth,
        days=days,
        band=cache.calendar.band[rows].astype(np.int64),
        first=float(days.min()),
        cut=days_since_epoch(end_times=np.array([common.to_64(moment=cut)]))[0],
    )
    y = np.full(len(cache.end_times), np.nan)
    y[rows] = base + drift + np.random.default_rng(seed).normal(0.0, noise_sd, size=len(rows))
    synthetic = replace(features, y=y)
    return replace(cache, sites={**cache.sites, truth.label: synthetic})


def _knot_paths(*, fit: GamFit) -> tuple[np.ndarray, np.ndarray]:
    """Return a fit's drift and cumulative step levels at its knots, shape (bands, knots) each."""
    n_knots = len(fit.layout.knot_days)
    drift = fit.theta[0, fit.layout.slices["drift"]].reshape(N_BANDS, n_knots)
    steps = np.zeros((N_BANDS, n_knots))
    if "steps" in fit.layout.slices:
        block = fit.theta[0, fit.layout.slices["steps"]]
        signed = (block[: block.size // 2] - block[block.size // 2 :]).reshape(N_BANDS, n_knots - 1)
        steps[:, 1:] = np.cumsum(signed, axis=1)
    return drift, steps


def _signed_steps(*, fit: GamFit) -> np.ndarray:
    """Return the fitted step sizes, shape (bands, knots - 1)."""
    n_knots = len(fit.layout.knot_days)
    block = fit.theta[0, fit.layout.slices["steps"]]
    return (block[: block.size // 2] - block[block.size // 2 :]).reshape(N_BANDS, n_knots - 1)


def test_one(*, cache: FeatureCache, vintages: pl.DataFrame, tuned: dict) -> list[dict]:
    """Run the synthetic drift-recovery test; return one result row per check."""
    arms = g.arm_definitions(tuned=tuned)
    results: list[dict] = []
    for position, truth in enumerate(SYNTHETIC):
        synthetic = _synthetic_cache(
            cache=cache, vintages=vintages, truth=truth, spec=arms["a1"].spec, seed=100 + position
        )
        features = synthetic.sites[truth.label]
        pinned = pinned_capacity(
            vintages=vintages, table=cache.sites[truth.base_site].table, cut=common.SCORED_CUT
        )
        rows = training_rows(
            features=features, cache=synthetic, cut=common.SCORED_CUT, window=TrainingWindow()
        )
        first = float(days_since_epoch(end_times=synthetic.end_times[rows]).min())
        cut_day = days_since_epoch(end_times=np.array([common.to_64(moment=common.SCORED_CUT)]))[0]
        # The synthetic series shares the base site's table, so give it its own site entry.
        fit = fit_gam(
            cache=synthetic,
            site=truth.label,
            specs=[arms["a1"].spec],
            cut=common.SCORED_CUT,
            window=TrainingWindow(),
            season_years=common.SEASON_YEARS,
            pinned=pinned,
        )
        drift_fit, _ = _knot_paths(fit=fit)
        knots = fit.layout.knot_days
        true_path = np.stack(
            [
                truth_level(
                    truth=truth,
                    days=knots,
                    band=np.full(len(knots), band),
                    first=first,
                    cut=cut_day,
                )
                for band in range(N_BANDS)
            ]
        )
        centred_fit = drift_fit - drift_fit.mean(axis=1, keepdims=True)
        centred_true = true_path - true_path.mean(axis=1, keepdims=True)
        rms = float(np.sqrt(((centred_fit - centred_true) ** 2).mean()))
        results.append(
            {
                "test": "1 drift recovery",
                "series": truth.label,
                "check": f"{truth.kind} rms",
                "value": rms,
                "limit": RMS_LIMIT,
                "passed": rms < RMS_LIMIT,
                "blocking": True,
            }
        )
        if truth.kind == "late_rise":
            held = float(np.abs(centred_fit[:, -1] - centred_true[:, -1]).max())
            results.append(
                {
                    "test": "1 drift recovery",
                    "series": truth.label,
                    "check": "held value at the cut",
                    "value": held,
                    "limit": HELD_LIMIT,
                    "passed": held < HELD_LIMIT,
                    "blocking": True,
                }
            )
        if truth.kind in ("step", "none"):
            active = arms["a1s_smooth"].spec
            results += _step_checks(
                cache=synthetic,
                truth=truth,
                spec=active,
                pinned=pinned,
                knots=knots,
                label="a1s_smooth",
                blocking=True,
            )
            results += _step_checks(
                cache=synthetic,
                truth=truth,
                spec=arms["a1s"].spec,
                pinned=pinned,
                knots=knots,
                label="tuned arm",
                blocking=False,
            )
    return results


def _step_checks(
    *,
    cache: FeatureCache,
    truth: Truth,
    spec: GamSpec,
    pinned: PinnedCapacity,
    knots: np.ndarray,
    label: str,
    blocking: bool,
) -> list[dict]:
    """Fit a D1-plus-steps arm to a synthetic series and check where its level path puts the step.

    The level path is the drift plus the cumulative steps, which is what a forecast uses. The
    smooth drift takes part of a step, so the step columns alone understate the step: the check
    reads the path's rise from two knots before the true step to one knot after it.

    Args:
        cache: The cache holding the synthetic series.
        truth: The synthetic series.
        spec: The arm's penalties; its steps must be able to switch on.
        pinned: The pinned capacity of the series' base site.
        knots: The drift knots of the fit, as days since the epoch.
        label: Names the arm in the report.
        blocking: Whether a failure stops scoring.

    Returns:
        One row per check.
    """
    fit = fit_gam(
        cache=cache,
        site=truth.label,
        specs=[spec],
        cut=common.SCORED_CUT,
        window=TrainingWindow(),
        season_years=common.SEASON_YEARS,
        pinned=pinned,
    )
    signed = _signed_steps(fit=fit)
    if truth.kind == "none":
        total = float(np.abs(signed).sum(axis=1).max())
        return [
            {
                "test": "1 drift recovery",
                "series": truth.label,
                "check": f"steps added at no drift ({label})",
                "value": total,
                "limit": NO_DRIFT_STEP_LIMIT,
                "passed": total < NO_DRIFT_STEP_LIMIT,
                "blocking": blocking,
            }
        ]
    step_day = truth.step_date.timestamp() / 86400.0 if truth.step_date else 0.0
    true_knot = int(np.searchsorted(knots, step_day))
    drift, steps = _knot_paths(fit=fit)
    level = drift + steps
    rows = []
    for band in range(N_BANDS):
        jumps = np.diff(level[band])
        found_knot = int(np.argmax(np.abs(jumps))) + 1
        size = float(
            level[band, min(true_knot + 1, level.shape[1] - 1)] - level[band, true_knot - 2]
        )
        placed = abs(found_knot - true_knot) <= 1
        sized = abs(size - STEP_SIZE) <= STEP_SIZE_TOLERANCE * STEP_SIZE
        rows.append(
            {
                "test": "1 drift recovery",
                "series": truth.label,
                "check": f"step placed and sized, band {band} ({label})",
                "value": size,
                "limit": STEP_SIZE,
                "passed": bool(placed and sized),
                "blocking": blocking,
            }
        )
    return rows


def _annual_fit(*, days: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Return the amplitude and the phase (radians) of the annual harmonic of a daily series."""
    angle = 2.0 * np.pi * days / 365.25
    design = np.column_stack([np.ones(len(days)), np.sin(angle), np.cos(angle)])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return float(np.hypot(coefficients[1], coefficients[2])), float(
        np.arctan2(coefficients[2], coefficients[1])
    )


def _daily_level(*, trace: CorrectionTrace, cache: FeatureCache) -> tuple[np.ndarray, np.ndarray]:
    """Return the day numbers and the mean over bands of the level after each day's last reading."""
    days: list[float] = []
    levels: list[float] = []
    end_days = days_since_epoch(end_times=cache.end_times)
    per_band = []
    for band in range(N_BANDS):
        band_days = np.floor(end_days[trace.positions[band]])
        last = np.r_[np.flatnonzero(np.diff(band_days)), len(band_days) - 1]
        per_band.append(dict(zip(band_days[last], trace.level[band][last], strict=True)))
    common_days = sorted(set.intersection(*(set(d) for d in per_band)))
    for day in common_days:
        days.append(float(day))
        levels.append(float(np.mean([d[day] for d in per_band])))
    return np.array(days), np.array(levels)


def test_two(*, cache: FeatureCache, tuned: dict) -> list[dict]:
    """Run the filter's seasonal-misfit test; return one result row per check."""
    results: list[dict] = []
    hypers = tuned["filter_hyper"]["a0"]
    for site in common.DECISION_SITES:
        features = cache.sites[site]
        hyper = FilterHyperparameters.from_array(values=np.array(hypers[site]))
        present = features.present & np.isfinite(features.y)
        index = np.flatnonzero(present & (cache.end_times > np.datetime64("2021-07-01", "us")))
        generator = np.random.default_rng(7)
        noise = np.full(len(cache.end_times), np.nan)
        noise[index] = generator.normal(0.0, WHITE_NOISE_POINTS, size=len(index))
        days_all = days_since_epoch(end_times=cache.end_times)
        cycle = CYCLE_AMPLITUDE * np.sin(2.0 * np.pi * days_all / 365.25)
        for label, stream in (("with cycle", noise + cycle), ("white noise only", noise)):
            trace = run_kalman(residual=stream, band=cache.calendar.band, hyper=hyper)
            days, level = _daily_level(trace=trace, cache=cache)
            keep = days > days.min() + 90.0
            amplitude, phase = _annual_fit(days=days[keep], values=level[keep])
            if label == "with cycle":
                lag_days = -phase / (2.0 * np.pi) * 365.25
                lag_days = (lag_days + 182.625) % 365.25 - 182.625
                ok_amplitude = (
                    abs(amplitude - CYCLE_AMPLITUDE) <= CYCLE_AMPLITUDE_TOLERANCE * CYCLE_AMPLITUDE
                )
                ok_lag = -0.5 <= lag_days <= CYCLE_LAG_LIMIT_DAYS
                results += [
                    {
                        "test": "2a filter cycle",
                        "series": site,
                        "check": "amplitude",
                        "value": amplitude,
                        "limit": CYCLE_AMPLITUDE,
                        "passed": bool(ok_amplitude),
                        "blocking": True,
                    },
                    {
                        "test": "2a filter cycle",
                        "series": site,
                        "check": "lag in days (never leads)",
                        "value": float(lag_days),
                        "limit": CYCLE_LAG_LIMIT_DAYS,
                        "passed": bool(ok_lag),
                        "blocking": True,
                    },
                ]
            else:
                results.append(
                    {
                        "test": "2a filter cycle",
                        "series": site,
                        "check": "no-cycle annual harmonic",
                        "value": amplitude,
                        "limit": NO_CYCLE_LIMIT,
                        "passed": amplitude < NO_CYCLE_LIMIT,
                        "blocking": True,
                    }
                )
    return results


def real_cycle_diagnostic(
    *, cache: FeatureCache, vintages: pl.DataFrame, tuned: dict
) -> list[dict]:
    """Report the annual harmonic of the filter's level on the tuning year's real A0 residuals."""
    rows = []
    definition = g.arm_definitions(tuned=tuned)["a0"]
    for site in common.DECISION_SITES:
        run = g.run_site(
            cache=cache, vintages=vintages, site=site, cut=common.TUNING_CUT, definition=definition
        )
        hyper = FilterHyperparameters.from_array(values=np.array(tuned["filter_hyper"]["a0"][site]))
        trace = run_kalman(
            residual=run.residual_stream(cache=cache), band=cache.calendar.band, hyper=hyper
        )
        days, level = _daily_level(trace=trace, cache=cache)
        amplitude, _ = _annual_fit(days=days, values=level)
        temperature = cache.sites[site].smoothed_temperature_c[:, 3]
        end_days = np.floor(days_since_epoch(end_times=cache.end_times))
        weekly = {d: float(np.nanmean(temperature[end_days == d])) for d in days}
        corr = float(np.corrcoef(level, [weekly[d] for d in days])[0, 1])
        rows.append(
            {
                "site": site,
                "annual harmonic of the level (points)": amplitude,
                "correlation with 7-day mean temperature": corr,
            }
        )
    return rows


@dataclass(frozen=True)
class Transfer:
    """An injected transfer: onset and end positions on the grid and the moved power."""

    site: str
    onset: int
    end: int
    moved: np.ndarray


def _transfers(*, cache: FeatureCache, site: str, seed: int) -> list[Transfer]:
    """Choose transfers in the tuning year, away from every real switching mask."""
    features = cache.sites[site]
    generator = np.random.default_rng(seed)
    first = int(
        np.searchsorted(cache.end_times, common.to_64(moment=common.TUNING_CUT), side="right")
    )
    last = int(np.searchsorted(cache.end_times, common.to_64(moment=common.REPLICATE_CUT)))
    clearance = TRANSFER_CLEARANCE_HOURS * STEPS_PER_HOUR
    chosen: list[Transfer] = []
    taken = np.zeros(len(cache.end_times), dtype=bool)
    attempts = 0
    while len(chosen) < TRANSFERS_PER_SITE and attempts < 400:
        attempts += 1
        length = int(generator.integers(4, 9)) * 48
        onset = int(generator.integers(first + 7 * 48, last - length - 20 * 48))
        end = onset + length
        low, high = onset - clearance - 5 * 48, end + clearance + 5 * 48
        if features.switching_mask[low:high].any() or taken[low:high].any():
            continue
        if not features.present[onset:end].all() or not np.isfinite(features.y[onset:end]).all():
            continue
        share = float(generator.uniform(0.15, 0.35))
        drift = np.exp(
            np.interp(np.arange(length), [0, length], [0.0, generator.uniform(-0.7, 0.7)])
        )
        moved = share * np.clip(drift, 0.5, 2.0) * features.y[onset:end]
        chosen.append(Transfer(site=site, onset=onset, end=end, moved=moved))
        taken[low:high] = True
    return chosen


def _state_carried(*, trace: CorrectionTrace, cache: FeatureCache, position: int) -> float:
    """Return the mean over bands of level plus transient after the newest usable readings."""
    cutoff = int(
        usable_cutoff_positions(
            end_times=cache.end_times,
            initialisation=cache.end_times[position : position + 1],
            delay_hours=DEFAULT_DELAY_HOURS,
        )[0]
    )
    values = []
    for band in range(N_BANDS):
        found = int(np.searchsorted(trace.positions[band], cutoff, side="right")) - 1
        if found >= 0:
            values.append(trace.level[band][found] + trace.anomaly[band][found])
    return float(np.mean(values)) if values else 0.0


def _carried_share(
    *,
    cache: FeatureCache,
    trace: CorrectionTrace,
    base_trace: CorrectionTrace,
    onset: int,
    moved_scale: float,
    offset: int,
) -> float:
    """Return the share of the moved power the filter's state carries `offset` steps after onset.

    Args:
        cache: The feature cache.
        trace: The filter run on the stream with the transfer.
        base_trace: The filter run on the stream without it.
        onset: The transfer's first grid position.
        moved_scale: The transfer's mean moved power in points.
        offset: Half-hours after the onset.

    Returns:
        The difference between the two states, negated (a transfer lowers the donor's power),
        divided by the moved power.
    """
    position = onset + offset
    injected = _state_carried(trace=trace, cache=cache, position=position)
    natural = _state_carried(trace=base_trace, cache=cache, position=position)
    return -(injected - natural) / moved_scale


def test_three(*, cache: FeatureCache, vintages: pl.DataFrame, tuned: dict) -> list[dict]:
    """Run the injected-transfer test; return one result row per check."""
    results: list[dict] = []
    definition = g.arm_definitions(tuned=tuned)["a1"]
    gains: dict[str, list[tuple[float, float, float]]] = {}
    for site in TRANSFER_SITES:
        run = g.run_site(
            cache=cache, vintages=vintages, site=site, cut=common.TUNING_CUT, definition=definition
        )
        hyper = FilterHyperparameters.from_array(values=np.array(tuned["filter_hyper"]["a1"][site]))
        base = run.residual_stream(cache=cache)
        base_trace = run_kalman(residual=base, band=cache.calendar.band, hyper=hyper)
        for transfer in _transfers(cache=cache, site=site, seed=11):
            stream = base.copy()
            stream[transfer.onset : transfer.end] -= transfer.moved
            trace = run_kalman(residual=stream, band=cache.calendar.band, hyper=hyper)
            carried = partial(
                _carried_share,
                cache=cache,
                trace=trace,
                base_trace=base_trace,
                onset=transfer.onset,
                moved_scale=float(transfer.moved.mean()),
            )
            day = 48
            by_day_two = float(np.mean([carried(offset=2 * day - k) for k in range(9)]))
            inside = float(
                np.mean(
                    [carried(offset=k) for k in range(2 * day, transfer.end - transfer.onset, 12)]
                )
            )
            after_end = transfer.end - transfer.onset + 3 * day
            gone = float(np.mean([carried(offset=after_end + k) for k in range(9)]))
            gains.setdefault(site, []).append((by_day_two, inside, gone))
    flat = np.array([row for rows in gains.values() for row in rows])
    median = np.median(flat, axis=0)
    checks = [
        ("share carried by the end of day 2", median[0], "at least 0.5", median[0] >= 0.5),
        ("share carried from day 2 to the end", median[1], "0.6 to 1.3", 0.6 <= median[1] <= 1.3),
        ("share left 3 days after the end", median[2], "under 0.25", median[2] < 0.25),
    ]
    for name, value, limit, passed in checks:
        results.append(
            {
                "test": "3 injected transfers",
                "series": f"median of {len(flat)}",
                "check": name,
                "value": float(value),
                "limit": limit,
                "passed": bool(passed),
                "blocking": True,
            }
        )
    return results


def _fit_at_cut(
    on: FeatureCache, definition: g.ArmDefinition, *, site: str, pinned: PinnedCapacity
) -> GamFit:
    """Fit an arm at the scored cut, with the pinned capacity only if the arm pins it."""
    return fit_gam(
        cache=on,
        site=site,
        specs=[definition.spec],
        cut=common.SCORED_CUT,
        window=TrainingWindow(),
        season_years=common.SEASON_YEARS,
        pinned=pinned if definition.spec.capacity == "pinned" else None,
    )


def test_four(
    *, cache: FeatureCache, vintages: pl.DataFrame, tuned: dict
) -> tuple[list[dict], list[dict]]:
    """Run the injected-farm test; return the checks and the reported midday and capacity rows."""
    arms = g.arm_definitions(tuned=tuned)
    cut = common.SCORED_CUT
    checks: list[dict] = []
    reports: list[dict] = []
    order = list(TRANSFER_SITES)
    for position, site in enumerate(order):
        features = cache.sites[site]
        other = cache.sites[order[(position + 1) % len(order)]]
        pinned = pinned_capacity(vintages=vintages, table=features.table, cut=cut)
        rows = training_rows(features=features, cache=cache, cut=cut, window=TrainingWindow())
        started = rows[cache.end_times[rows] >= common.to_64(moment=FARM_START)]
        capacity_points = FARM_SHARE * pinned.held_points
        farm = np.zeros(len(cache.end_times))
        farm[started] = capacity_points * derated_unit_shape(features=other, index=started)
        injected_y = np.where(np.isfinite(features.y), features.y - farm, np.nan)
        injected = replace(cache, sites={**cache.sites, site: replace(features, y=injected_y)})

        fit = partial(_fit_at_cut, site=site, pinned=pinned)

        base_drift, _ = _knot_paths(fit=fit(cache, arms["a1"]))
        injected_drift, _ = _knot_paths(fit=fit(injected, arms["a1"]))
        knots = fit(cache, arms["a1"]).layout.knot_days
        after = knots >= FARM_START.timestamp() / 86400.0
        change = (injected_drift - base_drift)[:, after].mean(axis=1) - (
            injected_drift - base_drift
        )[:, ~after].mean(axis=1)
        outside = float(abs(change[NIGHT]))
        checks.append(
            {
                "test": "4 injected farm",
                "series": site,
                "check": "night-band drift moves",
                "value": outside,
                "limit": FARM_DRIFT_LIMIT,
                "passed": outside < FARM_DRIFT_LIMIT,
                "blocking": True,
            }
        )
        capacities_base = fit(cache, arms["a1_fc"]).capacity_points()
        capacities_injected = fit(injected, arms["a1_fc"]).capacity_points()
        years = [year for year in capacities_base if year >= FARM_START.year]
        recovered = float(np.mean([capacities_injected[y] - capacities_base[y] for y in years]))
        reports.append(
            {
                "site": site,
                "night drift change (points)": float(change[NIGHT]),
                "morning drift change (points)": float(change[1]),
                "midday drift change (points)": float(change[MIDDAY]),
                "evening drift change (points)": float(change[3]),
                "injected capacity (points)": capacity_points,
                "recovered-to-injected capacity ratio": recovered / capacity_points,
                "in 0.75 to 1.25": CAPACITY_RATIO_RANGE[0]
                <= recovered / capacity_points
                <= CAPACITY_RATIO_RANGE[1],
            }
        )
    return checks, reports


def real_transfer_report(*, cache: FeatureCache, vintages: pl.DataFrame, tuned: dict) -> list[dict]:
    """Report the filter's state on the real hand-filtered transfers that start at a decision site.

    The state is read from the out-of-sample residuals of arm A1 in each scored year (the fits at
    the three cuts), after the second day of a transfer of three days or more.

    Args:
        cache: The feature cache.
        vintages: The real-time capacity vintages.
        tuned: The tuned settings.

    Returns:
        One row per transfer: the donor, its length, the moved power, and the filter's state.
    """
    labelled = pl.read_parquet(
        ML_EXPERIMENTS_DATA_DIR / "00041-sw2-0a-labeller-gate" / "real" / "labelled_set.parquet"
    ).filter(
        ~pl.col("is_sentinel"),
        pl.col("kind") == "real_operation",
        pl.col("visibility").is_in(["conserved", "one_sided"]),
    )
    definition = g.arm_definitions(tuned=tuned)["a1"]
    rows = []
    for cut in common.CUTS.values():
        for site in common.DECISION_SITES:
            run = g.run_site(
                cache=cache, vintages=vintages, site=site, cut=cut, definition=definition
            )
            hyper = FilterHyperparameters.from_array(
                values=np.array(tuned["filter_hyper"]["a1"][site])
            )
            trace = run_kalman(
                residual=run.residual_stream(cache=cache), band=cache.calendar.band, hyper=hyper
            )
            effective = cache.sites[site].table.effective_capacity_mw
            for event in labelled.filter(
                pl.col("donor") == str(cache.sites[site].table.time_series_id)
            ).iter_rows(named=True):
                onset = np.datetime64(event["onset_utc"].replace(tzinfo=None), "us")
                end = np.datetime64(event["end_utc"].replace(tzinfo=None), "us")
                if not (
                    common.to_64(moment=cut) <= onset
                    and end <= common.to_64(moment=common.year_end(cut=cut))
                ):
                    continue
                if (end - onset) / np.timedelta64(1, "D") < 3.0:
                    continue
                start = int(np.searchsorted(cache.end_times, onset))
                stop = int(np.searchsorted(cache.end_times, end))
                late = [
                    _state_carried(trace=trace, cache=cache, position=p)
                    for p in range(start + 2 * 48, stop, 12)
                ]
                rows.append(
                    {
                        "donor": site,
                        "year starting": cut.year,
                        "days": float((end - onset) / np.timedelta64(1, "D")),
                        "moved (points)": 100.0 * float(event["moved_mw_p50"]) / effective,
                        "state after day 2 (points)": float(np.mean(late)),
                        "has a metered receiver": event["receivers"] != "[]",
                    }
                )
    return rows


def main() -> None:
    """Run the gate, print the report, write `work/gate.json`, and exit 1 on a blocking failure."""
    tuned = g.read_tuned()
    cache = common.load_cache()
    vintages = read_vintages()
    checks = test_one(cache=cache, vintages=vintages, tuned=tuned)
    checks += test_two(cache=cache, tuned=tuned)
    checks += test_three(cache=cache, vintages=vintages, tuned=tuned)
    farm_checks, farm_reports = test_four(cache=cache, vintages=vintages, tuned=tuned)
    checks += farm_checks
    diagnostic = real_cycle_diagnostic(cache=cache, vintages=vintages, tuned=tuned)
    real_transfers = real_transfer_report(cache=cache, vintages=vintages, tuned=tuned)
    table = pl.DataFrame(checks).with_columns(limit=pl.col("limit").cast(pl.String))
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=60):
        print(table)
        print("\n2b real tuning-year level, by site")
        print(pl.DataFrame(diagnostic))
        print("\n4 reported rows")
        print(pl.DataFrame(farm_reports))
        print("\n3 real hand-filtered transfers of three days or more at the decision sites")
        print(pl.DataFrame(real_transfers) if real_transfers else "none")
    failed = [c for c in checks if c["blocking"] and not c["passed"]]
    (common.WORK_DIR / "gate.json").write_text(
        json.dumps(
            {
                "checks": checks,
                "diagnostic": diagnostic,
                "farm": farm_reports,
                "real_transfers": real_transfers,
                "failed": len(failed),
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print(f"\n{len(checks) - len(failed)} of {len(checks)} blocking checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
