"""The known-answer gate of experiment 111, run on the tuned settings before any scored fit.

A failed blocking test stops scoring until it is fixed. All five tests read the tuning months only
(the odd months of 2024-07 to 2025-06).

1. **The calibration recovers a known bias (blocking).** A fake forecast is built from ERA5
   temperature and CAMS irradiance with a temperature bias that grows from +1.5 K at the first lead
   band to +3 K at the last, an irradiance factor of 0.8, and the real ensemble's own member
   anomalies as noise. The fitted shift and ratio must recover each bias within 15%. Then the
   deterministic fake (no noise, shrink 1) is calibrated with the fitted map and run through the
   convex model: its mean absolute error must lie within 0.05 points of the same model driven by
   ERA5 and CAMS.
2. **Ensemble irradiance has the right timing (blocking).** On the clear days of the tuning months
   (a CAMS daily peak of at least 90% of the clear-sky peak) the hour-of-day profile of the
   calibrated ensemble mean must peak within one half-hour of CAMS's at every site, and its mean
   over 11:00 to 13:00 UTC must lie within 0.85 to 1.15 of CAMS's.
3. **The filter on ensemble residuals tracks transfers (blocking).** Transfers of 15% to 35% of a
   site's own power are injected for 4 to 8 days into the proxy-driven residual stream of
   `gamk` at D10, D11, D14, and D16, only where no real switching mask lies within 48 hours and the
   transfer and the 3 days after it fall inside one tuning month. The share of the moved power that
   the filter's state carries must reach 50% by the end of day 2, lie between 0.6 and 1.3 from day
   2 to the end, and fall under 25% within 3 days of the end. The shapes follow Part 4's gate.
4. **The margin does not leak (blocking).** The mean absolute error of `gam0`'s control-member
   margin on the runs before each tuning month (the rows an XGBoost refit would train on) must lie
   within 20% of its error on the runs of the tuning month (the rows it would be used on). Reported:
   the margin of the same calendar day one year earlier against a zero margin cannot be built
   before 2025-04, so it is left to the experiment.
5. **The module's tests pass (blocking).** `test_dp_ens.py` and `test_dp_ens_weather.py`.

    uv run python ml-experiments/00111-*/e00111_gate.py
"""

import itertools
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import e00111_arms as arms
import e00111_common as common
import e00111_ens as ens
import numpy as np
import polars as pl
import torch
from studies.dp_assimilation import (
    DEFAULT_DELAY_HOURS,
    CorrectionTrace,
    FilterHyperparameters,
    run_kalman,
    usable_cutoff_positions,
)
from studies.dp_ens import member_means
from studies.dp_ens_weather import (
    N_LEAD_BANDS,
    EnsSiteRun,
    SunGeometry,
    lead_band_of_hours,
    season_of_months,
)
from studies.dp_features import FeatureCache
from studies.dp_gam import predict_gam

BIAS_LOW_K: Final[float] = 1.5
BIAS_HIGH_K: Final[float] = 3.0
IRRADIANCE_FACTOR: Final[float] = 0.8
RECOVERY_TOLERANCE: Final[float] = 0.15
NESTING_TOLERANCE_POINTS: Final[float] = 0.05
CLEAR_DAY_SHARE: Final[float] = 0.9
TRANSFER_SITES: Final[tuple[str, ...]] = ("D10", "D11", "D14", "D16")
TRANSFERS_PER_SITE: Final[int] = 4
CLEARANCE_HALF_HOURS: Final[int] = 96
STEPS_PER_DAY: Final[int] = 48
MARGIN_TOLERANCE: Final[float] = 0.20
PROXY_WARM_UP_RUNS: Final[int] = 9
"""Runs before this index lack a week of analysis proxy before their origin."""
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
MODULE_TESTS: Final[tuple[str, ...]] = (
    "packages/studies/tests/test_dp_ens.py",
    "packages/studies/tests/test_dp_ens_weather.py",
)


def lead_bias(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the fake forecast's temperature bias: 1.5 K in the first band, 3 K in the last."""
    band = lead_band_of_hours(lead_hours=lead_hours)
    return BIAS_LOW_K + (BIAS_HIGH_K - BIAS_LOW_K) * band / (N_LEAD_BANDS - 1)


def fake_run(*, cache: FeatureCache, site: str, run: EnsSiteRun, noise: bool) -> EnsSiteRun:
    """Return a run whose members are ERA5 and CAMS plus a known bias, with or without noise.

    Args:
        cache: The feature cache.
        site: The substation's label.
        run: The real run, whose member anomalies are the noise.
        noise: False for the deterministic fake.

    Returns:
        The fake run.
    """
    features = cache.sites[site]
    last = len(cache.end_times) - 1
    rows = np.minimum(
        np.arange(run.origin_position + 1, run.origin_position + 1 + run.n_targets), last
    )
    lead_hours = np.arange(1, run.n_targets + 1) / 2.0
    temperature = features.temperature_c[rows] + lead_bias(lead_hours=lead_hours)
    shortwave = features.ghi_w_m2[rows] * IRRADIANCE_FACTOR
    # The fake is the biased truth; the calibration should map it back with 1.25 and -bias.
    temperature = np.tile(temperature, (run.temperature_c.shape[0], 1))
    shortwave = np.tile(shortwave, (run.temperature_c.shape[0], 1))
    if noise:
        temperature = temperature + (run.temperature_c - run.temperature_c.mean(axis=0))
        shortwave = shortwave * (
            run.shortwave_w_m2 / np.maximum(run.shortwave_w_m2.mean(axis=0), 1.0)
        )
    return replace(run, temperature_c=temperature, shortwave_w_m2=shortwave)


def test_one(*, cache: FeatureCache) -> list[dict]:
    """Calibrate a fake forecast with a known bias and nest it against ERA5 and CAMS."""
    results = []
    shifts, ratios = [], []
    nesting = []
    for site in common.DECISION_SITES:
        setup = ens.load_setup(cache=cache, site=site)
        noisy = replace(
            setup,
            runs=[fake_run(cache=cache, site=site, run=run, noise=True) for run in setup.runs],
        )
        calibration = ens.fit_calibration(cache=cache, setup=noisy)
        daylight = slice(1, 4)
        for lead in range(N_LEAD_BANDS):
            expected = -(BIAS_LOW_K + (BIAS_HIGH_K - BIAS_LOW_K) * lead / (N_LEAD_BANDS - 1))
            fitted = float(np.median(calibration.temperature_shift_c[:, lead, :]))
            shifts.append((site, lead, fitted, expected))
        fitted_ratio = float(np.median(calibration.irradiance_ratio[:, :, daylight]))
        ratios.append((site, fitted_ratio, 1.0 / IRRADIANCE_FACTOR))
        # Nesting: the deterministic fake, calibrated, against ERA5 and CAMS.
        deterministic = replace(
            setup,
            runs=[fake_run(cache=cache, site=site, run=run, noise=False) for run in setup.runs],
        ).with_calibration(calibration=calibration)
        reference = setup
        fits = arms.load_arm_fits(arm=arms.ARMS["gam0"], site=site)
        errors = {"fake": [], "era5": []}
        for index in setup.indices(periods={"tuning"})[::6]:
            origin = setup.origins[index]
            fake = member_means(
                cache=cache,
                context=deterministic.context,
                run=deterministic.runs[index],
                origin=origin,
                fits=fits,
                members=[0],
            )
            rows = np.arange(fake.window.start, fake.window.stop)
            actual = cache.sites[site].y[rows]
            keep = np.isfinite(actual) & ~cache.sites[site].switching_mask[rows]
            era5 = predict_gam(cache=cache, fit=fake.fit, index=rows)
            errors["fake"].append(np.abs(fake.means[0][keep] - actual[keep]).mean())
            errors["era5"].append(np.abs(era5[keep] - actual[keep]).mean())
        del reference
        nesting.append((site, float(np.nanmean(errors["fake"])), float(np.nanmean(errors["era5"]))))
    worst_shift = max(abs(fitted - expected) / abs(expected) for _, _, fitted, expected in shifts)
    worst_ratio = max(abs(fitted - expected) / expected for _, fitted, expected in ratios)
    for name, value, limit, passed in (
        (
            "worst relative error of the recovered temperature shift",
            worst_shift,
            f"under {RECOVERY_TOLERANCE}",
            worst_shift < RECOVERY_TOLERANCE,
        ),
        (
            "worst relative error of the recovered irradiance ratio (expected 1.25)",
            worst_ratio,
            f"under {RECOVERY_TOLERANCE}",
            worst_ratio < RECOVERY_TOLERANCE,
        ),
    ):
        results.append(
            {
                "test": "1 calibration",
                "series": "5 decision sites",
                "check": name,
                "value": value,
                "limit": limit,
                "passed": bool(passed),
                "blocking": True,
            }
        )
    worst_nesting = max(abs(fake - era5) for _, fake, era5 in nesting)
    results.append(
        {
            "test": "1 calibration",
            "series": "5 decision sites",
            "check": "largest MAE difference, calibrated fake against ERA5 and CAMS (points)",
            "value": worst_nesting,
            "limit": f"under {NESTING_TOLERANCE_POINTS}",
            "passed": bool(worst_nesting < NESTING_TOLERANCE_POINTS),
            "blocking": True,
        }
    )
    return results


def test_two(*, cache: FeatureCache) -> list[dict]:
    """Compare the calibrated ensemble's clear-day irradiance profile with CAMS's."""
    results = []
    for site in common.ALL_SITES:
        setup = ens.load_setup(cache=cache, site=site)
        calibration = arms.load_calibration(site=site)
        features = cache.sites[site]
        ens_profile = np.zeros(48)
        cams_profile = np.zeros(48)
        counts = np.zeros(48)
        for index in setup.indices(periods={"tuning"}):
            run = setup.runs[index]
            rows = np.arange(run.origin_position + 1, run.origin_position + 1 + STEPS_PER_DAY)
            cams = features.ghi_w_m2[rows]
            table = features.table
            peak = SunGeometry.build(
                end_times=cache.end_times[rows], latitude=table.latitude, longitude=table.longitude
            ).clear_sky_w_m2.max()
            if cams.max() < CLEAR_DAY_SHARE * peak:
                continue
            mean = np.nanmean(run.shortwave_w_m2[:, :STEPS_PER_DAY], axis=0)
            band = cache.calendar.band[rows].astype(np.int64)
            month = (
                cache.calendar.local_date[rows].astype("datetime64[M]").astype(np.int64) % 12 + 1
            )
            season = season_of_months(months=month)
            lead = np.zeros(STEPS_PER_DAY, dtype=np.int64)
            calibrated = mean * calibration.irradiance_ratio[season, lead, band]
            slot = (np.arange(STEPS_PER_DAY) + run.origin_position + 1) % 48
            np.add.at(ens_profile, slot, calibrated)
            np.add.at(cams_profile, slot, cams)
            np.add.at(counts, slot, 1.0)
        if counts.min() == 0:
            results.append(
                {
                    "test": "2 irradiance timing",
                    "series": site,
                    "check": "clear days in every slot",
                    "value": float(counts.min()),
                    "limit": "at least 1",
                    "passed": False,
                    "blocking": True,
                }
            )
            continue
        ens_profile /= counts
        cams_profile /= counts
        offset = abs(int(np.argmax(ens_profile)) - int(np.argmax(cams_profile)))
        offset = min(offset, 48 - offset)
        midday = slice(22, 26)
        ratio = float(ens_profile[midday].mean() / cams_profile[midday].mean())
        results.append(
            {
                "test": "2 irradiance timing",
                "series": site,
                "check": "peak slot offset from CAMS (half-hours)",
                "value": float(offset),
                "limit": "at most 1",
                "passed": offset <= 1,
                "blocking": True,
            }
        )
        results.append(
            {
                "test": "2 irradiance timing",
                "series": site,
                "check": "midday ratio of calibrated ensemble to CAMS",
                "value": ratio,
                "limit": "0.85 to 1.15",
                "passed": 0.85 <= ratio <= 1.15,
                "blocking": True,
            }
        )
    return results


@dataclass(frozen=True)
class Transfer:
    """An injected transfer: onset and end positions on the grid and the moved power."""

    site: str
    onset: int
    end: int
    moved: np.ndarray


def tuning_blocks(*, cache: FeatureCache) -> list[tuple[int, int]]:
    """Return the grid ranges of the tuning months, as (first row, one past the last row)."""
    mask = arms.in_months(cache=cache, months=common.TUNING_MONTHS)
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)) != 0) + 1
    bounds = np.concatenate([[0], edges, [len(mask)]])
    return [(int(a), int(b)) for a, b in itertools.pairwise(bounds) if mask[a]]


def choose_transfers(*, cache: FeatureCache, site: str, seed: int) -> list[Transfer]:
    """Choose transfers inside one tuning month, away from every real switching mask."""
    features = cache.sites[site]
    generator = np.random.default_rng(seed)
    blocks = tuning_blocks(cache=cache)
    chosen: list[Transfer] = []
    attempts = 0
    while len(chosen) < TRANSFERS_PER_SITE and attempts < 800:
        attempts += 1
        first, last = blocks[int(generator.integers(0, len(blocks)))]
        length = int(generator.integers(4, 9)) * STEPS_PER_DAY
        low = first + 7 * STEPS_PER_DAY
        high = last - length - 4 * STEPS_PER_DAY
        if high <= low:
            continue
        onset = int(generator.integers(low, high))
        end = onset + length
        window = slice(onset - CLEARANCE_HALF_HOURS, end + CLEARANCE_HALF_HOURS)
        if features.switching_mask[window].any():
            continue
        if any(
            not (c.end + 7 * STEPS_PER_DAY < onset or end + 7 * STEPS_PER_DAY < c.onset)
            for c in chosen
        ):
            continue
        if not features.present[onset:end].all() or not np.isfinite(features.y[onset:end]).all():
            continue
        share = float(generator.uniform(0.15, 0.35))
        drift = np.exp(
            np.interp(np.arange(length), [0, length], [0.0, generator.uniform(-0.7, 0.7)])
        )
        moved = share * np.clip(drift, 0.5, 2.0) * features.y[onset:end]
        chosen.append(Transfer(site=site, onset=onset, end=end, moved=moved))
    return chosen


def state_carried(*, trace: CorrectionTrace, cache: FeatureCache, position: int) -> float:
    """Return the mean over bands of level plus transient after the newest usable readings."""
    cutoff = int(
        usable_cutoff_positions(
            end_times=cache.end_times,
            initialisation=cache.end_times[position : position + 1],
            delay_hours=DEFAULT_DELAY_HOURS,
        )[0]
    )
    values = []
    for band in range(len(trace.positions)):
        found = int(np.searchsorted(trace.positions[band], cutoff, side="right")) - 1
        if found >= 0:
            values.append(trace.level[band][found] + trace.anomaly[band][found])
    return float(np.mean(values)) if values else 0.0


def carried_share(
    *,
    cache: FeatureCache,
    trace: CorrectionTrace,
    base_trace: CorrectionTrace,
    transfer: Transfer,
    offset: int,
) -> float:
    """Return the share of the moved power the filter's state carries `offset` steps after onset."""
    position = transfer.onset + offset
    injected = state_carried(trace=trace, cache=cache, position=position)
    natural = state_carried(trace=base_trace, cache=cache, position=position)
    return -(injected - natural) / float(transfer.moved.mean())


def test_three(*, cache: FeatureCache) -> list[dict]:
    """Inject transfers into the proxy-driven residuals and read the share the filter carries."""
    tuned = arms.read_tuned()
    rows: list[tuple[float, float, float]] = []
    for site in TRANSFER_SITES:
        setup = ens.load_setup(cache=cache, site=site).with_calibration(
            calibration=arms.load_calibration(site=site)
        )
        arm = arms.ARMS["gamk"]
        fits = arms.load_arm_fits(arm=arm, site=site)
        base = arms.residual_stream(cache=cache, setup=setup, fits=fits, source="proxy")
        hyper = FilterHyperparameters.from_array(
            values=np.array(tuned["filter_hyper"]["gamk"][site])
        )
        base_trace = run_kalman(residual=base, band=cache.calendar.band, hyper=hyper)
        for transfer in choose_transfers(cache=cache, site=site, seed=11):
            stream = base.copy()
            stream[transfer.onset : transfer.end] -= transfer.moved
            trace = run_kalman(residual=stream, band=cache.calendar.band, hyper=hyper)
            share = {
                offset: carried_share(
                    cache=cache,
                    trace=trace,
                    base_trace=base_trace,
                    transfer=transfer,
                    offset=offset,
                )
                for offset in range(0, transfer.end - transfer.onset + 5 * STEPS_PER_DAY, 4)
            }
            length = transfer.end - transfer.onset
            by_day_two = float(
                np.mean(
                    [v for k, v in share.items() if 2 * STEPS_PER_DAY - 8 <= k <= 2 * STEPS_PER_DAY]
                )
            )
            inside = float(
                np.mean([v for k, v in share.items() if 2 * STEPS_PER_DAY <= k < length])
            )
            gone = float(
                np.mean(
                    [
                        v
                        for k, v in share.items()
                        if length + 3 * STEPS_PER_DAY <= k <= length + 3 * STEPS_PER_DAY + 8
                    ]
                )
            )
            rows.append((by_day_two, inside, gone))
    median = np.median(np.array(rows), axis=0)
    return [
        {
            "test": "3 injected transfers",
            "series": f"median of {len(rows)}",
            "check": name,
            "value": float(value),
            "limit": limit,
            "passed": bool(passed),
            "blocking": True,
        }
        for name, value, limit, passed in (
            ("share carried by the end of day 2", median[0], "at least 0.5", median[0] >= 0.5),
            (
                "share carried from day 2 to the end",
                median[1],
                "0.6 to 1.3",
                0.6 <= median[1] <= 1.3,
            ),
            ("share left 3 days after the end", median[2], "under 0.25", median[2] < 0.25),
        )
    ]


def test_four(*, cache: FeatureCache) -> list[dict]:
    """Compare the margin's error on earlier runs with its error on the tuning months' runs."""
    train_errors, forecast_errors = [], []
    for site in common.DECISION_SITES:
        setup = ens.load_setup(cache=cache, site=site)
        fits = arms.load_arm_fits(arm=arms.ARMS["gam0"], site=site)
        features = cache.sites[site]
        for index, run_date in enumerate(setup.run_dates):
            key = common.month_of_origin(run=run_date)
            period = common.period_of_run(run=run_date)
            if period not in {"warm_up", "tuning"} or key >= (2025, 6):
                continue
            if index % 3 or index < PROXY_WARM_UP_RUNS:
                continue
            result = member_means(
                cache=cache,
                context=setup.context,
                run=setup.runs[index],
                origin=setup.origins[index],
                fits=fits,
                members=[0],
            )
            rows = np.arange(result.window.start, result.window.stop)
            actual = features.y[rows]
            keep = np.isfinite(actual) & ~features.switching_mask[rows]
            if not keep.any():
                continue
            error = float(np.abs(result.means[0][keep] - actual[keep]).mean())
            (forecast_errors if period == "tuning" else train_errors).append(error)
    train, forecast = float(np.mean(train_errors)), float(np.mean(forecast_errors))
    ratio = train / forecast
    return [
        {
            "test": "4 margin leakage",
            "series": "5 decision sites",
            "check": "margin error on earlier runs over error on the tuning months' runs",
            "value": ratio,
            "limit": f"{1 - MARGIN_TOLERANCE} to {1 + MARGIN_TOLERANCE}",
            "passed": bool(1 - MARGIN_TOLERANCE <= ratio <= 1 + MARGIN_TOLERANCE),
            "blocking": True,
        }
    ]


def test_five() -> list[dict]:
    """Run the module's tests."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--run-studies",
            "-q",
            "-p",
            "no:cacheprovider",
            "-n",
            "0",
            *MODULE_TESTS,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return [
        {
            "test": "5 module tests",
            "series": "-",
            "check": "pytest exit code",
            "value": float(completed.returncode),
            "limit": "0",
            "passed": completed.returncode == 0,
            "blocking": True,
        }
    ]


def main() -> None:
    """Run the gate, print the report, write `work/gate.json`, and exit 1 on a blocking failure."""
    torch.set_num_threads(2)
    cache = common.load_cache()
    checks = []
    for name, test in (
        ("1", lambda: test_one(cache=cache)),
        ("2", lambda: test_two(cache=cache)),
        ("3", lambda: test_three(cache=cache)),
        ("4", lambda: test_four(cache=cache)),
        ("5", test_five),
    ):
        checks += test()
        print(f"test {name} done", flush=True)
    table = pl.DataFrame(checks).with_columns(limit=pl.col("limit").cast(pl.String))
    with pl.Config(tbl_rows=200, tbl_cols=20, fmt_str_lengths=70):
        print(table)
    failed = [c for c in checks if c["blocking"] and not c["passed"]]
    common.write_json_atomic(
        path=common.WORK_DIR / "gate.json", content={"checks": checks, "failed": len(failed)}
    )
    print(f"\n{len(checks) - len(failed)} of {len(checks)} readings pass")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
