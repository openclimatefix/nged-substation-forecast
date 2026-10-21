"""Controls of experiment 133: P1 reproduction, P2 selection on known answers, tracking gate.

**P1** compares the FIX arm's scored-year median forecast with experiment 111's stored GAM-K rows.

**P2** simulates streams on each decision substation's real grid with the filter's own settings
(FIX's values) as the truth, in two forms: `own` (a random-walk level, an AR(1) transient per hour
band, and white noise, plus a level shift) and `wobble` (the same plus a seasonal sine of amplitude
1 point). It chooses a cell by the step-1 day-ahead error on the tuning-month cases and compares
the chosen cell's error on the replicate-month cases with the true setting's. The score of the
selection, not the recovery of any parameter.

**Tracking gate** adds a PV-shaped level shift (peak half of the substation's residual standard
deviation, ramping up over 180 days from 2024-07-01) to real tuning-year residuals (circular month
blocks from months with the same clock signature) and to AR(1) noise at FIX's settings, each
paired with the same noise unshifted. A draw passes when the difference between the filtered
midday level of the shifted and unshifted streams, averaged over days 151 to 180, reaches 50% of
the injected midday shift averaged over the same days.

    uv run python ml-experiments/00133-*/e00133_gate.py p1
    uv run python ml-experiments/00133-*/e00133_gate.py p2
    uv run python ml-experiments/00133-*/e00133_gate.py track [--workers 6]
"""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from typing import Final

import e00133_arms as arms
import e00133_common as common
import e00133_filters as filters
import e00133_run as run
import e00133_select as select
import numpy as np
import polars as pl
from studies.dp_filter_grid import correction_at_cutoffs, run_kalman_grid

GATE_DIR: Final = common.WORK_DIR / "gate"
STREAM_START: Final = datetime(2024, 7, 1, tzinfo=UTC)
STREAM_END: Final = datetime(2025, 7, 1, tzinfo=UTC)
RAMP_DAYS: Final[float] = 180.0
PEAK_SD_SHARE: Final[float] = 0.5
READ_DAYS: Final[tuple[int, int]] = (151, 180)
PASS_SHARE: Final[float] = 0.5
MIDDAY_BAND: Final[int] = 2
DRAWS: Final[int] = 12
SELECTION_DRAWS: Final[int] = 2
GATE_ARMS: Final[tuple[str, ...]] = ("fix", "grid", "pool", "vk0")
WINTER_MONTHS: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
SUMMER_MONTHS: Final[frozenset[int]] = frozenset({5, 6, 7, 8, 9})


def seed_of(*parts: object) -> int:
    """Return a stable seed from the parts, independent of the scenario."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:8], 16)


def p1() -> dict:
    """Compare FIX's scored-year median with experiment 111's stored GAM-K rows."""
    worst = {}
    keys = ["power_fcst_init_time", "valid_time"]
    for site in common.SCORED_SITES:
        mine = pl.read_parquet(run.out_path(arm="fix", period="scored", site=site))
        theirs = pl.read_parquet(
            common.SOURCE_WORK_DIR / "dp_preds" / f"gamk_scored_{site}.parquet"
        )
        joined = mine.select([*keys, "q50"]).join(
            theirs.select([*keys, pl.col("q50").alias("q50_111")]), on=keys, how="inner"
        )
        worst[site] = {
            "rows_mine": mine.height,
            "rows_111": theirs.height,
            "rows_joined": joined.height,
            "max_abs_difference": float((joined["q50"] - joined["q50_111"]).abs().to_numpy().max()),
        }
    passed = all(v["max_abs_difference"] < 1e-4 and v["rows_joined"] > 0 for v in worst.values())
    record = {"passed": passed, "sites": worst}
    common.write_json_atomic(path=GATE_DIR / "p1.json", content=record)
    return record


def ar1_noise(
    *, context: arms.SiteContext, hyper: np.ndarray, seed: int, observed: np.ndarray
) -> np.ndarray:
    """Return AR(1) noise per hour band plus white noise, on the `observed` rows.

    Args:
        context: The substation's stored inputs.
        hyper: The four unconstrained values that set the noise.
        seed: The random seed.
        observed: A boolean mask of the grid rows that carry a reading.

    Returns:
        A stream on the grid, `NaN` off the mask.
    """
    settings = arms.settings_of_hyper(hyper=hyper)
    rng = np.random.default_rng(seed)
    out = np.full(len(observed), np.nan)
    band = context.cache.calendar.band
    for index in range(4):
        rows = np.flatnonzero(observed & (band == index))
        state = rng.normal(0.0, settings.sigma_z / np.sqrt(1.0 - settings.phi**2))
        previous = rows[0] if len(rows) else 0
        for row in rows:
            gap = int(row - previous)
            previous = row
            decay = settings.phi**gap
            innovation = settings.sigma_z * np.sqrt(1.0 - decay**2) / np.sqrt(1.0 - settings.phi**2)
            state = decay * state + rng.normal(0.0, innovation)
            out[row] = state + rng.normal(0.0, settings.sigma_eps)
    return out


def block_noise(*, context: arms.SiteContext, rows: np.ndarray, seed: int) -> np.ndarray:
    """Return the real stream with each month replaced by a month of the same clock signature.

    Args:
        context: The substation's stored inputs.
        rows: The grid rows of the stream year.
        seed: The random seed.

    Returns:
        A stream on the whole grid, `NaN` outside `rows`.
    """
    rng = np.random.default_rng(seed)
    cache = context.cache
    starts = cache.end_times[rows] - np.timedelta64(30, "m")
    month = starts.astype("datetime64[M]")
    names = np.unique(month)
    calendar_month = names.astype(np.int64) % 12 + 1
    out = np.full(len(cache.end_times), np.nan)
    for slot, number in zip(names, calendar_month, strict=True):
        pool = (
            WINTER_MONTHS
            if number in WINTER_MONTHS
            else SUMMER_MONTHS
            if number in SUMMER_MONTHS
            else {number}
        )
        sources = [m for m, n in zip(names, calendar_month, strict=True) if n in pool]
        source = sources[int(rng.integers(len(sources)))]
        target_rows = rows[month == slot]
        source_rows = rows[month == source]
        length = min(len(target_rows), len(source_rows))
        out[target_rows[:length]] = context.stream[source_rows[:length]]
    return out


def stream_rows(*, context: arms.SiteContext) -> np.ndarray:
    """Return the grid rows of the stream year, 2024-07-01 to 2025-06-30."""
    ends = context.cache.end_times
    start = np.datetime64(STREAM_START.replace(tzinfo=None), "us")
    stop = np.datetime64(STREAM_END.replace(tzinfo=None), "us")
    return np.flatnonzero((ends > start) & (ends <= stop))


def injected_shift(*, context: arms.SiteContext, rows: np.ndarray, peak: float) -> np.ndarray:
    """Return the PV-shaped level shift on the whole grid: ramp times the unit PV shape."""
    cache = context.cache
    features = cache.sites[context.site]
    shape = np.nan_to_num(features.pv_unit_shape)
    shape = shape / max(float(shape[rows].max()), 1e-9)
    days = (
        cache.end_times - np.datetime64(STREAM_START.replace(tzinfo=None), "us")
    ) / np.timedelta64(1, "D")
    ramp = np.clip(days / RAMP_DAYS, 0.0, 1.0)
    out = np.zeros(len(cache.end_times))
    out[rows] = peak * ramp[rows] * shape[rows]
    return out


def read_midday(*, context: arms.SiteContext, rows: np.ndarray, series: np.ndarray) -> float:
    """Return the mean of `series` over the midday half-hours of days 151 to 180."""
    cache = context.cache
    days = (
        cache.end_times - np.datetime64(STREAM_START.replace(tzinfo=None), "us")
    ) / np.timedelta64(1, "D")
    selected = rows[
        (cache.calendar.band[rows] == MIDDAY_BAND)
        & (days[rows] > READ_DAYS[0])
        & (days[rows] <= READ_DAYS[1])
    ]
    return float(np.nanmean(series[selected]))


def level_on_grid(
    *, context: arms.SiteContext, stream: np.ndarray, settings: filters.FilterSettings
) -> np.ndarray:
    """Return the filter's level after each midday reading, on the whole grid."""
    result = filters.filter_stream(
        residual=stream, band=context.cache.calendar.band, settings=settings
    )
    out = np.full(len(stream), np.nan)
    positions = result.trace.positions[MIDDAY_BAND]
    out[positions] = result.trace.level[MIDDAY_BAND]
    return out


def track_job(job: tuple[str, str, int, float]) -> list[dict]:
    """Run one noise draw of the tracking gate at one substation for every gate arm."""
    site, form, draw, share = job
    context = arms.load_site(site=site)
    rows = stream_rows(context=context)
    sd = float(np.nanstd(context.stream[rows]))
    peak = share * sd
    shift = injected_shift(context=context, rows=rows, peak=peak)
    seed = seed_of("e133", site, form, draw)
    if form == "real":
        noise = block_noise(context=context, rows=rows, seed=seed)
    else:
        observed = np.isfinite(context.stream) & np.isin(np.arange(len(context.stream)), rows)
        noise = ar1_noise(
            context=context, hyper=arms.fix_hyper(site=site), seed=seed, observed=observed
        )
    truth = read_midday(context=context, rows=rows, series=shift)
    out = []
    for arm in GATE_ARMS:
        settings = run.arm_settings(arm=arm, site=site)
        shifted = level_on_grid(context=context, stream=noise + shift, settings=settings)
        plain = level_on_grid(context=context, stream=noise, settings=settings)
        gain = read_midday(context=context, rows=rows, series=shifted) - read_midday(
            context=context, rows=rows, series=plain
        )
        out.append(
            {
                "site": site,
                "form": form,
                "draw": draw,
                "arm": arm,
                "truth": truth,
                "gain": gain,
                "passed": bool(gain >= PASS_SHARE * truth),
            }
        )
    return out


def track(*, workers: int, share: float = PEAK_SD_SHARE) -> None:
    """Run the tracking gate and write the per-draw results and the pass rates."""
    jobs = [
        (s, f, d, share)
        for s in common.DECISION_SITES
        for f in ("real", "ar1")
        for d in range(DRAWS)
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records = [r for part in pool.map(track_job, jobs) for r in part]
    rates = {}
    for form in ("real", "ar1"):
        for arm in GATE_ARMS:
            rows = [r for r in records if r["form"] == form and r["arm"] == arm]
            rates[f"{form}_{arm}"] = float(np.mean([r["passed"] for r in rows]))
    common.write_json_atomic(
        path=GATE_DIR / ("track.json" if share == PEAK_SD_SHARE else f"track_{share}.json"),
        content={"peak_share_of_sd": share, "rates": rates, "draws": records},
    )
    print(json.dumps(rates, indent=2), flush=True)


def simulate(*, context: arms.SiteContext, form: str, seed: int) -> np.ndarray:
    """Return a simulated stream on the substation's real grid with FIX's values as the truth."""
    settings = arms.settings_of_hyper(hyper=arms.fix_hyper(site=context.site))
    observed = np.isfinite(context.stream)
    noise = ar1_noise(
        context=context, hyper=arms.fix_hyper(site=context.site), seed=seed, observed=observed
    )
    rng = np.random.default_rng(seed + 1)
    steps = rng.normal(0.0, np.sqrt(settings.q / 48.0), size=len(noise))
    level = np.cumsum(steps)
    stream = noise + level
    stream = np.where(observed, stream, np.nan)
    if form == "wobble":
        doy = context.cache.calendar.day_of_year.astype(np.float64)
        stream = stream + np.sin(2.0 * np.pi * doy / 365.0)
    return stream


def p2_job(job: tuple[str, str, int]) -> dict:
    """Choose a cell on a simulated stream's tuning cases and score it on the replicate cases."""
    site, form, draw = job
    context = arms.load_site(site=site)
    stream = simulate(context=context, form=form, seed=seed_of("e133p2", site, form, draw))
    grid = np.vstack([select.cells(), arms.fix_hyper(site=site)[None, :]])
    trace = run_kalman_grid(
        residuals=stream[None, :],
        band=context.cache.calendar.band,
        hyper=grid,
        n_bands=4,
        keep_states=True,
    )
    errors = {}
    for name, periods in (("tuning", {"tuning"}), ("replicate", {"replicate"})):
        cases = select.day_ahead_cases(context=context, periods=frozenset(periods), stream=stream)
        correction = correction_at_cutoffs(
            trace=trace, cutoff=cases["cutoff"], target=cases["target"], band=cases["band"]
        )[0]
        errors[name] = np.abs(cases["actual"][None, :] - correction.cpu().numpy()).mean(axis=1)
    chosen = int(np.argmin(errors["tuning"][:-1]))
    return {
        "site": site,
        "form": form,
        "draw": draw,
        "chosen_replicate_error": float(errors["replicate"][chosen]),
        "true_replicate_error": float(errors["replicate"][-1]),
        "ratio": float(errors["replicate"][chosen] / errors["replicate"][-1]),
    }


def p2() -> dict:
    """Run P2 on the GPU, one simulated stream at a time."""
    records = [
        p2_job((site, form, draw))
        for site in common.DECISION_SITES
        for form in ("own", "wobble")
        for draw in range(SELECTION_DRAWS)
    ]
    summary = {
        form: float(np.mean([r["ratio"] for r in records if r["form"] == form]))
        for form in ("own", "wobble")
    }
    record = {
        "mean_ratio": summary,
        "passed": all(v <= 1.02 for v in summary.values()),
        "draws": records,
    }
    common.write_json_atomic(path=GATE_DIR / "p2.json", content=record)
    print(json.dumps({k: v for k, v in record.items() if k != "draws"}, indent=2), flush=True)
    return record


def main() -> None:
    """Run one control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("p1", "p2", "track"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--share", type=float, default=PEAK_SD_SHARE)
    args = parser.parse_args()
    if args.stage == "p1":
        print(json.dumps(p1(), indent=2))
    elif args.stage == "p2":
        p2()
    else:
        track(workers=args.workers, share=args.share)


if __name__ == "__main__":
    main()
