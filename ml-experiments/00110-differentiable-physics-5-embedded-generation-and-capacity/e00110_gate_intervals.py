"""The calibration gate of the capacity intervals: do 80% intervals cover the truth 70 to 90%?

The design session's ruling after experiment 112: where a gate uses a point tolerance, it also
reports whether the interval a card or a label will print is calibrated. For each capacity arm
(K-base, K-i, K-ii, K-iii) the script makes synthetic substations whose truth is the arm's own
fit to a real decision substation at the tuning cut, adds noise, refits the arm, and asks whether
the 80% month-block bootstrap interval of the last season-year's capacity covers the truth.

- **Twenty synthetic substations per arm and noise:** the five decision substations times four
  noise draws.
- **Two kinds of noise:** independent normal noise of 5.0 points (the standard deviation of
  experiment 112's gate), and noise made of the real residuals: each calendar month's residuals
  are replaced by those of a randomly chosen month from the same quarter of the year, so the
  synthetic noise keeps the real residuals' autocorrelation and their size.
- **Two intervals, both from 20 stratified month-block weightings** (each calendar quarter's months
  kept), widened about the median by sqrt(m / (m - 1)) for m months in the window (experiment
  113's advice): the 80% interval (10th to 90th percentile), which the design session's ruling
  names, and the 90% range (2nd to 19th of the 20 ordered values), which the substation cards and
  the `identified` rule print. The gate tests both.
- **Pass:** coverage of the 80% interval between 70 and 90%, and coverage of the 90% range between
  80 and 100%, over the 20 substations. Each is reported with its binomial 95% interval, the mean
  width, and the mean bias. With 20 substations the gate has little power: an interval that truly
  covers 80% passes about 84% of the time, one that covers 60% or 95% still passes about a quarter
  of the time, and all eight cells pass together only about a quarter of the time even for a
  calibrated interval. It is weak evidence, read beside the point tolerances of the main gate,
  which stay as the pre-registered record.
- **Months are drawn independently,** as the bootstrap draws them, so the gate cannot catch
  dependence between months.

The truth is the arm's own model, so the test measures the interval's calibration and not the
bias of a missing term; the main gate's mismatched plants and farms measure bias.

    uv run python ml-experiments/00110-*/e00110_gate_intervals.py [--workers 4]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any, Final

import e00110_arms as arms
import e00110_common as common
import numpy as np
from scipy.stats import beta
from studies.dp_features import FeatureCache
from studies.dp_gam import fit_gam, predict_gam, prepare_fit
from studies.dp_pinned_capacity import read_vintages

INTERVAL_ARMS: Final[tuple[str, ...]] = ("k_base", "k_i", "k_ii", "k_iii")
N_REPLICATES: Final[int] = 4
IID_NOISE_POINTS: Final[float] = 5.0
COVERAGE_BAND_80: Final[tuple[float, float]] = (0.70, 0.90)
"""The ruling's band for an 80% interval."""
COVERAGE_BAND_90: Final[tuple[float, float]] = (0.80, 1.00)
"""The band for the 90% range, by the same plus or minus 10 points."""
INTERVAL_QUANTILES: Final[tuple[float, float]] = (0.10, 0.90)
NOISE_KINDS: Final[tuple[str, ...]] = ("iid", "real")

_STATE: dict[str, Any] = {}


def _init_worker() -> None:
    """Load the cache, vintages, and arm definitions once per worker process, with two threads."""
    import torch

    torch.set_num_threads(2)
    _STATE["cache"] = common.load_cache()
    _STATE["vintages"] = read_vintages()
    _STATE["definitions"] = arms.arm_definitions(
        part_4=arms.read_part_4_tuned(), tuned=arms.read_tuned()
    )


def real_residual_noise(
    *, cache: FeatureCache, index: np.ndarray, residual: np.ndarray, generator: np.random.Generator
) -> np.ndarray:
    """Return noise whose calendar months are the real residuals of random months of the quarter.

    A row takes the donor month's residual at the same offset from the start of the month, so the
    half-hour of day and the day of the month line up. A half-hour the donor month lacks takes a
    residual of the donor at the same half-hour of another day.

    Args:
        cache: The cache, for the rows' times.
        index: The training rows.
        residual: The real residual at each row of `index`.
        generator: Draws the donor months.

    Returns:
        A noise array of the same length as `index`.
    """
    ends = cache.end_times[index]
    month = ends.astype("datetime64[M]")
    months, inverse = np.unique(month, return_inverse=True)
    quarter = (months.astype(np.int64) % 12) // 3
    offset = np.rint((ends - month.astype("datetime64[us]")) / np.timedelta64(30, "m")).astype(
        np.int64
    )
    noise = np.empty(len(index))
    in_quarter = quarter[inverse]

    def quarter_pool(*, quarter: int, half_hour: int) -> list[float]:
        """Residuals of every row of the quarter at one half-hour of day; all of them if none."""
        chosen = (in_quarter == quarter) & (offset % 48 == half_hour)
        return residual[chosen].tolist() or residual[in_quarter == quarter].tolist()

    for position in range(len(months)):
        donors = np.flatnonzero(quarter == quarter[position])
        donor = int(generator.choice(donors))
        source = inverse == donor
        by_offset = dict(zip(offset[source].tolist(), residual[source].tolist(), strict=True))
        by_half_hour: dict[int, list[float]] = {}
        for key, value in by_offset.items():
            by_half_hour.setdefault(key % 48, []).append(value)
        target = np.flatnonzero(inverse == position)
        for row in target:
            key = int(offset[row])
            if key in by_offset:
                noise[row] = by_offset[key]
            else:
                pool = by_half_hour.get(key % 48) or quarter_pool(
                    quarter=int(quarter[position]), half_hour=key % 48
                )
                noise[row] = pool[int(generator.integers(len(pool)))]
    return noise


def _job(job: tuple[str, str, int, str]) -> dict[str, Any]:
    """Make one synthetic substation, refit the arm, and score the interval's coverage."""
    arm, site, replicate, noise_kind = job
    cache: FeatureCache = _STATE["cache"]
    definition: arms.ArmDefinition = _STATE["definitions"][arm]
    cut = common.TUNING_CUT
    truth_fit = fit_gam(
        cache=cache,
        site=site,
        specs=[definition.spec],
        cut=cut,
        window=definition.window,
        season_years=common.SEASON_YEARS,
    )
    _, index = prepare_fit(
        cache=cache,
        site=site,
        spec=definition.spec,
        cut=cut,
        window=definition.window,
        season_years=common.SEASON_YEARS,
    )
    truth = truth_fit.capacity_points()[truth_fit.last_season_year]
    predicted = predict_gam(cache=cache, fit=truth_fit, index=index)
    features = cache.sites[site]
    generator = np.random.default_rng(
        [common.BOOTSTRAP_SEED, replicate, common.DECISION_SITES.index(site)]
    )
    if noise_kind == "iid":
        noise = IID_NOISE_POINTS * generator.normal(size=len(index))
    else:
        noise = real_residual_noise(
            cache=cache, index=index, residual=features.y[index] - predicted, generator=generator
        )
    y = features.y.copy()
    y[index] = predicted + noise
    synthetic = replace(cache, sites={**cache.sites, site: replace(features, y=y)})
    result = arms.capacity_with_bootstrap(
        cache=synthetic,
        vintages=_STATE["vintages"],
        site=site,
        cut=cut,
        definition=definition,
    )
    ordered = np.sort(result.bootstrap)
    median = float(np.median(ordered))
    widen = float(np.sqrt(result.n_months / max(result.n_months - 1, 1)))
    low, high = (
        median + widen * (float(np.quantile(ordered, q)) - median) for q in INTERVAL_QUANTILES
    )
    low_90, high_90 = result.range_90
    return {
        "arm": arm,
        "site": site,
        "replicate": replicate,
        "noise": noise_kind,
        "truth": truth,
        "estimate": result.capacity,
        "low": low,
        "high": high,
        "covered": bool(low <= truth <= high),
        "low_90": low_90,
        "high_90": high_90,
        "covered_90": bool(low_90 <= truth <= high_90),
    }


def clopper_pearson(*, covered: int, total: int, level: float = 0.95) -> tuple[float, float]:
    """Return the exact binomial interval of a coverage count."""
    tail = (1.0 - level) / 2.0
    low = 0.0 if covered == 0 else float(beta.ppf(tail, covered, total - covered + 1))
    high = 1.0 if covered == total else float(beta.ppf(1.0 - tail, covered + 1, total - covered))
    return low, high


def summarise(*, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return coverage, width, and bias per arm and noise, with the verdicts and binomial ranges."""
    summary = []
    for arm in INTERVAL_ARMS:
        for kind in NOISE_KINDS:
            part = [r for r in rows if r["arm"] == arm and r["noise"] == kind]
            if not part:
                continue
            total = len(part)
            line: dict[str, Any] = {
                "arm": arm,
                "noise": kind,
                "n_substations": total,
                "mean_width_80_points": float(np.mean([r["high"] - r["low"] for r in part])),
                "mean_width_90_points": float(np.mean([r["high_90"] - r["low_90"] for r in part])),
                "mean_bias_points": float(np.mean([r["estimate"] - r["truth"] for r in part])),
            }
            for label, key, band in (
                ("80", "covered", COVERAGE_BAND_80),
                ("90", "covered_90", COVERAGE_BAND_90),
            ):
                covered = sum(bool(r[key]) for r in part)
                low, high = clopper_pearson(covered=covered, total=total)
                line[f"covered_{label}"] = covered
                line[f"coverage_{label}"] = covered / total
                line[f"coverage_{label}_binomial_95"] = [low, high]
                line[f"calibrated_{label}"] = band[0] <= covered / total <= band[1]
            summary.append(line)
    return summary


def main() -> None:
    """Run the calibration gate and write `work/gate_intervals.json`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "2"
    common.load_cache()
    jobs = [
        (arm, site, replicate, noise)
        for arm in INTERVAL_ARMS
        for noise in NOISE_KINDS
        for site in common.DECISION_SITES
        for replicate in range(N_REPLICATES)
    ]
    with ProcessPoolExecutor(max_workers=arguments.workers, initializer=_init_worker) as pool:
        rows = list(pool.map(_job, jobs))
    summary = summarise(rows=rows)
    (common.WORK_DIR / "gate_intervals.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n"
    )
    for line in summary:
        print(
            f"{line['arm']:>8} {line['noise']:>5}: 80% covers {line['covered_80']}/"
            f"{line['n_substations']} {line['coverage_80_binomial_95']}, 90% range covers "
            f"{line['covered_90']}/{line['n_substations']} {line['coverage_90_binomial_95']}, "
            f"width {line['mean_width_80_points']:.1f}/{line['mean_width_90_points']:.1f}, "
            f"bias {line['mean_bias_points']:+.2f}, calibrated "
            f"{line['calibrated_80']}/{line['calibrated_90']}"
        )


if __name__ == "__main__":
    main()
