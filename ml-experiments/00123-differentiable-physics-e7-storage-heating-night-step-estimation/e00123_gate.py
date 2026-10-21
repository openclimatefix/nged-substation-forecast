"""Test 2 of experiment 123: the known-answer gate for estimator D.

Noise: the real design-winter nights (2023-24 and 2024-25) of each series, with D's fitted
cold-dependent part removed from the window rows, resampled in 14-day blocks within winter with the
real covariates. A known step of `S` points at the reference cold night (and the placebo `S = 0`)
is injected in five shapes, `S` in {4, 7}. The used estimator, with its 1,000-draw block bootstrap,
is fitted to every draw, and the gate reads the coverage of the true step by the 90% interval.
The gate can fail by design and is never redefined after it has run.
"""

import json
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from typing import Final

import e00123_common as common
import e00123_estimate as estimate
import e00123_pool as pool
import numpy as np
import polars as pl

N_DRAWS: Final[int] = 200
LOAD_PAUSE: Final[float] = 35.0
WORKERS: Final[int] = 6
RAMP_MEAN_WEIGHT: Final[float] = (0.0 + 0.5 + 1.0 + 1.0 + 1.0) / 5.0
CELLS: Final[tuple[tuple[str, float], ...]] = (
    ("matched", 4.0),
    ("matched", 7.0),
    ("ramp_plateau", 4.0),
    ("ramp_plateau", 7.0),
    ("flat", 4.0),
    ("flat", 7.0),
    ("night_temperature", 4.0),
    ("night_temperature", 7.0),
    ("placebo", 0.0),
)
"""The nine cells: a shape and the step `S` (points at a daily mean of 3 C)."""
POOLED_STEPS: Final[tuple[float, ...]] = (0.0, 0.0, 4.0, 4.0, 7.0, 7.0)
SERIES_BAND: Final[tuple[float, float]] = (0.80, 0.98)
POOLED_BAND: Final[tuple[float, float]] = (0.85, 0.95)
BIAS_BOUND: Final[float] = 0.5
GATED_SHAPES: Final[tuple[str, ...]] = ("matched", "ramp_plateau", "flat", "placebo")
SUMMER_YEARS_FOR_GATE: Final[tuple[int, ...]] = (2023, 2024)


def truth(*, shape: str, step: float) -> float:
    """Return the true step at the reference cold night of an injected shape."""
    return step * RAMP_MEAN_WEIGHT if shape == "ramp_plateau" else step


def injection(*, shape: str, step: float, nights: estimate.Nights) -> np.ndarray:
    """Return the points added to each winter night's `d`."""
    scale = step / common.REFERENCE_COLD_C
    if shape == "matched":
        return scale * nights.cold
    if shape == "ramp_plateau":
        return RAMP_MEAN_WEIGHT * scale * nights.cold
    if shape == "flat":
        return np.full(len(nights.cold), step)
    if shape == "night_temperature":
        return scale * nights.cold_night
    return np.zeros(len(nights.cold))


def residual_nights(*, nights: estimate.Nights) -> estimate.Nights:
    """Return the nights with `d` replaced by D's residual and the summer `d` centred on 0."""
    x = estimate.design_matrix(nights=nights)
    coefficients, *_ = np.linalg.lstsq(x, nights.d, rcond=None)
    return nights.with_arrays(
        d=nights.d - x @ coefficients, summer_d=nights.summer_d - nights.summer_d.mean()
    )


def resample_blocks(
    *, strata: np.ndarray, block: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw whole blocks within each stratum.

    Args:
        strata: The stratum of each night.
        block: The block index of each night within its stratum.
        rng: The random generator.

    Returns:
        The drawn rows' indices, and a new block index per drawn row so that two copies of one
        block are two blocks in the inner bootstrap.
    """
    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for stratum in np.unique(strata):
        members = np.flatnonzero(strata == stratum)
        n_blocks = int(block[members].max()) + 1
        for new_label, picked in enumerate(rng.integers(0, n_blocks, size=n_blocks)):
            chosen = members[block[members] == picked]
            rows.append(chosen)
            labels.append(np.full(len(chosen), new_label))
    return np.concatenate(rows), np.concatenate(labels)


def simulated_dataset(*, base: estimate.Nights, rng: np.random.Generator) -> estimate.Nights:
    """Draw one resampled noise dataset (before any injection)."""
    rows, blocks = resample_blocks(strata=base.winter, block=base.block, rng=rng)
    summer_rows, summer_blocks = resample_blocks(
        strata=base.summer_year, block=base.summer_block, rng=rng
    )
    return estimate.Nights(
        site=base.site,
        winter=base.winter[rows],
        cold=base.cold[rows],
        cold_night=base.cold_night[rows],
        saturday=base.saturday[rows],
        sunday=base.sunday[rows],
        d=base.d[rows],
        evening=base.evening[rows],
        block=blocks,
        summer_d=base.summer_d[summer_rows],
        summer_block=summer_blocks,
        summer_year=base.summer_year[summer_rows],
    )


@dataclass(frozen=True)
class SeriesJob:
    """What one worker needs.

    Attributes:
        site: The label.
        capacity_mw: The effective capacity.
        seed: The seed of the worker's random generator.
    """

    site: str
    capacity_mw: float
    seed: int


def pause_while_busy() -> None:
    """Wait while the 1-minute load average is above the pause threshold."""
    while os.getloadavg()[0] > LOAD_PAUSE:
        print("load average above 35: pausing 60 s", flush=True)
        time.sleep(60)


def run_series(job: SeriesJob) -> pl.DataFrame:
    """Run every draw and cell of one series; return one row per draw and cell."""
    common.cap_threads()
    table = pl.read_parquet(common.WORK_DIR / "night_tables.parquet")
    base = residual_nights(
        nights=estimate.load_nights(
            table=table,
            site=job.site,
            capacity_mw=job.capacity_mw,
            winters=common.DESIGN_WINTERS,
            summer_years=SUMMER_YEARS_FOR_GATE,
        )
    )
    rng = np.random.default_rng(job.seed)
    rows: list[dict[str, object]] = []
    for draw in range(N_DRAWS):
        if draw % 20 == 0:
            pause_while_busy()
        noise = simulated_dataset(base=base, rng=rng)
        for shape, step in CELLS:
            sim = noise.with_arrays(d=noise.d + injection(shape=shape, step=step, nights=noise))
            fit = estimate.fit_series(nights=sim, rng=rng)
            rows.append(
                {
                    "site": job.site,
                    "draw": draw,
                    "shape": shape,
                    "step_true": truth(shape=shape, step=step),
                    "S": step,
                    "step": fit.step.value,
                    "low": fit.step.low,
                    "high": fit.step.high,
                    "sd": fit.step.sd,
                    "slope": fit.slope.value,
                    "slope_low": fit.slope.low,
                    "slope_high": fit.slope.high,
                    "flat": fit.flat.value,
                }
            )
    return pl.DataFrame(rows)


def pooled_cell(*, draws: pl.DataFrame, seed: int) -> dict[str, float]:
    """Return the Hartung-Knapp coverage of the pooled mean with `S` drawn per series.

    Each draw assigns a permutation of {0, 0, 4, 4, 7, 7} to the six decision series, reads each
    series' matched-shape fit at its `S`, and pools them.
    """
    rng = np.random.default_rng(seed)
    matched = draws.filter(pl.col("shape").is_in(["matched", "placebo"]))
    covered_hk = 0
    covered_boot = 0
    used = 0
    for draw in range(N_DRAWS):
        steps = rng.permutation(POOLED_STEPS)
        values = []
        errors = []
        for site, step in zip(common.DECISION_SERIES, steps, strict=True):
            shape = "placebo" if step == 0.0 else "matched"
            row = matched.filter(
                (pl.col("site") == site)
                & (pl.col("draw") == draw)
                & (pl.col("shape") == shape)
                & (pl.col("S") == step)
            ).row(0, named=True)
            values.append(row["step"])
            errors.append(row["sd"])
        pooled = pool.pool(values=np.array(values), standard_errors=np.array(errors), rng=rng)
        true_mean = float(np.mean(steps))
        covered_hk += int(pooled.low_hk <= true_mean <= pooled.high_hk)
        covered_boot += int(pooled.low_boot <= true_mean <= pooled.high_boot)
        used += 1
    return {"draws": float(used), "hk": covered_hk / used, "boot": covered_boot / used}


def summarise(*, draws: pl.DataFrame) -> pl.DataFrame:
    """Return per series and cell: coverage, mean bias, mean width, and power."""
    return (
        draws.with_columns(
            covered=(pl.col("low") <= pl.col("step_true"))
            & (pl.col("step_true") <= pl.col("high")),
            above_zero=pl.col("low") > 0.0,
        )
        .group_by("site", "shape", "S")
        .agg(
            coverage=pl.col("covered").mean(),
            bias=(pl.col("step") - pl.col("step_true")).mean(),
            width=(pl.col("high") - pl.col("low")).mean(),
            power=pl.col("above_zero").mean(),
        )
        .sort("shape", "S", "site")
    )


def main() -> None:
    """Run the gate over the 14 fitted series and write `gate.json` and `gate_draws.parquet`."""
    common.cap_threads()
    pause_while_busy()
    infos = common.series_infos()
    jobs = [
        SeriesJob(site=site, capacity_mw=infos[site].effective_capacity_mw, seed=1000 + index)
        for index, site in enumerate(common.FITTED_SERIES)
    ]
    with multiprocessing.get_context("spawn").Pool(WORKERS) as workers:
        frames = workers.map(run_series, jobs, chunksize=1)
    draws = pl.concat(frames)
    draws.write_parquet(common.WORK_DIR / "gate_draws.parquet")
    summary = summarise(draws=draws)
    summary.write_parquet(common.WORK_DIR / "gate_summary.parquet")
    pooled = pooled_cell(draws=draws, seed=777)
    by_shape = (
        summary.group_by("shape", "S")
        .agg(
            pooled_coverage=pl.col("coverage").mean(),
            lowest_series=pl.col("coverage").min(),
            highest_series=pl.col("coverage").max(),
            worst_bias=pl.col("bias").abs().max(),
        )
        .sort("shape", "S")
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3):
        print(by_shape)
        print(summary.filter(pl.col("shape") == "placebo"))
    print("pooled interval coverage (Hartung-Knapp, series bootstrap):", pooled)
    (common.WORK_DIR / "gate.json").write_text(
        json.dumps({"pooled_interval": pooled, "by_shape": by_shape.to_dicts()}, indent=2)
    )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
