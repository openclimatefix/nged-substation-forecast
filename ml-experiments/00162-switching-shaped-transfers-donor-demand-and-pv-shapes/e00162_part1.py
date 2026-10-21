"""Part 1: fit logged real transfers, detected intervals, injected controls, and placebos.

    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python e00162_part1.py [real|controls|all]

Writes `part1/real.parquet`, `part1/detected.parquet`, and `part1/controls.parquet` to the private
folder: one row per (event, series, window) with the three models' scores and coefficients.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Final

import e00162_shape as shape
import numpy as np
import polars as pl
from e00162_common import (
    BLOCK,
    GRID_LENGTH,
    GRID_START,
    LABELLED_PATH,
    PANEL_PATH,
    SERIES_IDS,
    WORK_DIR,
    near_log_masks,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final[Path] = WORK_DIR / "part1"
TERMS: Final[Path] = WORK_DIR / "terms"
INTERVALS_147: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00147-causal-live-end-rule-for-ro-nc" / "run" / "intervals.parquet"
)
MARGIN: Final[int] = 14 * BLOCK
N_WORKERS: Final[int] = 4
N_PER_SCENARIO: Final[int] = 200
SEED: Final[int] = 162
FRACTION_DEMAND: Final[float] = 0.15
"""Share of the donor's demand term a control moves."""
FRACTION_SOLAR: Final[float] = 0.35
"""Share of the donor's solar output a control moves."""
_CACHE: dict[str, Any] = {}


def data() -> dict[str, Any]:
    """Power, the two terms, and the logged-closure masks of every series (loaded once)."""
    if not _CACHE:
        frame = pl.read_parquet(PANEL_PATH)
        power, demand, solar = {}, {}, {}
        for series in SERIES_IDS:
            group = frame.filter(pl.col("series") == series)
            p = np.full(GRID_LENGTH, np.nan)
            p[group["stamp"].to_numpy()] = group["obs_mw"].to_numpy()
            terms = np.load(TERMS / f"terms_{series}.npz")
            power[series] = p - terms["total"]
            demand[series] = terms["demand"]
            solar[series] = -terms["solar"]
        _CACHE.update(
            residual=power,
            demand=demand,
            solar=solar,
            logged=near_log_masks(pad_stamps=BLOCK),
            detected=detected_masks(),
        )
    return _CACHE


def detected_masks() -> dict[int, np.ndarray]:
    """True within a day of any interval RO-NC detected (experiment 147, setting R0), per series."""
    out = {s: np.zeros(GRID_LENGTH, dtype=bool) for s in SERIES_IDS}
    frame = pl.read_parquet(INTERVALS_147).filter(pl.col("setting") == "R0")
    for r in frame.iter_rows(named=True):
        if r["series"] in out:
            out[r["series"]][max(r["onset"] - BLOCK, 0) : min(r["end"] + BLOCK, GRID_LENGTH)] = True
    return out


def make_window(
    *,
    series: int,
    onset: int,
    end: int,
    add: tuple[float, float] = (0.0, 0.0),
    mask_detected: bool = False,
) -> shape.Window:
    """The window of one series around an interval; `add` injects (f, g) shares of the terms."""
    d = data()
    lo, hi = max(onset - MARGIN, 0), min(end + MARGIN, GRID_LENGTH)
    idx = np.arange(lo, hi)
    inside = (idx >= onset) & (idx < end)
    residual = d["residual"][series][lo:hi].copy()
    demand = d["demand"][series][lo:hi]
    solar = d["solar"][series][lo:hi]
    residual[inside] += -add[0] * demand[inside] + add[1] * solar[inside]
    residual[d["logged"][series][lo:hi] & ~inside] = np.nan
    if mask_detected:
        residual[d["detected"][series][lo:hi] & ~inside] = np.nan
    return shape.Window(
        residual=residual, demand=demand, solar_output=solar, inside=inside, day=idx // BLOCK
    )


def row_of(*, fit: shape.Fit, **keys: Any) -> dict[str, Any]:
    """Flatten a fit into a table row."""
    return {
        **keys,
        "n_inside": fit.n_inside,
        "n_days": fit.n_days,
        "constant": fit.constant,
        "alpha": fit.alpha,
        "beta": fit.beta,
        "alpha_lo": fit.alpha_ci[0],
        "alpha_hi": fit.alpha_ci[1],
        "beta_lo": fit.beta_ci[0],
        "beta_hi": fit.beta_ci[1],
        "alpha2": fit.alpha2,
        "beta2": fit.beta2,
        "alpha2_lo": fit.alpha2_ci[0],
        "alpha2_hi": fit.alpha2_ci[1],
        "beta2_lo": fit.beta2_ci[0],
        "beta2_hi": fit.beta2_ci[1],
        "demand_energy": fit.demand_energy,
        "solar_energy": fit.solar_energy,
        "corr_terms": fit.corr_terms,
        "solar_day_cv": fit.solar_day_cv,
        **{f"rss_{m}": fit.in_sample_rss[m] for m in shape.MODELS},
        **{f"held_{m}": fit.held_out_rss[m] for m in shape.MODELS},
    }


def interval_job(args: tuple[dict[str, Any], int]) -> list[dict[str, Any]]:
    """Fit every series of one interval (real, detected, or control)."""
    spec, seed = args
    rng = np.random.default_rng(seed)
    rows = []
    for series in spec["series"]:
        window = make_window(
            series=series,
            onset=spec["onset"],
            end=spec["end"],
            add=tuple(spec.get("add", (0, 0))),
            mask_detected=spec["kind"] == "control",
        )
        fit = shape.fit_window(window=window, rng=rng)
        if fit is None:
            continue
        rows.append(
            row_of(
                fit=fit,
                key=spec["key"],
                kind=spec["kind"],
                series=series,
                onset=spec["onset"],
                end=spec["end"],
                scenario=spec.get("scenario", ""),
                donor=spec.get("donor"),
                position=spec.get("position", -1),
                add_f=spec.get("add", (0, 0))[0],
                add_g=spec.get("add", (0, 0))[1],
            )
        )
    return rows


def stamp(when: Any) -> int:
    """The grid stamp of a UTC time."""
    return int((when - GRID_START).total_seconds() // 1800)


def real_specs() -> list[dict[str, Any]]:
    """The logged transfers over 1 day, with the series that have terms."""
    frame = pl.read_parquet(LABELLED_PATH).filter(
        ~pl.col("is_sentinel"),
        pl.col("kind") == "real_operation",
        pl.col("visibility").is_in(["conserved", "one_sided"]),
    )
    out = []
    for row in frame.iter_rows(named=True):
        receivers = [int(r["series"]) for r in json.loads(row["receivers"])]
        donor = int(row["donor"]) if row["donor"] not in (None, "") else None
        series = sorted(
            {int(s) for s in (donor, row["side_a"], row["side_b"], *receivers) if s is not None}
        )
        onset, end = stamp(row["onset_utc"]), stamp(row["end_utc"]) + 1
        if end - onset <= BLOCK:
            continue
        out.append(
            {
                "key": row["card_id"],
                "kind": "real",
                "series": [s for s in series if s in SERIES_IDS],
                "onset": onset,
                "end": end,
                "donor": donor,
            }
        )
    return out


def detected_specs() -> list[dict[str, Any]]:
    """RO-NC's real detected intervals of a day or more (experiment 147, setting R0)."""
    frame = pl.read_parquet(INTERVALS_147).filter(
        (pl.col("setting") == "R0") & (pl.col("end") - pl.col("onset") > BLOCK)
    )
    return [
        {
            "key": f"{r['series']}_{r['onset']}",
            "kind": "detected",
            "series": [int(r["series"])],
            "onset": int(r["onset"]),
            "end": int(r["end"]),
        }
        for r in frame.iter_rows(named=True)
    ]


def control_specs() -> list[dict[str, Any]]:
    """Injected shaped transfers and placebos in quiet stretches of the real residual."""
    rng = np.random.default_rng(SEED)
    d = data()
    scenarios = {
        "demand_only": (FRACTION_DEMAND, 0.0),
        "solar_only": (0.0, FRACTION_SOLAR),
        "both": (FRACTION_DEMAND, FRACTION_SOLAR),
        "placebo": (0.0, 0.0),
    }
    usable = [s for s in SERIES_IDS if np.isfinite(d["demand"][s]).sum() > 20_000]
    positions = []
    while len(positions) < N_PER_SCENARIO:
        series = int(rng.choice(usable))
        days = float(np.exp(rng.uniform(np.log(0.5), np.log(14.0))))
        length = max(int(days * BLOCK), 4)
        onset = int(rng.integers(MARGIN + 60 * BLOCK, GRID_LENGTH - length - MARGIN))
        span = slice(onset - 2 * BLOCK, onset + length + 2 * BLOCK)
        if d["logged"][series][span].any() or d["detected"][series][span].any():
            continue
        if np.isfinite(d["demand"][series][onset : onset + length]).mean() < 0.95:
            continue
        if np.isfinite(d["residual"][series][onset : onset + length]).mean() < 0.9:
            continue
        positions.append((series, onset, length))
    return [
        {
            "key": f"{name}_{k}",
            "kind": "control",
            "scenario": name,
            "series": [series],
            "onset": onset,
            "end": onset + length,
            "add": add,
            "position": k,
        }
        for name, add in scenarios.items()
        for k, (series, onset, length) in enumerate(positions)
    ]


def run(*, name: str, specs: list[dict[str, Any]]) -> None:
    """Fit every spec in a pool and write one table."""
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(spec, SEED + i) for i, spec in enumerate(specs)]
    with ProcessPoolExecutor(N_WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        rows = [r for part in pool.map(interval_job, jobs, chunksize=4) for r in part]
    pl.DataFrame(rows).write_parquet(OUT / f"{name}.parquet")
    print(f"{name}: {len(specs)} intervals, {len(rows)} series fits", flush=True)


def main() -> None:
    """Dispatch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["real", "controls", "all"])
    args = parser.parse_args()
    if args.stage in ("real", "all"):
        run(name="real", specs=real_specs())
        run(name="detected", specs=detected_specs())
    if args.stage in ("controls", "all"):
        run(name="controls", specs=control_specs())


if __name__ == "__main__":
    main()
