"""Run the arms of experiment 141: tune, score the injected worlds, and score the real year.

Stages: `tune` picks the switched regime's prior and variance multiplier on the tuning months of
the injected worlds; `injected` scores every arm on the replicate and scored months of the injected
worlds; `real` scores the arms on the real scored year. Each (site, world) job writes its own file.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import e00141_fed as fed
import numpy as np
import polars as pl
from e00141_common import WORK_DIR

N_WORKERS: Final[int] = 6
WORLDS: Final[Path] = WORK_DIR / "worlds"
OUT: Final[Path] = WORK_DIR / "scores"
REAL_INTERVALS: Final[Path] = (
    WORK_DIR.parent / "00147-causal-live-end-rule-for-ro-nc" / "run" / "intervals.parquet"
)
P_GRID: Final[tuple[float, ...]] = (0.5, 0.9)
MULTIPLIER_GRID: Final[tuple[float, ...]] = (4.0, 16.0, 64.0, 256.0)
PLACEBO_WEEKS: Final[tuple[int, ...]] = (-6, -3, 3, 6)
"""Whole-week shifts of the causal prior; one is drawn per (world, site) from the seed."""
HERE: Final[Path] = Path(__file__).resolve().parent


def worlds_list() -> list[int]:
    """The seeds of the stored worlds."""
    return sorted(int(p.stem.split("_")[1]) for p in WORLDS.glob("world_*.parquet"))


def load_world(*, seed: int) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Return a world's intervals and the megawatts added to each series, by stamp."""
    intervals = pl.read_parquet(WORLDS / f"world_{seed}.parquet")
    with np.load(WORLDS / f"world_{seed}.npz") as stored:
        delta = {k: stored[k] for k in stored.files if k.startswith("delta_")}
    return intervals, delta


def placebo_shift(*, prior: np.ndarray, seed: int, site: str) -> np.ndarray:
    """Shift a prior by whole weeks, which keeps alarm rate, durations, and the hour of day."""
    rng = np.random.default_rng(seed + 31 * sorted(fed.SERIES_OF_SITE).index(site))
    weeks = int(rng.choice(PLACEBO_WEEKS))
    return np.roll(prior, weeks * 7 * 48)


def arm_priors(*, signals: fed.Signals, p_on: float, seed: int, site: str) -> dict[str, np.ndarray]:
    """The priors of every filter arm, with the same on-probability."""
    return {
        "SKF": signals.causal,
        "SHE": signals.hindsight,
        "SORACLE": signals.oracle,
        "PLACEBO": placebo_shift(prior=signals.causal, seed=seed, site=site),
    }


def injected_job(args: tuple[str, int, str]) -> str:
    """Score one site in one world. `mode` is `tune` or `score`."""
    site, seed, mode = args
    path = OUT / mode / f"{site}_{seed}.parquet"
    if path.exists():
        return str(path)
    data = fed.get_site(site)
    intervals, delta = load_world(seed=seed)
    stamp_delta = delta[f"delta_{fed.SERIES_OF_SITE[site]}"]
    if not np.any(stamp_delta):
        return ""
    points = fed.to_positions(
        array=100.0 * np.nan_to_num(stamp_delta) / fed.CAPACITY_MW[site], data=data
    )
    stream = np.where(np.isfinite(data.stream), data.stream + points, np.nan)
    periods = ("tuning",) if mode == "tune" else ("replicate", "scored")
    chosen = json.loads((OUT / "chosen.json").read_text()) if mode == "score" else {}
    grid = (
        [(p, m) for p in P_GRID for m in MULTIPLIER_GRID]
        if mode == "tune"
        else [(chosen["p_on"], chosen["multiplier"])]
    )
    frames = []
    for p_on, multiplier in grid:
        signals = fed.build_signals(
            data=data, site=site, intervals=intervals, delta_mw=stamp_delta, p_on=p_on
        )
        corrections = {
            "K0": fed.correction_for(
                data=data, stream=stream, prior=np.zeros(len(stream)), multiplier=1.0
            )
        }
        priors = arm_priors(signals=signals, p_on=p_on, seed=seed, site=site)
        if mode == "tune":
            priors = {"SKF": priors["SKF"]}
        for name, prior in priors.items():
            corrections[name] = fed.correction_for(
                data=data, stream=stream, prior=prior, multiplier=multiplier
            )
        if mode == "score":
            corrections["VARINF"] = fed.correction_for(
                data=data,
                stream=stream,
                prior=priors["PLACEBO"],
                multiplier=multiplier,
                update_regime=False,
            )
        for period in periods:
            mask = signals.event_window | signals.alarm_window
            rows = fed.rows_where(data=data, mask_by_position=mask, period=period)
            if not len(rows):
                continue
            scored = fed.score_arms(
                data=data, site=site, corrections=corrections, delta_points=points, rows=rows
            )
            in_alarm = signals.alarm_window[data.position[rows]]
            in_event = signals.event_window[data.position[rows]]
            tags = pl.DataFrame({"row": rows, "in_alarm48": in_alarm, "in_event": in_event})
            frames.append(
                scored.join(tags, on="row").with_columns(
                    world=pl.lit(seed), period=pl.lit(period), p_on=p_on, multiplier=multiplier
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return ""
    pl.concat(frames).write_parquet(path)
    return str(path)


def run_pool(jobs: list) -> list:
    """Run the jobs in spawned workers."""
    with ProcessPoolExecutor(N_WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(injected_job, jobs))


def tune() -> None:
    """Pick the prior and multiplier with the lowest mean quantile score on the tuning months."""
    jobs = [(s, w, "tune") for s in fed.SERIES_OF_SITE for w in worlds_list()]
    run_pool(jobs)
    frames = [pl.read_parquet(p) for p in (OUT / "tune").glob("*.parquet")]
    table = pl.concat(frames)
    k0 = (
        table.filter(pl.col("arm") == "K0")
        .group_by("site", "world", "row")
        .agg(k0=pl.col("qs").mean())
    )
    skf = table.filter(pl.col("arm") == "SKF").join(k0, on=["site", "world", "row"])
    summary = (
        skf.group_by("p_on", "multiplier")
        .agg(gain=(pl.col("qs") - pl.col("k0")).mean(), rows=pl.len())
        .sort("gain")
    )
    print(summary)
    best = summary.row(0, named=True)
    (OUT / "chosen.json").write_text(
        json.dumps({"p_on": best["p_on"], "multiplier": best["multiplier"], "gain": best["gain"]})
    )
    summary.write_csv(OUT / "tune_summary.csv")


def main() -> None:
    """Run a stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["tune", "injected", "gate"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "tune":
        tune()
    elif args.stage == "injected":
        run_pool([(s, w, "score") for s in fed.SERIES_OF_SITE for w in worlds_list()])
    else:
        print(injected_job(("D5", worlds_list()[0], "tune")))


if __name__ == "__main__":
    main()
