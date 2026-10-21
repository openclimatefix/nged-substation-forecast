"""Tune the flow penalties and the drift variance on the tuning year, for experiment 131.

Stage 1 scores 60 settings (step constant x level constant x drift variance, with "no drift term"
as the benchmark) on the big transfers of set `T` and on 4 evenly spaced 72-day placebo tiles.
Stage 2 re-reads the 3 best settings of each group (with a drift term, and without) on every
placebo tile of the year, and compares the Huber threshold of the winner at 1.0 and 3.0. The rule
is in the plan: among settings under 0.05 MW-days a year on both placebo pairs in both path orders,
the largest margin, the smaller of (recall for 3 to 24 hours / 0.5) and (recall for 24 to 72 hours
/ 0.77).

    uv run python ml-experiments/00131-*/e00131_tune.py screen [--workers 4]
    uv run python ml-experiments/00131-*/e00131_tune.py finalists [--workers 4]
    uv run python ml-experiments/00131-*/e00131_tune.py huber [--workers 4]
    uv run python ml-experiments/00131-*/e00131_tune.py decide
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from typing import Any, Final

import e00131_common as common
import e00131_data as data_module
import e00131_inject as inject
import e00131_local as local
import e00131_threshold as threshold
import numpy as np
import polars as pl

STEP_CONSTANTS: Final[tuple[float, ...]] = (3.0, 10.0, 30.0, 100.0, 300.0)
LEVEL_CONSTANTS: Final[tuple[float, ...]] = (0.3, 1.0)
DRIFT_Q: Final[tuple[float | None, ...]] = (None, 0.0001, 0.0004, 0.0016, 0.0064, 0.0256)
HUBER_CHECKS: Final[tuple[float, ...]] = (1.0, 3.0)
PLACEBO_LIMIT_MW_DAYS: Final[float] = 0.05
RECALL_SHORT_TARGET: Final[float] = 0.5
RECALL_LONG_TARGET: Final[float] = 0.77
FINALISTS_PER_GROUP: Final[int] = 3
SHORT_BAND: Final[str] = "3-24h"
LONG_BAND: Final[str] = "1-3 days"
TUNE_DIR: Final[Path] = common.WORK_DIR / "tune"
FINAL_DIR: Final[Path] = common.WORK_DIR / "tune_full"
HUBER_DIR: Final[Path] = common.WORK_DIR / "tune_huber"
DECISIONS_PATH: Final[Path] = common.WORK_DIR / "decisions.json"

_STATE: dict[str, Any] = {}


def configurations() -> list[data_module.FlowConfig]:
    """Return every stage-1 configuration."""
    return [
        data_module.FlowConfig(step_c=step, level_c=level, drift_q=q)
        for step, level, q in product(STEP_CONSTANTS, LEVEL_CONSTANTS, DRIFT_Q)
    ]


def _setup() -> None:
    """Load the cache, the tuning year, its graph, and the big `T` transfers once per process."""
    if "cache" in _STATE:
        return
    cache = common.load_cache()
    data = data_module.load_year(cache=cache, cut_name="tuning", arm="a0")
    limit = threshold.thresholds(cache=cache, data=data)
    transfers = inject.draw_set(set_name="T", year_name="tuning", cache=cache)
    big = [
        row
        for row in transfers.iter_rows(named=True)
        if threshold.is_big(transfer=row, cache=cache, data=data, limit=limit)
    ]
    _STATE.update(
        cache=cache,
        data=data,
        edges=data_module.decision_graph(data=data),
        transfers=big,
    )


def group_of(*, config: dict[str, Any]) -> str:
    """Return `drift` for a setting with a drift term and `none` for one without."""
    return "none" if config["drift_q"] is None else "drift"


def recall_by_band(*, config: data_module.FlowConfig) -> dict[str, Any]:
    """Return the pooled megawatt-weighted recall of the big set-`T` transfers by duration band."""
    cache, data, edges = _STATE["cache"], _STATE["data"], _STATE["edges"]
    paths = data_module.build_graph_paths(data=data, edges=edges, config=config)
    rows = []
    for transfer in _STATE["transfers"]:
        window = local.fit_local(
            cache=cache, data=data, paths=paths, config=config, transfer=transfer
        )
        span = slice(window.onset, window.end)
        true, fitted = window.true_flow[span], window.fitted_edge()[span]
        matched = np.minimum(np.maximum(fitted * np.sign(true), 0.0), np.abs(true))
        rows.append(
            {
                "duration": transfer["duration"],
                "total": float(np.sum(np.abs(true))),
                "matched": float(np.sum(matched)),
            }
        )
    frame = pl.DataFrame(rows)
    by_band = {}
    for band in (SHORT_BAND, LONG_BAND):
        rows_in_band = frame.filter(pl.col("duration") == band)
        by_band[band] = float(
            np.sum(rows_in_band["matched"].to_numpy()) / np.sum(rows_in_band["total"].to_numpy())
        )
    return {"recall": by_band, "n_big": frame.height}


def placebo_readings(
    *, config: data_module.FlowConfig, tiles: list[tuple[int, int]] | None
) -> dict[str, float]:
    """Return the placebo flow of each pair with the placebo path first and last."""
    data, edges = _STATE["data"], _STATE["edges"]
    paths = data_module.build_graph_paths(data=data, edges=edges, config=config)
    readings = {}
    for pair in common.PLACEBO_PAIRS["tuning"]:
        for first in (True, False):
            key = f"{'-'.join(pair)}:{'first' if first else 'last'}"
            readings[key] = local.placebo_mw_days(
                data=data, paths=paths, config=config, pair=pair, placebo_first=first, tiles=tiles
            )
    return readings


def margin(*, recall: dict[str, float]) -> float:
    """Return the smaller of the two recall-over-target ratios."""
    return min(recall[SHORT_BAND] / RECALL_SHORT_TARGET, recall[LONG_BAND] / RECALL_LONG_TARGET)


def _write(*, target: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file atomically."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(target)


def _screen_worker(config: data_module.FlowConfig) -> str:
    """Score one configuration at stage 1 and write its result."""
    import torch

    torch.set_num_threads(1)
    target = TUNE_DIR / f"{config.label()}.json"
    if target.exists():
        return f"skip {config.label()}"
    _setup()
    result = {
        "config": asdict(config),
        **recall_by_band(config=config),
        "placebo": placebo_readings(config=config, tiles=local.screen_tiles(data=_STATE["data"])),
    }
    _write(target=target, payload=result)
    worst_placebo = max(result["placebo"].values())
    return f"done {config.label()}: {result['recall']}, worst placebo {worst_placebo:.3f}"


def _full_worker(config: data_module.FlowConfig) -> str:
    """Re-read one finalist's placebo flow on every tile of the year."""
    import torch

    torch.set_num_threads(1)
    target = FINAL_DIR / f"{config.label()}.json"
    if target.exists():
        return f"skip {config.label()}"
    _setup()
    _write(target=target, payload={"placebo_full": placebo_readings(config=config, tiles=None)})
    return f"done {config.label()}"


def _huber_worker(config: data_module.FlowConfig) -> str:
    """Score a Huber variant of the winner on transfers and the whole year of placebo tiles."""
    import torch

    torch.set_num_threads(1)
    target = HUBER_DIR / f"{config.label()}.json"
    if target.exists():
        return f"skip {config.label()}"
    _setup()
    result = {
        "config": asdict(config),
        **recall_by_band(config=config),
        "placebo_full": placebo_readings(config=config, tiles=None),
    }
    _write(target=target, payload=result)
    return f"done {config.label()}: {result['recall']}"


def read_dir(*, folder: Path) -> list[dict[str, Any]]:
    """Read every result file in a folder."""
    return [json.loads(path.read_text()) for path in sorted(folder.glob("*.json"))]


def label_of(*, config: dict[str, Any]) -> str:
    """Return the file label of a stored configuration."""
    return data_module.FlowConfig(**config).label()


def worst(*, readings: dict[str, float]) -> float:
    """Return the largest placebo flow over the pairs and both path orders."""
    return max(readings.values())


def finalists() -> dict[str, list[data_module.FlowConfig]]:
    """Return each group's best stage-1 settings that pass the placebo screen."""
    results = read_dir(folder=TUNE_DIR)
    chosen: dict[str, list[data_module.FlowConfig]] = {}
    for group in ("drift", "none"):
        pool = [r for r in results if group_of(config=r["config"]) == group]
        passing = [r for r in pool if worst(readings=r["placebo"]) < PLACEBO_LIMIT_MW_DAYS]
        ranked = sorted(
            passing,
            key=lambda r: (margin(recall=r["recall"]), r["config"]["step_c"]),
            reverse=True,
        )
        if not ranked:
            ranked = sorted(pool, key=lambda r: worst(readings=r["placebo"]))
        chosen[group] = [
            data_module.FlowConfig(**r["config"]) for r in ranked[:FINALISTS_PER_GROUP]
        ]
    return chosen


def winner_of(*, group: str) -> dict[str, Any]:
    """Return a group's chosen setting among its finalists, with its readings."""
    screened = {label_of(config=r["config"]): r for r in read_dir(folder=TUNE_DIR)}
    full = {path.stem: json.loads(path.read_text()) for path in FINAL_DIR.glob("*.json")}
    candidates = []
    for config in finalists()[group]:
        label = config.label()
        if label not in full:
            continue
        candidates.append({**screened[label], **full[label]})
    passing = [c for c in candidates if worst(readings=c["placebo_full"]) < PLACEBO_LIMIT_MW_DAYS]
    if passing:
        return {
            **max(passing, key=lambda c: (margin(recall=c["recall"]), c["config"]["step_c"])),
            "met_limit": True,
        }
    return {**min(candidates, key=lambda c: worst(readings=c["placebo_full"])), "met_limit": False}


def huber_pool(*, base: dict[str, Any]) -> dict[str, Any]:
    """Return the best of the base setting and its Huber variants by the same rule."""
    variants = read_dir(folder=HUBER_DIR)
    pool = [base] + [
        v for v in variants if group_of(config=v["config"]) == group_of(config=base["config"])
    ]
    passing = [c for c in pool if worst(readings=c["placebo_full"]) < PLACEBO_LIMIT_MW_DAYS]
    if passing:
        return {
            **max(passing, key=lambda c: (margin(recall=c["recall"]), c["config"]["step_c"])),
            "met_limit": True,
        }
    return {**min(pool, key=lambda c: worst(readings=c["placebo_full"])), "met_limit": False}


def decide() -> dict[str, Any]:
    """Choose the drift setting and the no-drift benchmark and write `decisions.json`."""
    decisions = {}
    for group in ("drift", "none"):
        base = winner_of(group=group)
        decisions[group] = huber_pool(base=base)
    DECISIONS_PATH.write_text(json.dumps(decisions, indent=2))
    return decisions


def _run(*, worker: Any, configs: list[data_module.FlowConfig], workers: int) -> None:
    """Run a worker over configurations in parallel and print each message."""
    common.load_cache()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for message in pool.map(worker, configs):
            print(message, flush=True)


def main() -> None:
    """Run one stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("screen", "finalists", "huber", "decide"))
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    if arguments.stage == "screen":
        _run(worker=_screen_worker, configs=configurations(), workers=arguments.workers)
    elif arguments.stage == "finalists":
        flat = [c for group in finalists().values() for c in group]
        _run(worker=_full_worker, configs=flat, workers=arguments.workers)
    elif arguments.stage == "huber":
        base = {group: winner_of(group=group) for group in ("drift", "none")}
        variants = [
            replace(data_module.FlowConfig(**b["config"]), huber_sigmas=h)
            for b in base.values()
            for h in HUBER_CHECKS
        ]
        _run(worker=_huber_worker, configs=variants, workers=arguments.workers)
    else:
        print(json.dumps({k: v["config"] for k, v in decide().items()}, indent=2))


if __name__ == "__main__":
    main()
