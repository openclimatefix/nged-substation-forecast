"""Tune the flow penalties on seeds T and the tuning placebo pairs, for experiment 116.

Each configuration is the step constant `c` (penalty `c / sigma_e`), the level constant `c_l`, and
the form (half-hourly flows, A2, or 3-hour blocks, A2-b). It is scored on the 400 transfers of set
`T` in the tuning year, by the MW-weighted recall of the fitted flow on the transfer's edge, and on
the megawatt-days of flow on the two tuning placebo pairs over the year. The choice rule is the
plan's: the largest recall with under 0.05 megawatt-days per pair-year on both placebo pairs; `rho
= 2` is kept only if it adds 0.05 recall to the chosen configuration. If no configuration meets the
placebo limit, the one with the smallest placebo flow is chosen and `decisions.json` says so.

    uv run python ml-experiments/00116-*/e00116_tune.py [--workers 6] [--choose-only]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import product
from typing import Any, Final

import e00116_common as common
import e00116_data as data_module
import e00116_inject as inject
import e00116_local as local
import numpy as np
import polars as pl

STEP_CONSTANTS: Final[tuple[float, ...]] = (3.0, 10.0, 30.0, 100.0, 300.0)
LEVEL_CONSTANTS: Final[tuple[float, ...]] = (0.0, 0.3, 1.0)
BLOCKS: Final[tuple[int, ...]] = (1, data_module.BLOCK_HALF_HOURS)
PLACEBO_LIMIT_MW_DAYS: Final[float] = 0.05
RHO_GAIN: Final[float] = 0.05
TUNE_DIR: Final = common.WORK_DIR / "tune"
DECISIONS_PATH: Final = common.WORK_DIR / "decisions.json"

_STATE: dict[str, Any] = {}


def configurations(*, outside_factor: float = 1.0) -> list[data_module.FlowConfig]:
    """Return every configuration of the grid."""
    return [
        data_module.FlowConfig(
            step_c=step, level_c=level, outside_factor=outside_factor, block=block
        )
        for block, step, level in product(BLOCKS, STEP_CONSTANTS, LEVEL_CONSTANTS)
    ]


def result_path(*, config: data_module.FlowConfig) -> Any:
    """Return the file that holds a configuration's tuning result."""
    return TUNE_DIR / f"{config.label()}.json"


def _setup() -> None:
    """Load the cache, the tuning year, its graph, and the seed-T transfers once per process."""
    if "cache" in _STATE:
        return
    cache = common.load_cache()
    data = data_module.load_year(cache=cache, cut_name="tuning", arm="a0")
    _STATE.update(
        cache=cache,
        data=data,
        edges=data_module.decision_graph(data=data),
        transfers=inject.draw_set(set_name="T", year_name="tuning", cache=cache),
    )


def evaluate(*, config: data_module.FlowConfig, limit: int | None = None) -> dict[str, Any]:
    """Score one configuration on seeds T and the tuning placebo pairs.

    Args:
        config: The configuration.
        limit: Score only the first `limit` transfers, for a smoke test.

    Returns:
        The pooled MW-weighted recall, the recall by cell, and the placebo megawatt-days.
    """
    _setup()
    cache, data, edges = _STATE["cache"], _STATE["data"], _STATE["edges"]
    paths = data_module.build_graph_paths(data=data, edges=edges, config=config)
    rows = []
    transfers = _STATE["transfers"]
    if limit is not None:
        transfers = transfers.head(limit)
    for transfer in transfers.iter_rows(named=True):
        window = local.fit_local(
            cache=cache, data=data, paths=paths, config=config, transfer=transfer
        )
        span = slice(window.onset, window.end)
        true, fitted = window.true_flow[span], window.fitted_edge()[span]
        matched = np.minimum(np.maximum(fitted * np.sign(true), 0.0), np.abs(true))
        rows.append(
            {
                "transfer_id": transfer["transfer_id"],
                "size": transfer["size"],
                "duration": transfer["duration"],
                "true_mw_half_hours": float(np.sum(np.abs(true))),
                "matched_mw_half_hours": float(np.sum(matched)),
            }
        )
    frame = pl.DataFrame(rows)
    total = frame.select(pl.col("true_mw_half_hours").sum()).item()
    recall = frame.select(pl.col("matched_mw_half_hours").sum()).item() / total
    placebo = {
        "-".join(pair): local.placebo_mw_days(data=data, paths=paths, config=config, pair=pair)
        for pair in common.PLACEBO_PAIRS["tuning"]
    }
    by_cell = (
        frame.group_by("size", "duration")
        .agg(
            (pl.col("matched_mw_half_hours").sum() / pl.col("true_mw_half_hours").sum()).alias(
                "recall"
            )
        )
        .sort("size", "duration")
    )
    return {
        "config": asdict(config),
        "n_transfers": frame.height,
        "recall": recall,
        "placebo_mw_days": placebo,
        "recall_by_cell": by_cell.to_dicts(),
    }


def _worker(config: data_module.FlowConfig) -> str:
    """Evaluate one configuration in a worker process and write its result."""
    import torch  # ty: ignore[unresolved-import]

    torch.set_num_threads(1)
    target = result_path(config=config)
    if target.exists():
        return f"skip {config.label()}"
    result = evaluate(config=config)
    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(target)
    placebo = max(result["placebo_mw_days"].values())
    return f"done {config.label()}: recall {result['recall']:.3f}, worst placebo {placebo:.3f}"


def meets_limit(*, result: dict[str, Any]) -> bool:
    """Return whether both placebo pairs are under the limit."""
    return max(result["placebo_mw_days"].values()) < PLACEBO_LIMIT_MW_DAYS


def choose(*, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick a form's configuration by the placebo rule.

    Args:
        results: The tuning results of one form's configurations.

    Returns:
        The chosen result, with a `met_limit` flag: the largest recall among those under the
        placebo limit, or else the smallest placebo flow.
    """
    allowed = [r for r in results if meets_limit(result=r)]
    if allowed:
        return {**max(allowed, key=lambda r: r["recall"]), "met_limit": True}
    smallest = min(results, key=lambda r: (max(r["placebo_mw_days"].values()), -r["recall"]))
    return {**smallest, "met_limit": False}


def read_results() -> list[dict[str, Any]]:
    """Read every tuning result written so far."""
    return [json.loads(path.read_text()) for path in sorted(TUNE_DIR.glob("*.json"))]


def decide() -> dict[str, Any]:
    """Choose the form and configuration, and write `decisions.json`.

    A2-b replaces A2 only if its chosen configuration has the larger recall under the placebo
    limit (or both miss the limit and its placebo flow is smaller).
    """
    results = [r for r in read_results() if r["config"]["outside_factor"] == 1.0]
    by_form = {
        block: choose(results=[r for r in results if r["config"]["block"] == block])
        for block in BLOCKS
    }
    half_hourly, blocked = by_form[1], by_form[data_module.BLOCK_HALF_HOURS]
    if half_hourly["met_limit"] != blocked["met_limit"]:
        winner = half_hourly if half_hourly["met_limit"] else blocked
    elif half_hourly["met_limit"]:
        winner = blocked if blocked["recall"] > half_hourly["recall"] else half_hourly
    else:
        worst = {"half_hourly": half_hourly, "blocked": blocked}
        winner = min(worst.values(), key=lambda r: max(r["placebo_mw_days"].values()))
    decisions = {"A2": half_hourly, "A2-b": blocked, "chosen": winner}
    DECISIONS_PATH.write_text(json.dumps(decisions, indent=2))
    return decisions


def main() -> None:
    """Evaluate every configuration in parallel, then choose."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--choose-only", action="store_true")
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    if not arguments.choose_only:
        common.load_cache()
        with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
            for message in pool.map(_worker, configurations()):
                print(message, flush=True)
    print(json.dumps(decide()["chosen"]["config"], indent=2))


if __name__ == "__main__":
    main()
