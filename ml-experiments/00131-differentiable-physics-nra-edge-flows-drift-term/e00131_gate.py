"""The known-answer gate of experiment 131: four readings on injected transfers and placebo pairs.

The gate reads the big transfers of set `G` with the settings `e00131_tune.py` chose (frozen), in
the tuning year and again in the replicate year, for the setting with a drift term and for the
benchmark without one. Set `L` (7 to 30 days) is fitted too and is descriptive only.

1. **Moved-MW ratio:** the median over big transfers of the fitted mean flow over the true mean
   flow lies between 0.8 and 1.25.
2. **Recall:** pooled megawatt-weighted recall at least 0.77 for transfers of 24 to 72 hours and at
   least 0.5 for transfers of 3 to 24 hours.
3. **Placebo:** under 0.05 MW-days a year on each placebo pair of the year, with the placebo path
   listed first and last.
4. **Release:** among big transfers whose hindsight recall is at least 0.5, at least half have a
   causal flow under 20% of the moved MW at the larger of 6 hours and 10% of the duration after the
   end, counting the telemetry delay.

    uv run python ml-experiments/00131-*/e00131_gate.py [--workers 4]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Final

import e00131_common as common
import e00131_data as data_module
import e00131_inject as inject
import e00131_local as local
import e00131_threshold as threshold
import numpy as np
import polars as pl
from studies.dp_assimilation import DEFAULT_DELAY_HOURS
from studies.dp_edge_flow import (
    FLOW_AS_MOVED_FRACTION,
    causal_window_fit,
    edge_name,
    moved_mw_ratio,
    mw_weighted_recall,
)

GATE_PATH: Final = common.WORK_DIR / "gate.json"
ROWS_PATH: Final = common.WORK_DIR / "gate_rows.parquet"
DECISIONS_PATH: Final = common.WORK_DIR / "decisions.json"
RATIO_RANGE: Final[tuple[float, float]] = (0.8, 1.25)
RECALL_LONG: Final[float] = 0.77
RECALL_SHORT: Final[float] = 0.5
RELEASE_SHARE: Final[float] = 0.5
DETECTED_RECALL: Final[float] = 0.5
PLACEBO_LIMIT: Final[float] = 0.05
MIN_FOUND_FOR_RELEASE: Final[int] = 30
"""The release reading needs this many found transfers; fewer counts as a fail."""
RELEASE_FLOOR_HALF_HOURS: Final[int] = 12
"""The release allowance is at least 6 hours."""
DELAY_HALF_HOURS: Final[int] = round(DEFAULT_DELAY_HOURS * 2)
WINDOW_HALF_HOURS: Final[int] = 30 * 48
"""The causal window: the trailing 30 days."""
YEARS: Final[tuple[str, ...]] = ("tuning", "replicate")
GROUPS: Final[tuple[str, ...]] = ("drift", "none")
_STATE: dict[str, Any] = {}


def _setup(*, year_name: str) -> None:
    """Load the cache, a year's data, and its thresholds once per process."""
    if year_name in _STATE:
        return
    cache = common.load_cache()
    data = data_module.load_year(cache=cache, cut_name=year_name, arm="a0")
    _STATE[year_name] = (
        cache,
        data,
        data_module.decision_graph(data=data),
        threshold.thresholds(cache=cache, data=data),
    )


def chosen_config(*, group: str) -> data_module.FlowConfig:
    """Return the frozen setting of a group."""
    return data_module.FlowConfig(**json.loads(DECISIONS_PATH.read_text())[group]["config"])


def release_test(
    *,
    cache: Any,
    config: data_module.FlowConfig,
    data: data_module.YearData,
    paths: Any,
    transfer: dict,
) -> bool:
    """Return whether the causal flow is under 20% of the moved MW at the release deadline."""
    duration = int(transfer["end"] - transfer["onset"])
    allowance = max(RELEASE_FLOOR_HALF_HOURS, round(0.1 * duration))
    cutoff = int(transfer["end"]) + allowance - DELAY_HALF_HOURS
    streams, added = local.injected_streams(cache=cache, data=data, transfer=transfer)
    nodes = (transfer["node_a"], transfer["node_b"])
    local_paths = local.neighbourhood(paths=paths, nodes=nodes)
    touched = {path.node_a for path in local_paths} | {
        path.node_b for path in local_paths if path.node_b
    }
    fit, _ = causal_window_fit(
        paths=local_paths,
        residuals={node: streams[node] for node in touched},
        noise=data.noise_mw(block=config.block),
        cutoff=cutoff,
        window_length=WINDOW_HALF_HOURS,
        max_sweeps=data_module.MAX_SWEEPS,
        sweep_tolerance=data_module.SWEEP_TOLERANCE,
        huber_sigmas=config.huber_sigmas,
        drift_step_variance=local.drift_for(data=data, paths=local_paths, config=config),
    )
    moved = float(np.mean(np.abs(added[transfer["receiver"]][transfer["onset"] : transfer["end"]])))
    name = edge_name(node_a=transfer["node_a"], node_b=transfer["node_b"])
    return bool(abs(fit.flows[name][-1]) < FLOW_AS_MOVED_FRACTION * moved)


def transfer_job(job: tuple[dict, str, str]) -> dict[str, Any]:
    """Fit one transfer's neighbourhood and return its hindsight reading and release test."""
    import torch

    torch.set_num_threads(1)
    transfer, year_name, group = job
    _setup(year_name=year_name)
    cache, data, edges, _ = _STATE[year_name]
    config = chosen_config(group=group)
    paths = data_module.build_graph_paths(data=data, edges=edges, config=config)
    window = local.fit_local(cache=cache, data=data, paths=paths, config=config, transfer=transfer)
    span = slice(window.onset, window.end)
    true, fitted = window.true_flow[span], window.fitted_edge()[span]
    total = float(np.sum(np.abs(true)))
    recall = mw_weighted_recall(fitted=fitted, true=true)
    row = {
        "transfer_id": transfer["transfer_id"],
        "set": transfer["set"],
        "year": year_name,
        "group": group,
        "size": transfer["size"],
        "duration": transfer["duration"],
        "hours": (transfer["end"] - transfer["onset"]) / 2.0,
        "ratio": moved_mw_ratio(fitted=fitted, true=true),
        "recall": recall,
        "matched": float(recall * total),
        "total": total,
        "released": None,
    }
    if transfer["set"] == "G" and recall >= DETECTED_RECALL:
        row["released"] = release_test(
            cache=cache, config=config, data=data, paths=paths, transfer=transfer
        )
    return row


def placebo_job(job: tuple[str, str, tuple[str, str], bool]) -> dict[str, Any]:
    """Return the placebo flow of one pair, group, year, and path order over the whole year."""
    import torch

    torch.set_num_threads(1)
    year_name, group, pair, first = job
    _setup(year_name=year_name)
    _, data, edges, _ = _STATE[year_name]
    config = chosen_config(group=group)
    paths = data_module.build_graph_paths(data=data, edges=edges, config=config)
    value = local.placebo_mw_days(
        data=data, paths=paths, config=config, pair=pair, placebo_first=first
    )
    return {
        "year": year_name,
        "group": group,
        "pair": "-".join(pair),
        "order": "first" if first else "last",
        "mw_days": value,
    }


def recall_over(*, frame: pl.DataFrame) -> float:
    """Return the megawatt-weighted recall pooled over transfers."""
    return float(np.sum(frame["matched"].to_numpy()) / np.sum(frame["total"].to_numpy()))


def read_group(*, rows: pl.DataFrame, placebo: pl.DataFrame, year: str, group: str) -> dict:
    """Return the four readings of one group in one year."""
    sub = rows.filter(pl.col("year") == year, pl.col("group") == group, pl.col("set") == "G")
    ratio = float(np.median(sub["ratio"].to_numpy()))
    found = sub.filter(pl.col("released").is_not_null())
    release = float(np.mean(found["released"].to_numpy())) if found.height else float("nan")
    recall_long = recall_over(frame=sub.filter(pl.col("duration") == inject.DURATION_LONG_BAND))
    recall_short = recall_over(frame=sub.filter(pl.col("duration") == inject.DURATION_SHORT_BAND))
    flows = placebo.filter(pl.col("year") == year, pl.col("group") == group)
    worst = float(np.max(flows["mw_days"].to_numpy()))
    return {
        "n_big_transfers": sub.height,
        "1_moved_mw_ratio": {"value": ratio, "pass": RATIO_RANGE[0] <= ratio <= RATIO_RANGE[1]},
        "2a_recall_24_to_72_h": {"value": recall_long, "pass": recall_long >= RECALL_LONG},
        "2b_recall_3_to_24_h": {"value": recall_short, "pass": recall_short >= RECALL_SHORT},
        "3_placebo_mw_days": {
            "value": flows.select("pair", "order", "mw_days").to_dicts(),
            "worst": worst,
            "pass": worst < PLACEBO_LIMIT,
        },
        "4_release_share": {
            "value": release,
            "n_found": found.height,
            "pass": found.height >= MIN_FOUND_FOR_RELEASE and release >= RELEASE_SHARE,
        },
    }


def passes(*, readings: dict) -> bool:
    """Return whether every reading of one group-year passed."""
    return all(v["pass"] for k, v in readings.items() if isinstance(v, dict))


def main() -> None:
    """Fit every transfer and placebo tile, then write `gate.json` and the per-transfer rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    cache = common.load_cache()
    jobs: list[tuple[dict, str, str]] = []
    placebo_jobs: list[tuple[str, str, tuple[str, str], bool]] = []
    limit = threshold.thresholds(
        cache=cache, data=data_module.load_year(cache=cache, cut_name="tuning", arm="a0")
    )
    for year in YEARS:
        data = data_module.load_year(cache=cache, cut_name=year, arm="a0")
        for set_name in ("G", "L"):
            transfers = inject.draw_set(set_name=set_name, year_name=year, cache=cache)
            for row in transfers.iter_rows(named=True):
                if threshold.is_big(transfer=row, cache=cache, data=data, limit=limit):
                    jobs += [(row, year, group) for group in GROUPS]
        placebo_jobs += [
            (year, group, pair, first)
            for group in GROUPS
            for pair in common.PLACEBO_PAIRS[year]
            for first in (True, False)
        ]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        rows = pl.DataFrame(list(pool.map(transfer_job, jobs, chunksize=4)), strict=False)
        placebo = pl.DataFrame(list(pool.map(placebo_job, placebo_jobs)))
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    rows.write_parquet(ROWS_PATH)
    placebo.write_parquet(common.WORK_DIR / "gate_placebo.parquet")
    readings = {
        group: {
            year: read_group(rows=rows, placebo=placebo, year=year, group=group) for year in YEARS
        }
        for group in GROUPS
    }
    summary = {
        group: all(passes(readings=readings[group][year]) for year in YEARS) for group in GROUPS
    }
    long_arm = {
        f"{group}:{year}": recall_over(
            frame=rows.filter(
                pl.col("set") == "L", pl.col("year") == year, pl.col("group") == group
            )
        )
        for group in GROUPS
        for year in YEARS
    }
    payload = {"readings": readings, "gate_passes": summary, "recall_7_to_30_days": long_arm}
    GATE_PATH.write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps({"gate_passes": summary, "recall_7_to_30_days": long_arm}, indent=2))


if __name__ == "__main__":
    main()
