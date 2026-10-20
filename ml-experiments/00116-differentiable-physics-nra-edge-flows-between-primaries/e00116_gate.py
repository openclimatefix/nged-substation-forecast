"""The known-answer gate of experiment 116: five readings on injected transfers.

The gate reads set `G` in the tuning year with the configuration `e00116_tune.py` chose, and a
failed reading stops C2 and C3 and caps the verdict at `inconclusive`. Readings 1 to 3 use the
transfers of 25% and 40% of the receiving load.

1. **Moved-MW ratio:** the median over transfers of the fitted mean flow over the true mean flow,
   between 0.8 and 1.25.
2. **Release:** the causal flow is under 20% of the moved MW at the larger of 6 hours and 10% of
   the duration after the end, counting the 3.5-hour telemetry delay, in at least half the
   transfers.
3. **Recall:** MW-weighted recall at least 0.77 over 1 day and 0.5 for 6 to 24 hours, and at least
   0.5 over 1 day on the realistic set (set `C1`, fitted on the replicate year's residuals).
4. **Placebo, out of sample:** the held-out pairs of the replicate year carry under 0.05
   megawatt-days of flow each.
5. **NRA gap:** on in-event and 48-hour post-event rows, the forecast of the normal running
   arrangement from the causal flows closes at least 50% of the gap between arm A0 and the
   oracle, in mean absolute error against the uninjected power. It reads the files that
   `e00116_run.py transfers --set G --year tuning` writes.

    uv run python ml-experiments/00116-*/e00116_gate.py [--workers 6]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Final

import e00116_causal as causal
import e00116_common as common
import e00116_data as data_module
import e00116_flows as flows_module
import e00116_inject as inject
import e00116_local as local
import e00116_run as run
import e00116_score as score
import numpy as np
import polars as pl
from studies.dp_edge_flow import (
    FLOW_AS_MOVED_FRACTION,
    causal_window_fit,
    edge_name,
    moved_mw_ratio,
    mw_weighted_recall,
)

GATE_PATH: Final = common.WORK_DIR / "gate.json"
BIG: Final[tuple[str, ...]] = ("25%", "40%")
RATIO_RANGE: Final[tuple[float, float]] = (0.8, 1.25)
RECALL_OVER_DAY: Final[float] = 0.77
RECALL_6_TO_24H: Final[float] = 0.5
RECALL_REALISTIC: Final[float] = 0.5
RELEASE_SHARE: Final[float] = 0.5
PLACEBO_LIMIT: Final[float] = 0.05
GAP_SHARE: Final[float] = 0.5
RELEASE_FLOOR_HALF_HOURS: Final[int] = 12
"""The release allowance is at least 6 hours."""
_STATE: dict[str, Any] = {}


def _setup(*, cut_name: str) -> None:
    """Load the cache, the year's data, and its graph once per process."""
    key = f"setup:{cut_name}"
    if key in _STATE:
        return
    cache = common.load_cache()
    config = flows_module.chosen_config()
    data = data_module.load_year(cache=cache, cut_name=cut_name, arm="a0")
    edges = data_module.decision_graph(data=data)
    _STATE[key] = (
        cache,
        config,
        data,
        data_module.build_graph_paths(data=data, edges=edges, config=config),
    )


def local_readings(job: tuple[dict, str, bool]) -> dict[str, Any]:
    """Fit one transfer's neighbourhood and return its hindsight reading and its release test."""
    import torch  # ty: ignore[unresolved-import]

    torch.set_num_threads(1)
    transfer, cut_name, with_release = job
    _setup(cut_name=cut_name)
    cache, config, data, paths = _STATE[f"setup:{cut_name}"]
    window = local.fit_local(cache=cache, data=data, paths=paths, config=config, transfer=transfer)
    span = slice(window.onset, window.end)
    true, fitted = window.true_flow[span], window.fitted_edge()[span]
    total = float(np.sum(np.abs(true)))
    row = {
        "transfer_id": transfer["transfer_id"],
        "size": transfer["size"],
        "duration": transfer["duration"],
        "hours": (transfer["end"] - transfer["onset"]) / 2.0,
        "ratio": moved_mw_ratio(fitted=fitted, true=true),
        "recall": mw_weighted_recall(fitted=fitted, true=true),
        "matched": float(mw_weighted_recall(fitted=fitted, true=true) * total),
        "total": total,
        "released": None,
    }
    if with_release:
        row["released"] = release_test(
            cache=cache, config=config, data=data, paths=paths, transfer=transfer
        )
    return row


def release_test(*, cache: Any, config: Any, data: Any, paths: Any, transfer: dict) -> bool:
    """Return whether the causal flow is under 20% of the moved MW at the release deadline."""
    duration = int(transfer["end"] - transfer["onset"])
    allowance = max(RELEASE_FLOOR_HALF_HOURS, round(0.1 * duration))
    cutoff = int(transfer["end"]) + allowance - causal.DELAY_HALF_HOURS
    streams, added = local.injected_streams(cache=cache, data=data, transfer=transfer)
    nodes = (transfer["node_a"], transfer["node_b"])
    local_paths = local.neighbourhood(paths=paths, nodes=nodes)
    fit, _ = causal_window_fit(
        paths=local_paths,
        residuals={node: streams[node] for node in data.sites},
        noise=data.noise_mw(block=config.block),
        cutoff=cutoff,
        window_length=causal.WINDOW_HALF_HOURS,
        max_sweeps=data_module.MAX_SWEEPS,
        sweep_tolerance=data_module.SWEEP_TOLERANCE,
    )
    moved = float(np.mean(np.abs(added[transfer["receiver"]][transfer["onset"] : transfer["end"]])))
    name = edge_name(node_a=transfer["node_a"], node_b=transfer["node_b"])
    return bool(abs(fit.flows[name][-1]) < FLOW_AS_MOVED_FRACTION * moved)


def recall_over(*, frame: pl.DataFrame) -> float:
    """Return the MW-weighted recall pooled over transfers."""
    return float(frame["matched"].sum() / frame["total"].sum())


def nra_gap() -> dict[str, float]:
    """Return the share of the A0-to-oracle gap that A5's normal-running forecast closes."""
    folder = run.transfers_dir(set_name="G", year_name="tuning")
    rows = pl.concat([pl.read_parquet(p) for p in sorted(folder.glob("*.parquet"))])
    rows = rows.filter(pl.col("target") >= 0).with_columns(
        error=(pl.col("clean") - pl.col("gam") - pl.col("level") - pl.col("anomaly")).abs()
    )
    mean_error = {
        arm: float(rows.filter(pl.col("arm") == arm)["error"].mean()) for arm in ("a0", "a5", "or")
    }
    gap = mean_error["a0"] - mean_error["or"]
    return {
        **mean_error,
        "closed": (mean_error["a0"] - mean_error["a5"]) / gap if gap > 0 else float("nan"),
    }


def main() -> None:
    """Read the five readings and write `gate.json`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    cache = common.load_cache()
    gate = inject.draw_set(set_name="G", year_name="tuning", cache=cache)
    realistic = inject.draw_set(set_name="C1", year_name="replicate", cache=cache)
    jobs = [(row, "tuning", row["size"] in BIG) for row in gate.iter_rows(named=True)]
    jobs += [(row, "replicate", False) for row in realistic.iter_rows(named=True)]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        results = list(pool.map(local_readings, jobs, chunksize=4))
    frame = pl.DataFrame(results, strict=False)
    gate_rows = frame.filter(pl.col("transfer_id").str.starts_with("G-"), pl.col("size").is_in(BIG))
    realistic_rows = frame.filter(pl.col("transfer_id").str.starts_with("C1-"))
    ratio = float(gate_rows["ratio"].median())
    release = float(gate_rows.filter(pl.col("released").is_not_null())["released"].mean())
    recall_day = recall_over(frame=gate_rows.filter(pl.col("duration") == "over 1 day"))
    recall_6_24 = recall_over(frame=gate_rows.filter(pl.col("duration") == "6-24h"))
    recall_real = recall_over(frame=realistic_rows.filter(pl.col("hours") > 24.0))
    _setup(cut_name="replicate")
    _, config, data, paths = _STATE["setup:replicate"]
    placebo = {
        "-".join(pair): local.placebo_mw_days(data=data, paths=paths, config=config, pair=pair)
        for pair in common.PLACEBO_PAIRS["replicate"]
    }
    gap = nra_gap()
    readings = {
        "1_moved_mw_ratio": {"value": ratio, "pass": RATIO_RANGE[0] <= ratio <= RATIO_RANGE[1]},
        "2_release_share": {"value": release, "pass": release >= RELEASE_SHARE},
        "3a_recall_over_1_day": {"value": recall_day, "pass": recall_day >= RECALL_OVER_DAY},
        "3b_recall_6_to_24_h": {"value": recall_6_24, "pass": recall_6_24 >= RECALL_6_TO_24H},
        "3c_recall_realistic_over_1_day": {
            "value": recall_real,
            "pass": recall_real >= RECALL_REALISTIC,
        },
        "4_placebo_mw_days": {
            "value": placebo,
            "pass": max(placebo.values()) < PLACEBO_LIMIT,
        },
        "5_nra_gap_closed": {"value": gap, "pass": gap["closed"] >= GAP_SHARE},
    }
    readings["gate_passes"] = {"value": all(r["pass"] for r in readings.values()), "pass": None}
    score.write_json(path=GATE_PATH, payload=readings)
    frame.write_parquet(common.WORK_DIR / "gate_local.parquet")
    for name, reading in readings.items():
        print(name, reading["value"], "PASS" if reading["pass"] else "FAIL")


if __name__ == "__main__":
    main()
