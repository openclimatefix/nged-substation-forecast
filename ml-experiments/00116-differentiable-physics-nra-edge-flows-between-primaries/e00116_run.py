"""Run the forecast pipelines on injected transfers and on real data, for experiment 116.

For each transfer, every origin from 6 hours before its onset to 48 hours after its end (every 6
hours) is forecast at both ends by four arms: `a0` (the plain filter on A0's fit), `a4` (the plain
filter on A4's fit), `a5` (A4's fit with causal flows), and `or` (A4's fit with the true flow, the
ceiling). The rows kept are those whose valid time falls inside the transfer or in the 48 hours
after it. The replica runs produce each arm's forecasts at daily 09:00 UTC origins on the year
before, from which the spread is fitted.

A job writes `work/transfers/<set>_<year>/<transfer_id>.parquet` or
`work/replica/<year>_<arm>_<node>.parquet` atomically and is skipped if it exists.

    uv run python ml-experiments/00116-*/e00116_run.py transfers --set G --year tuning
    uv run python ml-experiments/00116-*/e00116_run.py replica --year replicate
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Final

import e00116_causal as causal
import e00116_common as common
import e00116_flows as flows_module
import e00116_gam as gam
import e00116_inject as inject
import e00116_local as local
import numpy as np
import polars as pl

ORIGIN_STEP_HALF_HOURS: Final[int] = 12
"""Origins are every 6 hours."""
PRE_ONSET_HALF_HOURS: Final[int] = 12
"""The first origin is 6 hours before the onset."""
POST_END_HALF_HOURS: Final[int] = 96
"""Rows after the end that are scored, and origins continue to."""
REPLICA_NODES: Final[tuple[str, ...]] = common.GATE_SITES
SCORED_ARMS: Final[tuple[str, ...]] = ("a0", "a4", "a5", "or")
_STATE: dict[str, Any] = {}


def arm_for_cut(*, cut_name: str, arm: gam.ArmType) -> gam.ArmType:
    """Return the arm whose stored fit exists at a cut: A4 is A0 at the tuning cut."""
    if arm == "a4" and cut_name == "tuning":
        return "a0"
    return arm


def flows_survival(*, cut_name: str) -> Any:
    """Return the survival of the hindsight flows in a cut's window.

    The scored cut's window holds the tuning and replicate years' flows, the replicate cut's the
    tuning year's, and the tuning cut's own year stands in for its window, which has none.
    """
    years = {"tuning": ("tuning",), "replicate": ("tuning",), "scored": ("tuning", "replicate")}
    pooled: dict[str, np.ndarray] = {}
    for year in years[cut_name]:
        for name, flow in flows_module.read_flows(year_name=year).items():
            pooled[f"{year}:{name}"] = flow
    return causal.survival_from_flows(flows=pooled)


def contexts(*, cut_name: str) -> dict[str, causal.Context]:
    """Build (once per process) the contexts of A0 and A4 for a cut."""
    key = f"ctx:{cut_name}"
    if key not in _STATE:
        cache = common.load_cache()
        config = flows_module.chosen_config()
        survival = flows_survival(cut_name=cut_name)
        _STATE[key] = {
            arm: causal.build_context(
                cache=cache,
                cut_name=cut_name,
                arm=arm_for_cut(cut_name=cut_name, arm=arm),
                config=config,
                survival=survival,
            )
            for arm in ("a0", "a4")
        }
    return _STATE[key]


def origin_cutoffs(*, onset: int, end: int) -> list[int]:
    """Return the newest usable position of each origin of a transfer."""
    delay = causal.DELAY_HALF_HOURS
    first = onset - PRE_ONSET_HALF_HOURS
    return list(range(first - delay, end + POST_END_HALF_HOURS - delay, ORIGIN_STEP_HALF_HOURS))


def _rows(
    *,
    transfer: dict,
    node: str,
    arm: str,
    forecasts: list[causal.OriginForecast],
    context: causal.Context,
    actual: np.ndarray,
    clean: np.ndarray,
) -> pl.DataFrame:
    """Turn one arm's forecasts at one node into rows inside the transfer or its 48 hours after."""
    onset, end = int(transfer["onset"]), int(transfer["end"])
    end_times = context.cache.end_times
    frames = []
    for forecast in forecasts:
        target = forecast.target
        keep = (target >= onset) & (target < end + POST_END_HALF_HOURS)
        keep &= np.isfinite(actual[target]) & np.isfinite(context.mean_points[node][target])
        if not keep.any():
            continue
        tau = forecast.cutoff + causal.DELAY_HALF_HOURS
        t = target[keep]
        frames.append(
            pl.DataFrame(
                {
                    "transfer_id": transfer["transfer_id"],
                    "node": node,
                    "role": "receiver" if node == transfer["receiver"] else "donor",
                    "arm": arm,
                    "cutoff": forecast.cutoff,
                    "target": t,
                    "lead_hours": (end_times[t] - end_times[tau]) / np.timedelta64(1, "h"),
                    "in_event": (t >= onset) & (t < end),
                    "actual": actual[t],
                    "clean": clean[t],
                    "gam": context.mean_points[node][t],
                    "level": forecast.level[keep],
                    "anomaly": forecast.anomaly[keep],
                    "flow": forecast.flow[keep],
                }
            )
        )
    return pl.concat(frames) if frames else pl.DataFrame()


def run_transfer(*, transfer: dict, cut_name: str, only_a5: bool = False) -> pl.DataFrame:
    """Run the four arms on one transfer at its two ends.

    Args:
        transfer: One row of the transfers table.
        cut_name: The year's cut.
        only_a5: True to compute only arm A5, to refresh an existing file.

    Returns:
        Long rows, one per arm, node, origin, and forecast row, as described in the module
        docstring.
    """
    ctx = contexts(cut_name=cut_name)
    cache = ctx["a0"].cache
    nodes = (transfer["receiver"], transfer["donor"])
    cutoffs = origin_cutoffs(onset=int(transfer["onset"]), end=int(transfer["end"]))
    frames = []
    for arm in ("a4",) if only_a5 else ("a0", "a4"):
        streams, added = local.injected_streams(cache=cache, data=ctx[arm].data, transfer=transfer)
        scale = ctx[arm].scale
        for node in () if only_a5 else nodes:
            clean = cache.sites[node].y
            actual = clean + added[node] / scale[node]
            plain = causal.plain_forecasts(
                context=ctx[arm], streams_mw=streams, node=node, cutoffs=cutoffs
            )
            frames.append(
                _rows(
                    transfer=transfer,
                    node=node,
                    arm=arm,
                    forecasts=plain,
                    context=ctx[arm],
                    actual=actual,
                    clean=clean,
                )
            )
            if arm == "a4":
                oracle = causal.oracle_forecasts(
                    context=ctx[arm],
                    streams_mw=streams,
                    node=node,
                    true_node_flow_mw=added[node],
                    cutoffs=cutoffs,
                )
                frames.append(
                    _rows(
                        transfer=transfer,
                        node=node,
                        arm="or",
                        forecasts=oracle,
                        context=ctx[arm],
                        actual=actual,
                        clean=clean,
                    )
                )
        if arm == "a4":
            a5 = causal.causal_forecasts(
                context=ctx[arm], streams_mw=streams, nodes=nodes, cutoffs=cutoffs
            )
            for node in nodes:
                clean = cache.sites[node].y
                frames.append(
                    _rows(
                        transfer=transfer,
                        node=node,
                        arm="a5",
                        forecasts=a5[node],
                        context=ctx[arm],
                        actual=clean + added[node] / scale[node],
                        clean=clean,
                    )
                )
    return pl.concat([f for f in frames if f.height])


def transfers_dir(*, set_name: str, year_name: str) -> Path:
    """Return the folder that holds a set's per-transfer results."""
    return common.WORK_DIR / "transfers" / f"{set_name}_{year_name}"


def _transfer_worker(job: tuple[dict, str, str, bool]) -> str:
    """Run one transfer in a worker process with one thread, and write its file."""
    import torch  # ty: ignore[unresolved-import]

    torch.set_num_threads(1)
    transfer, set_name, year_name, refresh = job
    target = (
        transfers_dir(set_name=set_name, year_name=year_name) / f"{transfer['transfer_id']}.parquet"
    )
    if target.exists() and not refresh:
        return f"skip {target.name}"
    frame = run_transfer(transfer=transfer, cut_name=year_name, only_a5=refresh)
    if refresh:
        frame = pl.concat([pl.read_parquet(target).filter(pl.col("arm") != "a5"), frame])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    frame.write_parquet(temporary)
    temporary.replace(target)
    return f"done {target.name}: {frame.height} rows"


def run_replica(
    *, year_name: str, arm: str, chunk: int, n_chunks: int, own_year: bool = False
) -> pl.DataFrame:
    """Forecast the year before `year_name` at daily 09:00 UTC origins, for one arm and chunk.

    The replica of a year's forecasts is the same arm run over the previous year's real data, so
    its residuals are as stale as the scored forecasts. A5 refits the flows of the four gate sites'
    paths at every origin.

    Args:
        year_name: The year being scored (`replicate` or `scored`).
        arm: `a0`, `a4`, or `a5`.
        chunk: Which share of the origins this call runs.
        n_chunks: How many shares the origins are split into.
        own_year: True to run on the scored year's own real data instead of the previous year's,
            for the check that causal flows do no harm outside the injected transfers.

    Returns:
        Rows with the node, the origin's cutoff, the target, the lead, the actual power, the frozen
        mean, and the corrections, over the 96 forecast rows of each origin.
    """
    previous = year_name if own_year else {"replicate": "tuning", "scored": "replicate"}[year_name]
    ctx = contexts(cut_name=previous)
    context = ctx["a0" if arm == "a0" else "a4"]
    cache = context.cache
    data = context.data
    window = np.arange(data.first + 60 * 48, data.stop - POST_END_HALF_HOURS - 8)
    at_nine = (cache.end_times[window] - cache.end_times[window].astype("datetime64[D]")) == (
        np.timedelta64(9, "h")
    )
    cutoffs = [int(position) - causal.DELAY_HALF_HOURS for position in window[at_nine]]
    cutoffs = cutoffs[chunk::n_chunks]
    streams = data.out_of_sample()
    if arm == "a5":
        forecasts = causal.causal_forecasts(
            context=context, streams_mw=streams, nodes=REPLICA_NODES, cutoffs=cutoffs
        )
    else:
        forecasts = {
            node: causal.plain_forecasts(
                context=context, streams_mw=streams, node=node, cutoffs=cutoffs
            )
            for node in REPLICA_NODES
        }
    frames = []
    for node in REPLICA_NODES:
        actual = cache.sites[node].y
        for forecast in forecasts[node]:
            target = forecast.target
            keep = np.isfinite(actual[target]) & np.isfinite(context.mean_points[node][target])
            tau = forecast.cutoff + causal.DELAY_HALF_HOURS
            frames.append(
                pl.DataFrame(
                    {
                        "node": node,
                        "cutoff": forecast.cutoff,
                        "target": target[keep],
                        "lead_hours": (cache.end_times[target[keep]] - cache.end_times[tau])
                        / np.timedelta64(1, "h"),
                        "actual": actual[target[keep]],
                        "gam": context.mean_points[node][target[keep]],
                        "correction": forecast.correction[keep],
                        "switching": cache.sites[node].switching_mask[target[keep]],
                    }
                )
            )
    return pl.concat(frames)


def _replica_worker(job: tuple[str, str, int, int, bool]) -> str:
    """Run one replica chunk in a worker process with one thread, and write its file."""
    import torch  # ty: ignore[unresolved-import]

    torch.set_num_threads(1)
    year_name, arm, chunk, n_chunks, own_year = job
    prefix = "real_" if own_year else ""
    target = common.WORK_DIR / "replica" / f"{prefix}{year_name}_{arm}_{chunk}of{n_chunks}.parquet"
    if target.exists():
        return f"skip {target.name}"
    frame = run_replica(
        year_name=year_name, arm=arm, chunk=chunk, n_chunks=n_chunks, own_year=own_year
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    frame.write_parquet(temporary)
    temporary.replace(target)
    return f"done {target.name}: {frame.height} rows"


def main() -> None:
    """Run transfers or replicas in parallel worker processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="what", required=True)
    transfers = sub.add_parser("transfers")
    transfers.add_argument("--set", dest="set_name", choices=tuple(inject.SET_IDS), required=True)
    transfers.add_argument("--year", choices=tuple(common.CUTS), required=True)
    transfers.add_argument("--limit", type=int, default=None)
    transfers.add_argument(
        "--refresh-a5", action="store_true", help="recompute only arm a5 in the existing files"
    )
    replica = sub.add_parser("replica")
    replica.add_argument("--year", choices=("replicate", "scored"), required=True)
    replica.add_argument("--chunks", type=int, default=6)
    replica.add_argument(
        "--own-year", action="store_true", help="run on the year's own real data, arms a4 and a5"
    )
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    cache = common.load_cache()
    if arguments.what == "transfers":
        table = inject.draw_set(set_name=arguments.set_name, year_name=arguments.year, cache=cache)
        if arguments.limit is not None:
            table = table.head(arguments.limit)
        jobs: list = [
            (row, arguments.set_name, arguments.year, arguments.refresh_a5)
            for row in table.iter_rows(named=True)
        ]
        worker = _transfer_worker
    else:
        arms = ("a4", "a5") if arguments.own_year else ("a0", "a4", "a5")
        jobs = [
            (arguments.year, arm, chunk, arguments.chunks, arguments.own_year)
            for arm in arms
            for chunk in range(arguments.chunks)
        ]
        worker = _replica_worker
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for message in pool.map(worker, jobs):
            print(message, flush=True)


if __name__ == "__main__":
    main()
