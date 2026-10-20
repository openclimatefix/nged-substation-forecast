"""C1 of experiment 116: does the hindsight NRA reconstruction recover injected training transfers?

Each seed of set `C1` injects up to 8 realistic transfers on the two gate edges into the
replicate year (2024-07 to 2025-06), which lies inside the scored cut's training window. For every
seed the script refits arm A0's convex model at the four gate sites on the injected power
(`a0_inj`), fits the hindsight flows of the neighbourhood of each transfer on those in-sample
residuals, and scores the share of the injected megawatts that the edge flows reconstruct:

    share = 1 - sum |injected - fitted flow| / sum |injected|

over the transfer's rows at both ends. Beside it, three convex models trained on the injected
history forecast the clean scored year (2025-07 to 2026-06): `a0_inj` as it is, `a4_inj` refitted
on the injected power minus the fitted flows, and `or_inj` trained with the injected windows
masked (the oracle). Their mean absolute errors are descriptive: with no drift in the model a
transfer of days barely moves a 24-month fit.

    uv run python ml-experiments/00116-*/e00116_c1.py [--workers 6]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any

import e00116_common as common
import e00116_data as data_module
import e00116_flows as flows_module
import e00116_gam as gam
import e00116_inject as inject
import e00116_local as local
import numpy as np
import polars as pl
from studies.dp_edge_flow import edge_name, node_flow
from studies.dp_features import FeatureCache
from studies.dp_pinned_capacity import read_vintages

CUT_NAME = "scored"
YEAR_NAME = "replicate"
RESULT_DIR = common.WORK_DIR / "c1"
_STATE: dict[str, Any] = {}


def injected_cache(
    *, cache: FeatureCache, data: data_module.YearData, added_mw: dict[str, np.ndarray]
) -> FeatureCache:
    """Return the cache with each gate site's power plus its injected megawatts, in points."""
    sites = dict(cache.sites)
    for node, added in added_mw.items():
        scale = data.capacity_mw[node] / data_module.POINTS_PER_CAPACITY
        sites[node] = replace(cache.sites[node], y=cache.sites[node].y + added / scale)
    return replace(cache, sites=sites)


def fit_mean(*, cache: FeatureCache, vintages: pl.DataFrame, node: str) -> pl.DataFrame:
    """Fit arm A0's model at the scored cut on a cache; return its means over window and year."""
    return gam.fit_job(
        cache=cache, vintages=vintages, arm="a0", site=node, cut=common.CUTS[CUT_NAME]
    )


def residual_stream_mw(*, frame: pl.DataFrame, scale: float, length: int) -> np.ndarray:
    """Return a fit's residual (power minus mean) on the grid in megawatts."""
    stream = np.full(length, np.nan)
    stream[frame["position"].to_numpy()] = (frame["y"] - frame["mean"]).to_numpy() * scale
    return stream


def clean_error(*, frame: pl.DataFrame, clean_y: np.ndarray, first: int, stop: int) -> float:
    """Return a fit's mean absolute error against the clean power over rows `first` to `stop`."""
    position = frame["position"].to_numpy()
    keep = (position >= first) & (position < stop)
    truth = clean_y[position[keep]]
    return float(np.nanmean(np.abs(truth - frame["mean"].to_numpy()[keep])))


def run_seed(*, seed: int) -> pl.DataFrame:
    """Run one seed of set C1 and return one row per transfer and one per decision site."""
    if "cache" not in _STATE:
        cache = common.load_cache()
        config = flows_module.chosen_config()
        data = data_module.load_year(cache=cache, cut_name=CUT_NAME, arm="a0")
        edges = data_module.decision_graph(data=data)
        _STATE.update(
            cache=cache,
            config=config,
            data=data,
            paths=data_module.build_graph_paths(data=data, edges=edges, config=config),
            vintages=read_vintages(),
            transfers=inject.draw_set(set_name="C1", year_name=YEAR_NAME, cache=cache),
        )
    cache, config, data = _STATE["cache"], _STATE["config"], _STATE["data"]
    paths, vintages = _STATE["paths"], _STATE["vintages"]
    transfers = _STATE["transfers"].filter(pl.col("seed") == seed)
    length = len(cache.end_times)
    injection_rows = common.year_slice(cache=cache, cut=common.CUTS[YEAR_NAME])
    added: dict[str, np.ndarray] = {node: np.zeros(length) for node in common.GATE_SITES}
    for transfer in transfers.iter_rows(named=True):
        _, one = local.injected_streams(cache=cache, data=data, transfer=transfer)
        for node in common.GATE_SITES:
            added[node] = added[node] + one[node]
    injected = injected_cache(cache=cache, data=data, added_mw=added)
    fits = {
        node: fit_mean(cache=injected, vintages=vintages, node=node) for node in common.GATE_SITES
    }
    base_streams = dict(data.residual_mw)
    streams = {
        node: residual_stream_mw(
            frame=fits[node],
            scale=data.capacity_mw[node] / data_module.POINTS_PER_CAPACITY,
            length=length,
        )
        if node in fits
        else base
        for node, base in base_streams.items()
    }
    injected_data = replace(data, residual_mw=streams)
    edge_paths = flows_module.edge_paths(paths=paths)
    removed = {node: np.zeros(length) for node in common.GATE_SITES}
    rows = []
    for transfer in transfers.iter_rows(named=True):
        first = max(int(transfer["onset"]) - local.PAD_HALF_HOURS, injection_rows.start)
        stop = min(
            int(transfer["end"]) + inject.POST_HALF_HOURS + local.PAD_HALF_HOURS,
            injection_rows.stop,
        )
        window = local.fit_window(
            data=injected_data,
            paths=paths,
            config=config,
            residuals={node: r[first:stop] for node, r in streams.items()},
            nodes=(transfer["node_a"], transfer["node_b"]),
        )
        local_edges = tuple(p for p in edge_paths if p.name in window.flows and p.name != "outside")
        onset, end = int(transfer["onset"]) - first, int(transfer["end"]) - first
        error = injected_mw = 0.0
        for node in (transfer["receiver"], transfer["donor"]):
            fitted = node_flow(paths=local_edges, flows=window.flows, node=node)
            truth = added[node][first:stop]
            error += float(np.sum(np.abs(truth[onset:end] - fitted[onset:end])))
            injected_mw += float(np.sum(np.abs(truth[onset:end])))
            removed[node][first:stop] += fitted
        rows.append(
            {
                "kind": "transfer",
                "seed": seed,
                "transfer_id": transfer["transfer_id"],
                "month": int(
                    cache.end_times[int(transfer["onset"])].astype("datetime64[M]").astype(int)
                ),
                "error": error,
                "injected": injected_mw,
                "edge": edge_name(node_a=transfer["node_a"], node_b=transfer["node_b"]),
            }
        )
    clean_y = {node: cache.sites[node].y for node in common.GATE_SITES}
    refit_cache = injected_cache(
        cache=cache,
        data=data,
        added_mw={node: added[node] - removed[node] for node in common.GATE_SITES},
    )
    oracle_sites = dict(injected.sites)
    for node in common.GATE_SITES:
        keep = oracle_sites[node].default_rows & (added[node] == 0.0)
        oracle_sites[node] = replace(oracle_sites[node], default_rows=keep)
    oracle_cache = replace(injected, sites=oracle_sites)
    for node in common.GATE_SITES:
        a4 = fit_mean(cache=refit_cache, vintages=vintages, node=node)
        oracle = fit_mean(cache=oracle_cache, vintages=vintages, node=node)
        rows.append(
            {
                "kind": "clean_year",
                "seed": seed,
                "node": node,
                "a0_inj": clean_error(
                    frame=fits[node],
                    clean_y=clean_y[node],
                    first=data.stop,
                    stop=data.stop + common.HALF_HOURS_PER_YEAR,
                ),
                "a4_inj": clean_error(
                    frame=a4,
                    clean_y=clean_y[node],
                    first=data.stop,
                    stop=data.stop + common.HALF_HOURS_PER_YEAR,
                ),
                "or_inj": clean_error(
                    frame=oracle,
                    clean_y=clean_y[node],
                    first=data.stop,
                    stop=data.stop + common.HALF_HOURS_PER_YEAR,
                ),
            }
        )
    return pl.DataFrame(rows, strict=False)


def _worker(seed: int) -> str:
    """Run one seed in a worker process with one thread, and write its file."""
    import torch  # ty: ignore[unresolved-import]

    torch.set_num_threads(1)
    target = RESULT_DIR / f"seed_{seed:02d}.parquet"
    if target.exists():
        return f"skip seed {seed}"
    frame = run_seed(seed=seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    frame.write_parquet(temporary)
    temporary.replace(target)
    return f"done seed {seed}: {frame.height} rows"


def main() -> None:
    """Run every seed of set C1 in parallel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    common.load_cache()
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for message in pool.map(_worker, range(inject.SEEDS_PER_SET)):
            print(message, flush=True)


if __name__ == "__main__":
    main()
