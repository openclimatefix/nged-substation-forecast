"""Refit the convex model on power with the edge flows taken out (arm A4), for experiment 116.

A4 is A0's model refitted once on `y` minus the hindsight edge flows at the substation, so its
training power is the normal running arrangement (NRA). The outside paths stay in the power, since
their level also absorbs growth and new connections. The flows come from `e00116_flows.py`: the
replicate cut's refit removes the tuning year's flows, and the scored cut's refit removes the
tuning and replicate years' flows. A year without flows trains on the power as it is. A job writes
`work/gam/a4/<site>_<cut>.parquet` in the format of `e00116_gam.py`, so the same loaders read it.

    uv run python ml-experiments/00116-*/e00116_refit.py [--workers 6]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime
from typing import Any

import e00116_common as common
import e00116_data as data_module
import e00116_flows as flows_module
import e00116_gam as gam
import numpy as np
import polars as pl
from studies.dp_edge_flow import node_flow
from studies.dp_features import FeatureCache
from studies.dp_gam import TrainingWindow, fit_gam, predict_gam
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

FLOW_YEARS: dict[str, tuple[str, ...]] = {
    "replicate": ("tuning",),
    "scored": ("tuning", "replicate"),
}
"""Which years' hindsight flows each cut's refit removes."""
_STATE: dict[str, Any] = {}


def flow_free_cache(*, cache: FeatureCache, cut_name: str) -> FeatureCache:
    """Return the cache with each substation's power minus its edge flows, where flows exist.

    Args:
        cache: The feature cache.
        cut_name: `replicate` or `scored`.

    Returns:
        A cache whose `y` is in points of effective capacity with the flows (megawatts divided by
        capacity over 100) taken out, and whose other arrays are unchanged.
    """
    y = {site: cache.sites[site].y.copy() for site in gam.SITES}
    for year_name in FLOW_YEARS[cut_name]:
        data = data_module.load_year(cache=cache, cut_name=year_name, arm="a0")
        edges = data_module.decision_graph(data=data)
        paths = flows_module.edge_paths(
            paths=data_module.build_graph_paths(
                data=data, edges=edges, config=flows_module.chosen_config()
            )
        )
        flows = flows_module.read_flows(year_name=year_name)
        positions = flows.pop("position")
        year_flows = {name: flow for name, flow in flows.items() if not name.startswith("outside")}
        for site in {node for edge in edges for node in edge}:
            scale = data.capacity_mw[site] / data_module.POINTS_PER_CAPACITY
            removed = node_flow(paths=paths, flows=year_flows, node=site) / scale
            y[site][positions] = y[site][positions] - removed
    sites = {
        site: replace(features, y=y[site]) if site in y else features
        for site, features in cache.sites.items()
    }
    return replace(cache, sites=sites)


def job_path(*, site: str, cut: datetime) -> Any:
    """Return the file a job writes."""
    return gam.job_path(arm="a4", site=site, cut=cut)  # ty: ignore[invalid-argument-type]


def _worker(job: tuple[str, str]) -> str:
    """Refit one substation at one cut in a worker process with one thread."""
    import torch  # ty: ignore[unresolved-import]

    torch.set_num_threads(1)
    cut_name, site = job
    cut = common.CUTS[cut_name]
    target = job_path(site=site, cut=cut)
    if target.exists():
        return f"skip {target.name}"
    key = f"cache:{cut_name}"
    if key not in _STATE:
        base = common.load_cache()
        _STATE[key] = (base, flow_free_cache(cache=base, cut_name=cut_name))
        _STATE["vintages"] = read_vintages()
    base, free = _STATE[key]
    if gam.usable_months(cache=base, site=site, cut=cut) < common.MIN_TRAIN_MONTHS:
        return f"too little history: a4 {site} {cut_name}"
    features = base.sites[site]
    mva = features.table.units == "MVA"
    pinned = (
        None if mva else pinned_capacity(vintages=_STATE["vintages"], table=features.table, cut=cut)
    )
    fit = fit_gam(
        cache=gam.arm_cache(cache=free, site=site, arm="a0"),
        site=site,
        specs=[common.gam_spec(units=features.table.units)],
        cut=cut,
        window=TrainingWindow(months=common.WINDOW_MONTHS, dark_only=mva),
        season_years=common.SEASON_YEARS,
        pinned=pinned,
    )
    rows = np.zeros(len(base.end_times), dtype=bool)
    rows[common.window_slice(cache=base, cut=cut)] = True
    keep = rows & features.present & np.isfinite(features.y)
    if mva:
        keep &= features.dark
    index = np.flatnonzero(keep)
    frame = pl.DataFrame(
        {
            "position": index.astype(np.int64),
            "y": features.y[index],
            "mean": predict_gam(cache=free, fit=fit, index=index),
            "default_row": features.default_rows[index],
            "switching_mask": features.switching_mask[index],
        }
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    frame.write_parquet(temporary)
    temporary.replace(target)
    return f"done a4 {target.name}: {frame.height} rows"


def main() -> None:
    """Refit every job in parallel worker processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    jobs = [(cut_name, site) for cut_name in FLOW_YEARS for site in gam.SITES]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for message in pool.map(_worker, jobs):
            print(message, flush=True)


if __name__ == "__main__":
    main()
