"""Fit the hindsight flows of the whole graph on a year of real residuals, for experiment 116.

The residuals are those of arm A0's frozen fit at the year's cut, out of sample over the year. The
configuration is the one `e00116_tune.py` chose. The flows of every edge path are written to
`work/flows/<year>.parquet`, with the outside paths beside them for the cards; the graph's edge
count is printed before any residual is read.

    uv run python ml-experiments/00116-*/e00116_flows.py --year tuning|replicate|scored
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Final

import e00116_common as common
import e00116_data as data_module
import numpy as np
import polars as pl
from studies.dp_edge_flow import PathSpec

FLOWS_DIR: Final[Path] = common.WORK_DIR / "flows"
DECISIONS_PATH: Final[Path] = common.WORK_DIR / "decisions.json"


def chosen_config() -> data_module.FlowConfig:
    """Return the configuration `e00116_tune.py` chose."""
    chosen = json.loads(DECISIONS_PATH.read_text())["chosen"]["config"]
    return data_module.FlowConfig(**chosen)


def flows_path(*, year_name: str) -> Path:
    """Return the file that holds a year's hindsight flows."""
    return FLOWS_DIR / f"{year_name}.parquet"


def read_flows(*, year_name: str) -> dict[str, np.ndarray]:
    """Read a year's hindsight flows as `{path name: flow}` on the year's rows, plus `position`."""
    frame = pl.read_parquet(flows_path(year_name=year_name))
    return {name: frame[name].to_numpy() for name in frame.columns}


def edge_paths(*, paths: tuple[PathSpec, ...]) -> tuple[PathSpec, ...]:
    """Return the paths that join two nodes: the outside paths are never taken out of the power."""
    return tuple(path for path in paths if path.node_b is not None)


def main() -> None:
    """Fit one year's flows and write them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", choices=tuple(common.CUTS), required=True)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    cache = common.load_cache()
    data = data_module.load_year(cache=cache, cut_name=arguments.year, arm="a0")
    edges = data_module.decision_graph(data=data)
    config = chosen_config()
    paths = data_module.build_graph_paths(data=data, edges=edges, config=config)
    print(f"{len(edges)} decision edges among {len(data.sites)} sites, {len(paths)} paths")
    started = time.monotonic()
    fit = data_module.fit_year_flows(
        paths=paths,
        residuals=data.year(),
        noise=data.noise_mw(block=config.block),
        config=config,
    )
    print(f"{fit.sweeps} sweeps, converged {fit.converged}, {time.monotonic() - started:.0f} s")
    FLOWS_DIR.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame({"position": np.arange(data.first, data.stop), **fit.flows})
    target = flows_path(year_name=arguments.year)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    frame.write_parquet(temporary)
    temporary.replace(target)


if __name__ == "__main__":
    main()
