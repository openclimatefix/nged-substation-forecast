"""Post hoc diagnostic of experiment 131: where does the replicate-year placebo flow fall?

For the chosen setting with a drift term, this fits each replicate-year placebo pair in 72-day
tiles (placebo path listed first) and splits the flow into rows within 48 hours of a hand-filtered
switching event at either end of the pair, and all other rows. It also records each tile's flow.
The result is exploratory and plays no part in the verdict.

    uv run python ml-experiments/00131-*/e00131_placebo_where.py
"""

import json
from typing import Final

import e00131_common as common
import e00131_data as data_module
import e00131_local as local
import numpy as np
from studies.dp_edge_flow import PathSpec

DECISIONS_PATH: Final = common.WORK_DIR / "decisions.json"
OUTPUT_PATH: Final = common.WORK_DIR / "placebo_where.json"
BUFFER_HALF_HOURS: Final[int] = 2 * 48
"""Rows within 48 hours of a hand-filtered event count as near an event."""


def main() -> None:
    """Fit the replicate placebo pairs and write the split."""
    cache = common.load_cache()
    config = data_module.FlowConfig(**json.loads(DECISIONS_PATH.read_text())["drift"]["config"])
    data = data_module.load_year(cache=cache, cut_name="replicate", arm="a0")
    paths = data_module.build_graph_paths(
        data=data, edges=data_module.decision_graph(data=data), config=config
    )
    result = {}
    for pair in common.PLACEBO_PAIRS["replicate"]:
        node_a, node_b = sorted(pair)
        scale = data.edge_scale_mw(node_a=node_a, node_b=node_b, block=config.block)
        placebo = PathSpec(
            name="placebo",
            node_a=node_a,
            node_b=node_b,
            step_penalty=config.step_c / scale,
            level_penalty=config.level_c / scale,
        )
        blocked = cache.sites[node_a].switching_mask | cache.sites[node_b].switching_mask
        near = (
            np.convolve(
                blocked.astype(np.int32),
                np.ones(2 * BUFFER_HALF_HOURS + 1, dtype=np.int32),
                "same",
            )
            > 0
        )
        jump = np.zeros(len(cache.end_times), dtype=bool)
        if "D10" in pair:
            for start, end in common.D10_JUMP_WINDOWS:
                jump |= (cache.end_times > common.to_64(moment=start)) & (
                    cache.end_times <= common.to_64(moment=end)
                )
        near_days = other_days = jump_days = 0.0
        tiles = []
        for first, stop in local.placebo_tiles(data=data):
            residuals = {node: r[first:stop].copy() for node, r in data.residual_mw.items()}
            fit = local.fit_window(
                data=data,
                paths=(placebo, *paths),
                config=config,
                residuals=residuals,
                nodes=pair,
            )
            flow = np.abs(fit.flows["placebo"]) / 48.0
            inside = near[first:stop]
            jump_days += float(flow[jump[first:stop]].sum())
            near_days += float(flow[inside].sum())
            other_days += float(flow[~inside].sum())
            tiles.append({"first": first, "mw_days": float(flow.sum())})
        result["-".join(pair)] = {
            "near_event_mw_days": near_days,
            "in_d10_unlabelled_jump_windows_mw_days": jump_days,
            "other_mw_days": other_days,
            "share_of_rows_near_event": float(near[data.first : data.stop].mean()),
            "tiles": tiles,
        }
        print(pair, result["-".join(pair)])
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
