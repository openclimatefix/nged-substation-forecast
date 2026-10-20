"""POST HOC variants of the flow fit, run after the registered gate failed (experiment 116).

These variants were chosen after the gate's first run, to learn why the registered configuration
fails; they are exploratory, and none of them changes the verdict. Each variant is scored on a
sample of the gate's transfers of 25% and 40% of the receiving load (the first 100 by identifier)
with the gate's recall and moved-MW readings, and on the tuning year's two placebo pairs.

    uv run python ml-experiments/00116-*/e00116_variants.py [--workers 4]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Final

import e00116_common as common
import e00116_data as data_module
import e00116_inject as inject
import e00116_local as local
import e00116_score as score
import numpy as np
import polars as pl
from studies.dp_edge_flow import mw_weighted_recall

SAMPLE: Final[int] = 100
VARIANTS: Final[dict[str, data_module.FlowConfig]] = {
    "huber10": data_module.FlowConfig(step_c=300.0, level_c=0.0, huber_sigmas=10.0),
    "loose": data_module.FlowConfig(step_c=10.0, level_c=0.3),
    "loose_huber10": data_module.FlowConfig(step_c=10.0, level_c=0.3, huber_sigmas=10.0),
    "loosest_huber10": data_module.FlowConfig(step_c=3.0, level_c=0.0, huber_sigmas=10.0),
}
"""The registered configuration (step 300, level 0, Huber 1.5) is not a variant."""
RESULT_PATH: Final = common.WORK_DIR / "variants.json"


def run_variant(item: tuple[str, data_module.FlowConfig]) -> dict[str, Any]:
    """Score one variant on the sample and the placebo pairs."""
    import torch  # ty: ignore[unresolved-import]

    torch.set_num_threads(1)
    name, config = item
    cache = common.load_cache()
    data = data_module.load_year(cache=cache, cut_name="tuning", arm="a0")
    paths = data_module.build_graph_paths(
        data=data, edges=data_module.decision_graph(data=data), config=config
    )
    transfers = (
        inject.draw_set(set_name="G", year_name="tuning", cache=cache)
        .filter(pl.col("size").is_in(["25%", "40%"]))
        .sort("transfer_id")
        .head(SAMPLE)
    )
    rows = []
    for transfer in transfers.iter_rows(named=True):
        window = local.fit_local(
            cache=cache, data=data, paths=paths, config=config, transfer=transfer
        )
        span = slice(window.onset, window.end)
        true, fitted = window.true_flow[span], window.fitted_edge()[span]
        total = float(np.sum(np.abs(true)))
        rows.append(
            {
                "duration": transfer["duration"],
                "ratio": float(np.mean(fitted) / np.mean(true)),
                "matched": float(mw_weighted_recall(fitted=fitted, true=true) * total),
                "total": total,
            }
        )
    frame = pl.DataFrame(rows)

    def recall(duration: str) -> float:
        part = frame.filter(pl.col("duration") == duration)
        return float(np.sum(part["matched"].to_numpy()) / np.sum(part["total"].to_numpy()))

    placebo = {
        "-".join(pair): local.placebo_mw_days(data=data, paths=paths, config=config, pair=pair)
        for pair in common.PLACEBO_PAIRS["tuning"]
    }
    return {
        "variant": name,
        "config": config.label(),
        "n_transfers": frame.height,
        "median_ratio": float(np.median(frame["ratio"].to_numpy())),
        "recall_over_1_day": recall("over 1 day"),
        "recall_6_to_24_h": recall("6-24h"),
        "recall_3_to_6_h": recall("3-6h"),
        "placebo_mw_days": placebo,
    }


def main() -> None:
    """Run every variant in parallel and write `variants.json`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    common.load_cache()
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        results = list(pool.map(run_variant, VARIANTS.items()))
    score.write_json(path=RESULT_PATH, payload={"post_hoc_variants": results})
    for result in results:
        print(result, flush=True)


if __name__ == "__main__":
    main()
