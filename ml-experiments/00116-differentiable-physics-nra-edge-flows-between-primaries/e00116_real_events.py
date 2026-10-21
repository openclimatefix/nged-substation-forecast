"""Descriptive check of the hindsight flows against the real transfers (experiment 116).

No verdict depends on this. For each hand-filtered event between the two gate edges' substations
(D5 and D16, D10 and D11), whose onset falls in a year that has hindsight flows, the script reports
the event's moved megawatts (the labeller's median estimate) and the mean flow the fit puts on the
edge over the event, signed towards the receiving substation. Events that involve either of the two
series labelled G-c are never listed.

    uv run python ml-experiments/00116-*/e00116_real_events.py
"""

from typing import Final

import e00116_common as common
import e00116_flows as flows_module
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EDGES: Final[dict[frozenset[int], tuple[str, str]]] = {
    frozenset({17, 18}): ("D16", "D5"),
    frozenset({10, 8}): ("D10", "D11"),
}
"""Time-series identifiers of each gate edge's two ends, and the edge's first and second node."""
RECEIVER_OF_DONOR: Final[dict[int, str]] = {17: "D5", 18: "D16", 10: "D10", 8: "D11"}


def events() -> pl.DataFrame:
    """Return the hand-filtered events on the two gate edges, as experiment 101 filters them."""
    frame = (
        pl.read_parquet(
            ML_EXPERIMENTS_DATA_DIR / "00041-sw2-0a-labeller-gate" / "real" / "labelled_set.parquet"
        )
        .filter(~pl.col("is_sentinel"), pl.col("kind") == "real_operation")
        .filter(pl.col("visibility").is_in(["conserved", "one_sided"]))
        .filter(pl.col("side_a").is_not_null(), pl.col("side_b").is_not_null())
    )
    keep = [
        row
        for row in frame.iter_rows(named=True)
        if frozenset({row["side_a"], row["side_b"]}) in EDGES
    ]
    return pl.DataFrame(keep)


def main() -> None:
    """Print one line per event with flows available."""
    cache = common.load_cache()
    flows = {year: flows_module.read_flows(year_name=year) for year in common.CUTS}
    rows = []
    for event in events().iter_rows(named=True):
        node_a, node_b = EDGES[frozenset({event["side_a"], event["side_b"]})]
        onset = np.datetime64(event["onset_utc"].replace(tzinfo=None), "us")
        end = np.datetime64(event["end_utc"].replace(tzinfo=None), "us")
        first = int(np.searchsorted(cache.end_times, onset, side="right"))
        stop = int(np.searchsorted(cache.end_times, end, side="right"))
        for year, table in flows.items():
            positions = table["position"]
            if positions[0] <= first and stop <= positions[-1]:
                name = (
                    f"{node_a}-{node_b}" if f"{node_a}-{node_b}" in table else f"{node_b}-{node_a}"
                )
                mean_flow = float(np.mean(table[name][first - positions[0] : stop - positions[0]]))
                receiver_is_first = event["donor"] is not None and (
                    RECEIVER_OF_DONOR.get(int(event["donor"])) == node_b
                )
                signed = mean_flow if receiver_is_first else -mean_flow
                rows.append(
                    {
                        "edge": f"{node_a}-{node_b}",
                        "year": year,
                        "onset": str(event["onset_utc"].date()),
                        "hours": (event["end_utc"] - event["onset_utc"]).total_seconds() / 3600.0,
                        "visibility": event["visibility"],
                        "labelled_moved_mw": event["moved_mw_p50"],
                        "fitted_mean_flow_mw": signed,
                    }
                )
    frame = pl.DataFrame(rows)
    print(frame)
    frame.write_json(common.WORK_DIR / "real_events.json")
    print(f"{len(events())} events on the two gate edges; {frame.height} fall in a year with flows")


if __name__ == "__main__":
    main()
