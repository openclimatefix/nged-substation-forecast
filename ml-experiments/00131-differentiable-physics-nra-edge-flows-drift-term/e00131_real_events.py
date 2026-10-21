"""Descriptive comparison of the hindsight flows with the logged real transfers, experiment 131.

For each year, the chosen setting with a drift term is fitted to the whole decision graph over the
scored year. Every hand-filtered real transfer (experiment 41's labelled set, kinds
`real_operation`, visibility conserved or one-sided) whose two ends are both substations joined by
a decision edge is then compared with the flow on that edge. A transfer is found when the mean
flow over its rows, signed towards the receiver, reaches the big-event threshold of the receiving
substation and the transfer lasts at least 3 hours. A fitted flow run (rows where the flow is at
least the edge's threshold, at least 3 hours long) is a true alarm when a labelled transfer on the
same edge overlaps it within 24 hours. Pairs are named by D-label only. No verdict depends on this.

    uv run python ml-experiments/00131-*/e00131_real_events.py
"""

import json
import os
from typing import Final

import e00131_common as common
import e00131_data as data_module
import e00131_local as local
import e00131_threshold as threshold
import numpy as np
import polars as pl
from studies.dp_edge_flow import edge_name
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DECISIONS_PATH: Final = common.WORK_DIR / "decisions.json"
OUTPUT_PATH: Final = common.WORK_DIR / "real_events.json"
MIN_RUN_HALF_HOURS: Final[int] = 6
"""A big event lasts at least 3 hours."""
MATCH_HALF_HOURS: Final[int] = 48
"""A fitted run matches a labelled transfer on the same edge within 24 hours."""


def label_of_series() -> dict[int, str]:
    """Return the D-label of each time-series identifier, from the private files."""
    series = pl.read_parquet(common.SERIES_PATH).select("time_series_id", "site")
    monitors = pl.read_csv(common.HIERARCHY_DIR / "Monitor Direction.csv").select(
        "TimeSeriesID", "SubstationNumber"
    )
    joined = series.join(monitors, left_on="time_series_id", right_on="TimeSeriesID")
    return {int(row[0]): row[1] for row in joined.select("time_series_id", "site").rows()}


def labelled_events() -> pl.DataFrame:
    """Return the hand-filtered real transfers with both sides given."""
    return (
        pl.read_parquet(
            ML_EXPERIMENTS_DATA_DIR / "00041-sw2-0a-labeller-gate" / "real" / "labelled_set.parquet"
        )
        .filter(~pl.col("is_sentinel"), pl.col("kind") == "real_operation")
        .filter(pl.col("visibility").is_in(["conserved", "one_sided"]))
        .filter(pl.col("side_a").is_not_null(), pl.col("side_b").is_not_null())
    )


def main() -> None:
    """Fit each year's hindsight flows and compare them with the labelled transfers."""
    os.environ["OMP_NUM_THREADS"] = "1"
    cache = common.load_cache()
    config = data_module.FlowConfig(**json.loads(DECISIONS_PATH.read_text())["drift"]["config"])
    labels = label_of_series()
    events = labelled_events()
    tuning_data = data_module.load_year(cache=cache, cut_name="tuning", arm="a0")
    rows, runs = [], []
    for year in common.CUTS:
        data = data_module.load_year(cache=cache, cut_name=year, arm="a0")
        edges = data_module.decision_graph(data=data)
        limit = {
            node: threshold.THRESHOLD_SHARE
            * threshold.p99_demand_mw(cache=cache, data=tuning_data, node=node)
            for node in data.sites
            if node in tuning_data.sites
        }
        paths = data_module.build_graph_paths(data=data, edges=edges, config=config)
        residuals = data.year()
        fit = data_module.fit_year_flows(
            paths=paths,
            residuals=residuals,
            noise=data.noise_mw(block=config.block),
            config=config,
            drift_step_variance=local.drift_for(data=data, paths=paths, config=config),
        )
        spans = []
        for event in events.iter_rows(named=True):
            a, b = labels.get(int(event["side_a"])), labels.get(int(event["side_b"]))
            if a is None or b is None or tuple(sorted((a, b))) not in set(edges):
                continue
            onset = int(
                np.searchsorted(
                    cache.end_times,
                    np.datetime64(event["onset_utc"].replace(tzinfo=None), "us"),
                    side="right",
                )
            )
            end = int(
                np.searchsorted(
                    cache.end_times,
                    np.datetime64(event["end_utc"].replace(tzinfo=None), "us"),
                    side="right",
                )
            )
            if not (data.first <= onset and end <= data.stop):
                continue
            node_a, node_b = sorted((a, b))
            name = edge_name(node_a=node_a, node_b=node_b)
            receiver = labels.get(int(event["donor"])) if event["donor"] is not None else None
            receiver = node_b if receiver == node_a else node_a if receiver == node_b else None
            flow = fit.flows[name][onset - data.first : end - data.first]
            sign = 0.0 if receiver is None else (1.0 if receiver == node_a else -1.0)
            mean_towards_receiver = float(np.mean(flow)) * sign
            edge_limit = min(limit.get(node_a, np.inf), limit.get(node_b, np.inf))
            receiver_limit = limit.get(receiver, edge_limit) if receiver else edge_limit
            hours = (end - onset) / 2.0
            rows.append(
                {
                    "year": year,
                    "edge": name,
                    "hours": hours,
                    "visibility": event["visibility"],
                    "labelled_moved_mw": event["moved_mw_p50"],
                    "fitted_mean_flow_mw": mean_towards_receiver,
                    "direction_known": receiver is not None,
                    "found": bool(hours >= 3.0 and mean_towards_receiver >= receiver_limit),
                    "big": bool(
                        hours >= 3.0
                        and event["moved_mw_p50"] is not None
                        and abs(event["moved_mw_p50"]) >= receiver_limit
                    ),
                }
            )
            spans.append((name, onset, end))
        for name, flow in fit.flows.items():
            if name.startswith("outside:"):
                continue
            node_a, node_b = name.split("-")
            edge_limit = min(limit.get(node_a, np.inf), limit.get(node_b, np.inf))
            run_starts, run_sizes = run_lengths_of(flow=flow, limit=edge_limit)
            for start, length in zip(run_starts, run_sizes, strict=True):
                if length < MIN_RUN_HALF_HOURS:
                    continue
                first, stop = data.first + int(start), data.first + int(start + length)
                matched = any(
                    span == name
                    and first <= span_end + MATCH_HALF_HOURS
                    and span_start <= stop + MATCH_HALF_HOURS
                    for span, span_start, span_end in spans
                )
                runs.append(
                    {"year": year, "edge": name, "half_hours": int(length), "matched": matched}
                )
    frame, run_frame = pl.DataFrame(rows), pl.DataFrame(runs)
    summary = {
        "events": frame.height,
        "big_events": int(frame["big"].sum()),
        "big_events_found": int((frame["big"] & frame["found"]).sum()),
        "all_events_found": int(frame["found"].sum()),
        "flow_runs": run_frame.height,
        "flow_runs_matched": int(run_frame["matched"].sum()),
    }
    print(summary)
    print(frame.sort("year", "edge"))
    OUTPUT_PATH.write_text(
        json.dumps(
            {"summary": summary, "events": frame.to_dicts(), "runs": run_frame.to_dicts()},
            indent=2,
            default=str,
        )
    )


def run_lengths_of(*, flow: np.ndarray, limit: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the start and length of each run where the flow's magnitude reaches `limit`."""
    active = np.abs(flow) >= limit
    padded = np.r_[False, active, False].astype(int)
    change = np.flatnonzero(np.diff(padded))
    return change[::2], change[1::2] - change[::2]


if __name__ == "__main__":
    main()
