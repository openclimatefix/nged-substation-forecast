"""The big-event threshold of experiment 131, frozen before any gate reading.

A big event moves at least 10% of the receiving substation's 99th-percentile cleaned demand for at
least 3 hours (every injected transfer lasts at least 3 hours, so the test is on the mean moved
megawatts). The percentile uses the receiving substation's rows at or before the year's cut that
hold a reading and are not in a hand-filtered switching event. This script prints each gate
receiver's threshold in megawatts, the share of each injected size's transfers that clear it, and
the labelled size of 5 hand-filtered events beside it.

    uv run python ml-experiments/00131-*/e00131_threshold.py
"""

from typing import Final

import e00131_common as common
import e00131_data as data_module
import e00131_inject as inject
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache

THRESHOLD_SHARE: Final[float] = 0.10
"""The share of the receiving substation's p99 cleaned demand."""
MIN_BIG_HOURS: Final[float] = 3.0
THRESHOLD_PATH: Final = common.WORK_DIR / "threshold.json"


def p99_demand_mw(*, cache: FeatureCache, data: data_module.YearData, node: str) -> float:
    """Return the 99th percentile of a substation's cleaned demand up to the cut, in megawatts."""
    site = cache.sites[node]
    rows = site.present & ~site.switching_mask & np.isfinite(site.y)
    rows[data.cut_position + 1 :] = False
    return float(np.percentile(site.y[rows], 99)) * data.capacity_mw[node] / 100.0


def thresholds(*, cache: FeatureCache, data: data_module.YearData) -> dict[str, float]:
    """Return the big-event threshold of every gate receiver in megawatts."""
    return {
        node: THRESHOLD_SHARE * p99_demand_mw(cache=cache, data=data, node=node)
        for node in common.GATE_SITES
    }


def is_big(*, transfer: dict, cache: FeatureCache, data: data_module.YearData, limit: dict) -> bool:
    """Return whether a transfer's mean moved megawatts reach the receiver's threshold for 3 h."""
    onset, end = int(transfer["onset"]), int(transfer["end"])
    scale = {n: data.capacity_mw[n] / data_module.POINTS_PER_CAPACITY for n in data.sites}
    power = {
        node: cache.sites[node].y[onset:end] * scale[node]
        for node in (transfer["donor"], transfer["receiver"])
    }
    moved = inject.moved_mw(
        transfer=transfer,
        donor_power_mw=power[transfer["donor"]],
        receiver_power_mw=power[transfer["receiver"]],
    )
    long_enough = (end - onset) / 2.0 >= MIN_BIG_HOURS
    return bool(long_enough and float(np.mean(moved)) >= limit[transfer["receiver"]])


def main() -> None:
    """Print the thresholds and the shares of each injected size that clear them."""
    cache = common.load_cache()
    data = data_module.load_year(cache=cache, cut_name="tuning", arm="a0")
    limit = thresholds(cache=cache, data=data)
    print({node: round(value, 2) for node, value in limit.items()})
    transfers = inject.draw_set(set_name="T", year_name="tuning", cache=cache)
    flags = [
        is_big(transfer=row, cache=cache, data=data, limit=limit)
        for row in transfers.iter_rows(named=True)
    ]
    frame = transfers.with_columns(big=pl.Series(flags))
    shares = frame.group_by("size").agg(pl.col("big").mean()).sort("size")
    print(shares)
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    THRESHOLD_PATH.write_text(
        pl.DataFrame({"node": list(limit), "threshold_mw": list(limit.values())}).write_json()
    )
    labelled = (
        pl.read_parquet(
            common.ML_EXPERIMENTS_DATA_DIR
            / "00041-sw2-0a-labeller-gate"
            / "real"
            / "labelled_set.parquet"
        )
        .filter(~pl.col("is_sentinel"), pl.col("kind") == "real_operation")
        .filter(pl.col("visibility").is_in(["conserved", "one_sided"]))
        .filter(pl.col("moved_mw_p50").is_finite())
    )
    print(
        "labelled events: share with moved_mw_p50 over the smallest gate threshold",
        float(np.mean(labelled["moved_mw_p50"].abs().to_numpy() >= min(limit.values()))),
        "of",
        labelled.height,
    )
    print(labelled.sample(5, seed=131).select("moved_mw_p50", "onset_utc", "end_utc"))


if __name__ == "__main__":
    main()
