# ruff: noqa: E501
"""Price the scored-year arms with the node-only curtailment rule (copied from experiment 169).

The five decision substations are treated as nodes. The scaled limit is f times the node's largest
flow before the scored year (f = 0.9 headline, 0.75 and 0.5 sweep); flows are experiment 169's
tuning and replicate readings. Savings are against the manual heuristic, in GBP over the scored
year for the five substations, at GBP 2,000 per MWh-equivalent (experiment 169's price).

    uv run --no-sync python ml-experiments/00251-*/e00251_curtailment.py
"""

import datetime as dt
from typing import Final

import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_month_totals
from studies.dp_features import default_input_paths, load_or_build_feature_cache
from studies.node_curtailment import (
    NodeCurtailmentAssumptions,
    NodeTable,
    avoidable_cost,
    month_total_arrays,
    node_curtailment,
    scale_to_megawatts,
    scaled_limits,
    table_with_scaled_limits,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR

import e00251_common as common  # isort: skip

FRACTIONS: Final[tuple[float, ...]] = (0.9, 0.75, 0.5)
PRICE: Final[float] = 2000.0
SCORED_START: Final[dt.datetime] = dt.datetime(2025, 7, 2, tzinfo=dt.UTC)
E169_PREDS: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00169-calibration-combined-fixes-and-ensemble-spread" / "work" / "dp_preds"
)  # fmt: skip
CACHE_DIR: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00106-differentiable-physics-4-all-years-drift-and-recent-level" / "cache"
)  # fmt: skip
ARMS: Final[tuple[str, ...]] = (
    "manual_heuristic",
    "bg2",
    "climatology",
    "xgboost",
    "d",
    "arm1",
    "arm2",
    "arm2c",
)


def main() -> None:
    """Score every arm and write the savings against the manual heuristic."""
    cache = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)
    ids = {s: int(cache.sites[s].table.time_series_id) for s in common.DECISION_SITES}
    capacity = {ids[s]: float(cache.sites[s].table.effective_capacity_mw) for s in ids}
    pieces = [
        pl.read_parquet(
            E169_PREDS / f"ref_{period}_{site}.parquet",
            columns=["time_series_id", "valid_time", "actual"],
        )
        for site in common.DECISION_SITES
        for period in ("tuning", "replicate")
    ]
    flows = (
        pl.concat(pieces)
        .filter(pl.col("valid_time") < SCORED_START)
        .unique(subset=["time_series_id", "valid_time"])
        .select(
            pl.col("time_series_id").cast(pl.Int64),
            "valid_time",
            flow_mw=pl.col("actual").cast(pl.Float64)
            / 100.0
            * pl.col("time_series_id").replace_strict(capacity, return_dtype=pl.Float64),
        )
    )
    base = common.ARM_DIR / "scored"
    rows = pl.read_parquet(base / "rows.parquet")
    mask = rows["site"].is_in(list(common.DECISION_SITES)).to_numpy()
    keep = rows.filter(pl.Series(mask)).select(
        pl.col("time_series_id").cast(pl.Int64), "power_fcst_init_time", "valid_time", "seed",
        (pl.col("actual").cast(pl.Float64) / 100.0).alias("actual"),
    )  # fmt: skip
    frames = {}
    for arm in ARMS:
        q = np.load(base / f"{arm}.npy")[mask].astype(np.float64) / 100.0
        frame = keep.with_columns(
            [pl.Series(name, q[:, i]) for i, name in enumerate(common.QUANTILES)]
        )
        frames[arm] = scale_to_megawatts(frame=frame, capacity_mw=capacity)
    frames["perfect"] = frames["bg2"].with_columns(
        *[pl.col("actual").alias(name) for name in common.QUANTILES]
    )
    nodes = pl.DataFrame(
        {
            "node_id": list(ids), "time_series_id": list(ids.values()),
            "node_type": ["BSP"] * len(ids), "parent_node_id": [None] * len(ids),
            "export_sign": [1] * len(ids),
            "sign_source": ["assumed: a positive demand reading is the flow that can breach"] * len(ids),
        },
        schema_overrides={"parent_node_id": pl.String, "export_sign": pl.Int64},
    )  # fmt: skip
    limits = pl.DataFrame(
        schema={"node_id": pl.String, "season": pl.String, "export_limit_mw": pl.Float64,
                "import_limit_mw": pl.Float64, "limit_source": pl.String}
    )  # fmt: skip
    generators = pl.DataFrame(
        schema={"generator_id": pl.Int64, "node_id": pl.String, "capacity_mw": pl.Float64,
                "connection_date": pl.Date, "is_forecast": pl.Boolean}
    )  # fmt: skip
    table = NodeTable(nodes=nodes, limits=limits, generators=generators)
    out: dict = {}
    for fraction in FRACTIONS:
        scaled = scaled_limits(table=table, flows=flows, cutoff=SCORED_START, fraction=fraction)
        result = node_curtailment(
            table=table_with_scaled_limits(table=table, scaled=scaled, fraction=fraction),
            node_forecasts=frames,
            generator_forecasts={},
            assumptions=NodeCurtailmentAssumptions(use_generators=False),
        )
        avoid = avoidable_cost(per_arm=result.per_arm, perfect_arm="perfect")
        column = "energy_proxy_mwh_equivalent"
        sums = {}
        for arm in ARMS:
            sums[arm], counts = month_total_arrays(
                month_totals=result.month_totals, arm=arm, column=column
            )
        rows_out = {}
        for arm in ARMS:
            diff = sums["manual_heuristic"] - sums[arm]
            low, high = bootstrap_month_totals(month_sums=diff, month_counts=counts)
            total = float(counts.sum()) * PRICE
            rows_out[arm] = {
                "saving_vs_manual_gbp": float(diff.sum(axis=1).mean()) * PRICE,
                "low_gbp": low * total,
                "high_gbp": high * total,
            }
        out[str(fraction)] = {"savings": rows_out, "avoidable": avoid.to_dicts(),
                              "excluded": result.excluded.to_dicts()}  # fmt: skip
        print(fraction, {a: round(v["saving_vs_manual_gbp"]) for a, v in rows_out.items()})
    common.write_json_atomic(path=common.REPORT_DIR / "curtailment_scored.json", content=out)


if __name__ == "__main__":
    main()
