# ruff: noqa: E501
"""Curtailment-cost stand-in at D8: demand treated as the flow that must stay under a scaled limit.

D8 never exports, so no export breach can occur. The stand-in scores `studies.node_curtailment` in
node-only mode with D8 as a node whose `export_sign` is +1, so its demand plays the part of the
export and the q95 forecast quantile is the planned-curtailment quantile. The limit is `f` times
the maximum reading from July to September 2025, and the score covers valid times from 2025-10-01.
One point of effective capacity is 0.1 MW.

    uv run --no-sync python ml-experiments/00265-*/e00265_curtail.py
"""

import datetime as dt

import e00265_common as common
import numpy as np
import polars as pl
from studies.node_curtailment import (
    NodeCurtailmentAssumptions,
    NodeTable,
    node_curtailment,
    scaled_limits,
    table_with_scaled_limits,
)

CUTOFF = dt.datetime(2025, 10, 1, tzinfo=dt.UTC)
MW_PER_POINT = 0.1
FRACTIONS = (0.9, 0.75, 0.5)
ARMS = ("arm1", "control", "arm2p", "arm3p")


def build_table() -> NodeTable:
    """Return a one-node table; limits are filled in per fraction."""
    nodes = pl.DataFrame({"node_id": ["D8"], "time_series_id": [26], "node_type": ["BSP"], "parent_node_id": [None], "export_sign": [1], "sign_source": ["stand-in: demand as export"]}, schema_overrides={"parent_node_id": pl.String, "export_sign": pl.Int64})  # fmt: skip
    limits = pl.DataFrame({"node_id": ["D8"], "season": ["Winter"], "export_limit_mw": [1.0], "import_limit_mw": [None], "limit_source": ["placeholder"]}, schema_overrides={"import_limit_mw": pl.Float64})  # fmt: skip
    generators = pl.DataFrame(schema={"generator_id": pl.Int64, "node_id": pl.String, "capacity_mw": pl.Float64, "connection_date": pl.Date, "is_forecast": pl.Boolean})  # fmt: skip
    return NodeTable(nodes=nodes, limits=limits, generators=generators)


def node_frame(*, arm: str) -> pl.DataFrame:
    """Return an arm's D8 forecast in MW with the columns the scorer needs."""
    frame = pl.read_parquet(common.PREDS_DIR / arm / "D8.parquet")
    columns = ["q05", "q95", "actual"]
    return frame.select(
        time_series_id=pl.lit(26, dtype=pl.Int64),
        power_fcst_init_time=pl.col("power_fcst_init_time"),
        valid_time=pl.col("valid_time"),
        seed=pl.lit(0, dtype=pl.Int64),
        **{c: pl.col(c).cast(pl.Float64) * MW_PER_POINT for c in columns},
    )


def main() -> None:
    """Score the arms at each scaled limit and print the energy proxy."""
    reference = node_frame(arm="arm1")
    flows = reference.select("time_series_id", "valid_time", flow_mw="actual").unique(
        subset=["valid_time"]
    )
    table = build_table()
    arms = {arm: node_frame(arm=arm) for arm in ARMS}
    arms["perfect"] = reference.with_columns(q05=pl.col("actual"), q95=pl.col("actual"))
    for fraction in FRACTIONS:
        scaled = scaled_limits(table=table, flows=flows, cutoff=CUTOFF, fraction=fraction)
        limit_table = table_with_scaled_limits(table=table, scaled=scaled, fraction=fraction)
        scored = {a: f.filter(pl.col("valid_time") >= CUTOFF) for a, f in arms.items()}
        result = node_curtailment(table=limit_table, node_forecasts=scored, generator_forecasts={}, assumptions=NodeCurtailmentAssumptions(use_generators=False), scaled=None)  # fmt: skip
        print(
            f"\n=== scaled limit f={fraction}: limit {scaled['export_limit_mw'][0]:.2f} MW (from Jul-Sep 2025 maximum)"
        )
        print(result.per_arm.filter(pl.col("is_headline_price")).select("arm", "n_node_half_hours", "needed_mwh", "planned_mwh", "curtailed_mwh", "breach_mwh", "energy_proxy_gbp").to_pandas().to_string(index=False) if False else result.per_arm.filter(pl.col("is_headline_price")).select("arm", "n_node_half_hours", "needed_mwh", "planned_mwh", "curtailed_mwh", "breach_mwh", "energy_proxy_gbp"))  # fmt: skip
        print(
            "excluded:",
            result.excluded.to_dicts(),
            "dropped nonfinite:",
            result.n_rows_dropped_nonfinite_forecast,
        )
        common.write_parquet_atomic(
            path=common.WORK_DIR / "curtail" / f"per_arm_f{fraction}.parquet", frame=result.per_arm
        )


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
