"""Split each arm's avoidable cost into unneeded cuts, energy lost on real events, and breach.

The scored rows are split in two by the metered flow: rows where the metered export is beyond the
limit (an event) and rows where it is not. `node_curtailment` runs on each part. On a row without an
event the breach is zero by definition, so everything an arm is charged there is energy lost to a
cut that the metered flow did not need. On an event row the arm is charged energy lost and breach.
The perfect forecast is charged nothing on a row without an event.

The script also compares the census with the scored rows: how many of the metered half-hours and of
the events have a stored forecast, and what share of the excess energy they hold.

    uv run python ml-experiments/00084-*/e00084_split.py
"""

from collections.abc import Mapping

import e00084_inputs as inputs
import e00084_report as report
import polars as pl
from studies.node_curtailment import (
    HALF_HOUR_H,
    NodeCurtailmentAssumptions,
    NodeTable,
    node_curtailment,
    scaled_limits,
    season_expression,
    table_with_scaled_limits,
)

KEYS: tuple[str, ...] = ("time_series_id", "power_fcst_init_time", "valid_time")


def _event_keys(*, frame: pl.DataFrame, table: NodeTable) -> pl.DataFrame:
    """Return the node rows whose metered export is beyond the limit of the row's season."""
    nodes = table.nodes.select("node_id", "time_series_id", "export_sign")
    limits = table.limits.select("node_id", "season", limit_mw="export_limit_mw")
    return (
        frame.filter(
            (pl.col("valid_time") - pl.col("power_fcst_init_time")) >= pl.duration(hours=12),
            (pl.col("valid_time") - pl.col("power_fcst_init_time")) < pl.duration(hours=36),
        )
        .select(*KEYS, "actual")
        .unique(subset=KEYS)
        .join(nodes, on="time_series_id")
        .with_columns(season=season_expression(time_column="valid_time"))
        .join(limits, on=["node_id", "season"])
        .filter(pl.col("export_sign") * pl.col("actual") > pl.col("limit_mw"))
        .select(*KEYS)
    )


def _subset(
    *,
    frames: Mapping[str, pl.DataFrame],
    keys: pl.DataFrame,
    table: NodeTable,
    node: bool,
    keep: bool,
) -> dict[str, pl.DataFrame]:
    """Keep (or drop) the rows of every arm's frame that match `keys`.

    A generator row matches when the row of its node, at the same run and valid time, does.
    """
    how = "semi" if keep else "anti"
    if node:
        return {arm: frame.join(keys, on=list(KEYS), how=how) for arm, frame in frames.items()}
    mapping = table.generators.join(table.nodes, on="node_id").select(
        pl.col("generator_id").alias("time_series_id"),
        pl.col("time_series_id").alias("node_series"),
    )
    node_keys = keys.rename({"time_series_id": "node_series"})
    on = ["node_series", "power_fcst_init_time", "valid_time"]
    return {
        arm: frame.join(mapping, on="time_series_id")
        .join(node_keys, on=on, how=how)
        .drop("node_series")
        for arm, frame in frames.items()
    }


def main() -> None:
    """Print the cost split and the scoring coverage, and save them."""
    capacity_mw = inputs.read_capacity_mw()
    forecast_ids = set(
        inputs.read_arm_frames(names=("gen_monthly",))["gen_monthly"]["time_series_id"]
        .unique()
        .to_list()
    )
    scoring = inputs.scoring_table(
        table=inputs.build_pilot_table(capacity_mw=capacity_mw, forecast_ids=forecast_ids)
    )
    flows = inputs.read_flows(table=scoring)
    node_frames, generator_frames = report.build_arms(table=scoring, capacity_mw=capacity_mw)
    assumptions = NodeCurtailmentAssumptions()
    price = assumptions.headline_price_gbp_per_mwh
    ratio = assumptions.breach_ratio
    rows = []
    coverage = []
    scored_flows = flows.filter(pl.col("valid_time") >= inputs.SCORED_START).join(
        scoring.nodes.select("node_id", "time_series_id", "export_sign"), on="time_series_id"
    )
    for fraction in (0.9, 0.75, 0.5):
        scaled = scaled_limits(
            table=scoring, flows=flows, cutoff=inputs.SCORED_START, fraction=fraction
        )
        table = table_with_scaled_limits(table=scoring, scaled=scaled, fraction=fraction)
        keys = _event_keys(frame=node_frames[report.BEST_GUESS], table=table)
        parts = {}
        for name, keep in (("events", True), ("no_event", False)):
            parts[name] = node_curtailment(
                table=table,
                node_forecasts=_subset(
                    frames=node_frames, keys=keys, table=table, node=True, keep=keep
                ),
                generator_forecasts=_subset(
                    frames=generator_frames, keys=keys, table=table, node=False, keep=keep
                ),
                assumptions=assumptions,
            )
        totals = {
            name: result.per_arm.filter(pl.col("is_headline_price")).select(
                "arm", "n_node_half_hours", "needed_mwh", "curtailed_mwh", "breach_mwh"
            )
            for name, result in parts.items()
        }
        events = totals["events"].rename(lambda c: c if c == "arm" else f"{c}_events")
        other = totals["no_event"].rename(lambda c: c if c == "arm" else f"{c}_other")
        joined = events.join(other, on="arm")
        perfect = joined.filter(pl.col("arm") == report.PERFECT).row(0, named=True)
        rows.extend(
            {
                "fraction": fraction,
                "arm": row["arm"],
                "unneeded_cuts_gbp": price * row["curtailed_mwh_other"],
                "energy_lost_on_events_gbp": price
                * (row["curtailed_mwh_events"] - perfect["curtailed_mwh_events"]),
                "breach_gbp": price
                * ratio
                * (row["breach_mwh_events"] - perfect["breach_mwh_events"]),
                "perfect_absolute_gbp": price
                * (perfect["curtailed_mwh_events"] + ratio * perfect["breach_mwh_events"]),
            }
            for row in joined.iter_rows(named=True)
        )
        # Coverage: census events and excess energy against what the scored rows hold.
        limits = scaled.select("node_id", limit_mw="export_limit_mw")
        excess = (
            scored_flows.join(limits, on="node_id")
            .with_columns(
                export_mw=pl.col("export_sign") * pl.col("flow_mw"),
            )
            .filter(pl.col("export_mw") > pl.col("limit_mw"))
            .select(
                n_events=pl.len(),
                excess_mwh=((pl.col("export_mw") - pl.col("limit_mw")) * HALF_HOUR_H).sum(),
            )
            .row(0, named=True)
        )
        scored = parts["events"].per_arm.filter(
            pl.col("is_headline_price"), pl.col("arm") == report.PERFECT
        )
        coverage.append(
            {
                "fraction": fraction,
                "metered_half_hours": scored_flows.height,
                "scored_rows": int(
                    parts["events"].per_arm["n_node_half_hours"][0]
                    + parts["no_event"].per_arm["n_node_half_hours"][0]
                ),
                "census_events": excess["n_events"],
                "scored_events": int(scored["n_node_half_hours"][0]),
                "census_excess_mwh": excess["excess_mwh"],
                "scored_excess_mwh": float(scored["needed_mwh"][0]),
            }
        )
    split = pl.DataFrame(rows).with_columns(
        total_avoidable_gbp=pl.col("unneeded_cuts_gbp")
        + pl.col("energy_lost_on_events_gbp")
        + pl.col("breach_gbp")
    )
    cover = pl.DataFrame(coverage).with_columns(
        unscored_excess_share=1.0 - pl.col("scored_excess_mwh") / pl.col("census_excess_mwh")
    )
    split.write_parquet(report.RESULTS_DIR / "cost_split.parquet")
    cover.write_parquet(report.RESULTS_DIR / "coverage.parquet")
    text = "\n\n".join(
        [
            "# Cost split (avoidable GBP at GBP 2,000 per MWh)",
            report._markdown(frame=split.sort("fraction", "arm")),
            "# Coverage of the census by the scored rows",
            report._markdown(frame=cover),
        ]
    )
    (inputs.DATA_DIR / "split.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
