"""Score experiment 48's stored forecasts by what substation-limit curtailment would have cost.

Prints the event census first, then scores six paired node-and-generator arms at the real firm limit
and at three scaled limits, and writes every number the page quotes into `report.md`. The numbers
behind the figures go to `results/` beside it, in the private data folder.

    uv run python ml-experiments/00084-*/e00084_report.py
"""

import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Final

import e00084_inputs as inputs
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_month_totals
from studies.node_curtailment import (
    SCALED_LIMIT_FRACTIONS,
    NodeCurtailmentAssumptions,
    NodeCurtailmentResult,
    NodeTable,
    avoidable_cost,
    capacity_forecast,
    event_census,
    month_total_arrays,
    node_curtailment,
    scaled_limits,
    table_with_scaled_limits,
)

BEST_GUESS: Final[str] = "best_guess"
STACK: Final[str] = "best_guess_stack"
CLIMATOLOGY: Final[str] = "climatology"
PERFECT: Final[str] = "perfect"
DIAGNOSTIC: Final[str] = "best_guess_nodes_actual_generators"
NGED_BASELINE: Final[str] = "nged_baseline_unconfirmed"
ARM_LABELS: Final[Mapping[str, str]] = {
    BEST_GUESS: "Best guess (weekly retrained)",
    STACK: "Best-guess stack",
    CLIMATOLOGY: "Climatology",
    PERFECT: "Perfect forecast",
    DIAGNOSTIC: "Best-guess nodes, actual generation",
    NGED_BASELINE: "NGED baseline (unconfirmed)",
}
"""Display names; `NGED_BASELINE` is the maintainer's understanding of NGED's practice."""

SET_NAMES: Final[tuple[str, ...]] = (
    "weekly",
    "frozen_cs",
    "climatology",
    "manual_heuristic",
    "gen_monthly",
    "gen_frozen",
)
QUANTILE_COLUMNS: Final[tuple[str, ...]] = (
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
)
LIMIT_KINDS: Final[tuple[str, ...]] = ("real", *(f"scaled f={f}" for f in SCALED_LIMIT_FRACTIONS))
RESULTS_DIR: Final = inputs.DATA_DIR / "results"
PRICE_COLUMN: Final[str] = "price_gbp_per_mwh"


def _perfect(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return `frame` with every quantile set to the actual value."""
    return frame.with_columns(*(pl.col("actual").alias(q) for q in QUANTILE_COLUMNS))


def build_arms(
    *, table: NodeTable, capacity_mw: Mapping[int, float]
) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
    """Return each arm's node and generator forecast frames, in MW.

    Args:
        table: The scoring table.
        capacity_mw: Effective capacity in MW by series.

    Returns:
        The node forecasts and the generator forecasts, keyed by arm.
    """
    node_series = set(table.nodes["time_series_id"].to_list())
    generator_series = set(table.generators["generator_id"].to_list())
    raw = inputs.read_arm_frames(names=SET_NAMES)
    raw = {n: f.with_columns(pl.col("time_series_id").cast(pl.Int64)) for n, f in raw.items()}

    def nodes(name: str) -> pl.DataFrame:
        return raw[name].filter(pl.col("time_series_id").is_in(node_series))

    def generators(name: str) -> pl.DataFrame:
        return raw[name].filter(pl.col("time_series_id").is_in(generator_series))

    node_frames = inputs.to_megawatts(
        frames={
            BEST_GUESS: nodes("weekly"),
            STACK: nodes("frozen_cs"),
            CLIMATOLOGY: nodes("climatology"),
            NGED_BASELINE: nodes("manual_heuristic"),
        },
        capacity_mw=capacity_mw,
    )
    generator_frames = inputs.to_megawatts(
        frames={
            BEST_GUESS: generators("gen_monthly"),
            STACK: generators("gen_frozen"),
            CLIMATOLOGY: generators("climatology"),
        },
        capacity_mw=capacity_mw,
    )
    node_frames[PERFECT] = _perfect(frame=node_frames[BEST_GUESS])
    generator_frames[PERFECT] = _perfect(frame=generator_frames[BEST_GUESS])
    node_frames[DIAGNOSTIC] = node_frames[BEST_GUESS]
    generator_frames[DIAGNOSTIC] = generator_frames[PERFECT]
    generator_frames[NGED_BASELINE] = capacity_forecast(
        generator_frame=generator_frames[BEST_GUESS], table=table
    )
    capacities = pl.DataFrame(
        {
            "time_series_id": table.generators["generator_id"],
            "capacity_mw": table.generators["capacity_mw"],
        }
    )
    generator_frames = {
        arm: _clipped_to_capacity(frame=frame, capacities=capacities, arm=arm)
        for arm, frame in generator_frames.items()
    }
    return node_frames, generator_frames


def _clipped_to_capacity(
    *, frame: pl.DataFrame, capacities: pl.DataFrame, arm: str
) -> pl.DataFrame:
    """Cap each generator forecast quantile at the generator's capacity, leaving `actual` alone.

    A fitted forecast, and the perfect forecast of an output above the effective capacity, can sit
    above the capacity, which `node_curtailment` refuses. Metered output above capacity is real
    and stays in `actual`. The number of rows changed is printed.
    """
    joined = frame.join(capacities, on="time_series_id", how="left", maintain_order="left")
    n_changed = int(joined.select((pl.col("q50") > pl.col("capacity_mw")).sum()).item())
    print(f"arm {arm}: {n_changed} of {joined.height} generator rows had q50 above capacity")
    return joined.with_columns(
        *(pl.min_horizontal(q, "capacity_mw").alias(q) for q in QUANTILE_COLUMNS)
    ).drop("capacity_mw")


def interval_table(*, result: NodeCurtailmentResult, price: float) -> pl.DataFrame:
    """Return each arm's cost and avoidable cost in GBP with month-resampled 95% intervals.

    Args:
        result: A scored result that contains the perfect arm.
        price: The price in GBP per MWh.

    Returns:
        One row per arm: `cost_gbp`, `cost_low_gbp`, `cost_high_gbp`, `avoidable_gbp`,
        `avoidable_low_gbp`, and `avoidable_high_gbp`.
    """
    column = "energy_proxy_mwh_equivalent"
    perfect_sums, counts = month_total_arrays(
        month_totals=result.month_totals, arm=PERFECT, column=column
    )
    total_rows = float(counts.sum())
    rows = []
    for arm in sorted(result.month_totals["arm"].unique()):
        sums, arm_counts = month_total_arrays(
            month_totals=result.month_totals, arm=arm, column=column
        )
        if not np.array_equal(arm_counts, counts):
            msg = f"arm {arm} does not share the perfect arm's rows"
            raise ValueError(msg)
        cost = float(sums.sum(axis=1).mean()) * price
        low, high = bootstrap_month_totals(month_sums=sums, month_counts=counts)
        avoid_low, avoid_high = bootstrap_month_totals(
            month_sums=sums - perfect_sums, month_counts=counts
        )
        avoidable = float((sums - perfect_sums).sum(axis=1).mean()) * price
        rows.append(
            {
                "arm": arm,
                "cost_gbp": cost,
                "cost_low_gbp": low * total_rows * price,
                "cost_high_gbp": high * total_rows * price,
                "avoidable_gbp": avoidable,
                "avoidable_low_gbp": avoid_low * total_rows * price,
                "avoidable_high_gbp": avoid_high * total_rows * price,
            }
        )
    return pl.DataFrame(rows)


CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    (BEST_GUESS, NGED_BASELINE),
    (BEST_GUESS, CLIMATOLOGY),
    (STACK, BEST_GUESS),
)
"""Pairs `(arm, reference)`: the page reports each arm's cost minus its reference's."""


def contrast_table(*, result: NodeCurtailmentResult, price: float) -> pl.DataFrame:
    """Return each pair's paired cost difference in GBP with a month-resampled 95% interval."""
    column = "energy_proxy_mwh_equivalent"
    rows = []
    for arm, reference in CONTRASTS:
        sums, counts = month_total_arrays(month_totals=result.month_totals, arm=arm, column=column)
        ref_sums, _ = month_total_arrays(
            month_totals=result.month_totals, arm=reference, column=column
        )
        low, high = bootstrap_month_totals(month_sums=sums - ref_sums, month_counts=counts)
        total = float(counts.sum()) * price
        rows.append(
            {
                "contrast": f"{arm} minus {reference}",
                "difference_gbp": float((sums - ref_sums).sum(axis=1).mean()) * price,
                "low_gbp": low * total,
                "high_gbp": high * total,
            }
        )
    return pl.DataFrame(rows)


def _markdown(*, frame: pl.DataFrame) -> str:
    """Render a frame as a pipe table."""
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = [
        "| " + " | ".join(f"{v:,.3f}" if isinstance(v, float) else str(v) for v in row) + " |"
        for row in frame.iter_rows()
    ]
    return "\n".join([header, rule, *body])


def census(*, full_table: NodeTable, flows: pl.DataFrame) -> pl.DataFrame:
    """Count exceedances per node, season, direction and limit, for two periods.

    Args:
        full_table: The table of all four nodes.
        flows: Metered flows up to the holdout.

    Returns:
        The rows of `event_census` with a `period` column.
    """
    scaled = {
        f: scaled_limits(table=full_table, flows=flows, cutoff=inputs.SCORED_START, fraction=f)
        for f in SCALED_LIMIT_FRACTIONS
    }
    periods = {
        "all history": flows,
        "scored year": flows.filter(pl.col("valid_time") >= inputs.SCORED_START),
    }
    return pl.concat(
        event_census(table=full_table, flows=f, scaled=scaled).with_columns(period=pl.lit(name))
        for name, f in periods.items()
    )


def score_all(
    *, scoring: NodeTable, flows: pl.DataFrame, capacity_mw: Mapping[int, float]
) -> dict[str, NodeCurtailmentResult]:
    """Score every arm at the real limit and at each scaled limit."""
    node_frames, generator_frames = build_arms(table=scoring, capacity_mw=capacity_mw)
    tables = {"real": scoring}
    for f in SCALED_LIMIT_FRACTIONS:
        scaled = scaled_limits(table=scoring, flows=flows, cutoff=inputs.SCORED_START, fraction=f)
        tables[f"scaled f={f}"] = table_with_scaled_limits(table=scoring, scaled=scaled, fraction=f)
    assumptions = NodeCurtailmentAssumptions()
    return {
        kind: node_curtailment(
            table=table,
            node_forecasts=node_frames,
            generator_forecasts=generator_frames,
            assumptions=assumptions,
        )
        for kind, table in tables.items()
    }


def main() -> None:
    """Run the census and the scoring, and write the report."""
    capacity_mw = inputs.read_capacity_mw()
    forecast_ids = set(
        inputs.read_arm_frames(names=("gen_monthly",))["gen_monthly"]["time_series_id"]
        .unique()
        .to_list()
    )
    full_table = inputs.build_pilot_table(capacity_mw=capacity_mw, forecast_ids=forecast_ids)
    scoring = inputs.scoring_table(table=full_table)
    inputs.write_private_tables(
        table=full_table,
        notes=(
            "# Lineage of the experiment 84 node and generator tables\n\n"
            "Built by `ml-experiments/00084-*/e00084_inputs.py` from `data/NGED/metadata.parquet`\n"
            "and `data/NGED/substation_hierarchy_2026-10-15/Substations.csv` (NGED, private).\n"
            "`generators.parquet` joins each generator series to a node: never commit it, "
            "and never\n"
            "publish a row. Limits are NGED's published pre-event limits and Technical Limits\n"
            "(Supported by NGED Open Data). Signs come from our own power data.\n"
            "Capacities are experiment 48's effective capacities.\n"
        ),
    )
    flows = inputs.read_flows(table=full_table)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    counts = census(full_table=full_table, flows=flows)
    counts.write_parquet(RESULTS_DIR / "census.parquet")
    flows.join(
        full_table.nodes.filter(pl.col("export_sign").is_not_null()).select(
            "node_id", "time_series_id", "export_sign"
        ),
        on="time_series_id",
    ).with_columns(export_mw=pl.col("export_sign") * pl.col("flow_mw")).select(
        "node_id", "valid_time", "export_mw"
    ).write_parquet(RESULTS_DIR / "exports.parquet")
    pre_scored_maxima = {
        f: scaled_limits(table=full_table, flows=flows, cutoff=inputs.SCORED_START, fraction=f)
        for f in SCALED_LIMIT_FRACTIONS
    }
    limits_frame = pl.concat(
        frame.with_columns(fraction=pl.lit(f)) for f, frame in pre_scored_maxima.items()
    )
    limits_frame.write_parquet(RESULTS_DIR / "scaled_limits.parquet")

    lines = ["# Experiment 84 report", ""]
    lines += ["## Event census (flows from cleaned unflagged power)", ""]
    lines += [
        "Half-hours per node: "
        + str(
            flows.join(full_table.nodes.select("node_id", "time_series_id"), on="time_series_id")
            .group_by("node_id")
            .agg(
                all_history=pl.len(),
                scored_year=(pl.col("valid_time") >= inputs.SCORED_START).sum(),
            )
            .sort("node_id")
            .rows()
        ),
        "",
        "Scaled limits (MW), from flows strictly before 2025-07-02:",
        "",
        _markdown(frame=limits_frame.sort("fraction", "node_id")),
        "",
    ]
    totals = (
        counts.group_by("period", "node_id", "direction", "limit_kind")
        .agg(pl.col("n_exceedances").sum(), pl.col("limit_mw").first())
        .sort("period", "node_id", "direction", "limit_kind")
    )
    lines += [_markdown(frame=totals), ""]
    lines += ["Per season:", "", _markdown(frame=counts.drop("n_half_hours")), ""]

    results = score_all(scoring=scoring, flows=flows, capacity_mw=capacity_mw)
    price = NodeCurtailmentAssumptions().headline_price_gbp_per_mwh
    for kind, result in results.items():
        # An arm that plans no curtailment has no generator pairs, so its DCUSA payment is null; it
        # is a payment of zero.
        result = replace(  # noqa: PLW2901
            result,
            per_arm=result.per_arm.with_columns(pl.col("dcusa_form_gbp").fill_null(0.0)),
        )
        intervals = interval_table(result=result, price=price)
        intervals.with_columns(limit_kind=pl.lit(kind)).write_parquet(
            RESULTS_DIR / f"intervals_{kind.replace(' ', '_').replace('=', '')}.parquet"
        )
        per_arm = avoidable_cost(per_arm=result.per_arm, perfect_arm=PERFECT)
        per_arm.with_columns(limit_kind=pl.lit(kind)).write_parquet(
            RESULTS_DIR / f"avoidable_{kind.replace(' ', '_').replace('=', '')}.parquet"
        )
        result.per_arm.with_columns(limit_kind=pl.lit(kind)).write_parquet(
            RESULTS_DIR / f"absolute_{kind.replace(' ', '_').replace('=', '')}.parquet"
        )
        result.per_node_season.with_columns(limit_kind=pl.lit(kind)).write_parquet(
            RESULTS_DIR / f"node_season_{kind.replace(' ', '_').replace('=', '')}.parquet"
        )
        label = (
            "REFERENCE ONLY: the published thermal limit, a ceiling NGED does not operate to"
            if kind == "real"
            else "CALIBRATION TO HISTORY, NOT A SAVING (scaled limit)"
        )
        lines += [f"## Scoring at the {kind} limit: {label}", ""]
        lines += [
            (
                f"Rows dropped: non-finite {result.n_rows_dropped_nonfinite_forecast}, "
                f"generator-half-hours missing a forecast "
                f"{result.n_generator_rows_missing_forecast}, missing an actual "
                f"{result.n_generator_rows_missing_actual}, "
                f"not in every arm {result.n_rows_dropped_not_in_every_arm}. "
                f"Excluded nodes: {result.excluded.rows()}."
            ),
            "",
            (
                "Absolute cost and avoidable cost at GBP 2,000 per MWh, energy proxy, Curtailment "
                "Limit 0, with month-resampled 95% intervals:"
            ),
            "",
            _markdown(frame=intervals),
            "",
            "Paired cost differences (negative means the first arm is cheaper):",
            "",
            _markdown(frame=contrast_table(result=result, price=price)),
            "",
            "Per price (avoidable cost):",
            "",
            _markdown(
                frame=per_arm.select(
                    "arm",
                    PRICE_COLUMN,
                    "n_node_half_hours",
                    "needed_mwh",
                    "planned_mwh",
                    "curtailed_mwh",
                    "breach_mwh",
                    "energy_proxy_gbp",
                    "dcusa_form_gbp",
                )
            ),
            "",
            "Absolute, per price:",
            "",
            _markdown(
                frame=result.per_arm.select(
                    "arm",
                    PRICE_COLUMN,
                    "energy_proxy_gbp",
                    "dcusa_form_gbp",
                    "energy_proxy_gbp_per_mw_year",
                )
            ),
            "",
            "Per node and season (absolute, headline price):",
            "",
            _markdown(
                frame=result.per_node_season.select(
                    "arm",
                    "node_id",
                    "season",
                    "needed_mwh",
                    "curtailed_mwh",
                    "breach_mwh",
                    "energy_proxy_gbp",
                )
            ),
            "",
        ]
    (RESULTS_DIR / "arm_labels.json").write_text(json.dumps(dict(ARM_LABELS)))
    report = "\n".join(lines)
    (inputs.DATA_DIR / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
