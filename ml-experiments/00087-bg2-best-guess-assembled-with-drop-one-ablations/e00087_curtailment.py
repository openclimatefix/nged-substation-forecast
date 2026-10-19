"""Price the arms of experiment 00087 by what curtailment would have cost at the BSP and GSP nodes.

Every number this script writes is **calibration to history, not a saving**: it shows how the
forecasts would have scored against the metered flows of 2025-07-01 to 2026-06-30 under an assumed
curtailment rule, not what NGED would have saved. The rule is `studies.node_curtailment` (module
docstring and `docs/roadmap/cost-savings-metrics.md`, Metric 2). The pattern is experiment 00084's
`e00084_inputs.py` and `e00084_report.py`, copied here (an experiment may not import another).

**What is scored.** The four bulk and grid supply point (BSP and GSP) nodes with the forecast
generators beneath them, on the 292 runs of `fold_a_daily_gap_v1`, at forecast leads of 12 to 36
hours. A node whose sign (which way a positive metered reading flows) is unknown is excluded by
the metric and listed. The signs are those of experiment 00084: Boston and Skegness are confirmed
from our own data, Bicker Fen's is unconfirmed (open question 1318 to NGED), and Walpole's is
unknown. `--nodes confirmed` scores only Boston and Skegness; `--nodes all` (the default) also
scores Bicker Fen, labelled unconfirmed. The 16 Primary substations are scored node-only for the
event census alone (`primary_census`): no generator beneath them is forecast.

**Limits.** The headline limit is the scaled limit `L = f x` the node's largest observed export
before 2025-07-01, with `f = 0.9`, swept to 0.75 and 0.5. NGED curtails so that one more fault
still leaves headroom, so the published thermal limit is never the headline: it appears only as a
reference mark and in the event census. (Experiment 00084 cut the history at 2025-07-02; this
script cuts at 2025-07-01, the start of the scored year.)

**Breach ratio.** `r` is the price of a residual breach in units of the price of curtailed energy,
and the forecast quantile that plans the cut is `tau = 1 - 1/r`. The roadmap page fixes `r = 20`
(`tau = 0.95`) and lists no other value, so `r = 20` is the default. The sweep adds `r = 4`
(`tau = 0.75`), `r = 10` (0.90), and `r = 100` (0.99), the values whose `tau` is among the 11
quantile levels. The manual heuristic is scored at the same `tau`.

**Arms.** `nged_method` is NGED's method as the maintainer understands it (unconfirmed): the
`manual_heuristic` forecast of each node, and every generator assumed to run at its capacity
(`node_curtailment.capacity_forecast`). `previous_best`, `frozen`, `monthly`, and `bg2` pair the
BSP and GSP staircase rungs `b_stack`, `b_frozen`, `b_monthly`, and `b_weekly` with the generator
rungs `g_prev_best_era5`, `g_ens_frozen`, `g_ens_frozen`, and `g_ens_weekly` (the generator
staircase has no monthly rung). `bg2_nodes_capacity_generators` and `nged_nodes_bg2_generators`
split BG2's gain between the node forecasts and the generator forecasts. `climatology` and
`xgboost_baseline` use the stored baselines for nodes and generators alike. `perfect` uses the
actual flow and output; the avoidable cost of an arm is its cost minus the perfect arm's.

**Units.** The headline is the energy proxy `price x (curtailed MWh + r x breach MWh)` at GBP
2,000 per MWh, per MW of forecast generator capacity beneath the scored nodes, per day. A day is
one of the 292 scored runs. The annual total multiplies the daily figure by 365. A missing
generator forecast counts as not instructed. Intervals resample whole origin months
(`studies.bootstrap.bootstrap_month_totals`) and are read at 95%; a cost difference is
"statistically significant at the 5% level" when its interval lies wholly on one side of zero.

    uv run python ml-experiments/00087-*/e00087_curtailment.py [--nodes confirmed|all]
    uv run python ml-experiments/00087-*/e00087_curtailment.py --check-inputs
    uv run python ml-experiments/00087-*/e00087_curtailment.py --self-test
"""

import argparse
import datetime as dt
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import e00087_arms as arms
import e00087_assemble as assemble
import e00087_common as common
import e00087_gen_common as gen
import e00087_score as scoring
import numpy as np
import polars as pl
from nged_data.storage import scan_cleaned_power
from studies.bootstrap import bootstrap_month_totals
from studies.node_curtailment import (
    SCALED_LIMIT_FRACTIONS,
    NodeCurtailmentAssumptions,
    NodeCurtailmentResult,
    NodeTable,
    avoidable_cost,
    capacity_forecast,
    month_total_arrays,
    node_curtailment,
    node_table_pilot,
    scale_to_megawatts,
    scaled_limits,
    table_with_scaled_limits,
)

LABEL: Final[str] = "calibration to history, not a saving"
SCORED_START: Final[dt.datetime] = dt.datetime(2025, 7, 1, tzinfo=dt.UTC)
"""The scored year starts here; a scaled limit uses only flows strictly before it."""
SCORED_END: Final[dt.datetime] = common.HOLDOUT
HEADLINE_FRACTION: Final[float] = 0.9
BREACH_RATIOS: Final[tuple[int, ...]] = (4, 10, 20, 100)
"""Breach ratios swept at the headline limit; their `tau` values 0.75, 0.90, 0.95, and 0.99 are
among the 11 quantile levels."""
DEFAULT_BREACH_RATIO: Final[int] = 20
DAYS_PER_YEAR: Final[float] = 365.0
CAPACITY_TOLERANCE: Final[float] = 1e-3
"""Relative tolerance between an arm's `capacity_mw` and the capacity table the node table uses."""

SUBSTATIONS_PATH: Final[Path] = (
    common.REPO_DATA_DIR / "NGED" / "substation_hierarchy_2026-10-15" / "Substations.csv"
)
CLEANED_POWER_PATH: Final[Path] = common.REPO_DATA_DIR / "NGED" / "cleaned_power_time_series.delta"
CAPACITY_PATH: Final[Path] = common.FOLD_A_FRAMES_DIR / "capacity.parquet"
BOSTON: Final[str] = "BOSTON 33kV S STN"
SKEGNESS: Final[str] = "SKEGNESS 33kV S STN"
BICKER_FEN: Final[str] = "BICKER FEN 132kV S STN"
WALPOLE: Final[str] = "WALPOLE 132kV S STN"
EXPORT_LIMITS_MW: Final[Mapping[str, tuple[float, ...]]] = {
    BOSTON: (45.0, 45.1),
    SKEGNESS: (29.8, 31.3),
    BICKER_FEN: (263.5,),
    WALPOLE: (73.4,),
}
"""NGED's published per-transformer export pre-event limits and Technical Limits (Supported by NGED
Open Data), as experiment 00084 used them. The firm limit of a node is the smaller of its pair."""
SIGNS: Final[Mapping[str, tuple[int | None, str]]] = {
    BOSTON: (-1, "our data: a positive reading is import (matches the Primaries beneath it)"),
    SKEGNESS: (-1, "our data: a positive reading is import (matches the Primaries beneath it)"),
    BICKER_FEN: (1, "UNCONFIRMED: a positive reading looks like export; awaiting NGED"),
    WALPOLE: (None, "unknown; awaiting NGED"),
}
CONFIRMED_NODES: Final[tuple[str, ...]] = (BOSTON, SKEGNESS)
NODE_LABELS: Final[Mapping[str, str]] = {
    BOSTON: "Boston",
    SKEGNESS: "Skegness",
    BICKER_FEN: "Bicker Fen (sign unconfirmed)",
    WALPOLE: "Walpole (sign unknown)",
}
NGED_ARM: Final[str] = "nged_method"
PERFECT_ARM: Final[str] = "perfect"
BG2_ARM: Final[str] = "bg2"
ARM_PAIRS: Final[dict[str, tuple[str, str | None]]] = {
    "previous_best": ("b_stack", "g_prev_best_era5"),
    "frozen": ("b_frozen", "g_ens_frozen"),
    "monthly": ("b_monthly", "g_ens_frozen"),
    BG2_ARM: ("b_weekly", "g_ens_weekly"),
    "bg2_nodes_capacity_generators": ("b_weekly", None),
    "nged_nodes_bg2_generators": ("ref_manual_heuristic", "g_ens_weekly"),
    "climatology": ("ref_climatology", "ref_climatology"),
    "xgboost_baseline": ("ref_xgboost_baseline", "ref_xgboost_baseline"),
    NGED_ARM: ("ref_manual_heuristic", None),
}
"""Each arm's node forecast and generator forecast (arm names of `e00087_arms`); a generator
forecast of `None` means every generator runs at its capacity (NGED's method)."""
ARM_LABELS: Final[Mapping[str, str]] = {
    NGED_ARM: "NGED's method (unconfirmed)",
    "previous_best": "Previous best (stack and weather-trained generators)",
    "frozen": "Frozen model",
    "monthly": "Refitted monthly",
    BG2_ARM: "Best guess (refitted weekly)",
    "bg2_nodes_capacity_generators": "Best-guess nodes, generators at capacity",
    "nged_nodes_bg2_generators": "NGED nodes, best-guess generators",
    "climatology": "Climatology",
    "xgboost_baseline": "Production-style XGBoost model",
    PERFECT_ARM: "Perfect forecast",
}
QUANTILE_COLUMNS: Final[tuple[str, ...]] = common.LEVEL_COLUMNS


# --- inputs (copied from experiment 00084's e00084_inputs.py) ------------------------------------


def read_capacity_mw() -> dict[int, float]:
    """Return each series' effective capacity in MW, as experiment 00048 normalised by."""
    capacity = pl.read_parquet(CAPACITY_PATH)
    return {
        int(s): float(c)
        for s, c in zip(capacity["time_series_id"], capacity["effective_capacity_mw"], strict=True)
    }


def build_table(*, capacity_mw: Mapping[int, float], forecast_ids: set[int]) -> NodeTable:
    """Build the node and generator tables of the four nodes from NGED's private files."""
    return node_table_pilot(
        metadata=pl.read_parquet(common.NGED_METADATA_PATH),
        substations=pl.read_csv(SUBSTATIONS_PATH),
        transformer_export_limits_mw=EXPORT_LIMITS_MW,
        signs=SIGNS,
        capacity_mw=capacity_mw,
        forecast_ids=forecast_ids,
    )


def only_nodes(*, table: NodeTable, keep: tuple[str, ...]) -> NodeTable:
    """Return the table restricted to some nodes, with the parent links cleared."""
    node_ids = list(keep)
    return NodeTable(
        nodes=table.nodes.filter(pl.col("node_id").is_in(node_ids)).with_columns(
            parent_node_id=pl.lit(None, dtype=pl.String)
        ),
        limits=table.limits.filter(pl.col("node_id").is_in(node_ids)),
        generators=table.generators.filter(pl.col("node_id").is_in(node_ids)),
    )


def read_flows(*, time_series_ids: list[int]) -> pl.DataFrame:
    """Return the metered flows of some series from cleaned, unflagged power, before the holdout."""
    unflagged = scan_cleaned_power(str(CLEANED_POWER_PATH))
    plain = pl.LazyFrame._from_pyldf(unflagged._ldf)  # strip the Patito model, zero-copy
    return (
        plain.filter(pl.col("time_series_id").is_in(time_series_ids), pl.col("time") < SCORED_END)
        .select(
            pl.col("time_series_id").cast(pl.Int64),
            pl.col("time").alias("valid_time"),
            flow_mw=pl.col("power").cast(pl.Float64),
        )
        .collect()
    )


# --- arms in megawatts -------------------------------------------------------------------------


def read_arm_mw(
    *, arms_dir: Path, name: str, series_ids: set[int], capacity_mw: Mapping[int, float]
) -> pl.DataFrame:
    """Read an arm's rows of some series on the gap runs, converted to MW.

    Args:
        arms_dir: The folder of arm tables.
        name: The arm's name.
        series_ids: The series to keep.
        capacity_mw: The capacity table of the node table.

    Returns:
        The keys, `seed`, `actual`, and the 11 quantiles in MW.

    Raises:
        FileNotFoundError: If the arm has no file.
        ValueError: If the arm's `capacity_mw` differs from `capacity_mw` for a series.
    """
    path = arms_dir / f"{arms.resolve(name=name)}.parquet"
    if not path.exists():
        msg = f"arm {name} has no file {path}"
        raise FileNotFoundError(msg)
    frame = pl.read_parquet(path).filter(
        pl.col("time_series_id").cast(pl.Int64).is_in(series_ids)
        & pl.col("power_fcst_init_time").is_in(common.origins_of_subset(subset="gap"))
        & (pl.col("valid_time") < SCORED_END)
    )
    table = frame.group_by("time_series_id").agg(c=pl.col("capacity_mw").first())
    for series_id, value in table.iter_rows():
        if abs(value - capacity_mw[int(series_id)]) > CAPACITY_TOLERANCE * value:
            msg = f"arm {name}: capacity of a series differs from the node table's"
            raise ValueError(msg)
    keep = [*common.KEYS, "seed", "actual", *QUANTILE_COLUMNS]
    return scale_to_megawatts(
        frame=frame.select(keep).cast({"time_series_id": pl.Int64}), capacity_mw=capacity_mw
    ).drop("capacity_missing")


def perfect_frame(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return `frame` with every quantile set to the actual value."""
    return frame.with_columns(*(pl.col("actual").alias(q) for q in QUANTILE_COLUMNS))


def clipped_to_capacity(*, frame: pl.DataFrame, capacities: pl.DataFrame) -> pl.DataFrame:
    """Cap each generator quantile at the generator's capacity, leaving `actual` alone.

    A fitted forecast, and the perfect forecast of an output above the effective capacity, can sit
    above the capacity, which `node_curtailment` refuses. Metered output above capacity is real
    and stays in `actual`.
    """
    joined = frame.join(capacities, on="time_series_id", how="left", maintain_order="left")
    return joined.with_columns(
        *(pl.min_horizontal(q, "capacity_mw").alias(q) for q in QUANTILE_COLUMNS)
    ).drop("capacity_mw")


def build_arm_frames(
    *, arms_dir: Path, table: NodeTable, capacity_mw: Mapping[int, float]
) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
    """Return each arm's node and generator forecast frames in MW.

    Args:
        arms_dir: The folder of arm tables.
        table: The scoring table.
        capacity_mw: The capacity table.

    Returns:
        The node frames and the generator frames, keyed by the arm names of `ARM_PAIRS`,
        `perfect` included.
    """
    node_ids = {int(i) for i in table.nodes["time_series_id"].to_list()}
    generator_ids = {int(i) for i in table.generators["generator_id"].to_list()}
    capacities = pl.DataFrame(
        {
            "time_series_id": table.generators["generator_id"],
            "capacity_mw": table.generators["capacity_mw"],
        }
    )
    cache: dict[tuple[str, str], pl.DataFrame] = {}

    def frame(*, name: str, side: str) -> pl.DataFrame:
        ids = node_ids if side == "node" else generator_ids
        key = (name, side)
        if key not in cache:
            cache[key] = read_arm_mw(
                arms_dir=arms_dir, name=name, series_ids=ids, capacity_mw=capacity_mw
            )
        return cache[key]

    nodes: dict[str, pl.DataFrame] = {}
    generators: dict[str, pl.DataFrame] = {}
    bg2_generators = frame(name=ARM_PAIRS[BG2_ARM][1] or "", side="generator")
    for arm, (node_arm, generator_arm) in ARM_PAIRS.items():
        nodes[arm] = frame(name=node_arm, side="node")
        generators[arm] = (
            capacity_forecast(generator_frame=bg2_generators, table=table)
            if generator_arm is None
            else frame(name=generator_arm, side="generator")
        )
    nodes[PERFECT_ARM] = perfect_frame(frame=nodes[BG2_ARM])
    generators[PERFECT_ARM] = perfect_frame(frame=bg2_generators)
    return nodes, {
        arm: clipped_to_capacity(frame=value, capacities=capacities)
        for arm, value in generators.items()
    }


# --- scoring -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """One scoring run: a limit, a breach ratio, and the result.

    Attributes:
        limit_kind: `real` or `scaled f=<fraction>`.
        breach_ratio: `r`.
        table: The node table whose limits were scored.
        result: The metric's output.
    """

    limit_kind: str
    breach_ratio: int
    table: NodeTable
    result: NodeCurtailmentResult


def score_scenario(
    *,
    table: NodeTable,
    limit_kind: str,
    nodes: Mapping[str, pl.DataFrame],
    generators: Mapping[str, pl.DataFrame],
    breach_ratio: int,
) -> Scenario:
    """Score every arm at one limit and breach ratio, with a null DCUSA figure read as zero."""
    result = node_curtailment(
        table=table,
        node_forecasts=nodes,
        generator_forecasts=generators,
        assumptions=NodeCurtailmentAssumptions(breach_ratio=breach_ratio),
    )
    filled = replace(
        result, per_arm=result.per_arm.with_columns(pl.col("dcusa_form_gbp").fill_null(0.0))
    )
    return Scenario(limit_kind=limit_kind, breach_ratio=breach_ratio, table=table, result=filled)


def forecast_capacity_mw(*, scenario: Scenario) -> float:
    """Return the capacity of the forecast generators beneath the scored nodes, in MW."""
    scored = scenario.result.excluded["node_id"].to_list()
    nodes = [n for n in scenario.table.nodes["node_id"].to_list() if n not in scored]
    return float(
        scenario.table.generators.filter(pl.col("is_forecast") & pl.col("node_id").is_in(nodes))[
            "capacity_mw"
        ].sum()
    )


def event_counts(*, scenario: Scenario) -> tuple[int, int]:
    """Return the scored export half-hours beyond the scenario's limit, and the half-hours scored.

    The metric's census counts over the node rows it scored, so the count is of events among the
    scored rows (lead 12 to 36 hours, runs of the gap set).
    """
    census = scenario.result.event_census.filter(
        (pl.col("direction") == "export") & (pl.col("limit_kind") == "real")
    )
    return int(census["n_exceedances"].sum()), int(census["n_half_hours"].sum())


def cost_rows(
    *, scenario: Scenario, price: float, n_days: int, label_arms: Mapping[str, str]
) -> pl.DataFrame:
    """Return each arm's cost, its cost against NGED's method, and its avoidable cost.

    Args:
        scenario: A scored scenario that holds the NGED and perfect arms.
        price: The price in GBP per MWh.
        n_days: The scored days (one per run).
        label_arms: Each arm's display name.

    Returns:
        One row per arm: the total energy-proxy cost in GBP, the cost minus NGED's method and minus
        the perfect arm (negative against NGED's method means a lower cost), each per MW of forecast
        generator capacity per day and per year, with 95% month-resampled intervals.
    """
    column = "energy_proxy_mwh_equivalent"
    totals = scenario.result.month_totals
    nged_sums, counts = month_total_arrays(month_totals=totals, arm=NGED_ARM, column=column)
    perfect_sums, _ = month_total_arrays(month_totals=totals, arm=PERFECT_ARM, column=column)
    total_rows = float(counts.sum())
    capacity = forecast_capacity_mw(scenario=scenario)
    scale = price * total_rows / (capacity * n_days) if capacity > 0 else float("nan")
    events, scored = event_counts(scenario=scenario)
    rows = []
    for arm in sorted(totals["arm"].unique()):
        sums, arm_counts = month_total_arrays(month_totals=totals, arm=arm, column=column)
        if not np.array_equal(arm_counts, counts):
            msg = f"arm {arm} does not share the NGED arm's rows"
            raise ValueError(msg)
        row: dict[str, object] = {
            "limit_kind": scenario.limit_kind,
            "breach_ratio": scenario.breach_ratio,
            "arm": arm,
            "label": label_arms.get(arm, arm),
            "price_gbp_per_mwh": price,
            "scored_events": events,
            "scored_half_hours": scored,
            "forecast_capacity_mw": capacity,
            "days": n_days,
            "note": LABEL,
        }
        for name, reference in (
            ("cost", None),
            ("vs_nged", nged_sums),
            ("avoidable", perfect_sums),
        ):
            values = sums if reference is None else sums - reference
            low, high = bootstrap_month_totals(month_sums=values, month_counts=counts)
            mean = float(values.sum(axis=1).mean() / total_rows)
            row[f"{name}_gbp_per_mw_day"] = mean * scale
            row[f"{name}_gbp_per_mw_day_lower"] = low * scale
            row[f"{name}_gbp_per_mw_day_upper"] = high * scale
            row[f"{name}_gbp_per_mw_year"] = mean * scale * DAYS_PER_YEAR
            row[f"{name}_gbp_per_mw_year_lower"] = low * scale * DAYS_PER_YEAR
            row[f"{name}_gbp_per_mw_year_upper"] = high * scale * DAYS_PER_YEAR
        rows.append(row)
    return pl.DataFrame(rows)


def significance_text(*, lower: float, upper: float) -> str:
    """Return whether an interval of a cost difference lies wholly on one side of zero."""
    if upper < 0:
        return "lower than NGED's method, statistically significant at the 5% level"
    if lower > 0:
        return "higher than NGED's method, statistically significant at the 5% level"
    return "not statistically significant at the 5% level"


# --- the Primary substations' event census -------------------------------------------------------


def primary_census(*, flows: pl.DataFrame, series: pl.DataFrame) -> pl.DataFrame:
    """Count, for each Primary, the scored-year half-hours whose export beats a scaled limit.

    A Primary has no published export limit and no forecast generator beneath it, so only the
    scaled limits `f x` the largest export before 2025-07-01 are counted. The sign follows
    experiment 00084's finding for the substations: a positive metered reading is import, so
    export is minus the flow. A Primary that never exported before the scored year has no limit.

    Args:
        flows: Metered flows of the Primary series (`time_series_id`, `valid_time`, `flow_mw`).
        series: `common.series_table()`.

    Returns:
        One row per Primary and fraction with the limit, the scored half-hours, and the count of
        half-hours beyond the limit.
    """
    primaries = series.filter(pl.col("substation_type") == "Primary").select(
        pl.col("time_series_id").cast(pl.Int64), "site"
    )
    export = (
        flows.join(primaries, on="time_series_id")
        .filter(pl.col("flow_mw").is_finite())
        .with_columns(export_mw=-pl.col("flow_mw"))
    )
    history = (
        export.filter(pl.col("valid_time") < SCORED_START)
        .group_by("site")
        .agg(max_export_mw=pl.col("export_mw").max())
    )
    scored = export.filter(pl.col("valid_time") >= SCORED_START)
    rows = []
    for fraction in SCALED_LIMIT_FRACTIONS:
        limits = history.filter(pl.col("max_export_mw") > 0).with_columns(
            limit_mw=fraction * pl.col("max_export_mw")
        )
        counted = (
            scored.join(limits, on="site")
            .group_by("site")
            .agg(
                half_hours=pl.len(),
                events=(pl.col("export_mw") > pl.col("limit_mw")).sum(),
                limit_mw=pl.col("limit_mw").first(),
            )
        )
        rows.append(counted.with_columns(fraction=pl.lit(fraction)))
    return pl.concat(rows).sort("fraction", "site")


# --- the full run --------------------------------------------------------------------------------


def markdown(*, frame: pl.DataFrame, digits: int = 3) -> str:
    """Render a frame as a pipe table."""
    return scoring.markdown(frame=frame, digits=digits)


def run(*, arms_dir: Path, results_dir: Path, nodes_choice: str) -> None:
    """Build the inputs, score every scenario, and write the tables and the report.

    Args:
        arms_dir: The folder of arm tables.
        results_dir: Where the tables and `curtailment_report.txt` go.
        nodes_choice: `confirmed` for Boston and Skegness, or `all` for the four nodes.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    capacity_mw = read_capacity_mw()
    forecast_ids = {
        int(i)
        for i in pl.read_parquet(arms_dir / "g_ens_weekly.parquet", columns=["time_series_id"])[
            "time_series_id"
        ].unique()
    }
    full = build_table(capacity_mw=capacity_mw, forecast_ids=forecast_ids)
    table = full if nodes_choice == "all" else only_nodes(table=full, keep=CONFIRMED_NODES)
    series = common.series_table()
    primary_ids = [
        int(i) for i in series.filter(pl.col("substation_type") == "Primary")["time_series_id"]
    ]
    flows = read_flows(
        time_series_ids=[*[int(i) for i in full.nodes["time_series_id"]], *primary_ids]
    )
    node_flows = flows.filter(
        pl.col("time_series_id").is_in(full.nodes["time_series_id"].to_list())
    )
    nodes, generators = build_arm_frames(arms_dir=arms_dir, table=table, capacity_mw=capacity_mw)
    n_days = len(common.origins_of_subset(subset="gap"))
    lines = [
        "# Experiment 00087 curtailment report",
        "",
        (
            f"All figures: {LABEL}. Nodes scored: {nodes_choice}. Energy proxy at GBP 2,000 per "
            "MWh, Curtailment Limit 0, leads 12 to 36 hours, 292 runs."
        ),
        "",
    ]
    census = pl.concat(
        [scaled_census(table=full, flows=node_flows, fraction=f) for f in SCALED_LIMIT_FRACTIONS]
    )
    census.write_parquet(results_dir / "curtailment_census.parquet")
    lines += [
        "## Event census at the nodes (metered flow, scored year, all half-hours)",
        "",
        markdown(frame=census),
        "",
    ]
    primary = primary_census(flows=flows, series=series)
    primary.write_parquet(results_dir / "primary_census.parquet")
    totals = (
        primary.group_by("fraction")
        .agg(events=pl.col("events").sum(), half_hours=pl.col("half_hours").sum(), series=pl.len())
        .sort("fraction", descending=True)
    )
    lines += [
        "## Event census at the 16 Primaries (node-only; scaled limits; no generator scored)",
        "",
        markdown(frame=totals),
        "",
    ]
    scenarios: list[Scenario] = []
    real = score_scenario(
        table=table,
        limit_kind="real",
        nodes=nodes,
        generators=generators,
        breach_ratio=DEFAULT_BREACH_RATIO,
    )
    scenarios.append(real)
    for fraction in SCALED_LIMIT_FRACTIONS:
        scaled = scaled_limits(
            table=table, flows=node_flows, cutoff=SCORED_START, fraction=fraction
        )
        scaled_table = table_with_scaled_limits(table=table, scaled=scaled, fraction=fraction)
        ratios = BREACH_RATIOS if fraction == HEADLINE_FRACTION else (DEFAULT_BREACH_RATIO,)
        scenarios.extend(
            score_scenario(
                table=scaled_table,
                limit_kind=f"scaled f={fraction}",
                nodes=nodes,
                generators=generators,
                breach_ratio=ratio,
            )
            for ratio in ratios
        )
    price = NodeCurtailmentAssumptions().headline_price_gbp_per_mwh
    frames = []
    for scenario in scenarios:
        costs = cost_rows(scenario=scenario, price=price, n_days=n_days, label_arms=ARM_LABELS)
        frames.append(costs)
        per_arm = avoidable_cost(per_arm=scenario.result.per_arm, perfect_arm=PERFECT_ARM)
        per_arm.with_columns(
            limit_kind=pl.lit(scenario.limit_kind), breach_ratio=pl.lit(scenario.breach_ratio)
        ).write_parquet(
            results_dir
            / f"curtailment_avoidable_{_slug(scenario.limit_kind)}_r{scenario.breach_ratio}.parquet"
        )
        headline = (
            scenario.limit_kind == f"scaled f={HEADLINE_FRACTION}"
            and scenario.breach_ratio == DEFAULT_BREACH_RATIO
        )
        lines += _scenario_lines(scenario=scenario, costs=costs, headline=headline)
    pl.concat(frames).write_parquet(results_dir / "curtailment_costs.parquet")
    report = "\n".join(lines)
    (results_dir / "curtailment_report.txt").write_text(report)
    print(report)


def _slug(kind: str) -> str:
    return kind.replace(" ", "_").replace("=", "")


def scaled_census(*, table: NodeTable, flows: pl.DataFrame, fraction: float) -> pl.DataFrame:
    """Count the scored-year half-hours beyond the real limit and one scaled limit, per node.

    Args:
        table: The node table.
        flows: Metered flows of the nodes.
        fraction: The scaled-limit fraction.

    Returns:
        One row per node with the number of half-hours in the scored year, and the counts beyond
        the real limit and beyond the scaled limit (export direction).
    """
    nodes = table.nodes.filter(pl.col("export_sign").is_not_null()).select(
        "node_id", "time_series_id", "export_sign"
    )
    exports = (
        flows.filter(pl.col("flow_mw").is_finite() & (pl.col("valid_time") >= SCORED_START))
        .join(nodes, on="time_series_id")
        .with_columns(export_mw=pl.col("export_sign") * pl.col("flow_mw"))
    )
    real = table.limits.group_by("node_id").agg(real_limit_mw=pl.col("export_limit_mw").min())
    scaled = scaled_limits(table=table, flows=flows, cutoff=SCORED_START, fraction=fraction).select(
        "node_id", scaled_limit_mw="export_limit_mw"
    )
    return (
        exports.join(real, on="node_id")
        .join(scaled, on="node_id", how="left")
        .group_by("node_id")
        .agg(
            half_hours=pl.len(),
            real_limit_mw=pl.col("real_limit_mw").first(),
            events_beyond_real=(pl.col("export_mw") > pl.col("real_limit_mw")).sum(),
            scaled_limit_mw=pl.col("scaled_limit_mw").first(),
            events_beyond_scaled=(pl.col("export_mw") > pl.col("scaled_limit_mw")).sum(),
        )
        .with_columns(fraction=pl.lit(fraction))
        .sort("node_id")
    )


def _scenario_lines(*, scenario: Scenario, costs: pl.DataFrame, headline: bool) -> list[str]:
    """Return the report lines of one scenario."""
    events = int(costs["scored_events"][0])
    scored = int(costs["scored_half_hours"][0])
    title = "HEADLINE " if headline else ""
    reference = (
        "REFERENCE ONLY: the published thermal limit, a ceiling NGED does not operate to"
        if scenario.limit_kind == "real"
        else f"{LABEL} (scaled limit)"
    )
    lines = [
        (
            f"## {title}{scenario.limit_kind}, r = {scenario.breach_ratio} "
            f"(tau {1 - 1 / scenario.breach_ratio:.2f}): {reference}"
        ),
        "",
        (
            f"Scored events (export half-hours beyond the limit among {scored:,} scored node "
            f"half-hours): {events}. Excluded nodes: {scenario.result.excluded.rows()}. "
            f"Generator-half-hours missing a forecast: "
            f"{scenario.result.n_generator_rows_missing_forecast}; missing an actual: "
            f"{scenario.result.n_generator_rows_missing_actual}. A scored event count this small "
            "cannot rank forecasts."
        ),
        "",
    ]
    table = costs.select(
        "label",
        pl.col("cost_gbp_per_mw_day").alias("cost per MW per day"),
        pl.col("vs_nged_gbp_per_mw_day").alias("minus NGED per MW per day"),
        pl.col("vs_nged_gbp_per_mw_day_lower").alias("lower"),
        pl.col("vs_nged_gbp_per_mw_day_upper").alias("upper"),
        pl.col("vs_nged_gbp_per_mw_year").alias("per MW per year"),
        pl.col("avoidable_gbp_per_mw_day").alias("avoidable per MW per day"),
    ).sort("minus NGED per MW per day")
    lines += [markdown(frame=table), ""]
    bg2 = costs.filter(pl.col("arm") == BG2_ARM)
    if not bg2.is_empty():
        row = bg2.row(0, named=True)
        verdict = significance_text(
            lower=row["vs_nged_gbp_per_mw_day_lower"], upper=row["vs_nged_gbp_per_mw_day_upper"]
        )
        lines += [
            (
                f"BG2 minus NGED's method: {row['vs_nged_gbp_per_mw_day']:+.2f} GBP per MW per day "
                f"({row['vs_nged_gbp_per_mw_day_lower']:+.2f} to "
                f"{row['vs_nged_gbp_per_mw_day_upper']:+.2f}), "
                f"{row['vs_nged_gbp_per_mw_year']:+.0f} GBP per MW per year pro-rated from "
                f"{row['days']} days; {verdict}. ({LABEL}.)"
            ),
            "",
        ]
    return lines


# --- checks --------------------------------------------------------------------------------------


def check_inputs() -> None:
    """Build the node table from the private files and print only counts, never identifiers."""
    capacity_mw = read_capacity_mw()
    forecast_ids = {int(i) for i in gen.site_table()["time_series_id"]}
    table = build_table(capacity_mw=capacity_mw, forecast_ids=forecast_ids)
    print(
        f"nodes {table.nodes.height}, "
        f"signs known {table.nodes['export_sign'].is_not_null().sum()}, "
        f"generators {table.generators.height}, forecast generators "
        f"{int(table.generators['is_forecast'].sum())}"
    )
    print(table.nodes.select("node_id", "node_type", "export_sign", "sign_source"))
    flows = read_flows(time_series_ids=[int(i) for i in table.nodes["time_series_id"]])
    print(f"node flows: {flows.height:,} half-hours, {flows['time_series_id'].n_unique()} series")
    for fraction in SCALED_LIMIT_FRACTIONS:
        print(scaled_census(table=table, flows=flows, fraction=fraction))


def synthetic_table() -> NodeTable:
    """Return a one-node, two-generator table with a 10 MW export limit, for the self-test."""
    nodes = pl.DataFrame(
        {
            "node_id": ["N1"],
            "time_series_id": [1],
            "node_type": ["BSP"],
            "parent_node_id": [None],
            "export_sign": [1],
            "sign_source": ["synthetic"],
        },
        schema={
            "node_id": pl.String,
            "time_series_id": pl.Int64,
            "node_type": pl.String,
            "parent_node_id": pl.String,
            "export_sign": pl.Int64,
            "sign_source": pl.String,
        },
    )
    limits = pl.DataFrame(
        {
            "node_id": ["N1"] * 4,
            "season": ["Winter", "Intermediate Cool", "Intermediate Warm", "Summer"],
            "export_limit_mw": [10.0] * 4,
            "import_limit_mw": [None] * 4,
            "limit_source": ["synthetic"] * 4,
        },
        schema={
            "node_id": pl.String,
            "season": pl.String,
            "export_limit_mw": pl.Float64,
            "import_limit_mw": pl.Float64,
            "limit_source": pl.String,
        },
    )
    generators = pl.DataFrame(
        {
            "generator_id": [2, 3],
            "node_id": ["N1", "N1"],
            "capacity_mw": [6.0, 8.0],
            "connection_date": [None, None],
            "is_forecast": [True, True],
        },
        schema={
            "generator_id": pl.Int64,
            "node_id": pl.String,
            "capacity_mw": pl.Float64,
            "connection_date": pl.Date,
            "is_forecast": pl.Boolean,
        },
    )
    return NodeTable(nodes=nodes, limits=limits, generators=generators)


def synthetic_frames(
    *, table: NodeTable
) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
    """Return fake arm frames over 160 daily runs (months January to June), lead 12 to 36 hours.

    The node's export is 8 MW plus a daily bump that pushes it to 12 MW on some days. The
    generators run at 3 and 4 MW. The `good` arm forecasts the node nearly exactly, the `bad` arm is
    2 MW too low, and NGED's method (generators at capacity, node from `bad`) over-curtails.
    """
    rng = np.random.default_rng(11)
    rows = []
    start = dt.datetime(2025, 7, 1, 9, tzinfo=dt.UTC)
    for day in range(160):
        origin = start + dt.timedelta(days=day)
        bump = 4.0 if day % 9 == 0 else 0.0
        rows.extend(
            (origin, origin + step * common.STEP, 8.0 + bump + rng.normal(0, 0.2))
            for step in range(24, 72)
        )
    base = pl.DataFrame(
        rows,
        schema={
            "power_fcst_init_time": pl.Datetime("us", "UTC"),
            "valid_time": pl.Datetime("us", "UTC"),
            "actual": pl.Float64,
        },
        orient="row",
    )

    def node(*, offset: float, spread: float) -> pl.DataFrame:
        frame = base.with_columns(
            time_series_id=pl.lit(1, dtype=pl.Int64), seed=pl.lit(0, dtype=pl.Int8)
        )
        return frame.with_columns(
            **{
                q: pl.col("actual") + offset + spread * z
                for q, z in zip(
                    QUANTILE_COLUMNS,
                    (-2.3, -1.6, -1.3, -0.7, -0.25, 0.0, 0.25, 0.7, 1.3, 1.6, 2.3),
                    strict=True,
                )
            }
        )

    def generator(*, series: int, output: float) -> pl.DataFrame:
        frame = base.select("power_fcst_init_time", "valid_time").with_columns(
            time_series_id=pl.lit(series, dtype=pl.Int64),
            seed=pl.lit(0, dtype=pl.Int8),
            actual=pl.lit(output),
        )
        return frame.with_columns(**{q: pl.lit(output) for q in QUANTILE_COLUMNS})

    gens = pl.concat([generator(series=2, output=3.0), generator(series=3, output=4.0)])
    nodes = {
        BG2_ARM: node(offset=0.0, spread=0.4),
        NGED_ARM: node(offset=-2.0, spread=0.4),
        "previous_best": node(offset=-2.0, spread=0.4),
    }
    nodes[PERFECT_ARM] = perfect_frame(frame=nodes[BG2_ARM])
    generators = {
        BG2_ARM: gens,
        NGED_ARM: capacity_forecast(generator_frame=gens, table=table),
        "previous_best": gens,
        PERFECT_ARM: perfect_frame(frame=gens),
    }
    return nodes, generators


def self_test() -> None:
    """Score a synthetic node whose answer is known, and check the cost table's logic."""
    table = synthetic_table()
    nodes, generators = synthetic_frames(table=table)
    scenario = score_scenario(
        table=table,
        limit_kind="real",
        nodes=nodes,
        generators=generators,
        breach_ratio=DEFAULT_BREACH_RATIO,
    )
    costs = cost_rows(scenario=scenario, price=2000.0, n_days=160, label_arms=ARM_LABELS)
    by_arm = {r["arm"]: r for r in costs.iter_rows(named=True)}
    events = by_arm[BG2_ARM]["scored_events"]
    assert events > 0, events
    perfect = by_arm[PERFECT_ARM]
    assert abs(perfect["avoidable_gbp_per_mw_day"]) < 1e-9, perfect
    # An accurate node forecast beats NGED's method (a low node forecast, generators at capacity).
    assert by_arm[BG2_ARM]["vs_nged_gbp_per_mw_day"] < 0, by_arm[BG2_ARM]
    assert by_arm[BG2_ARM]["avoidable_gbp_per_mw_day"] >= -1e-9
    assert abs(by_arm[NGED_ARM]["vs_nged_gbp_per_mw_day"]) < 1e-9
    # The per-year figure is the per-day figure times 365.
    ratio = by_arm[BG2_ARM]["cost_gbp_per_mw_year"] / by_arm[BG2_ARM]["cost_gbp_per_mw_day"]
    assert abs(ratio - DAYS_PER_YEAR) < 1e-6, ratio
    print(
        markdown(
            frame=costs.select(
                "arm",
                "cost_gbp_per_mw_day",
                "vs_nged_gbp_per_mw_day",
                "vs_nged_gbp_per_mw_day_lower",
                "vs_nged_gbp_per_mw_day_upper",
                "scored_events",
            )
        )
    )
    print(f"self-test passed: {events} scored events; the accurate arm beats NGED's method")


def write_mock_arms(*, folder: Path) -> None:
    """Write fake arm tables for the four nodes and nine generators, on real keys and real actuals.

    Each arm's median is the observation plus a bias with noise, so the costs differ by arm; 40 of
    the gap runs are used. This tests the private-data path (the node table, the flows, the scaled
    limits) end to end without any fit. The numbers mean nothing.

    Args:
        folder: Where the fake `arms` folder is written.
    """
    origins = common.origins_of_subset(subset="gap")[::7]
    context = assemble.make_context(smoke=False, store=False, origins=origins, root=folder)
    node_ids = {
        int(r["time_series_id"])
        for r in context.series.iter_rows(named=True)
        if r["substation_type"] in {"BSP", "GSP"}
    }
    generator_ids = {int(i) for i in context.sites["time_series_id"]}
    biases = {
        "b_frozen": (node_ids, 0.04),
        "b_stack": (node_ids, 0.03),
        "b_monthly": (node_ids, 0.02),
        "b_weekly": (node_ids, 0.01),
        "ref_manual_heuristic": (node_ids | generator_ids, 0.05),
        "ref_climatology": (node_ids | generator_ids, 0.08),
        "ref_xgboost_baseline": (node_ids | generator_ids, 0.06),
        "g_prev_best_era5": (generator_ids, 0.03),
        "g_ens_frozen": (generator_ids, 0.02),
        "g_ens_weekly": (generator_ids, 0.01),
    }
    z = np.array([-2.326, -1.645, -1.282, -0.674, -0.253, 0.0, 0.253, 0.674, 1.282, 1.645, 2.326])
    rng = np.random.default_rng(2)
    for name, (ids, bias) in biases.items():
        pieces = []
        for series_id in sorted(ids):
            pieces.extend(
                pl.DataFrame(
                    {
                        "time_series_id": [series_id] * 96,
                        "power_fcst_init_time": [origin] * 96,
                        "valid_time": [origin + (k + 1) * common.STEP for k in range(96)],
                    },
                    schema={
                        "time_series_id": pl.Int32,
                        "power_fcst_init_time": pl.Datetime("us", "UTC"),
                        "valid_time": pl.Datetime("us", "UTC"),
                    },
                )
                for origin in origins
            )
        keys = pl.concat(pieces).with_columns(seed=pl.lit(0).cast(pl.Int8))
        actual = np.nan_to_num(assemble.actual_array(frame=keys, power_of=context.power_of))
        centre = actual + bias + rng.normal(0.0, 0.03, size=actual.size)
        quantiles = centre[:, None] + 0.06 * z
        frame = keys.hstack(
            pl.DataFrame(quantiles.astype(np.float32), schema=list(common.LEVEL_COLUMNS))
        )
        assemble.write_arm(
            ctx=context, name=name, frame=assemble.finalise(ctx=context, frame=frame)
        )


def self_test_private() -> None:
    """Run the full scoring on mock arms and the real private node table, in a scratch folder."""
    with tempfile.TemporaryDirectory(dir=common.WORK_DIR.parent, prefix="selftest_") as name:
        folder = Path(name)
        write_mock_arms(folder=folder)
        run(arms_dir=folder / "arms", results_dir=folder / "results", nodes_choice="all")
        written = sorted(p.name for p in (folder / "results").glob("*.parquet"))
        assert "curtailment_costs.parquet" in written, written
        assert "primary_census.parquet" in written, written
        costs = pl.read_parquet(folder / "results" / "curtailment_costs.parquet")
        assert costs["note"].unique().to_list() == [LABEL]
        print(f"private self-test passed: {len(written)} tables written")


def main() -> None:
    """Run the requested check or the full scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", choices=("confirmed", "all"), default="all")
    parser.add_argument("--arms-dir", type=Path, default=common.ARMS_DIR)
    parser.add_argument("--results-dir", type=Path, default=common.RESULTS_DIR)
    parser.add_argument("--check-inputs", action="store_true", help="build the private tables")
    parser.add_argument("--self-test", action="store_true", help="score a synthetic node")
    parser.add_argument(
        "--self-test-private", action="store_true", help="score mock arms on the private node table"
    )
    args = parser.parse_args()
    if args.check_inputs:
        check_inputs()
    elif args.self_test:
        self_test()
    elif args.self_test_private:
        self_test_private()
    else:
        run(arms_dir=args.arms_dir, results_dir=args.results_dir, nodes_choice=args.nodes)


if __name__ == "__main__":
    main()
