# ruff: noqa: E501
"""Score one world at the two bulk supply points: forecast quality by row group, and curtailment cost.

**Every arm shares experiment 48's node forecast and experiment 224's plant forecast of the six
farms; an arm differs only by the change that the injected load shifts make to its forecast of the
affected Primaries.** The rule, the limits, and the rows are experiment 255's causal rows at 0.5
times each node's maximum export before 2025-09-01, fixed from real flows in every world.

Row groups of a node row (cutoff = newest usable half-hour, target = valid time), for the shifts
that touch a Primary under that node: G1 ordinary (target and cutoff outside every shift and the
144 half-hours after it), G2 BEFORE_ALARM (cutoff from the onset to 96 half-hours after it), G3
DURING (cutoff 96 half-hours after the onset or later, target before the end), G4 AFTER (target
within 96 half-hours after the end, cutoff at or after the onset), G5 NEAR (target within 336
half-hours either side of a shift), G0 all rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import e00271_arms as arms_module
import e00271_data as dat
import e00271_harness as h
import e00271_inputs as inputs
import e00271_worlds as worlds
import numpy as np
import polars as pl
from studies.ensemble_scores import tail_weighted_crps
from studies.node_curtailment import (
    NodeCurtailmentAssumptions,
    NodeTable,
    avoidable_cost,
    node_curtailment,
    scaled_limits,
    table_with_scaled_limits,
)

FRACTION: Final[float] = 0.5
GROUPS: Final[tuple[str, ...]] = ("G0", "G1", "G2", "G3", "G4", "G5")
TARGETS: Final[tuple[str, ...]] = ("observed", "nra")
PERFECT: Final[str] = "perfect"
AFTER_HALF_HOURS: Final[int] = 96
ORDINARY_MARGIN: Final[int] = 144
NEAR_HALF_HOURS: Final[int] = 336
QUANTILES: Final[tuple[str, ...]] = inputs.NODE_QUANTILES
LEVELS: Final[np.ndarray] = np.array([0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95])
_EDGES: Final[np.ndarray] = np.concatenate([[0.0], LEVELS, [1.0]])
WEIGHTS: Final[np.ndarray] = (
    (_EDGES[2:] - _EDGES[:-2]) / 2.0 / ((_EDGES[2:] - _EDGES[:-2]) / 2.0).sum()
)
"""Quantile weights proportional to the spacing of the levels."""


@dataclass(frozen=True)
class Context:
    """Everything that does not change between worlds."""

    table: NodeTable
    base_nodes: pl.DataFrame
    generator_frame: pl.DataFrame
    perfect_generators: pl.DataFrame
    node_name: dict[int, str]
    limit_mw: dict[str, float]
    row_index: np.ndarray
    lead: np.ndarray
    valid_pos: np.ndarray
    cutoff_pos: np.ndarray
    node_of_row: np.ndarray
    event_days: dict[str, np.ndarray]


def _positions(*, column: pl.Expr, frame: pl.DataFrame) -> np.ndarray:
    return (
        frame.select(((column - pl.lit(dat.GRID_START)).dt.total_seconds() // 1800).cast(pl.Int64))
        .to_series()
        .to_numpy()
    )


def build_context() -> Context:
    """Read the inputs once: the table, experiment 48's node rows, and the plant generator rows."""
    capacity_mw = inputs.read_capacity_mw()
    forecast_ids = set(
        inputs.read_arm_frames(names=("gen_monthly",))["gen_monthly"]["time_series_id"]
        .unique()
        .to_list()
    )
    full = inputs.build_pilot_table(capacity_mw=capacity_mw, forecast_ids=forecast_ids)
    scoring = inputs.scoring_table(table=full)
    flows = inputs.read_flows(table=full)
    scaled = scaled_limits(
        table=scoring, flows=flows, cutoff=inputs.SCORED_START, fraction=FRACTION
    )
    table = table_with_scaled_limits(table=scoring, scaled=scaled, fraction=FRACTION)
    forecasts = inputs.read_farm_forecasts()
    arms = arms_module.build_arms(
        table=scoring,
        capacity_mw=capacity_mw,
        pairing="causal",
        forecasts={arms_module.PLANT: forecasts[arms_module.PLANT]},
        reference_arm=arms_module.PLANT,
    )
    base = arms.node_frames[arms_module.PLANT].sort(
        "time_series_id", "power_fcst_init_time", "valid_time"
    )
    o06 = pl.col("power_fcst_init_time").dt.truncate("1d").dt.offset_by("6h")
    base = base.with_columns(o06_pos=_positions(column=o06, frame=base))
    node_name = dict(
        zip(table.nodes["time_series_id"].to_list(), table.nodes["node_id"].to_list(), strict=True)
    )
    limit_mw = dict(
        zip(scaled["node_id"].to_list(), scaled["export_limit_mw"].to_list(), strict=True)
    )
    valid_pos = _positions(column=pl.col("valid_time"), frame=base)
    o06_pos = base["o06_pos"].to_numpy()
    row_index = (o06_pos - h.FIRST_ORIGIN) // h.ORIGIN_STEP
    lead = valid_pos - o06_pos
    if (lead < 0).any() or (lead >= h.N_LEADS).any() or (row_index < 0).any():
        msg = "a node row falls outside the harness origins or leads"
        raise ValueError(msg)
    node_of_row = np.array([node_name[t] for t in base["time_series_id"].to_list()])
    event_days = {}
    export = -base["actual"].to_numpy()
    for name in node_name.values():
        chosen = (node_of_row == name) & (export > limit_mw[name])
        days = np.unique((valid_pos[chosen] // 48) * 48)
        event_days[name] = days
    return Context(
        table=table,
        base_nodes=base,
        generator_frame=arms.generator_frames[arms_module.PLANT],
        perfect_generators=arms.generator_frames[arms_module.PERFECT],
        node_name=node_name,
        limit_mw=limit_mw,
        row_index=row_index,
        lead=lead,
        valid_pos=valid_pos,
        cutoff_pos=o06_pos - h.DELAY_HALF_HOURS - 1,
        node_of_row=node_of_row,
        event_days=event_days,
    )


def group_masks(*, ctx: Context, shifts: list[worlds.Shift]) -> dict[str, np.ndarray]:
    """Boolean mask of each row group G0 to G5, for the shifts that touch each row's BSP."""
    bsp = worlds.bsp_of_series()
    n = len(ctx.valid_pos)
    out = {g: np.zeros(n, dtype=bool) for g in GROUPS}
    out["G0"][:] = True
    in_shift = np.zeros(n, dtype=bool)
    for name in set(bsp.values()):
        mine = ctx.node_of_row == name
        v, c = ctx.valid_pos, ctx.cutoff_pos
        for s in shifts:
            if bsp[s.donor] != name and bsp[s.receiver] != name:
                continue
            out["G2"] |= mine & (c >= s.onset) & (c < s.onset + 96)
            out["G3"] |= mine & (c >= s.onset + 96) & (v < s.end)
            out["G4"] |= mine & (v >= s.end) & (v < s.end + AFTER_HALF_HOURS) & (c >= s.onset)
            out["G5"] |= mine & (v >= s.onset - NEAR_HALF_HOURS) & (v < s.end + NEAR_HALF_HOURS)
            in_shift |= mine & (
                ((v >= s.onset) & (v < s.end + ORDINARY_MARGIN))
                | ((c >= s.onset) & (c < s.end + ORDINARY_MARGIN))
            )
    out["G1"] = ~in_shift
    return out


def quality_sums(
    *, ctx: Context, delta: np.ndarray, actual: np.ndarray, masks: dict[str, np.ndarray]
) -> pl.DataFrame:
    """Row count and sums of the median error, quantile score, and twCRPS per row group, one arm.

    The quantile score is twice the mean pinball loss over the nine quantiles of experiment 48;
    twCRPS is the CRPS of the nine quantiles clamped at minus the node's limit (the flow is
    import-positive, so the export tail is the lower tail of the flow).
    """
    q = ctx.base_nodes.select(list(QUANTILES)).to_numpy() + delta[:, None]
    q = np.sort(q, axis=1)
    difference = actual[:, None] - q
    pinball = np.where(difference >= 0, difference * LEVELS, -difference * (1.0 - LEVELS))
    qs = 2.0 * pinball.mean(axis=1)
    median_error = q[:, QUANTILES.index("q50")] - actual
    mean_error = q @ WEIGHTS - actual
    finite = np.isfinite(actual)
    threshold = np.zeros(len(actual))
    for name, limit in ctx.limit_mw.items():
        threshold = np.where(ctx.node_of_row == name, -limit, threshold)
    tw = tail_weighted_crps(
        values=q, weights=WEIGHTS, actual=actual, threshold=threshold, tail="lower"
    )
    rows = []
    for g, raw_mask in masks.items():
        mask = raw_mask & finite
        rows.append(
            {
                "group": g,
                "n": int(mask.sum()),
                "abs_median": float(np.abs(median_error[mask]).sum()),
                "err_median": float(median_error[mask].sum()),
                "err_mean": float(mean_error[mask].sum()),
                "qs": float(qs[mask].sum()),
                "tw": float(tw[mask].sum()),
            }
        )
    return pl.DataFrame(rows)


def score_world(
    *,
    ctx: Context,
    shifts: list[worlds.Shift],
    forecast: dict[str, dict[str, np.ndarray]],
    observed: dict[str, np.ndarray],
    arm_names: tuple[str, ...] = worlds.ARMS,
    targets: tuple[str, ...] = TARGETS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return the curtailment cost and the forecast-quality sums of every arm and target.

    Args:
        ctx: The shared inputs.
        shifts: The world's shifts (empty for the real record).
        forecast: Each arm's forecast change in MW by BSP, from `worlds.node_deltas`.
        observed: The change in each BSP's observed power in MW by BSP, from `worlds.node_deltas`.
        arm_names: The arms to score.
        targets: `observed`, `nra`, or both.

    Returns:
        The cost table (`target`, `group`, `arm`, `energy_proxy_gbp`, `avoidable_gbp`, rows) and
        the quality table (`target`, `arm`, `group`, sums).
    """
    masks = group_masks(ctx=ctx, shifts=shifts)
    base = ctx.base_nodes
    n = base.height
    delta = {}
    for arm in arm_names:
        d = np.zeros(n)
        for name in ctx.node_name.values():
            rows = ctx.node_of_row == name
            if arm in forecast and name in forecast[arm]:
                d[rows] = forecast[arm][name][ctx.row_index[rows], ctx.lead[rows]]
        delta[arm] = d
    obs = np.zeros(n)
    for name in ctx.node_name.values():
        rows = ctx.node_of_row == name
        if name in observed:
            obs[rows] = observed[name][ctx.valid_pos[rows]]
    cost_rows, quality_rows = [], []
    key = ["time_series_id", "power_fcst_init_time", "valid_time", "seed"]
    for target in targets:
        actual = base["actual"].to_numpy() + (obs if target == "observed" else 0.0)
        for group in ("G0", "G5", "G2", "G3", "G4", "G1"):
            keep = masks[group]
            if not keep.any():
                continue
            sub_keys = base.filter(pl.Series(keep)).select(key)
            node_frames = {}
            for arm in arm_names:
                q = base.with_columns(
                    *(pl.col(c) + pl.Series(delta[arm]) for c in QUANTILES),
                    actual=pl.Series(actual),
                ).drop("o06_pos", "capacity_missing", strict=False)
                node_frames[arm] = q.filter(pl.Series(keep))
            perfect = base.with_columns(
                *(pl.Series(actual).alias(c) for c in QUANTILES), actual=pl.Series(actual)
            ).drop("o06_pos", "capacity_missing", strict=False)
            node_frames[PERFECT] = perfect.filter(pl.Series(keep))
            generators = {
                arm: ctx.generator_frame.join(sub_keys, on=[*key[1:3], "seed"], how="semi")
                for arm in arm_names
            }
            generators[PERFECT] = ctx.perfect_generators.join(
                sub_keys, on=[*key[1:3], "seed"], how="semi"
            )
            result = node_curtailment(
                table=ctx.table,
                node_forecasts=node_frames,
                generator_forecasts=generators,
                assumptions=NodeCurtailmentAssumptions(),
            )
            per_arm = avoidable_cost(per_arm=result.per_arm, perfect_arm=PERFECT)
            headline = per_arm.filter(pl.col("is_headline_price"))
            cost = result.per_arm.filter(pl.col("is_headline_price")).select(
                "arm", cost_gbp="energy_proxy_gbp"
            )
            both = headline.select(
                "arm", avoidable_gbp="energy_proxy_gbp", rows="n_node_half_hours"
            ).join(cost, on="arm")
            cost_rows.append(both.with_columns(target=pl.lit(target), group=pl.lit(group)))
        quality_rows.extend(
            quality_sums(ctx=ctx, delta=delta[arm], actual=actual, masks=masks).with_columns(
                target=pl.lit(target), arm=pl.lit(arm)
            )
            for arm in arm_names
        )
    return pl.concat(cost_rows), pl.concat(quality_rows)
