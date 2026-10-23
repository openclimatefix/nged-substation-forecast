# ruff: noqa: E501
"""Re-score experiment 255's causal rows at f = 0.5 by the quantile path and the member path.

The quantile path is `node_curtailment` on the stored quantile columns. The member path replaces
every row's quantile columns with those that `quantiles_from_members` returns from 101 members drawn
from the row's own stored quantile function (probabilities 0, 0.01, ..., 1, linear between the
stored levels and flat beyond q01 and q99). Both are compared with the avoidable costs and
contrasts that experiment 255 stored, under the tolerances pre-registered in `result.md`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00268-*/e00268_reproduce.py
"""

import json
from collections.abc import Mapping
from typing import Final

import e00268_arms as arms_module
import e00268_inputs as inputs
import e00268_report as report
import numpy as np
import polars as pl
from e00268_arms import H5, H5_MEAN5, PERFECT
from studies.node_curtailment import (
    MEMBER_ROW_KEYS,
    NodeCurtailmentAssumptions,
    NodeCurtailmentResult,
    node_curtailment,
    quantiles_from_members,
    scaled_limits,
    table_with_scaled_limits,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR

FRACTION: Final[float] = 0.5
N_MEMBERS: Final[int] = 101
"""Members at probabilities 0, 0.01, ..., 1, so that linear interpolation between sorted members
returns the stored quantile at every whole-percent level exactly."""
STORED_DIR: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00255-metered-farm-node-curtailment-value" / "results"
)
ARM_TOLERANCE_PERCENT: Final[float] = 1.0
"""The member path's avoidable cost per arm may differ from experiment 255's by this percent."""
CONTRAST_TOLERANCE_PERCENT_POINTS: Final[float] = 1.0
"""Each contrast, as a percent of the baseline's avoidable cost, may differ by this many points."""
QUANTILE_PATH_TOLERANCE_PERCENT: Final[float] = 0.01
"""The quantile path is the same code path, so it must agree to floating-point noise."""


def _level(column: str) -> float:
    return int(column[1:]) / 100.0


def members_from_quantiles(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Draw `N_MEMBERS` members per row from the row's stored quantile function.

    Returns one row per row of `frame` and member: `MEMBER_ROW_KEYS`, `member` (1 to `N_MEMBERS`),
    and `value`. Beyond q01 and q99 the function is flat.
    """
    columns = sorted(c for c in frame.columns if len(c) == 3 and c[0] == "q" and c[1:].isdigit())
    levels = np.array([_level(c) for c in columns])
    stored = frame.select(columns).to_numpy()
    probabilities = np.linspace(0.0, 1.0, N_MEMBERS)
    values = np.empty((stored.shape[0], N_MEMBERS))
    for j, p in enumerate(probabilities):
        clipped = min(max(p, levels[0]), levels[-1])
        upper = int(np.searchsorted(levels, clipped, side="left"))
        if upper == 0 or levels[upper] == clipped:
            values[:, j] = stored[:, upper]
            continue
        weight = (clipped - levels[upper - 1]) / (levels[upper] - levels[upper - 1])
        values[:, j] = (1 - weight) * stored[:, upper - 1] + weight * stored[:, upper]
    keys = frame.select(MEMBER_ROW_KEYS)
    return (
        keys.with_columns(value=pl.Series(values.tolist(), dtype=pl.List(pl.Float64)))
        .explode("value")
        .with_columns(member=pl.int_range(pl.len()).over(MEMBER_ROW_KEYS) + 1)
    )


def through_members(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Replace a frame's quantile columns with those of its drawn members."""
    quantile_columns = [c for c in frame.columns if len(c) == 3 and c[0] == "q" and c[1:].isdigit()]
    reduced = quantiles_from_members(
        members=members_from_quantiles(frame=frame), value_column="value"
    ).drop("n_members")
    return frame.drop(quantile_columns).join(reduced, on=list(MEMBER_ROW_KEYS), how="left")


def sorted_quantiles(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Sort each row's stored quantile columns into increasing order (the rearrangement).

    A set of members cannot have crossing quantiles, so this is the quantile path's equivalent of
    the member path for rows whose stored quantiles cross.
    """
    columns = sorted(c for c in frame.columns if len(c) == 3 and c[0] == "q" and c[1:].isdigit())
    ordered = np.sort(frame.select(columns).to_numpy(), axis=1)
    return frame.with_columns(
        **{c: pl.Series(ordered[:, i], dtype=frame.schema[c]) for i, c in enumerate(columns)}
    )


def _score(*, arms: arms_module.Arms, table: object) -> NodeCurtailmentResult:
    return node_curtailment(
        table=table,  # ty: ignore[invalid-argument-type]
        node_forecasts=arms.node_frames,
        generator_forecasts=arms.generator_frames,
        assumptions=NodeCurtailmentAssumptions(),
    )


def _tables(*, result: NodeCurtailmentResult) -> tuple[pl.DataFrame, pl.DataFrame]:
    intervals = report.interval_table(result=result)
    avoidable = {r["arm"]: r["avoidable_gbp"] for r in intervals.iter_rows(named=True)}
    return intervals, report.contrast_table(result=result, avoidable=avoidable)


def _compare(
    *, name: str, intervals: pl.DataFrame, contrasts: pl.DataFrame, tolerance_percent: float
) -> tuple[dict[str, object], list[str]]:
    stored_intervals = pl.read_parquet(STORED_DIR / "causal_intervals_scaled_f0.5.parquet")
    stored_contrasts = pl.read_parquet(STORED_DIR / "causal_contrasts_scaled_f0.5.parquet")
    arms = intervals.join(
        stored_intervals.select("arm", stored_gbp="avoidable_gbp"), on="arm", how="inner"
    ).filter(pl.col("arm") != PERFECT)
    arms = arms.with_columns(
        percent_difference=100.0
        * (pl.col("avoidable_gbp") - pl.col("stored_gbp"))
        / pl.col("stored_gbp")
    )
    baseline = float(
        stored_intervals.filter(pl.col("arm") == arms_module.CAPACITY_BASELINE)[
            "avoidable_gbp"
        ].item()
    )
    both = contrasts.join(
        stored_contrasts.select(
            "contrast",
            stored=pl.col("difference_gbp"),
            stored_low95=pl.col("low95_gbp"),
            stored_high95=pl.col("high95_gbp"),
        ),
        on="contrast",
    ).with_columns(
        points_of_baseline=100.0 * (pl.col("difference_gbp") - pl.col("stored")) / baseline,
        same_sign=(pl.col("difference_gbp") < 0) == (pl.col("stored") < 0),
        same_interval_verdict=((pl.col("low95_gbp") <= 0) & (pl.col("high95_gbp") >= 0))
        == ((pl.col("stored_low95") <= 0) & (pl.col("stored_high95") >= 0)),
    )
    worst_arm = float(arms["percent_difference"].abs().max())
    worst_contrast = float(both["points_of_baseline"].abs().max())
    lines = [
        f"## {name}",
        "",
        report._markdown(
            frame=arms.select("arm", "avoidable_gbp", "stored_gbp", "percent_difference")
        ),
        "",
        report._markdown(
            frame=both.select(
                "contrast",
                "difference_gbp",
                "stored",
                "points_of_baseline",
                "same_sign",
                "same_interval_verdict",
            )
        ),
        "",
        f"Largest arm difference: {worst_arm:.4f}% (tolerance {tolerance_percent}%).",
        (
            f"Largest contrast difference: {worst_contrast:.4f} points of the baseline's avoidable cost "
            f"(tolerance {CONTRAST_TOLERANCE_PERCENT_POINTS})."
        ),
        (
            f"All contrasts keep sign: {bool(both['same_sign'].all())}; keep the 95% interval verdict: "
            f"{bool(both['same_interval_verdict'].all())}."
        ),
        "",
    ]
    passed = (
        worst_arm <= tolerance_percent
        and worst_contrast <= CONTRAST_TOLERANCE_PERCENT_POINTS
        and bool(both["same_sign"].all())
        and bool(both["same_interval_verdict"].all())
    )
    return {
        "name": name,
        "passed": passed,
        "worst_arm_percent": worst_arm,
        "worst_contrast_points": worst_contrast,
    }, lines


def main() -> None:
    """Score both paths and write `reproduction.md` and `reproduction.json`."""
    capacity_mw = inputs.read_capacity_mw()
    forecast_ids = set(
        inputs.read_arm_frames(names=("gen_monthly",))["gen_monthly"]["time_series_id"]
        .unique()
        .to_list()
    )
    full_table = inputs.build_pilot_table(capacity_mw=capacity_mw, forecast_ids=forecast_ids)
    scoring = inputs.scoring_table(table=full_table)
    flows = inputs.read_flows(table=full_table)
    forecasts: Mapping[str, pl.DataFrame] = {
        k: v for k, v in inputs.read_farm_forecasts().items() if k != H5_MEAN5
    }
    scaled = scaled_limits(
        table=scoring, flows=flows, cutoff=inputs.SCORED_START, fraction=FRACTION
    )
    table = table_with_scaled_limits(table=scoring, scaled=scaled, fraction=FRACTION)
    arms = arms_module.build_arms(
        table=scoring,
        capacity_mw=capacity_mw,
        pairing="causal",
        forecasts=forecasts,
        reference_arm=H5,
    )
    member_arms = arms_module.Arms(
        node_frames={k: through_members(frame=v) for k, v in arms.node_frames.items()},
        generator_frames={k: through_members(frame=v) for k, v in arms.generator_frames.items()},
        n_node_keys_before=arms.n_node_keys_before,
        n_node_keys_after=arms.n_node_keys_after,
        farm_frames_mw={},
    )
    sorted_arms = arms_module.Arms(
        node_frames={k: sorted_quantiles(frame=v) for k, v in arms.node_frames.items()},
        generator_frames={k: sorted_quantiles(frame=v) for k, v in arms.generator_frames.items()},
        n_node_keys_before=arms.n_node_keys_before,
        n_node_keys_after=arms.n_node_keys_after,
        farm_frames_mw={},
    )
    outcomes = []
    results: dict[str, pl.DataFrame] = {}
    quantile_columns = [c for c in arms.generator_frames[H5].columns if len(c) == 3 and c[0] == "q"]
    member_sorted_arms = arms_module.Arms(
        node_frames={k: through_members(frame=v) for k, v in sorted_arms.node_frames.items()},
        generator_frames={
            k: through_members(frame=v) for k, v in sorted_arms.generator_frames.items()
        },
        n_node_keys_before=arms.n_node_keys_before,
        n_node_keys_after=arms.n_node_keys_after,
        farm_frames_mw={},
    )
    crossing_rows = {
        arm: int(
            (np.diff(frame.select(sorted(quantile_columns)).to_numpy(), axis=1) < 0)
            .any(axis=1)
            .sum()
        )
        for arm, frame in arms.generator_frames.items()
    }
    lines = ["# Experiment 268 reproduction of experiment 255 at f = 0.5, causal rows", ""]
    for name, path_arms, tolerance in (
        ("Quantile path", arms, QUANTILE_PATH_TOLERANCE_PERCENT),
        ("Member path (101 members per row)", member_arms, ARM_TOLERANCE_PERCENT),
        ("Sorted-quantile path (post hoc)", sorted_arms, ARM_TOLERANCE_PERCENT),
        ("Member path on sorted quantiles (post hoc)", member_sorted_arms, ARM_TOLERANCE_PERCENT),
    ):
        intervals, contrasts = _tables(result=_score(arms=path_arms, table=table))
        outcome, block = _compare(
            name=name, intervals=intervals, contrasts=contrasts, tolerance_percent=tolerance
        )
        outcomes.append(outcome)
        lines += block
        results[name] = intervals
    gap = (
        results["Member path on sorted quantiles (post hoc)"]
        .join(results["Sorted-quantile path (post hoc)"], on="arm", suffix="_sorted")
        .with_columns(
            percent_difference=100.0
            * (pl.col("avoidable_gbp") - pl.col("avoidable_gbp_sorted"))
            / pl.col("avoidable_gbp_sorted")
        )
        .filter(pl.col("arm") != PERFECT)
    )
    explained = float(gap["percent_difference"].abs().max())
    lines += [
        "## Do crossing stored quantiles explain the member path's difference?",
        "",
        f"Generator rows with crossing stored quantiles, by arm: {crossing_rows}.",
        f"Largest percent difference between the member path and the quantile path when both start from sorted quantiles (no crossing rows): {explained:.4f}%.",
        "",
    ]
    outcomes.append({"name": "member vs sorted", "worst_arm_percent": explained})
    inputs.DATA_DIR.mkdir(parents=True, exist_ok=True)
    (inputs.DATA_DIR / "reproduction.md").write_text("\n".join(lines))
    (inputs.DATA_DIR / "reproduction.json").write_text(json.dumps(outcomes, indent=2))
    print("\n".join(lines))
    print(json.dumps(outcomes, indent=2))


if __name__ == "__main__":
    main()
