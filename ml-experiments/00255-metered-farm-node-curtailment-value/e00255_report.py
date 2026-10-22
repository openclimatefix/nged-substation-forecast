# ruff: noqa: E501
"""Score the six farms' stored forecasts by what substation-limit curtailment would have cost.

Prints the event census first, then scores every arm at the real firm limit and at three scaled
limits, under the causal pairing (the headline) and the valid-time pairing (a sensitivity), and
writes every number the page quotes into `report.md` beside the private results.

    uv run --no-sync python ml-experiments/00255-*/e00255_report.py
"""

import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Final

import e00255_arms as arms_module
import e00255_inputs as inputs
import numpy as np
import polars as pl
from e00255_arms import (
    CAPACITY_BASELINE,
    DIAGNOSTIC,
    H5,
    H5_MEAN5,
    PERFECT,
    PLANT,
    XGBOOST,
    PairingType,
)
from studies.bootstrap import bootstrap_month_totals
from studies.node_curtailment import (
    HOURS_PER_YEAR,
    SCALED_LIMIT_FRACTIONS,
    NodeCurtailmentAssumptions,
    NodeCurtailmentResult,
    NodeTable,
    avoidable_cost,
    event_census,
    month_total_arrays,
    node_curtailment,
    scaled_limits,
    table_with_scaled_limits,
)
from studies.tail_calibration import tail_calibration_markdown, tail_calibration_table

PAIRINGS: Final[tuple[PairingType, ...]] = ("causal", "valid_time")
LIMIT_KINDS: Final[tuple[str, ...]] = ("real", *(f"scaled f={f}" for f in SCALED_LIMIT_FRACTIONS))
HEADLINE_KIND: Final[str] = "scaled f=0.5"
CONTRAST_LEVEL: Final[float] = 99.0
"""95% with a Bonferroni correction over the five planned contrasts."""
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    (PLANT, CAPACITY_BASELINE),
    (XGBOOST, CAPACITY_BASELINE),
    (H5, CAPACITY_BASELINE),
    (H5, XGBOOST),
    (H5, PLANT),
)
"""C1 to C5: `(arm, reference)`; the page reports each arm's cost minus its reference's."""
COLUMN: Final[str] = "energy_proxy_mwh_equivalent"
PRICE: Final[float] = NodeCurtailmentAssumptions().headline_price_gbp_per_mwh
SEED_CHECK_ARMS: Final[tuple[str, ...]] = ("h5_s1", "h5_s2", "h5_s3", "h5_s4")


def _slug(kind: str) -> str:
    return kind.replace(" ", "_").replace("=", "")


def _markdown(*, frame: pl.DataFrame) -> str:
    """Render a frame as a pipe table."""
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = [
        "| " + " | ".join(f"{v:,.3f}" if isinstance(v, float) else str(v) for v in row) + " |"
        for row in frame.iter_rows()
    ]
    return "\n".join([header, rule, *body])


def interval_table(*, result: NodeCurtailmentResult) -> pl.DataFrame:
    """Return each arm's cost and avoidable cost in GBP with month-resampled 95% intervals."""
    perfect_sums, counts = month_total_arrays(
        month_totals=result.month_totals, arm=PERFECT, column=COLUMN
    )
    total_rows = float(counts.sum())
    rows = []
    for arm in sorted(result.month_totals["arm"].unique()):
        sums, arm_counts = month_total_arrays(
            month_totals=result.month_totals, arm=arm, column=COLUMN
        )
        if not np.array_equal(arm_counts, counts):
            msg = f"arm {arm} does not share the perfect arm's rows"
            raise ValueError(msg)
        low, high = bootstrap_month_totals(month_sums=sums - perfect_sums, month_counts=counts)
        rows.append(
            {
                "arm": arm,
                "cost_gbp": float(sums.sum(axis=1).mean()) * PRICE,
                "avoidable_gbp": float((sums - perfect_sums).sum(axis=1).mean()) * PRICE,
                "avoidable_low_gbp": low * total_rows * PRICE,
                "avoidable_high_gbp": high * total_rows * PRICE,
            }
        )
    return pl.DataFrame(rows)


def contrast_table(
    *, result: NodeCurtailmentResult, avoidable: Mapping[str, float]
) -> pl.DataFrame:
    """Return each planned contrast in GBP with a month-resampled interval at `CONTRAST_LEVEL`."""
    rows = []
    for arm, reference in CONTRASTS:
        sums, counts = month_total_arrays(month_totals=result.month_totals, arm=arm, column=COLUMN)
        ref_sums, _ = month_total_arrays(
            month_totals=result.month_totals, arm=reference, column=COLUMN
        )
        low, high = bootstrap_month_totals(
            month_sums=sums - ref_sums, month_counts=counts, level=CONTRAST_LEVEL
        )
        low95, high95 = bootstrap_month_totals(month_sums=sums - ref_sums, month_counts=counts)
        total = float(counts.sum()) * PRICE
        difference = float((sums - ref_sums).sum(axis=1).mean()) * PRICE
        scale = avoidable.get(XGBOOST) or float("nan")
        rows.append(
            {
                "contrast": f"{arm} minus {reference}",
                "difference_gbp": difference,
                "low_gbp": low * total,
                "high_gbp": high * total,
                "low95_gbp": low95 * total,
                "high95_gbp": high95 * total,
                "half_width_gbp": (high - low) * total / 2.0,
                "difference_pct_of_xgboost": 100.0 * difference / scale,
                "low_pct_of_xgboost": 100.0 * low * total / scale,
                "high_pct_of_xgboost": 100.0 * high * total / scale,
            }
        )
    return pl.DataFrame(rows)


def score_calls(
    *,
    scoring: NodeTable,
    flows: pl.DataFrame,
    capacity_mw: Mapping[int, float],
    pairing: PairingType,
    forecasts: Mapping[str, pl.DataFrame],
    seed_frames: Mapping[str, pl.DataFrame],
) -> tuple[dict[str, dict[str, NodeCurtailmentResult]], arms_module.Arms]:
    """Score the main arms, the five-seed mean, and the other seeds at every limit.

    Args:
        scoring: The scoring table (two bulk supply points).
        flows: Metered node flows, for the scaled limits.
        capacity_mw: Effective capacity in MW by series.
        pairing: `causal` or `valid_time`.
        forecasts: Each farm arm's forecast on the common rows, in fractions of capacity.
        seed_frames: The other single seeds of H5, on the same rows.

    Returns:
        The results keyed by call (`main`, `mean5`, `seeds`) and limit kind, and the main call's arms.
    """
    main_forecasts = {k: v for k, v in forecasts.items() if k != H5_MEAN5}
    calls = {
        "main": (main_forecasts, H5),
        "mean5": ({H5_MEAN5: forecasts[H5_MEAN5]}, H5_MEAN5),
        "seeds": (dict(seed_frames), "h5_s1"),
    }
    tables = {"real": scoring}
    for f in SCALED_LIMIT_FRACTIONS:
        scaled = scaled_limits(table=scoring, flows=flows, cutoff=inputs.SCORED_START, fraction=f)
        tables[f"scaled f={f}"] = table_with_scaled_limits(table=scoring, scaled=scaled, fraction=f)
    results: dict[str, dict[str, NodeCurtailmentResult]] = {}
    main_arms = None
    for call, (call_forecasts, reference) in calls.items():
        arms = arms_module.build_arms(
            table=scoring,
            capacity_mw=capacity_mw,
            pairing=pairing,
            forecasts=call_forecasts,
            reference_arm=reference,
        )
        if call == "main":
            main_arms = arms
        results[call] = {
            kind: node_curtailment(
                table=table,
                node_forecasts=arms.node_frames,
                generator_forecasts=arms.generator_frames,
                assumptions=NodeCurtailmentAssumptions(),
            )
            for kind, table in tables.items()
        }
    if main_arms is None:
        msg = "the main call did not run"
        raise RuntimeError(msg)
    return results, main_arms


def _per_mw_year(*, result: NodeCurtailmentResult, scoring: NodeTable) -> tuple[float, float]:
    """Return the forecast capacity (MW) and the scored years that divide a total cost."""
    capacity = float(scoring.generators.filter(pl.col("is_forecast"))["capacity_mw"].sum())
    # Node half-hours over both nodes; each valid time is scored once per node.
    node_half_hours = float(result.per_arm["n_node_half_hours"].cast(pl.Float64).to_numpy().max())
    years = node_half_hours / len(scoring.nodes) * 0.5 / HOURS_PER_YEAR
    return capacity, years


MASSES: Final = np.diff(
    np.concatenate([[0.0], (np.array(inputs.LEVELS[:-1]) + np.array(inputs.LEVELS[1:])) / 2, [1.0]])
)
"""The probability mass each of the 11 quantiles stands for, so that the mean forecast is q @ MASSES."""


def checks_table(*, forecasts: Mapping[str, pl.DataFrame], valid_times: pl.Series) -> pl.DataFrame:
    """Return, per arm, the mean over farms of the median minus mean forecast and the two MAEs.

    Values are in points of capacity (100 times a fraction of capacity), on the daylight farm rows of
    the scored node rows, with equal weight per farm.
    """
    rows = []
    for arm, frame in forecasts.items():
        kept = frame.filter(pl.col("valid_time").is_in(valid_times.implode()))
        quantiles = kept.select(list(inputs.QUANTILES)).to_numpy()
        mean_forecast = quantiles @ MASSES
        actual = kept["actual"].to_numpy()
        per_row = kept.select("farm").with_columns(
            median_minus_mean=pl.Series(100.0 * (quantiles[:, 5] - mean_forecast)),
            mae_median=pl.Series(100.0 * np.abs(actual - quantiles[:, 5])),
            mae_mean=pl.Series(100.0 * np.abs(actual - mean_forecast)),
        )
        by_farm = per_row.group_by("farm").agg(pl.all().exclude("farm").mean())
        rows.append({"arm": arm, **by_farm.drop("farm").mean().row(0, named=True)})
    return pl.DataFrame(rows)


def per_node_table(*, result: NodeCurtailmentResult) -> pl.DataFrame:
    """Return each arm's avoidable cost in GBP by node, a sum over the generators beneath it."""
    totals = result.per_node_season.group_by("arm", "node_id").agg(
        cost_gbp=pl.col("energy_proxy_gbp").sum()
    )
    perfect = totals.filter(pl.col("arm") == PERFECT).select(
        "node_id", perfect_gbp=pl.col("cost_gbp")
    )
    return (
        totals.join(perfect, on="node_id")
        .with_columns(avoidable_gbp=pl.col("cost_gbp") - pl.col("perfect_gbp"))
        .select("arm", "node_id", "avoidable_gbp")
        .sort("node_id", "arm")
    )


def tails_markdown(*, forecasts: Mapping[str, pl.DataFrame], valid_times: pl.Series) -> str:
    """Return the tail calibration of every arm on the daylight farm rows of the scored node rows."""
    tables = {}
    for arm, frame in forecasts.items():
        kept = frame.filter(pl.col("valid_time").is_in(valid_times.implode())).with_columns(
            power_fcst_init_time=pl.col("valid_time").dt.offset_by("-12h").dt.truncate("1d")
        )
        tables[arm] = tail_calibration_table(frame=kept)
    return tail_calibration_markdown(tables=tables, unit="fractions of capacity")


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
    flows = inputs.read_flows(table=full_table)
    inputs.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    inputs.PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    forecasts = inputs.read_farm_forecasts()
    keys = forecasts[H5].select("farm", "valid_time")
    seed_frames = {
        f"h5_s{k}": inputs.seed_forecasts(name="h5_quantile", setting="own_history", seed=k).join(
            keys, on=["farm", "valid_time"], how="inner"
        )
        for k in (1, 2, 3, 4)
    }
    lines = ["# Experiment 255 report", ""]
    scored_flows = flows.filter(pl.col("valid_time") >= inputs.SCORED_START)
    scaled = {
        f: scaled_limits(table=scoring, flows=flows, cutoff=inputs.SCORED_START, fraction=f)
        for f in SCALED_LIMIT_FRACTIONS
    }
    lines += ["## Scaled limits (MW), from flows strictly before 2025-09-01", ""]
    lines += [
        _markdown(
            frame=pl.concat(f.with_columns(fraction=pl.lit(x)) for x, f in scaled.items()).sort(
                "fraction", "node_id"
            )
        ),
        "",
    ]
    full_census = event_census(table=scoring, flows=scored_flows, scaled=scaled)
    lines += [
        "## Event census over all scored-window flows (valid times 2025-09-01 to 2026-06-30)",
        "",
        _markdown(
            frame=full_census.group_by("direction", "limit_kind")
            .agg(pl.col("n_exceedances").sum())
            .sort("direction", "limit_kind")
        ),
        "",
    ]
    for pairing in PAIRINGS:
        results, arms = score_calls(
            scoring=scoring,
            flows=flows,
            capacity_mw=capacity_mw,
            pairing=pairing,
            forecasts=forecasts,
            seed_frames=seed_frames,
        )
        lines += [
            f"## Pairing: {pairing}",
            "",
            (
                f"Node (origin, valid time) keys before the pairing {arms.n_node_keys_before}, "
                f"after {arms.n_node_keys_after}."
            ),
            "",
        ]
        valid_times = arms.node_frames[H5]["valid_time"].unique()
        night = arms.farm_frames_mw[H5].filter(pl.col("night")).unique(["site", "valid_time"])
        all_rows = arms.farm_frames_mw[H5].unique(["site", "valid_time"])
        day_mwh = float(all_rows["actual"].drop_nulls().to_numpy().sum()) * 0.5
        night_mwh = float(night["actual"].drop_nulls().to_numpy().sum()) * 0.5
        lines += [
            (
                f"Metered farm output in half-hours with a forecast of 0 MW (sun under 5 degrees): "
                f"{night_mwh:,.1f} MWh of {day_mwh:,.1f} MWh "
                f"summed over the six farms."
            ),
            "",
            "Tail calibration on the daylight farm rows of the scored node rows:",
            "",
            tails_markdown(forecasts=forecasts, valid_times=valid_times),
            "",
            "Checks on the same rows (points of capacity, mean over farms):",
            "",
            _markdown(frame=checks_table(forecasts=forecasts, valid_times=valid_times)),
            "",
        ]
        for kind in LIMIT_KINDS:
            main = results["main"][kind]
            main = replace(
                main, per_arm=main.per_arm.with_columns(pl.col("dcusa_form_gbp").fill_null(0.0))
            )
            per_arm = avoidable_cost(per_arm=main.per_arm, perfect_arm=PERFECT)
            intervals = interval_table(result=main)
            avoidable = dict(zip(intervals["arm"], intervals["avoidable_gbp"], strict=True))
            capacity, years = _per_mw_year(result=main, scoring=scoring)
            intervals = intervals.with_columns(
                avoidable_gbp_per_mw_year=pl.col("avoidable_gbp") / (capacity * years)
            )
            census_rows = (
                main.event_census.group_by("limit_kind", "direction")
                .agg(pl.col("n_exceedances").sum())
                .sort("limit_kind", "direction")
            )
            contrasts = contrast_table(result=main, avoidable=avoidable)
            extras = []
            for call in ("mean5", "seeds"):
                res = results[call][kind]
                ex = interval_table(result=res).filter(
                    ~pl.col("arm").is_in([PERFECT, DIAGNOSTIC, CAPACITY_BASELINE])
                )
                extras.append(ex)
            for frame_name, frame in (
                ("intervals", intervals),
                ("avoidable", per_arm),
                ("contrasts", contrasts),
                ("extras", pl.concat(extras)),
                ("census", census_rows),
                ("per_node", per_node_table(result=main)),
            ):
                frame.write_parquet(
                    inputs.RESULTS_DIR / f"{pairing}_{frame_name}_{_slug(kind)}.parquet"
                )
            label = (
                "REFERENCE ONLY: the published thermal limit, a ceiling NGED does not operate to"
                if kind == "real"
                else "CALIBRATION TO HISTORY, NOT A SAVING (scaled limit)"
            )
            lines += [
                f"### {pairing}, {kind}: {label}",
                "",
                (
                    f"Rows dropped: non-finite {main.n_rows_dropped_nonfinite_forecast}, "
                    f"generator-half-hours missing a forecast {main.n_generator_rows_missing_forecast}, "
                    f"missing an actual {main.n_generator_rows_missing_actual}, "
                    f"not in every arm {main.n_rows_dropped_not_in_every_arm}. "
                    f"Excluded nodes: {main.excluded.rows()}. Forecast capacity {capacity:.2f} MW, "
                    f"scored years {years:.4f} (node half-hours "
                    f"{main.per_arm['n_node_half_hours'].max():,.0f})."
                ),
                "",
                "Event half-hours on the scored rows:",
                "",
                _markdown(frame=census_rows),
                "",
                (
                    "Cost and avoidable cost at GBP 2,000 per MWh, energy proxy, Curtailment Limit 0, "
                    "with month-resampled 95% intervals on the avoidable cost:"
                ),
                "",
                _markdown(frame=intervals),
                "",
                f"Planned contrasts (negative: first arm cheaper), {CONTRAST_LEVEL:.0f}% intervals:",
                "",
                _markdown(frame=contrasts),
                "",
                "Five-seed mean of H5 and the other single seeds of H5 (own calls on the same rows):",
                "",
                _markdown(frame=pl.concat(extras)),
                "",
                "Avoidable cost by node, GBP at the headline price (a sum over the generators beneath):",
                "",
                _markdown(frame=per_node_table(result=main)),
                "",
                "Avoidable cost per price:",
                "",
                _markdown(
                    frame=per_arm.select(
                        "arm",
                        "price_gbp_per_mwh",
                        "needed_mwh",
                        "planned_mwh",
                        "curtailed_mwh",
                        "breach_mwh",
                        "energy_proxy_gbp",
                        "dcusa_form_gbp",
                    )
                ),
                "",
            ]
    (inputs.RESULTS_DIR / "arm_labels.json").write_text(json.dumps(dict(arms_module.ARM_LABELS)))
    report = "\n".join(lines)
    (inputs.DATA_DIR / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
