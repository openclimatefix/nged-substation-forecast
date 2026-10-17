"""Apply the curtailment-cost metric and the tail-calibration table to experiment 48's predictions.

Reads experiment 48's stored quantile forecasts of Primary, BSP, and GSP demand (the weekly
retrained XGBoost model, the best-guess stack, the manual heuristic, and the climatology), scores
them with `studies.curtailment_cost` and `studies.tail_calibration`, prints every number the
write-up quotes into `report.md`, and draws the one figure.

    uv run python ml-experiments/00083-*/e00083_report.py
"""

import math
from datetime import UTC, datetime
from typing import Final

import plotting.ocf_theme as ocf  # noqa: F401  (registers the OCF Altair theme)
import polars as pl
from contracts.settings import PROJECT_ROOT
from studies.bootstrap import AbsoluteInterval, BootstrapInterval
from studies.curtailment_cost import (
    PERFECT_FORECAST,
    CurtailmentAssumptions,
    CurtailmentResult,
    align_arms,
    cost_intervals,
    cost_markdown,
    curtailment_cost_for_arms,
    read_arms,
    sweep_assumptions,
)
from studies.sources import ML_EXPERIMENTS_DATA_DIR
from studies.tail_calibration import tail_calibration_markdown, tail_calibration_table

from studies import charts

SOURCE_EXPERIMENT_ID: Final[int] = 48
ROW_SET_ID: Final[str] = "fold_a_daily_gap_v1"
ARMS: Final[tuple[str, ...]] = ("weekly", "frozen_cs", "manual_heuristic", "climatology")
ARM_LABELS: Final[dict[str, str]] = {
    "weekly": "weekly retrained XGBoost",
    "frozen_cs": "best-guess stack",
    "manual_heuristic": "manual heuristic",
    "climatology": "climatology",
}
REFERENCE_ARM: Final[str] = "frozen_cs"
FIXED_TAU: Final[float] = 0.95
"""The highest quantile level these prediction sets deliver."""
CALIBRATION_END: Final[datetime] = datetime(2025, 10, 1, tzinfo=UTC)
EXPERIMENT_ID: Final[int] = 83
SLUG: Final[str] = "curtailment-cost-and-tail-calibration-worked-example"
REPORT_PATH = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "report.md"
FIGURE_PATH = (
    PROJECT_ROOT / "docs" / "ml-experiments" / f"{EXPERIMENT_ID:05d}-{SLUG}" / "figure_1_cost.svg"
)


def main() -> None:
    """Score the four arms, print the report, and draw the figure."""
    raw = read_arms(experiment_id=SOURCE_EXPERIMENT_ID, names=ARMS, row_set_id=ROW_SET_ID)
    groups = raw["weekly"].select("time_series_id", "group").unique()
    raw = {name: frame.drop("site", "group") for name, frame in raw.items()}
    arms = align_arms(arms=raw)
    lines = [
        "# Experiment 83: curtailment cost and tail calibration on experiment 48",
        "",
        f"Rows per arm before alignment: { {n: f.height for n, f in raw.items()} }.",
        (
            "Rows per arm after alignment (shared keys, finite actual): "
            f"{ {n: f.height for n, f in arms.items()} }."
        ),
        "",
    ]
    base = CurtailmentAssumptions(calibration_end=CALIBRATION_END)
    fixed = dict.fromkeys(ARMS, FIXED_TAU)
    result = curtailment_cost_for_arms(arms=arms, assumptions=base, fixed_tau=fixed)
    intervals = cost_intervals(result=result, reference=REFERENCE_ARM)
    lines += [
        (
            f"## A. Every arm procures at tau {FIXED_TAU} (limit: p99 of Jul-Sep 2025 export; "
            "leads 12 to 36 h; breach ratio 1/(1-tau) = 20)"
        ),
        "",
        (
            f"Export-constrained series: {result.per_series['time_series_id'].n_unique()} of "
            f"{result.per_series['time_series_id'].n_unique() + len(result.excluded_series)}."
        ),
        (
            "Cost is GBP per MW of capacity per day, mean over scored rows. Scored rows are runs "
            "initialised on or after 2025-10-01."
        ),
        "",
        cost_markdown(result=result, intervals=intervals),
        (
            "Paired differences from the best-guess stack (negative = cheaper), 95% interval over "
            "whole months:"
        ),
        "",
    ]
    for arm, interval in intervals[1].items():
        lines.append(
            f"- {arm}: {interval['difference']:+.2f} ({interval['lower_95']:+.2f} to "
            f"{interval['upper_95']:+.2f}), {interval['n_months']} months"
        )
    by_group = (
        result.per_series.join(groups, on="time_series_id")
        .group_by("group", "arm")
        .agg(series=pl.len(), gbp_per_day=pl.col("gbp_per_day").mean())
        .pivot(on="arm", index=["group", "series"], values="gbp_per_day")
        .sort("group")
    )
    lines += [
        "",
        "Mean cost per series at tau 0.95, by substation type (series = export-constrained):",
        "",
        str(by_group),
        "",
    ]
    at_five_percent = curtailment_cost_for_arms(arms=arms, assumptions=base, fixed_tau=None).taus
    lines += [
        "## B. Is the 5% unmet target reachable? (tau clamped means no)",
        "",
        str(at_five_percent),
        "",
        "## C. Limit-scale and breach-ratio sweep (tau fixed; rank 1 = cheapest)",
        "",
        str(
            sweep_assumptions(arms=arms, base=base, fixed_tau=fixed).pivot(
                on="arm", index=["limit_scale", "breach_ratio"], values="rank"
            )
        ),
        "",
        "## D. Tail calibration, all 20 demand series pooled, every lead and run from 2025-07-02",
        "",
        tail_calibration_markdown(
            tables={name: tail_calibration_table(frame=frame) for name, frame in arms.items()},
            unit="fraction of capacity",
        ),
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print("\n".join(lines))
    draw_figure(result=result, intervals=intervals)


def draw_figure(
    *,
    result: CurtailmentResult,
    intervals: tuple[dict[str, AbsoluteInterval], dict[str, BootstrapInterval]],
) -> None:
    """Draw the cost of each arm at the same tau, with its 95% interval."""
    absolute = intervals[0]
    rows = pl.DataFrame(
        [
            {
                "label": ARM_LABELS[arm],
                "family": "weather model",
                "value": interval["value"],
                "lower_95": interval["lower_95"],
                "upper_95": interval["upper_95"],
                "reference": arm in ("manual_heuristic", "climatology"),
            }
            for arm, interval in sorted(absolute.items(), key=lambda item: item[1]["value"])
        ]
    )
    cheapest = min(absolute, key=lambda arm: absolute[arm]["value"])
    n_series = result.per_series["time_series_id"].n_unique()
    perfect = result.per_arm.filter(pl.col("arm") == PERFECT_FORECAST)["gbp_per_day"].item()
    upper = math.ceil(max(rows["upper_95"].to_list()) / 10.0) * 10.0
    panel = charts.leaderboard_panel(
        rows=rows,
        x_domain=(0.0, upper),
        x_title="Cost, GBP per MW of capacity per day (smaller is better)",
        keys=False,
    )
    chart = charts.figure(
        panels=[panel],
        number=1,
        title=f"{ARM_LABELS[cheapest]} has the lowest total curtailment cost of the four forecasts",
        subtitle=[
            (
                "Mean total cost per day: curtailing in advance at GBP 100 per MWh, plus GBP 2,000 "
                f"per MWh on energy left unmet, when each forecast procures its {FIXED_TAU:.0%} "
                "quantile of net export above the export limit."
            ),
            (
                f"{n_series} export-constrained demand substations, scored on forecasts made from "
                "October 2025 to June 2026. A perfect forecast would cost "
                f"{perfect:.1f}."
            ),
            "Dot is the mean; line is the 95% interval from resampling whole months.",
        ],
        figure_planning=None,
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    chart.save(FIGURE_PATH)
    print(f"wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()
