"""Tabulate the four runs of the known-answer gate, line by line.

The runs are stored beside the gate (`work/gate/gate_run1.json` to `gate_run4.json`, with run 3's
second pass in `gate.json` at the time). The pre-registered definition of each line is the plan's;
the two post hoc changes are labelled in the table. Writes `gate_runs.json` and `gate_runs.csv`
into the docs folder, so the page quotes the same numbers.

    uv run python ml-experiments/00109-*/e00109_gate_table.py
"""

import json
from pathlib import Path
from typing import Any, Final

import e00109_common as common
import polars as pl

GATE_DIR: Final[Path] = common.WORK_DIR / "gate"
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
RUNS: Final[dict[str, tuple[str, str]]] = {
    "run 1": ("gate_run1.json", "raw heating coefficients; original closed form"),
    "run 2": ("gate_run2.json", "post hoc 1: heating contribution; original closed form"),
    "run 3": ("gate_run3.json", "post hoc 2: closed form with serially correlated noise"),
    "run 4": ("gate_run4.json", "post hoc 3: joint oracle posterior mean"),
    "run 5": ("gate_run5.json", "post hoc 4: oracle prior mean corrected for clipping"),
}
LINES: Final[dict[str, str]] = {
    "a_fleet_mean": "(a) fleet mean",
    "b_shrinkage_on_4_weeks": "(b) shrinkage on 4 June weeks",
    "c_full_history_keeps_own": "(c) full history keeps its own",
    "d_tuning_responds": "(d) tuning rule responds",
    "e_capacity_from_a_short_window": "(e) capacity from a short window",
}


def headline(*, key: str, check: dict[str, Any]) -> str:
    """Return the line's headline numbers as one short string."""
    if key == "a_fleet_mean":
        return (
            f"largest band error {max(check['band_relative_error']):.1%}, "
            f"profile {check['profile_rms_points']:.2f} points"
        )
    if key == "b_shrinkage_on_4_weeks":
        slope = check.get("slope_on_heating_contribution", check.get("slope"))
        reference = check.get(
            "oracle_slope_on_heating_contribution",
            check.get("closed_form_slope", check.get("closed_form_factor")),
        )
        return (
            f"slope {slope:.2f} against {reference:.2f}; heating RMSE {check['rmse']:.2f} "
            f"against {check['oracle_rmse']:.2f}"
        )
    if key == "c_full_history_keeps_own":
        slope = check.get("slope_on_heating_contribution", check.get("slope"))
        return f"slope {slope:.2f}"
    if key == "d_tuning_responds":
        return (
            f"lambda* {check['lambda_base']:g} to {check['lambda_tau_x10']:g} (10x tau) and "
            f"{check['lambda_tau_x0.1']:g} (tau/10)"
        )
    within = check["within_25_percent_of_8"]
    return f"within 25% of 8: 4 weeks {within['4w']}, 3 months {within['3m']}"


def main() -> None:
    """Write the table of runs."""
    rows = []
    for run, (name, definition) in RUNS.items():
        result = json.loads((GATE_DIR / name).read_text())
        for key, label in LINES.items():
            check = result["checks"][key]
            rows.append(
                {
                    "run": run,
                    "definition": definition,
                    "line": label,
                    "passed": bool(check["passed"]),
                    "numbers": headline(key=key, check=check),
                }
            )
    table = pl.DataFrame(rows)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    table.write_csv(DOCS_DIR / "gate_runs.csv")
    print(table.write_csv())


if __name__ == "__main__":
    main()
