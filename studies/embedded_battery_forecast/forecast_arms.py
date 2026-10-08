"""Which arms run for which battery, issue time, and hyperparameter setting.

Printed into `forecast_report.md` so a reader can check each arm's method and feature recipe
against the plan.
"""

from datetime import UTC, datetime, timedelta
from typing import Final

from forecast_fit import ArmDefinition, SettingType
from forecast_inputs import (
    DAY_AHEAD_ARMS,
    GATE_CLOSURE_ARMS,
    NGED_BATTERY_A_GATE_CLOSURE_ARMS,
    TESTBED_GATE_CLOSURE_ARMS,
    ArmSpec,
    Battery,
)
from studies.battery_forecast import IssueType, issue_time_for, persistence_source_times

SENSITIVITY_ARMS: Final[dict[str, tuple[str, ...]]] = {
    "DA-early": ("price_model", "price_shuffled"),
    "DA-late": ("price_actual", "price_shuffled"),
    "ID-1h": ("no_neighbour", "neighbour_fpn", "neighbour_fpn_shuffled"),
}
"""The recipes refitted at the second setting, for a testbed battery: the arms of the planned
contrasts D1 to D4 (D1 and D2 at `DA-late`, D3 at `DA-early`, D4 at `ID-1h`) and, for D4's
false-alarm rule, `no_neighbour`."""

NGED_BATTERY_A_SENSITIVITY_ARMS: Final[dict[str, tuple[str, ...]]] = {
    "DA-early": ("price_model", "price_shuffled"),
    "DA-late": ("price_actual", "price_shuffled"),
    "ID-1h": ("no_neighbour", "fleet_fpn", "fleet_fpn_shuffled"),
}
"""The same for NGED battery A, which runs the fleet arms at gate closure."""


def recipes(*, battery: Battery, issue: IssueType) -> dict[str, ArmSpec]:
    """Return the feature recipes an XGBoost quantile model is fitted with.

    Args:
        battery: The target battery.
        issue: The issue type.

    Returns:
        The day-ahead price sources at `DA-early` (with the model price) and `DA-late` (without
        it), and the gate-closure recipes the battery can run at `ID-1h`.
    """
    if issue == "DA-early":
        return dict(DAY_AHEAD_ARMS)
    if issue == "DA-late":
        return {n: a for n, a in DAY_AHEAD_ARMS.items() if a.price_source != "model"}
    names = TESTBED_GATE_CLOSURE_ARMS if battery.has_fpn else NGED_BATTERY_A_GATE_CLOSURE_ARMS
    chosen = {name: GATE_CLOSURE_ARMS[name] for name in names}
    if battery.has_fpn and battery.lead_party is not None:
        chosen["neighbour_fpn_same_party"] = GATE_CLOSURE_ARMS["neighbour_fpn_same_party"]
    return chosen


def arm_definitions(
    *, battery: Battery, issue: IssueType, setting: SettingType
) -> list[ArmDefinition]:
    """Return the arms to run for one battery, issue time, and setting.

    Args:
        battery: The target battery.
        issue: The issue type.
        setting: `primary` runs every arm. `sensitivity` runs `xgb_quantile` on the recipes of the
            planned contrasts only.

    Returns:
        The arms, baselines first.
    """
    available = recipes(battery=battery, issue=issue)
    if setting == "sensitivity":
        table = SENSITIVITY_ARMS if battery.has_fpn else NGED_BATTERY_A_SENSITIVITY_ARMS
        return [
            ArmDefinition(name=f"xgb_quantile__{name}", method="xgb_quantile", spec=available[name])
            for name in table[issue]
        ]
    arms = [
        ArmDefinition(name="clim", method="clim", spec=None),
        ArmDefinition(name="persistence_conformal", method="persistence_conformal", spec=None),
    ]
    if issue == "ID-1h":
        arms.append(
            ArmDefinition(
                name="rank_conformal__no_neighbour",
                method="rank_conformal",
                spec=available["no_neighbour"],
            )
        )
    else:
        arms += [
            ArmDefinition(name=f"rank_conformal__{name}", method="rank_conformal", spec=spec)
            for name, spec in available.items()
        ]
    arms += [
        ArmDefinition(name=f"xgb_quantile__{name}", method="xgb_quantile", spec=spec)
        for name, spec in available.items()
    ]
    return arms


def issue_cutoff_lines() -> list[str]:
    """Return report lines showing the issue-time cut-offs for one example target half-hour.

    The example is the half-hour starting 12:00 UTC on Tuesday 10 March 2026. The lines give the
    issue time, the source half-hour of the persistence value, and the last day the climatology
    window can contain, as the code computes them.
    """
    target = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
    lines = [
        f"Example target half-hour: {target:%Y-%m-%d %H:%M} UTC (a Tuesday).",
        "",
        "| Issue type | Issue time (UTC) | Persistence read from | Last day in climatology |",
        "|---|---|---|---|",
    ]
    for issue in ("DA-early", "DA-late", "ID-1h"):
        issued = issue_time_for(target_start=target, issue=issue)
        (source,) = persistence_source_times(target_times=[target], issue=issue)
        last_day = issued.date() - timedelta(days=1)
        lines.append(
            f"| {issue} | {issued:%Y-%m-%d %H:%M} | {source:%Y-%m-%d %H:%M} | {last_day} |"
        )
    return lines
