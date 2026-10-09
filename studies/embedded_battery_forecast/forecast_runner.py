"""Run the arms of one battery at one issue time and setting, from the saved input frames.

Shared by `forecast_bmus.py` (Part A) and `forecast_nged_battery_a.py` (Part B).
"""

import polars as pl
from census import BMU_LIST_PATH
from forecast_arms import arm_definitions
from forecast_fit import SettingType, run_job, with_model_price
from forecast_inputs import (
    Battery,
    half_hour_grid,
    load_physical_notifications,
    scored_arms,
    shuffled_time,
    slot_columns,
)
from forecast_price_model import model_price_by_fold, plain_forecast
from studies.battery_forecast import IssueType, neighbour_ids, neighbour_statistics
from studies.battery_market import p99_output_mw
from studies.sources import EMBEDDED_BATTERY_FORECAST_INPUTS_DIR

NGED_BATTERY_A_FILE_ID = "nged_battery_a"
ISSUES: tuple[IssueType, ...] = ("DA-early", "DA-late", "ID-1h")


def testbed_ids() -> list[str]:
    """Return the testbed batteries' identifiers, from the saved `ID-1h` input files."""
    files = EMBEDDED_BATTERY_FORECAST_INPUTS_DIR.glob("ID-1h__E_*.parquet")
    return sorted(f.stem.removeprefix("ID-1h__") for f in files)


def batteries_with_idle_lead_in() -> list[str]:
    """Return the testbed batteries whose saved input frame has half-hours outside service."""
    return [
        battery
        for battery in testbed_ids()
        if not pl.read_parquet(
            EMBEDDED_BATTERY_FORECAST_INPUTS_DIR / f"ID-1h__{battery}.parquet",
            columns=["in_service"],
        )["in_service"].all()
    ]


def lead_parties() -> dict[str, str]:
    """Return each testbed battery's lead party, from the reviewed BMU list."""
    listed = pl.read_csv(BMU_LIST_PATH)
    return dict(zip(listed["elexon_bmu_id"], listed["lead_party_name"], strict=True))


def battery_for(*, battery_id: str) -> Battery:
    """Return the facts about a battery that decide its arms (not its output)."""
    if battery_id == NGED_BATTERY_A_FILE_ID:
        return Battery(
            battery_id=battery_id,
            output=pl.DataFrame(),
            p99_mw=1.0,
            lead_party=None,
            has_fpn=False,
        )
    return Battery(
        battery_id=battery_id,
        output=pl.DataFrame(),
        p99_mw=1.0,
        lead_party=lead_parties()[battery_id],
        has_fpn=True,
    )


def with_largest_party_removed(*, base: pl.DataFrame, battery_id: str) -> pl.DataFrame:
    """Add the neighbour slots of a set that leaves out the testbed's largest lead party.

    The set is the target's different-party neighbours less every battery of the lead party with
    the most testbed batteries. For a target of that party, the set equals its different-party
    set, which excludes the party already.

    Args:
        base: The `ID-1h` wide input frame of the battery.
        battery_id: The battery's identifier.

    Returns:
        The frame with `without_largest_party__*` and `without_largest_party_shuffled__*` columns.
    """
    parties = {b: lead_parties()[b] for b in testbed_ids()}
    largest = max(sorted(set(parties.values())), key=list(parties.values()).count)
    neighbours = [
        b for b in neighbour_ids(target=battery_id, lead_party=parties) if parties[b] != largest
    ]
    p99 = {
        b: p99_output_mw(
            frame=pl.read_parquet(
                EMBEDDED_BATTERY_FORECAST_INPUTS_DIR / f"ID-1h__{b}.parquet", columns=["output_mw"]
            ).drop_nulls()
        )
        for b in parties
    }
    stats = neighbour_statistics(
        fpn=load_physical_notifications(), neighbours=neighbours, p99_mw=p99
    )
    slots = slot_columns(
        stats=stats, prefix="without_largest_party", shuffle=shuffled_time(grid=half_hour_grid())
    )
    return base.join(slots, on="time", how="left")


def battery_job(task: tuple[str, IssueType, SettingType]) -> list[str]:
    """Run every arm of one battery, issue time, and setting that has no saved file yet.

    Args:
        task: The battery identifier, the issue type, and the hyperparameter setting.

    Returns:
        The names of the arms that were run.
    """
    battery_id, issue, setting = task
    battery = battery_for(battery_id=battery_id)
    base = pl.read_parquet(EMBEDDED_BATTERY_FORECAST_INPUTS_DIR / f"{issue}__{battery_id}.parquet")
    by_fold = None
    if issue == "DA-early":
        base = with_model_price(base=base, price_model=plain_forecast())
        by_fold = model_price_by_fold()
    if issue == "ID-1h" and battery.lead_party is not None and setting == "primary":
        base = with_largest_party_removed(base=base, battery_id=battery_id)
    arms = [
        arm
        for arm in arm_definitions(battery=battery, issue=issue, setting=setting)
        if arm.spec is None
        or arm.spec.neighbour_slot != "same_party"
        or base["same_party__mean"].null_count() < base.height
    ]
    return run_job(
        battery_id=battery_id,
        issue=issue,
        setting=setting,
        base=base,
        arms=arms,
        scoring_arms=scored_arms(battery=battery, issue=issue, has_model_price=issue == "DA-early"),
        model_price_by_fold=by_fold,
    )
