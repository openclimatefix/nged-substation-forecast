"""Run the arms of one battery at one issue time and setting, from the saved input frames.

Shared by `forecast_bmus.py` (Part A) and `forecast_nged_battery_a.py` (Part B).
"""

import polars as pl
from census import BMU_LIST_PATH
from forecast_arms import arm_definitions
from forecast_fit import SettingType, run_job, with_model_price
from forecast_inputs import Battery, scored_arms
from forecast_price_model import model_price_by_fold, plain_forecast
from studies.battery_forecast import IssueType
from studies.sources import EMBEDDED_BATTERY_FORECAST_INPUTS_DIR

NGED_BATTERY_A_FILE_ID = "nged_battery_a"
ISSUES: tuple[IssueType, ...] = ("DA-early", "DA-late", "ID-1h")


def testbed_ids() -> list[str]:
    """Return the testbed batteries' identifiers, from the saved `ID-1h` input files."""
    files = EMBEDDED_BATTERY_FORECAST_INPUTS_DIR.glob("ID-1h__E_*.parquet")
    return sorted(f.stem.removeprefix("ID-1h__") for f in files)


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
