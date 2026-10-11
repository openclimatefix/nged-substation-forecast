"""Choose the plant and the derated months of round 3's exclusion fit, by a stated rule.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1138>. The
third science review suspected that the plant-month dominating the 8 unselected months is a
derated plant in the training data: the plant produced about two thirds of the other plants'
output in one training month and about three quarters in another, and the baseline can memorise a
low month through `day_of_year`. **Every output is post hoc** and sits under
`<output-root>/ens_mean/followups3/`.

**The plant** is the one with the largest mean B0 error in any single month of the 8 unselected
months (the first run's lead-day 1 losses, the rule `report_followups2.py` uses), chosen without
reference to any arm. **The derated months** are the months in which that plant's output, as a
share of the mean of the other five plants' output, is below `DERATED_RATIO_BELOW`. The share is
computed on the hours at which all six plants are scored, capacity-normalised, month by month.
Months in which a plant has no data, or no hour at which all six are scored, have no ratio and are
never chosen. The rule names neither a plant nor a month.

`followup3_plan.py` writes `derated_plan.json` (the plant, the months, and the plant's ratio for
every month) and `output_ratios_ens_mean.parquet` (every plant's ratio in every month). The fit
script and the report read them, so a fit and its report always use the same plan.

Run it with `uv run python studies/lag_features/followup3_plan.py`.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Final

import polars as pl
from build_lag_frame import (
    FULL_SWEEP_LEAD_DAY,
    LAG_FEATURES_DIR,
    output_paths,
    run_suffix,
    write_parquet_atomic,
)
from fit_lag_arms import METRIC
from followup_frames import PRODUCT
from report_lag_features import P2_FIRST_MONTH
from studies.guards import refuse_to_overwrite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("followup3_plan")

DERATED_RATIO_BELOW: Final[float] = 0.8
"""A month is derated when the plant's output is below this share of the other plants' output."""

PLANTS_SCORED: Final[int] = 6
"""The ratio uses only hours at which every plant has a scored row."""


def followup3_dir(*, root: Path) -> Path:
    """Return the folder every round-3 output lives in.

    Args:
        root: The output root.

    Returns:
        `<root>/ens_mean/followups3`.
    """
    return root / PRODUCT / "followups3"


def plan_path(*, root: Path) -> Path:
    """Return the plan's file.

    Args:
        root: The output root.

    Returns:
        `derated_plan.json` under the round-3 folder.
    """
    return followup3_dir(root=root) / "derated_plan.json"


def output_ratios(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return each plant's output as a share of the other plants', by month.

    Args:
        frame: The lead-day 1 frame, with `site`, `time`, `month`, `power_mw` and
            `effective_capacity_mw`.

    Returns:
        `site`, `month`, `hours` (the hours at which all plants are scored), `plant` (the plant's
        mean capacity-normalised output on them), `others` (the mean of the other plants' mean)
        and `ratio`.
    """
    rows = frame.select(
        "site", "time", "month", x=pl.col("power_mw") / pl.col("effective_capacity_mw")
    )
    totals = rows.group_by("time").agg(n=pl.len(), total=pl.col("x").sum())
    joined = rows.join(totals, on="time").filter(pl.col("n") == PLANTS_SCORED)
    return (
        joined.with_columns(others=(pl.col("total") - pl.col("x")) / (pl.col("n") - 1))
        .group_by("site", "month")
        .agg(hours=pl.len(), plant=pl.col("x").mean(), others=pl.col("others").mean())
        .with_columns(ratio=pl.col("plant") / pl.col("others"))
        .sort("site", "month")
    )


def dominant_plant(*, losses: pl.DataFrame) -> str:
    """Return the plant with the largest mean B0 error in a single unselected month.

    Args:
        losses: The first run's lead-day 1 primary B0 losses, with `site`, `month` and the metric.

    Returns:
        The plant's label.
    """
    ranked = (
        losses.filter(pl.col("month") >= P2_FIRST_MONTH)
        .group_by("site", "month")
        .agg(error=pl.col(METRIC).mean())
        .sort("error", "site", "month", descending=[True, False, False])
    )
    return str(ranked["site"][0])


def choose_plan(*, losses: pl.DataFrame, frame: pl.DataFrame) -> dict[str, object]:
    """Choose the plant and its derated months by the stated rule.

    Args:
        losses: The first run's lead-day 1 primary B0 losses.
        frame: The lead-day 1 frame.

    Returns:
        The plant, the months, the threshold and the plant's ratio in every month that has one.

    Raises:
        ValueError: If the plant has no derated month, which would make the fit meaningless.
    """
    site = dominant_plant(losses=losses)
    ratios = output_ratios(frame=frame).filter(pl.col("site") == site).drop_nulls("ratio")
    months = sorted(ratios.filter(pl.col("ratio") < DERATED_RATIO_BELOW)["month"].to_list())
    if not months:
        msg = f"plant {site} has no month below {DERATED_RATIO_BELOW:.0%} of the others' output"
        raise ValueError(msg)
    return {
        "site": site,
        "months": months,
        "threshold": DERATED_RATIO_BELOW,
        "ratio_by_month": dict(
            zip(ratios["month"].to_list(), ratios["ratio"].to_list(), strict=True)
        ),
    }


def main() -> int:
    """Write the plan and the output-ratio table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=LAG_FEATURES_DIR)
    arguments = parser.parse_args()
    root: Path = arguments.output_root
    ratios_path = followup3_dir(root=root) / f"output_ratios_{PRODUCT}.parquet"
    refuse_to_overwrite(paths=[plan_path(root=root), ratios_path])
    suffix = run_suffix(smoke=False)
    losses = (
        pl.scan_parquet(root / PRODUCT / f"losses_{PRODUCT}{suffix}.parquet")
        .filter(
            (pl.col("scope") == "lead1")
            & (pl.col("setting") == "primary")
            & (pl.col("arm") == "B0")
        )
        .select("site", "month", METRIC)
        .collect()
    )
    frame = pl.read_parquet(
        output_paths(root=root, product=PRODUCT, lead_days=(FULL_SWEEP_LEAD_DAY,))["day1"],
        columns=["site", "time", "month", "power_mw", "effective_capacity_mw"],
    )
    plan = choose_plan(losses=losses, frame=frame)
    plan_path(root=root).parent.mkdir(parents=True, exist_ok=True)
    plan_path(root=root).write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    write_parquet_atomic(frame=output_ratios(frame=frame), path=ratios_path)
    sys.stdout.write(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
