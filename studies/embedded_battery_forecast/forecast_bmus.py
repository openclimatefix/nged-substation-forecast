"""Part A, rungs A1 to A5: forecast the 35 testbed battery BMUs out of fold.

Fits every arm of `forecast_arms.arm_definitions` for every testbed battery at the issue times
asked for, and saves each arm's out-of-fold quantiles and per-row losses under
`fits_as_written/<setting>/<issue>/`. An arm whose file exists is skipped, so an interrupted run
resumes.
Rung A1 is `clim`, rung A2 the two conformal baselines, rung A3 the XGBoost quantile model at
`DA-early` and `DA-late`, rung A4 the `own_fpn` and `no_neighbour` arms at `ID-1h`, and rung A5 the
`neighbour_fpn` and `neighbour_fpn_shuffled` arms at `ID-1h`.

The default variant, `as_written`, writes `fits_as_written/`: it links the earlier fits that the
published-price rule leaves unchanged from `fits/` and refits the arms that use the actual price.
With `FIT_VARIANT=idle_dropped` it refits only the batteries that have an idle lead-in, without the
lead-in rows, into `fits_idle_dropped/`, and links every other fit there from `fits_as_written/`.

Run: `OMP_NUM_THREADS=2 uv run python
studies/embedded_battery_forecast/forecast_bmus.py <primary|sensitivity> [DA-early] [DA-late]
[ID-1h]`. `DA-early` needs `forecast_price_model.py`
to have run first.
"""

import sys

from forecast_fit import FIT_VARIANT, SettingType, link_unchanged_fits, run_in_pool
from forecast_runner import ISSUES, batteries_with_idle_lead_in, battery_job, testbed_ids
from studies.battery_forecast import IssueType


def main() -> None:
    """Fit Part A at the setting and issue times named on the command line."""
    arguments = sys.argv[1:]
    setting: SettingType = "sensitivity" if "sensitivity" in arguments else "primary"
    issues: list[IssueType] = [i for i in ISSUES if i in arguments] or list(ISSUES)
    batteries = batteries_with_idle_lead_in() if FIT_VARIANT == "idle_dropped" else testbed_ids()
    tasks = [(b, issue, setting) for issue in issues for b in batteries]
    print(f"Linked {link_unchanged_fits(affected=batteries)} unchanged fits.")
    run_in_pool(function=battery_job, tasks=tasks)


if __name__ == "__main__":
    main()
