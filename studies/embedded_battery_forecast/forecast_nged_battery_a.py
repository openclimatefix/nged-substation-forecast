"""Part B, rungs B1 and B2: forecast NGED battery A out of fold.

NGED battery A is a fraction of its own 99th-percentile absolute output, so nothing in the saved
files carries its megawatts. It is forecast by the same arms as a testbed battery, except that it
has no Physical Notification of its own and no lead party, so at `ID-1h` the fleet arms
(`fleet_fpn` and `fleet_fpn_shuffled`) stand in for the neighbour arms (rung B2).

Run: `OMP_NUM_THREADS=2 uv run python
studies/embedded_battery_forecast/forecast_nged_battery_a.py <primary|sensitivity>`.
"""

import sys

from forecast_fit import SettingType, run_in_pool
from forecast_runner import ISSUES, NGED_BATTERY_A_FILE_ID, battery_job


def main() -> None:
    """Fit Part B at the setting named on the command line."""
    setting: SettingType = "sensitivity" if "sensitivity" in sys.argv[1:] else "primary"
    run_in_pool(
        function=battery_job,
        tasks=[(NGED_BATTERY_A_FILE_ID, issue, setting) for issue in ISSUES],
        workers=3,
    )


if __name__ == "__main__":
    main()
