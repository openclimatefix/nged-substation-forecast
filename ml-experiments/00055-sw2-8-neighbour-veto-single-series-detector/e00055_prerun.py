"""Before any recall is scored: how many unexplained odd-month alarms does the veto remove?

The plan's predicted fall in the matched threshold comes from these counts. No injected transfer is
read here.

    uv run python ml-experiments/00055-*/e00055_prerun.py
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
from e00055_common import TUNING_MONTHS, load_panel, near_log_masks, stamp_months
from e00055_detectors import alarms as raw_alarms
from e00055_detectors import neighbour_check, prepare

EXPLAIN_PAD: Final[int] = 48
VETO_ZS: Final[tuple[float, ...]] = (1.5, 2.0, 3.0)
SETTINGS: Final[dict[int, float]] = {8: 6.0, 48: 4.0}
"""Window in half-hours and the D0 threshold of experiment 43."""
GRID: Final[tuple[float, ...]] = tuple(np.round(np.arange(1.5, 6.01, 0.25), 2))


def main() -> None:
    """Print veto shares at experiment 43's thresholds and the threshold that matches D0's count."""
    panel = load_panel()
    logged = near_log_masks(pad_stamps=EXPLAIN_PAD)
    months = stamp_months()
    odd_set = list(TUNING_MONTHS)
    for window, old in SETTINGS.items():
        inputs = prepare(residuals=panel["residual"], window=window)
        base = raw_alarms(inputs=inputs, detector="D0", threshold=old)
        explained = np.array([logged[s][t] for s, t in base.select("series", "stamp").iter_rows()])
        odd = np.isin(months[base["stamp"].to_numpy()], odd_set)
        target = int((odd & ~explained).sum())
        print(f"\nwindow {window}: D0 at {old}: {base.height} alarms, odd unexplained {target}")
        for veto_z in VETO_ZS:
            tiers = neighbour_check(inputs=inputs, table=base, veto_z=veto_z)["tier"].to_numpy()
            vetoed = tiers == "vetoed"
            print(
                f"  veto_z {veto_z}: vetoed {vetoed.sum()} of {base.height} alarms; "
                f"of odd unexplained {(odd & ~explained & vetoed).sum()} of {target}; "
                f"of explained {(explained & vetoed).sum()} of {explained.sum()}; "
                f"confirmed {(tiers == 'confirmed').sum()}"
            )
            counts = {}
            for threshold in GRID:
                table = raw_alarms(inputs=inputs, detector="D0", threshold=float(threshold))
                if table.is_empty():
                    counts[threshold] = 0
                    continue
                t = neighbour_check(inputs=inputs, table=table, veto_z=veto_z)
                keep = (t["tier"] != "vetoed").to_numpy()
                expl = np.array(
                    [logged[s][st] for s, st in table.select("series", "stamp").iter_rows()]
                )
                o = np.isin(months[table["stamp"].to_numpy()], odd_set)
                counts[threshold] = int((o & ~expl & keep).sum())
            matched = min((th for th, c in counts.items() if c <= target), default=None)
            print("    odd unexplained after veto by threshold:", counts)
            print(f"    lowest grid threshold with count <= {target}: {matched}")


if __name__ == "__main__":
    pl.Config.set_tbl_rows(40)
    main()
