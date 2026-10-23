"""Check for experiment 286: is method B's interval as wide as method A's on the same months?

Method B keeps fewer months than method A (48 to 56 against 59 to 65), so a wider interval could
come from fewer months and not from the index. This script refits method A on only the months
that method B kept, and refits both with a straight line through the months of the other.
"""

from pathlib import Path
from typing import Final

import e00286_injection as e
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00286-metered-farm-drift-soiling-injection"


def main() -> None:  # noqa: D103
    wide, info = e.load_farms()
    sites = sorted(info)
    rows = []
    for target in sites:
        frame = e.build_target_frame(wide=wide, target=target, sites=sites)
        one = np.ones(frame.height)
        a = e.monthly_rel(frame, factor=one, method="A")
        b = e.monthly_rel(frame, factor=one, method="B")
        common = a.join(b.select("month"), on="month")
        slope_a, lo_a, hi_a = e.fit_trend(t=common["t"].to_numpy(), rel=common["rel"].to_numpy())
        rows.append(
            {
                "farm": target,
                "months_a_all": a.height,
                "months_b": b.height,
                "half_width_a_on_b_months": (hi_a - lo_a) / 2,
                "trend_a_on_b_months": slope_a,
            }
        )
    out = pl.DataFrame(rows)
    out.write_csv(DATA_DIR / "same_months_check.csv")
    print(out)


if __name__ == "__main__":
    main()
