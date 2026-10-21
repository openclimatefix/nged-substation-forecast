r"""The write-up's checks: tails of each step statistic, and leave-one-donor-out contrasts.

- The 99.9th percentile of the absolute odd-month 24-hour step statistic divided by its robust
  scale, per arm. A heavy tail makes the scale (a median absolute deviation) understate the noise
  that an alarm threshold meets, and so explains an arm whose steps shrink less than its scale.
- C1 and C2 with each donor series left out in turn (audit trigger 4).

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00130-*/e00130_checks.py
"""

from __future__ import annotations

import numpy as np
import polars as pl
from e00130_common import SERIES_IDS, TUNING_MONTHS, stamp_months
from e00130_contrasts import BLANK_SERIES, contrast, fmt, pooled, populations
from e00130_detectors import level_shift, robust_scale
from e00130_variants import load_both

WINDOW = 48


def tail_ratios() -> None:
    """Median over series of q99.9 / robust scale of the odd-month step statistic, per arm."""
    odd = np.isin(stamp_months(), list(TUNING_MONTHS))
    print("\nTail of the odd-month 24-hour step statistic: q99.9 of |shift| over robust scale")
    for source in ("A15", "B", "C"):
        panel, _ = load_both(source=source)
        ratios = []
        for series in SERIES_IDS:
            if series == BLANK_SERIES:
                continue
            shift = level_shift(values=panel["residual"][series], window=WINDOW)
            scale = robust_scale(shift=shift)
            chosen = np.abs(shift[odd & np.isfinite(shift)])
            ratios.append(float(np.percentile(chosen, 99.9)) / scale)
        print(
            f"  {source}: median {np.median(ratios):.1f}, "
            f"range {min(ratios):.1f} to {max(ratios):.1f}"
        )


def leave_one_donor_out(*, data: pl.DataFrame) -> None:
    """C1 and C2 with each donor series left out in turn."""
    events = pl.read_parquet(__import__("e00130_common").WORK_DIR / "A15" / "events.parquet")
    donor = dict(zip(events["event_id"].to_list(), events["donor"].to_list(), strict=True))
    print("\nLeave one donor series out (base, even months): C1 B minus A, C2 C minus A")
    base = pooled(data=data, population="base", parity="even")
    for series in SERIES_IDS:
        if series == BLANK_SERIES:
            continue
        kept = [
            f.filter(pl.Series([donor[e] != series for e in f["event_id"].to_list()])) for f in base
        ]
        c1 = contrast(frames=kept, arm="B_XL", base="A15_XL")
        c2 = contrast(frames=kept, arm="C_XL", base="A15_XL")
        print(f"  without donor {series:2d}: C1 {fmt(*c1)}  C2 {fmt(*c2)}")


def main() -> None:
    """Print both checks."""
    tail_ratios()
    leave_one_donor_out(data=populations())


if __name__ == "__main__":
    main()
