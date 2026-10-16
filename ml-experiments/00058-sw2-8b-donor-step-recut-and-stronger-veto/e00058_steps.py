r"""Write each injected transfer's donor step in noise units.

The step is the mean MW the injection removes from the donor over the first 24 hours of the
transfer (or the whole transfer if it is shorter), divided by the noise scale of the 24-hour
level-shift statistic for the donor series. `largest_step_z` is the largest such step over the
donor and the metered receivers, because an alarm on any of them finds the transfer. The injection
is re-drawn from experiment 55's seeds, through the batches `e00058_run.py score` uses, so the
steps belong to the transfers scored.

    uv run python ml-experiments/00058-*/e00058_steps.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from e00058_common import BLOCK, WORK_DIR
from e00058_detect import SEED_43, SEED_NEW, load, seeded_batches
from e00058_inject import inject_power
from e00058_run import event_sets

WINDOW: int = 48
"""The 24-hour window, in half-hours."""


def donor_steps(*, out: Path) -> pl.DataFrame:
    """Return one row per injected transfer: its id, donor step in MW, and in noise units."""
    panel, scales = load()
    original, extra, holiday, _ = event_sets(out=out, power=panel["power"])
    groups = seeded_batches(events=original, seed=SEED_43) + seeded_batches(
        events=extra + holiday, seed=SEED_NEW
    )
    rows = []
    for seed, batch in groups:
        injection = inject_power(power=panel["power"], events=batch, seed=seed)
        for event in batch:
            span = slice(event.onset, min(event.end, event.onset + BLOCK))
            step_mw = float(-np.mean(injection.delta[event.donor][span]))
            donor_z = step_mw / scales[WINDOW][event.donor]
            largest_z = max(
                abs(float(np.mean(injection.delta[s][span]))) / scales[WINDOW][s]
                for s in event.series
            )
            rows.append(
                {
                    "event_id": event.event_id,
                    "donor_step_mw": step_mw,
                    "donor_step_z": donor_z,
                    "largest_step_z": largest_z,
                }
            )
    steps = pl.DataFrame(rows)
    zero = steps.filter(pl.col("donor_step_z").abs() < 1e-9).height
    print(f"{steps.height} transfers; {zero} move no MW (donor power zero or missing there)")
    if (
        steps.height != len(original) + len(extra) + len(holiday)
        or steps["donor_step_z"].is_nan().any()
        or steps["largest_step_z"].is_nan().any()
    ):
        raise SystemExit("STOP: re-drawn injection does not match the scored events")
    return steps


def main() -> None:
    """Write `steps.parquet`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK_DIR)
    args = parser.parse_args()
    steps = donor_steps(out=args.out)
    steps.write_parquet(args.out / "steps.parquet")
    print(steps.describe())


if __name__ == "__main__":
    main()
