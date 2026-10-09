"""The second setting and the extra negative controls (science review M8 and M10).

- **Second setting.** Rungs 1 and 2 re-run with the estimator's `sensitivity` setting: the duration
  priors' log standard deviations doubled and the efficiency prior's 95% range widened to 0.70 to
  0.95. (The differentiable estimator has no state-of-charge limits or cap to move; the usable
  duration absorbs the limits, and the merchant battery's cap is a learned weight between 1 and 2
  cycles a day. The Agile unit's cap is fixed at 1 cycle a day in both settings, and the second
  setting's cap of 2 and limits of 0% and 100% apply only to the grid estimator.) Writes
  `rung1_sensitivity_posteriors.parquet` and `rung2_sensitivity_posteriors.parquet`.
- **Rung 2 negative control with the Agile price also moved.** The one-hour-early windows move only
  the four fixed-window units, and the Agile unit, which carries rung 2's detection, is unchanged.
  This control fits rung 2 on the coarse stacks twice: with the real windows and prices
  (`rung2_coarse_real_posteriors.parquet`), and with the windows one hour early and both the N2EX
  and Agile prices taken from 7 days later (`rung2_coarse_placebo_posteriors.parquet`). Both use
  the coarse stacks so that neither is favoured by a finer interpolation.

Run: `OMP_NUM_THREADS=2 uv run python studies/unmetered_battery_capacity/capacity_sensitivity.py`
"""

import capacity_rung1
import capacity_rung2
from capacity_inputs import OUTPUT_DIR
from capacity_runs import fit_all_blocks
from capacity_stacks import coarse_stack_path


def main() -> None:
    """Run the second setting for rungs 1 and 2, and the rung 2 placebo with moved prices."""
    aggregates, metadata = capacity_rung1.build()
    fit_all_blocks(
        aggregates=aggregates, metadata=metadata, label="rung 1 sensitivity", setting="sensitivity"
    ).write_parquet(OUTPUT_DIR / "rung1_sensitivity_posteriors.parquet")
    aggregates, metadata, _ = capacity_rung2.build()
    capacity_rung2.fit(
        aggregates=aggregates,
        metadata=metadata,
        window_shift_hours=0.0,
        setting="sensitivity",
    ).write_parquet(OUTPUT_DIR / "rung2_sensitivity_posteriors.parquet")
    capacity_rung2.fit(
        aggregates=aggregates,
        metadata=metadata,
        window_shift_hours=0.0,
        stack_path=coarse_stack_path(weeks=0),
    ).write_parquet(OUTPUT_DIR / "rung2_coarse_real_posteriors.parquet")
    capacity_rung2.fit(
        aggregates=aggregates,
        metadata=metadata,
        window_shift_hours=capacity_rung2.SHIFT_HOURS,
        stack_path=coarse_stack_path(weeks=1),
    ).write_parquet(OUTPUT_DIR / "rung2_coarse_placebo_posteriors.parquet")


if __name__ == "__main__":
    main()
