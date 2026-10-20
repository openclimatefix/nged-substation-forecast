# Calibration and uncertainty: research plan

**A lower number in the "Priority" column runs sooner.**

## Priority

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| [P14](#p14-coupling) | Which coupling step, ECC-Q or the Schaake shuffle, gives better coherence scores at short and long leads? | 5 | The scoring harness and experiment 00016 | Deferred; no consumer needs coherent members yet |
| [Experiment 125 (N8)](../../00125-differentiable-physics-n8-ens-members-through-gam-k-and-the-spread-around-them/result.md) | Does GAM-K's residual table count the weather error twice, and does a table fitted on analysis-driven forecasts, placed around each ensemble member, calibrate the tails better? | 1 | Nothing | Planned |

## Planned experiments

<!-- planned-table -->

## Proposals in detail

### P14: coupling

**The coupling step turns calibrated quantiles back into 51 coherent members.** The arms are ECC-Q
with the interpolation, tail, and tie rules of the [staged forecast roadmap
page](../../../roadmap/staged-probabilistic-forecast.md), and the Schaake shuffle, scored by the
variogram score and the CRPS of summed solar farms in a weather cell, by lead band.
