# Calibration and uncertainty: research plan

**A lower number in the "Priority" column runs sooner.**

## Priority

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| [335](../../00335-primary-demand-default-3-tail-calibration-at-0-5-to-240-hours/result.md) | Do per-lead-band, per-group tail calibrations bring the q01, q05, q95, q99 of Primary demand research default 3 to nominal from 0.5 to 240 hours? | 1 | Experiment 322 | Planned |
| [P14](#p14-coupling) | Which coupling step, ECC-Q or the Schaake shuffle, gives better coherence scores at short and long leads? | 5 | The scoring harness and experiment 00016 | Deferred; no consumer needs coherent members yet |

## Planned experiments

<!-- planned-table -->

## Proposals in detail

### P14: coupling

**The coupling step turns calibrated quantiles back into 51 coherent members.** The arms are ECC-Q
with the interpolation, tail, and tie rules of the [staged forecast roadmap
page](../../../roadmap/staged-probabilistic-forecast.md), and the Schaake shuffle, scored by the
variogram score and the CRPS of summed solar farms in a weather cell, by lead band.
