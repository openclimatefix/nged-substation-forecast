# Calibration and uncertainty: research plan

**A lower number in the "Priority" column runs sooner.**

## Priority

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| [E160](#e160-band-width) | Does reading the spread table smoothly along the lead remove the q05-q95 width staircase of the stage 2 winner without hurting the quantile score? | 1 | Experiment 155 (merged) | Planned |
| [P14](#p14-coupling) | Which coupling step, ECC-Q or the Schaake shuffle, gives better coherence scores at short and long leads? | 5 | The scoring harness and experiment 00016 | Deferred; no consumer needs coherent members yet |

## Planned experiments

<!-- planned-table -->

## Proposals in detail

### E160: band width

**The spread table has four lead bands, so the band widens in steps.** Experiment 160 interpolates
each hour band by quarter cell between the lead-band centres, forces the width to be non-decreasing
in lead, or both, on the stage 2 winning arm, without refitting the mean. It passes when the upper
95% bound of the quantile-score change is under +0.01 points and the staircase is gone.

### P14: coupling

**The coupling step turns calibrated quantiles back into 51 coherent members.** The arms are ECC-Q
with the interpolation, tail, and tie rules of the [staged forecast roadmap
page](../../../roadmap/staged-probabilistic-forecast.md), and the Schaake shuffle, scored by the
variogram score and the CRPS of summed solar farms in a weather cell, by lead band.
