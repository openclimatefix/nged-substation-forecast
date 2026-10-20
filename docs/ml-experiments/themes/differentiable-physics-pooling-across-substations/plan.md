# Differentiable physics: pooling across substations: research plan

**Part 3 of the series is the first experiment of this theme.** The later rows depend on its result.

| Experiment | Question | Priority | Waits for | State |
|---|---|---|---|---|
| 00112 | Does the pooled DP match a per-site DP once every linear coefficient is solved exactly at each step (variable projection) and only the non-linear parameters take gradient steps? | High | The `fit_pooled_vp` module pull request | Planned |
| Part 4 (proposed) | Does a pooled drift term, trained from 2019-09, beat the same model on its best window? | 2 | 00105 confirmation tier | Proposed |
| Part 5 (proposed) | Does the pooled MVA observation layer help the MW sites and recover MW at MVA sites? | 3 | 00105 | Proposed |

## Planned experiments

<!-- planned-table -->
