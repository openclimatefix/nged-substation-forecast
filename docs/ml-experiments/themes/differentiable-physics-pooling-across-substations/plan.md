# Differentiable physics: pooling across substations: research plan

**Part 3 of the series (experiment 105) is finished; the rows below are what is planned next.**

| Experiment | Question | Priority | Waits for | State |
|---|---|---|---|---|
| 00109 | Does a convex model with a fleet mean plus ridge-penalised per-substation deviations match a model per substation, and beat it for a new substation with 4 weeks or 3 months of history? | High | The `studies.dp_gam_pooled` module pull request | Planned |
| Onboarding design (proposed) | Does a pooled model for onboarding and a per-site model for mature substations beat either alone? | 2 | Part 3 (105): pooled onboarding 4.04 against 14.38 for a per-site model on 4 weeks, worse at full history | Proposed |
| Part 4 (proposed) | Does a drift term, trained from 2019-09, beat the same model on its best window? | 2 | The pooled convex experiment (109) and a variable-projection fit; Part 3 (105) found the pooled neural model worse than a per-site model at full history | Proposed |
| Part 5 (proposed) | Does the MVA observation layer help the MW sites and recover MW at MVA sites? | 3 | Part 3 (105) and the capacity fit | Proposed |

## Planned experiments

<!-- planned-table -->
