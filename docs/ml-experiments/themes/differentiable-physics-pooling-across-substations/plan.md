# Differentiable physics: pooling across substations: research plan

**The rows below are the planned and proposed experiments of this theme, in priority order (1 is
first).** Experiment 105, the first of the theme, is finished: pooling helped a new or short-history
substation and was worse at full history.

| Experiment | Question | Priority | Waits for | State |
|---|---|---|---|---|
<<<<<<< HEAD
| 00109 | Does a convex model with a fleet mean plus ridge-penalised per-substation deviations match a model per substation, and beat it for a new substation with 4 weeks or 3 months of history? | High | The `studies.dp_gam_pooled` module pull request | Planned |
| Part 4 (proposed) | Does a pooled drift term, trained from 2019-09, beat the same model on its best window? | 2 | 00105 confirmation tier | Proposed |
| Part 5 (proposed) | Does the pooled MVA observation layer help the MW sites and recover MW at MVA sites? | 3 | 00105 | Proposed |
=======
| 00112 | Does the pooled DP match a per-site DP once every linear coefficient is solved exactly at each step (variable projection) and only the non-linear parameters take gradient steps? | 1 | The `fit_pooled_vp` module pull request | Planned |
| Onboarding design (proposed) | Does a pooled model for onboarding and a per-site model for mature substations beat either alone? | 2 | Experiment 105: pooled onboarding scored 4.04 against 14.38 for a per-site model on 4 weeks, and was worse at full history | Proposed |
| Part 4 (proposed) | Does a drift term (a slowly changing demand level), trained from 2019-09, beat the same model on its best window? | 3 | Experiment 112 (a variable-projection fit, an optimiser that solves the linear coefficients exactly at each step) and a rerun of experiment 109's gate | Proposed |
| Part 5 (proposed) | Does the MVA observation layer (a model of how a meter that reads apparent power relates to real power) help the MW sites and recover MW at MVA sites? | 4 | Experiment 105 and a better capacity fit | Proposed |

**Experiment 109 gives the pooled convex model N for the next experiments to start from, once its
known-answer gate is rerun.** The gate failed under its pre-registered definition, so experiment
112, Part 5, and Part 6 depend on N and on a rerun of the gate with an oracle that estimates every
coefficient and a tuning rule checked on several seeds.
>>>>>>> origin/main

## Planned experiments

<!-- planned-table -->
