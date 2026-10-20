# Differentiable physics: pooling across substations: research plan

**The rows below are the planned and proposed experiments of this theme, in priority order (1 is
first).** Experiment 105, the first of the theme, is finished: pooling helped a new or short-history
substation and was worse at full history.

| Experiment | Question | Priority | Waits for | State |
|---|---|---|---|---|
| Onboarding design (proposed) | Does a pooled model for onboarding and a per-site model for mature substations beat either alone? | 2 | Experiment 105: pooled onboarding scored 4.04 against 14.38 for a per-site model on 4 weeks, and was worse at full history | Proposed |
| Experiment 120, SH (planned) | Do the Primaries fall into a few kinds, so that a mix of K shared daily shapes per substation forecasts better than one fleet-mean profile? | 2 | Module `studies.dp_shape_library` (own pull request) and its Opus review | Planned; tuning and the known-answer gate may run before the review |
| Part 4 (proposed) | Does a drift term (a slowly changing demand level), trained from 2019-09, beat the same model on its best window? | 3 | Experiment 112 (finished: an exact variable-projection fit did not improve the pooled model, and its known-answer test failed at realistic noise) and a rerun of experiment 109's gate | Proposed |
| Part 5 (proposed) | Does the MVA observation layer (a model of how a meter that reads apparent power relates to real power) help the MW sites and recover MW at MVA sites? | 4 | Experiment 105 and a better capacity fit | Proposed |

**Experiment 109 gives the pooled convex model N for the next experiments to start from, once its
known-answer gate is rerun.** The gate failed under its pre-registered definition, so Part 5
and Part 6 depend on N and on a rerun of the gate with an oracle that estimates every
coefficient and a tuning rule checked on several seeds.

## Planned experiments

<!-- planned-table -->
