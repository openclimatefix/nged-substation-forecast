# Differentiable physics: the forecast engine: research plan

**Part 6 of the series is the first experiment of this theme.** The confirmation run depends on its
margin contrast.

| Experiment | Question | Priority | Waits for | State |
|---|---|---|---|---|
| 00111 | Does the convex differentiable-physics model, driven by ECMWF ensemble forecasts, beat lagless XGBoost, and does using its forecast as the starting point of the production recipe's lagged XGBoost improve that recipe? | High | The `studies.dp_features` ensemble layer pull request | Planned |
| 00115 | Does a within-band diurnal-shape state in GAM-K's recent-error filter lower the day-ahead quantile score on ENS, at five decision substations, in months that did not choose its form and in the scored year? | High | Experiment 111 merged, for its stored fits | Planned |
| BG2-margin confirmation (inside 00111) | Does the production blend with the DP margin in its weekly member beat the rebuilt production blend like for like? | Conditional | 00111's margin contrast (C3) estimate below zero | Proposed |

## Planned experiments

<!-- planned-table -->
