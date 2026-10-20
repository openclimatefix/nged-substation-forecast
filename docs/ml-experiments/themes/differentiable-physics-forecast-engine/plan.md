# Differentiable physics: the forecast engine: research plan

**Part 6 of the series was the first experiment of this theme, and its follow-ups wait on the
maintainer.** The rows below are proposals, not agreed work.

| Experiment | Question | Priority | Waits for | State |
|---|---|---|---|---|
| Margin arms at weekly cadence (proposed) | Does the DP's forecast as the starting point of lagged XGBoost help when the XGBoost model refits weekly, as the production recipe does, rather than monthly? | 3 | The maintainer's GPU budget (about 60 GPU-hours at the measured speed) | Proposed |
| ENS history of 2021 to 2024 (proposed) | Do three more years of ensemble history give XGBoost a fair training history, and a second validation year for the DP against XGBoost? | 2 | The maintainer's approval of a fetch from Source Cooperative | Proposed |
| Kalman filter on the convex model's effects, on ENS (experiment 118) | Does a weight on each frozen effect of the convex model (level, profile scale, heating, PV capacity, holiday), following a random walk with a fitted noise matrix, lower the day-ahead score below GAM-K's level-and-anomaly filter? | 1 | The Opus review of the module PR (before the scored-year run) | Planned |
| Filter in the production serving path (proposed) | Does a Kalman filter on the recent residuals of the production XGBoost blend lower its day-ahead score as it does for the convex model? | 1 | Nothing; a candidate for the next release review | Proposed |

## Planned experiments

<!-- planned-table -->
