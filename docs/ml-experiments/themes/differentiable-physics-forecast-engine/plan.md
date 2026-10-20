# Differentiable physics: the forecast engine: research plan

**Part 6 of the series was the first experiment of this theme, and its follow-ups wait on the
maintainer.** The rows below are proposals, not agreed work.

| Experiment | Question | Priority | Waits for | State |
|---|---|---|---|---|
| Experiment 119 (ENC, planned) | Does a small neural network reading the ENS boxes around a Primary lower GAM-K's quantile score, on average and on cold, showery, and badly forecast days? | 1 | Nothing; known-answer gate first | Planned |
| Margin arms at weekly cadence (proposed) | Does the DP's forecast as the starting point of lagged XGBoost help when the XGBoost model refits weekly, as the production recipe does, rather than monthly? | 3 | The maintainer's GPU budget (about 60 GPU-hours at the measured speed) | Proposed |
| ENS history of 2021 to 2024 (proposed) | Do three more years of ensemble history give XGBoost a fair training history, and a second validation year for the DP against XGBoost? | 2 | The maintainer's approval of a fetch from Source Cooperative | Proposed |
| Filter in the production serving path (proposed) | Does a Kalman filter on the recent residuals of the production XGBoost blend lower its day-ahead score as it does for the convex model? | 1 | Nothing; a candidate for the next release review | Proposed |

## Planned experiments

<!-- planned-table -->
