# Differentiable physics: long history and drift: research plan

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| Part 6a | The recent-level filter on ensemble weather forecasts | Done | None | Experiment 111: the Kalman recent-level filter lowered the convex model's score by 0.151 points |
| Part 6b (N10) | Does training the Kalman filter jointly with the demand model, as a differentiable fit, lower the score at 1 to 2 days' lead beyond what the two-stage filter gives? | Proposed | The learned-noise filter; W1 (experiment 115, inconclusive) and the KFE experiment (experiment 118, inconclusive) have reported; about 2 to 4 GPU hours | Queued |
| Shared day-level anomaly and sparse process noise (experiment 128) | Does one day-level anomaly state shared across the hour bands, with the process noise estimated sparsely from zero, fix the weights that the known-answer gate could not recover on real residuals in GAM-K's filter? | Proposed | Nothing | In flight (planned, code review before the first scored run) |

The later parts of the differentiable-physics series are planned one at a time, from the result of
the part before.

## Planned experiments

<!-- planned-table -->
