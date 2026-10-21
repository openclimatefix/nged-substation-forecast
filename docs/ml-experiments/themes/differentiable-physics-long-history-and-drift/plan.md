# Differentiable physics: long history and drift: research plan

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| Part 6a | The recent-level filter on ensemble weather forecasts | Done | None | Experiment 111: the Kalman recent-level filter lowered the convex model's score by 0.151 points |
| Part 6b (N10) | Does training the Kalman filter jointly with the demand model, as one differentiable fit, lower the score at 1 to 2 days' lead beyond what the two-stage filter gives? | Done | None | Experiment 135: not supported. The untied joint fit scored 5.4 points worse, and the tied fit was within 0.013 points of the two-stage fit. No further joint-fit variants are queued |
| Shared day-level anomaly and sparse process noise (experiment 128) | Does one day-level anomaly state shared across the hour bands, with the process noise estimated sparsely from zero, fix the weights that the known-answer gate could not recover on real residuals in GAM-K's filter? | Done | None | Experiment 128: inconclusive, stopped at its known-answer gate. No rerun is queued, and no further per-substation learned-noise experiments after experiment 133 are queued |
| Experiment 149 | With the solar term explicit, does training the convex model on all history back to 2019 beat a 24-month window, with and without a drift term, at 12 rolling origins? | Queued by the maintainer | Experiment 139 (PR 1440 merged; scoring in progress) | Planned: result.md committed, fits not started |

The later parts of the differentiable-physics series are planned one at a time, from the result of
the part before.

## Planned experiments

<!-- planned-table -->
