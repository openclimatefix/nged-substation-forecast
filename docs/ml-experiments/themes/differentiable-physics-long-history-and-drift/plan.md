# Differentiable physics: long history and drift: research plan

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| Part 6a | The recent-level filter on ensemble weather forecasts | Done | None | Experiment 111: the Kalman recent-level filter lowered the convex model's score by 0.151 points |
| Part 6b (N10) | Does training the Kalman filter jointly with the demand model, as one differentiable fit, lower the score at 1 to 2 days' lead beyond what the two-stage filter gives? | Done | None | Experiment 135: not supported. The untied joint fit scored 5.4 points worse, and the tied fit was within 0.013 points of the two-stage fit. No further joint-fit variants are queued |
| Shared day-level anomaly and sparse process noise (experiment 128) | Does one day-level anomaly state shared across the hour bands, with the process noise estimated sparsely from zero, fix the weights that the known-answer gate could not recover on real residuals in GAM-K's filter? | Done | None | Experiment 128: inconclusive, stopped at its known-answer gate. No rerun is queued, and no further per-substation learned-noise experiments after experiment 133 are queued |
| Experiment 167 | With the PV capacity explicit and the heating split by day type, does the GAM trained on every year back to late 2019 beat the last 24 months, with or without a drift term, at 12 quarterly refits on past weather? | In progress | None | Planned; fits start after this commit. Prediction that can fail: all years with drift beats 24 months by at least 0.05 points pooled and at 8 or more of 12 origins |

The later parts of the differentiable-physics series are planned one at a time, from the result of
the part before.

## Planned experiments

<!-- planned-table -->
