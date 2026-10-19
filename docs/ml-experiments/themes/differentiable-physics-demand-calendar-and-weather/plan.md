# Differentiable physics (demand, calendar, and weather): research plan

**The next step is a drift term that re-estimates its level after the end of the training data.**
Experiment 104 found that a drift held constant halved the gain from more years of history, and that
it lost most when the held value was freshest.

| Priority | Proposal | Waits for |
|---|---|---|
| 1 | Test a drift that re-estimates its level from recent demand, and examine the drift's last knot | Part 3, which pools the model across substations |

Later parts are listed in their own themes.

## Planned experiments

<!-- planned-table -->
