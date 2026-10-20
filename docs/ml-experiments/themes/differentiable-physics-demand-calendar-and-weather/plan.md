# Differentiable physics (demand, calendar, and weather): research plan

**The next step is a drift term that re-estimates its level after the end of the training data.**
Experiment 104 found that all years of history without drift scored worse than 12 months, and that a
drift held constant recovered that and helped in every scored month, leaving all years with drift
level with 12 months.

| Priority | Proposal | Waits for |
|---|---|---|
| 1 | Test a drift that re-estimates its level from recent demand | Part 3, which pools the model across substations |
| 2 | Experiment 121 (MH): multi-hinge heating curve and night-step terms in the convex model | Experiment 114 merge for the scored-year fits |

Later parts are listed in their own themes.

## Planned experiments

<!-- planned-table -->
