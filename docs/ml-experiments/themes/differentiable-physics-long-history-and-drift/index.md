# Differentiable physics: long history and drift

**The first experiment of this theme shows that the data lets the series go ahead, with three
changes: start most Primary substation series later than their first record, fix the lockdown window
in advance, and count eligible long-history sites at each tuning year's first day. The second shows
that a simple model of demand minus solar forecasts better when trained on every usable year than on
the last 12 months, but that a drift term fitted to the end of the training data gives half of the
gain back.** This theme covers whether a differentiable-physics model of Primary substation demand
can use old years of history and cope with drift in the demand level over the years. The
[Experiments](experiments.md) page lists every experiment tried, with its verdict and a take-home
message, and the [Research plan](plan.md) lists what is planned next.

## What the experiments found

- **[00103 - Differentiable physics, part
  1](../../00103-differentiable-physics-1-data-look/result.md):** The first months of 13 of the 16
  Primary substations are cut (19.4 months at D4), the data does not isolate the 2020 lockdown from
  drift, and the power timestamps show no 30-minute error on either side of NGED's fix of
  2026-03-26.
- **[00104 - Differentiable physics, part
  2](../../00104-differentiable-physics-2-simple-structural-model/result.md):** A simple model of
  demand minus solar, fitted as one convex optimisation per substation, scored 0.40 points lower
  (better) when trained on every usable year than on the last 12 months (4 of 5 substations better).
  Adding a drift term halved that gain, and the planned comparison is inconclusive (0.20 points
  better, 98.33% interval from 0.68 better to 0.22 worse), because it rests on one substation, D14.
  The experiment belongs mainly to the [demand: calendar and weather
  theme](../differentiable-physics-demand-calendar-and-weather/index.md).
