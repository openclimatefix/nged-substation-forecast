# Differentiable physics: long history and drift

**The first experiment of this theme shows that the data lets the series go ahead, with three
changes: start most Primary substation series later than their first record, fix the lockdown window
in advance, and count eligible long-history sites at each tuning year's first day. The second shows
that a simple model of demand minus solar does not clearly gain from every usable year: without a
drift term it scored worse than on the last 12 months, and with a drift term the model was level
with 12 months. The third shows that a Kalman filter on recent residuals lowers the score by 0.61
points, and that the benefit of old years changed sign between two scored years.** This theme covers whether a differentiable-physics model of Primary substation
demand can use old years of history and cope with drift in the demand level over the years. The
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
  demand minus solar, fitted as one convex optimisation per substation, scored 0.25 points higher
  (worse) when trained on every usable year without a drift term than on the last 12 months. Adding
  a drift term lowered the score by 0.29, so all years with drift was level with 12 months (0.04
  points better, 98.33% interval from 0.30 better to 0.21 worse, 2 of 5 substations better), and the
  planned comparison is inconclusive. A first version of the page had the opposite sign for two
  contrasts, because its forecast spread carried the previous year's mean bias; the page now carries
  a Correction section. The experiment belongs mainly to the [demand: calendar and weather
  theme](../differentiable-physics-demand-calendar-and-weather/index.md).
- **[00106 - Differentiable physics, part
  4](../../00106-differentiable-physics-4-all-years-drift-and-recent-level/result.md):** A Kalman
  filter that corrects each day-ahead forecast with the errors observed up to 3.5 hours earlier
  lowered the quantile score by 0.61 points in the scored year and 1.02 in the year before, at all
  five long-history substations, and with the filter running a drift term and refits added nothing
  measurable. More years of history gave -0.22 points in the scored year and +0.32 in the year
  before, so the verdict is partly supported. The filter's residuals use estimates of past weather.
