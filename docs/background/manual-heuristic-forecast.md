# The Manual Heuristic Forecast

NGED has been investing in a suite of different forecasting tools for use cases stretching from
multi-year planning forecasts through to day-ahead dispatch of flexibility services. However, until
recently, the normal approach to forecasting among distribution network operators relied more
heavily on the operator. That approach used no weather model and no machine learning: for each
substation it assembled a small ensemble of **historical analogues** from that substation's own
past, plotted them, and let a human operator read a forecast off the spread. That analogue method
is the **manual heuristic** — the method our models will be compared against — so we reproduce it
faithfully as the `manual_heuristic` [baseline
forecaster](../roadmap/metrics-and-leaderboard.md#the-headline-baseline-manual_heuristic).

## The recipe

For a given substation and target half-hour, the analogue ensemble is the observed power at the
**same time-of-day on the same weekday** from:

- the **last 6 weeks** (6 analogues), and
- the 7 weeks spanning **49–55 weeks back** — roughly a year ago, bracketing the same calendar
  point (7 analogues).

That is **13 analogues** in total.

**The 13 analogues are used as observed and weighted equally.** The manual heuristic forecast
itself does not adjust for holidays, switching events, or load growth.

**The output is the plot itself.** An operator looks at the 13 traces and their spread and forms a
judgement. If a single deterministic number is needed, the operator reads off an upper percentile
chosen to match the company's risk appetite. Our scoring uses the 95th percentile. An upper
percentile is a deliberately **conservative** operating point: the tool's job is to warn when
demand might approach the distribution network's flex/firm capacity limit, so erring high is the
safe direction.

## The operator's view

![Mock-up of an analogue-ensemble forecasting
tool](assets/manual_heuristic_forecast_mock_up.png)

The image above is a mock-up of what an operator might see for one feeder over a week. Demand is
plotted as **headroom against the constraint** — the y-axis is "MW Exceedance of Constraint", where
the blue **Flex Profile** line at `0` is the limit and more-negative values are further below it.
Reading the overlays:

- **Red/pink traces** — the individual historical analogues (the 13 above).
- **Green line** and **green band** — the mean of the analogues and the spread around it.
- **Purple dashed line** — a smooth recent-trend line (a quadratic fit) the tool overlays as a
  sense-check.
- **Yellow band** — a 5% warning zone.
- **Black cross** — missing data.

The strong twice-daily peaks and the weekday/weekend difference are exactly the structure the
same-weekday analogue selection is built to capture.

## Why it matters for us

**Because the manual heuristic uses no weather and no ML, beating the manual heuristic is the
project's core deliverable.**

**Because the manual heuristic forecast does not adjust for holidays, switching events, or load
growth, automating those adjustments is useful work in its own right** — for example, aligning bank
holidays and moveable feasts. Both the faithful replica and the variant that automates those
adjustments are specified in [Metrics & leaderboard →
Baseline forecasters](../roadmap/metrics-and-leaderboard.md#baseline-forecasters).
