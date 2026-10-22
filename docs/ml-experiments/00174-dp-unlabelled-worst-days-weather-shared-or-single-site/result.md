---
id: '00174'
slug: dp-unlabelled-worst-days-weather-shared-or-single-site
part: BG-EB2
title: >-
  Is the loss on the forecast's unlabelled worst days weather error, a day shared with other
  substations, or an error at one substation only?
theme: differentiable-physics-forecast-engine
also_relevant_to:
- production-reference-and-best-guess
- switching-labels-in-the-forecast
- weather-products-and-blending
status: planned
size: >-
  medium. Stored: private tables under data/ml-experiments/00174-dp-unlabelled-worst-days-weather-shared-or-single-site/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible (which past-weather estimate, whether the spread is refitted, which forecast ranks the
  other Primaries' days), and this plan pre-registers one of each and reports the alternatives as
  labelled sensitivities. Callers: this folder's scripts, which read experiment 155's and
  experiment 136's stored forecasts and fits, experiment 172's stored day tables, and experiment
  87's stored BG2 forecasts, and call the reviewed `studies.*` modules without changing them. No
  model is fitted.
hypothesis: >-
  Most of the excess loss on the unlabelled worst days of experiment 172 is weather error, and the
  unlabelled worst days are mostly shared with other Primaries. Replacing the ensemble-mean weather
  with the estimate of past weather (ERA5 temperature and CAMS irradiance) in arm D's forecast
  removes at least 50% of the unlabelled days' excess loss, and at least a third of the unlabelled
  days are also in the worst 5% of days of at least one other Primary.
prediction: >-
  Written before any score is read. Past weather removes 40% to 70% of the excess loss on the
  unlabelled worst days (the labelled switching days keep most of theirs, because switching is a
  demand-side event). 35% to 60% of the unlabelled days coincide with another Primary's worst-5%
  day. The chance rate of that coincidence is high with 15 other Primaries (about 54% if days were
  independent), so the test that carries information is the excess over a circular-shift control.
  A residual third group, the excess that stays with perfect weather on a day no other Primary
  shares, is 10% to 30% of the unlabelled days' excess.
decision_rule: >-
  The 72 worst substation-days of experiment 172 (five decision substations, arm D, D+1 band, the
  scored year) are split into its 37 unlabelled days and 35 labelled days. Weather share W is the
  sum over the unlabelled days of min(max(E_ens - E_era5, 0), E_ens) divided by the sum of E_ens,
  where E_ens is the day's excess loss in arm D and E_era5 is the same day's excess in arm D with
  ERA5 temperature and CAMS irradiance in place of the ensemble mean (the excess of a day is its
  mean quantile score minus the substation's median day, each arm against its own median, with the
  spread refitted on each arm's tuning months). Shared share S is the fraction of the unlabelled
  days on which at least one of the other 15 Primaries of experiment 87 is in its own worst 5% of
  days by BG2's score. The hypothesis is `supported` when W is at least 0.5 and S is at least 1/3,
  `partly supported` when exactly one holds, and `refuted` when neither holds. The 95% interval of
  each is a bootstrap over the 12 calendar months of the scored year (20,000 resamples). The split
  of the unlabelled days' excess is: weather (E_ens minus E_era5, clipped to the day's excess),
  shared (the remaining excess on days with S true), and neither (the remaining excess on days with
  S false). The neighbour question is the same count restricted to the Primaries that the
  interconnection table joins to the substation. The verdict is capped at `inconclusive` if the
  injected controls do not recover at least 90% of their injected days in the right group.
benchmark: >-
  The labelled worst days (35) as the comparison set, and a circular-shift control for S (the other
  Primaries' day series shifted by random whole numbers of days, 1,000 draws)
row_set_id: fold_a_daily_gap_v1
metric: >-
  Daily quantile score above the site's median day, in points of effective capacity, over the D+1
  rows (lead 15 to 39 hours) outside the hand-filtered switching masks, 2025-07-03 to 2026-06-30,
  as in experiment 172
code: ml-experiments/00174-dp-unlabelled-worst-days-weather-shared-or-single-site/
---

**Plan only.** The write-up replaces this page when the experiment has run.

We asked what the 37 worst days that experiment 172 could not label have in common. We re-ran arm
D's forecast with the estimate of past weather in place of the ensemble mean, to see how much of the
excess loss is weather error. We also counted how many other Primaries were in their own worst 5% on
the same day. The remainder is demand-side error at one substation.

## Plan

- **Weather.** Arm D is experiment 136's A2 (a convex demand model, monthly T-all fits, a Kalman
  filter on the recent errors) plus experiment 155's winter night offset and 14-day mean-error
  profile. We keep every fit and the filter's state, and replace only the weather window of each run
  with the feature cache's own ERA5 temperature and CAMS irradiance. The offset and the profile are
  rebuilt from the new forecast's own errors with experiment 155's code. ERA5 arrives about 5 days
  late, so the result is an attribution ceiling and not a forecast anyone can issue.
- **Shared days.** Experiment 87's BG2 forecasts cover 16 Primaries over the scored year. Each
  Primary's worst 5% of days are ranked by BG2's own score, so the other Primaries' days do not
  depend on arm D.
- **Controls.** Ten random days per substation get 10 points added to the observed demand (a
  demand-side error that past weather cannot remove, expected W near 0) and, separately, 10 points
  added to the ensemble-driven forecast only (a weather-side error that past weather removes,
  expected W near 1). 172's 5-point control was too small for the heaviest-tailed substation.
- **Reading.** Shares are approximate. Experiment 172's own control recovered 89.2% against the
  required 90%, and its labelled fraction sits on the 50% bar.
