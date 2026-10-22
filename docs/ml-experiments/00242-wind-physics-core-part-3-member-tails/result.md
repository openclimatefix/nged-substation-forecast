---
id: "00242"
slug: wind-physics-core-part-3-member-tails
title: >-
  Does running all 51 ensemble members through the wind model give better 1% and 99% quantiles for
  the three metered wind farms than running the ensemble mean once?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
status: planned
date: 2026-10-22
size: >-
  Medium. Stored: private files under data/ml-experiments/00242-wind-physics-core-part-3-member-tails/
  only. Serving path: not touched. Degradation rule: not touched. Designs: two defensible ways to
  summarise the ensemble (the mean of the weather inputs, or the mean of the outputs), and the plan
  pre-registers the first, because it is the cheap serving route the question is about. Callers:
  this folder's scripts, which copy experiment 230's helpers and read the same stored inputs. No new
  model family and no new data, so one Sonnet code review, one Sonnet write-up review, and one
  Sonnet prose review.
hypothesis: >-
  For the three metered wind farms (W1, W2, W3), running each of the 51 ensemble members through an
  ERA5-trained wind model and mixing the 51 forecasts gives better 0.01 and 0.99 quantiles than
  running the model once on the ensemble mean of the weather inputs and applying the same spread
  factor, because the model is non-linear near cut-in and rated wind, so a model of the mean weather
  is not the mean of the model.
prediction: >-
  Written before any fit. On the rows kept by experiment 210's mask over all leads, the tail score T
  (the mean of the 0.01 and 0.99 pinball losses, in points of capacity, about 0.47 for experiment
  230's arms) is lower for the members route than the mean route by 0.02 to 0.08 points for the
  physics core (C1) and by a similar amount for the XGBoost recipe (C2), with the gap largest at
  lead days 3 to 6, where the ensemble spread is widest. Before the spread factor, the 5% to 95%
  band of the mean route covers under 88% of rows at lead days 1 and 2, and the members route covers
  88% to 92%. The core's residual quantiles add 0.02 or more to the tail score over the members'
  expected powers alone (C3). The core and XGBoost members routes are within 0.03 points of each
  other on T (C4).
decision_rule: >-
  Decisive contrasts, each the first arm minus the second on T, on the rows kept by experiment 210's
  mask over all leads, after the spread factor, with whole months (of the ENS run) and the three
  seeds resampled, 50,000 resamples, 95% intervals Bonferroni-corrected to 98.75% across the four:
  C1 B1_members minus B1_mean; C2 A0r_members minus A0r_mean; C3 B1_members minus B1d_members; C4
  B1_members minus A0r_members. Supported: C1 and C2 intervals both wholly below zero (members
  better). The threshold is zero, because production already runs the 51 members through the model
  and keeping them costs nothing more. The mean route would be adopted only if an interval were
  wholly above zero (the ensemble mean is better and cheaper). Anything else is inconclusive for
  that model. Everything else (the quantile score, the other quantile levels, coverage by lead band,
  per farm, the raw forecasts before the spread factor, the row sets) is descriptive and labelled
  so. The intervals describe these three farms, because farms are not resampled.
benchmark: >-
  The mean route (B1_mean, A0r_mean) is the benchmark for the members route. Experiment 230's
  stored B1 and A0r are the reference rows, and the refits here must reproduce them on the quantile
  score to within 0.02 points over all rows (checked, not assumed).
row_set_id: >-
  experiment 33's wind cases (`fold_a_tuesdays_v1`: 53 Tuesday runs from 2025-07-01 to 2026-06-23,
  W1 to W3, three seeds), as in experiment 230. Rows kept by experiment 210's mask are the headline
  row set; all rows are descriptive.
metric: >-
  Tail score T, the mean of the 0.01 and 0.99 pinball losses in points of capacity (smaller is
  better); quantile score (twice the mass-weighted mean pinball loss over nine levels); coverage of
  the 5% to 95% band by lead band; pinball loss at 0.01, 0.05, 0.5, 0.95, and 0.99
production_status: not-for-production
adopted_in: none
code: ml-experiments/00242-wind-physics-core-part-3-member-tails/
---

This page is the plan. It is committed before any fit, so the commit order shows the hypothesis
came first.

**Question.** Experiment 230 ran all 51 ensemble members through each wind model and mixed the 51
forecasts, then applied a spread factor that widens or narrows the mixture until the 5% to 95%
band covers 90% of earlier rows. This experiment asks whether running the 51 members is better for
the 1% and 99% quantiles than running the ensemble mean once and applying the same spread factor.
NGED uses the tails to decide how much generation to curtail, so the tails matter most.

**Arms** (all trained on ERA5, scored on experiment 230's cases):

- **B1_members and B1_mean:** the physics core of [experiment 230](../00230-wind-physics-core-part-1/result.md)
  with its residual quantiles, run on each member or on the ensemble mean of the weather.
- **B1d_members:** the core's expected power for each member with no residual quantiles, so the
  ensemble supplies all the spread.
- **A0r_members and A0r_mean:** the XGBoost best-guess recipe refitted with the 0.01 and 0.99 levels
  (experiment 230's A0r), run the same two ways.
- Each arm is scored raw and after the spread factor. The headline is after the spread factor,
  because that is how the best guess is served.

**Weather source.** The training weather is ERA5, which arrives about 5 days late. The inference
weather is the ENS members, which exist at forecast time, so nothing here uses weather unavailable
at forecast time.

**Retraining schedule.** One fit per farm on valid times before 2025-06-17, no refits. The
contrasts do not depend on a retraining cadence.

**Neural components.** None. Experiment 230 found that the small neural adapter added nothing to
the core (B9 minus B1 was +0.002), so none is pre-registered here.

**Reuse of earlier advice.** This plan follows the advice to train wind models on ERA5 and to put
effort into the weather input. It departs from nothing.

**Curtailment cost.** Not scored. The node-plus-generators metric needs a node with its generators,
and this experiment scores the three farms alone; the tail scores are the proxy.
