---
id: '00167'
slug: dp-full-history-gam-24-months-or-all-years-with-drift
part: n/a
title: >-
  Does the GAM with explicit PV gain from training on every year of history back to late 2019, with
  or without a drift term, compared with the last 24 months?
theme: differentiable-physics-long-history-and-drift
also_relevant_to:
- retraining-and-training-history
- differentiable-physics-embedded-generation
status: done
size: >-
  medium. Stored: private files under
  data/ml-experiments/00167-dp-full-history-gam-24-months-or-all-years-with-drift/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: more than one defensible (the history
  rule for dropping a site, the spread, the treatment of D10's jumps), and this plan pre-registers
  one of each. Callers: this folder's scripts, which copy experiment 107's refit layout and
  experiment 136's T-all design and call the reviewed `studies.dp_*` modules without changing them.
hypothesis: >-
  With the PV capacity pinned to the as-of table (explicit PV) and the heating split by day type
  (T-all), the convex demand model (the GAM) trained on every year of history back to late 2019
  with a drift term scores better than the same model trained on the last 24 months, when both are
  refitted at 12 quarterly origins on past weather (ERA5 and CAMS). The mechanism is that explicit
  PV removes the changing solar capacity from the target, so the old years teach the heating and
  calendar terms without the capacity growth that made the all-years fits of experiments 104 and
  107 unstable, and the drift term absorbs the remaining slow change in the demand level.
prediction: >-
  Written before any fit. C1 (all years with drift, minus 24 months) is -0.10 to +0.05 points
  pooled, with 5 to 9 of 12 origins below zero, so the 0.05-point rule is likely to fail. C2 (all
  years without drift, minus 24 months) is 0.00 to +0.30, because experiments 104 and 107 found all
  years without drift worse than a shorter window at 8 of 12 origins. C3 (all years with drift,
  minus all years without) is -0.15 to -0.40, as experiment 107's C2 (-0.451, three-fifths of it one
  substation, which this experiment masks). Winter-quarter origins favour more years more than
  summer-quarter ones, as experiment 107's C3 did (-0.461 against +0.058).
decision_rule: >-
  The headline contrast C1 is all years with drift minus 24 months, both with T-all and the pinned
  capacity, as the quantile score (twice the mean pinball loss over experiment 87's 11 quantiles)
  in points of effective capacity, on the same rows, outside the hand-filtered switching events,
  pooled over the five decision substations (D5, D10, D11, D14, D16) and the 12 scored origins.
  The interval resamples whole calendar months at the 98.33% level (a Bonferroni correction over
  the three planned contrasts C1 to C3). `supported` needs a pooled C1 of at most -0.05 points, an
  interval wholly below zero, and C1 below zero at 8 or more of the 12 origins (the maintainer's
  rule of 2026-10-21). `partly supported` is an interval wholly below zero with either a size under
  0.05 or fewer than 8 negative origins. `refuted` is an interval wholly above zero. `inconclusive`
  is everything else. The 0.05 threshold is not zero because all years triples the rows of every
  refit and adds a drift term to fit, which a 24-month window does not need. Winter-quarter
  origins (cuts on 1 October and 1 January) are reported separately and never change the verdict.
  A leave-one-site-out table is part of the verdict check: a verdict that changes when one site is
  left out is capped at `inconclusive`.
benchmark: >-
  The 24-month T-all arm (m24) on the same rows; absolute skill of climatology and manual_heuristic
  as stored by experiment 107 (same origins and sites) for the reference row
row_set_id: >-
  dp_mw_v1 (experiment 103); MW sites; 12 quarterly origins 2023-07-01 to 2026-04-01, each scored on
  the following 3 months; a site-origin pair is scored only if the site has 24 months of usable rows
  before the origin
metric: >-
  Quantile score in points of effective capacity, on rows outside the hand-filtered switching
  events; tail coverage and pinball loss at q01, q05, q95, and q99
verdict: supported
best_arm: >-
  all years with drift (all_d1), quantile score 2.673 points, against 3.177 for 24 months without drift
headline: >-
  All years with drift scored 0.503 points better than 24 months (98.33% interval -0.770 to -0.234),
  with 10 of 12 refits below zero, but drift alone on 24 months gives 0.402 of that, so the extra
  years add 0.101 (post hoc, interval -0.188 to -0.019).
production_status: candidate
adopted_in: none
date: 2026-10-21
take_home_message: >-
  Giving the demand model a drift term lowered its score at 11 of 12 quarterly refits, by 0.40
  points on the last 24 months and by 0.55 on all years, so a drift term is the change to port.
  Training on all years instead of 24 months adds only 0.10 points on top of drift, and without
  drift it gives no gain, so old history helps little.
advice:
  - >-
    Give the GAM a drift term (D1) at every refit, with the last 24 months or all years: it lowered
    the quantile score by 0.402 points on 24 months (98.33% interval -0.639 to -0.154) and by 0.547
    on all years (-0.756 to -0.341), at 11 of 12 refits each (E1, C3).
  - >-
    Treat extra history as worth little: with drift, all years beat 24 months by 0.101 points (post
    hoc E2, -0.188 to -0.019, 8 of 12 refits), and without drift they were level (C2, +0.043, -0.092
    to +0.167).
  - >-
    Check D14 before porting a drift term: it was worse with drift (+0.186 on C1) while every other
    substation improved (per-site table).
  - >-
    Widen the tails with the upper quantiles: even the best arm covers 91.6% at q95 and 97.4% at
    q99, and 3.9% below q01 (tail checks).
code: ml-experiments/00167-dp-full-history-gam-24-months-or-all-years-with-drift/
---

**A drift term is what lowered the score; extra years of history added little.** Training the demand
model on all years with a drift term scored 0.50 points better than training it on the last 24
months without one. Drift alone on the last 24 months gives 0.40 of that. The extra years add 0.10
more, and without drift the extra years gave no gain. The verdict is `supported` by the rule written
before the fits, but it needs this reading: the rule tested "all years with drift against 24 months",
and most of that gain belongs to the drift term.

We asked whether the demand model with an explicit solar term can now train on every year of
history back to late 2019, instead of the last 24 months. The model is the GAM (the convex model of
demand minus solar output) from [experiment
107](../00107-differentiable-physics-rolling-origin-history-and-tails/result.md), with the heating
split by day type from [experiment
136](../00136-dp-bg-stage1-assembly-on-the-ensemble-mean/result.md) and the solar capacity pinned to
the as-of table. We refitted it every quarter from July 2023 to April 2026, 12 refits, each scored on
the next 3 months at 5 Primary substations (D5, D10, D11, D14, and D16), on past weather only
(ERA5 and CAMS). It followed experiment 107's rolling-origin design and its baseline spread. "Drift"
is a slow, smooth level term fitted to the history and held at its last value over the scored
months. "Score" is the quantile score: twice the mean pinball loss over 11 quantiles, in points of
each substation's effective capacity, where lower is better.

![Three contrasts at each refit: drift lowers the score at most refits](figure_01_per_origin.svg)

**Figure 1 shows the drift gain at most refits and the extra-years gain mixed in sign.** C1 is below
zero at 10 of 12 refits, with one clear exception (the July 2025 refit, +0.563). C3 (drift on all
years) is below zero at 11 of 12. E2, the contrast of all years against 24 months with drift in both
arms, is below zero at 8 of 12 and ranges from -0.45 to +0.13.

![Mean quantile score of each arm at each substation](figure_02_absolute.svg)

**Figure 2 shows both drift arms ahead of both no-drift arms at D5, D10, D11, and D16, and behind
at D14.** The mean scores are in the bullets below.

![Contrasts at each substation: D10 gains most from drift and D14 loses](figure_03_per_site.svg)

**Figure 3 shows that the drift gain is largest at D10 and has the opposite sign at D14.** With D10
left out, C1 is still -0.311 (interval -0.531 to -0.104), 9 of 12 refits below zero.

![Tail coverage of each arm: the drift arms are closer to nominal but still too narrow](figure_04_tails.svg)

**Figure 4 shows that every arm is too narrow in the tails, and the drift arms least.** At q95 the
all-years drift arm covers 91.6% of measured values.

## Results

- **Arms and rows.** T-all GAM with pinned capacity, experiment 104's penalties, past weather.
  m24 (24 months, no drift), all (all years, no drift), all_d1 (all years, drift), and the
  exploratory m24_d1 (24 months, drift). The five decision substations at 12 origins (243,741 scored
  half-hours per arm, outside the hand-filtered switching events), identical across arms. D10's two
  unlabelled-jump windows are removed from training and scoring.
- **Absolute scores (points).** m24 3.177, all 3.220, all_d1 2.673, m24_d1 2.774. D9, described
  beside the five and scored at 3 origins only: 2.232, 2.249, 2.376, and 2.360.
- **C1, all_d1 minus m24 (headline):** -0.503 (98.33% interval -0.770 to -0.234); 10 of 12 refits
  below zero. Winter-quarter refits -0.772 (-1.123 to -0.403), 5 of 6 below zero; summer-quarter
  refits -0.233 (-0.571 to +0.092), 5 of 6. Leaving one site out gives -0.311 to -0.671, always
  `supported`.
- **C2, all minus m24:** +0.043 (-0.092 to +0.167); 5 of 12 below zero.
- **C3, all_d1 minus all:** -0.547 (-0.756 to -0.341); 11 of 12 below zero.
- **E1, m24_d1 minus m24 (exploratory):** -0.402 (-0.639 to -0.154); 11 of 12 below zero.
- **E2, all_d1 minus m24_d1 (post hoc, not pre-registered):** -0.101 (-0.188 to -0.019); 8 of 12
  below zero; winter-quarter -0.133 (-0.269 to +0.001), summer-quarter -0.069 (-0.168 to +0.012).
  The drift penalty scales with the training rows, so all years smooths the drift about three times
  more than 24 months, and E2 mixes history length with that smoothing.
- **Tails, decision sites (coverage at q01 / q05 / q95 / q99; nominal 1 / 5 / 95 / 99%).** m24 5.2 /
  9.9 / 84.9 / 93.8%; all 7.1 / 13.5 / 86.7 / 94.4%; all_d1 3.9 / 9.9 / 91.6 / 97.4%; m24_d1 4.2 /
  10.5 / 91.0 / 96.9%. Pinball loss at q99: 0.302, 0.288, 0.205, 0.224.
- **Without the D10 mask,** C1 is -0.455 (-0.723 to -0.191) and C3 -0.484 (-0.703 to -0.277), so
  the mask moves each by about 0.05 and the verdict is unchanged.

## Checks

- **Median minus mean forecast.** Zero at every site and arm, because the baseline spread is centred
  on the median residual.
- **Mean-forecast error beside the quantile score.** The mean absolute error contrast has the same
  sign as the quantile score on every contrast: C1 -0.780, C2 +0.026, C3 -0.806, E1 -0.618, E2
  -0.162 points.
- **Triggers.** Trigger 2 (the result contradicts its prediction) held: the prediction was C1 of
  -0.10 to +0.05 and C3 of -0.15 to -0.40, and the printed C1 is -0.503 and C3 -0.547. The
  explanation was tested: experiment 107 found a drift gain of -0.451 as well, and masking D10's
  jump windows moves C3 by only 0.06, so the prediction wrongly assumed the jumps carried the drift
  gain. Trigger 5 did not fire, because all-years-without-drift against 24 months is a tie
  (+0.043, interval spanning zero). Triggers 1, 3, 4, and 6 did not fire (leave-one-site-out keeps
  the verdict), and trigger 7 is open until a later experiment depends on this result. No Opus audit
  was run. A Sonnet code review found no lookahead; it caught a history rule that counted meter
  readings instead of trainable rows (fixed: D14 and D16 are no longer scored at the first origin,
  which moved C1 from -0.483 to -0.503).

## Caveats

- **The pinned capacity is hindsight at the earliest origins.** The first capacity vintage dates
  from 2023-08-07, so the origin of 2023-07-01 uses an early vintage and the four burn-in cuts use
  a growth prior. All arms share this, but the all-years arms lean on the old capacity path more.
- **Drift is a recent-level term here.** Its last value is held over the scored 3 months, so part of
  its gain may be a recent-level effect, which [experiment
  111](../00111-differentiable-physics-6-ens-forecast-engine-against-bg2/result.md) did not
  separate.
- **Past weather, not forecasts.** Both arms use ERA5 and CAMS at forecast time, which is not
  available live; the contrasts rank training windows, not forecasts.
- **Not scored:** the curtailment-cost saving against the manual heuristic, because the experiment
  ranks training windows on past weather.
- **Schedule.** Quarterly refits on the whole history, frozen between refits, as in experiment 107.

## What to try next

- Port the drift term to the GAM with ensemble weather and check D14 first.
- Test whether all-years training still adds 0.10 points under the same drift smoothing in both
  windows.
