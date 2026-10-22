---
id: '00178'
slug: dp-sunny-daylight-gap-diagnosis
title: >-
  Where does the best GAM lose to the XGBoost model of BG2 on sunny daylight half-hours: which hours,
  clear-sky bands, leads, or substations?
theme: differentiable-physics-embedded-generation
also_relevant_to:
- production-reference-and-best-guess
status: finished
size: >-
  simple. Stored: private tables under data/ml-experiments/00178-dp-sunny-daylight-gap-diagnosis/
  only. Serving path: not touched. Degradation rule: not touched. Designs: one defensible design,
  because the maintainer fixed the four cuts. Callers: this folder's scripts, which read experiment
  164's stored forecasts (the GAM that experiment 169 calls arm A), experiment 87's BG2 forecasts,
  experiment 106's feature cache, and experiment 111's stored ensemble irradiance, and call
  reviewed `studies.*` modules without changing them. No forecast model is fitted.
hypothesis: >-
  On sunny daylight half-hours the best GAM scores worse than the BG2 XGBoost model, and the gap is
  concentrated in one level of one cut (hour of day, clear-sky band of the ensemble irradiance,
  lead band, or substation), which names the cause as a fixed-shape mismatch or an irradiance-input
  problem.
prediction: >-
  Written before any score is read. At least 60% of the summed GAM-minus-BG2 absolute error on sunny
  daylight half-hours sits in one level of one cut. The most likely level is the morning and evening
  shoulder hours (a share of 60% to 70%), because experiments 139 and 157 made capacity accuracy an
  unlikely cause and the fixed PV shape is south-facing. A fail (no level holds 60%) means the gap is
  spread evenly, which points at the irradiance input as a whole.
decision_rule: >-
  Each of the four cuts divides the sunny daylight half-hours into levels (hour of day in 2-hour
  bins, ensemble clear-sky index in four bands, lead in four bands, substation). The gap is the sum
  over rows of the GAM's absolute error minus BG2's absolute error. A level concentrates the gap
  when it holds at least 60% of the total gap while holding fewer than 40% of the rows. The
  hypothesis is `supported` when one level of one cut does, `refuted` when no level of any cut
  does, and `inconclusive` when the total gap's month-block 95% interval includes zero. No threshold
  of effect size is needed because the experiment changes nothing in production.
benchmark: >-
  BG2 (experiment 87, `p_s6_bg2`), same substations, runs, and half-hours as the GAM
row_set_id: fold_a_daily_gap_v1
metric: >-
  Mean absolute error of the median forecast (q50) in MW, and pinball loss averaged over the 11
  quantiles in MW, on rows outside the hand-filtered switching events
code: ml-experiments/00178-dp-sunny-daylight-gap-diagnosis/
verdict: partly supported
production_status: not-for-production
adopted_in: none
take_home_message: >-
  On the sunny daylight half-hours we could reconstruct, the best GAM is only 0.03 to 0.05 MW worse
  than BG2, not the 0.13 MW the by-condition plot showed. Most of that gap is one substation, D5,
  where the GAM is worse than BG2 on sunny half-hours at every hour, clear-sky band, and lead. The
  other five substations show a smaller loss on partly cloudy half-hours that clear half-hours
  offset, so the next experiment should look at D5's solar shape before touching the irradiance
  input.
date: 2026-10-22
headline: >-
  Sunny daylight, six substations, all leads: GAM minus BG2 mean absolute error +0.050 MW (95%
  interval +0.013 to +0.079); D5 alone +0.202 MW (+0.110 to +0.274), the other five +0.022 MW
  (-0.009 to +0.047).
best_arm: none; a diagnosis with no new fit
lessons:
  - >-
    The by-condition plot's rows could not be rebuilt: its code is not in the repository, and no
    definition of sunny daylight that we tried gave a GAM error of 0.58 MW with BG2 at 0.45 MW.
  - >-
    The gap is not a shoulder-hour effect. No 2-hour bin holds more than 32% of it.
  - >-
    D5 holds 63% of the gap with 16% of the rows, and without D5 the gap's interval includes zero.
advice:
  - >-
    Look at D5's solar shape (orientation, share of demand that is solar) before changing the
    irradiance input for all substations; D5 carries 63% of the sunny-daylight gap (figure 4).
  - >-
    Do not describe the sunny-daylight loss as a shoulder-hour or azimuth problem; the gap is spread
    over the hours of the day in proportion to the rows (figure 1).
  - >-
    When quoting a by-condition gap, state the row definition and the lead range, because the gap
    falls from 0.050 MW on all leads to 0.031 MW at 12 to 36 hours (results table).
---

**The best GAM is worse than BG2 on sunny daylight half-hours by 0.05 MW, and one substation, D5,
holds most of that.** We asked which hour of day, clear-sky band, lead band, or substation holds
the gap between the best convex model (a GAM, from experiment 155 and calibrated by experiments 160
and 164) and the XGBoost model of BG2. We joined both models' stored forecasts for the scored year
on the same substation, run, and half-hour, and cut the gap four ways. Without D5 the gap shrinks
to 0.02 MW and its interval includes zero. The earlier by-condition plot showed 0.13 MW, and we
could not reproduce that figure, so the size of the loss is smaller than the plot suggested.

The GAM is experiment 164's fix O, which is arm A of experiment 169; its median forecast equals
that of experiment 155's arm D on these rows. BG2 is experiment 87's
`p_s6_bg2`. A sunny daylight half-hour is one in the top third of each substation's daylight
half-hours by the unit solar shape of experiment 106's feature cache (the definition experiment
139 uses for its sunniest third). The by-condition plot's own code is not in the repository, so
this definition is our reconstruction. It gives 90,143 half-hours over all leads and 13,198 at
12 to 36 hours, against 12,125 in the plot.

## Figures

![Gap by hour of day](figure_01_hour.svg)

![Gap by clear-sky band](figure_02_clear_sky_band.svg)

![Gap by lead band](figure_03_lead.svg)

![Gap by substation](figure_04_site.svg)

![Gap by clear-sky band without D5](figure_05_clear_sky_band_without_d5.svg)

## Results

- **Rows and scope:** six substations (D5, D10, D11, D14, D16, D9), the 292 daily runs of
  `fold_a_daily_gap_v1` from 2025-07-02 to 2026-06-29, rows outside the hand-filtered switching
  events. The plan did not fix the lead range, so the headline uses all leads to 168 hours
  (post hoc); the 12 to 36 hour range, nearest to the plot's row count, is beside it.
- **Absolute errors (MW), sunny daylight, all leads:** GAM 0.590 against BG2 0.541; pinball loss
  0.139 against 0.125 (gap +0.0143, interval +0.0054 to +0.0226).
- **Gap (GAM minus BG2 mean absolute error, MW), month-resampled 95% intervals:**

| Scope | Rows | Gap | Interval | Pinball gap |
|---|---|---|---|---|
| All leads, six substations | 90,143 | +0.050 | +0.013 to +0.079 | +0.0143 |
| All leads, five decision substations | 74,728 | +0.044 | +0.007 to +0.072 | +0.0123 |
| 12 to 36 h, six substations | 13,198 | +0.031 | -0.014 to +0.064 | +0.0083 |
| 12 to 36 h, five decision substations | 10,952 | +0.027 | -0.017 to +0.059 | +0.0072 |
| All leads, D5 only | 14,045 | +0.202 | +0.110 to +0.274 | +0.0519 |
| All leads, without D5 | 76,098 | +0.022 | -0.009 to +0.047 | +0.0073 |

- **Cut that concentrates the gap (all leads; rule: at least 60% of the gap in a level holding under
  40% of the rows):**
    - Substation: D5 holds 63% of the gap with 16% of the rows. The rule is met.
    - Clear-sky band: the 0.4 to 0.7 band holds 83% of the gap with 32% of the rows. The rule is
      met, but the band is partly cloudy, where the GAM is worse (0.749 MW against 0.622 MW) and
      better than BG2 at 0.9 and above in five substations (0.438 MW against 0.499 MW).
    - Hour of day: no 2-hour bin holds more than 32% of the gap (10:00 to 12:00 UK local time), and
      the two busiest bins hold 26% and 29% of the rows. The rule is not met.
    - Lead band: 72 to 168 hours holds 77% of the gap and 58% of the rows. The rule is not met.
- **D5 is worse than BG2 in every clear-sky band** (0.873 against 0.590 MW at 0.4 to 0.7, 0.523
  against 0.412 at 0.7 to 0.9, 0.602 against 0.304 at 0.9 and above) and at every hour from 08:00 to
  18:00, so the cause is the site, not the weather band.
- **Solar share:** D5 has the largest relative fall from the night mean to the sunny-daylight mean
  (3.44 to 1.14 MW, 67%), which we use as a proxy for its solar share because the fitted capacities
  need the GPU environment. D10 has the next largest relative fall (2.42 to 1.26 MW, 48%) and the
  GAM beats BG2 there (-0.009 MW), so solar share alone does not explain D5.
- **Ensemble clear-sky index on the sunny rows:** the median is 0.77, and the 5th to 95th
  percentile range is 0.52 to 0.98.

## Checks

- **Median minus mean forecast:** zero at every site, because the GAM's stored mean equals its
  median. BG2 stores no mean, so its median is used.
- **Error of the mean forecast beside the quantile score:** the mean absolute error (of the median
  forecast) and the pinball loss gaps point the same way in every row of the table above.
- **Triggers fired:** trigger 4 (leaving one site out changes the verdict). Without D5 the all-lead
  gap's interval includes zero. That leave-one-out is the finding of this experiment, not a hidden
  fragility, and no planned contrast of the experiment depends on it. No audit was run; the manager
  decides whether one is needed. Trigger 2 does not apply: the experiment makes no claim against an
  earlier experiment, it fails to reproduce a plot.

## Caveats

- **The row set is our reconstruction.** The plot's MAE (0.58 against 0.45 MW) is not reproduced:
  the same definition gives 0.59 against 0.54 over all leads and 0.52 against 0.49 at 12 to 36
  hours. A definition from realised daily clear-sky index (top third of days) gave no gap at all
  (0.44 against 0.45 MW, five substations at 12 to 36 hours, 11,670 half-hours); that check is
  exploratory and not in the script.
- **Per-level intervals are not computed.** Only the totals carry month-resampled intervals, so the
  level shares are descriptive.
- **Lead was cut at all leads.** The plot's 12,125 rows resemble the 12 to 36 hour range, where the
  total gap's interval includes zero.

## What to try next

- Check D5's solar shape (orientation, tilt, and the capacity fitted for it) against D10's, where
  the GAM does well.
- Ask for the by-condition plot's row definition and the models it scored, so the 0.13 MW gap can
  be reproduced before another experiment is built on it.
