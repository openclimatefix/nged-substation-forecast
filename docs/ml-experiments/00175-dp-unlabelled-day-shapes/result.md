---
id: '00175'
slug: dp-unlabelled-day-shapes
part: BG-EB3
title: >-
  What shape does the forecast's error take on the unlabelled worst days: a sharp-edged block, a
  smooth offset, a ramp, or a spike?
theme: differentiable-physics-forecast-engine
also_relevant_to:
- switching-detectors
- switching-labels-in-the-forecast
status: done
size: >-
  medium. Stored: private tables under data/ml-experiments/00175-dp-unlabelled-day-shapes/ only.
  Serving path: not touched. Degradation rule: not touched. Designs: more than one is defensible
  (the step test, the spike test, the window of the neighbour check), so this plan fixes each rule
  in numbers before any day is looked at. Callers: this folder's scripts, which read experiment
  174's day tables, experiment 155's arm D forecasts as copied into experiment 174's folder,
  experiment 87's BG2 forecasts, experiment 147's and experiment 156's stored detector intervals, and
  NGED's interconnection table, and call reviewed `studies.*` modules without changing them. No
  forecast model is fitted.
hypothesis: >-
  At least half of the 37 unlabelled worst substation-days of experiment 174 are sharp-edged blocks
  in the forecast's error (a switching-like change in the power a substation draws), and in at
  least a third of those a substation interconnected with the first shows a step of opposite sign
  within 2 hours (a load transfer between the two).
prediction: >-
  Written before any day is looked at. Of the 37 days, 19 to 28 are sharp-edged blocks (class a),
  5 to 10 are smooth offsets (class b), 2 to 6 are ramps or midday-only errors (class c), and 0 to 3
  are spikes (class d). Of the class (a) days with a metered neighbour, 30% to 60% have an
  opposite-signed neighbour step within 2 hours, against a chance rate of 10% to 25% from shifting
  the neighbour's residuals by whole days. The detector of experiments 141 and 147 fires on 25% to
  50% of the class (a) days, and the step persists over the following 7 days on 15% to 35%.
decision_rule: >-
  Both halves are read from counts with 95% Wilson intervals. The first half passes when class (a)
  holds at least 19 of the 37 days (at least half). The second half passes when at least a third of
  the class (a) days that have a metered neighbour show an opposite-signed neighbour step within 2
  hours. `supported` when both pass, `partly supported` when exactly one passes, `refuted` when
  neither does. If the first half fails, the take-home message is that the lever is the Kalman
  filter's level adaptation, not switching detection. If the first half passes and the second
  fails, it is that the steps are real but the interconnection table cannot explain them, a direct
  case for a latent normal-running-arrangement node. The second half is read next to its chance
  rate from the shifted-neighbour control: the second half also needs the observed rate to exceed
  the mean chance rate by at least 10 points, otherwise it counts as failed (an observed rate near
  chance is evidence against a load transfer). The second half is "not assessed" when fewer than 10
  class (a) days have a metered neighbour, and the verdict is then `partly supported` or `refuted`
  on the first half alone. The first half is also read against the class (a) rate of ordinary days
  (below): if ordinary days reach class (a) as often as the unlabelled worst days, class (a) says
  nothing about worst days and the verdict is `inconclusive`. If the first half fails, the
  take-home message names the Kalman filter's level adaptation only when class (b) holds more days
  than any other class; otherwise it says the shape is mixed.
benchmark: >-
  Ordinary days (for each substation, the days whose excess loss lies in the middle half of its
  scored days, 40 drawn at random per substation) given the same classifier; a known-answer control
  that injects a block, an offset, a ramp, and a spike into ordinary days; and a shifted-neighbour
  control (each neighbour's residual series rolled by a whole number of days, separately per
  neighbour, 1,000 draws) for the opposite-signed-step rate. Experiment 172's known-answer control
  (89.2% of 10-point injections found in the worst 5%, under its 90% bar) is quoted, so that every
  share here is read as approximate
row_set_id: fold_a_daily_gap_v1
metric: >-
  Counts of days per shape class, and the share of class (a) days with a neighbour step of
  opposite sign. Residual is observed minus forecast of experiment 155's arm D (ensemble-weather
  forecast), D+1 band (lead 15 to 39 hours), outside the hand-filtered switching masks,
  2025-07-03 to 2026-06-30, as in experiments 172 and 174
code: ml-experiments/00175-dp-unlabelled-day-shapes/
date: 2026-10-22
verdict: refuted
best_arm: >-
  Not applicable (descriptive). 17 of the 37 unlabelled worst days (45.9%, 95% interval 31.0% to
  61.6%) are sharp-edged blocks, against the 19 needed. 15 are smooth offsets (40.5%). On the 17 block
  days, an interconnected Primary steps the opposite way within 2 hours on 52.9%, below the 70.9%
  that rolling the neighbours by whole days gives.
headline: >-
  Sharp-edged blocks make up 17 of 37 unlabelled worst days (45.9%, 31.0% to 61.6%), under the 19
  needed, and smooth offsets make up 15 (40.5%). Ordinary days reach block shape on 27.5% of days
  and offset shape on 16.0%. An interconnected Primary steps the opposite way within 2 hours on 52.9%
  of the block days, below the 70.9% (52.9% to 88.2%) that rolling the neighbours' residuals by
  whole days gives. The detectors fired within 3 hours on none of the 17 block days.
lessons:
  - >-
    The unlabelled worst days are two shapes about equally: sharp-edged blocks (46%) and smooth
    same-sign offsets (41%), with offsets 2.5 times as common as on ordinary days (16%).
  - >-
    Interconnected neighbours show an opposite step no more often than by chance, so the
    interconnection table does not explain these blocks.
  - >-
    The shape rules recover only about 60% of injected blocks and offsets and almost no injected
    ramps or spikes, so every count is approximate.
take_home_message: >-
  The forecast's worst unlabelled days are roughly half sharp-edged blocks (a switching-like change
  in the power a substation draws) and 40% smooth offsets (a change in level or behaviour that lasts
  the day). Interconnected neighbours do not show the opposite step more often than chance would
  give, and the existing switching detectors fired on none of the block days. A fix therefore needs
  both a faster level adaptation for the offsets and a way to see blocks that the interconnection
  table does not explain.
advice:
  - >-
    Do not expect the interconnection table to explain unlabelled block days: an opposite-signed
    neighbour step is no commoner than the 70.9% that rolled neighbours give (figure 2).
  - >-
    Treat the Kalman filter's level adaptation and a block-aware detector as separate fixes: the
    unlabelled worst days split 17 blocks to 15 offsets (figure 1).
  - >-
    Read any day-shape count from rules with a known-answer control: this experiment's rules recover
    about 60% of injected blocks and offsets and almost no ramps or spikes (known-answer control).
production_status: not-for-production
adopted_in: none
pr: 1480
---

**The forecast's worst unlabelled days are about half sharp-edged blocks and 40% smooth offsets, and
interconnected neighbours do not explain the blocks.**
[Experiment 174](../00174-dp-unlabelled-worst-days-weather-shared-or-single-site/result.md) left 37
worst substation-days (five Primary substations, one year) whose forecast loss is not weather error
and not shared with other Primaries. We took the forecast's error on each day, observed minus
forecast power in half-hours, and sorted each day into a shape with rules fixed before any day was
looked at: a sharp-edged block (the error jumps and stays shifted for hours, as a switching event
would), a smooth offset (the same sign most of the day), a ramp or midday-only error, or a spike of
1 to 3 readings. For each block day we asked whether an interconnected Primary shows a step of
opposite sign within 2 hours (a load transfer), and whether the switching detectors fired.

**The pre-registered verdict is `refuted`.** Class (a), the sharp-edged block, holds 17 of the 37
days against the 19 needed, and a neighbour's opposite step on the block days (52.9%) is below what
rolling the neighbours' residuals by whole days gives (70.9%), so the second half failed the
excess-over-chance rule. Two cautions apply. The shape rules recover only about 60% of injected
blocks, so 17 against 19 does not exclude a majority of blocks. The block count is also well above
ordinary days (27.5%), so switching-like days are over-represented among the worst days.
[Experiment 172](../00172-dp-error-budget-worst-days/result.md)'s own known-answer control
recovered 89.2% of injected days against a 90% bar, so every share here is approximate. This
experiment fits no new forecast model.

![Share of days in each error shape, unlabelled worst days against ordinary days](fig1_classes.svg)

![Share of block days with an opposite-signed neighbour step, against rolling the neighbours by days](fig2_neighbour.svg)

Figures 3 to 6 show every one of the 37 days, one panel each, grouped by substation:

![The 8 days at D10](fig3_days_d10.svg)

![The 10 days at D11](fig4_days_d11.svg)

![The 11 days at D14](fig5_days_d14.svg)

![The 8 days at D16 and D5](fig6_days_d16_d5.svg)

## Results

- **Shape counts (37 days):** sharp-edged block 17 (45.9%, 95% Wilson interval 31.0% to 61.6%),
  smooth offset 15 (40.5%, 26.3% to 56.5%), ramp or midday-only 4 (10.8%), none of the four 1,
  spike 0. All 4 ramp days put at least half their squared residual in 10:00 to 16:00.
- **By substation:** D10 6 blocks and 2 offsets; D11 0 blocks, 8 offsets, and 2 ramps; D14 6 blocks,
  3 offsets, 1 ramp, 1 none; D16 2 blocks and 2 offsets; D5 3 blocks and 1 ramp.
- **Ordinary days (200 days, 40 per substation from the middle half by excess loss):** block 27.5%
  (21.8% to 34.1%), offset 16.0%, ramp or midday-only 31.0%, none of the four 25.5%.
- **Neighbour step on the block days:** all 17 block days have at least one neighbour with a BG2
  forecast. 9 show an opposite-signed step within 2 hours (52.9%, 31.0% to 73.8%). Rolling each
  neighbour's residuals by 1 to 90 whole days (1,000 draws) gives a mean of 70.9% (95% range 52.9%
  to 88.2%). The 55 ordinary-day blocks give 49.1% (36.4% to 61.9%). The neighbour test is lenient
  (2 spreads, 8 candidate times, a noisy neighbour forecast), so chance is high.
- **Detectors on the 17 block days:** the refit-offset neighbour-count detector (RO-NC, called XL in
  experiment 82; experiments 141 and 147) fired within 3 hours of the edge on 0 with either the
  causal-live end rule (R2) or the hindsight end rule (R0), and the pair alarm of experiment 156
  on 0. The five substations have 2 to 26 RO-NC onsets over the whole stored history, and the
  ordinary-day base rate is 0 of 55 for RO-NC and 1 of 55 for the pair alarm.
- **Return and persistence:** 4 of the 17 blocks return within the local day. The step persists over
  the following 7 days on 1 of 17 and reverts on 14 (2 unknown), but the Kalman filter inside arm D
  adapts to a level shift, so persistence of the real step is understated. Ordinary-day blocks
  persist on 1 of 55.
- **Known-answer control** (a shape added to 40 ordinary days each): intended class recovered on
  24 of 40 injected blocks, 24 of 40 offsets, 1 of 40 ramps, and 3 of 40 spikes. Blocks and offsets
  are found about 60% of the time, and ramps and spikes are almost always classed otherwise, so
  the ramp and spike counts are lower bounds and every count is approximate.

## Checks

- **Median forecast minus mean forecast, per site:** 0.0 points at every substation, as in
  experiments 172 and 174 (the arm D quantile forecast has its median equal to its mean).
- **Mean-forecast error beside the quantile score:** not applicable; no score is contrasted here.
- **Audit triggers:** trigger 2 (a result contradicting its own prediction) fired, because class (a)
  holds 17 days against the predicted 19 to 28. The explanations tested are the ordinary-day rate
  (27.5%, so blocks are over-represented) and the known-answer control (about 60% recovery, so the
  shortfall of 2 days is inside the classifier's miss rate). Neither contradicts the neighbour
  result, which points the same way, so no audit was requested.

## Method and caveats

- **Row set:** `fold_a_daily_gap_v1`, experiment 155's arm D (a convex demand model with a Kalman
  filter, driven by the ensemble-mean weather forecast), D+1 band (lead 15 to 39 hours),
  hand-filtered switching rows left out, 2025-07-03 to 2026-06-30. Weather at forecast time is the
  ensemble forecast, so nothing here uses weather that arrives late.
- **Rules:** fixed in the plan commit of this pull request before any day was looked at, and
  coded in `e00175_classify.py`. A day is a block when its residual has a jump of more than 3
  day-before spreads within 1 hour that lasts 4 hours. The classes are tested in the order block,
  spike, offset, ramp or midday-only.
- **Neighbours:** Primaries joined to the substation's Primary by NGED's interconnection table that
  have a BG2 forecast (experiment 87, 16 series). BG2's residual is observed minus its median
  forecast, in the same points of effective capacity as arm D, and it has no switching mask.
- **Block edge at the day boundary:** a block that starts before the day's first half-hour is not
  found, and the 8-reading windows need at least 4 valid readings on each side.
- **One year, five substations:** the denominators are small (17 block days), so the 95% intervals
  are wide (about 30 points).

## What to try next

- Read the 17 block days against NGED's switching logs (`Switching Logs.xlsx` in the hierarchy
  folder), to see which blocks are logged switching that the detectors missed and which are
  unlogged.
- Test a faster level adaptation in the Kalman filter on the 15 offset days, which are 2.5 times as
  common as on ordinary days and concentrated at D11 (8 of 10 days).
