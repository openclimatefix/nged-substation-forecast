---
id: '00155'
slug: dp-bg-stage2-assembly-night-offset-and-error-correction
part: BG-S2
theme: differentiable-physics-forecast-engine
also_relevant_to:
- production-reference-and-best-guess
- calibration-and-uncertainty
- lagged-power-and-error-correction
status: planned
size: >-
  medium. Stored: private files under
  data/ml-experiments/00155-dp-bg-stage2-assembly-night-offset-and-error-correction/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: more than one is defensible (the order
  of the night offset and the error correction, whether each arm refits its spread, whether the
  error correction applies at every lead), and this plan pre-registers one of each and runs the
  alternative of the last as a labelled variant. Callers: this folder's scripts, which read
  experiment 136's stored forecasts and copy the night-term code of experiment 137 and the
  error-correction code of experiment 145; they call the reviewed `studies.*` modules without changing them. No new modelling.
hypothesis: >-
  A GAM-K forecast (the convex demand model with a Kalman filter on recent errors, experiment 111)
  that already has the heating split by day type, one shared filter noise setting, the pinned solar
  capacity, and the ensemble mean as its weather input (experiment 136's arm A2) improves further
  when two post-forecast corrections are added: a winter night offset (experiment 137) and the
  14-day mean error profile of recipe R (experiment 115). The mechanism is that the two corrections
  remove different errors, a level error that repeats on winter nights and a half-hour-of-day shape
  error that repeats day to day, so their gains add. The assembled arm then scores below experiment
  111's GAM-K and BG2 (experiment 87) over leads 0.5 to 168 hours, and its tails on cold nights and
  on holidays are no worse than the uncorrected arm's.
prediction: >-
  Written before any score is read. C1 (offset, minus arm A) is -0.02 to -0.04 points over all leads
  and -0.10 to -0.30 points on cold nights (experiment 137 found -0.032 over all rows with a
  temperature term). C2 (recipe R, minus A) is -0.03 to -0.08 over all leads and -0.07 to -0.12 at
  day D+1 (R gained 0.088 against GAM-K at D+1 in experiment 145 but A already has the filter and
  T-all). C3 (offset and R, minus A) lies within 0.02 of the sum of C1 and C2. R limited to leads of
  48 hours or less (the variants) scores 0.00 to 0.02 better than R at every lead, because
  experiments 134 and 145 found R lost 0.018 points at 48 to 144 hours. The analog correction of
  experiment 145 (C1 of that experiment) gains -0.01 to -0.04 against R at D+1 and under 0.01 over
  all leads. The best arm scores 2.69 to 2.75 points over all leads against A's 2.791 and BG2's 2.780,
  so the best arm minus BG2 is -0.03 to -0.09. The q95 coverage on cold nights rises by 2 to 8
  percentage points with the offset. Errors do not grow from leads of 3.5 hours to 9.5 hours by more
  than they grow in arm A.
decision_rule: >-
  The headline contrast is C3 (arm D, offset and R at every lead, minus arm A), over leads 0.5 to 168
  hours, at the five decision substations D5, D10, D11, D14, and D16 in the scored year (D9 beside
  them in the tables, in no contrast). Each planned contrast is read at the 99% level of the
  month-resampled paired bootstrap (the 12 calendar months of the scored year, 20,000 resamples).
  The hypothesis is `supported` when C3's interval lies wholly below -0.02 points, D minus BG2 has an
  upper bound of at most +0.05 points, the cold-night q95 coverage of D is at least that of A minus
  0.03, and the score of D on holiday and Christmas rows is at most A's plus 0.10 points. It is
  `partly supported` when C3's interval lies wholly below zero but one of the other three conditions
  fails or the interval's upper bound is above -0.02. It is `refuted` when C3's interval lies wholly
  above zero. It is `inconclusive` otherwise. The threshold of -0.02 is set by cost. The offset adds
  a fitted term of 72 coefficients per substation and R adds a rolling profile that the serving path
  must store, so a gain smaller than 0.02 points (under 1% of the score) does not pay for the two
  added pieces. The winning arm, which is the start for the calibration experiments, is the arm
  among A, B, C, D, the variants, and E with the lowest all-lead score in the scored year whose
  contrast against A has an interval wholly below zero, and the simpler arm (fewer added pieces)
  when two arms are within 0.005 points. The winner is A when no contrast against A is wholly below
  zero. Contrasts C1, C2, C4, C5, and C6 are read the same way and carry no verdict of their own. The
  verdict is capped at `inconclusive` if the reproduction control (arm A rescored from the stored
  forecasts of experiment 136) or the leakage control fails.
benchmark: >-
  Arm A, which is experiment 136's A2 as stored (2.791 points over leads 0.5 to 168 hours at the five
  decision substations in the scored year); BG2 as stored (experiment 87, 2.780); experiment 111's
  GAM-K as stored (2.800); climatology and manual_heuristic as stored by experiment 111, for absolute
  skill
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) in points of
  effective capacity, on rows outside the hand-filtered switching events, over leads 0.5 to 168
  hours, then under 24 hours, 24 to 72 hours, 72 to 168 hours, day D+1, winter (November to March),
  cold nights, and holiday and Christmas rows; coverage and pinball loss at the 0.01, 0.05, 0.95, and
  0.99 quantiles; the quantile score by lead hour up to 24 hours, to check that errors do not grow
  from 3.5 to 9.5 hours of lead
code: ml-experiments/00155-dp-bg-stage2-assembly-night-offset-and-error-correction/
production_status: not-for-production
adopted_in: none
---

# Experiment 155: does GAM-K with a winter night offset and recipe R beat the stage 1 assembly?

Plan only; the write-up follows the runs.

## Plan

**Question.** Experiment 136 assembled a GAM-K with the heating split (T-all), one shared filter
noise setting (POOL), the pinned solar capacity, and the ensemble mean as its weather input, and
could not rank it against BG2. Two later experiments found corrections that act on the finished
forecast: a winter night offset (the audit of
[experiment 137](../00137-differentiable-physics-e7-economy-7-night-step-term-in-the-forecast/result.md))
and the 14-day error profile of recipe R
([experiment 115](../00115-differentiable-physics-w1-diurnal-shape-state-on-ens/result.md)). This
experiment adds them to experiment 136's A2 one at a time and together, scores every arm on the same
rows, and picks the winning arm that the calibration experiments start from. No new modelling: every
piece exists, and the code is copied from experiments 136, 137, and 145.

**Arms.** All arms share the rows and the weather of experiment 136's A2, the pinned capacity of
[experiment 139](../00139-differentiable-physics-pv-as-of-capacity-in-gam-k/result.md) (which found
that replacing the pin does not help, so the pin is kept), and the filter noise of experiment 136.

| Arm | Built from | Added piece |
|---|---|---|
| A | Experiment 136's A2 forecasts as stored | none: GAM-K, T-all, POOL, pinned capacity, ensemble mean |
| B | A | winter night offset |
| C | A | recipe R at every lead |
| D | B | recipe R at every lead, fitted on B's own past errors |
| C48 | A | recipe R at leads of 48 hours or less (variant) |
| D48 | B | recipe R at leads of 48 hours or less (variant) |
| E | B | the analog correction of experiment 145 (ANALOG+R, tuned values) at every lead, in place of R |
| B0 | A | the offset's per-slot constants and weekend flag only, without the winter indicator (ablation, no verdict) |

**Reference arms.** BG2, experiment 111's GAM-K, `ref_climatology`, and `ref_manual_heuristic`, as
stored and read as experiment 136 reads them.

**The winter night offset (arm B).** For each of the 24 half-hours that end between 21:30 and 09:00
UTC, the offset is `a[h] + w[h] * winter + c[h] * weekend`, where `winter` is 1 on evenings from
November to March and 0 otherwise, and `weekend` flags Saturday and Sunday evenings. The
coefficients are a ridge regression of arm A's own past errors (actual minus mean forecast) on those
72 columns. They are fitted per substation on all rows whose run origin and valid time are both
before the first day of the month being forecast (an expanding window, refitted monthly, starting
from 2024-07), as in experiment 137, with experiment 137's ridge penalty of 100. The offset replaces
experiment 137's temperature term because the audit of that experiment found a winter indicator
matched it and so the data show a winter night bias and not a step that grows with cold. The offset
shifts every quantile and the mean.

**Recipe R (arms C, D, C48, D48).** For a run with origin on UTC day `O`, the target day is `T = O +
1`. The profile is the mean over the source days `T - 15` to `T - 2` (at least 3 of them usable) of
each source day's D+1 error block, by local half-hour with the day's mean removed. A source day's
block is the error of the pre-R forecast of the run issued on the day before it, over its leads of
15.5 to 39 hours. The shift added to every row of the run is minus 0.75 times the profile at the
row's local half-hour. The code is `e00145_analog.py`'s recipe R, copied. In arm D the pre-R forecast
is B, so R is fitted on the errors that remain after the offset and the two corrections do not
count the same error twice. The 48-hour variants zero the shift on rows with a lead above 48 hours.

**The analog correction (arm E).** Experiment 145's ANALOG+R with its tuned values (20 source days of
the same day type within 13 weeks of the target's day of the year, weight 0.5 of the analog profile
against R's 14-day profile, kappa 1.0), added to B at every lead. The values are not retuned here.
The loss beyond 48 hours that experiment 145 found is why the variants exist, and arm E is read
at D+1 and at every lead.

**Spread.** Each arm's quantiles are its corrected mean plus a residual table (hour band by
calendar quarter by lead band) fitted on that arm's own corrected forecasts in the tuning months
(the odd months of 2024-07 to 2025-06), every spread factor 1, as experiment 136 and experiment
145 do. Experiment 137 left the spread unchanged instead. The spread's width is therefore not a
second thing the arms differ in, and the comparison with the residual table of arm A's uncorrected
forecast (a post hoc sensitivity row for arm B) shows how much the refit matters. Experiment 125
found factors tuned on alternate months scored worse out of period, so no factor is tuned. The
comparison spread (the pooled quantile model with factor 1.15 of the BG2 recipe) is the stored BG2
and is not recomputed.

**Rows and periods.** `fold_a_daily_gap_v1`, 09:00 UTC `power_fcst_init_time`, leads 0.5 to 168
hours, rows outside the hand-filtered switching events. The scored year is the 292 runs from
2025-07-02 to 2026-06-29. The tuning months are the odd months and the replicate months the even
months of 2024-07 to 2025-06, as in experiment 136. D-labels only. The error stream that R and the
offset read is the corrected forecasts of the stored tuning, replicate, and scored runs. The
every-fifth-day stride runs of the scored year are in no stored forecast, so R's 14-day profile has
about 11 source days there, and the minimum of 3 is met.

**Weather and what exists at forecast time.** The weather is the calibrated ensemble mean of
experiment 136, and the filter uses an analysis proxy built from earlier runs. The offset uses the
calendar only, and R and the analog correction use errors of days that ended before the origin. Every
input exists at forecast time.

**Scoring.** The headline is the quantile score over leads 0.5 to 168 hours at the five decision
substations in the scored year. Beside it:

- lead bands (under 24 hours, 24 to 72 hours, 72 to 168 hours) and day D+1;
- **cold nights**: night and evening half-hours (the night and evening hour bands of experiment 136)
  of the winter days whose ERA5 daily mean 2 m temperature at the substation's grid box is below 2
  °C, at all leads; experiment 136's definition (an ERA5 half-hourly temperature of 3 °C or below)
  is reported as well so that the two experiments can be compared, and q95 coverage is the second
  headline for the cold-night rows because experiment 125 found it was 73% to 78%;
- **holiday and Christmas rows**: rows on a bank holiday or in the Christmas run days (the
  `holiday_period` of the calendar), with the Christmas rows reported alone;
- tail calibration at the 0.01, 0.05, 0.95, and 0.99 quantiles;
- the quantile score by lead hour for leads up to 24 hours, and the score at leads of 0.5 to 3.5
  hours, 3.5 to 9.5 hours, and 9.5 to 24 hours, for each arm. The production service will refresh
  its forecast every 6 hours with an input that is 3.5 hours old, so a forecast used in production
  is up to 9.5 hours older than its origin. The check is whether an arm's error grows between those
  lead bands by more than arm A's, which would make the corrections fragile under that cadence. The
  production cadence itself is not tested here;
- absolute scores for every arm and for the reference arms, and the mean absolute error of the mean
  forecast beside every quantile-score contrast.

**Planned contrasts.** C1 B minus A (the offset), C2 C minus A (R), C3 D minus A (both: the
headline), C4 D minus C (the offset on top of R), C5 D minus B (R on top of the offset), C6 D
minus BG2, C7 E minus D (the analog correction in place of R), C8 C48 minus C (limiting R to 48
hours), and C9 D48 minus D. Each is read at the 99% level. The replicate months give the sign.

**Controls, run before any contrast is read.**

- **Reproduction.** Arm A, rescored from the stored forecasts of experiment 136 on the same common
  rows, scores 2.791 over leads 0.5 to 168 hours, within 0.001.
- **Leakage.** Add 100 points to `actual` at every valid time on or after each month's cut, in a copy
  of the frames, and rebuild every correction. Pass: no correction applied to the month changes (a
  maximum change of 0.0), for the offset and for R. For R the control also corrupts `actual` from the
  origin's own day and checks that the run's shift is unchanged, because the newest source day is
  `T - 2`.
- **Injected known answer.** Add a winter-night step of known size (6 points on the half-hours from
  01:00 to 08:30 UTC, November to March) to the `actual` of the rows before 2025-07-01, and a
  daily shape error of known size (3 points on the half-hours ending 17:00 to 19:00 local time, every
  day) to the `actual` of the same rows. Pass: the fitted winter-night offset at the cuts of 2025-11
  onward is within 25% of 6 points at 5 of 6 substations, and R's shift on those half-hours has
  a mean within 25% of -0.75 times 3 points, both evaluated on the scored year's runs.
- **Placebo.** Shift the winter indicator by six months (so `winter` is 1 from May to September) and
  refit. Pass: the offset's gain in the scored year is at most half the gain of arm B. The offset's
  per-slot constants carry about a third of its gain (experiment 137's audit), so the placebo is not
  expected to be zero, and the ablation B0 shows how much the constants carry.

**Five triggers (size).** Stored: private files only. Serving path: not touched. Degradation rule:
not touched. More than one design defensible: yes, three, each pre-registered above. Callers not
nameable without searching: no, the scripts copy experiments 136, 137, and 145.

**Audit triggers to watch.** Trigger 2 (a result contradicts its prediction) and trigger 5 (a nested
arm beating the arm that contains it, which could be arm D against arm B and arm C) are the likely
ones. The write-up lists which fired and tells the manager if any did.

**Resources.** Central processing unit (CPU) only, at most 8 cores with one thread per worker, each
job under `systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue`. No GPU. The jobs read
stored forecasts, so a full run is expected to take minutes.

**Outputs.** Predictions for every arm in the `dp_preds` layout (`q01` to `q99`, `actual`,
`switching_mask`, 09:00 UTC origins, leads up to 168 hours) for D5, D9, D10, D11, D14, and D16, under
`data/ml-experiments/00155-dp-bg-stage2-assembly-night-offset-and-error-correction/work/dp_preds/`,
so that the substation cards can be rebuilt from the winning arm.

**Advice followed and departures.** From experiment 136: keep T-all; check POOL's cold-night tails
(the tail table reports them). From experiment 137: test a seasonal offset and judge on cold winter
nights and also on the 22:00 to 23:00 half-hours (the by-hour figure shows them). From experiment
145: apply R at D+1 only (the variants test this; the brief's recipe applies it at every lead) and
leave holiday-like days on R's 14-day profile. From experiment 139: keep the pinned capacity.
Provisional: the offset, because experiment 137 failed its placebo and its readability rule; this
experiment is a second winter of evidence only in the sense that the scored year is the same year,
so it adds no independent winter.
