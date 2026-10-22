---
id: '00168'
slug: dp-gamk-production-cadence-and-3-5-hour-delay
part: BG-S2
theme: differentiable-physics-forecast-engine
also_relevant_to:
- production-reference-and-best-guess
- lagged-power-and-error-correction
status: done
size: >-
  simple, by the five triggers. Stored: private files under
  data/ml-experiments/00168-dp-gamk-production-cadence-and-3-5-hour-delay/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: one defensible design (re-issue the stored
  forecasts of experiments 136 and 155 at the four production origins), with the 3.5-hour delay
  removed as a labelled diagnostic arm. Callers: this folder's scripts, which copy the filter and
  T-all code of experiment 136 and call the reviewed `studies.*` modules without changing them. No
  new modelling.
hypothesis: >-
  The GAM-K forecast of experiment 155's arm D (the convex demand model with a Kalman filter on
  recent errors, the winter night offset, and recipe R) keeps its score when it is issued the way
  production would issue it. Production refreshes the forecast every 6 hours from the newest
  ECMWF run, so the weather forecast is 12 to 30 hours old, and the filter reads only power
  readings that are at least 3.5 hours old. Experiments 136 and 155 issued one forecast a day, 9
  hours after the run. The mechanism is that a fresher filter state makes up for an older weather
  forecast, so the score at a given lead changes little.
prediction: >-
  Written before any score is read. C1 (arm D issued at the four production origins, minus arm D
  issued once a day, at the same lead and run, pooled over leads 0.5 to 24 hours) is +0.00 to +0.03
  points of effective capacity. The older weather forecast costs little at short leads, and the
  filter gains nothing from the later origins at the same lead. C3 (arm D at the production origins
  with the 3.5-hour delay, minus the same forecasts with no delay) is +0.01 to +0.04 pooled at
  leads under 24 hours, and larger than +0.04 at leads under 3.5 hours. C4 (production origins
  minus once-a-day origins at the same valid time) is -0.02 to -0.06 at leads under 24 hours,
  because the later origin has a shorter lead and a fresher filter. The 06:00 origin, whose
  weather forecast is 30 hours old, has the largest C1 of the four origins. On cold nights (daily
  mean temperature below 2 degrees Celsius) C1 is at most +0.10.
decision_rule: >-
  The headline contrast is C1, at the five decision substations D5, D10, D11, D14, and D16 in the
  scored year, pooled over the four production origins and over leads 0.5 to 24 hours (D9 beside
  them in the tables and in no contrast). Each contrast is read at the 99% level of the
  month-resampled paired bootstrap (the 12 calendar months of the scored year, 20,000 resamples).
  The hypothesis is `supported` when C1's point estimate is below +0.05 points and its interval's
  upper bound is below +0.05. It is `partly supported` when the point estimate is below +0.05 and
  the upper bound is not. It is `refuted` when the point estimate is +0.05 or more. The threshold
  of +0.05 is set by cost: it is larger than the 0.037 points that recipe R gained in experiment
  155, so a delay cost above it would remove the gain of the stage 2 corrections, and a cost below
  it is within what the production cadence can carry without a redesign. Where C1 is +0.05 or more,
  the write-up says which origin, lead band, and subset carry it. The lead check of experiment 155
  is repeated: the score at leads of 3.5 to 9.5 hours minus the score at 0.5 to 3.5 hours, for the
  production origins minus the same difference for once-a-day origins, is reported with a 99%
  interval, and passes when its upper bound is at most +0.02 points. The verdict is capped at
  `inconclusive` if the reproduction control fails (an origin shift of zero hours must reproduce
  the stored means to within 0.001 points and the stored quantiles to within 0.001 points).
benchmark: >-
  Arm D and arm A of experiment 155, as stored, issued once a day (2.756 and 2.791 points over
  leads 0.5 to 168 hours at the five decision substations in the scored year)
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over the 11 quantiles of experiment 87) in points of
  effective capacity, on rows outside the hand-filtered switching events, by lead from the
  forecast's own origin, over leads 0.5 to 147 hours at all four origins (leads up to 165 hours at
  fewer origins), then under 24 hours, by origin, and on cold nights; coverage and pinball loss at
  the 0.01, 0.05, 0.95, and 0.99 quantiles
code: ml-experiments/00168-dp-gamk-production-cadence-and-3-5-hour-delay/
verdict: refuted
best_arm: >-
  none; the contrast of interest is 6-hourly arm D minus once-a-day arm D at the same lead, +0.076
  points over leads 0.5 to 24 hours (99% interval +0.009 to +0.129), against 2.306 points for
  once-a-day arm D and 2.386 for 6-hourly arm D
headline: >-
  Issued every 6 hours with power readings 3.5 hours old, arm D scores 0.076 points worse than
  once a day at leads under 24 hours, and 0.017 points worse (-0.021 to +0.042) over all leads to
  147 hours. The 3.5-hour delay accounts for 0.047 points of the 0.076 and for 0.30 points in the
  first 3.5 hours of lead.
lessons:
  - >-
    The same-lead contrast cannot hold the hour of the day equal: the once-a-day origin is always
    09:00 UTC, so each lead falls at one hour of the day. Only leads pooled over 0 to 24 hours, which
    cover every hour in both arms, are comparable.
  - >-
    The delay costs 0.30 points in the first 3.5 hours of lead and under 0.02 points beyond 9.5 hours,
    so its cost is a short-lead cost.
  - >-
    Arm D's corrections (winter night offset and recipe R) keep their gain at the production
    cadence: -0.056 points under 24 hours and -0.034 over all leads to 147 hours.
take_home_message: >-
  Issued every 6 hours from the newest weather forecast with power readings 3.5 hours old, the
  stage 2 forecast scores 0.076 points (3%) worse in the first 24 hours than the once-a-day forecast
  at the same lead, and about the same beyond that. The 3.5-hour power-data delay explains 0.047
  points of the gap and 0.30 points of the first 3.5 hours, so a fresher power feed is the one
  change that would help the short leads. The offset and recipe R still gain 0.056 points at the
  production cadence.
advice:
  - >-
    Expect the first 3.5 hours of lead to score about 0.3 points worse than a no-delay forecast
    when the filter reads only readings at least 3.5 hours old, and plan the serving design around
    that figure (C3).
  - >-
    Do not read the lead check of experiment 155 (error growth from 0.5-3.5 to 3.5-9.5 hours) as a
    property of the filter: at the daily 09:00 origin the growth is -0.52 points because the first
    hours fall on the morning ramp, so compare an origin only with arms issued at the same hour of
    the day (C1 lead check).
  - >-
    Keep the winter night offset and recipe R in a production port: they still gain 0.056 points at
    leads under 24 hours when issued every 6 hours (C5).
  - >-
    Compare production-cadence forecasts with once-a-day forecasts over leads pooled across a whole
    day, never by lead band, because each lead band of a once-a-day origin sits at one hour of the
    day (C1, C4).
date: 2026-10-22
production_status: not-for-production
adopted_in: none
---

# Experiment 168: the 6-hourly cadence costs 0.08 points in the first day, mostly the delay

**Issued every 6 hours with power readings 3.5 hours old, the stage 2 forecast scores 0.076 points
worse than once a day over the first 24 hours of lead, and about the same beyond that.** The stage 2
forecast is arm D of [experiment
155](../00155-dp-bg-stage2-assembly-night-offset-and-error-correction/result.md): the convex demand
model with a Kalman filter on recent errors, driven by the ensemble-mean weather forecast, with a
winter night offset and recipe R added. Experiments 136 and 155 issued one forecast a day, 9 hours
after the 00:00 UTC ECMWF run. Production would issue one every 6 hours from the newest run, so the
weather forecast is 12 to 30 hours old. We re-issued each stored run at 12:00, 18:00, 00:00, and
06:00 UTC, with the filter reading only power readings that ended at least 3.5 hours before the new
origin, and scored the result on the same substations and year. The score is twice the mean pinball
loss over 11 quantiles, in points of a substation's effective capacity (lower is better), at the five
Primary substations D5, D10, D11, D14, and D16 in the year from July 2025.

The pre-registered contrast, C1, is 6-hourly arm D minus once-a-day arm D at the same run and lead,
pooled over leads 0.5 to 24 hours. It is +0.076 (99% interval from resampling calendar months +0.009
to +0.129), above the pre-registered +0.05, so the hypothesis is `refuted`. Over all leads to 147
hours it is +0.017 (-0.021 to +0.042). The 3.5-hour delay explains most of the first-day gap: the
same 6-hourly forecasts with no delay score 0.047 points better (C3, +0.047 for the delay; +0.020 to
+0.074), and 0.30 points better in the first 3.5 hours. The corrections still help: arm D scores
0.056 points below the filter alone at the production cadence (C5, -0.056; -0.091 to -0.025).

![Contrasts over leads under 24 hours](figure_1_headline.svg)

![Score by hour of the valid time](figure_2_by_hour.svg)

![Contrasts by origin](figure_3_by_origin.svg)

![The delay by lead band](figure_4_delay_by_lead.svg)

## Results

- **Method.** Each stored run of experiments 136 and 155 is re-issued at origins 3, 9, 15, and 21
  hours after its original 09:00 UTC origin. The weather input is the same run's ensemble mean and
  the GAM fit is the same, so only the filter's state and the lead change. The GAM's raw mean is
  experiment 136's stored mean minus its stored filter correction. Arm D's shift (offset and
  recipe R) is experiment 155's stored shift for the same run and valid time, and each arm's
  spread is refitted on its own tuning-month forecasts. Experiment 136's T-all fits, filter, and
  pooled filter noise are copied unchanged. No new modelling.
- **Reproduction control (passes).** An origin shift of zero hours reproduces the stored means to
  within 1.3e-6 points and the stored quantiles to within 1.5e-5 points at every substation.
- **Rows.** `fold_a_daily_gap_v1`, scored year, five decision substations (D9 beside them in the
  tables), outside the hand-filtered switching events. Leads run to 147 hours at all four origins
  and to 165 hours at fewer, because the stored window ends 168 hours after the original origin.
  The once-a-day arms have 464,363 rows and each 6-hourly arm 1,723,837.
- **Absolute scores over leads 0.5 to 24 hours.** Once-a-day arm D 2.306, once-a-day arm D with no
  delay 2.277, 6-hourly arm D 2.386, 6-hourly arm D with no delay 2.339, once-a-day arm A 2.364,
  6-hourly arm A 2.441, BG2 2.433, manual heuristic 3.506, climatology 3.801. Over leads 0.5 to 147
  hours: once-a-day arm A 2.774, arm D 2.738; 6-hourly arm A 2.797, arm D 2.763; BG2 2.771;
  manual heuristic 3.532; climatology 3.832. BG2, the manual heuristic, and climatology are
  once-a-day forecasts.
- **C1 (headline), 6-hourly D minus once-a-day D at the same lead.** Under 24 hours +0.076 (+0.009
  to +0.129; 5 of 5 substations, 10 of 12 months worse); all leads to 147 hours +0.017 (-0.021 to
  +0.042). By origin under 24 hours: 12:00 +0.028, 18:00 +0.101, 00:00 +0.091, 06:00 +0.085, each
  with an interval that includes zero except 18:00 and 00:00. Replicate months (even months of
  2024-07 to 2025-06): +0.041 (-0.020 to +0.095) under 24 hours, +0.021 (+0.007 to +0.040) over
  all leads.
- **C3, the 3.5-hour delay at the 6-hourly origins (no-delay forecasts subtracted).** Under 24 hours
  +0.047 (+0.020 to +0.074); leads 0.5 to 3.5 hours +0.299 (+0.184 to +0.420); 3.5 to 9.5 hours
  -0.011; 9.5 to 24 hours +0.009; all leads +0.014 (+0.004 to +0.026). By origin under 24 hours: 06:00
  +0.067 (+0.038 to +0.095), the others +0.034 to +0.045 with intervals that include zero. Replicate
  months +0.044 (+0.025 to +0.060) under 24 hours.
- **C7, the same delay at the once-a-day 09:00 origin.** Under 24 hours +0.029 (-0.018 to +0.081);
  leads 0.5 to 3.5 hours +0.260 (+0.114 to +0.413), close to the 6-hourly origins' +0.299.
- **C4 (post hoc), 6-hourly D minus once-a-day D at the same run and valid time.** Under 24 hours
  -0.024 (-0.110 to +0.029); all leads -0.019 (-0.058 to +0.004). The later origin has a shorter
  lead for the same valid time, so C4 favours the 6-hourly arm, and C1 and C4 bracket the cost of
  the production cadence.
- **C5, arm D minus arm A at the 6-hourly origins.** Under 24 hours -0.056 (-0.091 to -0.025); all
  leads -0.034 (-0.061 to -0.012); 5 of 5 substations. Winter nights -0.142.
- **Error of the mean forecast beside C1.** The mean absolute error contrast under 24 hours is
  +0.136 (+0.024 to +0.225) and over all leads +0.032 (-0.022 to +0.073), the same signs as the
  quantile score.
- **Lead check of experiment 155.** The 6-hourly arm D scores +0.021 points higher at 3.5 to 9.5
  hours than at 0.5 to 3.5 hours. The once-a-day arm D scores -0.522 points lower, because its
  first 3.5 hours of lead are 09:00 to 12:30 UTC, among the hardest hours of the day (figure 2). The
  difference in growth is +0.543 (+0.232 to +0.906) and fails the pre-registered limit of +0.02,
  but the once-a-day growth is set by the hour of the day, so the check cannot tell us about
  growth with lead.
- **Cold nights (daily mean temperature below 2 degrees Celsius, 3,186 once-a-day rows).** C1 is
  -0.146 (-0.562 to +0.146) and C3 is +0.123 (-0.002 to +0.143), against a prediction of at most
  +0.10 for C1: the interval is too wide to rank. Winter nights overall: C1 -0.473.
- **Predictions against outcome.** C1 +0.076 against +0.00 to +0.03 (a miss, high). C3 +0.047
  against +0.01 to +0.04 under 24 hours (just outside) and +0.299 against more than +0.04 in the
  first 3.5 hours (inside). C4 -0.024 against -0.02 to -0.06 (inside). The 06:00 origin has the
  largest C3 (+0.067) but not the largest C1 (18:00, +0.101), so that prediction missed.

## Tail calibration (scored year, five decision substations)

| Subset | Arm | q01 cov | q05 cov | q95 cov | q99 cov | Pinball q95 | Pinball q99 |
|---|---|---|---|---|---|---|---|
| 12-36 h | Daily D | 0.024 | 0.069 | 0.945 | 0.982 | 0.648 | 0.201 |
| 12-36 h | 6-hourly D | 0.025 | 0.072 | 0.948 | 0.983 | 0.666 | 0.206 |
| 12-36 h | BG2 | 0.009 | 0.036 | 0.948 | 0.985 | 0.699 | 0.242 |
| Cold nights | Daily D | 0.041 | 0.089 | 0.838 | 0.946 | 0.933 | 0.316 |
| Cold nights | 6-hourly D | 0.045 | 0.094 | 0.841 | 0.946 | 0.929 | 0.303 |
| Cold nights | BG2 | 0.019 | 0.033 | 0.761 | 0.866 | 1.347 | 0.788 |

- **The tails match the daily forecast's.** The 6-hourly q95 and q99 coverage at 12 to 36 hours is
  within 0.3 percentage points of the daily forecast's, and the cold-night q95 coverage stays near
  84%, as in experiment 155.
- **Curtailment cost.** The node-level curtailment metric and the saving against the manual heuristic
  need node forecasts, which this experiment does not produce, so neither is reported.

## Checks

- **Median minus mean forecast, per substation.** 0.00 points at every substation for both arms at
  12 to 36 hours, because the spread is centred on the mean.
- **Triggers that fired.** Trigger 2 fired: C1 (+0.076) contradicts the prediction (+0.00 to +0.03).
  The explanation tested is the hour of day: C4, C6 (6-hourly D with no delay minus once-a-day D,
  +0.031 under 24 hours), and C7 above. Trigger 7 fired: the GAM-K serving design depends on the
  headline, and the replicate months agree in sign (+0.041 under 24 hours). Triggers 1, 3, 4, 5, and
  6 did not fire: the reproduction control passed, the mean-error and quantile-score contrasts agree
  in sign, no substation leaves the sign of C1 when left out (substation contrasts +0.014 to
  +0.127), and no sister experiment exists for trigger 6. The audit's findings are triaged below.
- **Order of work.** The planning commit precedes the scoring script, but the build script had run
  once, and its reproduction control had been read, before the planning commit. No score had been
  read.

## Caveats and what to try next

- **The same-lead comparison is confounded with the hour of day, and the same-valid-time comparison
  favours the 6-hourly arm.** The true cost of the production cadence lies between C4 (-0.024)
  and C1 (+0.076). Only an arm issued at the same hours, with the same weather age, would close
  the gap, and that needs ensemble runs at 06:00, 12:00, and 18:00 UTC, which the project does not
  store.
- **Leads beyond 147 hours have fewer origins.** The stored window ends 168 hours after the
  original origin, so leads from 147 to 165 hours are not in the all-lead figures.
- **The monthly GAM refit is not re-timed.** A 00:00 or 06:00 origin on the first of a month uses
  the previous month's weights, where production would have used the new ones.
- **Arm D's shift is the stored shift.** The offset and recipe R are those fitted on the
  once-a-day forecasts' errors, not on the 6-hourly forecasts' errors.
- **Next.** Measure how soon after a half-hour ends production can read its power telemetry. These
  arms hold only delays of 0 and 3.5 hours, so the cost of a delay between them is not measured.
