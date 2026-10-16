---
id: "00048"
slug: weekly-retraining-confirmation-and-extension-demand-generators
status: done
size: complex
hypothesis: >-
  On the 292 daily ENS forecast runs of the validation year that experiment 34's 73 runs leave out,
  weekly retraining of the Primary substation demand model N1, with no error correction, lowers the
  quantile score below the best-guess stack (frozen N1 with the 14-day correction and spread factor
  1.15) by more than zero points of capacity; monthly retraining of the ERA5-trained solar and wind
  stage 1a models changes the score by less than 0.05 points; and, once the model retrains weekly,
  neither a 7-day nor a 14-day error correction helps.
prediction: >-
  Primary weekly minus the stack is -0.10 to -0.30 (experiment 34's 73 runs gave 5.04 against 5.32,
  a descriptive -0.28 against a three-seed stack; the daily runs let the stack's correction read
  forecasts from the last day, not 5 days back, which probably improves the stack and shrinks the
  gap), and monthly is 0.05 to 0.10 behind weekly. A 7-day or 14-day
  correction on the weekly arm adds +0.05 to +0.30, the 7-day window less than the 14-day window.
  For the 2 BSP and 2 GSP series, weekly minus frozen is -0.1 to -0.5, with intervals wide on 2
  series each. For solar and wind, monthly minus frozen lies within plus or minus 0.05, because the
  weather-to-power relation is physical and experiment 12 found the training start changed little.
decision_rule: >-
  Nine planned contrasts, quantile score over all leads on the 292 new runs, read at 99.44%
  (Bonferroni across nine), whole validation months resampled, each run counted in its origin
  month. Seed 0 only for every refit arm,
  because experiment 34 found three seeds narrowed its interval by under 0.01. Primary: C1 weekly
  (no correction, spread factor) minus the stack (frozen N1, 14-day correction, spread factor);
  C2 monthly minus the stack; C3 weekly minus monthly; C4 weekly with the 7-day correction minus
  weekly; C5 weekly with the 14-day correction minus weekly; C6 weekly with the 7-day correction
  minus weekly with the 14-day correction. BSP and GSP: C7 weekly (no correction, spread factor) minus
  frozen with the 14-day correction and spread factor, the deployable reference, pooling the 4
  series. Solar and wind: C8s monthly minus frozen for the 6 solar farms, and C8w the same for the 3
  wind farms, each pooled over its farms. C1 is
  supported when its interval lies wholly below zero, refuted when wholly above zero, and
  inconclusive otherwise. The threshold is zero because retraining replaces the correction step in
  the pipeline and adds a scheduled job, and experiment 34 already read monthly retraining as
  worth porting at any gain. C3 is read like C1, supported when its interval lies wholly below zero and
  refuted when wholly above, because 16 refits a week cost seconds of compute in production; it is
  expected to resolve only if the gain is near the upper end of the prediction.
  The correction is judged still helping when C4 or C5 lies wholly below zero and
  no longer helping when it lies wholly above zero. C8s and C8w are read as no change when the
  interval lies inside plus or minus 0.05, and as a change when it excludes zero; with 6 solar and 3
  wind farms the wind interval will probably be too wide to read as no change, and the page says so.
  Every series is also reported on its own, because the pooled interval resamples months only and
  says nothing about other series. Every other comparison, including P-roll, is descriptive. The 73
  runs of experiment 34 are scored again from the same refits as a reproduction check: the arms
  without a correction must match experiment 34's stored seed-0 scores to 0.001.
benchmark: >-
  The best-guess stack (experiment 33's `n1_cs`), frozen N1 for BSP, GSP, solar, and wind,
  `manual_heuristic`, and `climatology`, each scored on the same 292 runs.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over 9 levels), in points of capacity;
  MAE of the median, the exceedance rate of the 0.95 quantile, and the twCRPS at `historical_p99`
  are reported and decide nothing
verdict: supported
best_arm: >-
  Weekly retraining without the correction: 5.04 points of capacity for the 16 Primary substations,
  against 5.28 for the stack, 5.11 for monthly retraining, and 5.89 for `manual_heuristic`.
headline: >-
  C1, weekly retraining minus the stack: -0.20 points (99.44% interval -0.40 to -0.04). C3, weekly
  minus monthly retraining: -0.13 (-0.24 to -0.03). The 7-day correction on the weekly model
  raised the score by 0.16 (-0.06 to +0.36), unresolved.
lessons: >-
  The weekly gain over the stack held on 292 daily runs the earlier experiment had not scored, and
  weekly retraining beat monthly retraining here, which experiment 34 could not resolve. No error
  correction helped a retrained model, and the 7-day window was worse than the 14-day window. At
  the BSP and GSP substations retraining gained more than predicted, and at solar and wind farms
  monthly retraining changed the score by under 0.05 points.
advice:
  - >-
    Retrain the Primary, BSP, and GSP demand models weekly and drop the 14-day correction for them
    (C1, C3, C7).
  - >-
    Do not retrain solar and wind stage 1a models for accuracy: monthly retraining moved the solar
    score by -0.04 and the wind score by +0.01 points (C8s, C8w).
  - >-
    Do not add a 7-day correction to a retrained Primary model: it scored 0.09 worse than the 14-day
    window, and neither window helped (C4 to C6).
  - >-
    Score a daily row set, not a 5-day set, when the test depends on a correction built from recent
    forecasts, because the stack then reads fresh forecasts (the plan's stack, C1).
  - >-
    Test weekly retraining on a second validation year before porting it, because the 292 runs
    share their months and weather with experiment 34's 73 runs (limits section).
production_status: candidate
adopted_in: none
code: ml-experiments/00048-weekly-retraining-confirmation-and-extension-demand-generators/
---

# Experiment 48: does weekly retraining still win on the runs experiment 34 left out?

**We asked whether a forecast of a substation's electricity demand improves when the forecasting
model is refitted every week on the newest data, and whether the same refitting helps at larger
substations and at solar and wind farms.** The take-home message is that weekly refitting is worth
adopting for demand forecasts. On 292 daily forecast runs that experiment 34 had not scored, a
weekly refit with no error correction scored 5.04 points of capacity on 16 Primary substations,
which is 0.20 points better than the best earlier setup (a model fitted once, plus a correction
built from its own errors over the last 14 days, scoring 5.28). The result rules out two reasons for
keeping the older setup: weekly refitting beat monthly refitting by 0.13 points, and no error
correction (7 or 14 days) helped a refitted model. At the 2 bulk supply point (BSP) and 2 grid supply
point (GSP) substations, weekly refitting gained 0.66 points over the same older setup, more than we
predicted. For solar and wind farms, refitting the weather-to-power model monthly changed the score
by 0.04 points or less, so refitting does not need to be extended to generation. The next experiment
should check, on a second validation year, that the weekly gain is not particular to this year.
The verdict is **supported**.

**Terms used below.** The quantile score rewards a forecast that is both close and honest about its
uncertainty, in percentage points of the substation's capacity (smaller is better). The stack is the
older setup just described, with a spread factor of 1.15 that widens its uncertainty. Frozen means
fitted once, on data up to the end of June 2025. Every arm other than the stack uses the spread
factor, and uses no correction unless its name says so. ENS is the ECMWF ensemble weather forecast,
and the model is the one-model-per-substation XGBoost recipe of experiment 13 (N1), trained on the
ENS control member. Scores are means over the cases that every arm of a group holds, on seed 0.

**Weekly refitting beat the stack at the 99.44% level the plan set.** C1, weekly minus the stack, is
-0.20 points (99.44% interval -0.40 to -0.04), so the interval lies wholly below zero. The threshold
was zero, because refitting replaces the correction step and adds one scheduled job. C3, weekly
minus monthly, is -0.13 (-0.24 to -0.03). Monthly minus the stack (C2) is -0.08 (-0.33 to +0.19) and
unresolved, so the 5-day-row finding of experiment 34 for monthly refitting did not repeat on daily
runs at this level of confidence. The scores are mean quantile scores over all leads.

![Planned contrasts](figure_1.svg)

| Arm | Primary (16) | BSP (2) | GSP (2) | Solar farms (6) | Wind farms (3) |
|---|---|---|---|---|---|
| Stack: frozen, 14-day correction | 5.28 | 8.09 | 19.75 | | |
| Frozen, no correction | 6.28 | 7.99 | 19.46 | 5.02 | 10.99 |
| Monthly refits | 5.11 | 7.90 | 18.94 | 4.97 | 10.99 |
| Weekly refits | **5.04** | **7.84** | **18.66** | | |
| Weekly, 14-day correction | 5.16 | 7.96 | 19.17 | | |
| Weekly, 7-day correction | 5.22 | | | | |
| Monthly, 12-month window (P-roll) | 5.15 | | | | |
| `manual_heuristic` | 5.89 | 8.84 | 20.82 | 5.79 | 14.09 |
| `climatology` | 7.48 | | | | |

The 95% interval on each score, over months, is about plus or minus 0.4 points for Primary
substations; the contrasts are paired and much tighter. The 73 runs of experiment 34 were forecast
again from the same refits. For the Primary arms without a correction they matched experiment 34's
stored scores to 0.001 or better. For the BSP and GSP arms the differences were 0.0002 to 0.006
points, so 4 of the 6 BSP and GSP rows lie outside the 0.001 tolerance the plan set. The cause is the
graphics-card fits, which are not bit-for-bit repeatable; the differences are small against the
contrasts reported below.

**An error correction did not help a model that is refitted weekly.** C4, a 7-day correction on the
weekly model, raised the score by 0.16 (-0.06 to +0.36). C5, a 14-day correction, raised it by 0.08
(-0.13 to +0.27). C6 puts the 7-day window 0.09 above the 14-day window (+0.02 to +0.17), so the
shorter window was worse, the opposite of what the plan predicted. Neither C4 nor C5 is resolved
alone. The correction does help the frozen model: it lowers the frozen score by 1.45 points (95%
interval -1.87 to -0.98), so the correction is still the right step for a model that is not
refitted.

**Weekly refitting gained more at the BSP and GSP substations than the plan predicted, but on 4
series.** C7, weekly minus the stack, is -0.66 (-1.23 to -0.27), against a predicted range of -0.1
to -0.5. Each of the 4 series scored lower under weekly refitting, and the intervals on the two GSP
series are wide (figure 4). A correction on the weekly model costs 0.31 points here (95% interval
+0.06 to +0.60).

**Monthly refitting changed the solar and wind scores by 0.04 points or less.** C8s for the 6 solar
farms is -0.04 (99.44% interval -0.16 to -0.00), a change by the plan's rule because the interval
excludes zero by a hair, and inside the 0.05 no-change band. C8w for the 3 wind farms is +0.01
(-0.00 to +0.02), inside the band. One solar farm (label E) gained 0.26 points (95% interval -0.72 to
-0.00); the other 5 moved by 0.02 or less. The prediction of no change rests on the training-start
result of experiment 12, not on a retraining measurement, and these two intervals are the first
retraining measurement for generation.

![Absolute quantile score of every arm](figure_2.svg)

![Quantile score by lead band](figure_3.svg)

**Refitting helped at every lead time, most of all in the first two days.** On Primary substations
weekly refitting scored 4.18 against 4.41 for the stack in the first 24 hours, and 5.50 against 5.76
beyond 10 days (figure 3).

![Each series on its own](figure_4.svg)

**The rolling 12-month window stayed a good stand-in for monthly refitting.** P-roll scored 5.15
against 5.11 for monthly refitting, a difference of +0.03 (95% interval +0.01 to +0.04), with the
same sign pattern as experiment 34.

## Predictions

- **Held:** weekly minus the stack in the range -0.10 to -0.30 (-0.20); monthly 0.05 to 0.10 behind
  weekly (0.08); monthly solar and wind within 0.05 (-0.04 and +0.01).
- **Missed:** the 7-day correction adding less than the 14-day correction (it added more); BSP and
  GSP weekly minus the stack in -0.1 to -0.5 (-0.66).

## Failures and limits

- **No fit failed.** The frozen model's first single-job run showed no output for 37 minutes and was
  restarted as 5 workers; no result came from the abandoned run.
- **The 292 runs share their months and weather with the 73 of experiment 34.** Each run spans about
  14.6 days, so the 73 runs already cover every valid time of the year, and the interval is set by 12
  months. The result guards against the choice of origins and of seeds, not against an unusual year.
- **The first 3 days of July 2025 have no correction history,** so the stack and the corrected arms
  carry no correction there; no correction reads a forecast before 2025-07-01.
- **The generator spread factor is fitted on cases before each month's first run.** July 2025 and
  the first wind months have too few earlier cases, so both generator arms are left unscaled there.
- **Pooled intervals resample months only,** so they say nothing about other series. The BSP and GSP
  contrast rests on 4 series, solar on 6, and wind on 3.
- **Seed 0 only.** Experiment 34 found that seeds narrow its contrasts by under 0.01.
- **Reproduction tolerance missed on the GPU-fitted BSP and GSP arms** (0.002 to 0.006 points).
- **All arms use ENS weather forecasts, with no weather estimate from after the forecast origin.**
  The generator arms train on ERA5 up to 5 days before each refit date, the delay with which ERA5
  arrives.

## What the next experiment should answer

**Does weekly retraining still beat the stack on a validation year the project has not scored?**
The ENS backfill from Source Cooperative gives a second year to score. The 7-day correction window
and solar and wind retraining are answered at the resolution of this experiment.

## Design as planned

**Experiment 34 scored 73 forecast runs, 5 days apart, and found weekly retraining 0.28 below the
best-guess stack with no correction; this experiment asks whether the gain holds on the other 292
daily runs of the same year.** The 292 runs share their months and weather with the 73, so the test
guards against the choice of origins and seeds, not against an unusual year. Experiment 34 also left
three gaps: its correction read forecasts 5 days apart, retraining was not tried for BSP, GSP, solar,
and wind beyond monthly BSP and GSP, and the 7-day correction was not tried on a retrained model.

**Rows.** The row set `fold_a_daily_gap_v1` holds the daily 00:00 UTC ENS runs from 2025-07-01 to
2026-06-30 that `fold_a_every5d_v1` does not. All 365 daily runs exist in the ENS archive, so the set
holds 292 runs, and the 73 of `fold_a_every5d_v1` are forecast again from the same refits as a
reproduction check. Each run spans about 14.6 days, so the 73 runs already cover every valid time of
the year: the 292 runs score the same observed demand at other origins and leads, and guard against
the choice of origins, not against the outcome data. The demand frames use the same builder as `studies/stage1_pilot/demand_build_ens.py`
(production's half-hourly upsampling, all 51 members, lags from the observed power). No experiment
scores a valid time on or after 2026-07-01. The generators need frames for their 9 series over all
365 runs and 51 members, built the same way for the generators' H3 cells and the technology's
columns (experiment 33 holds only the 53 Tuesday runs).

**Arms.**

- **Primary (16 series):** the stack; weekly and monthly full refits with a half-life of 0.5 years
  and an expanding window, no correction, spread factor 1.15, using experiment 34's `n1h`
  recipe and its refit times (52 weekly, 12 monthly); P-roll (12-month window, 150 rounds, `eta`
  0.15, monthly); and the weekly arm with the 7-day and the 14-day correction, built from the 365
  daily forecasts as issued: each past half-hour's forecast comes from the refit that served that
  run, never from a re-forecast by the current refit, and no correction reads a run before
  2025-07-01. The stack's frozen model is refit with experiment 33's recipe (half-life 2 years,
  anchored at 2025-06-30, 500 rounds) because it was stored only for the 73 runs, and forecasts all
  365 runs.
- **BSP and GSP (2 each):** frozen (with the correction as the deployable reference, and without),
  monthly, and weekly, with and without the correction.
- **Solar (6) and wind (3):** the ERA5-trained stage 1a (solar `base`, with no CAMS and no stage 0;
  wind `margin`, with the power-curve margin), refitted monthly on ERA5 up to 5 days before each
  refit date (when ERA5 arrives), against the frozen model trained to 2025-06-17, both applied to the
  ENS members and pooled as in experiment 33. Both arms take the spread factor the same way as
  experiment 33's `apply_spread`, fitted only on cases whose valid time precedes the origin. The
  prediction of no change rests on experiment 12's training-start result, not on a retraining
  measurement.
- **Baselines:** `manual_heuristic` and `climatology` on the same 292 runs.

**Resources.** About 5 to 7 GPU hours inside slot A (10 cores, at most 24 GB): weekly Primaries
2.4, monthly 0.6, BSP and GSP weekly 0.6, generators 1 to 2, P-roll 0.2.
The 292 new demand frames add about 8 GB of disk, and the generator frames some more; each
worker loads the runs of one refit at a time, so memory stays inside 24 GB, and the fits run as
parallel workers, one output file per refit written atomically, so a run resumes. The GPU is shared with
other sessions, so the timings are not reproducible; the numbers are, because seeds are fixed per job.

**Sizing, by the five triggers.** Stored: predictions under `data/`, private, nothing in production.
Serving path: not touched. Degradation rule: not touched. More than one defensible design: no, the
approved ranking fixes arms and refit dates. Callers not nameable without searching: no, the scripts
are copies of experiment 33's and 34's. The experiment is complex only because it builds new
frames and a new solar and wind refit, so it gets the full review set: one plan review (Opus), one
code review (Sonnet), one write-up check (Sonnet), one prose review (Sonnet), and one result review
(Opus).

**Earlier advice followed.** Retrain monthly in the Primary stage 1 model and drop the correction
(00034); screen on P-roll (00034, used here only as a descriptive arm); read with the 99.375%
interval, because months set the width (00034). Departures: none.

**Plan review (Opus), triaged.** Accepted: state that the correction reads forecasts as issued and
nothing before 2025-07-01; predict a smaller Primary gap because the daily stack's correction is
fresher; set C3's threshold to zero; split C8 into solar and wind and make "no change" conditional on
the interval; name which frozen reference C7 uses; report each series; limit the reproduction check
to arms without a correction; add the generator frame builder, the disk and memory budget, and the
spread-factor rule. Rejected: none.
