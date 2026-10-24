---
id: '00328'
slug: lsc-regime-changes-detect-then-model-part-1
title: >-
  Does modelling permanent regime changes beat retraining on a recent window?
theme: lsc-detectors
also_relevant_to:
  - retraining-and-training-history
status: done
kind: other
size: >-
  complex on two of the five triggers, so it gets the plan, a Sonnet code review, a Sonnet
  write-up review and a Sonnet prose review (Sonnet only, by the maintainer's instruction).
  Stored: private tables under data/ml-experiments/00328-*/ only (the regime log is a table in
  that folder). Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible (three detectors, four modelling arms), each fixed below before any score is read.
  Callers: this folder's scripts, which call the reviewed `studies.dp_gam` functions and the code
  of experiments 295 and 297 unchanged; the only change under `packages/` is a reviewed addition to
  `studies.ecr_reader` that returns the connection date.
hypothesis: >-
  A permanent change in the configuration of an electricity network (a large new generator, a new
  factory) can be found offline from a Primary's own history, and a forecast that keeps the effects
  of the old behaviour fitted on the full history and adds new effects that exist only from the
  change onward scores better after the change than the same forecast refitted on a recent window
  only.
prediction: >-
  Written before any score is read; each can fail. (Detector) The rolling lagless GAM finds at
  least 80% of injected 10%-of-load steps within 60 days at one false change or fewer per
  Primary-year. The weekly power-mode histogram test and the rolling GAM both date D8's break
  (2023-08-15) within 30 days; Embedded Capacity Register (ECR) connection dates agree with at least
  one detected change at D8. (Forecast) In the 90 days after a detected regime change, arm C or D
  beats arm B (recent window, retrained monthly) by at least 0.05 points on the quantile score with
  a 95% interval above zero, and q99 coverage moves towards nominal, without raising the rows
  outside the 90 days by more than 0.01 points. At D8 the as-of band widening of experiment 264
  (score -9% to -19%) is matched or beaten by arm C or D. It fails if the recent-window refit
  matches arms C and D.
decision_rule: >-
  Five plan items, as the research-default rule asks, written before any score is read. (1) Family
  and base: Primary demand, monthly-refitted GAM_ONLY with Kalman filter on past weather (ERA5 and
  CAMS, lab conditions) as the stage 2 base, because the full Primary demand research default 3
  chain (ensemble members, heads, solar summary) cannot be refitted per arm on CPU in the night's
  budget. This is why `kind` is `other`: a win here is a candidate for a later `change` experiment
  that ports the arm onto Primary demand research default 3. (2) Arms: A the base on all history,
  B the base on the last 24 months, C1 two-stage regime-stacked GAM, C2 joint regime-stacked GAM,
  D the better of C1 and C2 plus a Kalman reset at each detected change, E (optional) a regime
  indicator feature. (3) Tier: a new fitted step (0.02 points minimum gain at pooled level; the
  90-day prediction above is judged at 0.05). (4) Pre-registered arms: A, B, C1, C2, D; E is
  descriptive. (5) Controls: injected permanent steps and ramps (known answer), a placebo with no
  regime change at the 14 Primaries without injection (arm C must equal arm A there within 0.01
  points), and a leakage check that every detector input and every regime date lies at or before
  the forecast origin. Detection: delay in days, false changes per Primary-year, shift in change
  date, by detector, with intervals that resample Primaries and months. Forecast: 95% intervals
  that resample whole months and whole regime changes; pooled non-inferiority at the other
  Primaries with margin 0.01 points; D8 scored on its own. The verdict is `supported` when the
  forecast prediction holds, `partly supported` when the detector prediction holds and the forecast
  prediction does not, `not supported` when the recent window ties arms C and D, and `inconclusive`
  when fewer than 3 real or injected regime changes can be scored.
benchmark: >-
  Arm B (the base on a recent 24-month window, retrained monthly); the as-of widening of
  experiment 264 at D8; `manual_heuristic` and `climatology` as references.
row_set_id: >-
  Own rows (not comparable with the leaderboard): the 15 Primaries of experiment 106's cache,
  monthly forecast origins, 1 to 10 day leads on past weather, rows outside the hand-filtered
  load-shift spans.
metric: >-
  Detection delay in days, false changes per Primary-year, mean absolute error of the median and
  the quantile score in points of effective capacity, twCRPS, q01 and q99 coverage.
code: ml-experiments/00328-lsc-regime-changes-detect-then-model-part-1/
date: 2026-10-24
production_status: not-for-production
adopted_in: none
verdict: not supported
best_arm: >-
  None. The best pre-registered arm, D2 (regime effects, bands from the regime's own residuals,
  filter reset), scores 0.089 points worse than arm B (last 24 months) in the first 90 days after
  an injected 10% step (98.75% interval 0.022 to 0.167 worse). Post hoc arm D2t (trailing bands)
  ties arm B there (B minus D2t +0.009, interval -0.023 to +0.042).
headline: >-
  Both detectors alarm too late and too rarely for regime-specific effects to help: at one false
  change per Primary-year the rolling lagless GAM finds 26% of injected 10% steps within 60 days
  and 56% of 25% steps, and the weekly-histogram detector finds 0% and 10%. A GAM refitted on the
  last 24 months scores as well as every regime-aware arm in the first 90 days after a step.
lessons:
  - >-
    A recent-window refit already handles an injected permanent step as well as the regime-stacked
    arms do, so regime machinery is not needed for the forecast on this base.
  - >-
    Estimating bands from only the residuals since a detected change costs about 0.1 points on
    Primaries with no change (arm Ar minus A), because false alarms leave a short, noisy sample.
  - >-
    Freezing the old effects at a detected change (C1) is 1.4 points worse on Primaries with no
    injected change, because every false alarm leaves a stale fit.
production_status: not-for-production
adopted_in: none
take_home_message: >-
  Adding effects for a detected permanent change does not forecast better than refitting the
  model on the last 24 months, on a model fed past weather. The detectors find a 10% step of a
  Primary's load within 60 days only 26% of the time at an acceptable false-alarm rate, so there
  is little time after the alarm for the extra effects to help. Regime changes are therefore a
  retraining-cadence question, and the detector is useful at most for substation cards.
advice:
  - >-
    Handle a permanent step in a Primary's load by refitting on a recent window, not by regime
    effects: B matched or beat C1, C2, D1, and D2 in the first 90 days (injected 10% and 25%
    steps).
  - >-
    Do not estimate forecast bands from the residuals since a detected change alone: it costs
    0.098 points on Primaries with no change (Ar minus A, real records).
  - >-
    Do not rely on a CUSUM of the rolling lagless GAM's daily residual for steps under 15% of
    load: it finds 26% of 10% steps within 60 days at one false change per Primary-year (detection
    table).
---

**Take-home: adding effects for a detected permanent change does not beat refitting on the last 24 months.**
The regime arms lose to a recent-window refit in the first 90 days after an injected step, because
both detectors alarm late and miss small steps. At D8 the extra effects change the first-90-day
score by no more than the noise. The verdict is `not supported`.

We asked whether a substation that gains a lasting new behaviour (a large generator, a new
factory) is forecast better by keeping the old effects and adding regime-specific effects than by
refitting on a recent window. The model is a lagless additive model (GAM) fitted to past weather
(ERA5 and CAMS), with a one-day-ahead level filter, so this is a lab setting and not Primary
demand research default 3. The step is injected as a permanent 10% or 25% of load at 14 Primaries.
D8's gas engine is the one real change. Fewer than three real changes exist, so the real-record
evidence is D8 alone.

![Recall of 10% steps against false changes](detection_curve.svg)

![B minus each regime arm, injected steps](headline_contrasts.svg)

![D8 by days since the break](d8_by_group.svg)

## Checks

- Median minus mean forecast: not applicable. Every arm forecasts quantiles and the median directly;
  no mean forecast exists. The mean absolute error of the median is printed beside every score in
  `printed_report.txt` and points the same way as the quantile score.
- Audit triggers that fired: (1) the pre-registered placebo (C arms against A at 14 Primaries with
  no change) failed at +0.054 (C2) and +1.362 (C1); the explanation, tested, is the regime-band
  rule (arm Ar minus A is +0.098) and stale frozen fits after false alarms, and the oracle placebo
  C2o equals A exactly (+0.000). (5) Nested arm A beats C1 and C2 on the real records. A Sonnet
  audit (Opus was excluded for this session) is triaged in the pull request.
- Leaving D8 out is not applicable to the injected contrasts, which exclude D8.

## Detection

- Thresholds were fixed on 7 tune Primaries (at most 1 false change per Primary-year) and read on 7
  held-out Primaries with even-month onsets. The first threshold grid never reached 1 false change
  per Primary-year, so the grid was widened before any recall was read.
- Rolling GAM (CUSUM 64): recall within 60 days of 10% steps 0.26 (0.12 to 0.42), of 25% steps 0.56
  (0.36 to 0.70), of 30-day ramps to 10% 0.21; median delay 40 days (10%) and 19 days (25%).
- Weekly histogram (penalty 128): recall 0.00 for 10% steps and ramps, 0.10 for 25% steps.
- D8 (break 2023-08-15): rolling GAM dates the change 2023-08-20 but alarms on 2023-12-06 (113 days
  late); histogram dates it 2023-08-17 and alarms 2023-10-19 (65 days late).
- The Embedded Capacity Register holds 3 assets of 1 MW or more connected since 2020-06-01 at these
  Primaries, none at D8 near 2023-08, so it gave no independent check of the D8 date.
- False changes count real load shifts, so the rates are upper bounds.

## Forecast

- Quantile score: twice the mean pinball loss over 11 levels (0.01 to 0.99), in points of capacity.
  Intervals are 98.75% (four pre-registered contrasts) and resample whole injected worlds.
- Injected 10% steps, first 90 days: A 3.273, B 3.270, C2 3.389, D2 3.359. B minus C2 -0.119
  (-0.200 to -0.049); B minus D2 -0.089 (-0.167 to -0.022); C1 and D1 are about 1.4 points worse.
- Injected 25% steps, first 90 days: A 3.945, B 3.978, C2 4.221, D2 4.161; B minus C2 -0.243
  (-0.334 to -0.158).
- Post hoc, trailing bands (C2t, D2t): B minus C2t +0.003 (-0.027 to +0.035) at 10%, -0.078 (-0.134 to
  -0.022) at 25% in the first 90 days, and +0.051 (+0.026 to +0.077) at 25% after day 90.
- D8, first 90 days after the break: B minus C2 -0.040 (-0.262 to +0.154); post hoc B minus C2t
  +0.065 (+0.037 to +0.129); all rows after the break B minus C2t +0.150 (-0.057 to +0.372). The
  q99 exceedance after the break is 0.031 for arm B and 0.031 for C2t against a nominal 0.010, so
  no arm is calibrated at D8.
- Primaries with no injection (14, real records): B minus A -0.021 (-0.037 to -0.004), C2t minus A
  -0.030 (-0.047 to -0.013), D2t minus A -0.034 (-0.052 to -0.017), against the 0.01 margin.
  C2 minus A is +0.054 (+0.018 to +0.101) and fails it.
- Tail calibration: q01 and q99 exceedance and pinball loss are in `printed_report.txt`. Twice the
  mean pinball loss at the 0.01 and 0.99 levels is the tail score. The tail-weighted continuous
  ranked probability score and the curtailment cost were not computed.

## Caveats and next steps

- Lab shortcuts: past weather in place of ensemble members, a filter that sees yesterday's
  reading, a lagless GAM base with a simple fixed-gain level filter instead of the effects filter.
  The experiment 264 as-of band widening at D8 was not reproduced, so the success criterion for D8
  against that comparator is untested.
- A plausible reason for the tie, not tested here, is that the level filter absorbs a step within
  days, leaving little for regime effects to add. A follow-up could repeat the contrast without the
  filter.
- Next: if a regime-aware arm is revisited, use trailing bands (C2t), a detector that alarms
  earlier, and the full Primary demand research default 3.
