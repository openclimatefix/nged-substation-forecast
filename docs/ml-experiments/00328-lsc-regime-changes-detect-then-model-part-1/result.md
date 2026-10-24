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
  defensible (two statistical detectors plus dates from the Embedded Capacity Register, and five
  pre-registered arms), each fixed below before any score is read.
  Callers: this folder's scripts, which call the reviewed `studies.dp_gam` functions and the code
  of experiments 295 and 297 unchanged. No change under `packages/`; the Embedded Capacity Register
  connection dates come from experiment 191's stored asset table.
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
  filter reset), scores 0.088 points worse than arm B (last 24 months) in the first 90 days after
  an injected 10% step (98.75% interval 0.038 to 0.146 worse). Post hoc arm D2t (trailing bands)
  ties arm B there (B minus D2t +0.009, interval -0.023 to +0.042).
headline: >-
  Both detectors alarm too late and too rarely for regime-specific effects to help: at one false
  change per Primary-year the rolling lagless GAM finds 26% of injected 10% steps within 60 days
  and 56% of 25% steps, and the weekly-histogram detector finds 0% and 10%. A GAM refitted on the
  last 24 months scores as well as every regime-aware arm in the first 90 days after a step.
lessons:
  - >-
    A recent-window refit handled an injected permanent step as well as the regime-stacked arms
    did, so this base needs no regime machinery for the forecast.
  - >-
    Setting the bands from only the residuals since a detected change lowers the score by about 0.1
    points on Primaries with no change (arm Ar minus A), because false alarms leave a short, noisy
    sample.
  - >-
    Freezing the old effects at a detected change (C1) is 1.4 points worse on Primaries with no
    injected change, because every false alarm leaves a stale fit.
take_home_message: >-
  Adding effects for a detected permanent change does not forecast better than refitting the
  additive model on the last 24 months, in a lab setting where the model is given past weather. The
  detectors find a 10% step of a Primary's load within 60 days only 26% of the time at an
  acceptable false-alarm rate, so there is little time after the alarm for the extra effects to
  help. On this base, handling a regime change comes down to how often the model is refitted. The
  rolling-GAM detector might still help when building substation cards, which this experiment did
  not test.
advice:
  - >-
    Handle a permanent step in a Primary's load by refitting on a recent window, not by regime
    effects: B matched or beat C1, C2, D1, and D2 in the first 90 days (injected 10% and 25%
    steps).
  - >-
    Do not estimate forecast bands from the residuals since a detected change alone: it worsens the
    score by 0.098 points on Primaries with no change (Ar minus A, real records).
  - >-
    Do not rely on the rolling lagless GAM detector (a cumulative sum of its daily forecast
    errors) for steps under 15% of load: it finds 26% of 10% steps within 60 days at one false
    change per Primary-year (detection
    table).
---

**Take-home: adding effects for a detected permanent change does not forecast better than refitting
the model on the last 24 months.**
We injected a permanent step of 10% or 25% of load into the recorded demand of 14 Primary
substations (the substations that feed local distribution feeders), and compared forecasts that add
new effects
after a detected change with forecasts refitted on the last 24 months (arm B). Arm B scored as well
as or better than every pre-registered regime arm in the first 90 days after a step, because the two
detectors raise their alarms late and miss small steps. At D8, the Primary with a real gas-engine
connection, the difference in the first 90 days was smaller than its interval (B minus C2 -0.040,
interval -0.262 to +0.154). The verdict is `not supported`, which means the hypothesis that regime
effects beat a recent-window refit did not hold.

The forecast model is a lagless generalised additive model (GAM) fitted to past weather (ERA5 and
CAMS), with a one-day-ahead level filter, so the result applies to a lab setting and not to Primary
demand research default 3. A "lagless" model reads no recent power. D8's gas engine is the only real
change we identified in the 15 Primaries' records, so the real-record evidence is D8 alone.

**Arm key.** A is one fit on all history, B one fit on the last 24 months, C1 a fit frozen at the
detected change plus new effects, and C2 a joint fit of old and new effects. D1 and D2 are C1 and C2
with the level filter reset at the change. Suffixes mark post hoc variants: `t` uses trailing
365-day bands, `o` uses the true change date (an oracle), `r` is arm A with bands from only the
residuals since the detected change, and `n` removes the level filter.

![Recall of 10% steps against false changes](detection_curve.svg)

![B minus each regime arm, injected steps](headline_contrasts.svg)

![D8 by days since the break](d8_by_group.svg)

## Controls and audit

- Median forecast against mean forecast: not applicable. Every arm forecasts quantiles and the
  median directly;
  no mean forecast exists. The mean absolute error of the median is printed beside every score in
  `printed_report.txt` and points the same way as the quantile score.
- Audit triggers that fired: (1) the pre-registered placebo (C arms against A at 14 Primaries with
  no change) failed at +0.055 (C2) and +1.364 (C1); the tested explanation is the regime-band
  rule (arm Ar minus A is +0.098) and stale frozen fits after false alarms. (5) nested arm A, which
  C1 and C2 contain, beats both on the real records.
- A Sonnet audit (Opus was excluded for this session) re-derived the 25% step contrasts from the
  stored forecasts and matched. Triage: the one-day look-ahead in `known_regime` (an alarm raised on
  the origin day) was fixed and every score re-run; the filter subtracting a step twice once a
  regime is known was tested with the no-filter arms below; the bootstrap clusters of one Primary
  and one onset were too fine, so the headline intervals now cluster by onset (eight clusters),
  which
  widens them. The oracle placebo C2o equals A by construction when no true change exists, so it
  checks the plumbing only.
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
- The Embedded Capacity Register holds 3 generators of at least 1 MW connected since 2020-06-01 at
  these
  Primaries, none at D8 near 2023-08, so the Register gave no independent check of the D8 date.
- False changes count real load shifts, so the rates are upper bounds.

## Forecast

- Quantile score: twice the mean pinball loss over 11 levels (0.01 to 0.99), in points of capacity.
  Intervals are 98.75% (four pre-registered contrasts) and resample the eight onset months of the
  injected worlds, or the calendar months of the real records.
- Injected 10% steps, first 90 days: A 3.273, B 3.270, C2 3.389, D2 3.358. B minus C2 is -0.118
  (-0.200 to -0.059); B minus D2 is -0.088 (-0.146 to -0.038); C1 and D1 are about 1.4 points worse.
- Injected 25% steps, first 90 days: A 3.945, B 3.978, C2 4.231, D2 4.171; B minus C2 -0.252
  (-0.320 to -0.181).
- Post hoc, trailing bands (C2t, D2t): B minus C2t +0.004 (-0.034 to +0.046) at 10% and -0.081
  (-0.183 to +0.029) at 25% in the first 90 days, and +0.051 (+0.011 to +0.092) at 25% after day 90.
- Post hoc, no level filter on any arm: the level filter matters more than the regime effects.
  Without it, C2 against the no-filter recent window (Bn) is +0.015 (-0.353 to +0.379) at 10% and
  +0.782 (-0.033 to +1.508) at 25% in the first 90 days, and +0.198 (+0.066 to +0.334) and +0.859
  (+0.419 to +1.277) after day 90. The true-date oracle gains +1.331 (+0.757 to +1.868) at 25%.
  Every
  filtered arm still scores far better than every unfiltered arm (arm B 3.978, arm Bn 8.047 at 25%
  steps, first 90 days).
- D8, first 90 days after the break: B minus C2 -0.040 (-0.262 to +0.154); post hoc B minus C2t
  +0.065 (+0.037 to +0.129); all rows after the break B minus C2t +0.150 (-0.057 to +0.372). The
  D8 intervals rest on 3 to 4 monthly forecast origins. The q99 exceedance after the break is 0.031
  for arm B
  and 0.031 for C2t against a nominal 0.010, so no arm is calibrated at D8.
- Primaries with no injection (14, real records): B minus A -0.021 (-0.037 to -0.004), C2t minus A
  -0.030 (-0.046 to -0.013), D2t minus A -0.033 (-0.050 to -0.016), against the 0.01 margin. C2
  minus A is +0.055 (+0.019 to +0.102) and fails it.
- Tail calibration: q01 and q99 exceedance and pinball loss are in `printed_report.txt`. The
  tail-weighted continuous ranked probability score and the curtailment cost were not computed.

## Caveats and next steps

- Lab shortcuts: past weather in place of ensemble members, a filter that sees yesterday's
  reading, a lagless GAM base with a simple fixed-gain level filter instead of the effects filter.
  The experiment 264 as-of band widening at D8 was not reproduced, so the success criterion for D8
  against that comparator is untested.
- The level filter absorbs a step within days, which leaves regime effects little to add. In this
  implementation the filter also subtracts a known step a second time once regime effects exist,
  which biases the regime arms against themselves; the no-filter rows above remove that bias but
  are far worse in absolute terms.
- Untested: confirming each candidate regime date by whether arm C2 improves out of sample on
  blocked time folds, then scanning 60 days either side of each candidate, needs a new design.
  A regime arm that replaces the level filter, instead of stacking on it, is also untested, and
  would need testing on the full Primary demand research default 3.
