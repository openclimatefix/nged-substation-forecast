---
id: '00328'
slug: lsc-regime-changes-detect-then-model-part-1
title: >-
  Does modelling permanent regime changes beat retraining on a recent window?
theme: lsc-detectors
also_relevant_to:
  - retraining-and-training-history
status: planned
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
---

**Planned. Written before any fit; the results and the take-home message replace this page.**

## Plan notes

- Reuses the segmentation code of
  [experiment 295](../00295-regime-segmentation-part-b-persistent-regimes-without-labels/result.md)
  and the injected worlds of experiment 297; the as-of widening comparator is
  [experiment 264](../00264-dp-gas-engine-asymmetric-bands-at-d8/result.md).
- Parallel pools use the `spawn` start method, one thread per worker, within a 6-core slot.
- No arm uses an engine-state input; regime dates use only data up to the forecast origin.
