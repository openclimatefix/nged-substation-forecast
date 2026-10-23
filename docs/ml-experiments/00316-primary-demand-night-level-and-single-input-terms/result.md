---
id: '00316'
slug: primary-demand-night-level-and-single-input-terms
title: >-
  Is the band heads' gain at one Primary substation a night-time level error, and which single-input
  term of the best-guess demand model is behind its April-to-June gap to BG2?
kind: other
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
status: planned
size: >-
  medium. Stored: private files under data/ml-experiments/00316-*/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one is defensible for the stage 1 graft (this
  plan pre-registers a surrogate-based emulation of grafting, because arm D's term contributions are
  not stored). Callers: this folder's scripts, which read the stored forecasts of experiments 251,
  266, 305, 311, and 314 and call reviewed `studies.*` modules without changing them. CPU only, no
  downloads. A stage 2 correction head would need its own addendum before it runs.
hypothesis: >-
  (Stage 0) The gain of the band heads of experiments 311 and 314 at D10, an anonymised Primary
  substation, is mostly a night-time level error of Primary demand research default 1 and not a
  better description of the uncertainty. (Stage 1) The April-to-June gap between arm D and BG2 at
  D5 and D9, two anonymised Primaries, sits mainly in arm D's response to sunshine and not in its
  recent-power or calendar terms.
prediction: >-
  Written before any score of the new arms is read; each line can fail. Lead range 0.5 to 168 hours,
  shorter than NGED's 1 to 10 days. (S0a) At least 70% of D10's gain under the pinball head (arm P)
  lies on night half-hours (21:00 to 07:30 UTC). (S0b) An as-of night offset (median residual of the
  previous 28 nights' half-hours, added to every quantile of D10's night half-hours, no head)
  reproduces at least 70% of D10's gain under arm P; under 30% means the head does something else.
  (S1a) Grafting the solar response recovers at least 50% of the advantage of the depth-1 XGBoost
  refit over arm D on the worst quarter of April-to-June daylight half-hours at D5 and D9. (S1b)
  Grafting the lag handling and grafting the calendar profile each recover under 25%. (S1c) Grafting
  all five families together recovers at least 50% (the check that the emulation is usable).
decision_rule: >-
  Intervals resample calendar months of forecast initialisation (12 in the scored year; the 3
  months of April to June for stage 1, so stage 1 intervals are wide and the page says so). S0a and
  S0b are read on the lower bound of the 95% interval of each share. Stage 1 names a family when
  its share S is at least 0.30; the 95% interval is reported beside it. Stage 2 runs only if
  stage 1 names a family, or if no single family reaches 0.30 but the graft of solar, lags, and
  calendar together reaches 0.50 (the brief's two rules for that case disagree, and this plan
  resolves them that way). A stage 2 addendum with its arms, cut protocol, and minimum gain (0.03
  points, a new head) is committed before stage 2 runs. Nothing is ported by stages 0 and 1, so no
  porting threshold applies to them.
benchmark: >-
  Primary demand research default 1 (experiment 251's arm 1) on row set fold_a_daily_gap_v1; BG2,
  `manual_heuristic`, and `climatology` beside it.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity;
  smaller is better).
code: ml-experiments/00316-primary-demand-night-level-and-single-input-terms/
---

**Plan only: no score of the new arms has been read.**

## Questions

- **Stage 0.** The pinball band head (experiment 314's arm P) and the mixture head (experiment 311's
  arm M) lower the quantile score almost only at D10. Is that gain the head covering D10's night
  level error?
- **Stage 1.** A depth-1 XGBoost refit (experiment 305's post hoc variant) keeps most of BG2's
  advantage over arm D in April to June at D5 and D9. Which family of arm D's single-input terms
  (sunshine, recent power, calendar) accounts for it?

## Method

- **Stage 0 splits.** D10's gain by day and night, local hour, quantile level, and by the centre
  shift versus the width change of the head's 5% to 95% band. Counterfactual arms keep only the
  head's centre shift or only its width change.
- **Stage 0 night offset.** For each run, the median of the previous 28 nights' residuals
  (actual minus the default's median, taken from the shortest-lead forecast of each half-hour,
  21:00 to 07:30 UTC), added to all 11 quantiles of D10's night half-hours in that run.
- **Stage 1 graft.** Arm D's term contributions are not stored. A depth-1 XGBoost surrogate of
  arm D's median is fitted per site and month on the rows of experiment 305's inputs, and
  family k is grafted by shifting all of arm D's quantiles by the refit's family-k contribution
  minus the surrogate's. S is (QS of arm D minus QS of the graft) over (QS of arm D minus QS of the
  refit). The surrogate's fit to arm D is reported; a poor fit makes stage 1 inconclusive.
- **Rows.** As experiment 305 for stage 1 (the worst quarter of daylight half-hours at D5 and D9
  by arm D's excess over stored BG2, and all daylight half-hours); the 561,207 rows of experiment
  251 for stage 0.

## Stop rule

If no single family reaches S of 0.30 and the three-family graft is under 0.50, write up and do not
run stage 2.
