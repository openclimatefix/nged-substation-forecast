---
id: '00322'
slug: primary-demand-default-2-daylight-only-heuristic-blend-and-sunshine-graft-replication
title: >-
  Can a blend with the manual heuristic confined to daylight (or to nights an ensemble share rule
  calls mild) improve Primary demand research default 2 at 0.5 to 240 hours, and does the sunshine
  graft of experiment 316 replicate?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: change
size: >-
  medium. Stored: private files under data/ml-experiments/00322-*/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one is defensible, but the arms are fixed by
  the question. Callers: this folder's scripts, which import the folders of experiments 184, 202,
  305, 316, 317, 319, and 320 and call reviewed `studies.*` modules without changing them. No change
  under `packages/`.
hypothesis: >-
  (Stages 1 and 2) In experiment 320 the heuristic blend's gain at 168 to 240 hours sits on sunny
  daylight (-0.236 points) and in April to October, while its harm sits on cold winter nights,
  which are night half-hours. Confining the blend of experiment 319 to half-hours with the sun above
  5 degrees (D1) keeps the gain with no harm on cold winter nights by construction, and a night
  weight that is a function of the share of the 51 ensemble members below 3 degrees C (D3) recovers
  the nights where the heuristic helps without harming cold ones. (Stage 3, a diagnostic) Experiment
  316's sunshine graft, which recovered 0.36 of the depth-1 XGBoost refit's advantage on the worst
  April-to-June half-hours, replicates in April and June 2025.
prediction: >-
  Written before any score is read; each can fail. (PD1) D1 keeps at least 70% of the blend's gain
  at 168 to 240 hours (a gain of at least 0.095 points, 99.5% interval below zero), at least 85% of
  the blend's gain on sunny daylight, and is at least 8% better than the manual heuristic at 168 to
  240 hours; cold winter nights equal the default exactly. If D1 keeps under 50% of the gain, the
  gain sits at night and the daylight gate is wrong. (PD3) The member share P (flag: at least half of
  the members below 3 C) flags at least 65% of realised cold winter nights at 168 to 240 hours
  (against 31% to 52% for the ensemble-mean rule) with a false-flag rate no more than twice the base
  rate of cold nights, D3 adds at least 0.02 points over D1 at 168 to 240 hours, and cold winter
  nights are harmed by under 0.03 points. If P flags under 50% of cold nights, D1 stands. (PG) In
  April and June 2025 the sunshine graft recovers at least 0.25 of the refit's advantage on the
  worst daylight half-hours at D5 and D9 together and at least 0.15 at each; the recent-power graft
  and the calendar graft each recover under 0.25. If sunshine recovers under 0.15, the experiment
  316 lead was noise.
decision_rule: >-
  Family rule of research-default.md, two pre-registered arms (D1, D3) read at the 1 - 0.01/2
  level, at most one adopted. Both are changed settings or a small fitted step on stored forecasts
  (tier 1): D1 minimum gain 0 points, D3 minimum gain 0.02 points over the default at 168 to 240
  hours. An arm is adopted when its interval at 168 to 240 hours lies wholly below the default's
  score, its point estimate reaches the minimum gain, no no-harm bound of the research-default page
  is broken at 0.5 to 240 hours or at 168 to 240 hours (pooled 0.02 points, q01 and q99 pinball
  0.05, cold winter nights 0.05, sunny daylight 0.05, twCRPS 2% relative), and leaving one Primary
  out does not flip the sign. The reproduction gate (default equals experiment 319 within 0.01
  points: 2.749 at 0.5 to 240 hours and 3.152 at 168 to 240 hours) comes first. Stage 3 is exempt
  and decides only whether a solar-term correction is worth a change experiment.
benchmark: >-
  Primary demand research default 2 (experiment 319's default plus x1: the convex chain with the
  two-number solar summary to 240 hours; experiment 251's A+H+S below 168 hours). The manual
  heuristic and climatology as stored by experiment 87 and BG2 to 168 hours are references, and
  the experiment 319 blend (all half-hours) is a reference arm.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity)
  per 24-hour lead band from day 1 to day 10, pooled over six Primaries, scored year 2025-07-02 to
  2026-06-30, rows outside the hand-filtered events, origin 09:00 UTC. Coverage and pinball loss at
  q01, q05, q95, q99 by lead band. Stage 3: share of the refit's advantage over arm D recovered by a
  graft, on the worst quarter of daylight half-hours by arm D's own score.
code: ml-experiments/00322-primary-demand-default-2-daylight-only-heuristic-blend-and-sunshine-graft-replication/
---

## Plan

Stage 1 tests D1 (the experiment 319 blend on daylight half-hours only, the default at night) with
D2 (day and night weights refitted separately) as a reference. Stage 2 tests D3 (D1 by day, and by
night a chain weight that is a logistic function of the share of the 51 ensemble members whose
24-hour mean temperature is below 3 C, fitted as-of per lead band and season group). Stage 3
repeats experiment 316's level-matched graft in April and June 2025 at D5 and D9 with D10 and D16
as placebos, from 0.5 to 168 hours (arm D is stored only to 168 hours).

Deviation from the brief, stated before any score is read: the replicate year has no stored BG2
forecast, so the worst quarter of daylight half-hours is chosen by arm D's own quantile score and
not by arm D's gap to BG2. The same selection is applied to the scored year as a bridge.

Size, by the five triggers: stores private files only; not on the serving path; no degradation rule;
more than one defensible design but the arms are fixed here; callers are this folder's scripts.
Medium. Sonnet plan, code, and prose reviews; no Opus.
