---
id: '00353'
slug: d8-research-default-with-mva-fold
title: >-
  How well does Primary demand research default 3's stack forecast D8 when the MVA meter's fold is
  modelled?
theme: production-reference-and-best-guess
also_relevant_to:
  - differentiable-physics-embedded-generation
status: planned
kind: other
size: >-
  medium. Stored: private files under data/ml-experiments/00353-d8-research-default-with-mva-fold/
  only. Serving path: not touched. Degradation rule: not touched. Designs: more than one is
  defensible (a full meter-fitted T-all stack, or the stored meter fit plus the stack's stage-2
  corrections), so the plan fixes the second, which needs no new fit. Callers: this folder's script
  reads the stored forecasts of experiments 152 and 252 and calls their stage-2 and scoring code,
  unchanged. Asset graph: not touched.
hypothesis: >-
  At D8 (Marsh Lane, a Primary metered only in MVA), a stack that fits its demand model through the
  fold-aware meter operator (experiment 152's arm A) and then applies the stack's night offset and
  recipe R scores better than the same stack fitted to the folded reading as if it were signed
  demand, and recovers reverse flow where the meter's fold hides it.
prediction: >-
  Written before the new arms are scored; the stored scores of experiment 252's arm A1 (11.43
  points at D8 against BG2's 7.59) and experiment 152's F minus N at D8 (-0.19) were already known.
  (1) B minus C (fold-aware minus fold-ignorant, both with the stage-2 corrections) is -0.30 to 0.00
  points over all rows, because D8's fitted flow never goes negative in experiment 152. (2) The
  fitted model finds reverse flow on 0 to 2% of D8's scored half-hours. (3) B stays worse than BG2
  by at least 1.5 points over all rows, because D8's gap to BG2 comes from the April 2025 change in
  what its series measures, not from the fold. (4) B scores within 0.5 points of experiment 252's
  arm A1 (A1 has the T-all design, a solar clip, and the 60% clip that B lacks).
decision_rule: >-
  C1, B minus C over all rows, is the only contrast that bears a verdict. Paired month-resampled
  bootstrap (20,000 resamples, 95%). `supported` when the interval lies wholly below zero and the
  point estimate is at most -0.10 points: a meter fit is existing reviewed code, so the threshold
  covers only its extra complexity in a forecast that must run for 8 Primaries. `small gain` when
  wholly below zero but above -0.10. `not supported` when the interval includes zero. C2, B minus
  A, and C3, B minus BG2 on sunny daylight half-hours, are descriptive and do not change the
  verdict. Control gate: the stored arms A and N of experiment 152 and A1 of experiment 252 must
  reproduce their printed D8 scores (7.29/7.56 are D12's; D8's are read from the stored files) when
  rescored by this script.
benchmark: >-
  BG2, climatology, and manual_heuristic as stored by experiment 87 and read by experiment 252's
  scoring code; experiment 252's arm A1 (stage-2 arm d) as the card's forecast
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of the series'
  effective capacity; slices sunny daylight, winter nights, cold nights, Christmas, bank holidays;
  tail coverage at q01, q05, q95, q99; median minus mean; MAE of the mean forecast
code: ml-experiments/00353-d8-research-default-with-mva-fold/
---

**This page is the plan; the write-up replaces it after the run.**

## Plan

1. **Question.** D8's forecast on the substation card comes from experiment 252's arm A1, which reads
   the MVA reading as signed demand. Does fitting the demand model through the meter's fold
   (`studies.dp_meter`, experiment 110's L0) change the score, and does it find reverse flow?
2. **Arms, all at D8, scored year 2025-07-02 to 2026-06-30, leads 0.5 to 168 hours.**
    - A: experiment 252's stage-2 arm d as stored (the card's forecast).
    - B: experiment 152's arm A (meter fit, symmetric quantiles), then the night offset and recipe R
      of experiment 252's stage 2. Also B-fold: 152's arm F_fold (quantiles folded at zero) given
      the same shifts.
    - C: experiment 152's arm N (the reading as signed demand), then the same stage 2.
    - References: BG2, `manual_heuristic`, climatology.
3. **What this does not test.** B does not use the T-all design, shared solar clip, or pinned solar
   capacity of the stack, because `studies.dp_meter` fits the B0 design only. The lead range stops
   at 168 hours because the stored ensemble forecasts stop there, so the heuristic blend of Primary
   demand research default 3 (168 to 240 hours) is not applied.
4. **Training length.** D8's series changes meaning in April 2025 (experiment 152); training starts
   2025-04-01 and the scored year has 12 monthly fits on 3 to 14 months of data. The spread comes
   from the tuning residuals of D2, D3, D4, D19, D21, and D22, as in experiment 252.
5. **Reverse flow.** Reported as the share of scored half-hours where experiment 152's fitted
   signed flow is negative, with the caveat that the sign is not observable in an MVA meter.

Compute: no fits. One process, under 4 GB. Reuse: experiment 152 and 252 folders.
