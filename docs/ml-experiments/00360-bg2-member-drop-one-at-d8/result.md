---
id: '00360'
slug: bg2-member-drop-one-at-d8
title: >-
  Which of BG2's five Primary members carries its lead over the GAM stack at D8?
theme: production-reference-and-best-guess
also_relevant_to:
  - retraining-and-training-history
status: planned
kind: other
size: >-
  simple. Stored: private files under data/ml-experiments/00360-bg2-member-drop-one-at-d8/ only.
  Serving path: not touched. Degradation rule: not touched. Designs: one (re-blend stored member
  forecasts by BG2's own rule). Callers: the script imports experiment 87's assembly code and
  experiment 355's stored D8 rows. Asset graph: not touched.
hypothesis: >-
  BG2's lead over the GAM stack at D8 is spread across its five-member blend, so no single member
  explains it.
prediction: >-
  Written before any score is read. Dropping any one member moves D8's quantile score by less than
  0.2 points (the five-member blend scores 7.593; the GAM stack scores 8.338 weekly). The best single
  member alone scores within 0.3 points of the blend.
decision_rule: >-
  Descriptive, no threshold: each drop-one arm minus the five-member blend, paired, whole valid months
  resampled (20,000 resamples, seed 0), 95% intervals. `supported` when every drop-one difference
  is under 0.2 points; `refuted` when one is 0.2 or more with an interval above zero; `inconclusive`
  otherwise. Gate: the re-blended five-member arm reproduces 7.593 on the 96,986 rows of experiment
  354 before any other arm is read.
benchmark: >-
  The five-member blend (BG2), the GAM stack (experiment 355 arm W), manual_heuristic, climatology
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of the series' effective
  capacity; median error in MW; q05 to q95 coverage
code: ml-experiments/00360-bg2-member-drop-one-at-d8/
---

# Experiment 360: which of BG2's five Primary members carries its lead over the GAM stack at D8?

**Plan (no score read yet).** Interim, one substation, one scored year; blending gains are not a
causal story about why a single model leads.

- **Rows.** The 96,986 rows of experiment 354 at D8 (scored year 2025-07-02 to 2026-06-30, leads
  0.5 to 168 h). Placebos D2, D3, D4 use all stored rows with a lead of up to 168 h and a finite
  actual.
- **Members.** Weekly-refitted XGBoost (W), monthly-refitted XGBoost (M), analogue (AN), pooled by
  substation type (PT), BG2's XGBoost stack (ST, with the 14-day correction). W, M, AN, and ST use
  vintage-capacity solar offsets at the five offset Primaries, as in BG2. D8 has no offset.
- **Arms.** The five-member blend; five four-member blends, each re-blended over all 16 Primaries
  with experiment 87's rolling monthly weights; five single members (spread factor 1.15 as in BG2).
- **Stop rule.** If the stored D8 member forecasts are missing, stop and report; nothing is refitted.
- **Experiment 87 check.** Experiment 87 reports Primary-wide ablations (A2, A3, A7) and none for
  D8 alone.
