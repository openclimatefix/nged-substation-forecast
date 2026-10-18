---
id: "00087"
slug: bg2-best-guess-assembled-with-drop-one-ablations
status: planned
size: complex
hypothesis: >-
  Assembling the best ideas of the last experiments into one forecast beats the previous best forecast
  of each target class on the 292 daily forecast runs of the validation year (July 2025 to June
  2026). For the 16 Primary substations the assembly is a five-member quantile blend of
  weekly-refitted and monthly-refitted XGBoost models, an analogue model, a pooled-by-type model, and
  the frozen stack, with an explicit solar offset at the 5 substations that have a clean solar
  capacity. For the 4 bulk and grid supply points it is the weekly-refitted XGBoost model with no
  error correction. For the 6 solar farms and 3 wind farms it is the weekly-refitted XGBoost model
  trained on the ensemble forecast's own history.
prediction: >-
  Primary: BG2 scores about 5.30 points of capacity on the quantile score, against 5.64 for the stack
  (experiment 48) and 6.26 for `manual_heuristic`; the blend adds -0.05 to -0.10 to weekly
  refitting and the offset adds -0.10 to -0.15. Bulk and grid supply points: weekly refitting scores
  0.2 to 0.7 below the frozen model with the correction (experiment 48, C7). Solar and wind farms:
  the ensemble-trained model is within 0.1 (solar) and 0.3 (wind) points of experiment 33's
  ERA5-trained mixture, which is the non-inferiority margin of decision D1.
decision_rule: >-
  Four verdict contrasts, one per target class, each the quantile score of BG2 minus the class's
  previous best forecast on the same runs, read at 98.75% (Bonferroni across four), whole origin
  months resampled, 20,000 resamples, seed 0. The previous best is the stack for Primary, the frozen
  XGBoost model with the 14-day correction for bulk and grid supply points, and experiment 33's
  ERA5-trained mixture for solar and for wind. Primary and bulk and grid supply points are supported
  when the interval lies wholly below zero (threshold zero: the weekly refit and the blend replace
  existing scheduled jobs and one fixed weight vector, and the offset needs one capacity table).
  Solar and wind are supported when the interval's upper end lies below +0.10 (solar) and +0.30
  (wind), because the swap to the ensemble's own history is worth making whenever it costs less than
  that. Every ablation, every staircase stage, and every comparison with `manual_heuristic` is read
  at 95% and decides no verdict.
benchmark: >-
  The stack, `manual_heuristic`, `climatology`, the production-style `xgboost_baseline` (decision
  D4), and a perfect forecast, each scored on the same 292 runs.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss) in points of capacity, over 11 levels
  (0.01, 0.05 to 0.95, 0.99); the mean absolute error of the median, tail coverage and pinball loss at
  0.01, 0.05, 0.95, and 0.99, and the threshold-weighted CRPS are reported and decide nothing.
verdict: null
production_status: null
adopted_in: none
advice: []
code: ml-experiments/00087-bg2-best-guess-assembled-with-drop-one-ablations/
---

# Experiment 87: BG2, the best guess so far and what each part contributes

**This page is the plan. The write-up replaces it when the fits have run.** The design is the
maintainer-approved Opus design in `briefs/bg2_design.md`, with the requirements in
`briefs/bg2_requirements.md`. It is followed exactly. This page lists only the choices the design
leaves to the runner and the deviations found while planning.

## Size, by the five triggers

- **What gets stored:** predictions for about 30 arms under `data/ml-experiments/00087-*/`, never
  committed. Nothing enters a Delta table.
- **Serving path:** not touched. All code lives under `ml-experiments/`.
- **Degradation rule:** not touched. The experiment reads the cleaned power table through the
  existing scripts' own paths.
- **Defensible designs:** one, fixed by the Opus design. Its undecided choices are the defaults D1 to
  D7, each run on its stated default.
- **Callers:** the scripts import only their own folder, `studies.*`, and reviewed packages.

The size is complex: the experiment fits about 10.5 GPU-hours of models. The reviews are one Sonnet
plan review, one Sonnet code review for leakage and for the fit stopping at the fold's `train_end`,
and one Sonnet prose review. The maintainer waived the Opus reviews because the design is already
Opus-reviewed.

## Earlier advice the plan follows

- Start every arm from XGBoost's own estimated starting value (experiment 80, experiment 74).
- Refit weekly at production settings for BG2, because experiment 48 found weekly refits beat monthly
  refits by 0.13 and experiment 85 reproduced the cadence gap.
- Train each model on the history of the weather forecast it will be fed, so BG2 trains the solar
  and wind models on ensemble data and keeps the ERA5-trained mixture as a reference row.
- Report absolute scores beside contrasts, and never compare R11's control-only numbers with
  51-member scores.

## Plan

- **Rows:** `fold_a_daily_gap_v1`, 292 daily runs, is the headline; all 365 daily runs are forecast
  because the stack's 14-day correction reads yesterday's forecast errors. `fold_a_every5d_v1`
  (73 runs) is reported as a check.
- **Levels:** 11 (0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99), so every arm
  writes `q01` and `q99`. Experiment 48's stored 9-level arms cannot supply the tails, so every
  member is refitted at 11 levels.
- **Reproduction check before scoring:** the weekly XGBoost model at the 9 levels inside the 11 must
  score within 0.02 of experiment 48's 5.44 on the same runs, and the bulk and grid supply points
  within 0.02 of 7.84 and 18.66. A larger miss stops the run and goes to the ML manager.
- **Leakage guards:** lags no older than the lead are nulled, the training rows of each refit end
  3.5 hours before the refit, and the offset's weights and capacity come from before the first
  validation run. The code reviewer checks each guard.
- **Fit stopping:** every fit uses a fixed number of rounds, so none can stop on data after the
  fold's `train_end`.
- **Compute:** one GPU job at a time inside the slot's 24 GB, 12 cores split across parallel
  workers, one seed. A repeat fit at 4 Primary substations gives the noise floor. The run stops at
  16 GPU-hours.

## Deviations from the design found while planning

- The design says the offset weights are refitted at every refit. Experiment 74's code applies
  experiment 64's weights, which were fitted on data before the validation year. BG2 does the same,
  because it has no leakage and it matches the scores the design quotes (0.149 and 0.152).
  The write-up records this as a deviation if the reviewers agree.
- Experiment 74's offset inputs stop at the end of May 2026, so BG2 rebuilds the missing runs. The
  offset arms start from the median of the training labels plus the offset, and the other arms from
  XGBoost's own start. Experiment 78's capacity vintages are used only where their estimation window
  ended before the run; otherwise the capacity table of experiment 64 is used and ablation A6 is
  identical by construction.
- The solar and wind reference is the ERA5-trained mixture refitted at 11 levels, not experiment
  33's stored arm. Weekly generator refits keep experiment 33's 2-year half-life.
- The spread factor, the blend scheme, the offset gate, and the half-life were chosen or scored on
  the same validation year, so the predicted scores carry selection bias.
- Defaults D1 to D7 run as stated. Any further deviation is listed in the write-up.
