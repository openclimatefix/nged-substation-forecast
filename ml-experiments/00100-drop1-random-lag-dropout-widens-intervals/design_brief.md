# Design brief for experiment 100 (DROP1): random lag dropout in training

Written for an Opus design reviewer. Nothing has been fitted yet. Repo worktree:
`/home/jack/dev/nged-substation-forecast/.claude/worktrees/drop1` (read-only for you; write only the
design file named at the end).

## The idea (maintainer, 2026-10-19)

Train the XGBoost Primary-demand quantile forecaster so that (1) hand-labelled switching events
(experiment 41's labelled set) are removed from the training targets, (2) every lagged power
feature whose time lies inside such an event is set to missing, in training and test, and (3) during
training only, other lagged power features are also set to missing at random, so the model learns
that a missing lag is normal and that the right response is wider quantiles, not a biased point
forecast.

## What we already know

- Experiment 75 (labelled-event masks, weekly): masking targets changes nothing outside switching
  (+0.004). Only 1.3% of scored half-hours have a lag in a logged event.
- Experiment 86: `plain` booster, quantile score on the 37,394 half-hours with a lag inside an oracle
  (logged) event: 10.778 points of capacity; 5-95% band coverage on those rows is 0.784 (nominal
  0.90); on all rows outside switching coverage is 0.893. Boosters are stored per arm, per monthly
  refit, per series (768 for 4 arms).
- Experiment 89: setting lags inside the ARA (union of RO-NC detector, logged events, logged closures;
  9-16% of half-hours per year) to missing with NO random dropout in training raised the score by
  +0.409 (+0.281 to +0.517) on rows with such a lag, because the model never saw missing lags except
  in exactly those rows. Its fit code is `ml-experiments/00089-*/e00089_fit.py` (12-month rolling
  window, 150 rounds, eta 0.15, half-life 0.5 years, 9 quantile levels, ENS control member,
  per-series boosters, monthly refits, mixture of 51 members' quantiles at predict time).
- Experiments 88 and 96: injection harness; 2,452 held-out family A injected cases (forecast origin
  inside an injected transfer), true unmodified series as truth.

## Planned arms (from the maintainer's brief; at most four)

1. `plain` (stored experiment 86 predictions; do not refit).
2. `mask`: labelled events removed from targets, lags in labelled events masked (train and test), no
   random dropout. Close to experiment 89's `M` but with labelled events only, not the ARA union.
3. `drop10`: arm 2 plus random dropout in training only. Each lag column independently set missing
   for a random 10% of TRAINING rows (augmentation: add masked copies, keep originals), chosen so the
   masked share is similar to the share of rows with a lag in an event.
4. `drop30` (rate 30%) or `dropgrp5` (whole lag groups, all lags of one origin, at 5%).

At test time lags are masked only when they lie inside a labelled event.

## Scoring (maintainer)

Quantile score and tail calibration (coverage and pinball at q01, q05, q95, q99; the 5-95% band
coverage plot) on in-event-lag rows minus plain; then all rows outside events; then the held-out
injected cases (experiment 88 family A, true series as truth, lags masked by the injected interval =
oracle masking; with RO-NC causal masks if cheap). Key question: does the 5-95% band coverage on
in-event-lag rows move from 0.784 towards 0.90 without hurting other rows? Report mean band width on
masked vs unmasked rows. Bonferroni intervals, months resampled. Monthly P-roll refits.

## Questions for you

1. The dropout mechanism for XGBoost: augmentation (add masked copies) versus replacement, per-lag
   versus per-group masks, the rate(s), how rows are weighted, and whether the masked copies should
   share the original row's recency weight. XGBoost routes missing values down a learned default
   branch; does this mean a missing lag behaves well at test time, and is there a risk that the
   default direction learns the wrong thing?
2. How to keep tail quantiles sensible when a lag is missing; whether the quantile heads (multi
   quantile objective) need monotone or non-crossing fixes.
3. What the training rows and masks should be. Which labelled set (the 35 real transfers over 1 day
   plus the rest; read `ml-experiments/00086-*/e00086_labels.py`), and what share of rows is masked
   by it, so the dropout rate can match.
4. The scoring, the pre-registered contrasts with a Bonferroni level, the decision rule, and a
   threshold set by the cost of porting. Which rows are the headline rows, and what rows make the
   no-harm check.
5. The pitfalls: confounds (more rows in augmented arms, so arm 3 vs 2 differs by sample size and by
   dropout; do we need an extra control that adds unmasked duplicates?), leakage, the lag columns that
   exist (`common.LAG_COLUMNS`), and anything in experiment 89 that made masking harmful for a reason
   other than missing training exposure.
6. How the injected-case scoring should reuse experiment 88/96 machinery with new boosters
   (`ml-experiments/00096-*/e00096_cases.py`, `e00096_fit.py`).

Read at least: `docs/ml-experiments/00086-*/result.md`, `00089-*/result.md`, `00096-*/result.md`,
`ml-experiments/00089-*/e00089_fit.py`, `ml-experiments/00086-*/e00086_labels.py`, and the
`ml-experiment` skill `.claude/skills/ml-experiment/SKILL.md`. Do not start sub-agents.

## Output

Write your design to
`ml-experiments/00100-drop1-random-lag-dropout-widens-intervals/opus_design.md`: concrete choices for
each question, the arms with exact definitions, the contrasts and decision rule with thresholds and
reasons, the pitfalls, and a checklist for the code reviewer. Keep it under about 250 lines.
