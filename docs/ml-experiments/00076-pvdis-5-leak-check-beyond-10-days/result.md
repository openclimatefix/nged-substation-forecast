---
id: "00076"
slug: pvdis-5-leak-check-beyond-10-days
status: done
size: simple
hypothesis: >-
  Experiment 00064's arm `am` (the XGBoost model given the forecast solar output as a base margin)
  scored 0.082 points of capacity below `d0` at leads of 10 days and beyond, where its margin is
  zero. That gain does not come from information that was unavailable at forecast time. If it is
  real, it comes from the offset in the training rows at shorter leads, which removes the solar
  swings from the training target so that the trees fit demand better at every lead (the
  cleaner-target effect), or from the fit starting at zero instead of at XGBoost's estimated base
  score.
prediction: >-
  The code audit finds no leak: the margin is zero at every row from 10 days, in training and at
  prediction; the weight, the capacity table, and the calibration slopes use only rows before
  2025-07-01; and the margin is set the same way on every prediction matrix. A model trained with
  the offset and given no offset at prediction (`am_train_only`) scores identically to `am` at 10
  days and beyond, because the margin is already zero there. A refit with a zero margin everywhere
  (`am0`) scores 0.03 to 0.08 points above `am` at 10 days and beyond, which would be the
  cleaner-target effect.
decision_rule: >-
  Leak found if any audit check fails, or if `am_train_only` differs from `am` at 10 days and
  beyond by more than 1e-6. Otherwise the gain beyond 10 days is the cleaner-target effect when the
  98.33% interval (a 95% interval split across three contrasts by a Bonferroni correction) of
  `am` minus `am0` at 10 days and beyond has an upper end below zero. It is a start-point effect
  when `am0` minus `d1l` has an upper end below zero and `am` minus `am0` does not. It is
  unexplained otherwise. Intervals come from 50,000 resamples of whole ISO weeks of forecast start
  and of seeds, on the quantile score over experiment 00064's rows. The threshold is zero because
  the question is whether a mechanism exists, and nothing is ported.
benchmark: >-
  `d0`, `d1l`, and `am` as stored by experiment 00064 on row set `fold_a_every5d_v1`. `am0` is
  refitted here with the same recipe, seeds (0 to 2), and series (D9, D10, D11, D14, D16) as
  `am`.
row_set_id: fold_a_every5d_v1
metric: >-
  Quantile score in points of capacity (1 point is 1% of a series' effective capacity), after the
  14-day correction and the 1.15 spread factor, on experiment 00064's rows, as scored there.
code: ml-experiments/00076-pvdis-5-leak-check-beyond-10-days/
verdict: partly supported
best_arm: >-
  Not applicable: a diagnostic. `am0` (the same fit with a zero margin everywhere) scores 4.939
  points of capacity at 10 days and beyond, against 4.904 for `am` and 5.025 for `d1l`.
headline: >-
  No leak. At 10 days and beyond, `am` is 0.121 points of capacity (98.33% interval -0.176 to
  -0.064) below `d1l`. Most of that comes from starting the fit at zero: `am0` is 0.086 (-0.135 to
  -0.040) below `d1l`. The offset in training adds 0.035 (-0.074 to +0.003), which the data cannot
  tell from zero.
lessons: >-
  The gain of `am` beyond 10 days is not a leak. The audit finds a margin of exactly zero at every
  row from 10 days in training and at prediction, and no validation actual in the weights. Most of
  the gain follows from the fit starting at zero, not from the solar offset. A cleaner-target effect
  of up to 0.07 points is not excluded. The contrast of `am0` with `d1l` also differs in the fitting
  pipeline (experiment 00046's stored `d1l`), so a refit of `d1l` in this pipeline would settle it.
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Compare an arm that has a base margin with a refit of the reference model in the same pipeline
    and with a zero margin, because a model given a margin starts from zero, not from XGBoost's
    estimated base score, and `am0` scores 0.086 points (-0.135 to -0.040) below `d1l` at 10 days
    and beyond (contrasts).
  - >-
    Read the gain of a solar offset at leads under 10 days against `am0`, not only against `d0`,
    because `am0` already scores 0.047 points below `d0` at 10 days and beyond, and the offset
    itself adds only 0.035 (-0.074 to +0.003) there (contrasts).
  - >-
    Do not score a zero-margin prediction of a margin-trained model after the 14-day correction
    without saying so, because the correction uses the arm's own recent medians, so `am_train_only`
    differs from `am` by 0.037 points (0.001 to 0.072) at 10 days and beyond although the raw
    quantiles are identical (audit).
---

# Experiment 76: does experiment 64's gain beyond 10 days come from a leak?

**Take-home message: the gain of experiment 64's best arm at 10 days and beyond is not a leak, and
most of it comes from how the fit starts, not from the solar offset.** The audit found no way for a
validation actual to reach the margin, and the margin is exactly zero from 10 days. A refit with a
zero margin everywhere scores 0.086 points of capacity below experiment 46's `d1l` there, which
accounts for about 70% of the 0.121-point gap. The offset in the training rows adds a further 0.035
points, and the data cannot tell that from zero. The next test is a refit of `d1l` in this
pipeline, which would show whether the start-at-zero effect is real or a difference between two
pipelines.

We asked whether the lower error of experiment 64's arm `am` at forecast leads of 10 days and
beyond could come from information that was not available at forecast time. In `am`, the XGBoost
model starts from a margin: the substation's solar capacity times a forecast of how sunny the day
will be, times a weight fitted on the training years. The weight, and so the margin, is zero from 10
days, yet `am` still scored 0.082 points below the best guess `d0` there. We read the code and
checked the margin at every training row and every validation row. We also refitted `am` with a zero
margin everywhere (`am0`) and predicted with the stored `am` models after setting the margin to zero
(`am_train_only`). The score is the quantile score, an error measure in which lower is better, and 1
point is 1% of a substation's capacity.

**The verdict is `partly supported`.** The prediction of no leak held. The prediction that `am0`
scores 0.03 to 0.08 points above `am` held in size (0.035) but not in the decision rule, because the
interval reaches zero.

## The audit finds no leak

All 20 checks of `e00076_audit.py` pass, for each of the five Primary substations.

- **The margin is zero from 10 days.** Of the training rows, 98,965 to 98,973 per series have a lead
  of 10 days or more, and their largest margin is exactly 0. The same holds for 1,784 validation
  rows per series in the first 8 forecast runs, and zeroing the margin changes no prediction there.
  At shorter leads, zeroing the margin changes the quantiles by up to 0.086 points.
- **The weights, the capacity, and the calibration use the training window only.** Every training
  row is at or before 2025-06-30 23:30 UTC, and the weights are cross-fitted on those rows. The
  capacity fit leaves out the week of 2025-06-30 entirely, and the calibration of the ENS sunshine
  anomaly stops with an error if a row is outside the training window.
- **Prediction applies the margin once, on the same path as in training.** Rebuilding `am`'s predictions
  from its stored models with the margin the validation code supplies reproduces experiment 64's
  stored quantiles exactly (largest difference 0) for seed 0. Each prediction matrix holds the
  margin, and the code raises if it does not.

The raw quantiles of `am_train_only` equal those of `am` at 10 days and beyond (largest difference
0 over 244,185 rows), and differ by up to 0.17 at shorter leads.

## Most of the gain comes from the start, and the offset adds little

![Contrasts at 10 days and beyond](figure_1_contrasts.svg)

Mean quantile score in points of capacity at 10 days and beyond: `am` 4.904, `am0` 4.939,
`am_train_only` 4.941, `d0` 4.986, and `d1l` 5.025. At leads under 10 days: `am` 4.267, `d0` 4.452,
`d1l` 4.586, and `am0` 4.502.

| Contrast at 10 days and beyond | Difference | Interval |
|---|---|---|
| `am` minus `am0` (offset in training) | -0.035 | -0.074 to +0.003 (98.33%) |
| `am0` minus `d1l` (start at zero) | -0.086 | -0.135 to -0.040 (98.33%) |
| `am` minus `d1l` (whole effect) | -0.121 | -0.176 to -0.064 (98.33%) |
| `am` minus `d0` (experiment 64) | -0.082 | -0.126 to -0.038 (95%, exploratory) |
| `d1l` minus `d0` (radiation inputs) | +0.039 | +0.006 to +0.074 (95%, exploratory) |
| `am_train_only` minus `am` | +0.037 | +0.001 to +0.072 (95%, exploratory) |

The intervals come from 50,000 resamples of whole ISO weeks of forecast start (26 weeks) and of 3
seeds, on 21,658 rows per seed. The decision rule is in the front matter. It calls the effect a
start-point effect when `am0` minus `d1l` is below zero and `am` minus `am0` is not, which is what
the table shows.

**`am_train_only` scores 0.037 points worse than `am` at 10 days and beyond although the raw
quantiles are identical.** The 14-day correction adds the mean of each arm's own recent median
errors, and `am_train_only` has no margin at the shorter leads that those errors include. This is a
property of the correction, not a leak, and it is why the audit compares raw quantiles.

**The model `am0` scores worse than `d0` at leads under 10 days (+0.051, 95% interval +0.002 to
+0.099, exploratory), so starting at zero does not help everywhere.** The solar offset is what
lowers `am` at those leads, as experiment 64 found.

## Limits

- **The start-point reading is not isolated.** `d1l` is experiment 46's stored model, and `am0` is
  fitted in experiment 64's pipeline, so any difference between the two pipelines is in the
  contrast. A refit of `d1l` here would remove it. The mechanism, that a model given a base margin
  starts from zero instead of XGBoost's estimated base score, is the documented behaviour and is
  not tested here.
- **A cleaner-target effect of up to 0.07 points is not excluded.** The interval of `am` minus `am0`
  ends at +0.003.
- **Experiment 64's rows only.** The result holds for the five Primary substations, April to
  September 2025 and 2026 daylight rows, and 26 weeks at 10 days and beyond.

## What the next experiment should answer

Refit `d1l` in this pipeline with the same seeds, and compare it with `am0`. If the two match, the
start-at-zero gain is an artefact of the pipeline, and `am`'s gain beyond 10 days is the offset
alone. If `am0` stays below, the production model's base score deserves a test of its own.
