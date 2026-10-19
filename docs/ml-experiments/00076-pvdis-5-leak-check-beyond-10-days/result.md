---
id: "00076"
slug: pvdis-5-leak-check-beyond-10-days
theme: solar-disaggregation
also_relevant_to: []
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
verdict: supported
best_arm: >-
  Not applicable: a diagnostic. Without the 14-day correction, `am0` (the same fit with a zero
  margin everywhere) scores 4.735 points of capacity at 10 days and beyond, `am` 4.739, and `d1l`
  4.837.
headline: >-
  No route for a leak was found. At 10 days and beyond, the gap between `am` and `d1l` is 0.098
  points of capacity without the 14-day correction (95% interval -0.150 to -0.048), and `am0`, which
  has no solar term at all, shows a gap of 0.102 (-0.158 to -0.047). The offset in training adds
  +0.004 (-0.012 to +0.021), so the gap matches a start-at-zero effect.
take_home_message: >-
  No route for a leak was found, and a refit with no solar term at all reproduces the gain of
  experiment 64's arm `am` at 10 days and beyond (0.102 below `d1l` without the 14-day correction,
  against 0.098 for `am`). The gain matches XGBoost starting from zero when a base margin is
  supplied, and not the solar offset, with the caveat that the reference `d1l` came from another
  pipeline.
lessons: >-
  The audit found a margin of exactly zero at every row from 10 days in training and at prediction,
  and no validation value in the weights. The gain of `am` beyond 10 days is reproduced by a refit
  with no solar term, which fits the XGBoost model starting at zero instead of at its estimated base
  score. The offset in the training rows adds nothing resolvable. A refit of `d1l` in the same
  pipeline would settle whether the start-at-zero effect is real, because `d1l` was fitted in
  experiment 00046's pipeline.
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Compare an arm that has a base margin with a refit of the reference model in the same pipeline
    and with a zero margin, because an XGBoost model given a margin starts from zero, not from its
    estimated base score, and `am0` scores 0.102 points (-0.158 to -0.047) below `d1l` at 10 days
    and beyond without the 14-day correction (raw contrasts).
  - >-
    Read the gain of a solar offset against `am0`, not only against `d0`, because `am0` scores
    0.047 points below `d0` at 10 days and beyond (-0.102 to +0.008, exploratory) and the offset
    itself adds +0.004 (-0.012 to +0.021) there without the correction (raw contrasts).
  - >-
    Score the arms without the 14-day correction as well, when comparing arms at long leads, because
    the correction uses each arm's own recent errors, which come mostly from shorter leads, and it
    moves `am_train_only` by 0.037 points (0.001 to 0.072) at 10 days and beyond although its raw
    quantiles equal `am`'s (audit).
---

# Experiment 76: does experiment 64's gain beyond 10 days come from a leak?

**Take-home message: we found no route for a leak, and a refit with no solar term at all reproduces
the gain of experiment 64's best arm at 10 days and beyond.** The gain therefore matches the XGBoost
model starting its fit at zero, which happens whenever a base margin is supplied, and not the solar
offset. A caveat limits that reading: the reference model `d1l` was fitted in experiment 46's
pipeline, so a refit of `d1l` in this pipeline would show whether the start-at-zero effect is real.

We asked whether the lower error of experiment 64's arm `am` at forecast leads of 10 days and
beyond could come from information that was not available at forecast time. In `am`, the XGBoost
model starts from a base margin: a starting prediction, added to each row, equal to the substation's
solar capacity times a forecast of how sunny the day will be, times a weight fitted on the training
years. The weight, and so the margin, is zero from 10 days. Yet `am` still scored 0.082 points below
the best guess `d0` there. The score is the quantile score, an error measure in which lower is
better, and 1 point is 1% of a Primary substation's capacity. A Primary substation feeds 11 kV
circuits from the 33 kV network. `d0` is experiment 33's best-guess recipe, and `d1l` is the same
recipe without the radiation inputs, from experiment 46.

We read the code and checked the margin at every training row and at the validation rows at 10 days
and beyond in the first 8 forecast runs of each series. We also made two new sets of forecasts.
`am0` is `am` refitted with a margin of zero in every row, so that it starts from zero like `am` and
carries no solar term. `am_train_only` is the stored `am` model predicted with the margin set to
zero at every lead.

**The verdict is `supported`.** The prediction of no leak held. The prediction that `am0` scores
0.03 to 0.08 points above `am` at 10 days and beyond held after the 14-day correction in size
(0.035) but its interval reaches zero, and without the correction the difference is +0.004.

## The audit found no route for a leak

All 20 checks of `e00076_audit.py` pass, 4 for each of the 5 Primary substations.

- **The margin is zero from 10 days.** Of the training rows, 98,965 to 98,973 per series have a
  lead of 10 days or more, and the largest margin among them is exactly 0. In the first 8 forecast
  runs, 1,784 validation rows per series have a lead of 10 days or more, and zeroing the margin
  changes none of their predictions. The weight is set to zero for those leads by construction, so
  this confirms that the lead bands agree and that the margin reaches the model as zero. At shorter
  leads, zeroing the margin changes the quantiles by up to 0.086 points.
- **Every training row is at or before 2025-06-30 23:30 UTC.** The weights are cross-fitted on those
  rows, the capacity fit leaves out the week of 2025-06-30, and the calibration of the ensemble
  forecast of sunshine stops with an error if a row is outside the training window. The audit checks
  the training rows themselves. The weights, capacity, and calibration were checked by reading the
  code of experiments 46 and 64.
- **Prediction applies the margin once.** Rebuilding `am`'s predictions from its stored models with
  the margin that the validation code supplies reproduces experiment 64's stored quantiles exactly
  for seed 0 (largest difference 0).

Across all 5 series and 3 seeds, the 244,185 prediction rows at 10 days and beyond have raw
quantiles that are identical between `am_train_only` and `am` (largest difference 0). At shorter
leads they differ by up to 0.17 points of capacity.

## The gain matches a start-at-zero effect

![Contrasts at 10 days and beyond](figure_1_contrasts.svg)

Mean quantile score at 10 days and beyond, in points of capacity, after the 14-day correction and
the 1.15 spread factor: `am` 4.904, `am0` 4.939, `am_train_only` 4.941, `d0` 4.986, and `d1l` 5.025.
Without the correction (the spread factor kept), `am` scores 4.739, `am0` 4.735, and `d1l` 4.837.

| Contrast at 10 days and beyond | Difference | Interval |
|---|---|---|
| `am` minus `am0`, corrected (offset in training) | -0.035 | -0.074 to +0.003 (98.33%) |
| `am0` minus `d1l`, corrected (start at zero) | -0.086 | -0.135 to -0.040 (98.33%) |
| `am` minus `d1l`, corrected (whole effect) | -0.121 | -0.176 to -0.064 (98.33%) |
| `am` minus `am0`, uncorrected | +0.004 | -0.012 to +0.021 (95%, exploratory) |
| `am0` minus `d1l`, uncorrected | -0.102 | -0.158 to -0.047 (95%, exploratory) |
| `am` minus `d1l`, uncorrected | -0.098 | -0.150 to -0.048 (95%, exploratory) |
| `am` minus `d0`, corrected (experiment 64) | -0.082 | -0.126 to -0.038 (95%, exploratory) |
| `d1l` minus `d0`, corrected | +0.039 | +0.006 to +0.074 (95%, exploratory) |
| `am_train_only` minus `am`, corrected | +0.037 | +0.001 to +0.072 (95%, exploratory) |

A negative difference means that the first arm has the lower error. The three corrected decision
contrasts are read at 98.33%, which is a 95% interval split across three contrasts (a Bonferroni
correction). Every other row is exploratory, meaning that it is not in the decision rule, and is read
at 95%. All intervals come from 50,000 resamples that draw whole ISO weeks of forecast start (26
weeks at 10 days and beyond) and the 3 seeds, on 21,658 scored rows per seed.

**The decision rule, as written before any code ran, names the effect a start-point effect.** The
rule applies when `am0` minus `d1l` has an upper end below zero and `am` minus `am0` does not, and
the corrected rows meet both conditions. The uncorrected rows agree: the offset in training adds
+0.004 (-0.012 to +0.021), and the refit with no solar term accounts for all of the gap to `d1l`.

**The 14-day correction moves the corrected contrasts by about 0.03 points, because it uses each
arm's own recent errors.** `am_train_only` scores 0.037 points worse than `am` at 10 days and beyond
although its raw quantiles are identical. We infer that the correction, which adds the mean of an
arm's own recent median errors, carries the margin's effect from shorter leads into the longer ones.
The uncorrected rows remove that route, and the corrected `am` minus `am0` of -0.035 shrinks to
+0.004.

**The XGBoost model `am0` scores worse than `d0` at leads under 10 days (+0.051, interval +0.002 to
+0.099, exploratory).** Starting at zero therefore does not help at every lead. The mean scores at
those leads are 4.267 for `am` and 4.502 for `am0`, so the solar offset lowers the error there by
about 0.24 points, which agrees with experiment 64 (no interval computed here).

## Limits

- **The start-at-zero reading is not isolated.** `d1l` is experiment 46's stored model, and `am0` is
  fitted in experiment 64's pipeline, so any difference between the two pipelines is in the
  contrast. The mechanism, that an XGBoost model given a base margin starts from zero instead of
  from its estimated base score, is the documented behaviour of XGBoost and is not tested here.
- **The validation rows are experiment 64's.** They are daylight half-hours at five Primary
  substations from July to September 2025 and from April to June 2026, and the result is measured
  on 26 weeks of forecast start at 10 days and beyond.

## What a next experiment could answer

A refit of `d1l` in this pipeline, with the same seeds, compared with `am0`, would show whether
the start-at-zero gain is real. If the two matched, the gain beyond 10 days would be an artefact of
the pipelines. If `am0` stayed lower, the base score of the production XGBoost model would merit a
test of its own.
