---
id: "00076"
slug: pvdis-5-leak-check-beyond-10-days
status: planned
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
---

# Experiment 76: does experiment 64's gain beyond 10 days come from a leak?

Planned. Sized `simple`: the diagnostic reuses experiment 64's code and inputs with one changed
setting.

- **What gets stored:** one new prediction set (`am0`), private, and no production table.
- **Serving path:** untouched.
- **Degradation rule:** untouched.
- **Defensible designs:** one. The brief fixes the audit and `am_train_only`; `am0` is added to tell
  a start-point effect from a cleaner-target effect.
- **Callers:** the scripts are copies of experiment 64's, and nothing imports them.
