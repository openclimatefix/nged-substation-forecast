---
id: "00078"
slug: pvdis-5-current-capacity-vintages
status: planned
size: medium
hypothesis: >-
  Experiment 00064's arm `am` (the XGBoost model given a forecast solar output as a base margin)
  used one embedded-solar capacity per Primary substation, fitted on data up to 2025-06-30 and held
  for the whole validation period. That capacity is 5% to 19% below experiment 00045's 2025/26
  constants at the five series. Giving `am` a capacity that is re-estimated before each forecast
  run, using only data from before the run (a real-time vintage of the season-year constant),
  lowers its quantile score further, and the fitted solar term still carries solar information,
  because the same arm with the forecast sunshine taken from another date scores worse.
prediction: >-
  The gain grows, but by little. Experiment 00076 puts the whole effect of the solar term at about
  0.24 points of capacity at leads under 10 days (`am` against `am0`), and the vintage capacities
  differ from the held capacity mostly in April to June 2026 (a rise of 5% to 20%). A margin 10% to
  20% larger is expected to lower the score by 0.01 to 0.04 points over all leads. So `am_sy`
  (`am` with the season-year vintage) minus `am` is expected at -0.02 (-0.04 to 0.00) over all
  leads, and at -0.04 (-0.09 to 0.00) at leads under 48 hours, where the solar term acts most. The
  weekly vintage `am_wk` is expected to score the same as `am_sy` or 0.00 to 0.05 points higher,
  because the latest weeks of a real-time weekly fit are noisy (experiment 00045 found 3.6 times the
  path error of the final fit). The refit of `d1l` in this pipeline, `d1l_r`, is expected to score
  0.00 to 0.10 points higher than `am0` at 10 days and beyond without the 14-day correction. A value
  near 0.10 would mean that starting from XGBoost's estimated base score costs that much. A value
  near 0.00 would mean that experiment 00076's gap between `am0` and the stored `d1l` came from the
  difference between experiment 00046's pipeline and experiment 00064's. The predicted C1 is far
  from the `supported` threshold (-0.05), so the plan expects `small gain` or `inconclusive`. The
  weight on the solar term was fitted with the held capacity, so a larger capacity at prediction
  could overshoot, and the prediction carries that uncertainty.
decision_rule: >-
  Two families of reads, each Bonferroni-corrected, from 50,000 resamples of whole ISO weeks of
  forecast start and of seeds, on the quantile score over experiment 00064's scored rows (daylight
  April-to-September half-hours at the five series D9, D10, D11, D14, and D16). The verdict family
  has four contrasts, read at 98.75% (a 95% interval split across four). C1 is `am_sy` minus `am`
  over all leads. C2 is `am_sy` minus `am` at leads under 48 hours. C3 is `am_sy` minus `am0`
  over all leads (the solar term itself). C4 is `am_sy` minus its placebo twin `am_sy_twin` over all
  leads (the forecast sunshine adds information). The threshold is 0.05 points of capacity, about a
  third of experiment 00064's gain of `am` over `d0`, because a capacity re-estimated every week is
  a production pipeline step (a weekly refit of the capacity table) that a fixed table does not
  need. Verdict `supported` when the upper end of C1 is below -0.05 and the upper ends of C3 and C4
  are below zero. `small gain` when the upper end of C1 is below 0 but not below -0.05, and the upper
  ends of C3 and C4 are below zero. `refuted` when the lower end of C1 is above -0.05 (a gain of the
  threshold size is ruled out) and the upper end of C1 is not below 0. `inconclusive` otherwise,
  including when C3 or C4 is not below zero, which the write-up reports as "no gain
  attributable to the solar term". C2 does not change the verdict. The write-up reports a gain at
  short leads when the upper end of C2 is below -0.05. The second family, for XGBoost's starting
  point, has three contrasts read at 98.33% and does not change the verdict. D1 is `d1l_r` minus
  `am0` at 10 days and beyond without the 14-day correction. D2 is `d0_r` minus `d0_m0` at 10 days
  and beyond without the correction. D3 is `d1l_r` minus `am0` over all leads with the correction.
  The starting point matters when the lower end of D1 is above 0 and the point estimate is at least
  0.05. The gap between `am0` and the stored `d1l` is an artefact of the two pipelines when the
  upper end of D1 is below 0.05. Otherwise the reading is unresolved. A positive D1 means that
  `d1l_r` scores worse than `am0`. Every other read is exploratory, labelled so, at 95%: `am_wk` minus `am`, `am_wk`
  minus `am_sy`, `am` minus `am0` (reported beside C3, because the solar term is the headline
  read), `am_sy` minus `am` restricted to the runs where step 1 of the fallback chain supplied the
  capacity, the three arms against `d0` (stored) and `d0_r` (refitted), lead
  bands, valid months, and series. The capacity vintages are inputs, not scores, and are printed
  before any arm is scored.
benchmark: >-
  `am`, `am_twin`, `d0`, and `d1l` as stored by experiment 00064 on row set `fold_a_every5d_v1`,
  `am0` as stored by experiment 00076, and `climatology` as stored by experiment 00064. `am_sy`,
  `am_wk`, and `am_sy_twin` reuse experiment 00064's fitted XGBoost models and change only the
  capacity in the base margin at prediction. `d1l_r`, `d0_r`, and `d0_m0` are fitted here with
  experiment 00064's recipe and seeds (0 to 2).
row_set_id: fold_a_every5d_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels, the estimate of the
  CRPS) in points of capacity (1 point is 1% of the series' effective capacity), after the 14-day
  correction and the 1.15 spread factor unless a read says otherwise, on experiment 00064's rows.
code: ml-experiments/00078-pvdis-5-current-capacity-vintages/
---

# Experiment 78: does a re-estimated solar capacity lower the error of experiment 64's best forecast? (PVDIS-5, check 2)

**Plan status: committed before any code runs.** The write-up replaces this page when the experiment
finishes.

## Question

Experiment 64 gave an XGBoost model a starting prediction (a base margin) equal to a weight times
the substation's embedded-solar capacity times a forecast of how sunny the day will be, and called
this arm `am`. The capacity was one number per Primary substation, fitted on data up to 2025-06-30
and held for validation days from July 2025 to June 2026. That number is 5% to 19% below experiment
45's 2025/26 constants. This experiment replaces it with a capacity that is re-estimated before
each forecast run, and asks whether the score falls further. The experiment also refits the reference model
`d1l` in the same pipeline, which experiment 76 left open, and tests whether giving XGBoost a base
margin of zero, instead of its estimated base score, changes the error of `d0` as well.

## Arms

- **`am`, `am_twin`, `d0`, `d1l`, `climatology`:** experiment 64's stored predictions.
- **`am0`:** experiment 76's stored predictions, fitted by a copy of the same code with the same seeds: the `am` recipe with a margin of zero in every row,
  so that it starts from zero like `am` and carries no solar term.
- **`am_sy`:** experiment 64's fitted `am` models, predicted with a margin whose capacity is the
  season-year vintage below. No model is refitted. The weight on the solar term, the training rows,
  and the capacity used in the training margin are experiment 64's.
- **`am_sy_twin`:** the same for experiment 64's `am_twin` models and placebo anomaly.
- **`am_wk`:** like `am_sy`, with the weekly vintage below (a sensitivity).
- **`d1l_r`:** `d1l` refitted here: experiment 64's recipe and 15 columns, no base margin, so
  XGBoost estimates its own base score.
- **`d0_r`:** `d0` (the 15 `d1l` columns plus `shortwave` and `longwave`) refitted here, no base
  margin.
- **`d0_m0`:** `d0_r` with a base margin of zero in every row and at prediction.

For `d1l_r` and `d0_r` the script prints XGBoost's estimated base score for each of the nine
quantile levels, per series and seed, and the fixed 500 rounds (the recipe has no early stopping, so
no interaction with early stopping exists in this pipeline).

## The capacity vintages

**Season-year vintage (`am_sy`).** For each validation forecast run, the script refits experiment
45's season-year constant on the usable half-hours of the weeks before the Monday of the run's week
minus one week (a one-week gap for the CAMS-based solar shape and the power data to arrive). The
fit is experiment 45's: one capacity per September-to-August season-year, non-negative least
squares, with the demand cells and heating degree-days projected out, divided by experiment 45's
stored recovery ratio of the series (a property of the injection test, computed from the full record
and not truncated at the origin, so each vintage carries a small look-ahead that `am`, `am_sy`, and
`am_wk` share). The run's capacity is chosen in this order:

1. the vintage's constant for the run's season-year, if that season-year has at least 8 observed
   weeks (weeks with 3 or more usable days) before the origin;
2. otherwise the vintage's constant for the previous season-year, if it has at least 8 observed
   weeks;
3. otherwise experiment 64's held capacity (for D14, experiment 40's 2023/24 constant, because D14's
   weeks from 2023-09-04 to 2025-08-25 are unobserved and its fitted constant there is about 0 MW).

**Weekly vintage (`am_wk`).** The run's capacity is the mean, over the 4 latest weeks that are not
`unobserved`, of experiment 45's `frozen` weekly fit in the vintage with the latest `origin_monday`
that is on or before the Monday of the run's week minus 7 days. Experiment 45 stored the weekly
vintages every 4 weeks. When no such weeks exist, the choice of step 3 above applies.

The script prints both vintages per series and run before any arm is scored, with the share of runs
that each step of the fallback chain supplied. The training margin of every fitted model keeps
experiment 64's training capacity, so the contrast measures the effect of the capacity at
prediction only.

## Contrasts

| Id | Contrast | Scope | Level |
|---|---|---|---|
| C1 | `am_sy` minus `am` | all leads | 98.75% |
| C2 | `am_sy` minus `am` | leads under 48 hours | 98.75% |
| C3 | `am_sy` minus `am0` | all leads | 98.75% |
| C4 | `am_sy` minus `am_sy_twin` | all leads | 98.75% |
| D1 | `d1l_r` minus `am0` | 10 days and beyond, no 14-day correction | 98.33% |
| D2 | `d0_r` minus `d0_m0` | 10 days and beyond, no 14-day correction | 98.33% |
| D3 | `d1l_r` minus `am0` | all leads | 98.33% |

Intervals come from 50,000 resamples of whole ISO weeks of forecast start and of the 3 seeds
(`studies.bootstrap.bootstrap_difference_at_level`), on the scored rows of experiment 64 (68,903 per
seed). The climatology has no seed and is shown as a reference row.

## Sizing

- **What gets stored:** private prediction sets and a private capacity-vintage table under `data/`;
  nothing in production.
- **Serving path:** untouched.
- **Degradation rule:** untouched. The fallback chain above is an experiment rule, not a production
  rule.
- **Defensible designs:** more than one (when the vintage may use data, how a young season-year is
  treated, and whether training margins also follow the vintage). The rules above fix one, and the
  limits section of the write-up will name the others.
- **Callers:** none; the scripts copy experiment 64's and 76's.

So the size is medium: a Sonnet plan review, a Sonnet code review, and a Sonnet review of the
write-up and its prose.

## Earlier advice followed and departed from

- **Followed:** read the gain of the solar offset against `am0` as well as `d0` (experiment 76); score
  without the 14-day correction when comparing arms at long leads (experiment 76); refit with a
  capacity table that is current for the validation year (experiment 64).
- **Departed from:** experiment 64 advised refitting `am`. This plan changes only the capacity at
  prediction and keeps the fitted models, because a base margin enters the trees only as an
  offset, and a refit with a vintage margin in the training rows is a larger design that this
  experiment does not test.
