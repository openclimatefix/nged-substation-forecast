---
id: "00035"
slug: lead-weighted-quantile-blend-of-stored-models
status: done
size: medium
hypothesis: >-
  For Primary substation demand, solar farms, and wind farms, averaging the quantiles of two or
  three forecast models that experiments 19, 20, 30, and 33 already scored (N1 is the ENS-trained
  single-stage XGBoost model, and the stack is N1 with the 14-day correction and a spread factor of
  1.15), with one weight per
  lead band chosen only on earlier months, lowers the quantile score below the best single member
  by more than the threshold set for each technology (0.05 points of capacity for Primary demand
  and solar, 0.10 for wind).
prediction: >-
  Wind: the blend of N1 and the best guess lowers the all-lead quantile score by 0.10 to 0.30
  against the best guess, mostly under 2 days, where N1 is 0.35 lower at 0 to 24 h. Solar: 0.03 to
  0.10 lower than the best guess. Primary demand: 0.03 to 0.10 lower than the best member of the
  monthly-retrained N1, `a7_cs`, and the stack. BSP and GSP are descriptive. The expected verdict
  is `supported` for wind and `small gain` for solar and Primary demand.
decision_rule: >-
  Three decisive contrasts, the rolling-origin lead-band blend minus the best single member of its
  technology, on the quantile score over all leads, read at 98.33% (Bonferroni across three) with
  whole months resampled (and the three seeds for solar and wind). The best single member is
  whichever member has the lowest all-lead score on the shared rows, so it is chosen with hindsight,
  which makes the contrast stricter, never looser. Per technology the outcome is "no combination"
  when the stop rule fires (below), "port" when the interval is wholly below zero and the estimate
  is at or below minus the threshold, "small gain" when the interval is wholly below zero and the
  estimate is above minus the threshold, "harm" when the interval is wholly above zero, and "null"
  otherwise. The stop rule is evaluated first, and when it fires the outcome is "no combination"
  whatever the
  interval shows. A cell is one scored month and one lead band, and it is a corner cell
  when the chosen weights put all the weight on one member; the stop rule fires when at least
  10/12 (83.3%) of a technology's 60 cells are corner cells, the denominator being 60 for every
  technology and July 2025's five cells counting as no corner cell. The verdict is `supported` when
  at
  least one technology is "port", `small gain` when none is but one is "small gain", `refuted` when
  every technology is "no combination" or "harm", and `inconclusive` otherwise. Thresholds
  are 0.05 points for Primary demand and solar, where the prediction is 0.03 to 0.10 and a gain
  under 0.05 does not justify serving a second XGBoost model, and 0.10 for wind, where N1's
  short-lead gain makes a larger effect likely and the blend adds a second model with a different
  training source. BSP and GSP, the equal-weights and
  single-global-weight arms, and the correction-placement contrast decide nothing.
benchmark: >-
  The best single member of each technology on the same rows; also the equal-weights blend and the
  single-global-weight blend.
row_set_id:
  - fold_a_every5d_v1
  - fold_a_tuesdays_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels, the estimate of the
  CRPS), in points of capacity; MAE of the median; coverage of the 5% to 95% band
verdict: supported
best_arm: >-
  Primary demand: the lead-band blend of the monthly-retrained N1, `a7_cs`, and the stack, 5.025
  points against 5.122 for the best member (difference -0.097, 98.33% interval -0.187 to -0.026).
  Solar (-0.039, -0.115 to +0.031) and wind (-0.074, -0.194 to +0.055) are null.
lessons: >-
  Averaging quantiles helps Primary demand by about 0.1 points, and one fixed weight vector scored
  within 0.03 points of weights per lead band. Solar and wind gains are too small to resolve on 6
  solar farms and 3 wind farms. Applying the 14-day correction after the blend scored worse than
  applying it to each member first.
advice:
  - >-
    Prefer one weight vector to weights per lead band if a blend is ported: it scored 5.020 against
    5.025 for Primary demand (lead-band minus one-weight +0.005, -0.008 to +0.023), and its contrast
    against the best member is -0.102 (-0.201 to -0.025); for solar and wind the difference is
    unresolved (+0.009 and +0.028; simpler-arm contrasts).
  - >-
    Apply the 14-day correction to each member before blending, not to the blend afterwards, which
    scored 0.048 points worse (+0.008 to +0.101, 95%); leave the monthly-retrained N1 uncorrected,
    because the headline blend with that member uncorrected beats the all-corrected blend
    (correction-placement contrast; C1).
  - >-
    Do not expect a gain larger than 0.08 points from blending N1 with `a7` for BSP and GSP: the
    lead-band blend minus the better member was -0.014 (-0.083 to +0.060) over 4 series
    (descriptive contrast).
  - >-
    Test a wind blend on more than 3 farms before deciding, because the short-lead gain of -0.37
    (-0.81 to +0.02) cannot be told from zero on 3 series (per-band contrast).
production_status: candidate
adopted_in: none
code: ml-experiments/00035-lead-weighted-quantile-blend-of-stored-models/
---

# Experiment 35: does a lead-weighted blend of quantiles beat the best single model?

We asked whether averaging the forecast ranges of several stored models, with a different weight at
each lead time, beats the best single model. We blended three stored models of demand at Primary
substations (substations that step power down to the local distribution network), and tested similar
blends for solar and wind. For Primary demand the blend lowered the error by 0.10 points of
capacity, which clears the threshold for moving it into production. For solar and wind the gains
were too small to resolve.

**Primary demand gains 0.10 points of capacity from blending; solar and wind gain too little to
resolve.** The stack is N1 (the single-stage XGBoost quantile model trained on ENS, the ensemble
weather forecast of the European Centre for Medium-Range Weather Forecasts) with the 14-day
correction and a spread factor of 1.15 (experiment 19). The lead-band blend of the monthly-retrained
N1, `a7_cs`, and the stack scores 5.025 points of capacity against 5.122 for the best member
(difference -0.097, 98.33% interval -0.187 to -0.026), so the outcome for Primary demand is "port"
(threshold 0.05). Solar is -0.039 (-0.115 to +0.031) and wind is -0.074 (-0.194 to +0.055), both
"null". The stop rule did not fire anywhere (corner cells: Primary 0 of 60, solar 6, wind 7), so the
verdict is `supported`, on Primary demand alone. The prediction held for Primary demand's size but
not its verdict (expected `small gain`, found "port"), and failed for wind, where the estimate was
-0.074 against a predicted -0.10 to -0.30. Intervals resample whole months, and for Primary demand
one seed, so they say nothing about other substations or about wind farms beyond the three scored.

![Blends minus the best single member over all leads](figure_1.svg)

## Absolute scores

**The quantile score (points of capacity, smaller is better) of every arm, over the cases that
experiment 33's final shared case sets hold.** Primary demand has 768,073 cases, solar 527,025,
wind 321,138, and BSP and GSP 592,950. The in-sample arm chooses its weights on all 12 months
and is an optimistic bound.

| Technology | Members | Equal | One weight | Lead bands | In sample |
|---|---|---|---|---|---|
| Primary demand | monthly 5.122, `a7_cs` 5.268, stack 5.318 | 5.059 | 5.020 | **5.025** | 5.008 |
| Solar | N1 5.113, best guess 5.035 | 4.978 | 4.986 | **4.996** | 4.973 |
| Wind | N1 11.359, best guess 11.042 | 10.965 | 10.941 | **10.969** | 10.907 |
| BSP and GSP (descriptive) | N1 13.665, `a7` 13.658 | 13.621 | 13.643 | 13.644 | 13.621 |

## Contrasts

- **Weights per lead band are no better than one weight.** The lead-band blend minus the one-weight
  blend is +0.005 (-0.008 to +0.023) for Primary demand, +0.009 (-0.005 to +0.028) for solar, and
  +0.028 (-0.019 to +0.082) for wind. The equal-weights blend beats the best member by -0.063 for
  Primary demand (-0.279 to +0.091), so equal weights are not resolved.
- **Primary demand gains at every lead band; wind's short-lead gains are unresolved.** The lead-band
  blend minus the
  best member for Primary demand is -0.17 at 0 to 24 h (-0.35 to -0.03) and -0.08 at 10 days or
  more (-0.20 to -0.002). Wind is -0.37 at 0 to 24 h (-0.81 to +0.02) and +0.05 at 2 to 6 days
  (-0.12 to +0.25).
- **Correct each member before blending, not the blend afterwards.** The blend with the correction
  applied afterwards scores 5.208 against 5.160 for the same blend with every member corrected
  first (+0.048, 95% interval +0.008 to +0.101; the contrast favours the corrected-first arm,
  because that arm's weights were chosen on it). Correcting the monthly-retrained N1 as well still
  scores 0.135 points worse than the headline blend, whose monthly-retrained N1 is uncorrected.
- **Against the stack of experiment 19, the Primary blend is -0.293** (-0.447 to -0.140, 95%). The
  monthly-retrained N1 alone is 0.196 below the stack on the same cases (5.122 against 5.318).

![Lead-band blend minus best member by lead band](figure_2.svg)

## Chosen weights

**The weights favour the member with the lower score in that band, and for wind and solar they
vary from band to band.** Mean chosen weights over the 11 months with history, by lead band from 0
to 24 h to 10 days or more: Primary demand gives the monthly-retrained N1 0.59 to 0.70, `a7_cs`
0.30 to 0.39, and the stack 0.00 to 0.05. Wind gives N1 0.68 and 0.61 under 2 days and 0.20 to 0.30
beyond. Solar gives N1 between 0.30 and 0.57. Weights lie on a 0.25 grid. The stop rule counts
corner cells over all 60 cells: Primary 0, solar 6, wind 7 (BSP and GSP 13).

## Porting cost

**A blend serves one more model per technology, and for Primary demand two more.** Wind and solar
would serve N1 beside the ERA5-trained mixture, which is a second training source and a second
feature pipeline. Primary demand would serve the monthly-retrained N1 (a refit each month for 16
series), `a7_cs` (seven weekly-analogue columns), and the stack, and a fixed weight vector of three
numbers suffices. The experiment does not decide a port; a release plan does.

## Failures and limits

- **One validation year and one seed for Primary demand.** The retrained N1 has seed 0 only.
- **Selection history is short.** Months 2 to 12 choose weights on 1 to 11 months, July 2025 uses
  equal weights, and the long-lead bands start with few cases.
- **The Primary member scores come from experiment 33's stored tables**, which that experiment may
  still be re-scoring; the case sets read here carry that experiment's `shared_keys` as of
  2026-10-15.
- **BSP and GSP are 4 series** and the result is descriptive.

## Lessons for later experiments

**See `advice` in the front matter.** In short: blend with one fixed weight vector, correct inside
the members, and test wind on more farms.
